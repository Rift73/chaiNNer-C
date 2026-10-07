"""Count/lifetime regressions, with CPU fakes and independently frozen functions.

No real weights, network, GPU, inference engine, timing or package startup. The
ncnn fakes are checked against the wheel's classes (Extractor, Option, an empty
Net), which need no Vulkan device. The D-18 tests import the port's native pixel
conversions inside them; the rest of the module runs without the native library.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from enum import Enum
from pathlib import Path
from threading import Event, Lock, get_ident
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from ncnn import ncnn

from api import Generator, NodeContext, NodeId, SettingsParser
from nodes.impl.pytorch import resource_cache as resources
from nodes.utils.utils import get_h_w_c, list_all_files_sorted

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "backend/src"
REF = Path(__file__).with_name("reference_framework_optimizations")
ALIGN = "packages/chaiNNer_pytorch/pytorch/processing/align_image_to_reference.py"
FACE = "packages/chaiNNer_pytorch/pytorch/restoration/upscale_face.py"
INTERP = "packages/chaiNNer_pytorch/pytorch/utility/interpolate_models.py"
LOAD = "packages/chaiNNer_ncnn/ncnn/batch_processing/load_models.py"
LOGGER = SimpleNamespace(
    debug=lambda *args: None, error=lambda *args: None, warning=lambda *args: None
)


def functions(path, **bindings):
    tree = ast.parse(path.read_text("utf-8"))
    definitions = [
        n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    ]
    for node in definitions:
        node.decorator_list = []
    tree.body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        *definitions,
    ]
    namespace = {
        "np": np,
        "Path": Path,
        "Enum": Enum,
        "logger": LOGGER,
        "os": os,
        **bindings,
    }
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


class Context(NodeContext):
    """A node context with the storage and chain-cleanup surface. Every other member
    raises AttributeError, as it would on a context without it."""

    def __init__(self, storage_dir: Path | None = None):
        self._storage_dir = storage_dir
        self.cleanup = set()

    @property
    def storage_dir(self) -> Path:
        if self._storage_dir is None:
            raise AttributeError("storage_dir")
        return self._storage_dir

    @property
    def node_id(self) -> NodeId:
        raise AttributeError("node_id")

    @property
    def settings(self) -> SettingsParser:
        raise AttributeError("settings")

    @property
    def aborted(self) -> bool:
        raise AttributeError("aborted")

    @property
    def paused(self) -> bool:
        raise AttributeError("paused")

    def set_progress(self, progress: float) -> None:
        raise AttributeError("set_progress")

    def add_cleanup(self, fn, after="chain"):
        assert after == "chain"
        self.cleanup.add(fn)

    def finish(self):
        callbacks, self.cleanup = self.cleanup, set()
        for callback in callbacks:
            callback()


def test_frozen_source_integrity():
    manifest = json.loads((REF / "manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((REF / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("path", [ALIGN, FACE, INTERP, LOAD])
def test_registered_metadata_and_signature_unchanged(path):
    def registration(root):
        tree = ast.parse((root / path).read_text())
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and any(
                isinstance(d, ast.Call)
                and isinstance(d.func, ast.Attribute)
                and d.func.attr == "register"
                for d in n.decorator_list
            )
        )
        assert fn.returns is not None
        return (
            ast.dump(fn.args),
            [ast.dump(d) for d in fn.decorator_list],
            ast.dump(fn.returns),
        )

    assert registration(SOURCE) == registration(REF)


@pytest.mark.parametrize("reuse", [False, True])
def test_resource_lifetime_invalidation_cleanup_and_rerun(tmp_path, reuse):
    path = tmp_path / "weights"
    path.write_bytes(b"first")
    context = Context(tmp_path)
    made = []

    def create():
        value = SimpleNamespace(index=len(made), weights=path.read_bytes())
        made.append(value)
        return value, True

    def get():
        with resources.lease_resource(
            context, lambda: resources.file_identity(path), create, reuse=reuse
        ) as value:
            return value

    first, second = get(), get()
    assert (first is second) == reuse
    path.write_bytes(b"changed weights")
    third = get()
    assert third is not second and third.weights == b"changed weights"
    assert len(context.cleanup) == 1
    context.finish()
    assert get() is not third
    assert len(context.cleanup) == 1
    context.finish()


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps", "dml"])
@pytest.mark.parametrize("budget", [0, 1])
@pytest.mark.parametrize("wipe", [False, True])
def test_device_policy_without_device_runtime(device, budget, wipe):
    assert resources.allow_model_reuse(device, budget, wipe) == (
        budget == 0 and (device == "cpu" or not wipe)
    )


def test_missing_checkpoint_not_retained_and_first_download_key(tmp_path):
    context = Context(tmp_path)
    path = tmp_path / "new weights"
    calls = []

    def create():
        if not path.exists():
            path.write_bytes(b"downloaded")
        value = object()
        calls.append(value)
        return value, True

    for _ in range(3):
        with resources.lease_resource(
            context, lambda: resources.file_identity(path), create, reuse=True
        ):
            pass
    assert len(calls) == 2
    context.finish()
    for _ in range(3):
        with resources.lease_resource(
            context, lambda: "incomplete", lambda: (object(), False), reuse=True
        ) as value:
            assert resources._ENTRIES[context].value is value
        assert resources._ENTRIES[context].value is None


def test_failed_use_or_factory_invalidates_and_retries():
    context = Context()
    made, resets = [], []

    def create():
        value = object()
        made.append(value)
        return value, True

    with pytest.raises(ValueError, match="inference"):
        with resources.lease_resource(
            context, lambda: 1, create, reuse=True, reset=resets.append
        ):
            raise ValueError("inference")
    with resources.lease_resource(
        context, lambda: 1, create, reuse=True, reset=resets.append
    ):
        pass
    assert len(made) == len(resets) == 2 and made[0] is not made[1]
    with pytest.raises(ValueError, match="loader"):
        with resources.lease_resource(
            context,
            lambda: 2,
            lambda: (_ for _ in ()).throw(ValueError("loader")),
            reuse=True,
        ):
            pytest.fail("factory should fail")
    assert resources._ENTRIES[context].value is None
    context.finish()


def test_reset_failure_does_not_mask_active_error():
    context = Context()
    primary = ValueError("original inference error")

    def reset(_):
        raise RuntimeError("reset error")

    with pytest.raises(ValueError) as caught:
        with resources.lease_resource(
            context, lambda: 1, lambda: (object(), True), reuse=True, reset=reset
        ):
            raise primary
    assert caught.value is primary
    assert resources._ENTRIES[context].value is None
    context.finish()


def test_parallel_calls_exclusively_lease_mutable_helper():
    context = Context()
    entered, release, waiting = Event(), Event(), Event()
    made = []

    def create():
        value = SimpleNamespace(frames=[])
        made.append(value)
        return value, True

    def run(index):
        if index == 2:
            waiting.set()
        with resources.lease_resource(
            context, lambda: 1, create, reuse=True, reset=lambda v: v.frames.clear()
        ) as value:
            assert value.frames == []
            value.frames.append(index)
            if index == 1:
                entered.set()
                assert release.wait(5)
            assert value.frames == [index]
            return index

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(run, 1)
        assert entered.wait(5)
        second = pool.submit(run, 2)
        assert waiting.wait(5)
        release.set()
        assert (first.result(), second.result()) == (1, 2)
    assert len(made) == 1 and made[0].frames == []
    context.finish()


def test_contexts_and_reentrant_calls_do_not_share_mutable_state():
    a, b = Context(), Context()

    class Resource:
        pass

    def create():
        return (Resource(), True)

    with resources.lease_resource(a, lambda: 1, create, reuse=True) as first:
        with resources.lease_resource(a, lambda: 1, create, reuse=True) as nested:
            assert nested is not first
        with resources.lease_resource(b, lambda: 1, create, reuse=True) as other:
            assert other is not first
    ref = weakref.ref(first)
    del first
    a.finish()
    assert ref() is None
    b.finish()


@pytest.mark.parametrize("backend", ["ncnn", "onnx"])
@pytest.mark.parametrize("count", [0, 1, 4, 16])
@pytest.mark.parametrize("collect_after", [False, True])
def test_collection_scales_with_invocation_not_tiles(backend, count, collect_after):
    calls = []
    image = np.ones((2, 3, 3), np.float32)

    class Mat:
        def __init__(self, data):
            self.data = data

        def clone(self):
            return Mat(self.data.copy())

    class Extractor:
        def __init__(self):
            self.value = None

        def input(self, _, value):
            self.value = value

        def extract(self, _):
            assert self.value is not None, "extract before input"
            return 0, self.value.data

    class Session:
        def create_extractor(self):
            return Extractor()

        def get_inputs(self):
            return [SimpleNamespace(name="in", type="tensor(float)")]

        def get_outputs(self):
            return [SimpleNamespace(name="out")]

        def run(self, _, inputs):
            return [inputs["in"]]

    def split(value, callback, _, **kwargs):
        return [callback(value, None) for _ in range(count)]

    module = functions(
        SOURCE / f"nodes/impl/{backend}/auto_split.py",
        use_gpu=False,
        ncnn=SimpleNamespace(Mat=Mat),
        get_h_w_c=lambda i: i.shape,
        to_uint8=lambda i: (i * 255).astype(np.uint8),
        ncnn_input=lambda i: i.transpose(2, 0, 1).astype(np.float32) / 255,
        cast_numpy=lambda i, dtype: i.astype(dtype),
        gc=SimpleNamespace(collect=lambda: calls.append("collect")),
        auto_split=split,
        swap_red_blue=lambda i: i[..., ::-1],
        SizeReq=lambda: SimpleNamespace(get_padding=lambda w, h: (0, 0)),
    )
    if backend == "ncnn":
        output = module.ncnn_auto_split(
            image, Session(), "in", "out", None, None, None, collect_after=collect_after
        )
    else:
        module.onnx_auto_split.__globals__["cast"] = lambda _, value: value
        output = module.onnx_auto_split(
            image, Session(), False, None, collect_after=collect_after
        )
    assert len(output) == count
    for result in output:
        np.testing.assert_array_equal(result, image)
    assert len(calls) == int(collect_after)


@pytest.mark.parametrize(
    "names",
    [
        [],
        ["model2.param", "model2.bin", "model10.param", "model10.bin"],
        ["inner/UP.PARAM", "inner/UP.BIN", "ignored.txt"],
        ["a.param"],
        ["a.param", "b.bin"],
    ],
)
def test_ncnn_pairing_one_discovery_exact_order_and_errors(tmp_path, names):
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    outcomes, counts = [], []
    for root in (REF, SOURCE):
        scans = []

        def discover(directory, extensions, scans=scans):
            scans.append(tuple(extensions))
            return list_all_files_sorted(directory, extensions)

        module = functions(
            root / LOAD,
            list_all_files_sorted=discover,
            Generator=Generator,
            load_model_node=lambda p, b: (str(b), p.parent, p.stem),
        )
        try:
            generator, directory = module.load_models_node(tmp_path, True)
            value = list(generator.supplier())
            outcomes.append((value, directory, generator.fail_fast))
        except Exception as error:
            outcomes.append((type(error), str(error)))
        counts.append(len(scans))
    assert outcomes[0] == outcomes[1]
    assert counts == [2, 1]


def test_face_reuses_helper_and_releases_frame_state(tmp_path):
    settings = SimpleNamespace(
        device=SimpleNamespace(type="cpu"), budget_limit=0, force_cache_wipe=False
    )
    calls = []

    class Helper:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.frames = []

        def clean_all(self):
            self.frames.clear()

    def upscale(img, bg, helper, *args):
        assert helper.frames == [] and not hasattr(helper, "input_img")
        helper.frames.append(img)
        helper.input_img = img
        return img.copy()

    module = functions(
        SOURCE / FACE,
        torch=SimpleNamespace(no_grad=nullcontext),
        get_settings=lambda _: settings,
        FaceRestoreHelper=Helper,
        lease_resource=resources.lease_resource,
        file_identity=resources.file_identity,
        allow_model_reuse=resources.allow_model_reuse,
        safe_cuda_cache_empty=lambda: None,
    )
    module.upscale_face_node.__globals__.update(
        denormalize=lambda i: i, upscale=upscale
    )
    context = Context(tmp_path)
    for i in range(3):
        image = np.full((2, 2, 3), i, np.float32)
        np.testing.assert_array_equal(
            module.upscale_face_node(context, image, None, None, 2, 0.7), image
        )
    assert len(calls) == 1
    helper = resources._ENTRIES[context].value
    assert helper.frames == [] and not hasattr(helper, "input_img")
    module.upscale_face_node(context, image, None, None, 4, 0.7)
    assert len(calls) == 2
    context.finish()


def test_align_reuses_models_but_invalidates_weights_options_and_incomplete_state(
    tmp_path,
):
    calls = []
    settings = SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        use_fp16=False,
        budget_limit=0,
        force_cache_wipe=False,
    )
    module = functions(
        SOURCE / ALIGN,
        lease_resource=resources.lease_resource,
        file_identity=resources.file_identity,
        allow_model_reuse=resources.allow_model_reuse,
    )
    retainable = [True]

    def create(*args):
        value = object()
        calls.append(value)
        return (value, None, None), retainable[0]

    module.align_images.__globals__.update(
        _load_alignment_models=create, _align_images=lambda *args: args[-1]
    )
    context = Context(tmp_path)

    def run(wide=False):
        return module.align_images(
            context, None, None, None, None, False, wide, settings
        )[0]

    first = run()
    assert run() is first and len(calls) == 1
    assert run(True) is not first and len(calls) == 2
    weights = tmp_path / "rife_v4.14/weights/flownet.pkl"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"new")
    run(True)
    assert len(calls) == 3
    settings.use_fp16 = True
    run(True)
    assert len(calls) == 4
    context.finish()
    retainable[0] = False
    assert run() is not run()
    context.finish()


def test_align_numerical_body_remains_original():
    old = next(
        n
        for n in ast.parse((REF / ALIGN).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "align_images"
    )
    new = next(
        n
        for n in ast.parse((SOURCE / ALIGN).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_align_images"
    )

    # Compare the entire image computation after constructor preparation. The
    # additional narrowing assertion merely checks the wide-search bundle.
    def pixel_body(fn):
        start = next(
            i
            for i, n in enumerate(fn.body)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "fclip" for t in n.targets)
        )
        tree = ast.Module(body=fn.body[start:], type_ignores=[])

        class StripNarrowing(ast.NodeTransformer):
            def visit_Assert(self, node):
                return None if "descriptor is not None" in ast.unparse(node) else node

        return ast.dump(StripNarrowing().visit(tree))

    assert pixel_body(old) == pixel_body(new)


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize("incomplete", [False, True])
def test_alignment_real_preparation_path_reuses_only_complete_weights(
    tmp_path, wide, half, incomplete
):
    settings = SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        use_fp16=half,
        budget_limit=0,
        force_cache_wipe=False,
    )
    counts = {"load": 0, "rife": 0, "xfeat": 0, "half": 0}

    class Model:
        def __init__(self, **kwargs):
            counts["rife"] += 1

        def to(self, device):
            return self

        def load_state_dict(self, state, strict=False):
            assert state == {"weight": 1} and strict is False
            return SimpleNamespace(missing_keys=["uninitialized"] if incomplete else [])

        def eval(self):
            return self

        def half(self):
            counts["half"] += 1
            return self

    class XModel(Model):
        def __init__(self, **kwargs):
            counts["xfeat"] += 1

    def load(path, **kwargs):
        counts["load"] += 1
        assert kwargs == {"map_location": settings.device, "weights_only": True}
        return {"module.weight": 1}

    for relative in ["rife_v4.14/weights/flownet.pkl", "xfeat/weights/xfeat.pt"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"pinned")
    module = functions(
        SOURCE / ALIGN,
        torch=SimpleNamespace(load=load),
        IFNet=Model,
        XFeat=XModel,
        lease_resource=resources.lease_resource,
        file_identity=resources.file_identity,
        allow_model_reuse=resources.allow_model_reuse,
    )
    module.align_images.__globals__.update(
        download_model=lambda **kwargs: kwargs["model_dir"] / kwargs["model_file"],
        _align_images=lambda *args: args[-1],
    )
    context = Context(tmp_path)
    result = [
        module.align_images(context, None, None, None, None, False, wide, settings)
        for _ in range(3)
    ]
    constructions = 3 if incomplete else 1
    assert counts == {
        "load": constructions * (2 if wide else 1),
        "rife": constructions,
        "xfeat": constructions * (2 if wide else 0),
        "half": constructions * (3 if wide else 1) * int(half),
    }
    assert (result[0][0] is result[1][0]) != incomplete
    context.finish()


def test_version_change_during_load_is_not_mislabeled_as_cached(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"old")
    context = Context(tmp_path)
    calls = []

    def create():
        loaded = path.read_bytes()
        calls.append(loaded)
        if len(calls) == 1:
            path.write_bytes(b"new version")
        return loaded, True

    values = []
    for _ in range(3):
        with resources.lease_resource(
            context, lambda: resources.file_identity(path), create, reuse=True
        ) as value:
            values.append(value)
    assert calls == [b"old", b"new version"]
    assert values == [b"old", b"new version", b"new version"]
    context.finish()


VK_FIELDS = ("blob_vkallocator", "workspace_vkallocator", "staging_vkallocator")


def public(cls):
    return {name for name in dir(cls) if not name.startswith("_")}


class VkOption:
    """A real ncnn.Option whose three Vulkan allocator fields hold CPU fakes.

    The real setters take only a VkAllocator, and one exists only on a Vulkan
    device, so those fields are kept here. Every other name goes to the real
    Option, which has no __dict__ and rejects a name the wheel lacks.
    """

    def __init__(self, on_write=lambda name, value: None):
        assert set(VK_FIELDS) <= public(ncnn.Option)
        object.__setattr__(self, "real", ncnn.Option())
        object.__setattr__(self, "vk", dict.fromkeys(VK_FIELDS))
        object.__setattr__(self, "on_write", on_write)

    def __getattr__(self, name):
        return self.vk[name] if name in VK_FIELDS else getattr(self.real, name)

    def __setattr__(self, name, value):
        if name not in VK_FIELDS:
            setattr(self.real, name, value)
            return
        self.vk[name] = value
        self.on_write(name, value)


def ncnn_gpu_auto_split(lock, **bindings):
    """ncnn_auto_split on its Vulkan branch, one tile, guarded by `lock`."""

    class Mat:
        def __init__(self, data):
            self.data = data

        def clone(self):
            return self

    return functions(
        SOURCE / "nodes/impl/ncnn/auto_split.py",
        use_gpu=True,
        _net_opt_lock=lock,
        ncnn=SimpleNamespace(Mat=Mat),
        get_h_w_c=lambda i: i.shape,
        to_uint8=lambda i: i,
        ncnn_input=lambda i: i,
        cast_numpy=lambda i, dtype: i.astype(dtype),
        auto_split=lambda image, callback, tiler, **kwargs: callback(image, None),
        **bindings,
    ).ncnn_auto_split


class Extractor:
    def __init__(self):
        self.mat = None

    def input(self, name, mat):
        self.mat = mat

    def extract(self, name):
        assert self.mat is not None, "extract before input"
        return 0, self.mat.data.transpose(2, 0, 1)


def test_ncnn_fakes_mirror_the_wheel():
    assert public(Extractor) <= public(ncnn.Extractor)
    option = VkOption()
    option.num_threads = 3
    assert option.real.num_threads == 3
    with pytest.raises(AttributeError):
        option.blob_vk_allocator = None


def test_real_ncnn_option_takes_none_on_its_vulkan_allocator_fields():
    option = ncnn.Option()
    net = ncnn.Net()
    for name in VK_FIELDS:
        setattr(option, name, None)
        setattr(net.opt, name, None)
        assert getattr(option, name) is None and getattr(net.opt, name) is None
    net.opt.num_threads = 3
    assert net.opt.num_threads == 3


@pytest.mark.parametrize(
    "failure", ["failed allocation", "vkQueueSubmit failure", "unrelated"]
)
def test_ncnn_error_collection_and_allocator_cleanup_retained(failure):
    calls = []

    class Failing(Extractor):
        def input(self, name, mat):
            raise RuntimeError(failure)

    assert public(Failing) <= public(ncnn.Extractor)
    allocator = SimpleNamespace(clear=lambda: calls.append("allocator"))
    split = object()
    ncnn_auto_split = ncnn_gpu_auto_split(
        Lock(),
        gc=SimpleNamespace(collect=lambda: calls.append("gc")),
        Split=lambda: split,
    )

    def run():
        return ncnn_auto_split(
            np.ones((1, 1, 3), np.uint8),
            SimpleNamespace(opt=VkOption(), create_extractor=Failing),
            "in",
            "out",
            allocator,
            allocator,
            None,
            collect_after=False,
        )

    if failure == "failed allocation":
        assert run() is split
    else:
        with pytest.raises(RuntimeError):
            run()
    assert calls == (["gc", "allocator", "allocator"] if failure != "unrelated" else [])


@pytest.mark.parametrize("create_fails", [False, True])
def test_ncnn_gpu_extractor_takes_allocators_from_opt_under_lock(create_fails):
    lock = Lock()
    events = []
    option = VkOption(lambda name, value: events.append((name, value, lock.locked())))
    blob = SimpleNamespace(clear=lambda: events.append("blob clear"))
    staging = SimpleNamespace(clear=lambda: events.append("staging clear"))

    def create_extractor():
        fields = tuple(getattr(option, name) for name in VK_FIELDS)
        events.append(("create", fields, lock.locked()))
        if create_fails:
            raise RuntimeError("no extractor")
        return Extractor()

    net = SimpleNamespace(opt=option, create_extractor=create_extractor)
    assert public(net) <= public(ncnn.Net)
    image = np.ones((2, 3, 3), np.float32)
    ncnn_auto_split = ncnn_gpu_auto_split(lock)
    if create_fails:
        with pytest.raises(RuntimeError, match="no extractor"):
            ncnn_auto_split(image, net, "in", "out", blob, staging, None, False)
    else:
        result = ncnn_auto_split(image, net, "in", "out", blob, staging, None, False)
        np.testing.assert_array_equal(result, image)
    assert events == [
        ("blob_vkallocator", blob, True),
        ("workspace_vkallocator", blob, True),
        ("staging_vkallocator", staging, True),
        ("create", (blob, blob, staging), True),
        *[(name, None, True) for name in VK_FIELDS],
        *([] if create_fails else ["blob clear", "staging clear"]),
    ]
    assert not lock.locked()
    assert all(getattr(option, name) is None for name in VK_FIELDS)


@pytest.mark.parametrize("guarded", [True, False])
def test_ncnn_concurrent_upscales_never_see_each_others_allocators(guarded):
    # The first thread to create an extractor waits for the other thread to write
    # net.opt. The lock keeps that write out until the wait times out; without
    # the lock, the write lands and the first extractor takes foreign allocators.
    first, other_wrote, seen, owners = [], Event(), [], {}

    def on_write(name, value):
        if first and first[0] != get_ident():
            other_wrote.set()

    option = VkOption(on_write)

    def create_extractor():
        first.append(get_ident())
        if first[0] == get_ident():
            other_wrote.wait(1)
        seen.append((get_ident(), tuple(getattr(option, n) for n in VK_FIELDS)))
        return Extractor()

    net = SimpleNamespace(opt=option, create_extractor=create_extractor)
    ncnn_auto_split = ncnn_gpu_auto_split(Lock() if guarded else nullcontext())
    image = np.ones((2, 3, 3), np.float32)

    def run(index):
        blob = SimpleNamespace(clear=lambda: None, index=index)
        staging = SimpleNamespace(clear=lambda: None, index=index)
        owners[get_ident()] = (blob, blob, staging)
        return ncnn_auto_split(image, net, "in", "out", blob, staging, None, False)

    with ThreadPoolExecutor(2) as pool:
        for result in [pool.submit(run, index) for index in (1, 2)]:
            np.testing.assert_array_equal(result.result(), image)
    assert len(owners) == len(seen) == 2
    own = [fields == owners[ident] for ident, fields in seen]
    assert all(own) if guarded else not all(own)
    assert all(getattr(option, name) is None for name in VK_FIELDS)


@pytest.mark.parametrize(
    "acquired", ["subclass", "unavailable", "base blob", "base staging"]
)
def test_ncnn_allocators_are_the_cleared_vk_subclasses(acquired):
    # The wheel declares acquire_*_allocator() as returning the base VkAllocator,
    # which has no clear(); pybind11 should hand back the registered subclass.
    calls = []

    class VkAllocator:
        pass

    class Blob(VkAllocator):
        def __init__(self, vkdev=None):
            if vkdev is not None:
                calls.append("blob built")

        def clear(self):
            calls.append("blob clear")

    class Staging(VkAllocator):
        def __init__(self, vkdev=None):
            if vkdev is not None:
                calls.append("staging built")

        def clear(self):
            calls.append("staging clear")

    class Device:
        def acquire_blob_allocator(self):
            if acquired == "unavailable":
                raise RuntimeError("no pooled allocator")
            return VkAllocator() if acquired == "base blob" else Blob()

        def acquire_staging_allocator(self):
            if acquired == "unavailable":
                raise RuntimeError("no pooled allocator")
            return VkAllocator() if acquired == "base staging" else Staging()

    assert public(VkAllocator) <= public(ncnn.VkAllocator)
    assert public(Blob) <= public(ncnn.VkBlobAllocator)
    assert public(Staging) <= public(ncnn.VkStagingAllocator)
    assert public(Device) <= public(ncnn.VulkanDevice)
    module = functions(
        SOURCE / "packages/chaiNNer_ncnn/ncnn/processing/upscale_image.py",
        ncnn=SimpleNamespace(VkBlobAllocator=Blob, VkStagingAllocator=Staging),
    )
    module.ncnn_allocators.__globals__["managed_blob_vkallocator"] = contextmanager(
        module.managed_blob_vkallocator
    )
    allocators = contextmanager(module.ncnn_allocators)
    if acquired.startswith("base"):
        kind = acquired.split()[1]
        with (
            pytest.raises(TypeError, match=f"VkAllocator as its {kind} allocator"),
            allocators(Device()),
        ):
            pytest.fail("a base VkAllocator must not be yielded")
        assert calls == ([] if kind == "blob" else ["blob clear"])
        return
    with allocators(Device()) as (blob, staging):
        assert isinstance(blob, Blob) and isinstance(staging, Staging)
        assert "clear" not in str(calls)
    built = ["blob built", "staging built"] if acquired == "unavailable" else []
    assert calls == [*built, "staging clear", "blob clear"]


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_ncnn_reads_lens_planar_layout_by_logical_index(channels):
    # Consult 11 D-18: Lens Blur's planar layout and its C copy give the network
    # identical planes. Upstream's Mat.from_pixels read the raw buffer instead.
    from nodes.impl.image_utils import to_uint8
    from nodes.impl.native_framework_images import ncnn_input
    from nodes.impl.native_tensors import cast_numpy

    planes = []

    class Planes(Extractor):
        def input(self, name, mat):
            planes.append(np.array(mat))
            self.mat = mat

        def extract(self, name):
            return 0, self.mat

    assert public(Planes) <= public(ncnn.Extractor)
    ncnn_auto_split = functions(
        SOURCE / "nodes/impl/ncnn/auto_split.py",
        use_gpu=False,
        ncnn=ncnn,
        get_h_w_c=get_h_w_c,
        to_uint8=to_uint8,
        ncnn_input=ncnn_input,
        cast_numpy=cast_numpy,
        auto_split=lambda image, callback, tiler, **kwargs: callback(image, None),
    ).ncnn_auto_split
    net = SimpleNamespace(opt=ncnn.Option(), create_extractor=Planes)
    rng = np.random.default_rng(18)
    copy = rng.uniform(-0.1, 1.1, (7, 9, channels)).astype(np.float32)
    lens = np.ascontiguousarray(copy.transpose(2, 0, 1)).transpose(1, 2, 0)
    results = [
        ncnn_auto_split(image, net, "in", "out", None, None, None, False)
        for image in (lens, copy)
    ]
    assert planes[0].shape == (channels, 7, 9)
    np.testing.assert_array_equal(planes[0], planes[1])
    np.testing.assert_array_equal(results[0], results[1])


@pytest.mark.parametrize("side", ["upstream", "port"])
def test_ncnn_node_hands_its_network_logical_order(side):
    # Consult 11 D-18: through Upscale Image (NCNN), upstream's and the port's, no
    # array that reaches upscale_impl has a memory order other than its logical
    # order. The node's cvtColor copies 3 and 4 channels into a new C array, and one
    # channel has one order, so Lens Blur's planar layout never reached upstream's
    # raw-buffer read through the node: D-18 changes no node output.
    reached = []

    def upscale_impl(settings, img, *args, **kwargs):
        reached.append(img)
        return img if img.ndim == 3 else img[:, :, None]  # NCNN's (h, w, c)

    if side == "upstream":
        buffers = REF.with_name("reference_buffers")
        utils = functions(buffers / "image_utils.py", cv2=cv2, get_h_w_c=get_h_w_c)
        shared = REF.with_name("reference_framework_shared")
        upscale = functions(
            buffers / "installed_convenient_upscale.py",
            get_h_w_c=get_h_w_c,
            clipped=functions(shared / "image_op.py").clipped,
            as_target_channels=utils.as_target_channels,
        ).convenient_upscale
        path = REF / "installed/packages/chaiNNer_ncnn/ncnn/processing/upscale_image.py"
    else:
        from nodes.impl.upscale.convenient_upscale import convenient_upscale as upscale

        path = SOURCE / "packages/chaiNNer_ncnn/ncnn/processing/upscale_image.py"
    node = functions(
        path,
        cv2=cv2,
        get_h_w_c=get_h_w_c,
        convenient_upscale=upscale,
        get_settings=lambda _: None,
        CUSTOM=-1,
    ).upscale_image_node
    node.__globals__["upscale_impl"] = upscale_impl
    rng = np.random.default_rng(18)
    for channels, alpha in ((1, None), (3, None), (4, "varying"), (4, "constant")):
        planes = rng.random((channels, 6, 5), dtype=np.float32)
        if alpha == "constant":
            planes[3] = 0.5
        image = planes.transpose(1, 2, 0)  # Lens Blur's layout
        for nc in (1, 3, 4):
            layers = [SimpleNamespace(outputs=["x"])]
            model = SimpleNamespace(
                in_nc=nc, out_nc=nc, model=SimpleNamespace(layers=layers)
            )
            for separate in (False, True):
                node(Context(), image, model, 0, 0, separate)
    assert len(reached) == 28
    assert all(np.array_equal(a.ravel(order="K"), a.ravel(order="C")) for a in reached)


@pytest.mark.parametrize("backend", ["ncnn", "onnx"])
@pytest.mark.parametrize("fail", [False, True])
def test_registered_upscale_defers_collection_and_drains_once(backend, fail):
    calls = []

    def collect():
        calls.append("gc")
        return 4

    def helper(image, *args, **kwargs):
        assert kwargs["collect_after"] is False
        calls.append("tile")
        if fail:
            raise ValueError("inference failed")
        return args[0] if backend == "ncnn" else image

    def convenient(image, in_nc, out_nc, callback, separate_alpha, **kwargs):
        return callback(image, None) if "progress" in kwargs else callback(image)

    settings = SimpleNamespace(
        gpu_index=0,
        execution_provider="CPUExecutionProvider",
        tensorrt_fp16_mode=False,
        tensorrt_cache_path="",
    )
    path = f"packages/chaiNNer_{backend}/{backend}/processing/upscale_image.py"
    module = functions(
        SOURCE / path,
        gc=SimpleNamespace(collect=collect),
        get_settings=lambda _: settings,
        convenient_upscale=convenient,
        get_h_w_c=lambda image: image.shape,
        get_onnx_session=lambda *args: object(),
        get_input_shape=lambda _: ("BCHW", 1, None, None),
        get_output_shape=lambda _: ("BCHW", 1, None, None),
        CUSTOM=-1,
    )
    node = module.upscale_image_node
    node.__globals__["upscale_impl" if backend == "ncnn" else "upscale"] = helper
    model = SimpleNamespace(
        in_nc=1,
        out_nc=1,
        model=SimpleNamespace(layers=[SimpleNamespace(outputs=["name"])]),
        info=SimpleNamespace(scale_width=None, scale_height=None),
    )
    context = Context()
    image = np.ones((2, 3, 1), np.float32)
    for _ in range(3):
        if fail:
            with pytest.raises(ValueError, match="inference failed"):
                node(context, image, model, 1, 1, False)
        else:
            assert node(context, image, model, 1, 1, False) is image
    assert calls == ["tile"] * 3 and len(context.cleanup) == 1
    context.finish()
    assert calls == ["tile"] * 3 + ["gc"]
    context.finish()
    assert calls.count("gc") == 1


@pytest.mark.parametrize(
    "path",
    [
        ALIGN,
        FACE,
        INTERP,
        LOAD,
        "packages/chaiNNer_ncnn/ncnn/processing/upscale_image.py",
        "packages/chaiNNer_onnx/onnx/processing/upscale_image.py",
    ],
)
def test_installed_delegate_registration_exact(path):
    def registration(root):
        tree = ast.parse((root / path).read_text("utf-8"))
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and any(
                isinstance(d, ast.Call)
                and isinstance(d.func, ast.Attribute)
                and d.func.attr == "register"
                for d in n.decorator_list
            )
        )
        assert node.returns is not None
        return (
            ast.dump(node.args),
            ast.dump(node.returns),
            [ast.dump(d) for d in node.decorator_list],
        )

    assert registration(REF / "installed") == registration(SOURCE)
