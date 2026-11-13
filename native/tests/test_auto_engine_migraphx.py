"""PyTorch Upscale Image's Use MIGraphX Program (nodes/impl/pytorch/auto_engine.py): MIGraphX's
route of its one flow, with the real MIGraphX package, Build Program and Upscale Image on
stand-ins for the AMD GPU, the compile and the runner (migraphx_stand_ins.py). No AMD GPU or
MIGraphX is needed. What the flow shares with NVIDIA's route, its tests cover
(native/tests/test_auto_engine.py)."""

from __future__ import annotations

import importlib.metadata
import inspect
import sys
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[2] / "backend" / "tests"))

from migraphx_stand_ins import (
    ABSENT,
    Here,
    Imported,
    Runner,
    import_package,
    package_imports,
)

import nodes.impl.migraphx.dual as dual_route
from api import Aborted, NodeContext, NodeId, SettingsParser
from gpu import AmdDevice, AmdInfo
from nodes.impl.migraphx import build_cache
from nodes.impl.migraphx.model import MIGraphXProgram, MIGraphXProgramInfo
from nodes.impl.onnx.model import OnnxGeneric, OnnxInfo
from nodes.impl.pytorch import auto_engine as auto
from nodes.impl.tensorrt import cache as tensorrt_cache
from nodes.impl.upscale.auto_split_tiles import (
    ESTIMATE,
    MAX_TILE_SIZE,
    NO_TILING,
    TileSize,
)
from nodes.impl.upscale.tile_report import TileReport

NODE = NodeId("upscale")
GPU = torch.device("cuda:0")
# A 2x model, as the stand-in runner's upscale.
IMAGE = np.full((4, 6, 3), 0.25, dtype=np.float32)


class Context(NodeContext):
    def __init__(self, root: Path) -> None:
        super().__init__()
        self.storage = root / "backend-storage"
        self.warnings: list[str] = []
        self.chain_cleanups: list = []

    @property
    def aborted(self) -> bool:
        return False

    @property
    def paused(self) -> bool:
        return False

    def set_progress(self, progress: float) -> None:
        self.check_aborted()

    @property
    def settings(self) -> SettingsParser:
        return SettingsParser({"gpu_index": "0"})

    @property
    def node_id(self) -> NodeId:
        return NODE

    @property
    def storage_dir(self) -> Path:
        return self.storage

    def add_cleanup(self, fn, after="chain") -> None:
        self.chain_cleanups.append(fn)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def end_chain(self) -> None:
        for fn in self.chain_cleanups:
            fn()
        self.chain_cleanups.clear()


class Model:
    def __init__(
        self,
        arch: str = "ESRGAN",
        weight: float = 1.0,
        multiple: int = 1,
        minimum: int = 0,
    ) -> None:
        self.architecture = SimpleNamespace(name=arch, id=arch)
        self.tags: list[str] = []
        self.scale = 2
        self.size_requirements = SimpleNamespace(minimum=minimum, multiple_of=multiple)
        self.model = torch.nn.Conv2d(1, 1, 1)
        torch.nn.init.constant_(self.model.weight, weight)
        torch.nn.init.constant_(self.model.bias, 0.0)


def info(size: tuple[int, int], scale: int = 2) -> MIGraphXProgramInfo:
    return MIGraphXProgramInfo(
        "fp16", 3, 3, scale, "gfx1100", "2.16.1", size[0], size[1]
    )


class Fakes:
    """What the real path reaches behind Build Program and Upscale Image."""

    def __init__(self, imported: Imported) -> None:
        self.imported = imported
        self.conversions = 0
        self.compiles: list[tuple[int, int]] = []
        self.compile_error: Exception | None = None
        self.program_scale = 2

    def convert(self, *_args: object) -> tuple[OnnxGeneric, str, str]:
        self.conversions += 1
        return (
            OnnxGeneric(b"onnx", OnnxInfo(20, "fp16", input_channels=3)),
            "fp16",
            "opset20",
        )

    def compile(self, _onnx: bytes, _gpu: int, size: tuple[int, int], **_kw) -> bytes:
        if self.compile_error is not None:
            raise self.compile_error
        self.compiles.append(size)
        return b"program of %dx%d" % size

    def read_info(self, program: bytes, _gpu: int) -> MIGraphXProgramInfo:
        height, width = program.removeprefix(b"program of ").split(b"x")
        return info((int(height), int(width)), self.program_scale)


@pytest.fixture
def amd_pc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[tuple[Fakes, Path]]:
    """This PC as an AMD one, with the MIGraphX package loaded and every keeper empty."""
    pytest.importorskip("packages.chaiNNer_pytorch.pytorch.utility.convert_to_onnx")
    from packages.chaiNNer_pytorch.pytorch.utility import convert_to_onnx

    Runner.made = []
    monkeypatch.setattr(build_cache, "_memory", OrderedDict())
    monkeypatch.setattr(build_cache, "_building", {})
    with package_imports():
        imported = import_package(monkeypatch, tmp_path, Here())
        fakes = Fakes(imported)
        monkeypatch.setattr(auto, "nvidia", SimpleNamespace(is_available=False))
        monkeypatch.setattr(
            auto, "amd", AmdInfo([AmdDevice(0, "Radeon", "gfx1100", compute_units=96)])
        )
        monkeypatch.setattr(convert_to_onnx, "convert_to_onnx_node", fakes.convert)
        build = imported.nodes["utility.build_program"]
        monkeypatch.setattr(build, "compile_in_child", fakes.compile)
        monkeypatch.setattr(build, "read_program_info", fakes.read_info)
        monkeypatch.setattr(
            build,
            "model_facts",
            lambda _m: SimpleNamespace(drft_manifest=None, dual_manifest=None),
        )
        monkeypatch.setattr(build, "runner_identity", lambda: "runner")
        monkeypatch.setattr(
            "nodes.impl.migraphx.runner.runner_identity", lambda: "runner"
        )
        for kept in (auto._outcomes, auto._engine_nodes, auto._unproven, auto._rebuilt):
            kept.clear()
        yield fakes, tmp_path
        tensorrt_cache.release_node(NODE)
        for kept in (auto._outcomes, auto._engine_nodes, auto._unproven, auto._rebuilt):
            kept.clear()


def pytorch(img: np.ndarray) -> np.ndarray:
    return img * 3


def run(
    context: Context,
    model: Model,
    tile: TileSize = ESTIMATE,
    image: np.ndarray = IMAGE,
    report: TileReport | None = None,
) -> np.ndarray:
    return auto.upscale_through_engine(
        context,
        model,
        GPU,
        0,
        tile,
        lambda op: op(image),
        pytorch,
        report,
    )


def kept_files(root: Path) -> list[Path]:
    folder = root / "cache" / "migraphx_program_cache"
    return list(folder.glob(f"*{build_cache.SUFFIX}"))


def test_the_route_is_migraphx_only_where_its_package_can_run(amd_pc, monkeypatch):
    assert auto.migraphx_here()
    assert auto._route() is auto._MIGraphX
    # Any NVIDIA GPU keeps NVIDIA's route, whatever else is here.
    monkeypatch.setattr(auto, "nvidia", SimpleNamespace(is_available=True))
    assert not auto.migraphx_here()
    assert auto._route() is auto._TensorRT
    # A processor's small graphics are no route, and neither GPU is none.
    monkeypatch.setattr(auto, "nvidia", SimpleNamespace(is_available=False))
    small = AmdInfo([AmdDevice(0, "Radeon Graphics", "gfx1036", compute_units=2)])
    monkeypatch.setattr(auto, "amd", small)
    assert not auto.migraphx_here()
    monkeypatch.setattr(auto, "amd", AmdInfo([]))
    assert not auto.migraphx_here()


def test_a_model_compiles_once_and_runs_through_the_runner(amd_pc):
    fakes, root = amd_pc
    context = Context(root)
    report = TileReport()
    for _ in range(3):
        output = run(context, Model(), TileSize(256), report=report)
        np.testing.assert_array_equal(
            output, IMAGE.repeat(2, 0).repeat(2, 1).clip(0, 1)
        )
    assert fakes.conversions == 1
    assert fakes.compiles == [(256, 256)]
    assert context.warnings == []
    assert len(Runner.made) == 1
    assert report.text == "Whole image"


def test_a_later_process_reads_the_program_from_disk_by_the_weights(amd_pc):
    fakes, root = amd_pc
    context = Context(root)
    run(context, Model(weight=1.0))
    assert len(kept_files(root)) == 1

    # A restart: nothing in memory, the model loaded anew from the same file.
    auto._outcomes.clear()
    build_cache._memory.clear()
    run(context, Model(weight=1.0))
    assert len(fakes.compiles) == 1

    run(context, Model(weight=2.0))
    assert len(fakes.compiles) == 2
    assert len(kept_files(root)) == 2


@pytest.mark.parametrize(
    ("tile", "size"),
    [
        (TileSize(128), 128),
        (TileSize(4096), 4096),
        (TileSize(100), 100),
        (ESTIMATE, 512),
        (MAX_TILE_SIZE, 512),
        (NO_TILING, 512),
    ],
)
def test_the_program_is_exactly_the_tile_size(amd_pc, tile, size):
    fakes, root = amd_pc
    run(Context(root), Model(), tile)
    assert fakes.compiles == [(size, size)]


@pytest.mark.parametrize(
    ("recipe", "tile", "multiple", "minimum", "size"),
    [
        ("fp16", 100, 64, 0, 128),
        ("fp16", 128, 64, 0, 128),
        ("fp16", 64, 8, 100, 104),
        # DUAL's ONNX takes multiples of 4, which its size requirement does not name.
        ("dual", 101, 1, 4, 104),
        ("dual", 104, 1, 4, 104),
    ],
)
def test_the_program_aligns_up_to_the_models_multiple(
    recipe, tile, multiple, minimum, size
):
    model = Model(multiple=multiple, minimum=minimum)
    assert auto._program_size(model, recipe, tile) == size


def test_a_legacy_drfts_program_is_at_least_two_windows():
    model = Model(arch="DRFT", minimum=16)
    model.model.window_size = 64
    assert auto._program_size(model, "drft", 64) == 128
    assert auto._program_size(model, "drft", 256) == 256


def test_each_tile_size_has_its_own_program_and_key(amd_pc):
    fakes, root = amd_pc
    context, model = Context(root), Model()
    for tile in (TileSize(256), TileSize(1024), TileSize(256), ESTIMATE):
        run(context, model, tile)
    assert fakes.compiles == [(256, 256), (1024, 1024), (512, 512)]
    assert len(kept_files(root)) == 3
    assert auto._program_key(context, model, "fp16", 256) != auto._program_key(
        context, model, "fp16", 1024
    )
    # One size is one program, whichever tiles align to it.
    aligned = Model(multiple=64)
    assert auto._program_key(context, aligned, "fp16", 100) == auto._program_key(
        context, aligned, "fp16", 128
    )


def test_the_key_follows_the_weights_the_gpu_the_runner_and_the_route(
    amd_pc, monkeypatch
):
    _fakes, root = amd_pc
    context = Context(root)
    base = auto._program_key(context, Model(), "fp16", 512)
    assert base == auto._program_key(context, Model(), "fp16", 512)
    assert base != auto._program_key(context, Model(weight=2.0), "fp16", 512)
    assert base != auto._program_key(context, Model(arch="SPAN"), "fp16", 512)
    assert base != auto._program_key(context, Model(), "dual", 512)
    monkeypatch.setattr("nodes.impl.migraphx.runner.runner_identity", lambda: "other")
    assert base != auto._program_key(context, Model(), "fp16", 512)


def test_the_program_sources_exist():
    # A source renamed away would turn every run into a fallback.
    for source in auto._PROGRAM_SOURCES:
        assert source.is_file(), source


def test_rgt_stops_at_its_size_and_says_so_once(amd_pc):
    fakes, root = amd_pc
    context, model = Context(root), Model(arch="RGT")
    for _ in range(2):
        run(context, model, TileSize(512))
    assert fakes.compiles == [(256, 256)]
    assert len(context.warnings) == 1
    assert context.warnings[0].startswith("Use MIGraphX Program:")
    assert "program stops at 256 px" in context.warnings[0]
    assert "512 px" in context.warnings[0]


@pytest.mark.parametrize("shape", [(4, 6, 3), (4, 6), (63, 512, 3)])
def test_a_small_image_runs_through_the_program_at_its_own_size(amd_pc, shape):
    """Padding it is the runner's (native/trt_runner/frame.cpp, which needs an AMD GPU to
    run): this route never leaves it to PyTorch, and its output has the image's size."""
    fakes, root = amd_pc
    context = Context(root)
    image = np.full(shape, 0.25, dtype=np.float32)
    report = TileReport()
    output = run(context, Model(), TileSize(512), image, report)
    assert output.shape[:2] == (shape[0] * 2, shape[1] * 2)
    assert np.allclose(output, 0.25)
    assert fakes.compiles == [(512, 512)]
    assert context.warnings == []
    assert report.text == "Whole image"


def test_a_larger_image_reports_the_programs_tiles(amd_pc):
    _fakes, root = amd_pc
    report = TileReport()
    image = np.full((600, 300, 3), 0.25, dtype=np.float32)
    assert run(Context(root), Model(), TileSize(256), image, report).shape[:2] == (
        1200,
        600,
    )
    assert report.text == "256x256 px"


def test_a_failed_build_falls_back_says_why_and_is_not_retried_in_the_chain(amd_pc):
    fakes, root = amd_pc
    context, model = Context(root), Model()
    fakes.compile_error = RuntimeError("MIGraphX program build failed")
    for _ in range(3):
        np.testing.assert_array_equal(run(context, model), IMAGE * 3)
    assert len(context.warnings) == 1
    assert "not through a MIGraphX program" in context.warnings[0]
    assert "MIGraphX program build failed" in context.warnings[0]
    assert kept_files(root) == []

    context.end_chain()
    fakes.compile_error = None
    assert run(context, model).shape == (8, 12, 3)
    assert fakes.compiles == [(512, 512)]


def test_stop_during_the_build_is_stop(amd_pc):
    fakes, root = amd_pc
    context = Context(root)
    fakes.compile_error = Aborted()
    with pytest.raises(Aborted):
        run(context, Model())
    assert context.warnings == []


def test_a_wrong_scale_falls_back(amd_pc):
    fakes, root = amd_pc
    context = Context(root)
    fakes.program_scale = 4
    np.testing.assert_array_equal(run(context, Model()), IMAGE * 3)
    assert "upscales 4x" in context.warnings[0]


def test_cpu_mode_stays_on_pytorch_and_says_why(amd_pc):
    _fakes, _root = amd_pc
    assert auto.no_engine_here(torch.device("cpu")) == "PyTorch is set to the CPU."
    assert auto.no_engine_here(GPU) is None


def test_a_package_that_is_not_installed_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(auto, "nvidia", SimpleNamespace(is_available=False))
    monkeypatch.setattr(
        auto, "amd", AmdInfo([AmdDevice(0, "Radeon", "gfx1100", compute_units=96)])
    )
    with package_imports():
        import_package(monkeypatch, tmp_path, Here(dependencies=[ABSENT]))
        assert "not installed" in str(auto.no_engine_here(GPU))


def test_a_package_that_did_not_load_says_why(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(auto, "nvidia", SimpleNamespace(is_available=False))
    monkeypatch.setattr(
        auto, "amd", AmdInfo([AmdDevice(0, "Radeon", "gfx1100", compute_units=96)])
    )
    with package_imports():
        import_package(monkeypatch, tmp_path, Here(load_error=OSError("no HIP")))
        assert "no HIP" in str(auto.no_engine_here(GPU))


def test_a_failing_session_gives_the_whole_image_to_pytorch(amd_pc, monkeypatch):
    _fakes, root = amd_pc
    context, model = Context(root), Model()
    run(context, model)
    monkeypatch.setattr(
        Runner, "upscale", lambda *_a: (_ for _ in ()).throw(RuntimeError("HIP error"))
    )
    np.testing.assert_array_equal(run(context, model), IMAGE * 3)
    assert "HIP error" in context.warnings[0]


def test_invalid_values_fall_back(amd_pc, monkeypatch):
    _fakes, root = amd_pc
    context = Context(root)

    def invalid(_self, _inputs, outputs):
        for out in outputs:
            out[...] = np.nan

    monkeypatch.setattr(Runner, "upscale", invalid)
    np.testing.assert_array_equal(run(context, Model()), IMAGE * 3)
    assert "returned invalid values" in context.warnings[0]


def test_a_program_file_that_migraphx_refuses_is_built_again(amd_pc, monkeypatch):
    fakes, root = amd_pc
    context = Context(root)
    run(context, Model())
    assert len(fakes.compiles) == 1
    context.end_chain()

    # A restart, and the record is whole but MIGraphX cannot start it.
    auto._outcomes.clear()
    build_cache._memory.clear()
    tensorrt_cache.release_node(NODE)
    refused = RuntimeError("Failed to load the program")
    real = Runner.__init__

    def refusing(self, *args):
        raise refused

    monkeypatch.setattr(Runner, "__init__", refusing)
    model = Model()
    np.testing.assert_array_equal(run(context, model), IMAGE * 3)
    assert kept_files(root) == []
    assert "built again" in context.warnings[0]
    assert "MIGraphX could not start" in context.warnings[0]

    # The next image builds.
    monkeypatch.setattr(Runner, "__init__", real)
    assert run(context, model).shape == (8, 12, 3)
    assert len(fakes.compiles) == 2


@pytest.mark.parametrize(
    "message",
    [
        "HIP error hipErrorOutOfMemory in allocating a buffer: out of memory",
        "MIGraphX: Out of memory on the device",
    ],
)
def test_a_kept_program_short_of_memory_is_not_built_again(
    amd_pc, monkeypatch, message
):
    fakes, root = amd_pc
    context = Context(root)
    run(context, Model())
    auto._outcomes.clear()
    build_cache._memory.clear()
    tensorrt_cache.release_node(NODE)

    def short(self, *args):
        raise RuntimeError(message)

    monkeypatch.setattr(Runner, "__init__", short)
    np.testing.assert_array_equal(run(context, Model()), IMAGE * 3)
    assert len(fakes.compiles) == 1
    assert len(kept_files(root)) == 1


def test_clear_frees_the_session(amd_pc):
    _fakes, root = amd_pc
    run(Context(root), Model())
    assert [runner.closed for runner in Runner.made] == [False]
    auto.release_engine(NODE)
    assert [runner.closed for runner in Runner.made] == [True]


def test_the_server_clear_hook_frees_it_too(amd_pc):
    _fakes, root = amd_pc
    run(Context(root), Model())
    tensorrt_cache.release_node(NODE)
    assert [runner.closed for runner in Runner.made] == [True]


def test_the_program_cache_setting_names_the_folder(amd_pc, tmp_path):
    _fakes, root = amd_pc
    chosen = tmp_path / "chosen"

    class Chosen(Context):
        @property
        def settings(self):
            other = SimpleNamespace(get_cache_location=lambda _key: str(chosen))
            parser = SettingsParser({"gpu_index": "0"})
            return SimpleNamespace(
                of_package=lambda _name: other,
                get_int=parser.get_int,
                get_cache_location=parser.get_cache_location,
            )

    run(Chosen(root), Model())
    assert len(list(chosen.glob(f"*{build_cache.SUFFIX}"))) == 1
    assert kept_files(root) == []


def test_the_real_nodes_take_these_arguments(amd_pc):
    """A call that no longer fits would only show as a fallback on every image."""
    fakes, _root = amd_pc
    build = fakes.imported.nodes["utility.build_program"]
    inspect.signature(build.build_program_node).bind(
        object(), object(), build.ProgramPrecision.FP16, 512, 512, False, False
    )
    upscale = fakes.imported.nodes["processing.upscale_image"]
    inspect.signature(upscale.program_upscale).bind(
        object(), IMAGE, object(), False, before_clip=print
    )


def test_migraphxs_own_upscale_image_still_returns_only_its_image(amd_pc):
    fakes, root = amd_pc
    context = Context(root)
    program = MIGraphXProgram(b"a program", info((256, 256)))
    output = fakes.imported.nodes["processing.upscale_image"].upscale_image_node(
        context, IMAGE, program, False
    )
    assert output.shape == (8, 12, 3)


def test_forgetting_a_program_takes_its_file_and_its_copy_in_memory(tmp_path):
    build_cache._memory.clear()
    program = MIGraphXProgram(b"a program", info((256, 256)))
    build_cache.cached_program("a" * 64, lambda: program, str(tmp_path))
    build_cache.forget_program("a" * 64, str(tmp_path))
    assert list(tmp_path.iterdir()) == []
    assert build_cache._recalled("a" * 64) is None
    # Gone already: nothing to do.
    build_cache.forget_program("a" * 64, str(tmp_path))


def test_dual_library_has_this_systems_name():
    assert dual_route.OP_LIBRARY.name == (
        "dual_migx_ops.dll"
        if dual_route.sys.platform == "win32"
        else "dual_migx_ops.so"
    )


def test_duals_identity_follows_triton_for_windows(monkeypatch: pytest.MonkeyPatch):
    def only_windows(version: str):
        def lookup(name: str) -> str:
            if name != "triton-windows":
                raise importlib.metadata.PackageNotFoundError(name)
            return version

        return lookup

    identities = []
    for version in ("3.8.0.post29", "3.8.0.post30"):
        monkeypatch.setattr(
            dual_route.importlib.metadata, "version", only_windows(version)
        )
        dual_route.build_identity.cache_clear()
        identities.append(dual_route.build_identity())
    dual_route.build_identity.cache_clear()
    assert identities[0] != identities[1]


def test_duals_identity_follows_the_vendored_exporter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A program kept by the weights is not read for a graph a later exporter writes."""
    names = {path.name for path in dual_route.EXPORTER_SOURCES}
    assert {"export.py", "dynamic.py", "graph.py", "spec.py", "dual_arch.py"} <= names
    exporter = tmp_path / "graph.py"
    monkeypatch.setattr(dual_route, "EXPORTER_SOURCES", (exporter,))
    identities = []
    for content in (b"one", b"two"):
        exporter.write_bytes(content)
        dual_route.build_identity.cache_clear()
        identities.append(dual_route.build_identity())
    dual_route.build_identity.cache_clear()
    assert identities[0] != identities[1]


NA_MANIFEST = {
    "specialization": {"dynamic": True, "factory": "dual_s", "scale": 2},
    "context": "na-v1",
    "local": "na-v1",
    "retrieval": "na-v1",
}


@pytest.mark.parametrize(
    ("keys", "named"),
    [
        (("local", "retrieval"), "local and retrieval"),
        (("local",), "local"),
        (("retrieval",), "retrieval"),
    ],
)
def test_a_dual_onnx_with_local_or_retrieval_na_has_no_program(
    monkeypatch: pytest.MonkeyPatch, keys: tuple[str, ...], named: str
):
    monkeypatch.setattr(dual_route, "run_child", pytest.fail)
    manifest = {"specialization": NA_MANIFEST["specialization"], "context": "na-v1"}
    manifest.update(dict.fromkeys(keys, "na-v1"))
    with pytest.raises(ValueError, match="has no MIGraphX program") as refused:
        dual_route.build_program(b"onnx", manifest, 0, (64, 64))
    assert f"ONNX's {named} neighborhood attention" in str(refused.value)


def test_a_dual_onnx_from_before_local_and_retrieval_na_builds_with_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    def child(args: list[str], *_: object, **__: object) -> str:
        Path(args[3]).write_bytes(b"lowered")
        return ""

    library = tmp_path / dual_route.OP_LIBRARY.name
    library.write_bytes(b"library")
    monkeypatch.setattr(dual_route, "OP_LIBRARY", library)
    monkeypatch.setattr(dual_route, "run_child", child)
    monkeypatch.setattr(dual_route, "gfx_of", lambda _gpu: "gfx1100")
    monkeypatch.setattr(
        dual_route.program, "compile_in_child", lambda *_a, **_kw: b"program"
    )
    for manifest in (
        {"specialization": {"dynamic": False, "height": 64, "width": 64}},
        {
            "specialization": {"dynamic": False, "height": 64, "width": 64},
            "context": "na-v1",
        },
    ):
        caplog.clear()
        assert dual_route.build_program(b"onnx", manifest, 0, None) == b"program"
        warned = [
            r
            for r in caplog.records
            if "old local windows and retrieval blocks" in r.getMessage()
        ]
        assert len(warned) == 1


def test_the_vendored_exporter_writes_local_and_retrieval_na():
    dual_route.exported_tensorrt_only_keys.cache_clear()
    assert dual_route.exported_tensorrt_only_keys() == ["local", "retrieval"]


def test_a_dual_falls_back_before_exporting_with_one_warning(
    amd_pc, monkeypatch: pytest.MonkeyPatch
):
    """The exporter's every ONNX has local and retrieval NA, which Build Program refuses:
    the model runs on PyTorch on ROCm, said once in the chain, and nothing is exported."""
    fakes, root = amd_pc
    monkeypatch.setattr(fakes.imported.dual, "tensorrt_only", ["local", "retrieval"])
    context, model = Context(root), Model(arch="DUAL")
    for _ in range(3):
        np.testing.assert_array_equal(run(context, model), IMAGE * 3)
    (warning,) = context.warnings
    assert "runs this DUAL model on PyTorch, not through a MIGraphX program" in warning
    assert "local and retrieval: no program" in warning
    assert fakes.conversions == 0
    assert kept_files(root) == []


def test_a_dual_onnx_with_local_and_retrieval_na_falls_back_with_one_warning(
    amd_pc, monkeypatch: pytest.MonkeyPatch
):
    """An exporter that writes no such keys exports; an ONNX that still holds them lowers
    to no program (dual.build_program refuses it before any child), so the model runs on
    PyTorch, said once in the chain."""
    fakes, root = amd_pc
    monkeypatch.setattr(
        fakes.imported.nodes["utility.build_program"],
        "model_facts",
        lambda _m: SimpleNamespace(drft_manifest=None, dual_manifest=NA_MANIFEST),
    )
    monkeypatch.setattr(fakes.imported.dual, "build_program", dual_route.build_program)
    monkeypatch.setattr(dual_route, "run_child", pytest.fail)
    context, model = Context(root), Model(arch="DUAL")
    for _ in range(3):
        np.testing.assert_array_equal(run(context, model), IMAGE * 3)
    (warning,) = context.warnings
    assert "runs this DUAL model on PyTorch, not through a MIGraphX program" in warning
    assert "local and retrieval neighborhood attention" in warning
    assert fakes.conversions == 1
    assert fakes.compiles == []
    assert kept_files(root) == []
