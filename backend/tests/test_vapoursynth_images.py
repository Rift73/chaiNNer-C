"""Images in the VapourSynth nodes: which methods take one (frame_methods.json, measured
by native/tools/vs_frame_probe.py), the types that refuse the others when connecting, and
in real VapourSynth the conversion to a clip and back, the same function on an image and
on a clip, and the chain run's frame process."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, cast

import cv2
import numpy as np
import pytest

# chaiNNer's inputs load the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

import process
from api import (
    Aborted,
    ExecutionOptions,
    InputId,
    Lazy,
    NodeContext,
    NodeData,
    NodeId,
    OutputId,
    Package,
    SettingsParser,
)
from chain.chain import Chain, Edge, EdgeSource, EdgeTarget, FunctionNode, GeneratorNode
from events import EventConsumer
from native_executor import NativeExecutor
from nodes.impl.image_utils import normalize
from nodes.impl.upscale.auto_split_tiles import TileSize
from nodes.impl.vapoursynth import catalog, runner
from nodes.impl.vapoursynth.catalog import CLASS, Catalog, load_catalog, overlay_paths
from nodes.impl.vapoursynth.runner import VSScriptError, clip_props, read_frames
from nodes.impl.vapoursynth.schema_gen import (
    _MAIN_HELP,
    CLIP_OPTION,
    FAMILIES,
    FILTER_ID,
    FRAME_OPTION,
    MAIN_ID,
    MAIN_TYPE,
    METHOD_ID,
    MODE_ID,
    OUTPUT_TYPE,
    SECOND_TYPE,
    input_ids,
    register_families,
)
from nodes.impl.vapoursynth.script import (
    VideoClip,
    VSFilter,
    compile_script,
    image_clip,
)
from packages.chaiNNer_standard.image.batch_processing import load_images
from packages.chaiNNer_standard.image.io import save_image
from packages.chaiNNer_standard.image.io.load_image import load_image_node
from packages.chaiNNer_vapoursynth.chains import (
    chain_dir,
    filter_image,
    image_node_output,
    render_images,
)
from packages.chaiNNer_vapoursynth.vapoursynth.ai_upscale import (
    AiUpscaleBackend,
    AiUpscalePrecision,
    ai_upscale_node,
)
from packages.chaiNNer_vapoursynth.vapoursynth.split_planes import split_planes_node
from process import ExecutionId, Executor, enforce_inputs

TABLE = json.loads(catalog.FRAME_METHODS.read_text(encoding="utf-8"))
KEY = {"vsjetpack": TABLE["vsjetpack"], "vapoursynth": TABLE["vapoursynth"]}
NODE = "vs-image"
HEIGHT, WIDTH = 48, 64


def _param(name: str, annotation: str, default: str | None) -> dict[str, Any]:
    return {
        "name": name,
        "annotation": annotation,
        "default": default,
        "kind": "keyword",
    }


def _function(name: str, *params: dict[str, Any]) -> dict[str, Any]:
    module = name.partition(".")[0]
    return {
        "id": name,
        "module": module,
        "source": "jetpack",
        "kind": "function",
        "call": name.partition(".")[2],
        "import": f"from {module} import {name.partition('.')[2]}",
        "returns": "VideoNode",
        "params": [_param("clip", "VideoNode", None), *params],
        "init_params": [],
        "doc": "",
        "arg_docs": {},
    }


def _small_catalog(tmp_path: Path, **key: str) -> Path:
    functions = [
        # What the table says of them decides: one that looks at one frame, one that
        # does with tr = 0, and one that looks at other frames.
        _function("vsrgtools.box_blur"),
        _function("vsdenoise.nl_means", _param("tr", "int", "1")),
        _function("vsdenoise.mc_degrain"),
        _function("vstools.merge", _param("other", "VideoNode", None)),
    ]
    for function in functions:
        function["module"] = "vsrgtools"
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {"schema": 1, "functions": functions, "enums": {}, "key": {**KEY, **key}}
        ),
        encoding="utf-8",
    )
    return path


def _frame_table(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    table = {
        **KEY,
        "frame": {
            "vsrgtools.box_blur": {},
            "vsdenoise.nl_means": {"args": {"tr": 0}},
            "vstools.merge": {"route": "yuv"},
        },
        "clip_only": {"vsdenoise.mc_degrain": "temporal"},
    }
    path = tmp_path / "frame_methods.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    monkeypatch.setattr(catalog, "FRAME_METHODS", path)


def _nodes(cat: Catalog, families: Iterable[str] = FAMILIES) -> dict[str, NodeData]:
    package = Package(
        where=__file__, id="t", name="T", description="", icon="", color=""
    )
    group = package.add_category("VS", "", "", "").add_node_group("Families")
    register_families(cat.methods, group, cat.enums, list(families), image_node_output)
    return {node.name: node for node in group.nodes}


@pytest.fixture
def small(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Catalog:
    _frame_table(monkeypatch, tmp_path)
    return load_catalog(_small_catalog(tmp_path), [])


def _methods(cat: Catalog) -> dict[str, Any]:
    return {method.id: method for method in cat.methods}


def test_a_method_takes_images_as_measured(small: Catalog) -> None:
    methods = _methods(small)
    assert methods["vsrgtools.box_blur"].frame_route == "rgb"
    assert methods["vstools.merge"].frame_route == "yuv"
    assert methods["vsdenoise.nl_means"].frame_args == {"tr": 0}
    assert methods["vsdenoise.mc_degrain"].frame_route == ""


def test_other_versions_take_clips_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _frame_table(monkeypatch, tmp_path)
    path = _small_catalog(tmp_path, vsjetpack="99")
    assert not any(m.frame_route for m in load_catalog(path, []).methods)


def test_an_overlay_says_otherwise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _frame_table(monkeypatch, tmp_path)
    overlay = tmp_path / "mine.toml"
    overlay.write_text(
        '[method."vsdenoise.mc_degrain"]\nimages = true\n'
        '[method."vsrgtools.box_blur"]\nimages = false\n',
        encoding="utf-8",
    )
    methods = _methods(load_catalog(_small_catalog(tmp_path), [overlay]))
    assert methods["vsdenoise.mc_degrain"].frame_route == "rgb"
    assert methods["vsrgtools.box_blur"].frame_route == ""


def _inputs(node: NodeData) -> dict[int, Mapping[str, Any]]:
    return {i.id: i.to_dict() for i in node.inputs}


def test_the_types_say_which_method_takes_an_image(small: Catalog) -> None:
    family = small.methods[0].family
    node = _nodes(small, [family])[family]
    inputs = _inputs(node)
    assert inputs[MAIN_ID]["type"] == [MAIN_TYPE, "null"]
    assert inputs[MAIN_ID]["followsInputs"] is True
    options = {o["value"]: o["type"] for o in inputs[METHOD_ID]["options"]}
    assert options == {
        "vsrgtools.box_blur": FRAME_OPTION,
        "vsdenoise.nl_means": FRAME_OPTION,
        "vstools.merge": FRAME_OPTION,
        "vsdenoise.mc_degrain": CLIP_OPTION,
    }
    other = input_ids(family, [("vstools.merge", "other")])["vstools.merge", "other"]
    assert inputs[other]["type"] == SECOND_TYPE
    assert OUTPUT_TYPE in str(node.outputs[0].to_dict()["type"])


def test_a_family_of_clips_only_keeps_its_types(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _frame_table(monkeypatch, tmp_path)
    path = _small_catalog(tmp_path, vsjetpack="99")
    cat = load_catalog(path, [])
    family = cat.methods[0].family
    node = _nodes(cat, [family])[family]
    inputs = _inputs(node)
    assert inputs[MAIN_ID]["type"] == ["VideoClip", "null"]
    assert inputs[MAIN_ID]["followsInputs"] is False
    assert all("type" not in o for o in inputs[METHOD_ID]["options"])
    assert "Image" not in str(node.outputs[0].to_dict()["type"])


def test_the_table_lists_a_method_once() -> None:
    assert not set(TABLE["frame"]) & set(TABLE["clip_only"])
    for entry in TABLE["frame"].values():
        assert set(entry) <= {"route", "args", "min_side"}
        assert entry.get("route", "rgb") in ("rgb", "yuv", "luma")


# As measured: frame n's output changes with other frames, or the clip's length changes.
PINNED_TEMPORAL = [
    "std.AverageFrames",
    "vsdenoise.mc_degrain",
    "vsdenoise.mpeg2stinx",
    "vsrgtools.flux_smooth",
    "vstools.sc_detect",
    "bwdif.Bwdif",
    "zsmooth.TemporalMedian",
    "zsmooth.TemporalSoften",
    "vivtc.VDecimate",
    "std.Reverse",
    "std.Trim",
    "std.Splice",
    "vstools.shift_clip",
]
PINNED_FRAME = [
    "std.BoxBlur",
    "std.Invert",
    "std.Crop",
    "std.Expr",
    "std.Levels",
    "vskernels.*.scale",
    "vsrgtools.box_blur",
    "vsrgtools.gauss_blur",
    "vsmasktools.*.edgemask",
    "vsdehalo.dehalo_alpha",
    "cas.CAS",
]
# A parameter switches these: an image's run sets it, and what is left looks at one frame.
PINNED_SWITCH = {
    "vsdenoise.nl_means": {"tr": 0},
    "vsdenoise.bm3d": {"tr": 0},
    "vsdenoise.DFTTest.denoise": {"tr": 0},
    "vsrgtools.soothe": {"temporal_radius": 0},
}


@pytest.mark.parametrize("method", PINNED_TEMPORAL)
def test_temporal_methods_stay_clip_only(method: str) -> None:
    assert method in TABLE["clip_only"]


@pytest.mark.parametrize(
    "method",
    [
        "vstools.get_u",
        "vstools.get_v",
        "std.MakeDiff",
        "std.MakeFullDiff",
        "std.MergeDiff",
        "std.MergeFullDiff",
        "atwt.ExtractFrequency",
        "atwt.ReplaceFrequency",
        "bm3d.RGB2OPP",
    ],
)
def test_signed_values_stay_clip_only(method: str) -> None:
    assert TABLE["clip_only"][method] == "signed"


# As measured: the same result whatever green and blue are. Detail Mask and the difference masks
# follow all three planes; Texture Mask gives one value on any input, on any route.
@pytest.mark.parametrize(
    "method",
    [
        "vsmasktools.limited_linemask",
        "vsmasktools.luma_credit_mask",
        "vsmasktools.ringing_mask",
        "vszip.AdaptiveGrainMask",
    ],
)
def test_luma_masks_run_on_yuv(method: str) -> None:
    assert TABLE["frame"][method]["route"] == "yuv"


@pytest.mark.parametrize(
    "method",
    [
        "vsrgtools.median_blur",
        "zsmooth.InterQuartileMean",
        "zsmooth.Median",
        "zsmooth.SmartMedian",
        "zsmooth.RemoveGrain",
        "zsmooth.Repair",
    ],
)
def test_methods_that_crash_on_tiny_images_say_so(method: str) -> None:
    assert TABLE["frame"][method]["min_side"] == 4


@pytest.mark.parametrize("method", PINNED_FRAME)
def test_frame_methods_take_images(method: str) -> None:
    assert method in TABLE["frame"]


@pytest.mark.parametrize(("method", "args"), PINNED_SWITCH.items())
def test_switched_methods_set_the_parameter(method: str, args: dict) -> None:
    assert TABLE["frame"][method]["args"] == args


def test_a_temporal_method_refuses_an_image_at_run_time_too(small: Catalog) -> None:
    family = small.methods[0].family
    node = _nodes(small, [family])[family]
    with pytest.raises(ValueError, match="takes clips only, not images"):
        _run(node, "vsdenoise.mc_degrain", np.zeros((8, 8, 3), np.float32), None)


def test_images_and_clips_do_not_mix(small: Catalog) -> None:
    family = small.methods[0].family
    node = _nodes(small, [family])[family]
    other = input_ids(family, [("vstools.merge", "other")])["vstools.merge", "other"]
    clip = VideoClip("n_a_0", "n_a_0 = 1", (), frozenset(), "a")
    values = {other: np.zeros((8, 8, 3), np.float32)}
    with pytest.raises(ValueError, match="do not mix"):
        _run(node, "vstools.merge", clip, None, raw=values)


# In real VapourSynth.


class Context:
    def __init__(self, storage: Path, node_id: str = NODE):
        self.storage_dir = storage
        self.node_id = node_id
        self.settings = SettingsParser({"cache_mb": 256})
        self.aborted = False

    def check_aborted(self) -> None:
        pass

    def add_cleanup(self, fn: Any, after: str = "chain") -> None:
        pass


def _context(storage: Path) -> NodeContext:
    return cast(NodeContext, Context(storage))


def _run(
    node: NodeData,
    method: str,
    main: object,
    storage: Path | None,
    raw: Mapping[int, object] | None = None,
    choice: Mapping[int, object] | None = None,
) -> Any:
    """Runs a family node as the executor does; `raw` by input id."""
    given: dict[int, object] = {MAIN_ID: main, METHOD_ID: method, MODE_ID: "simple"}
    given[FILTER_ID] = 0
    given.update(raw or {})
    given.update(choice or {})
    values = [
        given[i.id] if i.id in given else i.to_dict().get("def") for i in node.inputs
    ]
    enforced = enforce_inputs(values, node, NodeId(NODE), [])
    context = _context(storage) if storage else SimpleNamespace(node_id=NODE)
    return node.run(context, *enforced)


@pytest.fixture(scope="module")
def vs() -> Any:
    return pytest.importorskip("vapoursynth")


@pytest.fixture(autouse=True, params=["frame process", "vspipe"])
def evaluation(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterable[str]:
    """Each test on both routes: they must give the same, and the second is what remains
    of an image's run without the first."""
    if request.param == "vspipe":
        monkeypatch.setattr(runner, "evaluate_script", runner.evaluate_in_vspipe)
    yield request.param
    runner.end_frame_process()


@pytest.fixture(scope="module")
def real_nodes(
    vs: Any, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, NodeData]:
    storage = tmp_path_factory.mktemp("vapoursynth")
    path = catalog.ensure_catalog(storage)
    assert path is not None
    return _nodes(load_catalog(path, overlay_paths(storage)))


def _picture(seed: int, height: int = HEIGHT, width: int = WIDTH) -> np.ndarray:
    """A float32 BGR picture with detail, values 0 and 1 among them."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    picture = (
        np.stack(
            (
                0.5 + 0.4 * np.sin(x / 5 + seed) * np.cos(y / 4),
                0.3 + 0.5 * x / width,
                0.8 - 0.6 * y / height,
            ),
            axis=2,
        )
        + rng.standard_normal((height, width, 3), dtype=np.float32) * 0.02
    )
    picture = np.clip(picture, 0, 1).astype(np.float32)
    picture[0, 0] = (0, 1, 1e-30)
    return picture


def _clip_of(frames: Path, count: int, var: str = "n_source_0") -> VideoClip:
    """A clip of `count` frames saved in one NumPy file, by another route than an image's
    clip takes, tagged the same."""
    code = f"""
{var}_frames = __import__('numpy').load({str(frames)!r})
{var}_blank = core.std.BlankClip(format=vs.RGBS, width={var}_frames.shape[2], height={var}_frames.shape[1], length={count}, keep=True)
def {var}_fill(n, f):
    out = f.copy()
    for plane in range(3):
        __import__('numpy').copyto(__import__('numpy').asarray(out[plane]), {var}_frames[n, :, :, plane])
    return out
{var} = core.std.ModifyFrame({var}_blank, {var}_blank, {var}_fill).std.SetFrameProps(_Transfer=13, _Primaries=1, _Range=1, _FieldBased=0, _Matrix=0)
"""
    return VideoClip(var, code.strip(), (), frozenset(), var)


def _source_frames(path: Path, pictures: list[np.ndarray]) -> None:
    """The pictures in VapourSynth's plane order: (frame, y, x, R G B)."""
    stacked = np.stack(
        [p[:, :, ::-1] if p.ndim == 3 else p[:, :, None] for p in pictures]
    )
    np.save(path, stacked, allow_pickle=False)


def _frame_on_clip(
    tmp_path: Path,
    node: NodeData,
    method: str,
    frame: int,
    choice: Mapping[int, object],
) -> np.ndarray:
    """The same function on a clip of different frames: frame `frame` of its output."""
    frames = tmp_path / "frames.npy"
    pictures = [_picture(i) for i in range(5)]
    _source_frames(frames, pictures)
    out = _run(node, method, _clip_of(frames, 5), None, choice=choice)
    script = tmp_path / "clip.vpy"
    script.write_text(compile_script([(out, "x")], cache_mb=256), encoding="utf-8")
    return next(iter(read_frames(script, 0, frame, frame + 1)))


def _choice(node: NodeData, method: str, **values: object) -> dict[int, object]:
    keys = input_ids(node.name, [(method, key) for key in values])
    return {keys[method, key]: value for key, value in values.items()}


@pytest.mark.parametrize(
    "image",
    [
        _picture(1),
        _picture(2)[:, :, 0],
        np.dstack((_picture(3), _picture(4)[:, :, 1])),
    ],
    ids=["rgb", "gray", "rgba"],
)
def test_a_round_trip_is_exact(vs: Any, tmp_path: Path, image: np.ndarray) -> None:
    def same(clips: Any) -> VideoClip:
        return clips[0]

    out = filter_image(_context(tmp_path), [image], same)
    assert out.dtype == np.float32
    assert out.shape == image.shape
    assert out.tobytes() == image.tobytes()


def test_a_round_trip_keeps_values_outside_0_1(vs: Any, tmp_path: Path) -> None:
    image = _picture(1)
    image[1, 1] = (-0.5, 1.5, np.float32(1e-40))
    out = render_images(_context(tmp_path), [image], lambda clips: clips)[0]
    assert out.tobytes() == image.tobytes()


@pytest.mark.parametrize(
    ("image", "tags"),
    [
        (_picture(1), {"_Matrix": 0, "_Transfer": 13, "_Primaries": 1, "_Range": 1}),
        (_picture(1)[:, :, 0], {"_Transfer": 13, "_Primaries": 1, "_Range": 1}),
    ],
    ids=["rgb", "gray"],
)
def test_an_image_clip_is_tagged_as_an_image(
    vs: Any, tmp_path: Path, image: np.ndarray, tags: dict[str, int]
) -> None:
    path = tmp_path / "in.npy"
    np.save(path, image, allow_pickle=False)
    clip = image_clip(path, NODE, 0, "rgb")
    script = tmp_path / "tags.vpy"
    script.write_text(compile_script([(clip, "x")], cache_mb=256), encoding="utf-8")
    info, props, _ = clip_props(script, 0)
    # Progressive, as every image is.
    assert props == {**tags, "_FieldBased": 0}
    assert info.format_name == ("GrayS" if image.ndim == 2 else "RGBS")
    assert info.num_frames == 1


@pytest.mark.parametrize(
    ("family", "method", "values"),
    [
        ("Sharpen and Blur", "vsrgtools.box_blur", {}),
        ("Sharpen and Blur", "std.BoxBlur", {}),
        ("Colour", "std.Invert", {}),
        ("Resize", "std.Crop", {"left": 4, "right": 4, "top": 2, "bottom": 2}),
    ],
)
def test_an_image_gives_what_a_clip_gives(
    real_nodes: dict[str, NodeData],
    tmp_path: Path,
    family: str,
    method: str,
    values: dict[str, object],
) -> None:
    node = real_nodes[family]
    choice = _choice(node, method, **values)
    on_image = _run(node, method, _picture(2), tmp_path, choice=choice)
    on_clip = _frame_on_clip(tmp_path, node, method, 2, choice)
    assert on_image.shape == on_clip.shape
    assert on_image.tobytes() == np.clip(on_clip, 0, 1).tobytes()


GRAINER = "vsdeband.Grainer"


def _has_noise_plugin(vs: Any) -> bool:
    return any(plugin.namespace == "noise" for plugin in vs.core.plugins())


def _grain_choice(node: NodeData, grain: str) -> dict[int, object]:
    return _choice(node, GRAINER, **{CLASS: grain, "strength": 4})


@pytest.mark.parametrize(
    "grain", ["GAUSS", "PERLIN", "SIMPLEX", "FBM_SIMPLEX", "POISSON"]
)
def test_grainer_grains_an_image_and_a_clip(
    vs: Any, real_nodes: dict[str, NodeData], tmp_path: Path, grain: str
) -> None:
    if not _has_noise_plugin(vs):
        pytest.skip("this VapourSynth has no noise plugin")
    node = real_nodes["Deband and Grain"]
    choice = _grain_choice(node, grain)
    picture = _picture(2)
    on_image = _run(node, GRAINER, picture, tmp_path, choice=choice)
    on_clip = _frame_on_clip(tmp_path, node, GRAINER, 2, choice)
    on_gray = _run(node, GRAINER, picture[:, :, 0], tmp_path, choice=choice)
    # Grain around the picture: a layer of what memory held would move it far.
    for grained, source in (
        (on_image, picture),
        (on_clip, picture),
        (on_gray, picture[:, :, 0]),
    ):
        assert grained.shape == source.shape
        difference = grained - source
        assert np.isfinite(difference).all()
        assert abs(difference.mean()) < 0.01
        planes = difference.reshape(-1, *difference.shape[2:]).std(axis=0)
        assert (planes > 0.0005).all() and (planes < 0.1).all()


def test_the_noise_plugins_own_function_stays_clip_only() -> None:
    # Its Poisson type leaves float samples unwritten, which its default type does not show.
    assert TABLE["clip_only"]["noise.Add"] == "unwritten"


def test_grainer_without_the_noise_plugin_names_it(
    vs: Any, real_nodes: dict[str, NodeData], tmp_path: Path
) -> None:
    if _has_noise_plugin(vs):
        pytest.skip("this VapourSynth has the noise plugin")
    node = real_nodes["Deband and Grain"]
    with pytest.raises(VSScriptError, match="name noise"):
        _run(node, GRAINER, _picture(2), tmp_path, choice=_grain_choice(node, "GAUSS"))


def test_a_parameter_that_uses_other_frames_is_switched_off(
    real_nodes: dict[str, NodeData], tmp_path: Path
) -> None:
    node = real_nodes["Denoise"]
    method = "vsdenoise.bm3d"
    backend = "vsdenoise.bm3d.Backend.OLD"
    # tr = 2 is what a user may have left: an image has no other frames.
    picked = _run(
        node,
        method,
        _picture(2),
        tmp_path,
        choice=_choice(node, method, tr=2, backend=backend),
    )
    forced = _run(
        node,
        method,
        _picture(2),
        tmp_path,
        choice=_choice(node, method, tr=0, backend=backend),
    )
    assert picked.tobytes() == forced.tobytes()
    assert not np.array_equal(forced, _picture(2))


def test_a_luma_method_runs_on_yuv_and_touches_the_whole_picture(
    real_nodes: dict[str, NodeData], tmp_path: Path
) -> None:
    node = real_nodes["Dehalo"]
    method = "vsdehalo.dehalo_alpha"
    image = _picture(2)
    assert TABLE["frame"][method]["route"] == "yuv"
    out = _run(node, method, image, tmp_path)
    # In RGB it would filter the first plane alone, red, and shift the colours.
    assert all(not np.array_equal(out[:, :, c], image[:, :, c]) for c in range(3))

    frames = tmp_path / "frame.npy"
    _source_frames(frames, [image])
    source = _clip_of(frames, 1)
    yuv = VideoClip(
        "n_yuv_0",
        "n_yuv_0 = n_source_0.resize.Point(format=vs.YUV444PS, matrix=1)",
        (source,),
        frozenset(),
        "yuv",
    )
    on_clip = _run(node, method, yuv, None)
    script = tmp_path / "clip.vpy"
    script.write_text(compile_script([(on_clip, "x")], cache_mb=256), encoding="utf-8")
    reference = next(iter(read_frames(script, 0, 0, 1)))
    # RGB from YUV goes a little outside 0-1, which an image's output clips.
    assert np.clip(out, 0, 1).tobytes() == np.clip(reference, 0, 1).tobytes()


def test_ai_upscale_on_an_image_gives_what_it_gives_on_a_clip(
    vs: Any, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    model = (
        Path(os.environ.get("CHAINNER_C_TEST_MODELS", "missing"))
        / "2x_Waifaux-NL3-SuperLite_latest_G.pth"
    )
    if not model.exists():
        pytest.skip("CHAINNER_C_TEST_MODELS names no folder with the test model")
    image = _picture(2)
    arguments = (
        model,
        AiUpscaleBackend.PYTORCH,
        AiUpscalePrecision.FP32,
        TileSize(0),
        256,
        1,
    )
    out = cast(np.ndarray, ai_upscale_node(_context(tmp_path), image, *arguments))
    assert out.shape == (HEIGHT * 2, WIDTH * 2, 3)

    frames = tmp_path / "frame.npy"
    _source_frames(frames, [image])
    clip = cast(
        VideoClip, ai_upscale_node(_context(tmp_path), _clip_of(frames, 1), *arguments)
    )
    script = tmp_path / "clip.vpy"
    script.write_text(compile_script([(clip, "x")], cache_mb=256), encoding="utf-8")
    assert out.tobytes() == next(iter(read_frames(script, 0, 0, 1))).tobytes()


def test_load_image_to_spatial_node_to_save_image(
    real_nodes: dict[str, NodeData], tmp_path: Path
) -> None:
    source = tmp_path / "in.png"
    expected_input = (_picture(5) * 65535).round().astype(np.uint16)
    cv2.imwrite(str(source), expected_input)
    # As Load Image's output gives it.
    image = normalize(load_image_node(source)[0])
    node = real_nodes["Colour"]
    out = _run(node, "std.Invert", image, tmp_path)
    _save_png(out, tmp_path, "out")
    saved = cv2.imread(str(tmp_path / "out.png"), cv2.IMREAD_UNCHANGED)
    assert saved.dtype == np.uint16
    assert np.array_equal(saved, 65535 - expected_input)


def _save_png(image: np.ndarray, folder: Path, name: str) -> None:
    node = save_image.save_image_node
    formats = save_image
    node(
        Lazy.ready(image),
        folder,
        None,
        name,
        formats.ImageFormat.PNG,
        formats.PngColorDepth.U16,
        False,
        95,
        formats.JpegSubsampling.FACTOR_422,
        False,
        formats.TiffColorDepth.U8,
        formats.TiffCompression.LZW,
        "BC1_UNORM",
        formats.BC7Compression.DEFAULT,
        formats.DDSErrorMetric.PERCEPTUAL,
        False,
        0,
        0,
        formats.DDSMipFilter.DEFAULT,
        False,
        formats.DDSColourConversion.KEEP,
        False,
        False,
        formats.AvifSubsampling.FACTOR_420,
        formats.ExrColorDepth.F16,
        formats.ExrCompression.ZIP,
        False,
        None,
    )


def test_each_item_of_a_folder_gets_its_own_result(
    real_nodes: dict[str, NodeData], tmp_path: Path
) -> None:
    node = real_nodes["Colour"]
    sizes = [(32, 48), (48, 32), (16, 16)]
    for seed, (height, width) in enumerate(sizes):
        image = _picture(seed, height, width)
        out = _run(node, "std.Invert", image, tmp_path)
        assert out.tobytes() == (1 - image).tobytes()
    # Nothing of an item stays in the node's folder.
    assert not list(chain_dir(_context(tmp_path)).iterdir())


def test_alpha_stays_when_the_method_moves_nothing(
    real_nodes: dict[str, NodeData], tmp_path: Path
) -> None:
    alpha = _picture(3)[:, :, 0]
    colour = _picture(2)
    out = _run(real_nodes["Colour"], "std.Invert", np.dstack((colour, alpha)), tmp_path)
    assert out.shape == (HEIGHT, WIDTH, 4)
    assert out[:, :, 3].tobytes() == alpha.tobytes()
    assert out[:, :, :3].tobytes() == (1 - colour).tobytes()


def test_alpha_moves_with_the_pixels(
    real_nodes: dict[str, NodeData], tmp_path: Path
) -> None:
    alpha = _picture(3)[:, :, 0]
    out = _run(
        real_nodes["Resize"],
        "std.FlipHorizontal",
        np.dstack((_picture(2), alpha)),
        tmp_path,
    )
    assert out[:, :, 3].tobytes() == alpha[:, ::-1].tobytes()
    assert out[:, :, :3].tobytes() == _picture(2)[:, ::-1].tobytes()


def test_alpha_follows_a_new_size(
    real_nodes: dict[str, NodeData], tmp_path: Path
) -> None:
    node = real_nodes["Resize"]
    method = "std.Crop"
    choice = _choice(node, method, left=8, right=0, top=0, bottom=16)
    alpha = _picture(3)[:, :, 0]
    out = _run(node, method, np.dstack((_picture(2), alpha)), tmp_path, choice=choice)
    assert out.shape == (HEIGHT - 16, WIDTH - 8, 4)
    assert out[:, :, 3].tobytes() == alpha[: HEIGHT - 16, 8:].tobytes()


def test_a_gray_result_has_no_alpha(
    real_nodes: dict[str, NodeData], tmp_path: Path
) -> None:
    image = np.dstack((_picture(2), _picture(3)[:, :, 0]))
    out = _run(real_nodes["Planes"], "vstools.get_r", image, tmp_path)
    assert out.ndim == 2


def test_a_second_image(real_nodes: dict[str, NodeData], tmp_path: Path) -> None:
    node = real_nodes["Planes"]
    method = "std.Merge"
    a, b = _picture(2), _picture(3)
    other = input_ids(node.name, [(method, "clipb")])[method, "clipb"]
    out = _run(node, method, a, tmp_path, raw={other: b})
    reference = _frame_on_clip_two(tmp_path, node, method, other, a, b)
    assert out.tobytes() == np.clip(reference, 0, 1).tobytes()


def _frame_on_clip_two(
    tmp_path: Path,
    node: NodeData,
    method: str,
    other: int,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    first, second = tmp_path / "a.npy", tmp_path / "b.npy"
    _source_frames(first, [a])
    _source_frames(second, [b])
    clip_a = _clip_of(first, 1)
    out = _run(node, method, clip_a, None, raw={other: _clip_of(second, 1, "n_b")})
    script = tmp_path / "two.vpy"
    script.write_text(compile_script([(out, "x")], cache_mb=256), encoding="utf-8")
    return next(iter(read_frames(script, 0, 0, 1)))


def test_split_planes_gives_gray_images(vs: Any, tmp_path: Path) -> None:
    image = _picture(2)
    r, g, b = cast("list[np.ndarray]", split_planes_node(_context(tmp_path), image))
    assert r.tobytes() == image[:, :, 2].tobytes()
    assert g.tobytes() == image[:, :, 1].tobytes()
    assert b.tobytes() == image[:, :, 0].tobytes()


def test_split_planes_refuses_what_is_not_rgb(vs: Any, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="takes an RGB image"):
        split_planes_node(_context(tmp_path), _picture(2)[:, :, 0])


def test_a_second_image_with_alpha_is_refused(
    real_nodes: dict[str, NodeData], tmp_path: Path
) -> None:
    node = real_nodes["Planes"]
    method = "std.Merge"
    other = input_ids(node.name, [(method, "clipb")])[method, "clipb"]
    transparent = np.dstack((_picture(3), _picture(4)[:, :, 0]))
    with pytest.raises(ValueError, match="remove its alpha"):
        _run(node, method, _picture(2), tmp_path, raw={other: transparent})


def test_real_nodes_type_their_inputs(real_nodes: dict[str, NodeData]) -> None:
    denoise = _inputs(real_nodes["Denoise"])
    assert denoise[MAIN_ID]["type"] == [MAIN_TYPE, "null"]
    options = {o["value"]: o["type"] for o in denoise[METHOD_ID]["options"]}
    assert options["vsdenoise.mc_degrain"] == CLIP_OPTION
    assert options["vsdenoise.bm3d"] == FRAME_OPTION
    # A node with no method that takes an image stays as it was.
    cuts = real_nodes["Cut and Splice"]
    assert _inputs(cuts)[MAIN_ID]["type"] == ["VideoClip", "null"]
    assert "Image" not in str(cuts.outputs[0].to_dict()["type"])


def test_an_error_names_the_node(
    real_nodes: dict[str, NodeData], tmp_path: Path
) -> None:
    node = real_nodes["Resize"]
    method = "std.Crop"
    # More cropped than the image has.
    choice = _choice(node, method, left=WIDTH + 2, right=0, top=0, bottom=0)
    with pytest.raises(VSScriptError) as error:
        _run(node, method, _picture(2), tmp_path, choice=choice)
    assert error.value.node_id == NODE
    # Nothing stays behind.
    assert not list(chain_dir(_context(tmp_path)).iterdir())


def _node_of(group: Any, schema_id: str) -> NodeData:
    return next(node for node in group.nodes if node.schema_id == schema_id)


def _chain_node(node_id: str, data: NodeData) -> FunctionNode | GeneratorNode:
    kind = GeneratorNode if data.kind == "generator" else FunctionNode
    node = object.__new__(kind)
    node.id, node.schema_id, node.data = NodeId(node_id), data.schema_id, data
    return node


class _Events(EventConsumer):
    def put(self, event: object) -> None:
        pass


@pytest.mark.parametrize("engine", ["native", "python"])
def test_a_real_run_of_load_images_to_a_node_to_save_image(
    real_nodes: dict[str, NodeData],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    engine: str,
) -> None:
    """Per item under Load Images, in the executor: each image of a folder, its own
    result; the run's end takes the frame process."""
    monkeypatch.setattr(
        process.registry, "get_package", lambda _: SimpleNamespace(id="test")
    )
    folder = tmp_path / "in"
    folder.mkdir()
    saved = tmp_path / "out"
    saved.mkdir()
    sources = {}
    for seed, size in enumerate([(32, 48), (48, 32), (20, 20)]):
        pixels = (_picture(seed, *size) * 255).round().astype(np.uint8)
        sources[f"image{seed}"] = pixels
        cv2.imwrite(str(folder / f"image{seed}.png"), pixels)

    chain = Chain()
    loader = _node_of(load_images.batch_processing_group, "chainner:image:load_images")
    saver = _node_of(save_image.io_group, "chainner:image:save")
    colour = real_nodes["Colour"]
    for node_id, data in (("load", loader), ("filter", colour), ("save", saver)):
        chain.add_node(_chain_node(node_id, data))
        for item in data.inputs:
            default = item.to_dict().get("def")
            if default is not None:
                chain.inputs.set(NodeId(node_id), item.id, default)
    chain.inputs.set(NodeId("load"), InputId(0), str(folder))
    chain.inputs.set(NodeId("filter"), InputId(METHOD_ID), "std.Invert")
    chain.inputs.set(NodeId("filter"), InputId(MODE_ID), "simple")
    chain.inputs.set(NodeId("filter"), InputId(FILTER_ID), 0)
    chain.inputs.set(NodeId("save"), InputId(1), str(saved))
    chain.add_edge(
        Edge(
            EdgeSource(NodeId("load"), OutputId(0)),
            EdgeTarget(NodeId("filter"), InputId(MAIN_ID)),
        )
    )
    chain.add_edge(
        Edge(
            EdgeSource(NodeId("filter"), OutputId(0)),
            EdgeTarget(NodeId("save"), InputId(0)),
        )
    )
    chain.add_edge(
        Edge(
            EdgeSource(NodeId("load"), OutputId(3)),
            EdgeTarget(NodeId("save"), InputId(3)),
        )
    )

    async def main() -> None:
        pool = ThreadPoolExecutor(max_workers=8)
        try:
            if engine == "python":
                executor = Executor(
                    ExecutionId("vs-images"),
                    chain,
                    False,
                    ExecutionOptions({}),
                    asyncio.get_running_loop(),
                    _Events(),
                    pool,
                    tmp_path,
                    pool_size=8,
                )
            else:
                executor = NativeExecutor(
                    id=ExecutionId("vs-images"),
                    chain=chain,
                    send_broadcast_data=False,
                    options=ExecutionOptions({}),
                    loop=asyncio.get_running_loop(),
                    queue=_Events(),
                    storage_dir=tmp_path,
                    pool=pool,
                    pool_size=8,
                )
            await executor.run()
        finally:
            pool.shutdown(wait=False)

    asyncio.run(main())
    for name, pixels in sources.items():
        out = cv2.imread(str(saved / f"{name}.png"), cv2.IMREAD_UNCHANGED)
        assert out is not None, name
        assert np.array_equal(out, 255 - pixels), name
    assert runner.frame_process is None
    assert not list((tmp_path / "vapoursynth" / "chains").rglob("frames-*"))


def _statement_on(source: VideoClip, code: str) -> VideoClip:
    var = "n_result_0"
    return VideoClip(var, f"{var} = {code}", (source,), frozenset(), NODE)


def test_the_frame_process_serves_a_whole_run(
    evaluation: str, vs: Any, tmp_path: Path
) -> None:
    if evaluation == "vspipe":
        pytest.skip("a vspipe serves one image")
    context = _context(tmp_path)

    def render():
        return filter_image(context, [_picture(1)], lambda clips: clips[0])

    render()
    first = runner.frame_process
    assert first is not None
    render()
    assert runner.frame_process is first


def test_scripts_do_not_see_each_other(
    evaluation: str, vs: Any, tmp_path: Path
) -> None:
    context = _context(tmp_path)

    def run(code: str) -> np.ndarray:
        return filter_image(
            context, [_picture(1)], lambda clips: _statement_on(clips[0], code)
        )

    run(
        "(globals().__setitem__('earlier', 1), {})[1]".format(
            "core.std.BlankClip(length=1)"
        )
    )
    # A name of the script before would make this statement fail.
    out = run("core.std.BlankClip(length=1) if 'earlier' not in globals() else None")
    assert out.shape[:2] == (480, 640)


def test_what_a_script_prints_is_no_answer(vs: Any, tmp_path: Path) -> None:
    image = _picture(1)
    out = filter_image(
        _context(tmp_path),
        [image],
        lambda clips: _statement_on(
            clips[0],
            f"(print('noise', flush=True), {clips[0].var})[1]",
        ),
    )
    assert out.tobytes() == image.tobytes()


def test_a_process_that_ends_costs_its_image_only(
    evaluation: str, vs: Any, tmp_path: Path
) -> None:
    """As a plugin's crash ends it (without a real crash: this runs on Windows too)."""
    context = _context(tmp_path)
    with pytest.raises(VSScriptError, match="ended with code 3"):
        filter_image(
            context,
            [_picture(1)],
            lambda clips: _statement_on(clips[0], "__import__('os')._exit(3)"),
        )
    image = _picture(2)
    assert filter_image(context, [image], lambda c: c[0]).tobytes() == image.tobytes()


def test_stop_ends_the_script_at_once(evaluation: str, vs: Any, tmp_path: Path) -> None:
    class Stoppable(Context):
        def check_aborted(self) -> None:
            if self.aborted:
                raise Aborted

    context = Stoppable(tmp_path)
    timer = threading.Timer(0.5, lambda: setattr(context, "aborted", True))
    timer.start()
    began = time.perf_counter()
    with pytest.raises(Aborted):
        filter_image(
            cast(NodeContext, context),
            [_picture(1)],
            lambda clips: _statement_on(clips[0], "__import__('time').sleep(60)"),
        )
    assert time.perf_counter() - began < 10
    timer.join()
    context.aborted = False
    image = _picture(2)
    out = filter_image(cast(NodeContext, context), [image], lambda c: c[0])
    assert out.tobytes() == image.tobytes()


def test_alpha_that_comes_back_as_rgb_stays_one_channel(
    real_nodes: dict[str, NodeData], tmp_path: Path
) -> None:
    node = real_nodes["Convert Format"]
    method = "resize.Bicubic"
    choice = _choice(node, method, format="vs.RGBS", width=WIDTH * 2, height=HEIGHT * 2)
    rgba = np.dstack((_picture(2), _picture(3)[:, :, 0]))
    out = _run(node, method, rgba, tmp_path, choice=choice)
    assert out.shape == (HEIGHT * 2, WIDTH * 2, 4)


def test_a_method_that_fails_on_the_alpha_says_so(
    real_nodes: dict[str, NodeData], tmp_path: Path
) -> None:
    node = real_nodes["Resize"]
    method = "std.AddBorders"
    # Three values for the colour's planes, which a gray alpha does not have.
    choice = _choice(node, method, left=4, color="[1.0, 0.0, 0.0]")
    rgba = np.dstack((_picture(2), _picture(3)[:, :, 0]))
    with pytest.raises(VSScriptError, match="alpha"):
        _run(node, method, rgba, tmp_path, choice=choice)


def test_use_as_filter_with_an_image_at_clip_gives_the_filter(small: Catalog) -> None:
    family = small.methods[0].family
    node = _nodes(small, [family])[family]
    image = np.zeros((8, 8, 3), np.float32)
    out = _run(node, "vsrgtools.box_blur", image, None, raw={FILTER_ID: 1})
    assert isinstance(out, VSFilter)
    assert out.expr == "vsrgtools.box_blur"


def test_an_image_left_on_another_methods_input_does_not_count(
    real_nodes: dict[str, NodeData], tmp_path: Path
) -> None:
    node = real_nodes["Planes"]
    other = input_ids(node.name, [("std.Merge", "clipb")])["std.Merge", "clipb"]
    transparent = np.dstack((_picture(2), _picture(3)[:, :, 0]))
    # An RGBA image there would be refused if it counted.
    out = _run(node, "vstools.get_r", _picture(4), tmp_path, raw={other: transparent})
    assert out.tobytes() == _picture(4)[:, :, 2].tobytes()

    clip = VideoClip("n_a_0", "n_a_0 = 1", (), frozenset(), "a")
    statement = _run(node, "vstools.get_r", clip, None, raw={other: _picture(2)})
    assert isinstance(statement, VideoClip)

    resize = real_nodes["Resize"]
    left = input_ids(resize.name, [("vsscale.descale_detail_mask", "rescaled")])
    rgba = np.dstack((_picture(2), _picture(3)[:, :, 0]))
    flipped = _run(
        resize,
        "std.FlipHorizontal",
        rgba,
        tmp_path,
        raw={next(iter(left.values())): _picture(4)},
    )
    assert flipped.shape == rgba.shape


def test_the_help_says_a_temporal_radius_counts_as_0() -> None:
    assert "temporal radius" in _MAIN_HELP
    assert "counts as 0" in _MAIN_HELP


def test_a_yuv_result_still_tagged_rgb_comes_back(vs: Any, tmp_path: Path) -> None:
    """A function that makes YUV of an RGB image and keeps its _Matrix 0 (Range Mask): VapourSynth
    refuses that tag on YUV, which says nothing about the picture."""
    code = "{}.resize.Point(format=vs.YUV444PS, matrix=1).std.SetFrameProps(_Matrix=0)"
    image = _picture(2)
    out = filter_image(
        _context(tmp_path),
        [image],
        lambda clips: _statement_on(clips[0], code.format(clips[0].var)),
    )
    assert np.allclose(out, image, atol=1e-4)


def test_range_mask_on_an_rgb_image(
    real_nodes: dict[str, NodeData], tmp_path: Path
) -> None:
    out = _run(real_nodes["Masks"], "vsmasktools.range_mask", _picture(2), tmp_path)
    assert out.shape[:2] == (HEIGHT, WIDTH)


def test_a_result_of_varying_format_is_refused_by_name(vs: Any, tmp_path: Path) -> None:
    code = "core.std.Splice([{0}, {0}.resize.Point(format=vs.GRAYS, matrix=1)], mismatch=True)"
    with pytest.raises(VSScriptError, match="size or format varies"):
        filter_image(
            _context(tmp_path),
            [_picture(2)],
            lambda clips: _statement_on(clips[0], code.format(clips[0].var)),
        )


def test_a_process_that_ends_does_not_quote_an_earlier_image(
    evaluation: str, vs: Any, tmp_path: Path
) -> None:
    context = _context(tmp_path)
    earlier = (
        "(print('earlier image', file=__import__('sys').stderr, flush=True), {})[1]"
    )
    filter_image(
        context,
        [_picture(1)],
        lambda clips: _statement_on(clips[0], earlier.format(clips[0].var)),
    )
    with pytest.raises(VSScriptError) as error:
        filter_image(
            context,
            [_picture(1)],
            lambda clips: _statement_on(clips[0], "__import__('os')._exit(3)"),
        )
    assert "ended with code 3" in str(error.value)
    assert "earlier image" not in str(error.value)


def test_other_versions_are_named_in_a_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _frame_table(monkeypatch, tmp_path)
    with caplog.at_level(logging.WARNING):
        load_catalog(_small_catalog(tmp_path, vsjetpack="99"), [])
    messages = [
        r.getMessage() for r in caplog.records if "vs_frame_probe" in r.getMessage()
    ]
    assert len(messages) == 1
    assert "99" in messages[0] and TABLE["vsjetpack"] in messages[0]
    assert TABLE["vapoursynth"] in messages[0]


def test_a_method_that_crashes_on_tiny_images_refuses_them_by_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    table = json.loads(catalog.FRAME_METHODS.read_text(encoding="utf-8"))
    table["frame"] = {"vsrgtools.box_blur": {"min_side": 4}}
    path = tmp_path / "frame_methods.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    monkeypatch.setattr(catalog, "FRAME_METHODS", path)
    cat = load_catalog(_small_catalog(tmp_path), [])
    family = cat.methods[0].family
    node = _nodes(cat, [family])[family]
    with pytest.raises(ValueError, match="at least 4x4"):
        _run(node, "vsrgtools.box_blur", np.zeros((3, 5, 3), np.float32), tmp_path)


def _route_of(probe: Any, statement: str, vs: Any, family: str = "") -> tuple[str, str]:
    results = probe.measure_case({"stmt": statement, "imports": []}, vs)
    return probe.verdict(results, family)


def test_the_measurement_finds_what_reads_only_red(vs: Any) -> None:
    probe = pytest.importorskip("vs_frame_probe")
    red = "n_probe_0 = core.std.ShufflePlanes(n_src_0, 0, vs.GRAY)"
    # The same mask whatever green and blue are: VapourSynth's first plane, which vs-jetpack means
    # as luma, is red here. It runs on YUV, where it is luma.
    assert _route_of(probe, red, vs) == ("frame", "yuv")
    # In the Planes node the plane is what the user asked for.
    assert _route_of(probe, red, vs, "Planes") == ("frame", "rgb")
    # What reads all three, and what reads none, stay on RGB.
    every = "n_probe_0 = core.std.BoxBlur(n_src_0)"
    assert _route_of(probe, every, vs) == ("frame", "rgb")
    none = "n_probe_0 = core.std.BlankClip(n_src_0, keep=True)"
    assert _route_of(probe, none, vs) == ("frame", "rgb")


def test_the_measurement_makes_an_image_of_the_result(vs: Any) -> None:
    probe = pytest.importorskip("vs_frame_probe")
    statement = "n_probe_0 = core.std.BoxBlur(n_src_0)"
    results = probe.measure_case({"stmt": statement, "imports": []}, vs)
    assert results["RGBS"]["image"] == [probe.HEIGHT, probe.WIDTH, 3]
    assert results["GRAYS"]["image"] == [probe.HEIGHT, probe.WIDTH]
    assert results["RGBS"]["finite"] is True
    # A result no image can be is an error of that format.
    many = "n_probe_0 = core.std.SetFrameProps(n_src_0, _Matrix=0).std.Crop(left=1)"
    assert (
        probe.measure_case({"stmt": many, "imports": []}, vs)["RGBS"]["kind"] == "frame"
    )


@pytest.mark.parametrize(
    "method",
    [
        "vsmasktools.detail_mask",
        "vsmasktools.limited_linemask",
        "vsmasktools.luma_credit_mask",
        "vsmasktools.ringing_mask",
    ],
)
def test_a_luma_mask_looks_at_the_whole_picture(
    real_nodes: dict[str, NodeData], tmp_path: Path, method: str
) -> None:
    node = real_nodes["Masks"]

    def green_blocks(size: int) -> np.ndarray:
        """Flat red and blue, all the detail in green (BGR)."""
        y, x = np.mgrid[0:96, 0:128]
        image = np.full((96, 128, 3), 0.9, np.float32)
        image[:, :, 1] = ((x // size + y // size) % 2) * 0.85 + 0.1
        return image

    first = _run(node, method, green_blocks(9), tmp_path)
    second = _run(node, method, green_blocks(17), tmp_path)
    assert not np.array_equal(first, second)


def test_a_diff_is_no_image(real_nodes: dict[str, NodeData]) -> None:
    """A result centred on 0 would lose everything below 0 at the image's output."""
    for family, method in (("Planes", "vstools.get_u"), ("Planes", "std.MakeDiff")):
        options = {
            o["value"]: o["type"]
            for o in _inputs(real_nodes[family])[METHOD_ID]["options"]
        }
        assert options[method] == CLIP_OPTION, method


def test_tiny_images_are_refused_where_a_plugin_crashes(
    real_nodes: dict[str, NodeData], tmp_path: Path
) -> None:
    node = real_nodes["Sharpen and Blur"]
    with pytest.raises(ValueError, match="at least 4x4"):
        _run(node, "vsrgtools.median_blur", _picture(1, 3, 5), tmp_path)


def test_an_edge_mask_of_an_rgb_image_is_a_gray_mask_of_its_luma(
    real_nodes: dict[str, NodeData], tmp_path: Path
) -> None:
    node = real_nodes["Masks"]
    method = "vsmasktools.*.edgemask"
    assert TABLE["frame"][method]["route"] == "luma"
    keys = input_ids(node.name, [(method, CLASS)])
    choice = {keys[method, CLASS]: "Prewitt"}
    image = _picture(1, 96, 128)
    out = _run(node, method, image, tmp_path, choice=choice)
    # Not an edge mask per YUV plane read as colour.
    assert out.ndim == 2
    assert np.isfinite(out).all()

    weights = np.array([0.0722, 0.7152, 0.2126], np.float32)
    luma = (image * weights).sum(axis=2)
    alone = _run(node, method, luma, tmp_path, choice=choice)
    assert np.allclose(out, alone, atol=1e-3)


def test_a_source_window_moves_the_alpha_too(
    real_nodes: dict[str, NodeData], tmp_path: Path
) -> None:
    node = real_nodes["Convert Format"]
    method = "resize.Point"
    key = input_ids(node.name, [(method, "src_left")])[method, "src_left"]
    colour = _picture(1)
    image = np.dstack((colour, colour[:, :, 1]))
    raw = {MODE_ID: "advanced", key: 8.0}
    out = _run(node, method, image, tmp_path, raw=raw)
    assert not np.array_equal(out[:, :, 1], colour[:, :, 1])
    assert np.array_equal(out[:, :, 3], out[:, :, 1])


def test_a_result_that_is_not_numbers_is_an_error(vs: Any, tmp_path: Path) -> None:
    path = tmp_path / "in.npy"
    np.save(path, _picture(1), allow_pickle=False)
    source = image_clip(path, NODE, 0, "rgb")
    stmt = f"n_divided = {source.var}.std.Expr('x 0 /')"
    divided = VideoClip("n_divided", stmt, (source,), frozenset(), NODE)
    code = compile_script([(divided, "x")], cache_mb=256)
    out = tmp_path / "out.npy"
    with pytest.raises(VSScriptError, match="not numbers"):
        runner.save_frames(tmp_path / "x.vpy", code, [out], _context(tmp_path))
    assert not out.exists()


def test_the_measurement_refuses_what_differs_on_a_clip_of_one_frame(vs: Any) -> None:
    """SangNom's antialias on one frame is what it does on a clip's last frame, not on the
    others: on an image it would look unlike the same picture in a video."""
    probe = pytest.importorskip("vs_frame_probe")
    case = {
        "stmt": "n_probe_0 = vsaa.SangNom().antialias(n_src_0)",
        "imports": ["import vsaa"],
    }
    assert probe.verdict(probe.measure_case(case, vs))[0] == "needs_frames"
    # What changes with the frame number looks at no other frame.
    numbered = {"stmt": "n_probe_0 = core.text.FrameNum(n_src_0)", "imports": []}
    assert probe.verdict(probe.measure_case(numbered, vs))[0] == "frame"


def test_the_measurement_takes_a_case_of_another_size(vs: Any) -> None:
    probe = pytest.importorskip("vs_frame_probe")
    case = {"stmt": "n_probe_0 = core.zsmooth.CCD(n_src_0)", "imports": []}
    assert probe.verdict(probe.measure_case(case, vs))[0] == "fails"
    sized = {**case, "size": probe.BIG_PICTURES["zsmooth.CCD"]}
    results = probe.measure_case(sized, vs)
    assert probe.verdict(results)[0] == "frame"
    assert results["RGBS"]["image"] == [480, 640, 3]


def test_chroma_denoise_takes_an_image(
    real_nodes: dict[str, NodeData], tmp_path: Path
) -> None:
    image = _picture(1, 480, 640)
    out = _run(real_nodes["Denoise"], "vsdenoise.ccd", image, tmp_path)
    assert out.shape == image.shape
    assert not np.array_equal(out, image)


def test_a_luma_clip_is_gray_with_no_matrix(vs: Any, tmp_path: Path) -> None:
    path = tmp_path / "in.npy"
    np.save(path, _picture(1), allow_pickle=False)
    clip = image_clip(path, NODE, 0, "luma")
    script = tmp_path / "tags.vpy"
    script.write_text(compile_script([(clip, "x")], cache_mb=256), encoding="utf-8")
    info, props, _ = clip_props(script, 0)
    assert info.format_name == "GrayS"
    assert "_Matrix" not in props


def test_the_measurement_finds_what_the_table_says(vs: Any) -> None:
    """A sample measured again: what the table holds comes from this."""
    probe = pytest.importorskip("vs_frame_probe")
    samples = {
        "n_probe_0 = core.std.BoxBlur(n_src_0)": "frame",
        "n_probe_0 = core.std.Invert(n_src_0)": "frame",
        "n_probe_0 = core.std.Crop(n_src_0, top=20, bottom=20)": "frame",
        "n_probe_0 = core.std.Reverse(n_src_0)": "temporal",
        "n_probe_0 = core.std.SelectEvery(n_src_0, 2, 0)": "length",
        "n_probe_0 = core.zsmooth.TemporalSoften(n_src_0)": "temporal",
        "n_probe_0 = core.zsmooth.FluxSmoothT(n_src_0)": "temporal",
    }
    for statement, kind in samples.items():
        results = probe.measure_case({"stmt": statement, "imports": []}, vs)
        assert probe.verdict(results)[0] == kind, statement
