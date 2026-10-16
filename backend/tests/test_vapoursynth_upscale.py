"""AI Upscale in a VapourSynth script: chainner_vs.upscale runs chaiNNer's own
upscalers on every frame, with the same result as the Upscale Image nodes give for the
same images; a model is loaded once per process and called by one thread at a time.
CPU only (CUDA_VISIBLE_DEVICES=-1), on 96x64 frames."""

from __future__ import annotations

import dataclasses
import importlib
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from spandrel import ImageModelDescriptor

from api import NodeContext, SettingsParser
from gpu import cuda_pci_addresses
from nodes.impl.onnx.model import OnnxGeneric, OnnxInfo
from nodes.impl.upscale.auto_split_tiles import (
    CUSTOM,
    ESTIMATE,
    TILE_SIZE_256,
    TileSize,
)
from nodes.impl.upscale.basic_upscale import PaddingType
from nodes.impl.vapoursynth.runner import vs_env, vs_python, vspipe
from nodes.impl.vapoursynth.script import VideoClip, compile_script
from packages.chaiNNer_onnx.onnx.io.load_model import (
    load_model_node as load_onnx_model_node,
)
from packages.chaiNNer_onnx.onnx.processing.upscale_image import (
    upscale_image_node as onnx_upscale_image_node,
)
from packages.chaiNNer_pytorch.pytorch.io.load_model import (
    load_model_node as load_pytorch_model_node,
)
from packages.chaiNNer_pytorch.pytorch.processing.upscale_image import (
    upscale_image_node as pytorch_upscale_image_node,
)
from packages.chaiNNer_vapoursynth.vapoursynth.ai_upscale import (
    AiUpscaleBackend,
    AiUpscalePrecision,
    ai_upscale_node,
)

# VapourSynth lives in native\local\vapoursynth\venv, not in the runtime.
vs = pytest.importorskip("vapoursynth")
# Tiling runs in the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

# The bridge imports VapourSynth.
chainner_vs = importlib.import_module("nodes.impl.vapoursynth.chainner_vs")
bridge = importlib.import_module("nodes.impl.vapoursynth.chainner_vs.upscale")

# A 2x ESRGAN (7 MB) and a 4x Compact ONNX (2.5 MB) from the owner's model folder.
ESRGAN_2X = Path(r"C:\Executables\models\ESRGAN\2x_Waifaux-NL3-SuperLite_latest_G.pth")
COMPACT_4X_ONNX = Path(
    r"C:\Executables\models\ESRGAN\4xLSDIRCompactN-fp32-opset17.onnx"
)

pytestmark = pytest.mark.skipif(
    not ESRGAN_2X.is_file(), reason=f"{ESRGAN_2X} is not on this machine"
)

WIDTH, HEIGHT = 96, 64


def images(count: int, *, gray: bool = False) -> list[np.ndarray]:
    """Random float32 BGR images (HxW for gray), as chaiNNer holds them."""
    rng = np.random.default_rng(7)
    shape = (HEIGHT, WIDTH) if gray else (HEIGHT, WIDTH, 3)
    return [rng.random(shape, dtype=np.float32) for _ in range(count)]


def clip_of(imgs: list[np.ndarray]) -> Any:
    """An RGBS (or GRAYS) clip whose frames are `imgs`."""
    gray = imgs[0].ndim == 2
    blank = vs.core.std.BlankClip(
        width=WIDTH,
        height=HEIGHT,
        length=len(imgs),
        format=vs.GRAYS if gray else vs.RGBS,
    )

    def fill(n: int, f: Any) -> Any:
        frame = f.copy()
        if gray:
            np.copyto(np.asarray(frame[0]), imgs[n])
        else:
            for plane, channel in enumerate((2, 1, 0)):  # R, G, B from BGR
                np.copyto(np.asarray(frame[plane]), imgs[n][:, :, channel])
        return frame

    return blank.std.ModifyFrame(blank, fill)


def image_of(frame: Any) -> np.ndarray:
    """A frame as a float32 BGR image (HxW for gray)."""
    if frame.format.num_planes == 1:
        return np.array(frame[0])
    return np.stack([np.asarray(frame[plane]) for plane in (2, 1, 0)], axis=2)


class Context:
    """The NodeContext members the Load Model and Upscale Image nodes use."""

    aborted = False
    paused = False

    def __init__(self, settings: dict[str, object]) -> None:
        self.settings = SettingsParser(settings)

    def add_cleanup(self, fn: Callable[[], None], after: str = "chain") -> None:
        pass

    def check_aborted(self) -> None:
        pass

    def suspend(self) -> None:
        pass


def gpu_visible() -> bool:
    return bool(cuda_pci_addresses())


def pytorch_reference(imgs: list[np.ndarray], tile_size: TileSize, custom: int):
    """PyTorch's Upscale Image node on `imgs`, on the bridge's device (chaiNNer's CPU
    mode when CUDA sees no GPU)."""
    context = cast(
        NodeContext, Context({"use_cpu": not gpu_visible(), "use_fp16": False})
    )
    model = load_pytorch_model_node(context, ESRGAN_2X)[0]
    assert isinstance(model, ImageModelDescriptor)
    return [
        pytorch_upscale_image_node(
            context,
            img,
            model,
            False,
            model.scale,
            tile_size,
            custom,
            PaddingType.NONE,
            False,
        )
        for img in imgs
    ]


@pytest.mark.parametrize(("tile_size", "custom"), [(ESTIMATE, 0), (CUSTOM, 48)])
def test_bridge_matches_upscale_node(tile_size: TileSize, custom: int):
    imgs = images(3)
    expected = pytorch_reference(imgs, tile_size, custom)

    tile = custom if tile_size == CUSTOM else tile_size
    out = chainner_vs.upscale(clip_of(imgs), str(ESRGAN_2X), "PyTorch", "fp32", tile)

    assert (out.width, out.height, out.num_frames) == (2 * WIDTH, 2 * HEIGHT, 3)
    assert out.format.id == vs.RGBS
    for n, want in enumerate(expected):
        got = image_of(out.get_frame(n))
        assert got.dtype == want.dtype == np.float32
        assert got.shape == want.shape
        assert np.array_equal(got, want)


@pytest.mark.skipif(not COMPACT_4X_ONNX.is_file(), reason="no ONNX test model")
def test_onnx_bridge_matches_upscale_node():
    imgs = images(3)
    provider = "CUDAExecutionProvider" if gpu_visible() else "CPUExecutionProvider"
    context = cast(NodeContext, Context({"execution_provider": provider}))
    model = load_onnx_model_node(COMPACT_4X_ONNX)[0]
    assert isinstance(model, OnnxGeneric)
    expected = [
        onnx_upscale_image_node(context, img, model, TILE_SIZE_256, 0, False)
        for img in imgs
    ]

    # Auto is the ONNX node's default tile size (ONNX has no estimate).
    out = chainner_vs.upscale(
        clip_of(imgs), str(COMPACT_4X_ONNX), "ONNX", "fp32", ESTIMATE
    )

    assert (out.width, out.height, out.format.id) == (4 * WIDTH, 4 * HEIGHT, vs.RGBS)
    for n, want in enumerate(expected):
        assert np.array_equal(image_of(out.get_frame(n)), want)


def test_gray_clip_roundtrip():
    imgs = images(3, gray=True)
    expected = pytorch_reference(imgs, ESTIMATE, 0)
    source = clip_of(imgs).std.SetFrameProps(Marker=7)

    out = chainner_vs.upscale(source, str(ESRGAN_2X), "PyTorch", "fp32", ESTIMATE)

    assert (out.width, out.height, out.format.id) == (2 * WIDTH, 2 * HEIGHT, vs.GRAYS)
    for n, want in enumerate(expected):
        frame = out.get_frame(n)
        got = image_of(frame)
        assert want.size == got.size
        # The node's own gray path: replicated to BGR for the model, then back to
        # gray with BT.601 weights (as a gray RGBS clip resized to GRAY would be).
        assert np.array_equal(got, want.reshape(got.shape))
        assert frame.props["Marker"] == 7


def test_out_of_range_input_is_clipped_like_the_node():
    """chaiNNer's images are always within [0, 1]; a VapourSynth float clip need not
    be, and the model gets it clipped, as the node would."""
    imgs = [img * 2 - 0.5 for img in images(3)]
    expected = pytorch_reference([np.clip(img, 0, 1) for img in imgs], ESTIMATE, 0)

    out = chainner_vs.upscale(
        clip_of(imgs), str(ESRGAN_2X), "PyTorch", "fp32", ESTIMATE
    )

    for n, want in enumerate(expected):
        assert np.array_equal(image_of(out.get_frame(n)), want)


def fake_load(
    monkeypatch: pytest.MonkeyPatch, input_channels: int, output_channels: int
) -> None:
    """Every model loads as a 2x upscaler with these channel counts that makes black
    images."""

    def run(img: np.ndarray, tile: int) -> np.ndarray:
        h, w = img.shape[:2]
        return np.zeros((2 * h, 2 * w, output_channels), np.float32)

    monkeypatch.setattr(bridge, "loaded", {})
    monkeypatch.setattr(
        bridge,
        "load",
        lambda *key: bridge.Upscaler(2, 2, input_channels, output_channels, run),
    )


def test_colour_props_follow_a_colour_family_change(monkeypatch: pytest.MonkeyPatch):
    """A gray clip from YUV keeps its _Matrix; on an RGB result (a 1 -> 3 channel
    model) it must become RGB's, and an RGB clip's must go on a gray result, or the
    next resize fails."""
    fake_load(monkeypatch, 1, 3)
    gray = clip_of(images(1, gray=True)).std.SetFrameProps(_Matrix=6, Marker=7)
    rgb_out = chainner_vs.upscale(gray, str(ESRGAN_2X), "PyTorch", "fp32", 0)
    frame = rgb_out.get_frame(0)
    assert rgb_out.format.id == vs.RGBS
    assert (frame.props["_Matrix"], frame.props["Marker"]) == (0, 7)
    rgb_out.resize.Bicubic(format=vs.YUV444PS, matrix=1).get_frame(0)

    fake_load(monkeypatch, 3, 1)
    rgb = clip_of(images(1)).std.SetFrameProps(_Matrix=0, Marker=7)
    gray_out = chainner_vs.upscale(rgb, str(ESRGAN_2X), "PyTorch", "fp32", 0)
    frame = gray_out.get_frame(0)
    assert gray_out.format.id == vs.GRAYS
    assert "_Matrix" not in frame.props
    assert frame.props["Marker"] == 7
    gray_out.resize.Bicubic(format=vs.RGBS).get_frame(0)


def test_model_loaded_once(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bridge, "loaded", {})
    keys: list[tuple[str, ...]] = []
    load = bridge.load

    def counting_load(*key: str):
        keys.append(key)
        return load(*key)

    monkeypatch.setattr(bridge, "load", counting_load)
    source = clip_of(images(3))
    model = str(ESRGAN_2X)

    first = chainner_vs.upscale(source, model, "PyTorch", "fp32", ESTIMATE)
    second = chainner_vs.upscale(source, model, "PyTorch", "fp32", 48)
    for n in range(3):
        first.get_frame(n)
        second.get_frame(n)
    assert keys == [(model, "PyTorch", "fp32")]

    chainner_vs.upscale(source, model, "PyTorch", "fp16", ESTIMATE)
    assert keys == [(model, "PyTorch", "fp32"), (model, "PyTorch", "fp16")]


@pytest.mark.skipif(not COMPACT_4X_ONNX.is_file(), reason="no ONNX test model")
def test_onnx_model_loaded_once_whatever_the_precision(monkeypatch: pytest.MonkeyPatch):
    """ONNX models (and TensorRT engines) run in their own precision: two nodes on one
    file share one session, whatever their hidden precision input says."""
    monkeypatch.setattr(bridge, "loaded", {})
    keys: list[tuple[str, ...]] = []
    load = bridge.load

    def counting_load(*key: str):
        keys.append(key)
        return load(*key)

    monkeypatch.setattr(bridge, "load", counting_load)
    source = clip_of(images(1))
    model = str(COMPACT_4X_ONNX)

    chainner_vs.upscale(source, model, "ONNX", "fp32", ESTIMATE)
    chainner_vs.upscale(source, model, "ONNX", "fp16", ESTIMATE)
    assert len(keys) == 1


def test_repeated_requests_run_the_model_once(monkeypatch: pytest.MonkeyPatch):
    """vs-jetpack reads frame properties while a script is evaluated, through
    short-lived clips; VapourSynth's automatic cache mode would run the model again
    for every one of them."""
    monkeypatch.setattr(bridge, "loaded", {})
    calls = 0
    load = bridge.load

    def counting_load(*key: str):
        upscaler = load(*key)

        def run(img: np.ndarray, tile: int) -> np.ndarray:
            nonlocal calls
            calls += 1
            return upscaler.run(img, tile)

        return dataclasses.replace(upscaler, run=run)

    monkeypatch.setattr(bridge, "load", counting_load)
    out = chainner_vs.upscale(
        clip_of(images(1)), str(ESRGAN_2X), "PyTorch", "fp32", ESTIMATE
    )
    consumer = out.resize.Point(format=vs.RGBS)

    for _ in range(3):
        consumer.std.Expr("x").get_frame(0)
    consumer.get_frame(0)
    assert calls == 1


def test_model_calls_never_overlap(monkeypatch: pytest.MonkeyPatch):
    """VapourSynth calls one ModifyFrame's selector one frame at a time, but two AI
    Upscale clips in a script run on two worker threads at once; they share the
    model."""
    monkeypatch.setattr(bridge, "loaded", {})
    monkeypatch.setattr(vs.core, "num_threads", 4)
    load = bridge.load
    active = 0
    peak = 0
    count = threading.Lock()

    def slow_load(*key: str):
        upscaler = load(*key)

        def run(img: np.ndarray, tile: int) -> np.ndarray:
            nonlocal active, peak
            with count:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)  # a window in which another call would overlap
            try:
                return upscaler.run(img, tile)
            finally:
                with count:
                    active -= 1

        return dataclasses.replace(upscaler, run=run)

    monkeypatch.setattr(bridge, "load", slow_load)
    source = clip_of(images(6))
    model = str(ESRGAN_2X)
    both = vs.core.std.StackHorizontal(
        [
            chainner_vs.upscale(source, model, "PyTorch", "fp32", ESTIMATE),
            chainner_vs.upscale(source, model, "PyTorch", "fp32", 48),
        ]
    )

    futures = [both.get_frame_async(n) for n in range(6)]
    for future in futures:
        future.result()
    assert peak == 1


def test_parallel_frame_requests_are_serialized(tmp_path: Path):
    """The node's script in vspipe: 4 threads and 4 concurrent requests over 6 frames
    give the same bytes as 1 thread, with two AI Upscale nodes sharing one model."""
    source = VideoClip(
        var="n_src_0",
        stmt=(
            "n_src_0 = core.akarin.Expr(core.std.BlankClip(width=96, height=64,"
            " length=6, format=vs.RGBS), ['X 0.37 * Y 0.11 * + N 0.9 * + sin 0.5 *"
            " 0.5 +', 'X 0.13 * Y 0.29 * - N 0.7 * + cos 0.5 * 0.5 +', 'X Y * 0.003 *"
            " N 0.2 * + sin 0.5 * 0.5 +'])"
        ),
        deps=(),
        imports=frozenset(),
        node_id="src",
    )
    nodes = [
        ai_upscale_node(
            cast(NodeContext, SimpleNamespace(node_id=node_id)),
            source,
            ESRGAN_2X,
            AiUpscaleBackend.PYTORCH,
            AiUpscalePrecision.FP32,
            tile_size,
            48,
        )
        for node_id, tile_size in (("up_a", ESTIMATE), ("up_b", CUSTOM))
    ]
    stack = VideoClip(
        var="n_stack_0",
        stmt=f"n_stack_0 = core.std.StackHorizontal([{nodes[0].var}, {nodes[1].var}])",
        deps=tuple(nodes),
        imports=frozenset(),
        node_id="stack",
    )
    script = compile_script([(stack, "Upscaled")], cache_mb=256)
    assert "from nodes.impl.vapoursynth import chainner_vs" in script
    assert script.count("core = vs.core\n") == 1  # where the thread count goes

    rendered: dict[int, bytes] = {}
    for threads in (1, 4):
        path = tmp_path / f"threads{threads}.vpy"
        path.write_text(
            script.replace(
                "core = vs.core\n", f"core = vs.core\ncore.num_threads = {threads}\n"
            ),
            encoding="utf-8",
        )
        done = subprocess.run(
            [str(vspipe()), "-r", str(threads), "-e", "5", str(path), "-"],
            env=vs_env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=120,
        )
        assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
        rendered[threads] = done.stdout

    assert len(rendered[1]) == 6 * 3 * (2 * 2 * WIDTH) * (2 * HEIGHT) * 4
    assert rendered[4] == rendered[1]


# Evaluates a script twice in one process, as vsview's Reload Script does: between the
# runs it evicts the local packages (chaiNNer's api, gpu, navi, nodes, packages,
# system) from sys.modules, so the script imports them again.
RELOAD = """\
import runpy
import sys

import vapoursynth as vs

try:
    from vsview.app.workspace.utils import evict_packages, find_local_packages
except ImportError:
    LOCAL = {"api", "gpu", "navi", "nodes", "packages", "system"}

    def find_local_packages():
        return {name.split(".")[0] for name in sys.modules} & LOCAL

    def evict_packages(packages):
        for name in [n for n in sys.modules if n.split(".")[0] in packages]:
            del sys.modules[name]

for _ in range(2):
    runpy.run_path(sys.argv[1])
    vs.get_output(0).clip.get_frame(0)
    local = find_local_packages()
    print("evicted", " ".join(sorted(local)))
    evict_packages(local)
    vs.clear_outputs()
"""


def test_script_reloads_after_local_packages_are_evicted(tmp_path: Path):
    source = VideoClip(
        var="n_src_0",
        stmt=(
            "n_src_0 = core.std.BlankClip(width=96, height=64, length=1,"
            " format=vs.RGBS, color=[0.2, 0.5, 0.7])"
        ),
        deps=(),
        imports=frozenset(),
        node_id="src",
    )
    node = ai_upscale_node(
        cast(NodeContext, SimpleNamespace(node_id="up")),
        source,
        ESRGAN_2X,
        AiUpscaleBackend.PYTORCH,
        AiUpscalePrecision.FP32,
        ESTIMATE,
        256,
    )
    script = tmp_path / "chain.vpy"
    script.write_text(compile_script([(node, "Upscaled")], cache_mb=256), "utf-8")
    runner = tmp_path / "reload.py"
    runner.write_text(RELOAD, "utf-8")

    done = subprocess.run(
        [str(vs_python()), "-I", "-B", str(runner), str(script)],
        env=vs_env(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        timeout=120,
    )

    stderr = done.stderr.decode("utf-8", "replace")
    assert done.returncode == 0, stderr
    evicted = done.stdout.decode().splitlines()
    assert len(evicted) == 2
    for line in evicted:
        assert {"api", "nodes", "packages"} <= set(line.split()[1:])


def test_clear_errors(tmp_path: Path):
    rgb = clip_of(images(1))
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(b"not a model")
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"not an engine")

    with pytest.raises(ValueError, match="RGBS or GRAYS"):
        chainner_vs.upscale(
            rgb.resize.Point(format=vs.RGB24), str(ESRGAN_2X), "PyTorch", "fp32", 0
        )
    mixed = vs.core.std.Splice([rgb, clip_of(images(1, gray=True))], mismatch=True)
    with pytest.raises(ValueError, match="changes between frames"):
        chainner_vs.upscale(mixed, str(ESRGAN_2X), "PyTorch", "fp32", 0)
    sizes = vs.core.std.Splice([rgb, rgb.resize.Point(48, 32)], mismatch=True)
    with pytest.raises(ValueError, match="one size"):
        chainner_vs.upscale(sizes, str(ESRGAN_2X), "PyTorch", "fp32", 0)
    with pytest.raises(ValueError, match="ONNX backend"):
        chainner_vs.upscale(rgb, str(onnx_file), "PyTorch", "fp32", 0)
    with pytest.raises(ValueError, match="Unknown backend"):
        chainner_vs.upscale(rgb, str(ESRGAN_2X), "NCNN", "fp32", 0)
    with pytest.raises(ValueError, match="Unknown precision"):
        chainner_vs.upscale(rgb, str(ESRGAN_2X), "PyTorch", "bf16", 0)
    with pytest.raises(ValueError, match="tile size"):
        chainner_vs.upscale(rgb, str(ESRGAN_2X), "PyTorch", "fp32", CUSTOM)
    with pytest.raises(FileNotFoundError):
        chainner_vs.upscale(rgb, str(tmp_path / "missing.pth"), "PyTorch", "fp32", 0)
    if not gpu_visible():
        with pytest.raises(RuntimeError, match="NVIDIA GPU"):
            chainner_vs.upscale(rgb, str(engine), "TensorRT", "fp32", 0)


def test_models_that_are_not_upscalers_are_clear_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    rgb = clip_of(images(1))
    pth = tmp_path / "model.pth"
    pth.write_bytes(b"")
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(b"")
    pytorch_io = importlib.import_module(
        "packages.chaiNNer_pytorch.pytorch.io.load_model"
    )
    onnx_io = importlib.import_module("packages.chaiNNer_onnx.onnx.io.load_model")
    monkeypatch.setattr(bridge, "loaded", {})

    # spandrel's other descriptor: an inpainting model, which takes a mask.
    monkeypatch.setattr(pytorch_io, "load_model", lambda path, settings: object())
    with pytest.raises(ValueError, match="inpainting"):
        chainner_vs.upscale(rgb, str(pth), "PyTorch", "fp32", 0)

    monkeypatch.setattr(
        onnx_io, "load_model_node", lambda path: (object(), path.parent, path.stem)
    )
    with pytest.raises(ValueError, match="background removal"):
        chainner_vs.upscale(rgb, str(onnx_file), "ONNX", "fp32", 0)

    unknown_scale = OnnxGeneric(b"", OnnxInfo(13, "fp32"))
    monkeypatch.setattr(
        onnx_io,
        "load_model_node",
        lambda path: (unknown_scale, path.parent, path.stem),
    )
    with pytest.raises(ValueError, match="cannot tell the scale"):
        chainner_vs.upscale(rgb, str(onnx_file), "ONNX", "fp32", 0)

    fake_load(monkeypatch, 3, 4)
    with pytest.raises(ValueError, match="4-channel images"):
        chainner_vs.upscale(rgb, str(ESRGAN_2X), "PyTorch", "fp32", 0)
