"""The native TensorRT runner (native/trt_runner): its tile planner, frame checks and CPU
reference of the frame loop against the Python they replace (tiling.py, convenient_upscale,
colour_info, bfloat16, NumPy's float16 and np.pad), on the CPU: the module loads without
TensorRT or a GPU. The GPU tests at the end run only with CHAINNER_C_GPU_TESTS=1 and an
ESRGAN engine in CHAINNER_C_GPU_ENGINE."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections import deque
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from nodes.impl.colour_info import code_to_float, float_to_code
from nodes.impl.tensorrt.bfloat16 import bf16_bits_to_float32, float32_to_bf16_bits
from nodes.impl.tensorrt.runner import PLUGIN, RUNNER_VARIABLE, runner_enabled
from nodes.impl.tensorrt.tiling import (
    ShapeBounds,
    _pad_tile,
    _tile_extent,
    axis_weights,
    tile_starts,
    tiled_upscale,
)
from nodes.impl.upscale.convenient_upscale import convenient_upscale

BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
SCALE = 4
STATIC_512 = ShapeBounds(512, 512, 512, 512)
DYNAMIC_64_512 = ShapeBounds(64, 64, 512, 512)


def _module() -> ModuleType:
    try:
        return import_module("nodes.impl.tensorrt._chainner_trt")
    except ModuleNotFoundError:
        pytest.skip("_chainner_trt.pyd is not built (native/Build.ps1 with TensorRT)")


@pytest.fixture(scope="module")
def trt() -> ModuleType:
    return _module()


def _bounds(b: ShapeBounds) -> tuple[int, int, int, int, int]:
    return (b.min_h, b.min_w, b.max_h, b.max_w, b.alignment)


def fake_model(img: np.ndarray, scale: int = SCALE) -> np.ndarray:
    """A context-dependent 'model' (test_tensorrt_tiling's): 5x5 box blur, then
    nearest-neighbour upscale."""
    padded = np.pad(img, ((2, 2), (2, 2), (0, 0)), mode="reflect")
    h, w = img.shape[:2]
    blurred = np.zeros_like(img)
    for dy in range(5):
        for dx in range(5):
            blurred += padded[dy : dy + h, dx : dx + w]
    blurred /= 25
    return blurred.repeat(scale, axis=0).repeat(scale, axis=1)


class FakeEngine:
    """tiling.TileRunner over `fake_model`, enforcing the engine's profile."""

    def __init__(self, bounds: ShapeBounds) -> None:
        self.bounds = bounds
        self.queue: deque[np.ndarray] = deque()

    def submit(self, tile: np.ndarray) -> None:
        h, w = tile.shape[:2]
        b = self.bounds
        assert b.min_h <= h <= b.max_h and b.min_w <= w <= b.max_w, (h, w)
        self.queue.append(fake_model(tile))

    def collect(self, height: int, width: int) -> np.ndarray:
        return self.queue.popleft()[:height, :width]


def chw_model(tensor: np.ndarray) -> np.ndarray:
    """`fake_model` on the runner's tensors: channel-planar RGB in and out."""
    hwc = np.ascontiguousarray(tensor.transpose(1, 2, 0)[:, :, ::-1])
    return np.ascontiguousarray(fake_model(hwc)[:, :, ::-1].transpose(2, 0, 1))


def nearest(scale: int):
    return lambda tensor: tensor.repeat(scale, axis=1).repeat(scale, axis=2)


def bgr_planes(img: np.ndarray) -> list[np.ndarray]:
    """A BGR image's channel views in R, G, B order, as RunnerSession passes them."""
    return [img[:, :, c] for c in range(img.shape[2])[::-1]]


def _image(h: int, w: int, c: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 1.0, (h, w, c)).astype(np.float32)


PLAN_CASES = [
    (1024, 700, 4, STATIC_512),
    (1024, 1024, 4, DYNAMIC_64_512),
    (37, 100, 2, DYNAMIC_64_512),
    (300, 1200, 4, DYNAMIC_64_512),
    (2000, 3000, 1, ShapeBounds(64, 64, 1088, 1920, 4)),
    (300, 300, 2, ShapeBounds(100, 100, 200, 200)),
    (500, 90, 3, ShapeBounds(64, 64, 90, 90)),
    (130, 130, 2, ShapeBounds(50, 50, 50, 50)),
]


@pytest.mark.parametrize("case", PLAN_CASES, ids=[f"{c[0]}x{c[1]}" for c in PLAN_CASES])
def test_plan_matches_tiling(trt: ModuleType, case: tuple[int, int, int, ShapeBounds]):
    h, w, scale, b = case
    plan = trt.plan_tiles(h, w, scale, _bounds(b))
    tile_h = _tile_extent(b.max_h, b.min_h, b.alignment)
    tile_w = _tile_extent(b.max_w, b.min_w, b.alignment)
    ys, xs = tile_starts(h, tile_h), tile_starts(w, tile_w)
    assert (plan["tile_h"], plan["tile_w"], plan["ys"], plan["xs"]) == (
        tile_h,
        tile_w,
        ys,
        xs,
    )
    if len(ys) == 1 and len(xs) == 1:
        assert plan["wy"] == [] and plan["wx"] == []
        return
    for got, want in zip(
        (*plan["wy"], *plan["wx"]),
        (*axis_weights(ys, tile_h, h, scale), *axis_weights(xs, tile_w, w, scale)),
    ):
        assert got.dtype == np.float32
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32))


def test_plan_refuses_an_engine_too_small_to_tile(trt: ModuleType):
    with pytest.raises(ValueError) as python:
        tile_starts(100, 40)
    with pytest.raises(ValueError) as native:
        trt.plan_tiles(100, 30, 2, (40, 40, 40, 40, 4))
    assert str(native.value) == str(python.value)


FRAME_CASES = [
    ("tiled 1024x700 static", (1024, 700), STATIC_512),
    ("tiled 1024x1024 dynamic", (1024, 1024), DYNAMIC_64_512),
    ("one padded tile", (37, 100), DYNAMIC_64_512),
    ("tiled across, padded down", (300, 1200), DYNAMIC_64_512),
    ("pad longer than the image", (5, 3), DYNAMIC_64_512),
    ("one row: edge padding", (1, 50), DYNAMIC_64_512),
    ("one column", (50, 1), DYNAMIC_64_512),
]


@pytest.mark.parametrize("case", FRAME_CASES, ids=[c[0] for c in FRAME_CASES])
def test_frame_loop_matches_tiled_upscale(
    trt: ModuleType, case: tuple[str, tuple[int, int], ShapeBounds]
):
    """The runner's frame loop with the tiles' outputs made on the CPU gives exactly
    tiled_upscale's result: the same tiles, padding, blend weights and blend order."""
    _, (h, w), b = case
    img = _image(h, w)
    expected = tiled_upscale(img, SCALE, b, FakeEngine(b))
    out = np.empty((h * SCALE, w * SCALE, 3), dtype=np.float32)
    trt.reference_upscale(
        bgr_planes(img),
        bgr_planes(out),
        SCALE,
        _bounds(b),
        3,
        3,
        "fp32",
        "fp32",
        chw_model,
    )
    assert np.array_equal(out, expected)


@pytest.mark.parametrize("size", [(5, 3), (1, 50), (50, 1), (37, 100)])
def test_padding_is_tiling_pad_tile(trt: ModuleType, size: tuple[int, int]):
    """The tile the engine gets is _pad_tile's (np.pad reflect, edge for one pixel)."""
    h, w = size
    img = _image(h, w, seed=h)
    seen: list[np.ndarray] = []

    def engine(tensor: np.ndarray) -> np.ndarray:
        seen.append(tensor.copy())
        return nearest(2)(tensor)

    out = np.empty((h * 2, w * 2, 3), dtype=np.float32)
    trt.reference_upscale(
        bgr_planes(img),
        bgr_planes(out),
        2,
        _bounds(DYNAMIC_64_512),
        3,
        3,
        "fp32",
        "fp32",
        engine,
    )
    (tensor,) = seen
    padded = _pad_tile(img, tensor.shape[1], tensor.shape[2])
    assert np.array_equal(tensor, padded[:, :, ::-1].transpose(2, 0, 1))


def test_gray_through_an_rgb_engine_as_convenient_upscale(trt: ModuleType):
    """A gray image is replicated to the engine's three channels and its output clipped
    and narrowed back with BT.601 weights (OpenCV's scalar order: within an ulp of the
    CPU path, whose SIMD narrowing may fuse)."""
    gray = _image(200, 150, 1)[:, :, 0]
    expected = convenient_upscale(
        gray,
        3,
        3,
        lambda i: tiled_upscale(i, SCALE, STATIC_512, FakeEngine(STATIC_512)),
    )
    out = np.empty((800, 600), dtype=np.float32)
    trt.reference_upscale(
        [gray],
        [out],
        SCALE,
        _bounds(STATIC_512),
        3,
        3,
        "fp32",
        "fp32",
        chw_model,
        clip=True,
    )
    np.testing.assert_allclose(out, expected, rtol=0, atol=2e-7)


def test_rgb_through_a_gray_engine(trt: ModuleType):
    """RGB planes are narrowed to the gray engine's input; its output is replicated."""
    img = _image(40, 30)
    gray = (img[:, :, 0] * np.float32(0.114) + img[:, :, 1] * np.float32(0.587)) + img[
        :, :, 2
    ] * np.float32(0.299)
    out = np.empty((80, 60, 3), dtype=np.float32)
    trt.reference_upscale(
        bgr_planes(img),
        bgr_planes(out),
        2,
        _bounds(DYNAMIC_64_512),
        1,
        1,
        "fp32",
        "fp32",
        nearest(2),
    )
    for c in range(3):
        assert np.array_equal(out[:, :, c], gray.repeat(2, 0).repeat(2, 1))


@pytest.mark.parametrize(
    ("bits", "full_range", "planes"),
    [
        (10, False, 1),
        (8, True, 3),
        (8, False, 3),
        (12, True, 1),
        (16, True, 3),
        (16, False, 1),
        (6, False, 1),
    ],
)
def test_integer_planes_use_the_colour_info_convention(
    trt: ModuleType, bits: int, full_range: bool, planes: int
):
    """Codes become colour_info plane values, are clipped to 0-1 for the engine and
    after it, and go back to codes rounded once: colour_info's own functions."""
    rng = np.random.default_rng(bits)
    dtype = np.uint8 if bits <= 8 else np.uint16
    codes = [rng.integers(0, 1 << bits, (33, 47)).astype(dtype) for _ in range(planes)]
    out = [np.empty((66, 94), dtype=dtype) for _ in range(planes)]
    trt.reference_upscale(
        codes,
        out,
        2,
        _bounds(DYNAMIC_64_512),
        planes,
        planes,
        "fp32",
        "fp32",
        nearest(2),
        bits=bits,
        full_range=full_range,
        clip=True,
    )
    for got, plane in zip(out, codes):
        values = np.clip(code_to_float(plane, bits, full_range, False), 0, 1)
        want = float_to_code(values.repeat(2, 0).repeat(2, 1), bits, full_range, False)
        assert np.array_equal(got, want)


SPECIAL = np.array(
    [
        0.0,
        -0.0,
        0.333,
        -1.7,
        65504,
        65519,
        65520,
        1e5,
        -1e5,
        6.1e-5,
        6.0e-5,
        1e-7,
        5.96e-8,
        2.98e-8,
        2.97e-8,
        1e-9,
        -3.3e-6,
        np.inf,
        -np.inf,
        np.nan,
        1.00048828125,
        1.0009765625,
    ],
    dtype=np.float32,
)


def test_half_planes_and_tensors_round_as_numpy(trt: ModuleType):
    """float16 planes and fp16 tensors: NumPy's float32 -> float16 cast (round to nearest
    even, Inf from 65520, subnormals)."""
    plane = np.resize(SPECIAL, (4, 22)).astype(np.float32)
    out = np.empty((8, 44), dtype=np.float32)
    trt.reference_upscale(
        [plane], [out], 2, _bounds(DYNAMIC_64_512), 1, 1, "fp16", "fp16", nearest(2)
    )
    with np.errstate(over="ignore"):  # NumPy's cast overflows to Inf, as intended
        want = plane.astype(np.float16).astype(np.float32).repeat(2, 0).repeat(2, 1)
        half = np.resize(SPECIAL, (4, 22)).astype(np.float16)
    np.testing.assert_array_equal(out, want)
    out16 = np.empty((8, 44), dtype=np.float16)
    trt.reference_upscale(
        [half], [out16], 2, _bounds(DYNAMIC_64_512), 1, 1, "fp32", "fp32", nearest(2)
    )
    np.testing.assert_array_equal(out16, half.repeat(2, 0).repeat(2, 1))


def test_bf16_tensors_round_as_bfloat16_module(trt: ModuleType):
    plane = np.resize(SPECIAL, (4, 22)).astype(np.float32)
    out = np.empty((8, 44), dtype=np.float32)
    trt.reference_upscale(
        [plane], [out], 2, _bounds(DYNAMIC_64_512), 1, 1, "bf16", "bf16", nearest(2)
    )
    want = bf16_bits_to_float32(float32_to_bf16_bits(plane)).repeat(2, 0).repeat(2, 1)
    np.testing.assert_array_equal(out, want)


def test_packing_touches_only_the_planes(trt: ModuleType):
    """A BGR image goes up interleaved; planes of a wider array or with row padding are
    packed planar, and the result never writes the bytes beside its planes."""
    img = _image(30, 20)
    b = _bounds(DYNAMIC_64_512)
    expected = np.empty((60, 40, 3), dtype=np.float32)
    info = trt.describe(bgr_planes(img), bgr_planes(expected), 2, b, 3, 3)
    assert info["source_interleaved"] and info["result_interleaved"]
    assert not info["blended"] and info["tiles"] == [(0, 0, 30, 20, 64, 64, False)]
    trt.reference_upscale(
        bgr_planes(img), bgr_planes(expected), 2, b, 3, 3, "fp32", "fp32", nearest(2)
    )

    wide = np.full((60, 40, 4), 7.0, dtype=np.float32)
    padded = np.full((60, 53), 9.0, dtype=np.float32)
    info = trt.describe(bgr_planes(img), bgr_planes(wide[:, :, :3]), 2, b, 3, 3)
    assert info["source_interleaved"] and not info["result_interleaved"]
    trt.reference_upscale(
        bgr_planes(img),
        bgr_planes(wide[:, :, :3]),
        2,
        b,
        3,
        3,
        "fp32",
        "fp32",
        nearest(2),
    )
    assert np.array_equal(wide[:, :, :3], expected) and np.all(wide[:, :, 3] == 7.0)

    flipped = [np.ascontiguousarray(p)[::-1] for p in bgr_planes(img)]
    trt.reference_upscale(
        flipped,
        [
            padded[:, :40],
            np.empty((60, 40), np.float32),
            np.empty((60, 40), np.float32),
        ],
        2,
        b,
        3,
        3,
        "fp32",
        "fp32",
        nearest(2),
    )
    assert np.array_equal(padded[:, :40], expected[::-1, :, 2]) and np.all(
        padded[:, 40:] == 9.0
    )


def test_frame_arguments_are_checked(trt: ModuleType):
    b = _bounds(DYNAMIC_64_512)
    f32 = np.zeros((8, 8), np.float32)
    up = np.zeros((16, 16), np.float32)
    bad: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = [
        ("1 \\(gray\\) or 3", ([f32, f32], [up, up], 2, b, 3, 3), {}),
        ("result planes from", ([f32], [up], 2, b, 1, 3), {}),
        (
            "the input's size times 2",
            ([f32], [np.zeros((16, 15), np.float32)], 2, b, 1, 1),
            {},
        ),
        ("input planes' dtype", ([f32], [up.astype(np.float16)], 2, b, 1, 1), {}),
        ("one dtype and one size", ([f32, f32, f32[:4]], [up, up, up], 2, b, 3, 3), {}),
        ("2-D arrays", ([np.zeros((8, 8, 3), np.float32)], [up], 2, b, 1, 1), {}),
        ("at least one plane", ([], [up], 2, b, 1, 1), {}),
        ("1 or 3 input and output channels", ([f32], [up], 2, b, 4, 4), {}),
        (
            "9 to 16 bits",
            ([f32.astype(np.uint16)], [up.astype(np.uint16)], 2, b, 1, 1),
            {"bits": 8},
        ),
        (
            "1 to 8 bits",
            ([f32.astype(np.uint8)], [up.astype(np.uint8)], 2, b, 1, 1),
            {"bits": 10},
        ),
        (
            "uint8, uint16, float16 or float32",
            ([f32.astype(np.int16)], [up.astype(np.int16)], 2, b, 1, 1),
            {},
        ),
        ("Bounds are", ([f32], [up], 2, (64, 64, 512, 512), 1, 1), {}),
    ]
    for message, args, kwargs in bad:
        with pytest.raises(ValueError, match=message):
            trt.describe(*args, **kwargs)
    readonly = np.zeros((16, 16), np.float32)
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        trt.describe([f32], [readonly], 2, b, 1, 1)
    with pytest.raises(TypeError, match="NumPy arrays"):
        trt.describe([[0.0]], [up], 2, b, 1, 1)
    with pytest.raises(ValueError, match="engine stand-in returns"):
        trt.reference_upscale([f32], [up], 2, b, 1, 1, "fp32", "fp32", lambda t: t)
    with pytest.raises(ValueError, match="fp32, fp16 or bf16"):
        trt.reference_upscale([f32], [up], 2, b, 1, 1, "int8", "fp32", nearest(2))


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((b"", 0, 1), "engine is empty"),
        ((b"x", -1, 1), "GPU index"),
        ((b"x", 0, 0), "stream count"),
        ((b"x", 0, 17), "stream count"),
    ],
)
def test_runner_arguments_are_checked_before_cuda(
    trt: ModuleType, args: tuple[bytes, int, int], message: str
):
    with pytest.raises(ValueError, match=message):
        trt.Runner(*args)


def test_cpu_functions_load_no_cuda_or_tensorrt(trt: ModuleType):
    """In a fresh process: the module imports and its CPU functions run without loading
    CUDA's driver or runtime or TensorRT (nvinfer is delay-loaded, cudart is static)."""
    code = f"""
import sys
sys.path.insert(0, {str(BACKEND_SRC)!r})
import numpy as np, psutil
from nodes.impl.tensorrt import _chainner_trt as trt
trt.plan_tiles(1024, 700, 4, (512, 512, 512, 512, 64))
img = np.zeros((40, 30), np.float32)
out = np.zeros((80, 60), np.float32)
trt.reference_upscale([img], [out], 2, (64, 64, 512, 512, 64), 1, 1, "fp32", "fp32",
                      lambda t: t.repeat(2, 1).repeat(2, 2))
try:
    trt.Runner(b"", 0, 1)
except ValueError:
    pass
separator, prefixes = (
    (chr(92), ("nvcuda", "nvinfer", "cudart", "nvrtc"))
    if sys.platform == "win32"
    else ("/", ("libcuda", "libnvinfer", "libcudart", "libnvrtc"))
)
names = {{m.path.lower().rsplit(separator, 1)[-1] for m in psutil.Process().memory_maps()}}
print(sorted(n for n in names if n.startswith(prefixes)))
"""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        env=env,
        stdin=subprocess.DEVNULL,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


class _ReferenceRunner:
    """_chainner_trt.Runner with the CPU reference and `fake_model` in the GPU's place."""

    scale = SCALE
    input_channels = 3
    output_channels = 3

    def __init__(self, trt: ModuleType) -> None:
        self.trt = trt

    def upscale(self, inputs: list[np.ndarray], outputs: list[np.ndarray]) -> None:
        self.trt.reference_upscale(
            inputs, outputs, SCALE, _bounds(STATIC_512), 3, 3, "fp32", "fp32", chw_model
        )

    def capture_failure(self) -> str:
        return ""


def test_node_session_matches_python_session(
    trt: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    """The Upscale Image node's runner session (RGB, alpha by the transparency hack, gray
    through convenient_upscale) gives what the Python session gives."""
    from nodes.impl.tensorrt import runner as runner_module
    from nodes.impl.tensorrt.model import TensorRTEngine, TensorRTEngineInfo

    native = SimpleNamespace(Runner=lambda *_: _ReferenceRunner(trt))
    monkeypatch.setattr(runner_module, "native", lambda: native)
    info = TensorRTEngineInfo(
        "fp32", 3, 3, SCALE, "sm_120", "11.2", False, None, None, None
    )
    session = runner_module.RunnerSession(TensorRTEngine(b"engine", info), 0)
    rgba = _image(300, 700, 4)
    rgba[:, :350, 3] = 0.5
    for img in (_image(600, 700), rgba, _image(200, 150, 1)[:, :, 0]):
        expected = convenient_upscale(
            img,
            3,
            3,
            lambda i: tiled_upscale(i, SCALE, STATIC_512, FakeEngine(STATIC_512)),
        )
        assert np.array_equal(convenient_upscale(img, 3, 3, session.upscale), expected)


@pytest.mark.skipif(
    find_spec("vapoursynth") is None or not PLUGIN.is_file(),
    reason=f"needs VapourSynth (run under its venv, as the VapourSynth tests) and {PLUGIN.name}",
)
def test_vapoursynth_plugin_loads_and_checks_its_arguments(tmp_path: Path):
    """The plugin registers chainner_trt.Upscale (API 4) and refuses bad clips, stream
    counts and engines before any CUDA call."""
    compressed = tmp_path / "compressed.engine"
    compressed.write_bytes(b"\x28\xb5\x2f\xfd" + b"\0" * 16)
    code = f"""
import json
import vapoursynth as vs
core = vs.core
core.std.LoadPlugin({str(PLUGIN)!r})
rgb = core.std.BlankClip(format=vs.RGBS, width=64, height=48, length=1)
mixed = core.std.Splice([rgb, core.std.BlankClip(format=vs.GRAY8, width=32, height=48, length=1)],
                        mismatch=True)
def error(make):
    try:
        make()
    except vs.Error as e:
        return str(e)
    return "no error"
up = core.chainner_trt.Upscale
print(json.dumps({{
    "yuv": error(lambda: up(core.std.BlankClip(format=vs.YUV420P8), engine=b"x")),
    "mixed": error(lambda: up(mixed, engine=b"x")),
    "streams": error(lambda: up(rgb, engine=b"x", streams=0)),
    "empty": error(lambda: up(rgb, engine=b"")),
    "compressed": error(lambda: up(rgb, engine={str(compressed)!r})),
    "missing": error(lambda: up(rgb, engine={str(tmp_path / "missing.engine")!r})),
}}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        stdin=subprocess.DEVNULL,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    errors = json.loads(result.stdout.strip().splitlines()[-1])
    expected = {
        "yuv": "must be GRAY or RGB",
        "mixed": "one format and one frame size",
        "streams": "streams must be 1 to 16",
        "empty": "engine is empty",
        "compressed": "compressed",
        "missing": "Cannot open the engine file",
    }
    for case, text in expected.items():
        assert errors[case].startswith("Upscale: ") and text in errors[case], (
            case,
            errors[case],
        )


@pytest.mark.parametrize(
    ("value", "node", "clip"),
    [
        ("1", True, True),
        ("0", False, False),
        ("true", False, True),
        (None, False, True),
    ],
)
def test_runner_is_ai_upscales_default_and_the_nodes_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: str | None,
    node: bool,
    clip: bool,
):
    """AI Upscale's TensorRT backend uses the runner's plugin unless the switch is "0";
    the Upscale Image node only when it is "1"."""
    from nodes.impl.tensorrt import runner as runner_module

    plugin = tmp_path / runner_module.PLUGIN_FILE
    plugin.write_bytes(b"")
    monkeypatch.setattr(runner_module, "PLUGIN", plugin)
    monkeypatch.setattr(runner_module, "MODULE", "nodes.impl.tensorrt.tiling")
    if value is None:
        monkeypatch.delenv(RUNNER_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(RUNNER_VARIABLE, value)
    assert runner_enabled() is node
    assert runner_module.runner_plugin() == (plugin if clip else None)


def test_a_build_without_the_runner_uses_the_python_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Missing binaries are no error: the Python paths run, and each absence is logged
    once."""
    from nodes.impl.tensorrt import runner as runner_module

    warnings: list[str] = []
    monkeypatch.setattr(
        runner_module,
        "logger",
        SimpleNamespace(warning=lambda message, *args: warnings.append(message % args)),
    )
    monkeypatch.setattr(runner_module, "_reported", set())
    monkeypatch.setattr(runner_module, "PLUGIN", tmp_path / runner_module.PLUGIN_FILE)
    monkeypatch.setattr(runner_module, "MODULE", "nodes.impl.tensorrt._missing_runner")
    monkeypatch.setenv(RUNNER_VARIABLE, "1")
    for _ in range(2):
        assert runner_module.runner_plugin() is None
        assert runner_enabled() is False
    assert warnings == [
        f"The TensorRT runner's {name} is not in this build: TensorRT runs on the"
        " Python path."
        for name in (
            f"VapourSynth plugin {runner_module.PLUGIN_FILE}",
            f"module {runner_module.MODULE_FILE}",
        )
    ]


# GPU tests: an approved GPU session only (ESRGAN engines only).
GPU_ENGINE = os.environ.get("CHAINNER_C_GPU_ENGINE", "")
gpu = pytest.mark.skipif(
    os.environ.get("CHAINNER_C_GPU_TESTS") != "1" or not Path(GPU_ENGINE).is_file(),
    reason="GPU tests run with CHAINNER_C_GPU_TESTS=1 and an ESRGAN engine in CHAINNER_C_GPU_ENGINE",
)


def _python_session() -> Any:
    from nodes.impl.compressed_file import read_file
    from nodes.impl.tensorrt.engine_info import deserialize_engine, read_engine_info
    from nodes.impl.tensorrt.inference import TensorRTSession
    from nodes.impl.tensorrt.model import TensorRTEngine

    data = read_file(GPU_ENGINE)
    loaded = deserialize_engine(data, 0)
    engine = TensorRTEngine(data, read_engine_info(loaded), loaded)
    return engine, TensorRTSession(engine, 0)


def _session_engine(session: Any, runner: Any):
    """The Python session as reference_upscale's engine: it takes float32 tiles and casts
    them to the engine's input type itself, which gives back the runner's tensor exactly."""

    def run(tensor: np.ndarray) -> np.ndarray:
        h, w = tensor.shape[1:]
        values = (
            bf16_bits_to_float32(tensor)
            if runner.input_precision == "bf16"
            else tensor.astype(np.float32)
        )
        session.submit(np.ascontiguousarray(values.transpose(1, 2, 0)[:, :, ::-1]))
        out = session.collect(h * runner.scale, w * runner.scale)
        chw = np.ascontiguousarray(out[:, :, ::-1].transpose(2, 0, 1))
        if runner.output_precision == "bf16":
            return float32_to_bf16_bits(chw)
        return chw.astype(
            np.float16 if runner.output_precision == "fp16" else np.float32
        )

    return run


@gpu
@pytest.mark.parametrize("size", [(200, 150), (1100, 900)], ids=["one tile", "tiled"])
def test_gpu_runner_matches_python_session(size: tuple[int, int]):
    from nodes.impl.tensorrt.runner import native

    engine, session = _python_session()
    runner = native().Runner(engine.bytes, 0, 2)
    try:
        img = _image(*size)
        expected = tiled_upscale(img, engine.scale, session.bounds, session)
        out = np.empty_like(expected)
        for _ in range(3):  # enqueue, enqueue and capture, graph replay
            runner.upscale(bgr_planes(img), bgr_planes(out))
            assert np.array_equal(out, expected)
        assert runner.capture_failure() == ""
    finally:
        runner.close()
        session.close()
        engine.close()


@gpu
def test_gpu_kernels_match_cpu_reference():
    """10-bit limited gray planes through an RGB engine: the GPU's conversions, padding and
    blending give what the CPU reference gives with the Python session as its engine."""
    from nodes.impl.tensorrt.runner import native

    engine, session = _python_session()
    runner = native().Runner(engine.bytes, 0, 1)
    try:
        codes = (
            np.random.default_rng(5).integers(0, 1024, (700, 1300)).astype(np.uint16)
        )
        s = engine.scale
        got = np.empty((700 * s, 1300 * s), np.uint16)
        want = np.empty_like(got)
        runner.upscale([codes], [got], bits=10, full_range=False, clip=True)
        native().reference_upscale(
            [codes],
            [want],
            s,
            runner.bounds,
            runner.input_channels,
            runner.output_channels,
            runner.input_precision,
            runner.output_precision,
            _session_engine(session, runner),
            bits=10,
            full_range=False,
            clip=True,
        )
        assert np.array_equal(got, want)
    finally:
        runner.close()
        session.close()
        engine.close()


@gpu
def test_gpu_streams_run_frames_at_once_deterministically():
    from nodes.impl.tensorrt.runner import native

    engine, session = _python_session()
    session.close()
    runner = native().Runner(engine.bytes, 0, 4)
    try:
        img = _image(600, 500)
        first = np.empty((600 * engine.scale, 500 * engine.scale, 3), np.float32)
        runner.upscale(bgr_planes(img), bgr_planes(first))
        outs = [np.empty_like(first) for _ in range(8)]
        threads = [
            threading.Thread(
                target=runner.upscale, args=(bgr_planes(img), bgr_planes(o))
            )
            for o in outs
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(np.array_equal(o, first) for o in outs)
    finally:
        runner.close()
        engine.close()
