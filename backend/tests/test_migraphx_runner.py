"""The native MIGraphX runner (native/migraphx_runner): its tile planner and CPU frame loop
(the TensorRT runner's, shared), ONNX models compiled with custom ops on MIGraphX's CPU
reference target (ref) against NumPy, saved programs, and the checks made before any HIP
call; these need no AMD GPU. The custom-op tests take the build's example op library
(native/migraphx_runner/tests/example_ops.hip, chainner_example_migx_ops.so in the build
directory) from CHAINNER_C_MIGRAPHX_TEST_OPS. The tests marked amd_gpu run programs, the
runner and a GPU custom op on an AMD GPU (the AMD kit)."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from ctypes.util import find_library
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import ModelProto, NodeProto, TensorProto, helper, numpy_helper

from nodes.impl.tensorrt.tiling import (
    ShapeBounds,
    _tile_extent,
    axis_weights,
    tile_starts,
    tiled_upscale,
)

MODULE = "nodes.impl.migraphx._chainner_migraphx"
TEST_OPS = os.environ.get("CHAINNER_C_MIGRAPHX_TEST_OPS", "")
BACKEND_SRC = Path(__file__).resolve().parents[1] / "src"
# The example ops' attributes; the tag's NUL checks that strings arrive byte for byte.
GAIN = 1.5
ORDER = [2, 0, 1]
BIAS = [0.25, -0.5, 1.0]
REPEAT: dict[str, Any] = {
    "factor": 2,
    "gain": GAIN,
    "order": ORDER,
    "bias": BIAS,
    "tag": b"chain\0ner",
}


def _module() -> ModuleType:
    """_chainner_migraphx, MIGraphX's libraries loaded as the MIGraphX package loads them
    (from ONNX Runtime's MIGraphX build) when that is installed, else found by the loader
    (LD_LIBRARY_PATH)."""
    if find_spec(MODULE) is None:
        pytest.skip(
            "_chainner_migraphx.so is not built (native/build_linux.sh with"
            " CHAINNER_C_ROCM_ROOT and CHAINNER_C_MIGRAPHX_ROOT)"
        )
    try:
        return import_module("nodes.impl.migraphx.runner").load_migraphx()
    except ImportError:
        try:
            return import_module(MODULE)
        except ImportError as error:
            pytest.skip(f"MIGraphX's libraries do not load: {error}")


@pytest.fixture(scope="module")
def migx() -> ModuleType:
    return _module()


@pytest.fixture(scope="module")
def ops(migx: ModuleType) -> list[str]:
    if not Path(TEST_OPS).is_file():
        pytest.skip(
            "custom-op tests need the build's chainner_example_migx_ops.so in"
            " CHAINNER_C_MIGRAPHX_TEST_OPS"
        )
    return migx.load_op_library(TEST_OPS)


def model(
    nodes: list[NodeProto],
    inputs: dict[str, list[int | str]],
    outputs: list[str],
    weights: dict[str, np.ndarray] | None = None,
    elem: int = TensorProto.FLOAT,
) -> bytes:
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info(n, elem, s) for n, s in inputs.items()],
        [helper.make_tensor_value_info(n, elem, None) for n in outputs],
        [numpy_helper.from_array(a, n) for n, a in (weights or {}).items()],
    )
    proto: ModelProto = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("chainner", 1)],
    )
    return proto.SerializeToString()


def repeat_node(op: str, x: str, y: str, **overrides: Any) -> NodeProto:
    attributes: dict[str, Any] = {**REPEAT, **overrides}
    return helper.make_node(op, [x], [y], domain="chainner", **attributes)


def repeated(x: np.ndarray) -> np.ndarray:
    """The example ops on NumPy, in float32 as they compute."""
    bias = np.array(BIAS, np.float32)[None, :, None, None]
    x = x[:, ORDER].repeat(2, axis=2).repeat(2, axis=3)
    return x * np.float32(GAIN) + bias


def conv3x3(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """A 3x3 convolution with padding 1 (NCHW, OIHW), in float64."""
    h, wd = x.shape[2:]
    p = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (1, 1), (1, 1)))
    out = np.zeros((x.shape[0], w.shape[0], h, wd))
    for dy in range(3):
        for dx in range(3):
            out += np.einsum(
                "oc,nchw->nohw", w[:, :, dy, dx], p[:, :, dy : dy + h, dx : dx + wd]
            )
    return out


def run(program: Any, inputs: dict[str, np.ndarray], outputs: list[np.ndarray]) -> None:
    """A ref program's run on host arrays."""
    keep = {n: np.ascontiguousarray(a) for n, a in inputs.items()}
    program.run(
        {n: a.ctypes.data for n, a in keep.items()}, [o.ctypes.data for o in outputs]
    )


def _image(h: int, w: int, c: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 1.0, (h, w, c)).astype(np.float32)


def bgr_planes(img: np.ndarray) -> list[np.ndarray]:
    return [img[:, :, c] for c in range(img.shape[2])[::-1]]


def nearest(scale: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda tensor: tensor.repeat(scale, axis=1).repeat(scale, axis=2)


class NearestEngine:
    """tiling.TileRunner of a nearest-neighbour 2x upscale."""

    def __init__(self) -> None:
        self.tiles: list[np.ndarray] = []

    def submit(self, tile: np.ndarray) -> None:
        self.tiles.append(tile.repeat(2, axis=0).repeat(2, axis=1))

    def collect(self, height: int, width: int) -> np.ndarray:
        return self.tiles.pop(0)[:height, :width]


@pytest.mark.parametrize(
    ("h", "w", "bounds"),
    [
        (1024, 700, ShapeBounds(512, 512, 512, 512)),
        (300, 1200, ShapeBounds(64, 64, 512, 512)),
    ],
)
def test_planner_and_frame_loop_are_the_tensorrt_runners(
    migx: ModuleType, h: int, w: int, bounds: ShapeBounds
):
    """The runner shares trt_runner's planner and frame loop: tiling.py's tiles, weights
    and blend, exactly."""
    b = (bounds.min_h, bounds.min_w, bounds.max_h, bounds.max_w, bounds.alignment)
    plan = migx.plan_tiles(h, w, 2, b)
    tile_h = _tile_extent(bounds.max_h, bounds.min_h, bounds.alignment)
    tile_w = _tile_extent(bounds.max_w, bounds.min_w, bounds.alignment)
    assert (plan["ys"], plan["xs"]) == (tile_starts(h, tile_h), tile_starts(w, tile_w))
    for got, want in zip(
        (*plan["wy"], *plan["wx"]),
        (
            *axis_weights(plan["ys"], tile_h, h, 2),
            *axis_weights(plan["xs"], tile_w, w, 2),
        ),
    ):
        assert np.array_equal(got, want)
    img = _image(h, w)
    out = np.empty((h * 2, w * 2, 3), np.float32)
    migx.reference_upscale(
        bgr_planes(img), bgr_planes(out), 2, b, 3, 3, "fp32", "fp32", nearest(2)
    )
    assert np.array_equal(out, tiled_upscale(img, 2, bounds, NearestEngine()))


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((b"", 0, 1), "program is empty"),
        ((b"x", -1, 1), "GPU index"),
        ((b"x", 0, 0), "stream count"),
        ((b"x", 0, 17), "stream count"),
        ((b"not a program", 0, 1), "not a chaiNNer-C MIGraphX program"),
    ],
)
def test_runner_arguments_are_checked_before_hip(
    migx: ModuleType, args: tuple[bytes, int, int], message: str
):
    with pytest.raises(ValueError, match=message):
        migx.Runner(*args)


def test_compile_checks_its_arguments(migx: ModuleType):
    data = model([helper.make_node("Relu", ["x"], ["y"])], {"x": [1, 3, 8, 8]}, ["y"])
    with pytest.raises(ValueError, match="gpu or ref"):
        migx.compile_onnx(data, target="cpu")
    with pytest.raises(ValueError, match="GPU index"):
        migx.compile_onnx(data, device=-1)
    with pytest.raises(ValueError, match="fp16 or fp8"):
        migx.compile_onnx(data, target="ref", fp16=True, fp8=True)
    with pytest.raises(ValueError, match="calibration"):
        migx.compile_onnx(data, target="ref", fp8=True)
    with pytest.raises(ValueError, match="damaged"):
        migx.compile_onnx(b"\x08\x07garbage", target="ref")
    with pytest.raises(ValueError, match="not a chaiNNer-C MIGraphX program"):
        migx.load_program(b"\x93\x01", 0)


def conv_model(w: np.ndarray) -> ModelProto:
    graph = helper.make_graph(
        [helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        [numpy_helper.from_array(w, "w")],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_model_file_with_external_data(migx: ModuleType, tmp_path: Path):
    """A model file whose weights lie beside it compiles as its bytes with the weights
    inside."""
    rng = np.random.default_rng(7)
    proto = conv_model((rng.standard_normal((8, 3, 3, 3)) * 0.2).astype(np.float32))
    data = proto.SerializeToString()
    path = tmp_path / "model.onnx"
    onnx.save_model(
        proto,
        path,
        save_as_external_data=True,
        location="model.data",
        size_threshold=0,
    )
    x = rng.uniform(0, 1, (1, 3, 32, 32)).astype(np.float32)
    y, want = np.zeros((2, 1, 8, 32, 32), np.float32)
    run(migx.compile_onnx(path, target="ref"), {"x": x}, [y])
    run(migx.compile_onnx(data, target="ref"), {"x": x}, [want])
    assert np.array_equal(y, want)


def test_fp8_quantization_calibrates(migx: ModuleType):
    """fp8: MIGraphX's quantization, scaled from the calibration inputs, within FP8's
    rounding (e4m3: 3 mantissa bits) of float32."""
    rng = np.random.default_rng(8)
    w = (rng.standard_normal((8, 3, 3, 3)) * 0.2).astype(np.float32)
    data = conv_model(w).SerializeToString()
    x = rng.uniform(0, 1, (1, 3, 32, 32)).astype(np.float32)
    program = migx.compile_onnx(data, target="ref", fp8=True, calibration=[{"x": x}])
    assert program.compute_type == "fp8e4m3fn"
    assert program.input_types == ["float32"]
    y = np.zeros((1, 8, 32, 32), np.float32)
    run(program, {"x": x}, [y])
    want = conv3x3(x, w)
    assert np.abs(y - want).max() <= 0.1 * np.abs(want).max()


@pytest.mark.parametrize("fp16", [False, True], ids=["fp32", "fp16"])
def test_ref_program_matches_numpy_and_reloads(
    migx: ModuleType, tmp_path: Path, fp16: bool
):
    """Convolution, ReLU and a transpose at the end, with dynamic sizes fixed by
    input_dims: NumPy's result (within float16's rounding for fp16), packed, and the
    same again from the saved program, as bytes and as a file."""
    rng = np.random.default_rng(1)
    w = (rng.standard_normal((4, 3, 3, 3)) * 0.2).astype(np.float32)
    data = model(
        [
            helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]),
            helper.make_node("Relu", ["c"], ["r"]),
            helper.make_node("Transpose", ["r"], ["y"], perm=[0, 1, 3, 2]),
        ],
        {"x": [1, 3, "h", "w"]},
        ["y"],
        {"w": w},
    )
    program = migx.compile_onnx(
        data, target="ref", fp16=fp16, input_dims={"x": (1, 3, 20, 12)}
    )
    assert (program.target, program.gfx, program.custom_ops) == ("ref", "ref", [])
    assert program.compute_type == ("float16" if fp16 else "float32")
    assert program.input_names == ["x"]
    assert (program.input_shapes, program.output_shapes) == (
        [(1, 3, 20, 12)],
        [(1, 4, 12, 20)],
    )
    assert (program.input_types, program.output_types) == (["float32"], ["float32"])
    x = rng.uniform(0, 1, (1, 3, 20, 12)).astype(np.float32)
    want = np.maximum(conv3x3(x, w), 0).transpose(0, 1, 3, 2)
    y = np.zeros((1, 4, 12, 20), np.float32)
    run(program, {"x": x}, [y])
    assert np.allclose(y, want, atol=2e-2 if fp16 else 1e-5)
    path = tmp_path / "program.mxp"
    program.save(path)
    assert path.read_bytes() == program.to_bytes()
    for source in (program.to_bytes(), path, str(path)):
        again = migx.load_program(source, 0)
        assert again.compute_type == program.compute_type
        y2 = np.zeros_like(y)
        run(again, {"x": x}, [y2])
        assert np.array_equal(y2, y)
        again.close()
    program.close()
    with pytest.raises(RuntimeError, match="closed"):
        run(program, {"x": x}, [y])


def test_custom_op_takes_its_node_and_shapes_the_rest(migx: ModuleType, ops: list[str]):
    """A custom op that changes its input's size, its attributes from the node, followed by
    a convolution and a broadcast against the size it makes (which only parse once the
    op's output shape is known)."""
    assert ops == [
        "chainner_example_repeat",
        "chainner_example_repeat_float",
        "chainner_example_repeat_hip",
    ]
    assert migx.load_op_library(TEST_OPS) == ops
    rng = np.random.default_rng(2)
    w = (rng.standard_normal((3, 3, 3, 3)) * 0.2).astype(np.float32)
    big = rng.standard_normal((1, 3, 16, 16)).astype(np.float32)
    data = model(
        [
            repeat_node("chainner_example_repeat", "x", "r"),
            helper.make_node("Add", ["r", "big"], ["a"]),
            helper.make_node("Conv", ["a", "w"], ["y"], pads=[1, 1, 1, 1]),
        ],
        {"x": [1, 3, 8, 8]},
        ["y"],
        {"w": w, "big": big},
    )
    program = migx.compile_onnx(data, target="ref")
    assert program.custom_ops == ["chainner_example_repeat"]
    assert program.output_shapes == [(1, 3, 16, 16)]
    x = rng.standard_normal((1, 3, 8, 8)).astype(np.float32)
    y = np.zeros((1, 3, 16, 16), np.float32)
    run(program, {"x": x}, [y])
    assert np.allclose(y, conv3x3(repeated(x) + big, w), atol=1e-5)
    again = migx.load_program(program.to_bytes(), 0)
    y2 = np.zeros_like(y)
    run(again, {"x": x}, [y2])
    assert np.array_equal(y2, y)


def test_quantization_reaches_custom_ops_unless_they_keep_their_types(
    migx: ModuleType, ops: list[str]
):
    """fp16: the op without CHAINNER_MIGX_FIXED_TYPES gets MIGraphX's float16 of the input
    (the ref target truncates) and gives its float32 result rounded to float16; the one
    with it keeps float32 (it refuses float16)."""
    x = np.random.default_rng(3).standard_normal((1, 3, 8, 8)).astype(np.float32)
    casts = model(
        [
            helper.make_node("Cast", ["x"], ["h"], to=TensorProto.FLOAT16),
            helper.make_node("Cast", ["h"], ["y"], to=TensorProto.FLOAT),
        ],
        {"x": [1, 3, 8, 8]},
        ["y"],
    )
    half = np.zeros_like(x)
    run(migx.compile_onnx(casts, target="ref"), {"x": x}, [half])
    assert not np.array_equal(half, x)
    for op, want in (
        (
            "chainner_example_repeat",
            repeated(half).astype(np.float16).astype(np.float32),
        ),
        ("chainner_example_repeat_float", repeated(x)),
    ):
        data = model([repeat_node(op, "x", "y")], {"x": [1, 3, 8, 8]}, ["y"])
        y = np.zeros((1, 3, 16, 16), np.float32)
        run(migx.compile_onnx(data, target="ref", fp16=True), {"x": x}, [y])
        assert np.array_equal(y, want), op


@pytest.mark.parametrize(
    ("node", "error", "message"),
    [
        (
            helper.make_node(
                "chainner_example_missing", ["x"], ["y"], domain="chainner"
            ),
            ValueError,
            "chainner_example_missing, which is not loaded",
        ),
        (
            repeat_node("chainner_example_repeat", "x", "y", factor=0),
            RuntimeError,
            "chainner_example_repeat: missing attribute",
        ),
        (
            helper.make_node(
                "chainner_example_repeat", ["x"], ["y", "z"], domain="chainner"
            ),
            ValueError,
            "has 2 outputs",
        ),
        (
            helper.make_node(
                "chainner_example_repeat", ["x", ""], ["y"], domain="chainner"
            ),
            ValueError,
            "omits an input",
        ),
    ],
    ids=["not loaded", "op refuses", "two outputs", "omitted input"],
)
def test_custom_node_errors_name_the_node(
    migx: ModuleType,
    ops: list[str],
    node: NodeProto,
    error: type[Exception],
    message: str,
):
    data = model([node], {"x": [1, 3, 8, 8]}, list(node.output))
    with pytest.raises(error, match=message):
        migx.compile_onnx(data, target="ref")


def test_custom_op_refuses_input_types(migx: ModuleType, ops: list[str]):
    data = model(
        [repeat_node("chainner_example_repeat", "x", "y")],
        {"x": [1, 3, 8, 8]},
        ["y"],
        elem=TensorProto.INT32,
    )
    with pytest.raises(RuntimeError, match="takes float32 or float16"):
        migx.compile_onnx(data, target="ref")


def test_op_libraries_are_checked(migx: ModuleType, tmp_path: Path):
    with pytest.raises(RuntimeError, match="was not found"):
        migx.load_op_library(tmp_path / "missing.so")
    (tmp_path / "text.so").write_text("not a library")
    with pytest.raises(RuntimeError, match="did not load"):
        migx.load_op_library(tmp_path / "text.so")
    # A library without the export: Windows' kernel32.dll, Linux's C math library
    if sys.platform == "win32":
        library = Path(os.environ["SystemRoot"], "System32", "kernel32.dll")
    else:
        libm = find_library("m")
        if libm is None:
            pytest.skip("no libm to load")
        library = Path("/lib/x86_64-linux-gnu") / libm
    with pytest.raises(RuntimeError, match="does not export chainner_migx_ops"):
        migx.load_op_library(library)


def test_saved_program_needs_its_op_library(
    migx: ModuleType, ops: list[str], tmp_path: Path
):
    """In a fresh process: a program with a custom op loads only after its library."""
    data = model(
        [repeat_node("chainner_example_repeat", "x", "y")], {"x": [1, 3, 8, 8]}, ["y"]
    )
    path = tmp_path / "program.mxp"
    migx.compile_onnx(data, target="ref").save(path)
    code = f"""
import sys
sys.path.insert(0, {str(BACKEND_SRC)!r})
from importlib import import_module
try:
    migx = import_module("nodes.impl.migraphx.runner").load_migraphx()
except ImportError:
    migx = import_module({MODULE!r})
try:
    migx.load_program({str(path)!r}, 0)
except ValueError as error:
    print(error)
migx.load_op_library({TEST_OPS!r})
print(migx.load_program({str(path)!r}, 0).custom_ops)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        stdin=subprocess.DEVNULL,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert "chainner_example_repeat, which is not loaded" in lines[0]
    assert lines[-1] == "['chainner_example_repeat']"


# AMD GPU tests: the AMD kit (an AMD GPU and ROCm's HIP runtime).
def _amd_gpu() -> bool:
    gpu = import_module("gpu")
    return bool(getattr(gpu, "amd", None)) and gpu.amd.is_available


def amd_gpu(test: Callable[..., None]) -> Callable[..., None]:
    """Marks a test for the AMD kit and skips it without an AMD GPU."""
    return pytest.mark.amd_gpu(
        pytest.mark.skipif(not _amd_gpu(), reason="needs an AMD GPU")(test)
    )


class Hip:
    """HIP's device memory through its runtime library, for programs run on device
    pointers."""

    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libamdhip64.so.7")
        self.lib.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.hipMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.lib.hipFree.argtypes = [ctypes.c_void_p]
        self.buffers: list[int] = []

    def check(self, status: int, call: str) -> None:
        assert status == 0, f"{call}: HIP error {status}"

    def upload(self, array: np.ndarray) -> int:
        array = np.ascontiguousarray(array)
        pointer = ctypes.c_void_p()
        self.check(self.lib.hipMalloc(ctypes.byref(pointer), array.nbytes), "hipMalloc")
        self.buffers.append(pointer.value or 0)
        self.check(
            self.lib.hipMemcpy(pointer, array.ctypes.data, array.nbytes, 1), "hipMemcpy"
        )
        return pointer.value or 0

    def download(self, pointer: int, out: np.ndarray) -> np.ndarray:
        self.check(self.lib.hipDeviceSynchronize(), "hipDeviceSynchronize")
        self.check(
            self.lib.hipMemcpy(out.ctypes.data, pointer, out.nbytes, 2), "hipMemcpy"
        )
        return out

    def free(self) -> None:
        for pointer in self.buffers:
            self.check(self.lib.hipFree(pointer), "hipFree")
        self.buffers.clear()


def upscale_model() -> bytes:
    """A 2x nearest-neighbour upscale (Resize, asymmetric floor), exact everywhere."""
    return model(
        [
            helper.make_node(
                "Resize",
                ["input", "", "scales"],
                ["output"],
                mode="nearest",
                coordinate_transformation_mode="asymmetric",
                nearest_mode="floor",
            )
        ],
        {"input": [1, 3, "h", "w"]},
        ["output"],
        {"scales": np.array([1, 1, 2, 2], np.float32)},
    )


@amd_gpu
@pytest.mark.parametrize("fp16", [False, True], ids=["fp32", "fp16"])
def test_gpu_program_matches_ref(migx: ModuleType, fp16: bool):
    rng = np.random.default_rng(4)
    w = (rng.standard_normal((4, 3, 3, 3)) * 0.2).astype(np.float32)
    data = model(
        [helper.make_node("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1])],
        {"x": [1, 3, 64, 48]},
        ["y"],
        {"w": w},
    )
    program = migx.compile_onnx(data, device=0, fp16=fp16)
    assert program.target == "gpu" and program.gfx.startswith("gfx")
    x = rng.uniform(0, 1, (1, 3, 64, 48)).astype(np.float32)
    hip = Hip()
    try:
        out = hip.upload(np.zeros((1, 4, 64, 48), np.float32))
        program.run({"x": hip.upload(x)}, [out], 0)
        y = hip.download(out, np.zeros((1, 4, 64, 48), np.float32))
    finally:
        hip.free()
    assert np.allclose(y, conv3x3(x, w), atol=2e-2 if fp16 else 1e-4)
    again = migx.load_program(program.to_bytes(), 0)
    assert again.gfx == program.gfx


@amd_gpu
@pytest.mark.parametrize("size", [(50, 40), (150, 200)], ids=["one tile", "tiled"])
def test_gpu_runner_matches_reference_upscale(migx: ModuleType, size: tuple[int, int]):
    """The runner's HIP kernels and MIGraphX's program give what the CPU frame loop gives
    with NumPy's nearest-neighbour upscale: exactly, for float32 BGR and for 10-bit
    limited-range gray planes through the RGB program, clipped."""
    program = migx.compile_onnx(
        upscale_model(), device=0, input_dims={"input": (1, 3, 64, 64)}
    )
    runner = migx.Runner(program.to_bytes(), 0, 2)
    try:
        assert (runner.scale, runner.bounds) == (2, (64, 64, 64, 64, 64))
        assert (runner.input_precision, runner.output_precision) == ("fp32", "fp32")
        h, w = size
        img = _image(h, w)
        for _ in range(2):
            got = np.empty((h * 2, w * 2, 3), np.float32)
            runner.upscale(bgr_planes(img), bgr_planes(got))
            want = np.empty_like(got)
            migx.reference_upscale(
                bgr_planes(img),
                bgr_planes(want),
                2,
                runner.bounds,
                3,
                3,
                "fp32",
                "fp32",
                nearest(2),
            )
            assert np.array_equal(got, want)
        codes = np.random.default_rng(5).integers(0, 1024, (h, w)).astype(np.uint16)
        got16 = np.empty((h * 2, w * 2), np.uint16)
        want16 = np.empty_like(got16)
        runner.upscale([codes], [got16], bits=10, full_range=False, clip=True)
        migx.reference_upscale(
            [codes],
            [want16],
            2,
            runner.bounds,
            3,
            3,
            "fp32",
            "fp32",
            nearest(2),
            bits=10,
            full_range=False,
            clip=True,
        )
        assert np.array_equal(got16, want16)
        assert runner.capture_failure() == ""
    finally:
        runner.close()


@amd_gpu
def test_gpu_streams_run_frames_at_once_deterministically(migx: ModuleType):
    program = migx.compile_onnx(
        upscale_model(), device=0, input_dims={"input": (1, 3, 64, 64)}
    )
    runner = migx.Runner(program.to_bytes(), 0, 4)
    try:
        img = _image(150, 130)
        first = np.empty((300, 260, 3), np.float32)
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


@amd_gpu
@pytest.mark.parametrize("fp16", [False, True], ids=["fp32", "fp16"])
def test_gpu_custom_op_runs_on_the_device(migx: ModuleType, ops: list[str], fp16: bool):
    """The GPU example op (scratch memory, the program's stream, float32 kept under fp16)
    between two additions: NumPy's result exactly."""
    rng = np.random.default_rng(6)
    a = rng.standard_normal((1, 3, 1, 1)).astype(np.float32)
    data = model(
        [
            helper.make_node("Add", ["x", "a"], ["s"]),
            repeat_node("chainner_example_repeat_hip", "s", "r"),
            helper.make_node("Add", ["r", "r"], ["y"]),
        ],
        {"x": [1, 3, 32, 24]},
        ["y"],
        {"a": a},
    )
    program = migx.compile_onnx(data, device=0, fp16=fp16)
    assert program.custom_ops == ["chainner_example_repeat_hip"]
    x = rng.standard_normal((1, 3, 32, 24)).astype(np.float32)
    hip = Hip()
    try:
        out = hip.upload(np.zeros((1, 3, 64, 48), np.float32))
        program.run({"x": hip.upload(x)}, [out], 0)
        y = hip.download(out, np.zeros((1, 3, 64, 48), np.float32))
    finally:
        hip.free()
    s = x + a
    if fp16:
        s = s.astype(np.float16).astype(np.float32)
    r = repeated(s)
    want = r + r
    if fp16:
        want = (r.astype(np.float16) + r.astype(np.float16)).astype(np.float32)
    assert np.array_equal(y, want)
