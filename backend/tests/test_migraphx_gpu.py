"""MIGraphX programs on an AMD GPU: compiled in a child process, read back, saved and
loaded, and run by the runner on images larger than a tile, against NumPy. They need an
AMD GPU that ROCm supports, the runner's module and the MIGraphX package's dependencies;
without an AMD GPU they are skipped, and with one, anything missing fails them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from migraphx_stand_ins import context

import gpu
from nodes.impl.compressed_file import Compression, read_file, write_file
from nodes.impl.migraphx.model import MIGraphXProgram
from nodes.impl.migraphx.program import compile_in_child
from nodes.impl.migraphx.program_info import read_program_info
from nodes.impl.migraphx.session import RunnerSession

pytestmark = pytest.mark.amd_gpu

TILE = (64, 96)


@pytest.fixture(scope="module")
def device() -> gpu.AmdDevice:
    if not gpu.amd.is_available:
        pytest.skip("no AMD GPU")
    return gpu.amd.devices[0]


def upscale_onnx() -> bytes:
    """A 2x upscale of any size: a 1x1 identity convolution, then nearest-neighbour."""
    from onnx import TensorProto, helper

    weight = helper.make_tensor(
        "w", TensorProto.FLOAT, [3, 3, 1, 1], np.eye(3, dtype=np.float32).ravel()
    )
    scales = helper.make_tensor("scales", TensorProto.FLOAT, [4], [1, 1, 2, 2])
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["input", "w"], ["conv"]),
            helper.make_node(
                "Resize", ["conv", "", "scales"], ["output"], mode="nearest"
            ),
        ],
        "upscale",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "H", "W"])],
        initializer=[weight, scales],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model.SerializeToString()


def nearest(image: np.ndarray) -> np.ndarray:
    return image.repeat(2, 0).repeat(2, 1)


def image() -> np.ndarray:
    # Larger than one tile on both axes: the runner tiles and blends it.
    return np.random.default_rng(7).random((150, 200, 3), dtype=np.float32)


@pytest.mark.parametrize(
    ("fp16", "precision", "tolerance"), [(False, "fp32", 1e-5), (True, "fp16", 2e-3)]
)
def test_a_compiled_program_upscales_as_numpy(
    device: gpu.AmdDevice, fp16: bool, precision: str, tolerance: float
):
    saved = compile_in_child(
        upscale_onnx(),
        device.index,
        TILE,
        fp16=fp16,
        fp8=False,
        exhaustive_tune=False,
    )
    info = read_program_info(saved, device.index)
    assert (info.scale, info.input_channels, info.output_channels) == (2, 3, 3)
    assert (info.height, info.width, info.gfx) == (*TILE, device.gfx)
    assert info.precision == precision
    session = RunnerSession(MIGraphXProgram(saved, info), device.index)
    try:
        source = image()
        assert np.abs(session.upscale(source) - nearest(source)).max() < tolerance
    finally:
        session.close()


def test_fp8_runs_where_the_gpu_supports_it(device: gpu.AmdDevice):
    if not device.supports_fp8:
        pytest.skip(f"{device.gfx} has no FP8")
    saved = compile_in_child(
        upscale_onnx(), device.index, TILE, fp16=False, fp8=True, exhaustive_tune=False
    )
    info = read_program_info(saved, device.index)
    session = RunnerSession(MIGraphXProgram(saved, info), device.index)
    try:
        source = image()
        # E4M3 keeps 3 mantissa bits: values in [0, 1) within 1/16 relative.
        assert np.abs(session.upscale(source) - nearest(source)).max() < 0.07
    finally:
        session.close()


def test_a_saved_program_loads_and_runs_the_same(device: gpu.AmdDevice, tmp_path: Path):
    saved = compile_in_child(
        upscale_onnx(), device.index, TILE, fp16=False, fp8=False, exhaustive_tune=False
    )
    path = tmp_path / "model_compressed.mxr"
    write_file(path, saved, compression=Compression.ZSTD)
    loaded = read_file(path)
    assert loaded == saved
    session = RunnerSession(
        MIGraphXProgram(loaded, read_program_info(loaded, device.index)), device.index
    )
    try:
        source = image()
        assert np.abs(session.upscale(source) - nearest(source)).max() < 1e-5
    finally:
        session.close()


def test_the_nodes_build_and_upscale(device: gpu.AmdDevice, tmp_path: Path):
    import packages.chaiNNer_migraphx as package
    from nodes.impl.onnx.load import load_onnx_model
    from nodes.impl.tensorrt import cache
    from packages.chaiNNer_migraphx.migraphx.processing import upscale_image
    from packages.chaiNNer_migraphx.migraphx.utility import build_program

    assert package.migraphx_loaded, package.package.disabled_reason
    settings: dict[str, object] = {
        "gpu_index": str(device.index),
        "program_cache": str(tmp_path),
    }
    model = load_onnx_model(upscale_onnx())
    program = build_program.build_program_node(
        context(settings, "build"),
        model,
        build_program.ProgramPrecision.FP32,
        *TILE,
        False,
    )
    assert list(tmp_path.iterdir())  # kept in the Program Cache
    upscale_context = context(settings, "upscale")
    source = image()
    try:
        result = upscale_image.upscale_image_node(
            upscale_context, source, program, False
        )
        assert np.abs(result - nearest(source)).max() < 1e-5
    finally:
        cache.release_node(upscale_context.node_id)
