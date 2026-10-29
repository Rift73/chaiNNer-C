"""ONNX on an AMD GPU through ONNX Runtime's MIGraphX provider: a session gives the
CPU's results (within FP32 or FP16 rounding) at every input shape, the program cache
keeps one program per model, precision and shape and a later session loads them
instead of compiling, and Upscale Image's tiles come out as on the CPU. Needs an AMD
GPU and a MIGraphX build of ONNX Runtime; the AMD kit runs it."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

import gpu
from api import NodeContext, SettingsParser
from nodes.impl.onnx.load import load_onnx_model
from nodes.impl.onnx.model import OnnxGeneric
from nodes.impl.onnx.session import OnnxSession, get_onnx_session
from nodes.impl.upscale.auto_split_tiles import CUSTOM
from packages.chaiNNer_onnx.onnx.processing.upscale_image import upscale_image_node
from packages.chaiNNer_onnx.settings import can_cache_migraphx

MIGRAPHX = "MIGraphXExecutionProvider"
CPU = "CPUExecutionProvider"

pytestmark = [
    pytest.mark.amd_gpu,
    pytest.mark.skipif(
        MIGRAPHX not in ort.get_available_providers() or not gpu.amd.is_available,
        reason="needs an AMD GPU and ONNX Runtime's MIGraphX provider",
    ),
]
# Load Model and the tiling run native code, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

# (height, width): the third run returns to the first shape
SHAPES = [(48, 64), (72, 40), (48, 64)]
# Tolerances against the CPU's FP32 result
TOLERANCE = {False: 1e-4, True: 2e-2}


def upscaler(seed: int) -> OnnxGeneric:
    """A small 2x model (convolution, ReLU, convolution, depth to space) with dynamic
    height and width, its weights from `seed`."""
    rng = np.random.default_rng(seed)

    def weight(name: str, shape: tuple[int, ...]) -> TensorProto:
        return numpy_helper.from_array(
            (rng.standard_normal(shape) * 0.1).astype(np.float32), name
        )

    image = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "h", "w"])
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3, "h2", "w2"]
    )
    nodes = [
        helper.make_node("Conv", ["input", "w1", "b1"], ["c1"], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c1"], ["r1"]),
        helper.make_node("Conv", ["r1", "w2", "b2"], ["c2"], pads=[1, 1, 1, 1]),
        helper.make_node("DepthToSpace", ["c2"], ["output"], blocksize=2),
    ]
    initializers = [
        weight("w1", (16, 3, 3, 3)),
        weight("b1", (16,)),
        weight("w2", (12, 16, 3, 3)),
        weight("b2", (12,)),
    ]
    graph = helper.make_graph(nodes, "upscaler", [image], [output], initializers)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )
    loaded = load_onnx_model(model.SerializeToString())
    assert isinstance(loaded, OnnxGeneric)
    return loaded


def inputs() -> list[np.ndarray]:
    rng = np.random.default_rng(3)
    return [rng.random((1, 3, h, w), dtype=np.float32) for h, w in SHAPES]


def outputs(session: OnnxSession) -> list[np.ndarray]:
    return [np.asarray(session.run(None, {"input": x})[0]) for x in inputs()]


def cpu_outputs(model: OnnxGeneric) -> list[np.ndarray]:
    return outputs(ort.InferenceSession(model.bytes, providers=[CPU]))


def migraphx_session(
    model: OnnxGeneric, fp16: bool = False, cache: Path | None = None
) -> OnnxSession:
    # The GPU the ONNX settings default to: the strongest AMD GPU
    session = get_onnx_session(
        model,
        gpu.amd.default_index,
        MIGRAPHX,
        False,
        None,
        fp16,
        None if cache is None else str(cache),
    )
    assert session.get_providers()[0] == MIGRAPHX
    return session


def assert_close(got: list[np.ndarray], want: list[np.ndarray], fp16: bool) -> None:
    assert [g.shape for g in got] == [w.shape for w in want]
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(g, w, rtol=0, atol=TOLERANCE[fp16])


@pytest.mark.parametrize("fp16", [False, True], ids=["fp32", "fp16"])
def test_migraphx_gives_the_cpu_results_at_every_shape(fp16: bool):
    model = upscaler(1)
    assert_close(outputs(migraphx_session(model, fp16)), cpu_outputs(model), fp16)


# The ONNX settings offer the program cache only with a build that names a program by
# its whole input shape, and only where no NVIDIA GPU is detected.
needs_program_cache = pytest.mark.skipif(
    not can_cache_migraphx,
    reason="the ONNX settings offer no MIGraphX program cache here",
)


def program_files(cache: Path) -> dict[Path, int]:
    """The program files in the cache, with their modification times."""
    return {p: p.stat().st_mtime_ns for p in cache.rglob("*.mxr")}


@needs_program_cache
def test_a_later_session_loads_the_cached_programs(tmp_path: Path):
    model = upscaler(1)
    want = cpu_outputs(model)
    first = outputs(migraphx_session(model, cache=tmp_path))
    # Back at the first shape, the first shape's program runs (not the second's).
    assert_close(first, want, False)
    files = program_files(tmp_path)
    assert len(files) >= 2

    # The same model loaded again: a new session, which compiles nothing.
    again = upscaler(1)
    second = outputs(migraphx_session(again, cache=tmp_path))
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
    assert program_files(tmp_path) == files


@needs_program_cache
def test_models_of_one_architecture_keep_their_own_programs(tmp_path: Path):
    for seed in (1, 2):
        model = upscaler(seed)
        got = outputs(migraphx_session(model, cache=tmp_path))
        assert_close(got, cpu_outputs(model), False)
    # Each model compiled its own programs, as did FP16.
    fp16 = upscaler(1)
    assert_close(
        outputs(migraphx_session(fp16, True, tmp_path)), cpu_outputs(fp16), True
    )
    assert len(list(tmp_path.iterdir())) == 3


class Context:
    """The NodeContext members ONNX Upscale Image uses."""

    aborted = False
    paused = False

    def __init__(self, provider: str) -> None:
        self.settings = SettingsParser(
            {"execution_provider": provider, "gpu_index": str(gpu.amd.default_index)}
        )

    def add_cleanup(self, fn: Callable[[], None], after: str = "chain") -> None:
        pass

    def check_aborted(self) -> None:
        pass

    def suspend(self) -> None:
        pass

    def set_progress(self, progress: float) -> None:
        pass


def test_upscale_image_tiles_on_migraphx_as_on_the_cpu():
    # 32 px tiles of a 100x70 image: tiles of several shapes, one after another.
    image = np.random.default_rng(5).random((70, 100, 3), dtype=np.float32)

    def upscale(provider: str) -> np.ndarray:
        context = cast(NodeContext, Context(provider))
        return upscale_image_node(context, image, upscaler(1), CUSTOM, 32, False)

    got, want = upscale(MIGRAPHX), upscale(CPU)
    assert got.shape == want.shape == (140, 200, 3)
    np.testing.assert_allclose(got, want, rtol=0, atol=TOLERANCE[False])
