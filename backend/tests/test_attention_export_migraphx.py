"""On PyTorch for AMD GPUs (ROCm), Convert To ONNX lays HAT's and DAT's attention out for
MIGraphX. Every window attention takes the form MIGraphX fuses into one kernel (its
fuse_attention pass, then rocMLIR): the first MatMul reaches the Softmax through
elementwise operations and constants only (scale, relative position bias, and the
shifted windows' mask, which selects through a Where), and the Softmax feeds a single
MatMul as its first input. DAT's channel attention does not: its Softmax feeds a
Transpose. Cross-compiled with MIGraphX 2.16.1 for gfx942, such an export of a two-group
HAT-L or DAT2 fuses every window attention and compiles; the default export leaves the
shifted blocks' attention unfused (its mask goes through a reshape to five dimensions),
and DAT2's fails to compile there. Elsewhere the export keeps that default (TensorRT's).
Both layouts compute the same values; on an AMD GPU, ONNX Runtime's MIGraphX provider
runs the ROCm layout as PyTorch does (the AMD kit runs that test)."""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from test_convert_to_onnx import tiny_model

from nodes.impl.pytorch.convert_to_onnx_impl import convert_to_onnx_impl

MIGRAPHX = "MIGraphXExecutionProvider"

ELEMENTWISE = {"Add", "Sub", "Mul", "Div", "Where", "Cast"}

# HAT: window attention, shifted window attention, overlapping cross-attention; DAT:
# spatial and shifted spatial attention, each in two halves, and channel attention
ARCHS = ["HAT", "DAT"]
WINDOW_SOFTMAXES = {"HAT": 3, "DAT": 4}
CHANNEL_SOFTMAXES = {"HAT": 0, "DAT": 1}


def export(arch: str, half: bool, rocm: bool, monkeypatch: pytest.MonkeyPatch) -> bytes:
    with monkeypatch.context() as patch:
        # The torch the export sees: a ROCm build or not, whichever runs the tests.
        patch.setattr(torch.version, "hip", "7.14.60850" if rocm else None)
        return convert_to_onnx_impl(tiny_model(arch), torch.device("cpu"), half)


def attention_paths(graph: onnx.GraphProto) -> list[tuple[set[str], set[str], str]]:
    """For each Softmax, the kinds of operation its non-constant inputs reach first, the
    kinds of operation passed on the way, through elementwise operations and reshapes,
    and its users ("MatMul" when a single MatMul takes it as its first input). A value
    of the weights and shapes alone is a constant: MIGraphX folds it for its input
    size."""
    producers = {output: node for node in graph.node for output in node.output}
    inputs = {i.name for i in graph.input}
    constants: dict[str, bool] = {}

    def constant(name: str) -> bool:
        if name not in constants:
            node = producers.get(name)
            constants[name] = name not in inputs and (
                node is None
                or node.op_type == "Shape"
                or all(constant(i) for i in node.input if i)
            )
        return constants[name]

    paths: list[tuple[set[str], set[str], str]] = []
    for softmax in (n for n in graph.node if n.op_type == "Softmax"):
        users = [n for n in graph.node if softmax.output[0] in n.input]
        use = " ".join(u.op_type for u in users)
        if use == "MatMul" and users[0].input[0] != softmax.output[0]:
            use = "MatMul (second input)"
        reached: set[str] = set()
        passed: set[str] = set()
        stack, seen = [softmax.input[0]], set()
        while stack:
            name = stack.pop()
            if name in seen or constant(name):
                continue
            seen.add(name)
            node = producers[name]
            if node.op_type in ELEMENTWISE | {"Reshape"}:
                passed.add(node.op_type)
                stack.extend(i for i in node.input if i)
            else:
                reached.add(node.op_type)
        paths.append((reached, passed, use))
    return paths


@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize("arch", ARCHS)
def test_rocm_lays_attention_out_for_migraphx(
    arch: str, half: bool, monkeypatch: pytest.MonkeyPatch
):
    graph = onnx.load_model_from_string(export(arch, half, True, monkeypatch)).graph
    paths = attention_paths(graph)
    window = [path for path in paths if path[2] == "MatMul"]
    assert len(window) == WINDOW_SOFTMAXES[arch]
    for reached, passed, _ in window:
        assert reached == {"MatMul"}
        assert passed <= ELEMENTWISE
    assert [use for _, _, use in paths if use != "MatMul"] == [
        "Transpose"
    ] * CHANNEL_SOFTMAXES[arch]


@pytest.mark.parametrize("arch", ARCHS)
def test_elsewhere_shifted_windows_keep_tensorrts_view(
    arch: str, monkeypatch: pytest.MonkeyPatch
):
    graph = onnx.load_model_from_string(export(arch, False, False, monkeypatch)).graph
    paths = attention_paths(graph)
    assert all(reached == {"MatMul"} and use == "MatMul" for reached, _, use in paths)
    # one shifted block's attention: HAT's, or DAT's two halves
    assert (
        sum("Reshape" in passed for _, passed, _ in paths) == {"HAT": 1, "DAT": 2}[arch]
    )


@pytest.mark.parametrize("arch", ARCHS)
def test_rocms_export_computes_the_defaults_values(
    arch: str, monkeypatch: pytest.MonkeyPatch
):
    # Two images: the mask tiled over both, against the default's view of each image's
    # windows (DAT's transposed channel product rounds its own way); and each image
    # against PyTorch
    images = np.random.default_rng(0).random((2, 3, 37, 51), dtype=np.float32)
    outputs = [
        np.asarray(
            ort.InferenceSession(
                export(arch, False, rocm, monkeypatch),
                providers=["CPUExecutionProvider"],
            ).run(None, {"input": images})[0]
        )
        for rocm in (True, False)
    ]
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=0, atol=1e-6)
    session = ort.InferenceSession(
        export(arch, False, True, monkeypatch), providers=["CPUExecutionProvider"]
    )
    model = tiny_model(arch)
    for image in images:
        (actual,) = session.run(None, {"input": image[None]})
        with torch.no_grad():
            expected = model(torch.from_numpy(image)[None]).numpy()
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=0, atol=1e-5)


@pytest.mark.amd_gpu
@pytest.mark.skipif(
    MIGRAPHX not in ort.get_available_providers()
    or not (torch.version.hip and torch.cuda.is_available()),
    reason="needs an AMD GPU and ONNX Runtime's MIGraphX provider",
)
@pytest.mark.parametrize(("half", "tolerance"), [(False, 1e-4), (True, 2e-2)])
@pytest.mark.parametrize("arch", ARCHS)
def test_migraphx_runs_rocms_export_as_pytorch(
    arch: str, half: bool, tolerance: float, monkeypatch: pytest.MonkeyPatch
):
    session = ort.InferenceSession(
        export(arch, False, True, monkeypatch),
        providers=[(MIGRAPHX, {"device_id": 0, "migraphx_fp16_enable": half})],
    )
    image = np.random.default_rng(0).random((1, 3, 37, 51), dtype=np.float32)
    (actual,) = session.run(None, {"input": image})
    with torch.no_grad():
        expected = tiny_model(arch)(torch.from_numpy(image)).numpy()
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0, atol=tolerance)
