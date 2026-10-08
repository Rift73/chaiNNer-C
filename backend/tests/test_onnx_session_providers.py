"""An ONNX session runs on the execution provider chosen in the ONNX settings, or
fails with the reason. ONNX Runtime itself falls back to the next provider when one
cannot start (TensorRT onto CUDA, CUDA onto the CPU) and only logs why. Runs without
a GPU: the GPU sessions are stand-ins that report the providers ORT registered."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from nodes.impl.onnx.model import OnnxGeneric, OnnxInfo
from nodes.impl.onnx.session import get_onnx_session

TENSORRT = "TensorrtExecutionProvider"
CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"


def identity_model() -> OnnxGeneric:
    image = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 4])
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], "identity", [image], [output])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    )
    return OnnxGeneric(model.SerializeToString(), OnnxInfo(13, "fp32"))


class RegisteredSession:
    """Stands in for an ORT session that registered `providers`."""

    def __init__(self, providers: Sequence[str]) -> None:
        self.providers = list(providers)

    def get_providers(self) -> list[str]:
        return self.providers


def ort_registers(monkeypatch: pytest.MonkeyPatch, providers: Sequence[str]) -> None:
    monkeypatch.setattr(
        ort, "InferenceSession", lambda *args, **kwargs: RegisteredSession(providers)
    )


def test_tensorrt_that_cannot_load_is_reported_not_run_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
):
    # ORT 1.30's TensorRT provider needs nvinfer_10.dll; without it ORT runs on CUDA.
    ort_registers(monkeypatch, [CUDA, CPU])
    model = identity_model()
    for _ in range(2):  # not cached: the next run tries again
        with pytest.raises(RuntimeError) as raised:
            get_onnx_session(model, 0, TENSORRT, False)
        message = str(raised.value)
        assert f"({TENSORRT})" in message
        assert f"{CUDA}, {CPU} instead" in message
        assert "TensorRT 10 (nvinfer_10.dll)" in message
        assert "TensorRT nodes" in message


def test_cuda_that_cannot_load_is_reported_not_run_on_the_cpu(
    monkeypatch: pytest.MonkeyPatch,
):
    ort_registers(monkeypatch, [CPU])
    with pytest.raises(RuntimeError, match=rf"\({CUDA}\).* on {CPU} instead") as raised:
        get_onnx_session(identity_model(), 0, CUDA, False)
    assert "TensorRT" not in str(raised.value)


def test_a_provider_ort_does_not_have_is_reported():
    # Real ORT on the CPU: it drops a provider it does not have and runs on the CPU.
    with pytest.raises(RuntimeError, match=rf"\(NoSuchExecutionProvider\).* {CPU}"):
        get_onnx_session(identity_model(), 0, "NoSuchExecutionProvider", False)


@pytest.mark.parametrize(
    ("chosen", "registered"),
    [(TENSORRT, [TENSORRT, CUDA, CPU]), (CUDA, [CUDA, CPU])],
)
def test_a_provider_that_starts_gets_its_session(
    monkeypatch: pytest.MonkeyPatch, chosen: str, registered: list[str]
):
    ort_registers(monkeypatch, registered)
    model = identity_model()
    session = get_onnx_session(model, 0, chosen, False)
    assert isinstance(session, RegisteredSession)
    assert session.providers == registered
    assert get_onnx_session(model, 0, chosen, False) is session


def test_the_cpu_provider_gets_its_session():
    model = identity_model()
    session = get_onnx_session(model, 0, CPU, False)
    image = np.ones((1, 3, 4, 4), np.float32)
    np.testing.assert_array_equal(session.run(None, {"input": image})[0], image)
