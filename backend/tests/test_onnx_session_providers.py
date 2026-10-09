"""An ONNX session runs on the execution provider chosen in the ONNX settings, or
fails with the reason. ONNX Runtime itself falls back to the next provider when one
cannot start (TensorRT onto CUDA, CUDA onto the CPU) and only logs why. Runs without
a GPU: the GPU sessions are stand-ins that report the providers ORT registered."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from nodes.impl.onnx import session as session_module
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


def tensorrt_package(monkeypatch: pytest.MonkeyPatch, installed: bool) -> list[str]:
    """Stands in for the TensorRT package; returns the modules imported through it."""
    imported: list[str] = []
    monkeypatch.setattr(
        session_module, "_tensorrt_package_installed", lambda: installed
    )
    monkeypatch.setattr(
        session_module, "importlib", SimpleNamespace(import_module=imported.append)
    )
    return imported


def test_tensorrt_without_the_tensorrt_package_is_reported_not_run_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
):
    # The provider imports nvinfer_11.dll, which only the TensorRT package installs;
    # without it ORT runs on CUDA.
    imported = tensorrt_package(monkeypatch, installed=False)
    ort_registers(monkeypatch, [CUDA, CPU])
    model = identity_model()
    for _ in range(2):  # not cached: the next run tries again
        with pytest.raises(RuntimeError) as raised:
            get_onnx_session(model, 0, TENSORRT, False)
        message = str(raised.value)
        assert f"({TENSORRT})" in message
        assert f"{CUDA}, {CPU} instead" in message
        assert "Install the TensorRT package" in message
    assert imported == []


def test_tensorrt_that_cannot_start_with_the_package_is_reported(
    monkeypatch: pytest.MonkeyPatch,
):
    tensorrt_package(monkeypatch, installed=True)
    ort_registers(monkeypatch, [CUDA, CPU])
    with pytest.raises(RuntimeError) as raised:
        get_onnx_session(identity_model(), 0, TENSORRT, False)
    message = str(raised.value)
    assert f"{CUDA}, {CPU} instead" in message
    assert "Choose another execution provider" in message
    assert "Install the TensorRT package" not in message


def test_tensorrt_is_loaded_from_the_package_before_the_session(
    monkeypatch: pytest.MonkeyPatch,
):
    events = tensorrt_package(monkeypatch, installed=True)

    def session(*args: object, **kwargs: object) -> RegisteredSession:
        events.append("session")
        return RegisteredSession([TENSORRT, CUDA, CPU])

    monkeypatch.setattr(ort, "InferenceSession", session)
    get_onnx_session(identity_model(), 0, TENSORRT, False)
    assert events == ["tensorrt_libs", "session"]


def test_cuda_does_not_load_tensorrt(monkeypatch: pytest.MonkeyPatch):
    imported = tensorrt_package(monkeypatch, installed=True)
    ort_registers(monkeypatch, [CUDA, CPU])
    get_onnx_session(identity_model(), 0, CUDA, False)
    assert imported == []


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
    # Runs native code, which GitHub's checks do not build.
    pytest.importorskip("nodes.impl._chainner_graph")
    model = identity_model()
    session = get_onnx_session(model, 0, CPU, False)
    image = np.ones((1, 3, 4, 4), np.float32)
    np.testing.assert_array_equal(session.run(None, {"input": image})[0], image)
