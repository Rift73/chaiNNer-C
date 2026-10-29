"""An ONNX session runs on the execution provider chosen in the ONNX settings, or
fails with the reason. ONNX Runtime itself falls back to the next provider when one
cannot start (TensorRT onto CUDA, CUDA onto the CPU) and only logs why. Runs without
a GPU: the GPU sessions are stand-ins that report the providers ORT registered."""

from __future__ import annotations

import importlib.util
import os
import threading
import time
import weakref
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from nodes.impl.onnx import session as session_module
from nodes.impl.onnx.load import ModelFacts
from nodes.impl.onnx.model import OnnxGeneric, OnnxInfo
from nodes.impl.onnx.session import ProviderDesc, get_onnx_session

TENSORRT = "TensorrtExecutionProvider"
CUDA = "CUDAExecutionProvider"
DML = "DmlExecutionProvider"
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


def conv_model(seed: int) -> OnnxGeneric:
    """A 3x3 convolution with dynamic height and width, its weights from `seed`."""
    rng = np.random.default_rng(seed)
    weight = numpy_helper.from_array(
        rng.standard_normal((3, 3, 3, 3)).astype(np.float32), "weight"
    )
    image = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "h", "w"])
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3, "h", "w"]
    )
    node = helper.make_node("Conv", ["input", "weight"], ["output"], pads=[1, 1, 1, 1])
    graph = helper.make_graph([node], "conv", [image], [output], [weight])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    )
    return OnnxGeneric(model.SerializeToString(), OnnxInfo(13, "fp32"))


@pytest.fixture(autouse=True)
def cuda_loads(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """The CUDA and cuDNN loads before a CUDA or TensorRT session (_load_cuda), made
    once per test and recorded instead of made, as where PyTorch is installed."""
    loads: list[dict[str, object]] = []
    monkeypatch.setattr(ort, "preload_dlls", lambda **kwargs: loads.append(kwargs))
    monkeypatch.setattr(session_module, "_pytorch_installed", lambda: True)
    load_cuda = vars(session_module)["_load_cuda"]
    monkeypatch.setattr(session_module, "_load_cuda", cache(load_cuda.__wrapped__))
    return loads


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
        session_module,
        "importlib",
        SimpleNamespace(import_module=imported.append, util=importlib.util),
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


@pytest.mark.parametrize("provider", [CUDA, TENSORRT])
def test_a_gpu_provider_that_cannot_start_without_pytorch_is_reported(
    monkeypatch: pytest.MonkeyPatch, provider: str
):
    # The PyTorch package installs chaiNNer's only CUDA and cuDNN (on Linux, its NVIDIA
    # wheels).
    monkeypatch.setattr(session_module, "_pytorch_installed", lambda: False)
    tensorrt_package(monkeypatch, installed=True)
    ort_registers(monkeypatch, [CPU])
    with pytest.raises(
        RuntimeError, match=rf"\({provider}\).* on {CPU} instead"
    ) as raised:
        get_onnx_session(identity_model(), 0, provider, False)
    assert "Install the PyTorch package" in str(raised.value)


@pytest.mark.parametrize(
    ("platform", "torch", "loads"),
    [
        ("win32", "installed", [{"msvc": False}]),
        ("win32", "imported", []),
        ("win32", "absent", []),
        ("linux", "installed", [{"msvc": False, "directory": ""}]),
    ],
)
def test_cuda_and_cudnn_load_once_before_the_first_gpu_session(
    monkeypatch: pytest.MonkeyPatch,
    cuda_loads: list[dict[str, object]],
    platform: str,
    torch: str,
    loads: list[dict[str, object]],
):
    # The CUDA and TensorRT providers import CUDA 13 and cuDNN 9 by name. Where torch
    # was not imported first (a VapourSynth script's process), Windows took them from
    # PATH, and another CUDA's cuDNN there crashed the process at the first run:
    # PyTorch's load by full path (ORT's default folder on Windows), unless torch has
    # loaded them. Without PyTorch, chaiNNer has none to load.
    modules = {"torch": SimpleNamespace()} if torch == "imported" else {}
    monkeypatch.setattr(
        session_module, "sys", SimpleNamespace(platform=platform, modules=modules)
    )
    monkeypatch.setattr(session_module, "_pytorch_installed", lambda: torch != "absent")
    ort_creates(monkeypatch)
    # Each session's provider, with the loads made before it was created
    sessions: list[tuple[str, int]] = []

    def session(_model: bytes, providers: list[ProviderDesc]) -> RegisteredSession:
        names = [p if isinstance(p, str) else p[0] for p in providers]
        sessions.append((names[0], len(cuda_loads)))
        return RegisteredSession(names)

    monkeypatch.setattr(ort, "InferenceSession", session)
    for provider in (CPU, DML, CUDA, TENSORRT, CUDA):
        get_onnx_session(identity_model(), 0, provider, False)
    made = len(loads)
    assert sessions == [
        (CPU, 0),
        (DML, 0),
        (CUDA, made),
        (TENSORRT, made),
        (CUDA, made),
    ]
    assert cuda_loads == loads


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


def ort_creates(monkeypatch: pytest.MonkeyPatch) -> list[list[ProviderDesc]]:
    """Stands in for ORT sessions that start every provider, on the CPU as well (no
    native session); returns the provider lists the sessions were created with."""
    created: list[list[ProviderDesc]] = []

    def session(_model: bytes, providers: list[ProviderDesc]) -> RegisteredSession:
        created.append(providers)
        return RegisteredSession([p if isinstance(p, str) else p[0] for p in providers])

    monkeypatch.setattr(ort, "InferenceSession", session)
    monkeypatch.setattr(
        session_module,
        "model_facts",
        lambda _model: ModelFacts(native_session=False, dual_manifest=None),
    )
    tensorrt_package(monkeypatch, installed=True)
    return created


# (GPU index, execution provider, TensorRT FP16, TensorRT cache path)
Settings = tuple[int, str, bool, str | None]


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ((0, TENSORRT, False, None), (1, TENSORRT, False, None)),
        ((0, TENSORRT, False, None), (0, TENSORRT, True, None)),
        ((0, TENSORRT, False, None), (0, TENSORRT, False, "cache")),
        ((0, TENSORRT, False, None), (0, CUDA, False, None)),
        ((0, CUDA, False, None), (1, CUDA, False, None)),
        ((0, DML, False, None), (1, DML, False, None)),
        ((0, CUDA, False, None), (0, CPU, False, None)),
    ],
    ids=[
        "tensorrt-gpu",
        "tensorrt-fp16",
        "tensorrt-cache",
        "provider",
        "cuda-gpu",
        "dml-gpu",
        "cpu",
    ],
)
def test_a_changed_setting_gets_a_session_created_from_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, before: Settings, after: Settings
):
    # The relative TensorRT cache path's engine folder is made in the working folder
    monkeypatch.chdir(tmp_path)
    created = ort_creates(monkeypatch)
    model = identity_model()
    first = get_onnx_session(model, *before)
    assert get_onnx_session(model, *before) is first
    second = get_onnx_session(model, *after)
    assert second is not first
    assert len(created) == 2 and created[1] != created[0]
    assert get_onnx_session(model, *after) is second
    assert len(created) == 2


def test_the_tensorrt_fp16_setting_reaches_the_new_session(
    monkeypatch: pytest.MonkeyPatch,
):
    created = ort_creates(monkeypatch)
    model = identity_model()
    get_onnx_session(model, 0, TENSORRT, False)
    get_onnx_session(model, 0, TENSORRT, True)
    options = [p[1] for p in (created[0][0], created[1][0]) if not isinstance(p, str)]
    assert [o["trt_fp16_enable"] for o in options] == [False, True]


def test_tensorrt_engines_are_cached_per_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # ORT names an engine by the graph's tensor names and the model's file name, which
    # a model created from bytes has not: two models of one architecture with other
    # weights need their own folders, or the second runs the first one's engine. The
    # timing cache (tactic timings by layer) stays shared.
    created = ort_creates(monkeypatch)
    cache = str(tmp_path)

    def options(model: OnnxGeneric) -> dict[str, object]:
        get_onnx_session(model, 0, TENSORRT, False, cache)
        tensorrt = created[-1][0]
        assert isinstance(tensorrt, tuple) and tensorrt[0] == TENSORRT
        return tensorrt[1]

    first = options(conv_model(1))
    folder = str(first["trt_engine_cache_path"])
    assert os.path.isdir(folder) and Path(folder).parent == tmp_path
    # The same bytes in a model loaded again
    assert options(conv_model(1))["trt_engine_cache_path"] == folder
    other = options(conv_model(2))
    assert other["trt_engine_cache_path"] != folder
    assert os.path.isdir(str(other["trt_engine_cache_path"]))
    assert first["trt_timing_cache_path"] == other["trt_timing_cache_path"] == cache


@pytest.mark.parametrize(
    "after",
    [(1, CPU, False, None), (0, CPU, True, None), (0, CPU, False, "cache")],
    ids=["gpu", "tensorrt-fp16", "tensorrt-cache"],
)
def test_settings_a_cpu_session_does_not_use_keep_it(
    monkeypatch: pytest.MonkeyPatch, after: Settings
):
    created = ort_creates(monkeypatch)
    model = identity_model()
    first = get_onnx_session(model, 0, CPU, False)
    assert get_onnx_session(model, *after) is first
    assert len(created) == 1


def test_the_replaced_session_is_freed_before_its_successor_is_created(
    monkeypatch: pytest.MonkeyPatch,
):
    sessions: list[weakref.ref[RegisteredSession]] = []
    alive: list[bool] = []

    def session(_model: bytes, providers: list[ProviderDesc]) -> RegisteredSession:
        alive.append(any(s() is not None for s in sessions))
        created = RegisteredSession([CUDA, CPU])
        sessions.append(weakref.ref(created))
        return created

    monkeypatch.setattr(ort, "InferenceSession", session)
    model = identity_model()
    get_onnx_session(model, 0, CUDA, False)
    get_onnx_session(model, 1, CUDA, False)
    assert alive == [False, False]
    assert sessions[0]() is None and sessions[1]() is not None


def test_concurrent_first_uses_of_a_model_share_one_session(
    monkeypatch: pytest.MonkeyPatch,
):
    # Two upscales of one model that start together (a preview beside a run, say)
    # wait for one session instead of each creating one; another model's session
    # is created meanwhile.
    slow, other = conv_model(1), conv_model(2)
    created: list[bytes] = []
    creating, release = threading.Event(), threading.Event()
    entered = threading.Semaphore(0)

    def session(model: bytes, providers: list[ProviderDesc]) -> RegisteredSession:
        created.append(model)
        if model == slow.bytes:
            creating.set()
            assert release.wait(30)
        return RegisteredSession([CUDA, CPU])

    settings = vars(session_module)["_session_settings"]

    def entering(*args: Any) -> tuple[object, ...]:
        entered.release()
        return settings(*args)

    monkeypatch.setattr(ort, "InferenceSession", session)
    monkeypatch.setattr(session_module, "_session_settings", entering)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(get_onnx_session, slow, 0, CUDA, False)
        assert creating.wait(30) and entered.acquire(timeout=30)
        second = pool.submit(get_onnx_session, slow, 0, CUDA, False)
        assert entered.acquire(timeout=30)
        get_onnx_session(other, 0, CUDA, False)
        time.sleep(0.1)
        assert not second.done()
        release.set()
        assert first.result(30) is second.result(30)
    assert created == [slow.bytes, other.bytes]


def test_two_loads_of_one_file_share_a_session_that_builds_into_a_cache_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Their engines go to one folder, named by the file's content: two sessions would
    # build into it at once. The second load's first use waits for the first's session.
    created = ort_creates(monkeypatch)
    creating, release = threading.Event(), threading.Event()
    make = ort.InferenceSession

    def session(model: bytes, providers: list[ProviderDesc]) -> object:
        creating.set()
        assert release.wait(30)
        return make(model, providers=providers)

    monkeypatch.setattr(ort, "InferenceSession", session)
    first, again = conv_model(1), conv_model(1)
    cache = str(tmp_path)
    with ThreadPoolExecutor(2) as pool:
        one = pool.submit(get_onnx_session, first, 0, TENSORRT, False, cache)
        assert creating.wait(30)
        other = pool.submit(get_onnx_session, again, 0, TENSORRT, False, cache)
        time.sleep(0.1)
        assert not other.done()
        release.set()
        assert one.result(30) is other.result(30)
    assert len(created) == 1
    # Without a cache folder, each model keeps a session of its own
    assert get_onnx_session(first, 0, CUDA, False) is not get_onnx_session(
        again, 0, CUDA, False
    )
