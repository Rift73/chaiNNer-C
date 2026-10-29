from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import sys
import warnings
from functools import cache
from typing import Any, Dict, Protocol, Sequence, Tuple, Union
from weakref import WeakKeyDictionary

import onnxruntime as ort

from ..native_onnx_runtime import NativeSession
from .load import model_facts
from .model import OnnxModel
from .utils import OnnxParsedTensorShape, parse_onnx_shape

ProviderDesc = Union[str, Tuple[str, Dict[Any, Any]]]


class OnnxNodeArg(Protocol):
    """An input or output of an ONNX session: ORT's NodeArg or the native one."""

    @property
    def name(self) -> str: ...
    @property
    def type(self) -> str: ...
    @property
    def shape(self) -> Sequence[int | str | None]: ...


class OnnxSession(Protocol):
    """The session surface the backend uses. ort.InferenceSession and the native
    adapter (NativeSession) both provide it."""

    def get_inputs(self) -> Sequence[OnnxNodeArg]: ...
    def get_outputs(self) -> Sequence[OnnxNodeArg]: ...
    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, Any],
        run_options: Any = None,
    ) -> Sequence[object]: ...
    def get_providers(self) -> Sequence[str]: ...


def _tensorrt_package_installed() -> bool:
    return importlib.util.find_spec("tensorrt_libs") is not None


def _load_tensorrt() -> None:
    """ORT's TensorRT provider imports nvinfer_11.dll and nvonnxparser_11.dll by name
    (on Linux libnvinfer.so.11 and libnvonnxparser.so.11). Importing the TensorRT
    package's tensorrt_libs loads its TensorRT 11 by full path, so the provider finds
    those modules without a PATH change, and the TensorRT nodes use the same ones."""
    if _tensorrt_package_installed():
        importlib.import_module("tensorrt_libs")


def _pytorch_installed() -> bool:
    return importlib.util.find_spec("torch") is not None


@cache
def _load_cuda() -> None:
    """ORT's CUDA and TensorRT providers import CUDA 13 and cuDNN 9 by name, which
    resolve to the modules of those names already loaded (`import torch` loads
    PyTorch's). Else Linux finds no cuDNN, and Windows searches PATH, where another
    CUDA's cuDNN crashes the process at its first run. So they load once by full path:
    on Linux from the NVIDIA wheels; on Windows from PyTorch's folder (ORT's default
    there; its wheel carries chaiNNer's only CUDA), where torch is installed but not
    imported, as in a VapourSynth script's process."""
    if sys.platform == "linux":
        ort.preload_dlls(msvc=False, directory="")
    elif "torch" not in sys.modules and _pytorch_installed():
        ort.preload_dlls(msvc=False)


@cache
def _load_migraphx() -> None:
    """On Windows, ORT's MIGraphX provider imports HIP by name and MIGraphX loads its
    GPU target by name, which only the DLLs loaded first resolve to the right files
    (the AMD driver puts another HIP in System32): MIGraphX's DLLs and ROCm's load as the
    MIGraphX package loads them, once. Linux's find theirs through their run paths."""
    from ..migraphx.runner import load_migraphx_libraries

    load_migraphx_libraries()


def _model_cache_dir(model: OnnxModel, cache_path: str, variant: str = "") -> str:
    """The folder of `model`'s compiled programs or engines in the cache at
    `cache_path`, named by the SHA-256 of the model and the `variant` of its
    compilation. ONNX Runtime's MIGraphX and TensorRT providers name what they compile
    by the graph's tensor names, not its weights (and a model created from bytes has
    no file name): two models of one architecture with other weights would load each
    other's from one folder."""
    digest = hashlib.sha256(model.bytes).hexdigest()[:32]
    path = os.path.join(cache_path, digest + variant)
    os.makedirs(path, exist_ok=True)
    return path


def create_inference_session(
    model: OnnxModel,
    gpu_index: int,
    execution_provider: str,
    should_tensorrt_fp16: bool = False,
    tensorrt_cache_path: str | None = None,
    should_migraphx_fp16: bool = False,
    migraphx_cache_path: str | None = None,
    migraphx_exhaustive_tune: bool = False,
) -> NativeSession | ort.InferenceSession:
    if (
        execution_provider == "CPUExecutionProvider"
        and model_facts(model).native_session
    ):
        # The native adapter implements the image-session surface below; model
        # values outside dense numeric tensors and all other providers retain
        # the original framework binding explicitly. Native failures propagate.
        session = NativeSession(model.bytes)
        # Original providers=[CPU, CPU] emits this warning while deduplicating.
        warnings.warn(
            "Duplicate provider 'CPUExecutionProvider' encountered, ignoring.",
            stacklevel=2,
        )
        return session
    cuda: ProviderDesc = (
        "CUDAExecutionProvider",
        {
            "device_id": gpu_index,
        },
    )
    dml: ProviderDesc = (
        "DmlExecutionProvider",
        {
            "device_id": gpu_index,
        },
    )
    cpu: ProviderDesc = "CPUExecutionProvider"

    if sys.platform in ("linux", "win32") and execution_provider in (
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
    ):
        _load_cuda()
    if execution_provider == "TensorrtExecutionProvider":
        _load_tensorrt()
        # The engines go to the model's own folder; the timing cache (tactic timings
        # by layer, which carry no weights) stays one for all models.
        tensorrt: ProviderDesc = (
            "TensorrtExecutionProvider",
            {
                "device_id": gpu_index,
                "trt_engine_cache_enable": tensorrt_cache_path is not None,
                "trt_engine_cache_path": (
                    None
                    if tensorrt_cache_path is None
                    else _model_cache_dir(model, tensorrt_cache_path)
                ),
                "trt_fp16_enable": should_tensorrt_fp16,
                "trt_timing_cache_enable": tensorrt_cache_path is not None,
                "trt_timing_cache_path": tensorrt_cache_path,
            },
        )
        providers = [tensorrt, cuda, cpu]
    elif execution_provider == "CUDAExecutionProvider":
        providers = [cuda, cpu]
    elif execution_provider == "DmlExecutionProvider":
        providers = [dml, cpu]
    elif execution_provider == "MIGraphXExecutionProvider":
        if sys.platform == "win32":
            _load_migraphx()
        # MIGraphX compiles the graph for each input shape it runs at; without a
        # cache folder it keeps no program across sessions.
        migraphx_options: dict[str, object] = {
            "device_id": gpu_index,
            "migraphx_fp16_enable": should_migraphx_fp16,
            "migraphx_exhaustive_tune": migraphx_exhaustive_tune,
        }
        if migraphx_cache_path is not None:
            # MIGraphX's names also leave out the precision and the tuning
            precision = "fp16" if should_migraphx_fp16 else "fp32"
            tuning = "-exhaustive" if migraphx_exhaustive_tune else ""
            migraphx_options["migraphx_model_cache_dir"] = _model_cache_dir(
                model, migraphx_cache_path, f"-{precision}{tuning}"
            )
        providers = [("MIGraphXExecutionProvider", migraphx_options), cpu]
    else:
        providers = [execution_provider, cpu]

    session = ort.InferenceSession(
        model.bytes,
        providers=providers,
    )
    return session


def _session_settings(
    gpu_index: int,
    execution_provider: str,
    should_tensorrt_fp16: bool,
    tensorrt_cache_path: str | None,
    should_migraphx_fp16: bool,
    migraphx_cache_path: str | None,
    migraphx_exhaustive_tune: bool,
) -> tuple[object, ...]:
    """The settings `create_inference_session` builds a session of
    `execution_provider` from."""
    if execution_provider == "TensorrtExecutionProvider":
        return (
            execution_provider,
            gpu_index,
            should_tensorrt_fp16,
            tensorrt_cache_path,
        )
    if execution_provider == "MIGraphXExecutionProvider":
        return (
            execution_provider,
            gpu_index,
            should_migraphx_fp16,
            migraphx_cache_path,
            migraphx_exhaustive_tune,
        )
    if execution_provider in ("CUDAExecutionProvider", "DmlExecutionProvider"):
        return (execution_provider, gpu_index)
    return (execution_provider,)


# One session per model: a session created from other settings is replaced, which
# frees its memory before the new one is created.
__session_cache: WeakKeyDictionary[
    OnnxModel, tuple[tuple[object, ...], OnnxSession]
] = WeakKeyDictionary()


def get_onnx_session(
    model: OnnxModel,
    gpu_index: int,
    execution_provider: str,
    should_tensorrt_fp16: bool,
    tensorrt_cache_path: str | None = None,
    should_migraphx_fp16: bool = False,
    migraphx_cache_path: str | None = None,
    migraphx_exhaustive_tune: bool = False,
) -> OnnxSession:
    key = _session_settings(
        gpu_index,
        execution_provider,
        should_tensorrt_fp16,
        tensorrt_cache_path,
        should_migraphx_fp16,
        migraphx_cache_path,
        migraphx_exhaustive_tune,
    )
    cached = __session_cache.get(model)
    if cached is not None and cached[0] != key:
        del __session_cache[model]
        cached = None
    if cached is None:
        session = create_inference_session(
            model,
            gpu_index,
            execution_provider,
            should_tensorrt_fp16,
            tensorrt_cache_path,
            should_migraphx_fp16,
            migraphx_cache_path,
            migraphx_exhaustive_tune,
        )
        providers = session.get_providers()
        if execution_provider not in providers:
            # ORT falls back to the next provider when one cannot start (TensorRT
            # onto CUDA, CUDA onto the CPU) and only logs why.
            message = (
                "ONNX Runtime could not start the execution provider chosen in the"
                f" ONNX settings ({execution_provider}), so it would run this model on"
                f" {', '.join(providers)} instead. The log has ONNX Runtime's reason."
            )
            if (
                execution_provider == "TensorrtExecutionProvider"
                and not _tensorrt_package_installed()
            ):
                message += (
                    " Its TensorRT provider uses the TensorRT that chaiNNer-C's"
                    " TensorRT package installs. Install the TensorRT package in the"
                    " Dependency Manager, or choose CUDA in the ONNX settings."
                )
            elif (
                execution_provider
                in ("TensorrtExecutionProvider", "CUDAExecutionProvider")
                and not _pytorch_installed()
            ):
                message += (
                    " On NVIDIA GPUs it uses the CUDA and cuDNN that chaiNNer-C's PyTorch"
                    " package installs. Install the PyTorch package in the Dependency"
                    " Manager, or choose another execution provider in the ONNX settings."
                )
            else:
                message += " Choose another execution provider in the ONNX settings."
            raise RuntimeError(message)
        cached = (key, session)
        __session_cache[model] = cached
    return cached[1]


def get_input_shape(session: OnnxSession) -> OnnxParsedTensorShape:
    """
    Returns the input shape, input channels, input width (optional), and input height (optional).
    """

    return parse_onnx_shape(session.get_inputs()[0].shape)


def get_output_shape(session: OnnxSession) -> OnnxParsedTensorShape:
    """
    Returns the output shape, output channels, output width (optional), and output height (optional).
    """

    return parse_onnx_shape(session.get_outputs()[0].shape)
