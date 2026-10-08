from __future__ import annotations

import warnings
from typing import Any, Dict, Protocol, Sequence, Tuple, Union
from weakref import WeakKeyDictionary

import onnxruntime as ort

from ..native_onnx_runtime import NativeSession, supports_model
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


def create_inference_session(
    model: OnnxModel,
    gpu_index: int,
    execution_provider: str,
    should_tensorrt_fp16: bool = False,
    tensorrt_cache_path: str | None = None,
) -> NativeSession | ort.InferenceSession:
    if execution_provider == "CPUExecutionProvider" and supports_model(model.bytes):
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
    tensorrt: ProviderDesc = (
        "TensorrtExecutionProvider",
        {
            "device_id": gpu_index,
            "trt_engine_cache_enable": tensorrt_cache_path is not None,
            "trt_engine_cache_path": tensorrt_cache_path,
            "trt_fp16_enable": should_tensorrt_fp16,
            "trt_dump_subgraphs": tensorrt_cache_path is not None,
            "trt_timing_cache_enable": tensorrt_cache_path is not None,
            "trt_timing_cache_path": tensorrt_cache_path,
        },
    )
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

    if execution_provider == "TensorrtExecutionProvider":
        providers = [tensorrt, cuda, cpu]
    elif execution_provider == "CUDAExecutionProvider":
        providers = [cuda, cpu]
    elif execution_provider == "DmlExecutionProvider":
        providers = [dml, cpu]
    else:
        providers = [execution_provider, cpu]

    session = ort.InferenceSession(
        model.bytes,
        providers=providers,
    )
    return session


__session_cache: WeakKeyDictionary[OnnxModel, OnnxSession] = WeakKeyDictionary()


def get_onnx_session(
    model: OnnxModel,
    gpu_index: int,
    execution_provider: str,
    should_tensorrt_fp16: bool,
    tensorrt_cache_path: str | None = None,
) -> OnnxSession:
    cached = __session_cache.get(model)
    if cached is None:
        session = create_inference_session(
            model,
            gpu_index,
            execution_provider,
            should_tensorrt_fp16,
            tensorrt_cache_path,
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
            if execution_provider == "TensorrtExecutionProvider":
                # onnxruntime_providers_tensorrt.dll of onnxruntime-gpu 1.30.0
                # imports nvinfer_10.dll; the TensorRT package ships nvinfer_11.dll.
                message += (
                    " Its TensorRT provider needs TensorRT 10 (nvinfer_10.dll) on the"
                    " PATH; chaiNNer-C's TensorRT package installs TensorRT 11, which"
                    " only the TensorRT nodes use. Choose CUDA in the ONNX settings,"
                    " or use the TensorRT nodes."
                )
            else:
                message += " Choose another execution provider in the ONNX settings."
            raise RuntimeError(message)
        cached = session
        __session_cache[model] = cached
    return cached


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
