"""Dense CPU image-session adapter; the installed ONNX Runtime remains the engine.

C++ owns sessions, input copies and result buffers. Python retains ndarray owners
for each synchronous call, then receives independent arrays. Other provider or
graph-value kinds remain explicitly handled by the original runtime binding.
"""

from __future__ import annotations

import ctypes as ct
import importlib
import json
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from google.protobuf.message import DecodeError

from .native import lib

ort_errors = importlib.import_module("onnxruntime.capi.onnxruntime_pybind11_state")

_TYPES = {
    np.dtype(name): code
    for name, code in (
        ("float32", 1),
        ("uint8", 2),
        ("int8", 3),
        ("uint16", 4),
        ("int16", 5),
        ("int32", 6),
        ("int64", 7),
        ("bool", 9),
        ("float16", 10),
        ("float64", 11),
        ("uint32", 12),
        ("uint64", 13),
    )
}
_DTYPES = {v: k for k, v in _TYPES.items()}
_ERRORS = {
    1: ("Fail", "FAIL"),
    2: ("InvalidArgument", "INVALID_ARGUMENT"),
    3: ("NoSuchFile", "NO_SUCHFILE"),
    4: ("NoModel", "NO_MODEL"),
    5: ("EngineError", "ENGINE_ERROR"),
    6: ("RuntimeException", "RUNTIME_EXCEPTION"),
    7: ("InvalidProtobuf", "INVALID_PROTOBUF"),
    8: ("ModelLoaded", "MODEL_LOADED"),
    9: ("NotImplemented", "NOT_IMPLEMENTED"),
    10: ("InvalidGraph", "INVALID_GRAPH"),
    11: ("EPFail", "EP_FAIL"),
}


class _Input(ct.Structure):
    _fields_ = [
        ("name", ct.c_char_p),
        ("data", ct.c_void_p),
        ("bytes", ct.c_size_t),
        ("shape", ct.POINTER(ct.c_int64)),
        ("rank", ct.c_size_t),
        ("type", ct.c_int),
    ]


@lru_cache(maxsize=1)
def _api():
    dll = lib()
    error = [ct.c_char_p, ct.c_size_t]
    signatures = {
        "cn_ort_create": [
            ct.c_wchar_p,
            ct.c_void_p,
            ct.c_size_t,
            ct.POINTER(ct.c_uint64),
        ],
        "cn_ort_metadata": [
            ct.c_uint64,
            ct.c_void_p,
            ct.c_size_t,
            ct.POINTER(ct.c_size_t),
        ],
        "cn_ort_run": [
            ct.c_uint64,
            ct.POINTER(_Input),
            ct.c_size_t,
            ct.POINTER(ct.c_char_p),
            ct.c_size_t,
            ct.POINTER(ct.c_uint64),
        ],
        "cn_ort_result_info": [
            ct.c_uint64,
            ct.c_size_t,
            ct.POINTER(ct.c_int),
            ct.POINTER(ct.c_size_t),
            ct.POINTER(ct.c_int64),
            ct.c_size_t,
            ct.POINTER(ct.c_size_t),
        ],
        "cn_ort_result_copy": [ct.c_uint64, ct.c_size_t, ct.c_void_p, ct.c_size_t],
        "cn_ort_release_session": [ct.c_uint64],
        "cn_ort_release_result": [ct.c_uint64],
    }
    for name, arguments in signatures.items():
        function = getattr(dll, name)
        function.argtypes = [*arguments, *error]
        function.restype = ct.c_int
    dll.cn_ort_live_handles.argtypes = [
        ct.POINTER(ct.c_size_t),
        ct.POINTER(ct.c_size_t),
    ]
    dll.cn_ort_live_handles.restype = ct.c_int
    return dll


def _call(function: Callable[..., int], *args: Any):
    error = ct.create_string_buffer(16384)
    status = function(*args, error, len(error))
    if status:
        name, label = _ERRORS.get(status, ("Fail", "FAIL"))
        exception = getattr(ort_errors, name, RuntimeError)
        raise exception(
            f"[ONNXRuntimeError] : {status} : {label} : {error.value.decode('utf-8', errors='replace')}"
        )


def supports_model(model: bytes) -> bool:
    """Inspect value kinds without constructing a second inference session."""
    try:
        graph = onnx.load_model_from_string(model).graph
    except DecodeError:
        # Let ORT produce its original InvalidProtobuf error from these bytes.
        return True
    for value in (*graph.input, *graph.output):
        if (
            not value.type.HasField("tensor_type")
            or value.type.tensor_type.elem_type not in _DTYPES
        ):
            return False
    return True


@dataclass
class NodeArg:
    name: str
    type: str
    shape: list[int | str | None]


def _finalize_session(function: Callable[..., int], handle: int):
    # Finalizers cannot raise during interpreter teardown. Explicit close uses
    # the same release with normal error propagation and detaches this callback.
    error = ct.create_string_buffer(512)
    function(handle, error, len(error))


class NativeSession:
    """The get_inputs/get_outputs/run interface consumed by chaiNNer's images."""

    def __init__(self, model: bytes, *, library_path: Path | None = None):
        dll = _api()
        path = library_path or Path(ort.__file__).parent / "capi/onnxruntime.dll"
        handle = ct.c_uint64()
        owner = ct.create_string_buffer(model)
        _call(
            dll.cn_ort_create, str(path.resolve()), owner, len(model), ct.byref(handle)
        )
        self._handle = handle.value
        self._finalizer = weakref.finalize(
            self, _finalize_session, dll.cn_ort_release_session, self._handle
        )
        try:
            size = ct.c_size_t()
            _call(dll.cn_ort_metadata, self._handle, None, 0, ct.byref(size))
            text = ct.create_string_buffer(size.value)
            _call(dll.cn_ort_metadata, self._handle, text, len(text), ct.byref(size))
            metadata = json.loads(text.value)
            self._inputs = [NodeArg(**item) for item in metadata["inputs"]]
            self._outputs = [NodeArg(**item) for item in metadata["outputs"]]
            self.runtime_version: str = metadata["runtime_version"]
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._finalizer.alive:
            # Claim the finalizer once before calling native release, so two
            # simultaneous closes never release the same registry handle twice.
            claimed = self._finalizer.detach()
            if claimed is not None:
                _call(_api().cn_ort_release_session, self._handle)

    def get_inputs(self) -> list[NodeArg]:
        return self._inputs

    def get_outputs(self) -> list[NodeArg]:
        return self._outputs

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def get_provider_options(self) -> dict[str, dict[str, str]]:
        return {"CPUExecutionProvider": {}}

    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, Any],
        run_options: Any = None,
    ) -> list[np.ndarray]:
        if run_options is not None:
            raise NotImplementedError(
                "Native image sessions support default run options"
            )
        missing = [item.name for item in self._inputs if item.name not in input_feed]
        if missing:
            raise ValueError(
                f"Required inputs ({missing}) are missing from input feed ({list(input_feed.keys())})."
            )
        names = output_names or [item.name for item in self._outputs]
        known_outputs = {item.name for item in self._outputs}
        for name in names:
            if name not in known_outputs:
                # The Python ORT binding checks this before its C API call and
                # formats the colon differently from the engine's own error.
                raise ort_errors.InvalidArgument(
                    f"[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Invalid output name:{name}"
                )
        output_array = (ct.c_char_p * len(names))(
            *(name.encode("utf-8") for name in names)
        )
        owners = []
        descriptors = []
        for name, image in input_feed.items():
            if not isinstance(image, np.ndarray):
                raise TypeError("Native image session inputs must be NumPy arrays")
            type_code = _TYPES.get(image.dtype.newbyteorder("="))
            if type_code is None:
                raise TypeError(
                    f"Native image session does not support input dtype {image.dtype}"
                )
            source = np.require(image, requirements=["C", "A"])
            shape = (ct.c_int64 * source.ndim)(*source.shape)
            owners.append(source)
            descriptors.append(
                _Input(
                    name.encode("utf-8"),
                    source.ctypes.data,
                    source.nbytes,
                    shape,
                    source.ndim,
                    type_code,
                )
            )
        inputs = (_Input * len(descriptors))(*descriptors)
        dll = _api()
        result = ct.c_uint64()
        _call(
            dll.cn_ort_run,
            self._handle,
            inputs,
            len(inputs),
            output_array,
            len(output_array),
            ct.byref(result),
        )
        try:
            arrays = []
            for index in range(len(names)):
                dtype, rank, byte_count = ct.c_int(), ct.c_size_t(), ct.c_size_t()
                _call(
                    dll.cn_ort_result_info,
                    result,
                    index,
                    ct.byref(dtype),
                    ct.byref(rank),
                    None,
                    0,
                    ct.byref(byte_count),
                )
                shape = (ct.c_int64 * rank.value)()
                _call(
                    dll.cn_ort_result_info,
                    result,
                    index,
                    ct.byref(dtype),
                    ct.byref(rank),
                    shape,
                    rank.value,
                    ct.byref(byte_count),
                )
                array = np.empty(tuple(shape), dtype=_DTYPES[dtype.value])
                _call(
                    dll.cn_ort_result_copy,
                    result,
                    index,
                    array.ctypes.data,
                    array.nbytes,
                )
                arrays.append(array)
            return arrays
        finally:
            _call(dll.cn_ort_release_result, result)
