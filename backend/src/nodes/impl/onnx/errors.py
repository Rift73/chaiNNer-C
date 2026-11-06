from __future__ import annotations

import sys

# What lack of memory reads as: the arena's words, the GPU's, native/src/onnx_runtime.cpp's
# for a std::bad_alloc in the adapter, and a kernel's std::bad_alloc on Linux and Windows.
_OUT_OF_MEMORY_TEXTS = (
    "allocate memory",
    "out of memory",
    "cudaMalloc",
    "allocation failed",
    "bad_alloc",
    "bad allocation",
)


def ort_error_message(error: Exception) -> str:
    """The message of an error that an ONNX Runtime session raised.

    ORT's binding decodes an error message as UTF-8, but Windows formats system errors
    (such as a DirectML device error) in the ANSI code page. Such a message arrives as
    a UnicodeDecodeError that hides it; the error keeps the message's bytes, which
    decode in that code page.
    """
    if isinstance(error, UnicodeDecodeError):
        encoding = "mbcs" if sys.platform == "win32" else "utf-8"
        return error.object.decode(encoding, errors="replace")
    return str(error)


def is_out_of_memory(error: Exception) -> bool:
    """Whether an ONNX Runtime session's error is that it could not allocate memory
    (on the CPU or the GPU)."""
    message = ort_error_message(error)
    return "ONNXRuntimeError" in message and any(
        text in message for text in _OUT_OF_MEMORY_TEXTS
    )
