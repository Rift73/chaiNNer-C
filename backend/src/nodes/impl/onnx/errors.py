from __future__ import annotations

import sys


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
