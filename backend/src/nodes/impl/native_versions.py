"""The library versions the native paths are validated against (the tested stack).

A gate compares the installed version exactly. Any other version keeps the library's
own path, logged once per gate. ``native/python-stack.lock.txt`` records the full set.
"""

from __future__ import annotations

import cv2
from sanic.log import logger

CV2 = "5.0.0"
NUMPY = "2.5.3"
ORT = "1.30.0"

# OpenCV 5.0.0's channel ceiling (core/hal/interface.h CV_CN_MAX); the Python
# module does not export it.
CV_CN_MAX = 128

_fallbacks_logged: set[str] = set()


def cv2_is_tested(gate: str) -> bool:
    """Whether the installed OpenCV is CV2; otherwise log the gate's first fallback."""
    if cv2.__version__ == CV2:
        return True
    if gate not in _fallbacks_logged:
        _fallbacks_logged.add(gate)
        logger.warning(
            "%s: OpenCV %s is not the tested %s; using OpenCV's own path",
            gate,
            cv2.__version__,
            CV2,
        )
    return False
