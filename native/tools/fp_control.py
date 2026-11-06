"""The calling thread's rounding and denormal controls, in the words of the CRT's
_controlfp_s on every system: Windows through that call, Linux through fegetenv and
fesetenv, whose environment holds the whole MXCSR (glibc's fenv_t, __mxcsr at byte 28).
"""

from __future__ import annotations

import ctypes
import sys

ROUNDING_MASK = 0x300  # _MCW_RC: nearest 0, down 0x100, up 0x200, toward zero 0x300
DENORMAL_MASK = 0x03000000  # _MCW_DN: keep 0, flush both, operands, results

MXCSR_AT = 28
MXCSR_ROUNDING = 0x6000
MXCSR_DAZ, MXCSR_FTZ = 0x0040, 0x8000
# _MCW_DN's values and the MXCSR bits they stand for
DENORMALS = {
    0: 0,
    0x01000000: MXCSR_DAZ | MXCSR_FTZ,
    0x02000000: MXCSR_DAZ,
    0x03000000: MXCSR_FTZ,
}


def control(value: int, mask: int) -> int:
    """Sets the bits of mask to value's; the word from before, as _controlfp_s gives it
    (mask 0 only reads)."""
    if sys.platform == "win32":
        call = ctypes.CDLL("ucrtbase.dll")["_controlfp_s"]
        call.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.c_uint, ctypes.c_uint]
        call.restype = ctypes.c_int
        current = ctypes.c_uint()
        if call(ctypes.byref(current), 0, 0) or call(None, value, mask):
            raise OSError("_controlfp_s refused the word")
        return current.value
    libm = ctypes.CDLL("libm.so.6")
    environment = (ctypes.c_ubyte * 64)()
    if libm.fegetenv(environment):
        raise OSError("fegetenv failed")
    mxcsr = int.from_bytes(bytes(environment[MXCSR_AT : MXCSR_AT + 4]), "little")
    denormal = next(
        word
        for word, bits in DENORMALS.items()
        if bits == mxcsr & (MXCSR_DAZ | MXCSR_FTZ)
    )
    before = (mxcsr & MXCSR_ROUNDING) >> 5 | denormal
    word = (before & ~mask) | (value & mask)
    mxcsr &= ~(MXCSR_ROUNDING | MXCSR_DAZ | MXCSR_FTZ)
    mxcsr |= (word & ROUNDING_MASK) << 5 | DENORMALS[word & DENORMAL_MASK]
    environment[MXCSR_AT : MXCSR_AT + 4] = list(mxcsr.to_bytes(4, "little"))
    if libm.fesetenv(environment):
        raise OSError("fesetenv failed")
    return before
