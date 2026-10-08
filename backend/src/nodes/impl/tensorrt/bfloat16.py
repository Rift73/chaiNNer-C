"""bfloat16 <-> float32 for TensorRT engines with BF16 input or output tensors.

NumPy has no bfloat16 type, so a bf16 tensor is held as its uint16 bits. Both
directions give exactly what ml_dtypes gives. This module does not import
TensorRT, so it can be tested without a GPU.
"""

from __future__ import annotations

import numpy as np


def float32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round to nearest, ties to even. A NaN becomes the quiet NaN 0x7FC0 with its
    sign: rounding its payload could carry it into Inf or wrap it to zero."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    out = (rounded >> np.uint32(16)).astype(np.uint16)
    nan = (bits & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    out[nan] = np.where(bits[nan] >> np.uint32(31), 0xFFC0, 0x7FC0)
    return out


def bf16_bits_to_float32(x: np.ndarray) -> np.ndarray:
    """Exact: a bf16 value is the top half of a float32."""
    return (x.astype(np.uint32) << np.uint32(16)).view(np.float32)
