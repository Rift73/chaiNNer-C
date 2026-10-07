"""C CPU wavelet kernels with the installed oneDNN fused arithmetic order.

Device tensors, autograd, CPU autocast and requested reduced convolution
precision retain their framework semantics. Non-oneDNN helper calls preserve
their alternate BLAS reduction outside the verified normalized-image domain.
"""

from __future__ import annotations

import ctypes as ct
from functools import lru_cache

import numpy as np
import torch

from .native import check, lib, ptr


@lru_cache(maxsize=1)
def _api():
    library = lib()
    p, n = ct.POINTER(ct.c_float), ct.c_size_t
    library.cn_wavelet_blur_f32.argtypes = [p, p, n, n, n, n, ct.c_int]
    library.cn_wavelet_decompose_f32.argtypes = [p, p, p, n, n, n, ct.c_int, ct.c_int]
    library.cn_wavelet_reconstruct_f32.argtypes = [p, p, p, n, n, n, ct.c_int, ct.c_int]
    for name in ("blur", "decompose", "reconstruct"):
        getattr(library, "cn_wavelet_" + name + "_f32").restype = ct.c_int
    return library


def _image(image: torch.Tensor, levels: int) -> np.ndarray | None:
    if (
        type(image) is not torch.Tensor
        or image.device.type != "cpu"
        or image.layout != torch.strided
        or image.dtype != torch.float32
        or image.ndim != 4
        or image.shape[1] != 3
        or not all(image.shape)
        or image.requires_grad
        or image.is_conj()
        or image.is_neg()
        or torch.is_autocast_cpu_enabled()
    ):
        return None
    for backend in (
        torch.backends,
        torch.backends.mkldnn,
        getattr(torch.backends.mkldnn, "conv", None),
    ):
        if getattr(backend, "fp32_precision", "none") not in ("none", "ieee"):
            return None
    array = image.numpy()
    if not _fused():
        if not (0 <= np.min(array) <= np.max(array) <= 1):
            return None
        lower = float(np.finfo(np.float32).tiny) * 2.0 ** (4 * levels + 4)
        if np.logical_and(array > 0, array < lower).any():
            return None
    return np.require(array, requirements=["C", "A"])


def _fused() -> bool:
    return torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available()


def blur(image: torch.Tensor, radius: object) -> torch.Tensor | None:
    if not isinstance(radius, int) or not 1 <= radius <= 512:
        return None
    source = _image(image, 1)
    if source is None:
        return None
    output = np.empty(source.shape, np.float32)
    check(
        _api().cn_wavelet_blur_f32(
            ptr(source),
            ptr(output),
            source.shape[0],
            *source.shape[2:],
            radius,
            _fused(),
        )
    )
    return torch.from_numpy(output)


def decomposition(
    image: torch.Tensor, levels: object
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not isinstance(levels, int) or not 1 <= levels <= 10:
        return None
    source = _image(image, levels)
    if source is None:
        return None
    high, low = (np.empty(source.shape, np.float32) for _ in range(2))
    check(
        _api().cn_wavelet_decompose_f32(
            ptr(source),
            ptr(high),
            ptr(low),
            source.shape[0],
            *source.shape[2:],
            levels,
            _fused(),
        )
    )
    return torch.from_numpy(high), torch.from_numpy(low)


def reconstruction(
    content: torch.Tensor, style: torch.Tensor, levels: object
) -> torch.Tensor | None:
    if not isinstance(levels, int) or not 1 <= levels <= 10:
        return None
    a, b = _image(content, levels), _image(style, levels)
    if a is None or b is None or a.shape != b.shape:
        return None
    output = np.empty(a.shape, np.float32)
    check(
        _api().cn_wavelet_reconstruct_f32(
            ptr(a), ptr(b), ptr(output), a.shape[0], *a.shape[2:], levels, _fused()
        )
    )
    return torch.from_numpy(output)
