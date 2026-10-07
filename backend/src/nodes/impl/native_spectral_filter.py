"""C cross-correlation buffers around the retained OpenCV 4.8 DFT primitive.

The installed double-precision DFT uses Intel IPP. Replacing it with NumPy's
FFT or OpenCV's open-source DFT changes saved pixels at legitimate rounding
ties. Only public cv2.dft remains external; C stages the kernel and reflected
image tiles, multiplies their packed spectra, and converts/interleaves output.
The tile geometry follows OpenCV 4.8 modules/imgproc/src/templmatch.cpp.
"""

from __future__ import annotations

import ctypes as ct
import operator

import cv2
import numpy as np

from .native import check, f32, lib, ptr
from .native_prepared import PreparedCache, float_state
from .native_versions import CV_CN_MAX

_spectra = PreparedCache[np.ndarray](32 * 1024 * 1024, 32)

_lib = lib()
_fp = ct.POINTER(ct.c_float)
_dp = ct.POINTER(ct.c_double)
_n = ct.c_size_t
_lib.cn_spectral_kernel.argtypes = [ct.c_void_p, ct.c_int, _dp, _n, _n, _n, _n]
_lib.cn_spectral_tile.argtypes = [_fp, _dp] + [_n] * 13 + [ct.c_int]
_lib.cn_spectral_multiply.argtypes = [_dp, _dp, _n, _n]
_lib.cn_spectral_store.argtypes = [_dp, _fp] + [_n] * 10
for _name in ("kernel", "tile", "multiply", "store"):
    getattr(_lib, "cn_spectral_" + _name).restype = ct.c_int


def filter2d(
    image: np.ndarray,
    kernel: object,
    padding: int = 0,
    border: int = cv2.BORDER_REFLECT_101,
) -> np.ndarray:
    """Reproduce the float32, centered-anchor, zero-delta DFT filter2D path.

    ``padding`` adds the original Convolve node's zero frame before filtering.
    Callers select this helper only when filter2D would choose crossCorr;
    selecting it for a smaller spatial kernel changes accumulation order.
    """
    source = f32(image)
    if source.ndim not in (2, 3) or not all(source.shape):
        raise ValueError("Spectral filtering requires a nonempty float32 image")
    channels = source.shape[2] if source.ndim == 3 else 1
    if channels > CV_CN_MAX:
        raise cv2.error(f"Spectral filtering supports at most {CV_CN_MAX} channels")
    if not isinstance(kernel, np.ndarray) or kernel.ndim != 2 or not all(kernel.shape):
        raise ValueError("The spectral kernel must be a nonempty two-dimensional array")
    if kernel.dtype not in (np.float32, np.float64):
        raise TypeError("Spectral kernel coefficients must be float32 or float64")
    coefficients = np.require(kernel, requirements=["C", "A"])
    padding, border = operator.index(padding), operator.index(border)
    border &= ~cv2.BORDER_ISOLATED
    if padding < 0 or border not in range(5):
        raise ValueError("Invalid spectral padding or border type")
    h, w = source.shape[:2]
    kh, kw = coefficients.shape
    oh, ow = h + 2 * padding, w + 2 * padding
    if max(oh, ow, kh, kw) > np.iinfo(np.int32).max:
        raise OverflowError("Spectral dimensions exceed OpenCV's integer range")
    bw = min(max(round(kw * 4.5), 256 - kw + 1), ow)
    bh = min(max(round(kh * 4.5), 256 - kh + 1), oh)
    if max(bw + kw - 1, bh + kh - 1) > np.iinfo(np.int32).max:
        raise OverflowError(
            "Spectral transform dimensions exceed OpenCV's integer range"
        )
    dw = cv2.getOptimalDFTSize(bw + kw - 1)
    dh = cv2.getOptimalDFTSize(bh + kh - 1)
    if dh <= 0 or dw <= 0:
        raise OverflowError("Spectral transform dimensions exceed OpenCV's range")
    dw = max(dw, 2)
    bw, bh = min(dw - kw + 1, ow), min(dh - kh + 1, oh)
    plane = np.empty((dh, dw), np.float64)
    payload = coefficients.tobytes()
    key = (
        coefficients.shape,
        coefficients.dtype.str,
        payload,
        dh,
        dw,
        float_state(),
        cv2.useOptimized(),
        cv2.ipp.useIPP(),
        cv2.ipp.useIPP_NotExact(),
        cv2.getCPUFeaturesLine(),
    )

    def build_spectrum():
        spectrum = np.empty_like(plane)
        check(
            _lib.cn_spectral_kernel(
                coefficients.ctypes.data,
                int(coefficients.dtype == np.float64),
                spectrum.ctypes.data_as(_dp),
                kh,
                kw,
                dh,
                dw,
            )
        )
        cv2.dft(spectrum, spectrum, nonzeroRows=kh)
        spectrum.setflags(write=False)
        return spectrum, spectrum.nbytes + len(payload)

    spectrum = _spectra.get(key, build_spectrum)
    plane_ptr, spectrum_ptr = plane.ctypes.data_as(_dp), spectrum.ctypes.data_as(_dp)
    result = np.empty((oh, ow) if channels == 1 else (oh, ow, channels), np.float32)
    for y in range(0, oh, bh):
        th = min(bh, oh - y)
        for x in range(0, ow, bw):
            tw = min(bw, ow - x)
            for channel in range(channels):
                check(
                    _lib.cn_spectral_tile(
                        ptr(source),
                        plane_ptr,
                        h,
                        w,
                        channels,
                        padding,
                        y,
                        x,
                        th,
                        tw,
                        kh,
                        kw,
                        dh,
                        dw,
                        channel,
                        border,
                    )
                )
                cv2.dft(plane, plane, nonzeroRows=th + kh - 1)
                check(_lib.cn_spectral_multiply(plane_ptr, spectrum_ptr, dh, dw))
                cv2.dft(
                    plane, plane, flags=cv2.DFT_INVERSE | cv2.DFT_SCALE, nonzeroRows=th
                )
                check(
                    _lib.cn_spectral_store(
                        plane_ptr,
                        ptr(result),
                        oh,
                        ow,
                        channels,
                        y,
                        x,
                        th,
                        tw,
                        dh,
                        dw,
                        channel,
                    )
                )
    return result
