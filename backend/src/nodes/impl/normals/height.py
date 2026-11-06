from enum import Enum
from typing import Any, cast

import numpy as np

from ...utils.utils import get_h_w_c


class HeightSource(Enum):
    AVERAGE_RGB = 0
    MAX_RGB = 1
    # 1 - ((1-r) * (1-g) * (1-b))
    SCREEN_RGB = 2
    RED = 3
    GREEN = 4
    BLUE = 5
    ALPHA = 6


def get_height_map(img: np.ndarray, source: HeightSource) -> np.ndarray:
    """
    Converts the given color/grayscale image to a height map.
    """
    h, w, c = get_h_w_c(img)

    assert c in (1, 3, 4), "Only grayscale, RGB, and RGBA images are supported"

    if source == HeightSource.ALPHA:
        if c < 4:
            return np.ones((h, w), dtype=np.float32)
        return img[:, :, 3]

    if c == 1:
        if source == HeightSource.SCREEN_RGB:
            x = 1 - img
            return 1 - x * x * x
        return img

    r = img[:, :, 2]
    g = img[:, :, 1]
    b = img[:, :, 0]

    if source == HeightSource.RED:
        return r
    elif source == HeightSource.GREEN:
        return g
    elif source == HeightSource.BLUE:
        return b
    elif source == HeightSource.MAX_RGB:
        return np.maximum(np.maximum(r, g), b)
    elif source == HeightSource.AVERAGE_RGB:
        return (r + g + b) / 3
    elif source == HeightSource.SCREEN_RGB:
        return 1 - ((1 - r) * (1 - g) * (1 - b))
    else:
        raise AssertionError(f"Invalid height source {source}.")


def integrate_slopes(
    slope_x: np.ndarray, slope_y: np.ndarray, tileable: bool
) -> np.ndarray:
    """
    The height field whose slopes per column and per row are closest to the given ones in the
    least-squares sense (Frankot and Chellappa). The solve is exact for sampled slopes and,
    unlike a finite-difference Poisson solve, does not shift the result by half a pixel.

    The result has no overall tilt: the slopes' means have no effect. A tileable height field
    wraps around; otherwise its edges mirror, which ties nothing to the opposite edge.
    """
    # SciPy's import is slow, and every start of the backend imports this module.
    from scipy import fft as scipy_fft

    # Its stubs give these functions the type of their dispatch, not of the array they return.
    fft = cast("Any", scipy_fft)

    h, w = slope_x.shape
    p = slope_x - float(np.mean(slope_x, dtype=np.float64))
    q = slope_y - float(np.mean(slope_y, dtype=np.float64))

    if tileable:
        u = 2 * np.pi * fft.rfftfreq(w).astype(np.float32)
        v = 2 * np.pi * fft.fftfreq(h).astype(np.float32)
        # A wave at the sampling limit has no slope at the samples.
        if w % 2 == 0:
            u[-1] = 0
        if h % 2 == 0:
            v[h // 2] = 0
        denominator = u * u + (v * v)[:, None]
        denominator[denominator == 0] = 1
        spectrum = -1j * (u * fft.rfft2(p) + v[:, None] * fft.rfft2(q)) / denominator
        return fft.irfft2(spectrum, s=(h, w))

    # Mirrored edges make the height field a cosine series, and its slopes along an axis a
    # sine series whose coefficient i belongs to the height's frequency i + 1.
    b = fft.dct(fft.dst(p, type=2, axis=1, norm="ortho"), type=2, axis=0, norm="ortho")
    c = fft.dst(fft.dct(q, type=2, axis=1, norm="ortho"), type=2, axis=0, norm="ortho")
    u = np.pi * np.arange(w, dtype=np.float32) / w
    v = np.pi * np.arange(h, dtype=np.float32) / h
    numerator = np.zeros((h, w), dtype=np.float32)
    numerator[:, 1:] += u[1:] * b[:, :-1]
    numerator[1:] += v[1:, None] * c[:-1]
    denominator = u * u + (v * v)[:, None]
    denominator[0, 0] = 1
    return fft.idctn(-numerator / denominator, type=2, norm="ortho")
