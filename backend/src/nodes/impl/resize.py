from __future__ import annotations

from enum import Enum

import numpy as np

from ..utils.utils import get_h_w_c
from . import native_resample


class ResizeFilter(Enum):
    AUTO = -1
    NEAREST = 0
    BOX = 4
    LINEAR = 2
    CATROM = 3
    LANCZOS = 1

    HERMITE = 5
    MITCHELL = 6
    BSPLINE = 7
    HAMMING = 8
    HANN = 9
    LAGRANGE = 10
    GAUSS = 11


def resize(
    img: np.ndarray,
    out_dims: tuple[int, int],
    filter: ResizeFilter,
    separate_alpha: bool = False,
    gamma_correction: bool = False,
) -> np.ndarray:
    h, w, c = get_h_w_c(img)
    new_w, new_h = out_dims

    # check memory
    GB: int = 2**30  # noqa: N806
    MAX_MEMORY = 16 * GB  # noqa: N806
    new_memory = new_w * new_h * c * 4
    if new_memory > MAX_MEMORY:
        raise RuntimeError(
            f"Resize would require {round(new_memory / GB, 3)} GB of memory, but only {MAX_MEMORY // GB} GB are allowed."
        )

    if filter == ResizeFilter.AUTO:
        # automatically chose a method that works
        if new_w > w or new_h > h:
            filter = ResizeFilter.LANCZOS
        else:
            filter = ResizeFilter.BOX

    if (w, h) == out_dims and (filter in (ResizeFilter.NEAREST, ResizeFilter.BOX)):
        # no resize needed
        return img.copy()

    if filter == ResizeFilter.NEAREST:
        # we don't need premultiplied alpha for NN
        separate_alpha = True
        if new_w == 0 or new_h == 0:
            # The original binding returns HWC even for a 2D grayscale input.
            return native_resample.filtered(
                img, out_dims, ResizeFilter.BOX.value, False
            )
        return native_resample.nearest(img, out_dims)

    if not separate_alpha and c == 4:
        # pre-multiply alpha
        if img.dtype == np.float32:
            img = native_resample.premultiply(img)
        else:
            img = img.copy()
            img[:, :, 0] *= img[..., 3]
            img[:, :, 1] *= img[..., 3]
            img[:, :, 2] *= img[..., 3]

    img = native_resample.filtered(img, out_dims, filter.value, gamma_correction)

    if not separate_alpha and c == 4 and img.size:
        # undo pre-multiply alpha
        img = native_resample.finish_alpha_inplace(img)

    return img
