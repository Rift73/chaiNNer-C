from __future__ import annotations

from enum import Enum

import numpy as np
from PIL import Image

from ..utils.utils import get_h_w_c
from . import native_rotate
from .image_utils import FillColor, convert_to_bgra, normalize, to_uint8


class InterpolationMethod(Enum):
    AUTO = -1
    NEAREST = 0
    LANCZOS = 1
    LINEAR = 2
    CUBIC = 3
    BOX = 4


class RotationInterpolationMethod(Enum):
    CUBIC = InterpolationMethod.CUBIC.value
    LINEAR = InterpolationMethod.LINEAR.value
    NEAREST = InterpolationMethod.NEAREST.value

    @property
    def interpolation_method(self) -> InterpolationMethod:
        return InterpolationMethod(self.value)


INTERPOLATION_METHODS_MAP = {
    InterpolationMethod.NEAREST: Image.Resampling.NEAREST,
    InterpolationMethod.BOX: Image.Resampling.BOX,
    InterpolationMethod.LINEAR: Image.Resampling.BILINEAR,
    InterpolationMethod.CUBIC: Image.Resampling.BICUBIC,
    InterpolationMethod.LANCZOS: Image.Resampling.LANCZOS,
}


class RotateSizeChange(Enum):
    EXPAND = 1
    CROP = 0


def resize(
    img: np.ndarray, out_dims: tuple[int, int], interpolation: InterpolationMethod
) -> np.ndarray:
    """Perform PIL resize"""

    if interpolation == InterpolationMethod.AUTO:
        # automatically chose a method that works
        new_w, new_h = out_dims
        old_h, old_w, _ = get_h_w_c(img)
        if new_w > old_w or new_h > old_h:
            interpolation = InterpolationMethod.LANCZOS
        else:
            interpolation = InterpolationMethod.BOX

    resample = INTERPOLATION_METHODS_MAP[interpolation]

    pimg = Image.fromarray(to_uint8(img, normalized=True))
    pimg = pimg.resize(out_dims, resample=resample)  # type: ignore
    return normalize(np.array(pimg))


def rotate(
    img: np.ndarray,
    angle: float,
    interpolation: RotationInterpolationMethod,
    expand: RotateSizeChange,
    fill: FillColor,
) -> np.ndarray:
    """Perform Pillow-compatible byte rotation in C."""

    c = get_h_w_c(img)[2]
    if fill == FillColor.TRANSPARENT:
        img = convert_to_bgra(img, c)
    fill_color = tuple([x * 255 for x in fill.get_color(c)])

    return native_rotate.rotate(
        to_uint8(img, normalized=True),
        angle,
        interpolation.value,
        bool(expand.value),
        fill_color,
    )
