from enum import Enum

import numpy as np

from ..utils.utils import get_h_w_c
from .native_blend import composite


class BlendMode(Enum):
    NORMAL = 0
    DARKEN = 2
    MULTIPLY = 1
    COLOR_BURN = 5
    LINEAR_BURN = 22
    LIGHTEN = 3
    SCREEN = 12
    COLOR_DODGE = 6
    ADD = 4
    OVERLAY = 9
    SOFT_LIGHT = 17
    HARD_LIGHT = 18
    VIVID_LIGHT = 19
    LINEAR_LIGHT = 20
    PIN_LIGHT = 21
    REFLECT = 7
    GLOW = 8
    DIFFERENCE = 10
    EXCLUSION = 16
    NEGATION = 11
    SUBTRACT = 14
    DIVIDE = 15
    XOR = 13
    COLOR = 23
    LUMINOSITY = 24


def blend_images(overlay: np.ndarray, base: np.ndarray, blend_mode: BlendMode):
    """Blend equal-sized grayscale, BGR, or BGRA images using the C core.

    Input images remain unchanged. The output uses the greater channel count,
    with the same normalization and alpha rules as the original implementation.
    """
    o_shape = get_h_w_c(overlay)
    b_shape = get_h_w_c(base)
    assert o_shape[:2] == b_shape[:2], (
        "The overlay and the base image must have the same size"
    )

    def assert_sane(c: int, name: str):
        assert c in (1, 3, 4), f"The {name} has to be a grayscale, RGB, or RGBA image"

    assert_sane(o_shape[2], "overlay layer")
    assert_sane(b_shape[2], "base layer")
    return composite(overlay, base, blend_mode.value, o_shape[2], b_shape[2])
