import cv2
import numpy as np

from ..utils.utils import get_h_w_c
from .image_utils import as_2d_grayscale
from .native_analysis import cas_mask
from .native_analysis import cas_mix as native_cas_mix
from .native_cas import extrema, luminance
from .native_filters import morphology


def _luminance(img: np.ndarray) -> np.ndarray:
    """Returns the luminance of an image."""
    _, _, c = get_h_w_c(img)
    if c == 1:
        return as_2d_grayscale(img)
    if c == 2:
        return img[..., 0]
    return luminance(img)


def create_cas_mask(img: np.ndarray, kernel: np.ndarray, bias: float = 2) -> np.ndarray:
    """
    Uses contrast adaptive sharpening's method to create a mask to interpolate between the original
    and sharpened image.

    `kernel` is an element create by `cv2.getStructuringElement`. It determines the shape and size
    of each pixel's neighborhood.

    `bias` is used to bias the mask towards the more sharpening. A value of 1 means no bias, a
    value greater than 1 biases the mask towards the sharpened image, a value less than 1 biases
    the mask towards the original image.

    Reference:
    Lou Kramer, FidelityFX CAS, AMD Developer Day 2019, https://gpuopen.com/wp-content/uploads/2019/07/FidelityFX-CAS.pptx
    https://www.shadertoy.com/view/wtlSWB#
    """
    assert bias > 0, "Bias must be greater than or equal to 0."

    l = _luminance(img)
    if l.dtype == np.float64:
        min_l, max_l = extrema(l, kernel)
    elif kernel.shape == (3, 3) and np.array_equal(kernel, np.ones((3, 3), np.uint8)):
        min_l = morphology(l, cv2.MORPH_RECT, 1, 1, maximum=False)
        max_l = morphology(l, cv2.MORPH_RECT, 1, 1, maximum=True)
    elif kernel.shape == (3, 3) and np.array_equal(
        kernel, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    ):
        min_l = morphology(l, cv2.MORPH_CROSS, 1, 1, maximum=False)
        max_l = morphology(l, cv2.MORPH_CROSS, 1, 1, maximum=True)
    else:
        # Only external helper callers can supply other masks; High Boost's
        # grayscale/RGB/RGBA paths use the two C shapes above.
        min_l = cv2.erode(l, kernel)
        max_l = cv2.dilate(l, kernel)
    return cas_mask(min_l, max_l, bias)


def cas_mix(
    img: np.ndarray,
    sharpened: np.ndarray,
    kernel: np.ndarray,
    bias: float = 2,
) -> np.ndarray:
    mask = create_cas_mask(img, kernel, bias)
    return native_cas_mix(img, sharpened, mask)
