from __future__ import annotations

import cv2
import numpy as np

from ...utils.utils import get_h_w_c
from ..native_channels import concatenate_channels
from ..native_color_complete import cvt_color
from ..native_filters import morphology
from ..native_framework_images import matting_trimap, threshold_mask
from ..native_gaussian import gaussian
from ..native_layout import stack
from ..native_matting import estimate_alpha, estimate_foreground
from ..native_tensors import cast_numpy
from ..onnx.session import OnnxSession
from .session_factory import new_session


def assert_rgb(img: np.ndarray):
    assert get_h_w_c(img)[2] == 3


def assert_gray(img: np.ndarray):
    assert img.ndim == 2


def alpha_matting_cutout(
    img: np.ndarray,
    mask: np.ndarray,
    foreground_threshold: int,
    background_threshold: int,
    erode_structure_size: int,
) -> np.ndarray:
    assert_rgb(img)
    assert_gray(mask)

    trimap = matting_trimap(
        mask, foreground_threshold, background_threshold, erode_structure_size
    )
    img64 = cast_numpy(img, np.dtype(np.float64))
    alpha = estimate_alpha(img64, trimap)
    foreground = estimate_foreground(img64, alpha)
    assert isinstance(foreground, np.ndarray)

    return concatenate_channels(foreground, cast_numpy(alpha, np.dtype(np.float32)))


def naive_cutout(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    assert_rgb(img)
    assert_gray(mask)
    return concatenate_channels(img, mask)


def post_process(mask: np.ndarray) -> np.ndarray:
    """
    Post Process the mask for a smooth boundary by applying Morphological Operations
    Research based on paper: https://www.sciencedirect.com/science/article/pii/S2352914821000757
    args:
        mask: Binary Numpy Mask
    """
    mask = morphology(mask, cv2.MORPH_ELLIPSE, 1, 1, maximum=False)
    mask = morphology(mask, cv2.MORPH_ELLIPSE, 1, 1, maximum=True)
    mask = gaussian(mask, 2, 2, size=(5, 5), border=cv2.BORDER_DEFAULT)
    mask = threshold_mask(mask)
    return mask


def remove_bg(
    img: np.ndarray,
    ort_session: OnnxSession,
    alpha_matting: bool = False,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
    alpha_matting_erode_size: int = 10,
    post_process_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    # Flip channels to RGB mode
    assert_rgb(img)
    img = cvt_color(img, cv2.COLOR_BGR2RGB)
    session = new_session(ort_session)

    masks: list[np.ndarray] = session.predict(img)
    cutouts: list[np.ndarray] = []

    assert len(masks) > 0, "Model failed to generate masks"

    mask = None
    for mask in masks:
        if post_process_mask:
            mask = post_process(mask)  # noqa

        if alpha_matting:
            try:
                cutout = alpha_matting_cutout(
                    img,
                    mask,
                    alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold,
                    alpha_matting_erode_size,
                )
            except ValueError:
                cutout = naive_cutout(img, mask)
        else:
            cutout = naive_cutout(img, mask)

        cutouts.append(cutout)

    if mask is None or len(cutouts) == 0:
        raise ValueError("Model failed to generate masks")

    cutout = stack(cutouts, "vertical")

    return cvt_color(cutout, cv2.COLOR_RGBA2BGRA), mask
