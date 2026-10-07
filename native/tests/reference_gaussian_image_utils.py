"""Frozen original fast_gaussian_blur; numerical reference only."""
import math
import cv2
import numpy as np
from nodes.utils.utils import get_h_w_c

def fast_gaussian_blur(
    img: np.ndarray,
    sigma_x: float,
    sigma_y: float | None = None,
) -> np.ndarray:
    """
    Computes a channel-wise gaussian blur of the given image using a fast approximation.

    The maximum error of the approximation is guaranteed to be less than 0.1%.
    In addition to that, the error is guaranteed to be smoothly distributed across the image.
    There are no sudden spikes in error anywhere.

    Specifically, the method is implemented by downsampling the image, blurring the downsampled
    image, and then upsampling the blurred image. This is much faster than blurring the full image.
    Unfortunately, OpenCV's `resize` method has unfortunate artifacts when upscaling, so we
    apply a small gaussian blur to the image after upscaling to smooth out the artifacts. This
    single step almost doubles the runtime of the method, but it is still much faster than
    blurring the full image.
    """
    if sigma_y is None:
        sigma_y = sigma_x
    if sigma_x == 0 or sigma_y == 0:
        return img.copy()

    h, w, _ = get_h_w_c(img)

    def get_scale_factor(sigma: float) -> float:
        if sigma < 11:
            return 1
        if sigma < 15:
            return 1.25
        if sigma < 20:
            return 1.5
        if sigma < 25:
            return 2
        if sigma < 30:
            return 2.5
        if sigma < 50:
            return 3
        if sigma < 100:
            return 4
        if sigma < 200:
            return 6
        return 8

    def get_sizing(size: int, sigma: float, f: float) -> tuple[int, float, float]:
        """
        Return the size of the downsampled image, the sigma of the downsampled gaussian blur,
        and the sigma of the upscaled gaussian blur.
        """
        if f <= 1:
            # just use simple gaussian, the error is too large otherwise
            return size, 0, sigma

        size_down = math.ceil(size / f)
        f = size / size_down
        sigma_up = f
        sigma_down = math.sqrt(sigma**2 - sigma_up**2) / f
        return size_down, sigma_down, sigma_up

    # Handling different sigma values for x and y is difficult, so we take the easy way out
    # and just use the smaller one. There are potentially better ways of combining them, but
    # this is good enough for now.
    scale_factor = min(get_scale_factor(sigma_x), get_scale_factor(sigma_y))
    h_down, y_down_sigma, y_up_sigma = get_sizing(h, sigma_y, scale_factor)
    w_down, x_down_sigma, x_up_sigma = get_sizing(w, sigma_x, scale_factor)

    if h != h_down or w != w_down:
        # downsampled gaussian blur
        img = cv2.resize(img, (w_down, h_down), interpolation=cv2.INTER_AREA)
        img = cv2.GaussianBlur(
            img,
            (0, 0),
            sigmaX=x_down_sigma,
            sigmaY=y_down_sigma,
            borderType=cv2.BORDER_REFLECT,
        )
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    if x_up_sigma != 0 or y_up_sigma != 0:
        # post blur to smooth out artifacts
        img = cv2.GaussianBlur(
            img,
            (0, 0),
            sigmaX=x_up_sigma,
            sigmaY=y_up_sigma,
            borderType=cv2.BORDER_REFLECT,
        )

    return img
