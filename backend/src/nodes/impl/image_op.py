from typing import Callable

import numpy as np
from typing_extensions import Concatenate, ParamSpec

from .native_framework_shared import clip_image

ImageOp = Callable[[np.ndarray], np.ndarray]
"""
An image processing operation that takes an image and produces a new image.

The given image is guaranteed to *not* be modified.
"""


def clipped(op: ImageOp) -> ImageOp:
    """
    Ensures that all values in the returned image are between 0 and 1.

    A float32 image is returned read-only and registered as normalized, so an
    ImageOutput that receives it borrows it instead of clipping a copy again.
    """
    return lambda i: clip_image(op(i), frozen=True)


P = ParamSpec("P")


def to_op(fn: Callable[Concatenate[np.ndarray, P], np.ndarray]) -> Callable[P, ImageOp]:
    """
    Curries fn: to_op(resize)(size) is `lambda img: resize(img, size)`. The image must
    be fn's first argument.
    """

    def p(*args: P.args, **kwargs: P.kwargs) -> ImageOp:
        return lambda i: fn(i, *args, **kwargs)

    return p
