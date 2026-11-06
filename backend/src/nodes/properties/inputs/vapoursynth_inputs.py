from __future__ import annotations

from typing import Union

import numpy as np

import navi
from api import BaseInput

from ...impl.vapoursynth.script import VideoClip, VSFilter
from ...utils.utils import get_h_w_c

FILTER_OR_CLIP = navi.union("VideoClip", "VSFilter")


def _image(value: np.ndarray) -> np.ndarray:
    assert value.dtype == np.float32, "Expected the input image to be normalized."
    if get_h_w_c(value)[2] == 1 and value.ndim == 3:
        return value[:, :, 0]
    return value


class VideoClipInput(BaseInput[Union[VideoClip, np.ndarray]]):
    """A VapourSynth clip: a variable of the chain's script. With take_images, also an
    image, which the node runs through its function at once."""

    def __init__(
        self, label: str = "Clip", input_type: navi.ExpressionJson = "VideoClip"
    ):
        super().__init__(input_type, label, associated_type=VideoClip)
        self.images = False

    def take_images(self, input_type: navi.ExpressionJson):
        """Takes images too; `input_type` says when."""
        self.input_type = input_type
        self.images = True
        self.associated_type = Union[VideoClip, np.ndarray]
        return self

    def to_dict(self):
        return {**super().to_dict(), "followsInputs": self.images}

    def enforce(self, value: object) -> VideoClip | np.ndarray:
        if self.images and isinstance(value, np.ndarray):
            return _image(value)
        assert isinstance(value, VideoClip)
        return value


class VSFilterInput(BaseInput[Union[VideoClip, VSFilter, np.ndarray]]):
    """A function argument that takes a filter (a node with Use as filter on) or a
    clip. With take_images, also an image."""

    def __init__(
        self,
        label: str = "Filter",
        input_type: navi.ExpressionJson = FILTER_OR_CLIP,
    ):
        super().__init__(input_type, label, associated_type=Union[VideoClip, VSFilter])
        self.images = False

    def take_images(self, input_type: navi.ExpressionJson):
        """Takes images too; `input_type` says when."""
        self.input_type = input_type
        self.images = True
        self.associated_type = Union[VideoClip, VSFilter, np.ndarray]
        return self

    def to_dict(self):
        return {**super().to_dict(), "followsInputs": self.images}

    def enforce(self, value: object) -> VideoClip | VSFilter | np.ndarray:
        if self.images and isinstance(value, np.ndarray):
            return _image(value)
        assert isinstance(value, (VideoClip, VSFilter))
        return value
