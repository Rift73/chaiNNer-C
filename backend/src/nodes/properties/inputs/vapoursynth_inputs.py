from __future__ import annotations

from typing import Union

import navi
from api import BaseInput

from ...impl.vapoursynth.script import VideoClip, VSFilter

FILTER_OR_CLIP = navi.union("VideoClip", "VSFilter")


class VideoClipInput(BaseInput[VideoClip]):
    """A VapourSynth clip: a variable of the chain's script."""

    def __init__(
        self, label: str = "Clip", input_type: navi.ExpressionJson = "VideoClip"
    ):
        super().__init__(input_type, label, associated_type=VideoClip)

    def enforce(self, value: object) -> VideoClip:
        assert isinstance(value, VideoClip)
        return value


class VSFilterInput(BaseInput[Union[VideoClip, VSFilter]]):
    """A function argument that takes a filter (a node with Use as filter on) or a
    clip."""

    def __init__(
        self,
        label: str = "Filter",
        input_type: navi.ExpressionJson = FILTER_OR_CLIP,
    ):
        super().__init__(input_type, label, associated_type=Union[VideoClip, VSFilter])

    def enforce(self, value: object) -> VideoClip | VSFilter:
        assert isinstance(value, (VideoClip, VSFilter))
        return value
