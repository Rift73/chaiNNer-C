from __future__ import annotations

from typing import Union

import navi
from api import BaseOutput

from ...impl.vapoursynth.script import VideoClip, VSFilter


class VideoClipOutput(BaseOutput[Union[VideoClip, VSFilter]]):
    """A VapourSynth clip; with an output_type that can be VSFilter (a family node's
    Use as filter switch), a clip or a filter."""

    def __init__(
        self, label: str = "Clip", output_type: navi.ExpressionJson = "VideoClip"
    ):
        super().__init__(
            output_type,
            label,
            associated_type=(
                VideoClip if output_type == "VideoClip" else Union[VideoClip, VSFilter]
            ),
        )

    def enforce(self, value: object) -> VideoClip | VSFilter:
        assert isinstance(value, (VideoClip, VSFilter))
        return value
