from __future__ import annotations

from typing import Union

import numpy as np

import navi
from api import BaseOutput, BroadcastData

from ...impl.vapoursynth.script import VideoClip, VSFilter
from .numpy_outputs import ImageOutput


class VideoClipOutput(BaseOutput[Union[VideoClip, VSFilter, np.ndarray]]):
    """A VapourSynth clip; with an output_type that can be VSFilter (a family node's
    Use as filter switch), a clip or a filter; with images, an image when the node was
    given one."""

    def __init__(
        self,
        label: str = "Clip",
        output_type: navi.ExpressionJson = "VideoClip",
        images: bool = False,
    ):
        filters = "VSFilter" in str(output_type)
        associated_type = Union[VideoClip, VSFilter] if filters else VideoClip
        super().__init__(
            output_type,
            label,
            associated_type=(
                Union[associated_type, np.ndarray] if images else associated_type
            ),
        )
        self._image = ImageOutput(label) if images else None

    def enforce(self, value: object) -> VideoClip | VSFilter | np.ndarray:
        if self._image is not None and isinstance(value, np.ndarray):
            return self._image.enforce(value)
        assert isinstance(value, (VideoClip, VSFilter))
        return value

    def get_broadcast_data(self, value: object) -> BroadcastData | None:
        if self._image is not None and isinstance(value, np.ndarray):
            return self._image.get_broadcast_data(value)
        return None

    def get_broadcast_type(self, value: object) -> navi.ExpressionJson | None:
        if self._image is not None and isinstance(value, np.ndarray):
            return self._image.get_broadcast_type(value)
        if isinstance(value, VideoClip) and value.shown is not None:
            shown = value.shown
            fields: dict[str, navi.ExpressionJson] = {
                "width": shown.width,
                "height": shown.height,
                "fps": navi.literal(shown.fps),
            }
            if shown.field_order:
                fields["fieldOrder"] = navi.literal(shown.field_order)
            return navi.named("VideoClip", fields)
        return None
