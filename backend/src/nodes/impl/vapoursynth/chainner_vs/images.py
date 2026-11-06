"""Images in a VapourSynth script: an image as a clip of one frame, and a clip's frame as
an image, both as NumPy files, so that no value changes on the way.

chaiNNer's images are float32 sRGB-encoded BGR (HxW for gray), so a clip is RGBS or
GRAYS tagged so: _Matrix 0 for RGB, sRGB transfer, BT.709 primaries, full range,
progressive."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from .upscale import frame_image

# VapourSynth by name, as in upscale.py.
vs: Any = importlib.import_module("vapoursynth")

# H.273: identity, BT.709, sRGB.
RGB_MATRIX = 0
BT709 = 1
SRGB_TRANSFER = 13
# R81's _Range: 1 is full range.
FULL_RANGE = 1
PROGRESSIVE = 0


class NotFiniteError(ValueError):
    """A result with NaN or infinity among its values."""


def image_clip(path: str, route: str) -> Any:
    """The image saved at `path` as a clip of one frame. Route "yuv" makes an RGB image
    YUV444PS (BT.709), for a function that wants luma; route "luma" gives that clip's luma
    alone, for a mask, which on YUV would mask each plane; gray stays gray."""
    image = np.load(path, allow_pickle=False)
    gray = image.ndim == 2
    blank = vs.core.std.BlankClip(
        format=vs.GRAYS if gray else vs.RGBS,
        width=image.shape[1],
        height=image.shape[0],
        length=1,
        keep=True,
    )

    def fill(n: int, f: Any) -> Any:
        frame = f.copy()
        if gray:
            np.copyto(np.asarray(frame[0]), image)
        else:
            for plane, channel in enumerate((2, 1, 0)):
                np.copyto(np.asarray(frame[plane]), image[:, :, channel])
        return frame

    clip = blank.std.ModifyFrame(blank, fill).std.SetFrameProps(
        _Transfer=SRGB_TRANSFER,
        _Primaries=BT709,
        _Range=FULL_RANGE,
        _FieldBased=PROGRESSIVE,
    )
    if gray:
        return clip
    clip = clip.std.SetFrameProps(_Matrix=RGB_MATRIX)
    if route == "yuv":
        return clip.resize.Point(format=vs.YUV444PS, matrix=BT709)
    if route == "luma":
        yuv = clip.resize.Point(format=vs.YUV444PS, matrix=BT709)
        return yuv.std.ShufflePlanes(0, vs.GRAY).std.RemoveFrameProps("_Matrix")
    return clip


def save_frame(clip: Any, path: str) -> None:
    """The clip's only frame as an image at `path`: gray clips as gray, every other clip
    as RGB (YUV by its _Matrix, else BT.709)."""
    # R81 gives a varying format the id 0; the API allows no format at all.
    if clip.format is None or not clip.format.id or not clip.width or not clip.height:
        raise ValueError(
            "This function gives frames whose size or format varies, which is no image."
        )
    if clip.num_frames != 1:
        raise ValueError(
            f"This function makes {clip.num_frames} frames from one image, so it takes"
            " clips, not images."
        )
    family = clip.format.color_family
    target = vs.GRAYS if family == vs.GRAY else vs.RGBS
    if clip.format.id != target:
        matrix = {"matrix_in": BT709} if family == vs.YUV else {}
        # A function may make YUV of an RGB image and keep its _Matrix, which would beat
        # matrix_in, and which YUV cannot have.
        if family == vs.YUV and clip.get_frame(0).props.get("_Matrix") == RGB_MATRIX:
            clip = clip.std.RemoveFrameProps("_Matrix")
        clip = clip.resize.Bicubic(format=target, **matrix)
    image = frame_image(clip.get_frame(0))
    # A plugin's float code can make them (NL-Means on the CPU), and every later node would
    # show or save them as whatever its conversion makes of them.
    finite = np.isfinite(image)
    if not finite.all():
        raise NotFiniteError(
            f"This function's result has {image.size - np.count_nonzero(finite)} values"
            " that are not numbers (NaN or infinity), which is no image."
        )
    np.save(path, image, allow_pickle=False)
