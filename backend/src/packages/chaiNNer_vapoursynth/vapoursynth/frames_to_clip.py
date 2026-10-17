from __future__ import annotations

import uuid
from fractions import Fraction

import numpy as np
from sanic.log import logger

from api import Collector, IteratorInputInfo, NodeContext
from nodes.impl.vapoursynth.script import VideoClip, clip_var
from nodes.properties.inputs import ImageInput, NumberInput
from nodes.properties.outputs import VideoClipOutput
from packages.chaiNNer_standard.image.video_frames.save_video import (
    AudioSettings,
    SimpleVideoFormat,
    Simplicity,
    VideoEncoder,
    VideoFormat,
    VideoPreset,
    save_video_node,
)

from ..chains import chain_dir
from . import clips_group


def frame_rate(fps: float) -> Fraction:
    """The clip's frame rate: an NTSC rate (24000/1001 for 23.976) when fps is one to
    the FPS input's 4 decimals, else fps as a fraction."""
    ntsc = Fraction(round(fps * 1.001) * 1000, 1001)
    if abs(fps - ntsc) < 0.0005:
        return ntsc
    return Fraction(fps).limit_denominator(1001)


@clips_group.register(
    schema_id="chainner:vapoursynth:frames_to_clip",
    name="Frames to Clip",
    description=[
        (
            "Turns a sequence of images into a clip: the frames are saved as a lossless"
            " 8-bit RGB video (FFV1), which the clip reads with BestSource."
        ),
        (
            "The video is a temporary file in chaiNNer's storage; the node's next run"
            " replaces it, and a start of chaiNNer deletes it once it is a day old."
        ),
    ],
    icon="BsFilm",
    inputs=[
        ImageInput("Frames", channels=3),
        NumberInput("FPS", default=30, min=1, step=1, precision=4),
    ],
    iterator_inputs=IteratorInputInfo(inputs=0),
    outputs=[VideoClipOutput("Clip")],
    kind="collector",
    node_context=True,
)
def frames_to_clip_node(
    context: NodeContext, _: None, fps: float
) -> Collector[np.ndarray, VideoClip]:
    folder = chain_dir(context)
    # The previous run's video and its index, unless a clip in an open vsview still
    # reads them; a later run deletes those.
    for old in folder.iterdir():
        try:
            old.unlink()
        except OSError as error:
            logger.debug("Frames to Clip keeps %s: %s", old, error)

    name = f"frames-{uuid.uuid4().hex[:12]}"
    # Save Video's writer: FFV1 in Matroska, lossless 8-bit RGB. (FFV1 level 3 or
    # -slices corrupt frames smaller than a few pixels, and were no faster at 1080p.)
    video = save_video_node(
        context,
        None,
        folder,
        name,
        Simplicity.ADVANCED,
        VideoFormat.MKV,
        VideoEncoder.FFV1,
        VideoPreset.MEDIUM,
        0,
        "-pix_fmt bgr0",
        SimpleVideoFormat.MP4_H264,
        75,
        fps,
        None,
        AudioSettings.AUTO,
    )
    path = folder / f"{name}.mkv"
    rate = frame_rate(fps)
    var = clip_var(context.node_id)
    # cachemode=3 keeps BestSource's index (written only for long videos) beside the
    # video, so it goes with it; AssumeFPS sets the exact rate the container rounds.
    statement = (
        f"{var} = core.bs.VideoSource(source={str(path)!r}, cachemode=3)"
        f".std.AssumeFPS(fpsnum={rate.numerator}, fpsden={rate.denominator})"
    )

    def complete() -> VideoClip:
        video.on_complete()
        if not path.is_file():
            raise ValueError("Frames to Clip got no frames.")
        return VideoClip(
            var=var,
            stmt=statement,
            deps=(),
            imports=frozenset(),
            node_id=context.node_id,
        )

    return Collector(on_iterate=video.on_iterate, on_complete=complete)
