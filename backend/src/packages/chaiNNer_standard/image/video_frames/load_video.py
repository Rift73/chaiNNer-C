from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from sanic.log import logger

import navi
from api import Generator, IteratorOutputInfo, NodeContext, OutputId
from nodes.groups import if_enum_group
from nodes.impl.colour_info import (
    VIDEO_MODE_LABELS,
    ChromaLocation,
    ColourInfo,
    Matrix,
    Primaries,
    Range,
    Transfer,
    VideoMode,
)

# `x as x` imports: the native mirror reads these names in native/src/video_io.cpp
from nodes.impl.ffmpeg import FFMpegEnv as FFMpegEnv
from nodes.impl.native_graph import graph
from nodes.impl.video import ReadSettings, VideoMetadata, preview_frame
from nodes.impl.video import VideoLoader as VideoLoader
from nodes.properties.inputs import (
    EnumInput,
    NumberInput,
    VideoFileInput,
    VideoSeekInput,
    colour_override_inputs,
)
from nodes.properties.outputs import (
    AudioStreamOutput,
    ColourInfoOutput,
    DirectoryOutput,
    FileNameOutput,
    ImageOutput,
    LargeImageOutput,
    NumberOutput,
    TextOutput,
)
from nodes.utils.utils import split_file_path as split_file_path

from .. import video_frames_group


class RgbPrecision(Enum):
    FLOAT = "float"
    EIGHT_BIT = "8bit"


class FramesMode(Enum):
    ALL = "all"
    RANGE = "range"


@dataclass(frozen=True)
class LoadedVideo:
    """Load Video's generator metadata (native/src/video_io.cpp load_video)."""

    metadata: VideoMetadata
    mode: VideoMode
    colour: ColourInfo


def get_item_types(loaded: LoadedVideo):
    types: dict[OutputId, navi.ExpressionJson] = {}
    if loaded.mode is not VideoMode.PLANES:
        types[OutputId(0)] = navi.Image(
            width=loaded.metadata.width, height=loaded.metadata.height, channels=3
        )
    if loaded.mode is not VideoMode.RGB:
        (width, height), (chroma_width, chroma_height) = (
            loaded.colour.luma_size,
            loaded.colour.chroma_size,
        )
        types[OutputId(6)] = navi.Image(width=width, height=height, channels=1)
        for output_id in (7, 8):
            types[OutputId(output_id)] = navi.Image(
                width=chroma_width, height=chroma_height, channels=1
            )
    return types


def load_preview(node_id: str, loader: VideoLoader, n: int) -> np.ndarray:
    """The Preview (native/src/video_io.cpp load_video): frame `n` in RGB from the node's
    open preview reader (video.py preview_frame), whatever the Mode. A video read in
    Planes mode whose matrix RGB cannot convert (BT.2020 constant luminance) shows its Y
    plane in gray instead, so the node still reports its Frame Count and FPS."""
    try:
        loader.preview_format()
    except ValueError as reason:
        logger.warning(
            "%s: the Preview shows the Y plane in gray: %s",
            vars(loader)["path"],
            reason,
        )
        ((_, y, _, _),) = list(loader.stream_items(n, n + 1))
        assert y is not None
        return np.dstack((y, y, y))
    return preview_frame(node_id, loader, n)


RGB_MODES = "VideoMode::Rgb | VideoMode::Both"
PLANE_MODES = "VideoMode::Planes | VideoMode::Both"


def plane_output(label: str, output_id: int) -> ImageOutput:
    # Plane values outside 0-1 (limited range's foot- and headroom) are kept.
    return (
        ImageOutput(label, channels=1, assume_normalized=True)
        .with_id(output_id)
        .null_unless(3, PLANE_MODES)
    )


@video_frames_group.register(
    schema_id="chainner:image:load_video",
    name="Load Video",
    description=[
        "Iterate over the frames in a video as images: all of them, or a Range from In to Out.",
        "Uses FFMPEG to read video files.",
        "The seek bar picks a frame, which the Preview shows. In and Out set the Range there, and Run this frame runs the chain on that frame alone. A run of one frame (Run this frame, or a Range whose In is its Out) makes Save Video save the frame as a PNG image instead of a video.",
        "This iterator is much slower than just using FFMPEG directly, so if you are doing a simple conversion, just use FFMPEG outside chaiNNer instead.",
    ],
    icon="MdVideoCameraBack",
    inputs=[
        VideoFileInput(primary_input=True),
        VideoSeekInput(
            "Seek",
            frames_input=10,
            in_input=11,
            out_input=12,
            frame_count_output=12,
            fps_output=4,
        )
        .with_id(13)
        .with_docs(
            "The frame the Preview shows, by its number in the video (0 is the first)."
            " Past the last frame, the last one is shown."
        ),
        EnumInput(
            FramesMode,
            "Frames",
            default=FramesMode.ALL,
            option_labels={FramesMode.ALL: "All", FramesMode.RANGE: "Range"},
            preferred_style="tabs",
        )
        .with_id(10)
        .with_docs(
            "All reads every frame. Range reads the frames from In to Out, both"
            " included, by their numbers in the video (0 is the first), and cuts the"
            " audio to them."
        ),
        if_enum_group(10, FramesMode.RANGE)(
            NumberInput("In", default=0, min=0)
            .with_id(11)
            .with_docs(
                "The first frame read. Past the video's last frame, the run fails.",
                hint=True,
            ),
            NumberInput("Out", default=9, min=0)
            .with_id(12)
            .with_docs(
                "The last frame read. Past the video's last frame, the video is read to"
                " its end.",
                hint=True,
            ),
        ),
        EnumInput(
            VideoMode,
            "Mode",
            default=VideoMode.RGB,
            option_labels=VIDEO_MODE_LABELS,
            preferred_style="tabs",
        )
        .with_id(3)
        .with_docs(
            "RGB gives each frame as an RGB image. Planes gives the Y, U and V planes as"
            " they are stored, each at its own size (4:2:0 1080p: Y 1920x1080, U and V"
            " 960x540) and at the video's full bit depth: Y is 0 at black and 1 at white,"
            " U and V are 0.5 where neutral, and values outside 0-1 are kept. Both gives"
            " both. An RGB-coded video has no planes, and gray video no U and V."
        ),
        if_enum_group(3, (VideoMode.RGB, VideoMode.BOTH))(
            EnumInput(
                RgbPrecision,
                "RGB Precision",
                default=RgbPrecision.FLOAT,
                option_labels={
                    RgbPrecision.FLOAT: "Float",
                    RgbPrecision.EIGHT_BIT: "8-bit (as before)",
                },
            )
            .with_id(4)
            .with_docs(
                "Float converts the video's own bit depth to RGB in floating point, with"
                " the colour values below and the chroma siting. 8-bit converts as Load"
                " Video did before this option (8-bit RGB, chroma assumed centred), with"
                " a Matrix or Range chosen below; Chroma Siting does not apply to it."
            ),
        ),
        *colour_override_inputs(5),
    ],
    outputs=[
        ImageOutput("Frame", channels=3).null_unless(3, RGB_MODES),
        NumberOutput(
            "Index",
            output_type="""
                match Input10 {
                    FramesMode::Range => max(int, Input11) & min(int, Input12),
                    _ => min(uint, max(0, IterOutput0.length - 1)),
                }
            """,
        )
        .with_docs(
            "The frame's number in the video: 0 for its first frame, In for the first"
            " frame of a Range."
        )
        .with_never_reason("In is after Out."),
        DirectoryOutput("Video Directory", of_input=0),
        FileNameOutput("Name", of_input=0),
        NumberOutput("FPS", output_type="0.."),
        AudioStreamOutput().suggest(),
        plane_output("Y", 6),
        plane_output("U", 7),
        plane_output("V", 8),
        ColourInfoOutput().with_id(9),
        TextOutput("Colour")
        .with_id(10)
        .with_docs(
            "The colour information: tagged values, and which were guessed or chosen."
        ),
        LargeImageOutput("Preview", image_type=navi.Image(channels=3), has_handle=False)
        .with_id(11)
        .with_docs("The seek bar's frame in RGB, whatever the Mode."),
        NumberOutput("Frame Count", output_type="uint")
        .with_id(12)
        .with_docs(
            "The number of frames in the video. For a video with a variable frame rate,"
            " which is read at a constant one, it is the probe's estimate; for a video"
            " whose frames lack timestamps, an estimate by its video packets."
        ),
    ],
    iterator_outputs=IteratorOutputInfo(
        outputs=[0, 1, 6, 7, 8],
        length_type="""
            match Input10 {
                FramesMode::Range => max(0, Input12 - Input11 + 1),
                _ => uint,
            }
        """,
    ).with_item_types(LoadedVideo, get_item_types),
    node_context=True,
    side_effects=True,
    kind="generator",
)
def load_video_node(
    node_context: NodeContext,
    path: Path,
    seek: int,
    frames: FramesMode,
    first: int,
    last: int,
    mode: VideoMode,
    precision: RgbPrecision,
    matrix: Matrix,
    transfer: Transfer,
    primaries: Primaries,
    range_: Range,
    chroma_location: ChromaLocation,
) -> tuple[
    Generator[tuple[Any, ...]],
    Path,
    str,
    float,
    Any,
    ColourInfo,
    str,
    np.ndarray,
    int,
]:
    settings = ReadSettings(
        mode,
        precision is RgbPrecision.FLOAT,
        (matrix, transfer, primaries, range_, chroma_location),
    )
    generator, directory, name, fps, audio, preview, count = graph().video_load(
        globals(),
        node_context,
        path,
        frames is FramesMode.RANGE,
        first,
        last,
        seek,
        settings,
    )
    loaded = generator.metadata
    assert isinstance(loaded, LoadedVideo)
    return (
        generator,
        directory,
        name,
        fps,
        audio,
        loaded.colour,
        loaded.colour.summary(),
        preview,
        count,
    )
