from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import navi
from api import Generator, IteratorOutputInfo, NodeContext, OutputId
from nodes.groups import Condition, if_enum_group, if_group
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
from nodes.impl.video import ReadSettings, VideoMetadata
from nodes.impl.video import VideoLoader as VideoLoader
from nodes.properties.inputs import (
    BoolInput,
    EnumInput,
    NumberInput,
    VideoFileInput,
    colour_override_inputs,
)
from nodes.properties.outputs import (
    AudioStreamOutput,
    ColourInfoOutput,
    DirectoryOutput,
    FileNameOutput,
    ImageOutput,
    NumberOutput,
    TextOutput,
)
from nodes.utils.utils import split_file_path as split_file_path

from .. import video_frames_group


class RgbPrecision(Enum):
    FLOAT = "float"
    EIGHT_BIT = "8bit"


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
        "Iterate over all frames in a video as images.",
        "Uses FFMPEG to read video files.",
        "This iterator is much slower than just using FFMPEG directly, so if you are doing a simple conversion, just use FFMPEG outside chaiNNer instead.",
    ],
    icon="MdVideoCameraBack",
    inputs=[
        VideoFileInput(primary_input=True),
        BoolInput("Use limit", default=False).with_id(1),
        if_group(Condition.bool(1, True))(
            NumberInput("Limit", default=10, min=1)
            .with_docs(
                "Limit the number of frames to iterate over. This can be useful for testing the iterator without having to iterate over all frames of the video."
                " The audio is cut to the limited frames."
            )
            .with_id(2)
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
            output_type="min(uint, max(0, IterOutput0.length - 1))",
        ).with_docs("A counter that starts at 0 and increments by 1 for each frame."),
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
    ],
    iterator_outputs=IteratorOutputInfo(
        outputs=[0, 1, 6, 7, 8],
        length_type="if Input1 { min(uint, Input2) } else { uint }",
    ).with_item_types(LoadedVideo, get_item_types),
    node_context=True,
    side_effects=True,
    kind="generator",
)
def load_video_node(
    node_context: NodeContext,
    path: Path,
    use_limit: bool,
    limit: int,
    mode: VideoMode,
    precision: RgbPrecision,
    matrix: Matrix,
    transfer: Transfer,
    primaries: Primaries,
    range_: Range,
    chroma_location: ChromaLocation,
) -> tuple[Generator[tuple[Any, ...]], Path, str, float, Any, ColourInfo, str]:
    settings = ReadSettings(
        mode,
        precision is RgbPrecision.FLOAT,
        (matrix, transfer, primaries, range_, chroma_location),
    )
    generator, directory, name, fps, audio = graph().video_load(
        globals(), node_context, path, use_limit, limit, settings
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
    )
