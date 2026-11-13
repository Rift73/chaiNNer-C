from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator

import numpy as np
from sanic.log import logger

import navi
from api import Generator, IteratorOutputInfo, NodeContext, OutputId
from nodes.groups import if_enum_group
from nodes.impl.colour_info import (
    VIDEO_MODE_LABELS,
    ColourInfo,
    VideoMode,
    code_to_float,
    detect_colour_info,
)
from nodes.impl.vapoursynth.colour_props import colour_stream, vs_pix_fmt
from nodes.impl.vapoursynth.runner import (
    ClipInfo,
    VSScriptError,
    clip_props,
    read_frames,
    read_planes,
)
from nodes.impl.vapoursynth.script import VideoClip
from nodes.properties.inputs import (
    EnumInput,
    NumberInput,
    VideoClipInput,
    VideoSeekInput,
)
from nodes.properties.outputs import (
    ColourInfoOutput,
    ImageOutput,
    LargeImageOutput,
    NumberOutput,
)

from ..chains import chain_dir, write_script
from . import clips_group

# One item of the frames: Frame, Index, Y, U, V (None for another mode's outputs).
Item = tuple[
    np.ndarray | None, int, np.ndarray | None, np.ndarray | None, np.ndarray | None
]


class FrameSelection(Enum):
    # The seek bar's Set start and Set end buttons set these values (VideoSeekInput).
    ALL = "all"
    RANGE = "range"


def frame_range(selection: FrameSelection, first: int, last: int, count: int):
    """The frames start..end-1 a run renders: all, or Start to End (an End past the end
    reads to the end)."""
    if selection is FrameSelection.ALL:
        return 0, count
    if first > last:
        raise ValueError(f"Start ({first}) is after End ({last}).")
    if first >= count:
        raise ValueError(
            f"Start ({first}) is past the clip's last frame ({count - 1}). Set Start and"
            " End with the seek bar's Set start and Set end."
        )
    return first, min(last, count - 1) + 1


@dataclass(frozen=True)
class FramesInfo:
    """The generator's metadata: the clip, the mode and the clip's ColourInfo."""

    clip: ClipInfo
    mode: VideoMode
    colour: ColourInfo


def get_item_types(info: FramesInfo):
    clip = info.clip
    width, height = clip.width or None, clip.height or None
    types: dict[OutputId, navi.ExpressionJson] = {}
    if info.mode is not VideoMode.PLANES:
        channels = 1 if clip.format_name.startswith("Gray") else 3
        types[OutputId(0)] = navi.Image(width=width, height=height, channels=channels)
    if info.mode is not VideoMode.RGB:
        chroma_width, chroma_height = info.colour.chroma_size
        types[OutputId(3)] = navi.Image(width=width, height=height, channels=1)
        chroma = navi.Image(
            width=chroma_width or None, height=chroma_height or None, channels=1
        )
        types[OutputId(4)] = types[OutputId(5)] = chroma
    return types


@clips_group.register(
    schema_id="chainner:vapoursynth:clip_frames",
    name="Clip Frames",
    description=[
        (
            "Iterates over the frames of a clip as images, rendered by vspipe: all of"
            " them, or a Range from Start to End."
        ),
        (
            "The seek bar picks the frame the Preview shows, rendered at each run and"
            " while the bar moves. Set start and Set end set the Range there, and Run"
            " this frame runs the chain on that frame alone. A Range renders only its"
            " frames, through every filter before this node."
        ),
        (
            "Gray clips give gray images; every other clip is converted to RGB (YUV with"
            " the clip's matrix, else as Load Video reads untagged video: BT.709 at"
            " least 1280 pixels wide or above 576 lines, else BT.601). Add a Convert"
            " Format node before this one to choose the conversion yourself."
        ),
        (
            "Mode Planes gives a YUV clip's Y, U and V planes at their own sizes"
            " instead, as Load Video's Planes mode does; Both gives both."
        ),
        (
            "Use this node's own Index and FPS for Save Image or Save Video. After a"
            " filter that drops or adds frames (decimation, double-rate"
            " deinterlacing) they differ from the source's, and an output of another"
            " loop (Load Video's Index) cannot be used in this one."
        ),
    ],
    see_also=["chainner:vapoursynth:load_clip"],
    icon="MdVideoCameraBack",
    inputs=[
        VideoClipInput("Clip").with_docs(
            "The clip whose frames come out one by one as images.", hint=True
        ),
        VideoSeekInput(
            "Seek",
            frames_input=3,
            in_input=4,
            out_input=5,
            frame_count_output=8,
            fps_output=2,
        )
        .with_id(2)
        .with_docs(
            "The frame the Preview shows, by its number in this clip (0 is the first);"
            " each run renders it, and so does moving the bar, through the nodes"
            " before this one only. The bar is empty until the first run, which"
            " evaluates the chain to count the frames. Run this frame renders only that"
            " frame through the whole chain.",
            hint=True,
        ),
        EnumInput(
            FrameSelection,
            "Frames",
            default=FrameSelection.ALL,
            option_labels={FrameSelection.ALL: "All", FrameSelection.RANGE: "Range"},
            preferred_style="tabs",
        )
        .with_id(3)
        .with_docs(
            "All renders every frame. Range renders only the frames from Start to End,"
            " both included, still through every filter before this node, so a frame"
            " after field matching or decimation is the one All gives.\n\n"
            "Example: Start `100`, End `199` gives 100 frames.",
            hint=True,
        ),
        if_enum_group(3, FrameSelection.RANGE)(
            NumberInput("Start", default=0, min=0)
            .with_id(4)
            .with_docs(
                "The first frame rendered, by its number in this clip (0 is the first)."
                " Past the clip's last frame, the run fails.\n\nExample: `100`",
                hint=True,
            ),
            NumberInput("End", default=9, min=0)
            .with_id(5)
            .with_docs(
                "The last frame rendered, included. Past the clip's last frame, the"
                " clip is rendered to its end.\n\nExample: `199`",
                hint=True,
            ),
        ),
        EnumInput(
            VideoMode,
            "Mode",
            default=VideoMode.RGB,
            option_labels=VIDEO_MODE_LABELS,
        )
        .with_id(1)
        .with_docs(
            "RGB gives each frame as one image (gray for a gray clip). Planes gives"
            " the planes as stored: Y with 0 black and 1 white, U and V centred on 0.5,"
            " values outside 0-1 kept (the clip's range and bit depth undone). Both"
            " gives both. Planes and Both need a YUV clip.",
            hint=True,
        ),
    ],
    outputs=[
        ImageOutput("Frame").null_unless(1, "VideoMode::Rgb | VideoMode::Both"),
        NumberOutput(
            "Index",
            output_type="""
                match Input3 {
                    FrameSelection::Range => max(int, Input4) & min(int, Input5),
                    _ => min(uint, max(0, IterOutput0.length - 1)),
                }
            """,
        )
        .with_docs(
            "The frame's number in this clip: 0 for its first frame, Start for the first"
            " frame of a Range."
        )
        .with_never_reason("Start is after End."),
        NumberOutput("FPS", output_type="0..").with_docs(
            "The clip's frame rate; 0 when it varies."
        ),
        ImageOutput("Y", channels=1).null_unless(
            1, "VideoMode::Planes | VideoMode::Both"
        ),
        ImageOutput("U", channels=1).null_unless(
            1, "VideoMode::Planes | VideoMode::Both"
        ),
        ImageOutput("V", channels=1).null_unless(
            1, "VideoMode::Planes | VideoMode::Both"
        ),
        ColourInfoOutput().with_docs(
            "The clip's colour information, from its first frame's properties"
            " (_Matrix, _Transfer, _Primaries, _Range, _ChromaLocation); a property"
            " that is not set is guessed as Load Video guesses it."
        ),
        LargeImageOutput("Preview", has_handle=False).with_docs(
            "The seek bar's frame in RGB (gray for a gray clip), whatever the Mode."
        ),
        NumberOutput("Frame Count", output_type="uint").with_docs(
            "The number of frames in this clip, after every filter before this node."
        ),
    ],
    iterator_outputs=IteratorOutputInfo(
        outputs=[0, 1, 3, 4, 5],
        length_type="""
            match Input3 {
                FrameSelection::Range => max(0, Input5 - Input4 + 1),
                _ => uint,
            }
        """,
    ).with_item_types(FramesInfo, get_item_types),
    node_context=True,
    side_effects=True,
    kind="generator",
)
def clip_frames_node(
    context: NodeContext,
    clip: VideoClip,
    seek: int,
    selection: FrameSelection,
    first: int,
    last: int,
    mode: VideoMode,
) -> tuple[Generator[Item], float, ColourInfo, np.ndarray, int]:
    script = write_script(context, [(clip, "Clip")], chain_dir(context) / "frames.vpy")
    try:
        # One evaluation for the length, FPS, colour and the Preview.
        info, props, preview = clip_props(script, 0, context, preview=seek)
    except VSScriptError as error:
        # The Preview frame's own error names no line of the chain.
        if error.node_id is None:
            error.node_id = context.node_id
        raise
    assert preview is not None
    shown = max(0, min(seek, info.num_frames - 1))
    # The Preview is the frame Mode RGB reads, unless it is the Y plane of a matrix
    # RGB cannot convert.
    is_frame = preview.shape[2] == (1 if info.format_name.startswith("Gray") else 3)
    if not is_frame:
        logger.warning(
            "Clip Frames: the Preview shows the Y plane in gray: RGB cannot convert"
            " the clip's matrix."
        )
    pix_fmt = vs_pix_fmt(info.format_name)
    colour, warnings = detect_colour_info(
        colour_stream(props, pix_fmt, info.width, info.height)
    )
    for warning in warnings:
        logger.warning("Clip Frames: %s", warning)
    if mode is not VideoMode.RGB and not pix_fmt.startswith("yuv"):
        raise ValueError(
            f"Mode {VIDEO_MODE_LABELS[mode]} needs a YUV clip, and this clip is"
            f" {info.format_name}. Choose Mode RGB, or convert the clip to YUV before"
            " this node."
        )
    # A Range that fails fails at its first item, so that the node still reports its
    # Preview and Frame Count, by which the seek bar sets the Range again.
    try:
        start, end = frame_range(selection, first, last, info.num_frames)
        invalid = None
    except ValueError as error:
        start = end = 0
        invalid = error
    # Float samples as they are (VapourSynth's float chroma is centred on 0).
    float_samples = info.format_name.endswith(("H", "S"))

    def values(plane: np.ndarray, chroma: bool) -> np.ndarray:
        if float_samples:
            return plane + np.float32(0.5) if chroma else plane
        return code_to_float(plane, colour.bit_depth, colour.full_range, chroma)

    def items() -> Iterator[Item]:
        if invalid is not None:
            raise invalid
        if mode is VideoMode.RGB:
            if is_frame and (start, end) == (shown, shown + 1):
                # Run this frame: the Preview is the frame, without a second vspipe.
                yield preview, shown, None, None, None
                return
            # The info above: vspipe evaluates the script once more, for the frames.
            frames = read_frames(script, 0, start, end, info=info, context=context)
            # Stops vspipe when the run ends before the last frame.
            context.add_cleanup(frames.close)
            for index, frame in enumerate(frames, start):
                yield frame, index, None, None, None
            return
        planes = read_planes(
            script,
            0,
            start,
            end,
            info=info,
            rgb=mode is VideoMode.BOTH,
            context=context,
        )
        context.add_cleanup(planes.close)
        for index, (frame, y, u, v) in enumerate(planes, start):
            yield frame, index, values(y, False), values(u, True), values(v, True)

    def frames() -> Iterator[Item]:
        try:
            yield from items()
        except VSScriptError as error:
            # A plugin's error at frame time names no line of the script. The
            # iteration's error arrives without this node, so it goes on it here.
            if error.node_id is None:
                error.node_id = context.node_id
            raise

    # As Load Video's: items numbered from Start, and a one-frame Range (Run this frame)
    # is a one-item run of that frame, which Save Video saves as an image.
    one = selection is FrameSelection.RANGE and end - start == 1
    generator = (
        Generator.from_iter(supplier=frames, expected_length=end - start)
        .with_metadata(FramesInfo(info, mode, colour))
        .with_range(start, start if one else None)
    )
    # Frames come from vspipe's pipe, as Load Video's come from FFmpeg's.
    generator.source_paths = ()
    fps = info.fps_num / info.fps_den
    return generator, fps, colour, preview, info.num_frames
