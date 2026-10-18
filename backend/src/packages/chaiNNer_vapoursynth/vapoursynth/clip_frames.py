from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from sanic.log import logger

import navi
from api import Generator, IteratorOutputInfo, NodeContext, OutputId
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
from nodes.properties.inputs import EnumInput, VideoClipInput
from nodes.properties.outputs import ColourInfoOutput, ImageOutput, NumberOutput

from ..chains import chain_dir, write_script
from . import clips_group

# One item of the frames: Frame, Index, Y, U, V (None for another mode's outputs).
Item = tuple[
    np.ndarray | None, int, np.ndarray | None, np.ndarray | None, np.ndarray | None
]


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
        "Iterates over all frames of a clip as images, rendered by vspipe.",
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
    ],
    icon="MdVideoCameraBack",
    inputs=[
        VideoClipInput("Clip"),
        EnumInput(
            VideoMode,
            "Mode",
            default=VideoMode.RGB,
            option_labels=VIDEO_MODE_LABELS,
        )
        .with_id(1)
        .with_docs(
            "Planes gives the planes as stored: Y with 0 black and 1 white, U and V"
            " centred on 0.5, values outside 0-1 kept (the clip's range and bit depth"
            " undone). Planes and Both need a YUV clip.",
        ),
    ],
    outputs=[
        ImageOutput("Frame").null_unless(1, "VideoMode::Rgb | VideoMode::Both"),
        NumberOutput(
            "Index",
            output_type="min(uint, max(0, IterOutput0.length - 1))",
        ).with_docs("A counter that starts at 0 and increments by 1 for each frame."),
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
    ],
    iterator_outputs=IteratorOutputInfo(outputs=[0, 1, 3, 4, 5]).with_item_types(
        FramesInfo, get_item_types
    ),
    node_context=True,
    side_effects=True,
    kind="generator",
)
def clip_frames_node(
    context: NodeContext, clip: VideoClip, mode: VideoMode
) -> tuple[Generator[Item], float, ColourInfo]:
    script = write_script(context, [(clip, "Clip")], chain_dir(context) / "frames.vpy")
    info, props = clip_props(script, 0, context)
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
    # Float samples as they are (VapourSynth's float chroma is centred on 0).
    float_samples = info.format_name.endswith(("H", "S"))

    def values(plane: np.ndarray, chroma: bool) -> np.ndarray:
        if float_samples:
            return plane + np.float32(0.5) if chroma else plane
        return code_to_float(plane, colour.bit_depth, colour.full_range, chroma)

    def items() -> Iterator[Item]:
        # The info above: vspipe evaluates the script once more, for the frames.
        if mode is VideoMode.RGB:
            frames = read_frames(
                script, 0, 0, info.num_frames, info=info, context=context
            )
            # Stops vspipe when the run ends before the last frame.
            context.add_cleanup(frames.close)
            for index, frame in enumerate(frames):
                yield frame, index, None, None, None
            return
        planes = read_planes(
            script,
            0,
            0,
            info.num_frames,
            info=info,
            rgb=mode is VideoMode.BOTH,
            context=context,
        )
        context.add_cleanup(planes.close)
        for index, (frame, y, u, v) in enumerate(planes):
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

    generator = Generator.from_iter(
        supplier=frames, expected_length=info.num_frames
    ).with_metadata(FramesInfo(info, mode, colour))
    # Frames come from vspipe's pipe, as Load Video's come from FFmpeg's.
    generator.source_paths = ()
    return generator, info.fps_num / info.fps_den, colour
