from __future__ import annotations

from typing import Iterator

import numpy as np

import navi
from api import Generator, IteratorOutputInfo, NodeContext, OutputId
from nodes.impl.vapoursynth.runner import (
    ClipInfo,
    VSScriptError,
    clip_info,
    read_frames,
)
from nodes.impl.vapoursynth.script import VideoClip
from nodes.properties.inputs import VideoClipInput
from nodes.properties.outputs import ImageOutput, NumberOutput

from ..chains import chain_dir, write_script
from . import clips_group


def get_item_types(info: ClipInfo):
    return {
        OutputId(0): navi.Image(
            width=info.width or None,
            height=info.height or None,
            channels=1 if info.format_name.startswith("Gray") else 3,
        ),
    }


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
    ],
    icon="MdVideoCameraBack",
    inputs=[VideoClipInput("Clip")],
    outputs=[
        ImageOutput("Frame"),
        NumberOutput(
            "Index",
            output_type="min(uint, max(0, IterOutput0.length - 1))",
        ).with_docs("A counter that starts at 0 and increments by 1 for each frame."),
        NumberOutput("FPS", output_type="0..").with_docs(
            "The clip's frame rate; 0 when it varies."
        ),
    ],
    iterator_outputs=IteratorOutputInfo(outputs=[0, 1]).with_item_types(
        ClipInfo, get_item_types
    ),
    node_context=True,
    side_effects=True,
    kind="generator",
)
def clip_frames_node(
    context: NodeContext, clip: VideoClip
) -> tuple[Generator[tuple[np.ndarray, int]], float]:
    script = write_script(context, [(clip, "Clip")], chain_dir(context) / "frames.vpy")
    info = clip_info(script, 0, context)

    def frames() -> Iterator[tuple[np.ndarray, int]]:
        # The info above: vspipe evaluates the script once more, for the frames.
        reader = read_frames(script, 0, 0, info.num_frames, info=info, context=context)
        # Stops vspipe when the run ends before the last frame.
        context.add_cleanup(reader.close)
        try:
            for index, frame in enumerate(reader):
                yield frame, index
        except VSScriptError as error:
            # A plugin's error at frame time names no line of the script. The
            # iteration's error arrives without this node, so it goes on it here.
            if error.node_id is None:
                error.node_id = context.node_id
            raise

    generator = Generator.from_iter(
        supplier=frames, expected_length=info.num_frames
    ).with_metadata(info)
    # Frames come from vspipe's pipe, as Load Video's come from FFmpeg's.
    generator.source_paths = ()
    return generator, info.fps_num / info.fps_den
