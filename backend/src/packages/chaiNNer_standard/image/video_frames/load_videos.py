from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from api import Generator, InputId, IteratorOutputInfo, NodeContext, OutputId
from nodes.groups import Condition, if_group
from nodes.impl.colour_info import (
    ChromaLocation,
    ColourInfo,
    Matrix,
    Primaries,
    Range,
    Transfer,
    VideoMode,
)
from nodes.properties.inputs import (
    BoolInput,
    DirectoryInput,
    NumberInput,
    TextInput,
    VideoFileInput,
)
from nodes.properties.outputs import (
    AudioStreamOutput,
    DirectoryOutput,
    FileNameOutput,
    ImageOutput,
    NumberOutput,
    TextOutput,
)

from .. import video_frames_group
from ..batch_processing.load_images import list_glob
from .load_video import (
    RGB_MODES,
    FramesMode,
    LoadedVideo,
    RgbPrecision,
    get_item_types,
    later_outputs,
    load_video_node,
    read_inputs,
)

# What makes Load Video read a whole file, besides its path, in a run's per-file chain
# (folder_videos.py): its inputs that Load Videos does not have.
WHOLE_VIDEO: dict[InputId, object] = {
    InputId(13): 0,
    InputId(10): FramesMode.ALL.value,
    InputId(11): 0,
    InputId(12): 0,
}
VIDEO_FILE = InputId(0)
INDEX = OutputId(1)
VIDEO_DIRECTORY = OutputId(2)
NAME = OutputId(3)
SUBDIRECTORY = OutputId(13)
VIDEO_COUNT = OutputId(14)
# The inputs the file list follows from, in list_videos' order, are the node's first.
LISTING_INPUTS = 6


def list_videos(
    directory: Path,
    use_glob: bool,
    is_recursive: bool,
    glob_str: str,
    use_limit: bool,
    limit: int,
) -> list[Path]:
    """The folder's video files in Load Images' order. None is an error."""
    expression = glob_str if use_glob else ("**/*" if is_recursive else "*")
    listed = list_glob(directory, expression, VideoFileInput().filetypes)
    # A folder can be named like a video.
    paths = [path for path in listed if path.is_file()]
    if not paths:
        raise FileNotFoundError(f"{directory} has no video files.")
    return paths[:limit] if use_limit else paths


def subdirectory_of(path: Path, directory: Path) -> str:
    """The Subdirectory Path: the file's folder relative to the Directory, as Load Images
    gives it."""
    return os.path.relpath(path.parent, directory)


@video_frames_group.register(
    schema_id="chainner:image:load_videos",
    name="Load Videos",
    description=[
        "Iterate over all the video files in a directory/folder: the nodes it feeds run once per video, from its first frame to its last, each with that video's Name, FPS, audio and colour, so that a Save Video gives each one its own file.",
        "Connect Name to Save Video's Video Name. Files with one name (a.mp4 and a.mkv, or the same name in two subdirectories with Recursive) need more than Name to tell their outputs apart: a run that would write two videos to one file fails when it starts.",
        "The other nodes of the chain run once: a model is loaded once, for all the videos.",
    ],
    see_also=["chainner:image:load_video", "chainner:image:load_images"],
    icon="MdVideoCameraBack",
    inputs=[
        DirectoryInput(),
        BoolInput("Use WCMatch glob expression", default=False).with_id(14),
        if_group(Condition.bool(14, False))(
            BoolInput("Recursive")
            .with_docs("Iterate recursively over subdirectories.")
            .with_id(15)
        ),
        if_group(Condition.bool(14, True))(
            TextInput("WCMatch Glob expression", default="**/*")
            .with_docs(
                "For information on how to use WCMatch glob expressions, see [here](https://facelessuser.github.io/wcmatch/glob/)."
            )
            .with_id(16),
        ),
        BoolInput("Use limit", default=False).with_id(17),
        if_group(Condition.bool(17, True))(
            NumberInput("Limit", default=10, min=1)
            .with_docs(
                "Limit the number of videos to iterate over. This can be useful for testing the chain without having to iterate over all videos."
            )
            .with_id(18)
        ),
        BoolInput("Stop on first error", default=False)
        .with_docs(
            "Instead of going on with the next video when one fails (a damaged file, or a node's error on one of its frames) and showing the errors at the end, stop and show the error as soon as one occurs. A video that fails leaves no output file.",
            hint=True,
        )
        .with_id(19),
        *read_inputs(),
    ],
    outputs=[
        ImageOutput("Frame", channels=3).null_unless(3, RGB_MODES),
        NumberOutput(
            "Index", output_type="min(uint, max(0, IterOutput0.length - 1))"
        ).with_docs("The frame's number in its video: 0 for its first frame."),
        DirectoryOutput("Video Directory"),
        FileNameOutput("Name"),
        NumberOutput("FPS", output_type="0.."),
        AudioStreamOutput().suggest(),
        *later_outputs(
            "The first frame of the video, in RGB, whatever the Mode. Before a run, of the"
            " first video."
        ),
        TextOutput("Subdirectory Path")
        .with_id(SUBDIRECTORY)
        .with_docs("The video's folder relative to the Directory."),
        NumberOutput("Video Count", output_type="uint")
        .with_id(VIDEO_COUNT)
        .with_docs("The number of videos the run goes through."),
    ],
    iterator_outputs=IteratorOutputInfo(
        outputs=[0, 1, 6, 7, 8], length_type="uint"
    ).with_item_types(LoadedVideo, get_item_types),
    node_context=True,
    side_effects=True,
    kind="generator",
)
def load_videos_node(
    node_context: NodeContext,
    directory: Path,
    use_glob: bool,
    is_recursive: bool,
    glob_str: str,
    use_limit: bool,
    limit: int,
    _fail_fast: bool,
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
    str,
    int,
]:
    """The node on its own, as its preview runs it: the folder's first video. A run gives
    each video a Load Video of its own instead (folder_videos.py)."""
    paths = list_videos(directory, use_glob, is_recursive, glob_str, use_limit, limit)
    return (
        *load_video_node(
            node_context,
            paths[0],
            0,
            FramesMode.ALL,
            0,
            0,
            mode,
            precision,
            matrix,
            transfer,
            primaries,
            range_,
            chroma_location,
        ),
        subdirectory_of(paths[0], directory),
        len(paths),
    )
