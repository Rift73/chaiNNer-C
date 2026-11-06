"""A GIF of odd size. GIF has no encoder input, but Simple mode's format and Advanced
mode's hidden Encoder name H.264, whose even-size rule refused the frames with an error
about an encoder the GIF never uses."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from scratch_storage import NO_FFMPEG, STORAGE

# The node's modules load the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from api import NodeContext
from nodes.impl.colour_info import ChromaLocation, Matrix, Primaries, Range, Transfer
from nodes.impl.ffmpeg import get_executable_path
from packages.chaiNNer_standard.image.video_frames.save_video import (
    AudioSettings,
    BitDepth,
    NvencPreset,
    ProResProfile,
    SimpleVideoFormat,
    Simplicity,
    SitingCorrection,
    VideoEncoder,
    VideoFormat,
    VideoInputMode,
    VideoPreset,
    save_video_node,
)

FFMPEG, FFPROBE = map(str, get_executable_path(STORAGE / "ffmpeg"))

pytestmark = pytest.mark.skipif(not Path(FFMPEG).exists(), reason=NO_FFMPEG)


class Context:
    chosen_item = None
    node_id = "save"
    storage_dir = STORAGE

    def add_cleanup(self, _fn: Any) -> None:
        pass


def save(
    folder: Path,
    simplicity: Simplicity,
    container: VideoFormat,
    simple: SimpleVideoFormat,
    size: tuple[int, int],
) -> None:
    width, height = size
    collector = save_video_node(
        cast(NodeContext, Context()),
        None,
        folder,
        "out",
        VideoInputMode.RGB,
        None,
        None,
        None,
        simplicity,
        container,
        VideoEncoder.H264,
        VideoPreset.ULTRA_FAST,
        23,
        *(8, 35, ProResProfile.HQ, NvencPreset.P4, 23, 30),
        None,
        simple,
        75,
        10.0,
        None,
        AudioSettings.AUTO,
        *(None, Matrix.AUTO, Transfer.AUTO, Primaries.AUTO, Range.AUTO),
        *(ChromaLocation.AUTO, BitDepth.AUTO, SitingCorrection.AUTO),
    )
    for value in np.linspace(0, 1, 3, dtype=np.float32):
        collector.on_iterate(
            (np.full((height, width, 3), value, np.float32), None, None, None)
        )
    collector.on_complete()


@pytest.mark.parametrize("simplicity", list(Simplicity))
def test_a_gif_keeps_its_odd_size(tmp_path: Path, simplicity: Simplicity):
    save(tmp_path, simplicity, VideoFormat.GIF, SimpleVideoFormat.GIF, (65, 49))
    probe = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_streams",
            "-of",
            "json",
            str(tmp_path / "out.gif"),
        ],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    assert (stream["codec_name"], stream["width"], stream["height"]) == ("gif", 65, 49)


def test_h264_still_refuses_an_odd_size(tmp_path: Path):
    with pytest.raises(AssertionError, match="even-number frame resolution"):
        save(
            tmp_path,
            Simplicity.ADVANCED,
            VideoFormat.MP4,
            SimpleVideoFormat.MP4_H264,
            (65, 49),
        )
