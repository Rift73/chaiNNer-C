# `x as x` imports: the native mirror reads these names in native/src/video_io.cpp
import subprocess as subprocess
from dataclasses import dataclass
from io import BufferedIOBase as BufferedIOBase
from pathlib import Path
from typing import Iterator

import ffmpeg as ffmpeg

# Also read as "np" by global(globals_, "np") in native/src/video_io.cpp
import numpy as np
from sanic.log import logger as logger

from .ffmpeg import FFMpegEnv
from .native_graph import graph


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int

    @staticmethod
    def from_file(path: Path, ffmpeg_env: FFMpegEnv):
        return graph().video_metadata(globals(), path, ffmpeg_env)


class VideoLoader:
    def __init__(self, path: Path, ffmpeg_env: FFMpegEnv):
        graph().video_loader_init(globals(), self, path, ffmpeg_env)

    def get_audio_stream(self):
        return graph().video_audio_stream(globals(), self)

    def stream_frames(self) -> Iterator[np.ndarray]:
        return graph().video_stream_frames(globals(), self)
