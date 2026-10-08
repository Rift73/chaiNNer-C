# `x as x` imports: the native mirror reads these names in native/src/video_io.cpp
from __future__ import annotations

import os
import subprocess as subprocess
from dataclasses import dataclass
from functools import cache
from io import BufferedIOBase as BufferedIOBase
from pathlib import Path
from types import ModuleType
from typing import Generator, Iterator

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


class _UseCliReader(Exception):
    """The file needs the FFmpeg CLI reader; the message says why."""


@cache
def _pyav() -> ModuleType | None:
    """PyAV, or None (warned once per process) when it cannot be imported."""
    try:
        import av
    except ImportError as error:
        logger.warning(
            "PyAV unavailable (%s); videos are read through the FFmpeg CLI", error
        )
        return None
    return av


def _pyav_frames(
    path: Path, width: int, height: int, av: ModuleType
) -> Generator[np.ndarray, None, None]:
    """Frames decoded in-process, the bytes the CLI reader produces: FFmpeg's decoder
    and the same libswscale conversion to bgr24, without the pipe. Raises
    _UseCliReader before its first frame for a file the CLI would treat differently."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # The CLI resamples a variable frame rate to a constant one (rawvideo output
        # has no timestamps), duplicating or dropping frames; PyAV yields them as stored.
        if stream.average_rate is None or stream.average_rate != stream.guessed_rate:
            raise _UseCliReader("its frame rate is variable")
        stream.thread_type = "AUTO"
        convert = av.filter.Graph()
        source = convert.add_buffer(template=stream)
        # The CLI reader's -sws_flags (video_io.cpp FrameIterator).
        flags = "lanczos+accurate_rnd+full_chroma_int+full_chroma_inp+bitexact"
        scale = convert.add("scale", f"flags={flags}")
        bgr = convert.add("format", "bgr24")
        sink = convert.add("buffersink")
        source.link_to(scale)
        scale.link_to(bgr)
        bgr.link_to(sink)
        convert.configure()
        first = True
        for packet in container.demux(stream):
            try:
                frames = packet.decode()
            except av.error.InvalidDataError as error:
                # The CLI skips a broken packet too (no -xerror).
                logger.warning("Skipping undecodable data in %s: %s", path, error)
                continue
            for frame in frames:
                if first and frame.rotation:
                    raise _UseCliReader("it is rotated")
                convert.push(frame)
                image = np.ascontiguousarray(convert.pull().to_ndarray())
                if image.shape != (height, width, 3):
                    message = (
                        f"a {image.shape} frame where {(height, width, 3)} was expected"
                    )
                    if first:
                        raise _UseCliReader(message)
                    raise RuntimeError(f"Video {path} changed size: {message}")
                image.setflags(write=False)
                first = False
                yield image


def _resume(
    first: np.ndarray, rest: Generator[np.ndarray, None, None]
) -> Iterator[np.ndarray]:
    try:
        yield first
        yield from rest
    finally:
        rest.close()


class VideoLoader:
    def __init__(self, path: Path, ffmpeg_env: FFMpegEnv):
        graph().video_loader_init(globals(), self, path, ffmpeg_env)

    def get_audio_stream(self):
        return graph().video_audio_stream(globals(), self)

    def stream_frames(self) -> Iterator[np.ndarray]:
        """PyAV's in-process decoder when available, else (or for a file it would
        read differently) the FFmpeg CLI through a pipe. CHAINNER_C_VIDEO_READER=cli
        forces the CLI."""
        av = None if os.environ.get("CHAINNER_C_VIDEO_READER") == "cli" else _pyav()
        if av is not None:
            # path and metadata are set by video_loader_init (native/src/video_io.cpp).
            attributes = vars(self)
            path: Path = attributes["path"]
            metadata: VideoMetadata = attributes["metadata"]
            frames = _pyav_frames(path, metadata.width, metadata.height, av)
            try:
                first = next(frames)
            except StopIteration:
                return iter(())
            except (_UseCliReader, av.error.FFmpegError) as reason:
                logger.warning("Reading %s through the FFmpeg CLI: %s", path, reason)
            else:
                return _resume(first, frames)
        return graph().video_stream_frames(globals(), self)
