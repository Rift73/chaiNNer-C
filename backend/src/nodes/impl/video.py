# `x as x` imports: the native mirror reads these names in native/src/video_io.cpp
from __future__ import annotations

import itertools
import math
import os
import subprocess as subprocess
from dataclasses import dataclass
from functools import cache
from io import BufferedIOBase as BufferedIOBase
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Generator, Iterator

import ffmpeg as ffmpeg

# Also read as "np" by global(globals_, "np") in native/src/video_io.cpp
import numpy as np
from sanic.log import logger as logger

from .ffmpeg import FFMpegEnv
from .native_graph import graph

if TYPE_CHECKING:
    from av.video.stream import VideoStream


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


def _upright_filters(matrix: np.ndarray | None) -> list[tuple[str, str | None]]:
    """The filters the FFmpeg CLI inserts for a frame's display matrix, so frames
    come out upright (fftools/ffmpeg_filter.c configure_input_video_filter, with
    cmdutils.c get_rotation and libavutil av_display_rotation_get)."""
    if matrix is None:
        return []
    m = [int(v) for v in matrix]
    a, b, d, e = (v / 65536 for v in (m[0], m[1], m[3], m[4]))
    scale_x, scale_y = math.hypot(a, d), math.hypot(b, e)
    if scale_x == 0 or scale_y == 0:
        return []
    rotation = -(math.atan2(b / scale_y, a / scale_x) * 180 / math.pi)
    theta = -math.copysign(math.floor(abs(rotation) + 0.5), rotation)  # C round()
    theta -= 360 * math.floor(theta / 360 + 0.9 / 360)
    if abs(theta - 90) < 1.0:
        return [("transpose", "cclock_flip" if m[3] > 0 else "clock")]
    if abs(theta - 180) < 1.0:
        flips = [("hflip", m[0] < 0), ("vflip", m[4] < 0)]
        return [(name, None) for name, flip in flips if flip]
    if abs(theta - 270) < 1.0:
        return [("transpose", "clock_flip" if m[3] < 0 else "cclock")]
    if abs(theta) > 1.0:
        return [("rotate", f"{theta:f}*PI/180")]
    return [("vflip", None)] if m[4] < 0 else []


def _converter(
    av: ModuleType,
    stream: VideoStream,
    filters: list[tuple[str, str | None]],
    matrix: str | None,
):
    """buffer -> the upright filters -> scale -> bgr24 -> buffersink, as the CLI
    reader converts: its -sws_flags and, for untagged HD video, its scale filter's
    input matrix (video_io.cpp FrameIterator)."""
    graph = av.filter.Graph()
    flags = "lanczos+accurate_rnd+full_chroma_int+full_chroma_inp+bitexact"
    scale = (
        f"flags={flags}"
        if matrix is None
        else f"in_color_matrix={matrix}:flags={flags}"
    )
    chain = [graph.add_buffer(template=stream)]
    chain += [graph.add(name, arguments) for name, arguments in filters]
    chain += [graph.add("scale", scale), graph.add("format", "bgr24")]
    chain.append(graph.add("buffersink"))
    for upstream, downstream in itertools.pairwise(chain):
        upstream.link_to(downstream)
    graph.configure()
    return graph


def _pyav_frames(
    path: Path, width: int, height: int, matrix: str | None, av: ModuleType
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
        display = av.sidedata.sidedata.Type.DISPLAYMATRIX
        convert = None
        first = True
        for packet in container.demux(stream):
            try:
                frames = packet.decode()
            except av.error.InvalidDataError as error:
                # The CLI skips a broken packet too (no -xerror).
                logger.warning("Skipping undecodable data in %s: %s", path, error)
                continue
            for frame in frames:
                if convert is None:
                    # The CLI configures its filters from the first frame too.
                    side = frame.side_data.get(display)
                    rotation = (
                        None if side is None else np.frombuffer(bytes(side), "<i4")
                    )
                    convert = _converter(av, stream, _upright_filters(rotation), matrix)
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
            # path, metadata and input_matrix (the matrix for untagged HD video) are
            # set by video_loader_init (native/src/video_io.cpp).
            attributes = vars(self)
            path: Path = attributes["path"]
            metadata: VideoMetadata = attributes["metadata"]
            matrix: str | None = attributes["input_matrix"]
            frames = _pyav_frames(path, metadata.width, metadata.height, matrix, av)
            try:
                first = next(frames)
            except StopIteration:
                return iter(())
            except (_UseCliReader, av.error.FFmpegError) as reason:
                logger.warning("Reading %s through the FFmpeg CLI: %s", path, reason)
            else:
                return _resume(first, frames)
        return graph().video_stream_frames(globals(), self)
