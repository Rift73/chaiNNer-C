# `x as x` imports: the native mirror reads these names in native/src/video_io.cpp
from __future__ import annotations

import itertools
import math
import os
import subprocess as subprocess
import time
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from functools import cache, lru_cache
from io import BufferedIOBase as BufferedIOBase
from pathlib import Path
from threading import Lock, Timer
from types import ModuleType
from typing import TYPE_CHECKING, Generator, Iterator

import ffmpeg as ffmpeg

# Also read as "np" by global(globals_, "np") in native/src/video_io.cpp
import numpy as np
from sanic.log import logger as logger

from ..utils.utils import file_identity
from .colour_info import (
    MATRIX_LABELS,
    SWSCALE_CHROMA_POSITION,
    ChromaLocation,
    ColourInfo,
    Matrix,
    PixelLayout,
    Primaries,
    Range,
    Transfer,
    VideoMode,
    code_to_float,
    detect_colour_info,
    pixel_layout,
    swscale_matrix,
)
from .ffmpeg import FFMpegEnv
from .native_framework_images import planes_to_interleaved
from .native_graph import graph

if TYPE_CHECKING:
    from fractions import Fraction

    from av.container import InputContainer
    from av.filter import Graph
    from av.rational import AVRational
    from av.video.frame import VideoFrame
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


@dataclass(frozen=True)
class ReadSettings:
    """What Load Video reads: `mode`, RGB as float32 (`float_rgb`, else 8-bit as before)
    and the five colour choices (ColourInfo.with_choices's arguments)."""

    mode: VideoMode
    float_rgb: bool
    choices: tuple[Matrix, Transfer, Primaries, Range, ChromaLocation]


@dataclass(frozen=True)
class FrameFormat:
    """One conversion a reader applies to every frame: the scale filter `scale` (its
    arguments; None for no scale filter), then `pix_fmt`, after the display matrix's
    rotation when `upright`. A `size` (width, height) frame arrives as `planes`, (rows,
    bytes per row) each, one after another without padding, as the CLI's rawvideo pipe
    writes them; planes are coded as `colour` says."""

    pix_fmt: str
    scale: str | None
    upright: bool
    size: tuple[int, int]
    planes: tuple[tuple[int, int], ...]
    colour: ColourInfo | None = None

    @property
    def frame_bytes(self) -> int:
        return sum(rows * row_bytes for rows, row_bytes in self.planes)

    @property
    def input_options(self) -> dict[str, int]:
        """The CLI reader's input options: frames stay as stored unless `upright`."""
        return {} if self.upright else {"autorotate": 0}

    def unpack(self, data: np.ndarray) -> object:
        """A frame's bytes (uint8) as its values: a BGR uint8 image (bgr24), a BGR float32
        image (gbrpf32le) or the list of planes, float32 Y, U and V."""
        if data.size != self.frame_bytes:
            raise RuntimeError(
                f"FFmpeg's frame ended after {data.size} of {self.frame_bytes} bytes."
            )
        width, height = self.size
        if self.pix_fmt == "bgr24":
            return data.reshape(height, width, 3)
        if self.pix_fmt == "gbrpf32le":
            g, b, r = data.view(np.float32).reshape(3, height, width)
            return planes_to_interleaved((b, g, r))
        colour = self.colour
        assert colour is not None
        dtype = np.dtype(np.uint8 if colour.bit_depth <= 8 else "<u2")
        planes: list[np.ndarray] = []
        offset = 0
        for index, (rows, row_bytes) in enumerate(self.planes):
            codes = data[offset : offset + rows * row_bytes].view(dtype)
            planes.append(
                code_to_float(
                    codes.reshape(rows, row_bytes // dtype.itemsize),
                    colour.bit_depth,
                    colour.full_range,
                    chroma=index > 0,
                )
            )
            offset += rows * row_bytes
        return planes


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


def _reader_pyav() -> ModuleType | None:
    """PyAV for Load Video's readers, or None when the FFmpeg CLI reads every file:
    CHAINNER_C_VIDEO_READER=cli, or PyAV missing."""
    return None if os.environ.get("CHAINNER_C_VIDEO_READER") == "cli" else _pyav()


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
    frame_format: FrameFormat,
):
    """buffer -> the upright filters -> scale -> the format -> buffersink, as the CLI
    reader converts (video_io.cpp FrameIterator)."""
    graph = av.filter.Graph()
    chain = [graph.add_buffer(template=stream)]
    chain += [graph.add(name, arguments) for name, arguments in filters]
    if frame_format.scale is not None:
        chain.append(graph.add("scale", frame_format.scale))
    chain += [graph.add("format", frame_format.pix_fmt), graph.add("buffersink")]
    for upstream, downstream in itertools.pairwise(chain):
        upstream.link_to(downstream)
    graph.configure()
    return graph


def _free_converter(converter: Graph) -> None:
    """Lets a converter's filter graph go with its last reference, not at a later
    collection. PyAV's Graph and its FilterContexts refer to each other (the graph keeps
    its contexts in dicts, each context keeps .graph), and a successful run never
    collects, so the graph and the frames its filters pool (31.7 MiB for a 720p video)
    stayed until some collection. PyAV has no call that frees a graph or drops its
    contexts; the dicts are among its referents, so they are emptied here, once the
    reader is done with the graph."""
    # Imported here, as _pyav imports av: test_video_io.py runs this module's functions
    # and classes in a namespace of its own, without the module's imports.
    import gc

    for referent in gc.get_referents(converter):
        if isinstance(referent, dict):
            referent.clear()


def _frame_bytes(frame: VideoFrame, frame_format: FrameFormat) -> np.ndarray:
    """A converted frame's planes without their row padding, as the CLI pipes them."""
    data = np.empty(frame_format.frame_bytes, np.uint8)
    offset = 0
    for plane, (rows, row_bytes) in zip(frame.planes, frame_format.planes):
        source = np.lib.stride_tricks.as_strided(
            np.frombuffer(plane, np.uint8), (rows, row_bytes), (plane.line_size, 1)
        )
        data[offset : offset + rows * row_bytes].reshape(rows, row_bytes)[:] = source
        offset += rows * row_bytes
    data.setflags(write=False)
    return data


def _frame_values(frame: VideoFrame, frame_format: FrameFormat) -> object:
    """A converted frame's values, as unpack gives them for the CLI's bytes. A float
    RGB frame's G, B and R planes go straight into the BGR image, without the bytes."""
    if frame_format.pix_fmt != "gbrpf32le":
        return frame_format.unpack(_frame_bytes(frame, frame_format))
    width, height = frame_format.size
    green, blue, red = (
        np.lib.stride_tricks.as_strided(
            np.frombuffer(plane, np.float32), (height, width), (plane.line_size, 4)
        )
        for plane in frame.planes
    )
    return planes_to_interleaved((blue, green, red))


def _variable_timestamps(
    path: Path, rate: Fraction | AVRational, av: ModuleType
) -> bool:
    """Whether the first video track's frames leave the grid of `rate` by half a frame
    or more: its packets' timestamps, demuxed without decoding and sorted (B-frames
    store them out of order). The scan reads the whole file, so its answer is kept for
    the file's version (file_identity) and `rate`: a file read again is not scanned
    again until it changes."""
    return _timestamps_vary(file_identity(path), path, rate, av)


@lru_cache(maxsize=64)
def _timestamps_vary(
    _identity: tuple, path: Path, rate: Fraction | AVRational, av: ModuleType
) -> bool:
    """_variable_timestamps's scan; `_identity` is the file version it is kept for."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stamps = sorted(p.pts for p in container.demux(stream) if p.pts is not None)
        step = float(1 / (rate * stream.time_base))
    return any(
        abs(pts - stamps[0] - index * step) >= step / 2
        for index, pts in enumerate(stamps)
    )


def _variable_rate(path: Path, container: InputContainer, av: ModuleType) -> bool:
    """Whether the CLI reader would resample the first video track's frames: it makes a
    variable frame rate constant (rawvideo output has no timestamps), duplicating or
    dropping frames, where PyAV yields them as stored. Matroska and WebM report the
    track's default frame duration as both rates, so their timestamps decide there."""
    stream = container.streams.video[0]
    rate = stream.average_rate
    return (
        # None (no rate known), which PyAV's stubs leave out.
        not rate
        or rate != stream.guessed_rate
        or (
            "matroska" in container.format.name.split(",")
            and _variable_timestamps(path, rate, av)
        )
    )


@dataclass(frozen=True)
class FrameIndex:
    """A video's frames as PyAV's reader numbers them: `pts`, every frame's presentation
    timestamp (the first video track's time base) in display order, so frame n is the
    full read's item n; `keyframes`, the numbers of the frames a decoder can start at."""

    pts: tuple[int, ...]
    keyframes: tuple[int, ...]


def frame_index(path: Path) -> FrameIndex:
    """The first video track's FrameIndex, from its packets demuxed without decoding and
    sorted (B-frames store them out of order). A packet flagged as discarded (the frames
    an MP4's edit list hides) is decoded but never shown, so it is left out. The scan
    reads the whole file, so its index is kept for the file's version (file_identity),
    as _variable_timestamps keeps its answer."""
    av = _pyav()
    if av is None:
        raise RuntimeError(f"{path} has no frame index: PyAV is unavailable.")
    return _scan_frames(file_identity(path), path, av)


@lru_cache(maxsize=64)
def _scan_frames(_identity: tuple, path: Path, av: ModuleType) -> FrameIndex:
    """frame_index's scan; `_identity` is the file version it is kept for."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = sorted(
            (packet.pts, packet.is_keyframe)
            for packet in container.demux(stream)
            if packet.pts is not None and not packet.is_discard
        )
    return FrameIndex(
        tuple(pts for pts, _ in frames),
        tuple(n for n, (_, keyframe) in enumerate(frames) if keyframe),
    )


def _past_the_end(n: int, count: int) -> ValueError:
    return ValueError(f"frame {n} is past the end (the video has {count} frames)")


def _before_the_start(n: int) -> ValueError:
    return ValueError(f"frame {n} is before the start (the first frame is 0)")


def _decode(
    container: InputContainer, stream: VideoStream, path: Path, av: ModuleType
) -> Generator[VideoFrame, None, None]:
    """The track's frames decoded from where the container is (its start or a seek)."""
    for packet in container.demux(stream):
        try:
            frames = packet.decode()
        except av.error.InvalidDataError as error:
            # The CLI skips a broken packet too (no -xerror).
            logger.warning("Skipping undecodable data in %s: %s", path, error)
            continue
        yield from frames


# The keyframes _seek starts from, nearest first, before it decodes from the file's start.
_SEEK_TRIES = 3


def _from_timestamp(
    frames: Iterator[VideoFrame], target: int
) -> Generator[VideoFrame, None, bool]:
    """`frames` from the one with timestamp `target` on, and True; False, with none
    yielded, when they pass the timestamp or end first."""
    for frame in frames:
        if frame.pts is not None and frame.pts >= target:
            if frame.pts != target:
                return False
            yield frame
            yield from frames
            return True
    return False


def _seek(
    container: InputContainer,
    stream: VideoStream,
    path: Path,
    av: ModuleType,
    n: int,
) -> Generator[VideoFrame, None, None]:
    """The track's frames decoded from frame `n` (frame_index's number) on, as the full
    read decodes them: a seek to the nearest keyframe at or before it, then decoding up
    to its timestamp. When the decoded frames pass the timestamp, the seek starts again
    from the keyframe before (nearest first): a decoder started at an open GOP's
    keyframe skips the GOP's leading frames, which refer to the GOP before, and a
    demuxer that seeks by decoding timestamps (MPEG-TS) lands past the keyframe, so the
    decoder starts at the next one. After _SEEK_TRIES keyframes, or for a frame before
    the index's first keyframe (an edit list hides the one before it), the file is
    decoded from its start in a container of its own, as the full read decodes it."""
    index = frame_index(path)
    if n >= len(index.pts):
        raise _past_the_end(n, len(index.pts))
    target = index.pts[n]
    starts = [index.pts[k] for k in reversed(index.keyframes) if index.pts[k] <= target]
    for start in starts[:_SEEK_TRIES]:
        container.seek(start, stream=stream, backward=True, any_frame=False)
        frames = _decode(container, stream, path, av)
        if (yield from _from_timestamp(frames, target)):
            return
        frames.close()
    with av.open(str(path)) as again:
        stream = again.streams.video[0]
        stream.thread_type = "AUTO"
        frames = _decode(again, stream, path, av)
        try:
            if (yield from _from_timestamp(frames, target)):
                return
        finally:
            frames.close()
    raise RuntimeError(f"Frame {n} of {path} (timestamp {target}) was never decoded.")


class _Converters:
    """One decoder's conversions to each of `formats`, configured from its first frame,
    as the CLI configures its filters (the display matrix's upright filters)."""

    def __init__(
        self,
        path: Path,
        stream: VideoStream,
        formats: list[FrameFormat],
        av: ModuleType,
    ):
        self.path = path
        self.stream = stream
        self.formats = formats
        self.av = av
        # The filter graphs (_converter), one per format, once the first frame came.
        self.graphs = None
        self.first = True

    def values(self, frame: VideoFrame) -> list[object]:
        """The frame's values in each format, as the CLI reader unpacks them. Raises
        _UseCliReader on the first frame of a file the CLI would treat differently."""
        if self.graphs is None:
            display = self.av.sidedata.sidedata.Type.DISPLAYMATRIX
            side = frame.side_data.get(display)
            rotation = None if side is None else np.frombuffer(bytes(side), "<i4")
            upright = _upright_filters(rotation)
            self.graphs = [
                _converter(self.av, self.stream, upright if f.upright else [], f)
                for f in self.formats
            ]
        values: list[object] = []
        for frame_format, convert in zip(self.formats, self.graphs):
            convert.push(frame)
            converted = convert.pull()
            size = (converted.width, converted.height)
            if size != frame_format.size:
                message = (
                    f"a {size[0]}x{size[1]} frame where"
                    f" {frame_format.size[0]}x{frame_format.size[1]} was expected"
                )
                if self.first:
                    raise _UseCliReader(message)
                raise RuntimeError(f"Video {self.path} changed size: {message}")
            if any(plane.line_size < 0 for plane in converted.planes):
                # The vflip filter a vertically flipped display matrix asks for hands a
                # frame it passes through unconverted (the file's own pixel format)
                # bottom-up: a negative row stride, whose rows lie outside the plane's
                # buffer.
                message = "its display matrix flips the frames vertically"
                if self.first:
                    raise _UseCliReader(message)
                raise RuntimeError(f"Video {self.path}: {message}")
            values.append(_frame_values(converted, frame_format))
        self.first = False
        return values

    def free(self) -> None:
        """Lets the filter graphs go now (_free_converter)."""
        for convert in self.graphs or ():
            _free_converter(convert)
        self.graphs = None


def _pyav_frames(
    path: Path, formats: list[FrameFormat], av: ModuleType, start: int = 0
) -> Generator[list[object], None, None]:
    """Frames decoded in-process from frame `start` (frame_index's number) on, each
    converted to every one of `formats` and unpacked: the values the CLI reader gives,
    FFmpeg's decoder and the same libswscale conversion without the pipe. Raises
    _UseCliReader before its first frame for a file the CLI would treat differently."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if _variable_rate(path, container, av):
            raise _UseCliReader("its frame rate is variable")
        stream.thread_type = "AUTO"
        frames = (
            _decode(container, stream, path, av)
            if start == 0
            else _seek(container, stream, path, av, start)
        )
        converters = _Converters(path, stream, formats, av)
        try:
            for frame in frames:
                yield converters.values(frame)
        finally:
            frames.close()
            converters.free()


def _resume(
    first: list[object], rest: Generator[list[object], None, None]
) -> Iterator[list[object]]:
    try:
        yield first
        yield from rest
    finally:
        rest.close()


def _close(iterator: Iterator[object]) -> None:
    close = getattr(iterator, "close", None)
    if close is not None:
        close()


def _zip_readers(readers: list[Iterator[object]]) -> Iterator[list[object]]:
    """The CLI readers' frames side by side (one FFmpeg per format)."""
    try:
        for values in zip(*readers):
            yield list(values)
    finally:
        for reader in readers:
            _close(reader)


def _discarding(frames: Iterator[list[object]], start: int) -> Iterator[list[object]]:
    """The CLI readers' frames from frame `start` on. The CLI renumbers the frames to a
    constant rate, which a seek's timestamp would not match, so the frames before it
    are decoded and discarded."""
    try:
        for read in range(start):
            if next(frames, None) is None:
                raise _past_the_end(start, read)
        first = next(frames, None)
        if first is None:
            raise _past_the_end(start, start)
        yield first
        yield from frames
    finally:
        _close(frames)


def _first(frames: Iterator[list[object]], count: int) -> Iterator[list[object]]:
    """The first `count` frames; the reader closes after them, not at its end."""
    try:
        yield from itertools.islice(frames, max(count, 0))
    finally:
        _close(frames)


def _only(frames: Iterator[list[object]]) -> Iterator[np.ndarray]:
    """The frames of a reader with one format."""
    try:
        for (frame,) in frames:
            assert isinstance(frame, np.ndarray)
            yield frame
    finally:
        _close(frames)


def _items(
    frames: Iterator[list[object]], rgb: bool, planes: bool
) -> Iterator[
    tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]
]:
    """Each frame's values (RGB first, then the planes) as (RGB frame, Y, U, V)."""
    try:
        for values in frames:
            frame = values[0] if rgb else None
            assert frame is None or isinstance(frame, np.ndarray)
            plane_values = values[-1] if planes else [None, None, None]
            assert isinstance(plane_values, list)
            y, u, v = plane_values
            yield frame, y, u, v
    finally:
        _close(frames)


class VideoLoader:
    # Both readers' libswscale flags (native/src/video_io.cpp FrameIterator).
    SWS_FLAGS = "lanczos+accurate_rnd+full_chroma_int+full_chroma_inp+bitexact"

    def __init__(
        self, path: Path, ffmpeg_env: FFMpegEnv, settings: ReadSettings | None = None
    ):
        graph().video_loader_init(globals(), self, path, ffmpeg_env)
        self.settings = settings
        self.colour: ColourInfo | None = None
        self.formats: list[FrameFormat] = []
        if settings is not None:
            # The video stream video_loader_init probed.
            stream: dict[str, object] = vars(self)["video_stream"]
            colour, warnings = detect_colour_info(stream)
            if colour.transfer in (Transfer.PQ, Transfer.HLG) and colour.hdr is None:
                # HEVC and AV1 carry HDR10 metadata in the bitstream, which ffprobe
                # reports on frames only: the first frame's. Without it, the video is
                # read with no HDR10 metadata. The probe is kept for the file's version,
                # as video_loader_init's is (video_io.cpp probe_file).
                try:
                    probe = graph().video_probe(
                        globals(),
                        str(path),
                        ffmpeg_env.ffprobe,
                        select_streams="v:0",
                        read_intervals="%+#1",
                        show_frames=None,
                    )
                except ffmpeg.Error as error:
                    message = (
                        str(error)
                        if error.stderr is None
                        else error.stderr.decode("utf-8", "replace").strip()
                    )
                    logger.warning(
                        "%s: its first frame's HDR10 metadata could not be read, so it"
                        " is left out: %s",
                        path,
                        message,
                    )
                    probe = {}
                frames = probe.get("frames") or [{}]
                frame_side_data = frames[0].get("side_data_list") or []
                if frame_side_data:
                    stream_side_data = stream.get("side_data_list")
                    side_data = [
                        *(
                            stream_side_data
                            if isinstance(stream_side_data, list)
                            else []
                        ),
                        *frame_side_data,
                    ]
                    stream = {**stream, "side_data_list": side_data}
                    colour, warnings = detect_colour_info(stream)
            for warning in warnings:
                logger.warning("%s: %s", path, warning)
            self.colour = colour.with_choices(*settings.choices)
            self.formats = self._formats(path, stream, self.colour, settings)

    def _formats(
        self,
        path: Path,
        stream: dict[str, object],
        colour: ColourInfo,
        settings: ReadSettings,
    ) -> list[FrameFormat]:
        """The RGB format (RGB and Both) and the planes format (Planes and Both)."""
        metadata: VideoMetadata = vars(self)["metadata"]
        layout = pixel_layout(str(stream.get("pix_fmt") or ""))
        formats: list[FrameFormat] = []
        if settings.mode is not VideoMode.PLANES:
            chosen = any(choice.value != "auto" for choice in settings.choices)
            if settings.float_rgb or chosen:
                formats.append(self._rgb_format(metadata, layout, colour, settings))
            else:
                formats.append(self.legacy_format())
        if settings.mode is not VideoMode.RGB:
            if layout.kind == "rgb" or colour.matrix is Matrix.RGB:
                raise ValueError(
                    f"{path.name} is RGB-coded, so it has no Y, U and V planes; read it"
                    " in RGB mode."
                )
            if layout.kind == "gray":
                raise ValueError(
                    f"{path.name} is gray video, so it has no U and V planes; read it in"
                    " RGB mode."
                )
            if layout.planar is None:
                raise ValueError(
                    f"Planes mode cannot read {path.name}'s pixel format"
                    f" {stream.get('pix_fmt')}; read it in RGB mode."
                )
            side_data = stream.get("side_data_list")
            if isinstance(side_data, list) and any(
                entry.get("side_data_type") == "Display Matrix"
                and entry.get("rotation") not in (None, 0)
                for entry in side_data
            ):
                logger.warning(
                    "%s: its planes are read as stored, without the rotation the file"
                    " asks for.",
                    path,
                )
            (width, height), (chroma_width, chroma_height) = (
                colour.luma_size,
                colour.chroma_size,
            )
            sample = 1 if colour.bit_depth <= 8 else 2
            planes = (
                (height, width * sample),
                *((chroma_height, chroma_width * sample),) * 2,
            )
            formats.append(
                FrameFormat(
                    layout.planar, None, False, colour.luma_size, planes, colour
                )
            )
        return formats

    @staticmethod
    def _rgb_format(
        metadata: VideoMetadata,
        layout: PixelLayout,
        colour: ColourInfo,
        settings: ReadSettings,
    ) -> FrameFormat:
        """Float32 RGB (gbrpf32le) or 8-bit RGB (bgr24), upright, converted by the colour
        information's matrix and range; float also by its chroma siting (libswscale
        otherwise assumes centre siting, as 8-bit RGB did before RGB Precision)."""
        width, height = metadata.width, metadata.height
        arguments = []
        if layout.kind != "rgb":
            # libswscale would convert BT.2020 constant luminance as non-constant.
            matrix = swscale_matrix(colour.matrix.value)
            if matrix is None or colour.matrix is Matrix.BT2020_CL:
                raise ValueError(
                    f"RGB mode cannot convert the {MATRIX_LABELS[colour.matrix]} matrix;"
                    " read the video in Planes mode and convert it with Planes to RGB."
                )
            arguments += [
                f"in_color_matrix={matrix}",
                f"in_range={colour.range.value}",
            ]
            if settings.float_rgb:
                position = SWSCALE_CHROMA_POSITION[colour.chroma_location]
                arguments += [
                    f"in_h_chr_pos={position[0]}",
                    f"in_v_chr_pos={position[1]}",
                ]
        scale = ":".join([*arguments, f"flags={VideoLoader.SWS_FLAGS}"])
        if settings.float_rgb:
            planes = ((height, width * 4),) * 3
            return FrameFormat("gbrpf32le", scale, True, (width, height), planes)
        planes = ((height, width * 3),)
        return FrameFormat("bgr24", scale, True, (width, height), planes)

    def legacy_format(self) -> FrameFormat:
        """bgr24 as Load Video read it before its Mode: FFmpeg's conversion by the file's
        tags, and untagged HD video as BT.709. The CLI reader's format None."""
        # metadata and input_matrix (the matrix for untagged HD video) are set by
        # video_loader_init (native/src/video_io.cpp).
        metadata: VideoMetadata = vars(self)["metadata"]
        matrix: str | None = vars(self)["input_matrix"]
        flags = self.SWS_FLAGS
        scale = (
            f"flags={flags}"
            if matrix is None
            else f"in_color_matrix={matrix}:flags={flags}"
        )
        width, height = metadata.width, metadata.height
        planes = ((height, width * 3),)
        return FrameFormat("bgr24", scale, True, (width, height), planes)

    def preview_format(self) -> FrameFormat:
        """The Preview's format in every Mode: float32 RGB by the loader's colour
        information, as Mode RGB with float RGB Precision reads it."""
        settings = self.settings
        assert settings is not None and self.colour is not None
        metadata: VideoMetadata = vars(self)["metadata"]
        stream: dict[str, object] = vars(self)["video_stream"]
        layout = pixel_layout(str(stream.get("pix_fmt") or ""))
        rgb = ReadSettings(VideoMode.RGB, True, settings.choices)
        return self._rgb_format(metadata, layout, self.colour, rgb)

    def get_audio_stream(self, duration: float | None = None, start: float = 0.0):
        """The file's audio streams, for Save Video's mux; with a `duration` (seconds),
        only that much of them, and from `start` (seconds, the input's -ss) when it is
        past 0. The mux reads the file the stream's input names (its "filename")."""
        if start <= 0:
            return graph().video_audio_stream(globals(), self, duration)
        path: Path = vars(self)["path"]
        cut = {"ss": start} if duration is None else {"ss": start, "t": duration}
        return ffmpeg.input(path, **cut).audio

    def frame_count(self) -> int:
        """The number of frames the readers give: exact where PyAV reads the file (its
        frame_index's), else the probe's estimate (metadata.frame_count), which the
        FFmpeg CLI's renumbered frames can pass or fall short of."""
        path: Path = vars(self)["path"]
        av = _reader_pyav()
        if av is not None:
            try:
                with av.open(str(path)) as container:
                    variable = _variable_rate(path, container, av)
                if not variable:
                    return len(frame_index(path).pts)
            except av.error.FFmpegError as reason:
                # The reader goes to the CLI too (_pyav_reader).
                logger.warning("Counting %s's frames by its probe: %s", path, reason)
        metadata: VideoMetadata = vars(self)["metadata"]
        return metadata.frame_count

    def _pyav_reader(
        self, path: Path, formats: list[FrameFormat], start: int = 0
    ) -> Iterator[list[object]] | None:
        """PyAV's in-process decoder from frame `start` on, or None when the FFmpeg CLI
        reads the file: CHAINNER_C_VIDEO_READER=cli, PyAV missing, or a file PyAV would
        read differently."""
        av = _reader_pyav()
        if av is None:
            return None
        frames = _pyav_frames(path, formats, av, start)
        try:
            first = next(frames)
        except StopIteration:
            return iter(())
        except (_UseCliReader, av.error.FFmpegError) as reason:
            logger.warning("Reading %s through the FFmpeg CLI: %s", path, reason)
            return None
        return _resume(first, frames)

    def stream_frames(self) -> Iterator[np.ndarray]:
        """The frames as bgr24, as Load Video read them before its Mode: PyAV's
        in-process decoder when available, else (or for a file it would read
        differently) the FFmpeg CLI through a pipe. CHAINNER_C_VIDEO_READER=cli
        forces the CLI."""
        # path is set by video_loader_init (native/src/video_io.cpp).
        path: Path = vars(self)["path"]
        frames = self._pyav_reader(path, [self.legacy_format()])
        if frames is None:
            return graph().video_stream_frames(globals(), self)
        return _only(frames)

    def stream_items(
        self, start: int = 0, stop: int | None = None
    ) -> Iterator[
        tuple[
            np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None
        ]
    ]:
        """Load Video's items by its settings: (RGB frame, Y, U, V), None for what its
        Mode leaves out, from the readers stream_frames uses: frames `start` to `stop`
        - 1 (to the end without a `stop`, and as a slice, fewer when the video ends
        first). Frame n is the full read's item n, bit for bit: PyAV seeks to it, the
        CLI decodes and discards the frames before it. The CLI runs one FFmpeg per
        format (two in Both mode)."""
        settings = self.settings
        assert settings is not None
        if start < 0:
            raise _before_the_start(start)
        path: Path = vars(self)["path"]
        legacy = self.legacy_format()
        if self.formats == [legacy] and start == 0:
            frames = _zip_readers([self.stream_frames()])
        else:
            frames = self._pyav_reader(path, self.formats, start)
        if frames is None:
            frames = _zip_readers(
                [
                    graph().video_stream_frames(
                        globals(), self, None if f == legacy else f
                    )
                    for f in self.formats
                ]
            )
            if start > 0:
                frames = _discarding(frames, start)
        if stop is not None:
            frames = _first(frames, stop - start)
        rgb = settings.mode is not VideoMode.PLANES
        planes = settings.mode is not VideoMode.RGB
        return _items(frames, rgb, planes)


class PreviewReader:
    """A node's open reader for its Preview (preview_frame): float32 BGR frames in the
    loader's preview_format, frame n being the full read's frame n. PyAV keeps the
    container, the converter and the decoder's place: a frame in the same or the next
    GOP decodes forward, an earlier or farther one seeks (as stream_items(n, n + 1)
    does). A file the FFmpeg CLI reads keeps its pipe: a later frame reads forward, an
    earlier one reads again from the start. `key` is the file version and the format
    the reader is for."""

    def __init__(self, loader: VideoLoader, key: tuple):
        self.loader = loader
        self.key = key
        self.path: Path = vars(loader)["path"]
        self.format = loader.preview_format()
        self.last_used = time.monotonic()
        self._av = _reader_pyav()
        # PyAV's open file, the conversion of its video track and the decoded frames
        # from the decoder's place on; or the CLI's frames. _next is the next's number.
        self._container = None
        self._converters: _Converters | None = None
        self._decoded: Generator[VideoFrame, None, None] | None = None
        self._piped: Iterator[np.ndarray] | None = None
        self._next = 0

    def frame(self, n: int) -> np.ndarray:
        """Frame `n` (n >= 0). PyAV falls back to the CLI before its first frame, as
        stream_items does."""
        if self._av is not None:
            try:
                return self._decoded_frame(self._av, n)
            except (_UseCliReader, self._av.error.FFmpegError) as reason:
                if self._converters is not None and not self._converters.first:
                    raise
                logger.warning(
                    "Previewing %s through the FFmpeg CLI: %s", self.path, reason
                )
                self.close()
                self._av = None
        return self._piped_frame(n)

    def _decoded_frame(self, av: ModuleType, n: int) -> np.ndarray:
        if self._container is None:
            self._container = av.open(str(self.path))
            if _variable_rate(self.path, self._container, av):
                raise _UseCliReader("its frame rate is variable")
            stream = self._container.streams.video[0]
            stream.thread_type = "AUTO"
            self._converters = _Converters(self.path, stream, [self.format], av)
        converters = self._converters
        assert converters is not None
        index = frame_index(self.path)
        if n >= len(index.pts):
            raise _past_the_end(n, len(index.pts))
        # The keyframes between the decoder's place and the frame.
        keyframes = bisect_right(index.keyframes, n) - bisect_right(
            index.keyframes, self._next
        )
        decoded = self._decoded
        if decoded is None or n < self._next or keyframes > 1:
            if decoded is not None:
                decoded.close()
            decoded = _seek(self._container, converters.stream, self.path, av, n)
            self._decoded = decoded
            self._next = n
        frame = None
        while self._next <= n:
            frame = next(decoded, None)
            if frame is None:
                raise RuntimeError(f"{self.path} ended before its frame {n}.")
            self._next += 1
        assert frame is not None
        (image,) = converters.values(frame)
        assert isinstance(image, np.ndarray)
        return image

    def _piped_frame(self, n: int) -> np.ndarray:
        piped = self._piped
        if piped is None or n < self._next:
            if piped is not None:
                _close(piped)
            piped = graph().video_stream_frames(globals(), self.loader, self.format)
            self._piped = piped
            self._next = 0
        image = None
        while self._next <= n:
            # The CLI renumbers frames to a constant rate: decoded and discarded.
            image = next(piped, None)
            if image is None:
                raise _past_the_end(n, self._next)
            self._next += 1
        assert image is not None
        return image

    def close(self) -> None:
        """Closes the decoder and its file, or the pipe."""
        if self._decoded is not None:
            self._decoded.close()
            self._decoded = None
        if self._converters is not None:
            self._converters.free()
            self._converters = None
        if self._container is not None:
            self._container.close()
            self._container = None
        if self._piped is not None:
            _close(self._piped)
            self._piped = None


# preview_frame keeps a reader for at most PREVIEW_READERS nodes, and closes one idle for
# PREVIEW_IDLE_SECONDS: Windows renames or deletes no file a reader holds open.
PREVIEW_READERS = 4
PREVIEW_IDLE_SECONDS = 60.0
_previews_lock = Lock()
# By node id, the least recently used first.
_previews: OrderedDict[str, PreviewReader] = OrderedDict()
_previews_timer: Timer | None = None


def preview_frame(node_id: str, loader: VideoLoader, n: int) -> np.ndarray:
    """Frame `n` of the loader's video for node `node_id`'s Preview: float32 BGR, the
    frame Mode RGB with float RGB Precision reads (stream_items(n, n + 1)), whatever
    the loader's Mode. The node's PreviewReader stays open, so the frames after it come
    without a seek; it opens again when the node's file or settings change."""
    if n < 0:
        raise _before_the_start(n)
    path: Path = vars(loader)["path"]
    key = (file_identity(path), loader.preview_format())
    with _previews_lock:
        # The reader is this call's alone until it is kept again.
        reader = _previews.pop(node_id, None)
    if reader is not None and reader.key != key:
        reader.close()
        reader = None
    if reader is None:
        reader = PreviewReader(loader, key)
    try:
        image = reader.frame(n)
    except BaseException:
        reader.close()
        raise
    _keep_preview(node_id, reader)
    return image


def close_previews() -> None:
    """Closes every node's preview reader (at a chain's end and at shutdown)."""
    global _previews_timer
    with _previews_lock:
        closing = list(_previews.values())
        _previews.clear()
        if _previews_timer is not None:
            _previews_timer.cancel()
            _previews_timer = None
    _close_readers(closing)


def _keep_preview(node_id: str, reader: PreviewReader) -> None:
    """Keeps `reader` as the node's, closing the one it replaces (a concurrent call's)
    and the least recently used beyond PREVIEW_READERS."""
    global _previews_timer
    reader.last_used = time.monotonic()
    with _previews_lock:
        replaced = _previews.pop(node_id, None)
        closing = [] if replaced is None else [replaced]
        _previews[node_id] = reader
        while len(_previews) > PREVIEW_READERS:
            closing.append(_previews.popitem(last=False)[1])
        if _previews_timer is None:
            _previews_timer = _idle_timer(PREVIEW_IDLE_SECONDS)
    _close_readers(closing)


def _idle_timer(seconds: float) -> Timer:
    timer = Timer(max(seconds, 0.0), _close_idle_previews)
    timer.daemon = True
    timer.start()
    return timer


def _close_idle_previews() -> None:
    """The idle timer: closes the readers idle for PREVIEW_IDLE_SECONDS, and waits for
    the next one to be."""
    global _previews_timer
    now = time.monotonic()
    with _previews_lock:
        idle = [
            node_id
            for node_id, reader in _previews.items()
            if now - reader.last_used >= PREVIEW_IDLE_SECONDS
        ]
        closing = [_previews.pop(node_id) for node_id in idle]
        _previews_timer = None
        if _previews:
            oldest = min(reader.last_used for reader in _previews.values())
            _previews_timer = _idle_timer(oldest + PREVIEW_IDLE_SECONDS - now)
    _close_readers(closing)


def _close_readers(readers: list[PreviewReader]) -> None:
    """Closes each reader; the first error is raised once all have closed."""
    error: Exception | None = None
    for reader in readers:
        try:
            reader.close()
        except Exception as failure:
            error = error or failure
    if error is not None:
        raise error
