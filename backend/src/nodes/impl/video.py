# `x as x` imports: the native mirror reads these names in native/src/video_io.cpp
from __future__ import annotations

import itertools
import math
import os
import subprocess as subprocess
from dataclasses import dataclass
from functools import cache, lru_cache
from io import BufferedIOBase as BufferedIOBase
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Generator, Iterator

import ffmpeg as ffmpeg

# Also read as "np" by global(globals_, "np") in native/src/video_io.cpp
import numpy as np
from sanic.log import logger as logger

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
from .pytorch.resource_cache import file_identity

if TYPE_CHECKING:
    from fractions import Fraction

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


def _variable_timestamps(path: Path, rate: Fraction, av: ModuleType) -> bool:
    """Whether the first video track's frames leave the grid of `rate` by half a frame
    or more: its packets' timestamps, demuxed without decoding and sorted (B-frames
    store them out of order). The scan reads the whole file, so its answer is kept for
    the file's version (file_identity) and `rate`: a file read again is not scanned
    again until it changes."""
    return _timestamps_vary(file_identity(path), path, rate, av)


@lru_cache(maxsize=64)
def _timestamps_vary(
    _identity: tuple, path: Path, rate: Fraction, av: ModuleType
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


def _pyav_frames(
    path: Path, formats: list[FrameFormat], av: ModuleType
) -> Generator[list[object], None, None]:
    """Frames decoded in-process, each converted to every one of `formats` and unpacked:
    the values the CLI reader gives, FFmpeg's decoder and the same libswscale conversion
    without the pipe. Raises _UseCliReader before its first frame for a file the CLI
    would treat differently."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # The CLI resamples a variable frame rate to a constant one (rawvideo output
        # has no timestamps), duplicating or dropping frames; PyAV yields them as stored.
        # Matroska and WebM report the track's default frame duration as both rates, so
        # their timestamps decide there.
        if (
            stream.average_rate is None
            or stream.average_rate != stream.guessed_rate
            or (
                "matroska" in container.format.name.split(",")
                and _variable_timestamps(path, stream.average_rate, av)
            )
        ):
            raise _UseCliReader("its frame rate is variable")
        stream.thread_type = "AUTO"
        display = av.sidedata.sidedata.Type.DISPLAYMATRIX
        converters = None
        first = True
        for packet in container.demux(stream):
            try:
                frames = packet.decode()
            except av.error.InvalidDataError as error:
                # The CLI skips a broken packet too (no -xerror).
                logger.warning("Skipping undecodable data in %s: %s", path, error)
                continue
            for frame in frames:
                if converters is None:
                    # The CLI configures its filters from the first frame too.
                    side = frame.side_data.get(display)
                    rotation = (
                        None if side is None else np.frombuffer(bytes(side), "<i4")
                    )
                    upright = _upright_filters(rotation)
                    converters = [
                        _converter(av, stream, upright if f.upright else [], f)
                        for f in formats
                    ]
                values: list[object] = []
                for frame_format, convert in zip(formats, converters):
                    convert.push(frame)
                    converted = convert.pull()
                    size = (converted.width, converted.height)
                    if size != frame_format.size:
                        message = (
                            f"a {size[0]}x{size[1]} frame where"
                            f" {frame_format.size[0]}x{frame_format.size[1]} was expected"
                        )
                        if first:
                            raise _UseCliReader(message)
                        raise RuntimeError(f"Video {path} changed size: {message}")
                    values.append(_frame_values(converted, frame_format))
                first = False
                yield values


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

    def get_audio_stream(self, duration: float | None = None):
        """The file's audio streams, for Save Video's mux; with a `duration` (seconds),
        only that much of them."""
        return graph().video_audio_stream(globals(), self, duration)

    def _pyav_reader(
        self, path: Path, formats: list[FrameFormat]
    ) -> Iterator[list[object]] | None:
        """PyAV's in-process decoder, or None when the FFmpeg CLI reads the file:
        CHAINNER_C_VIDEO_READER=cli, PyAV missing, or a file PyAV would read
        differently."""
        av = None if os.environ.get("CHAINNER_C_VIDEO_READER") == "cli" else _pyav()
        if av is None:
            return None
        frames = _pyav_frames(path, formats, av)
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
        self,
    ) -> Iterator[
        tuple[
            np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None
        ]
    ]:
        """Load Video's items by its settings: (RGB frame, Y, U, V), None for what its
        Mode leaves out, from the readers stream_frames uses. The CLI runs one FFmpeg
        per format (two in Both mode)."""
        settings = self.settings
        assert settings is not None
        path: Path = vars(self)["path"]
        legacy = self.legacy_format()
        if self.formats == [legacy]:
            frames = _zip_readers([self.stream_frames()])
        else:
            frames = self._pyav_reader(path, self.formats)
        if frames is None:
            frames = _zip_readers(
                [
                    graph().video_stream_frames(
                        globals(), self, None if f == legacy else f
                    )
                    for f in self.formats
                ]
            )
        rgb = settings.mode is not VideoMode.PLANES
        planes = settings.mode is not VideoMode.RGB
        return _items(frames, rgb, planes)
