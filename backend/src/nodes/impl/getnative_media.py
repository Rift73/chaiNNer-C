"""Get Native Resolution's sources, as Vernier 0.2's media.cpp makes them: a still image's analysis
plane, and a video's frame candidates (even times, codec keyframes, scene midpoints or typed
times), the frame cap spread over them, and one decoded frame's Y plane per chosen candidate,
with its full-range RGB for the linear-light models.

Frames are read by Load Video's PyAV reader (video.py: its frame index from the packets, demuxed
without decoding, its seek, and its float RGB conversion by the video's colour information), one
frame per candidate; a file whose packets cannot number its frames is read by the FFmpeg CLI as
Vernier reads it, and its keyframes are listed by ffprobe as Vernier lists them. Scene cuts come
from the FFmpeg CLI with Vernier's filter. FFmpeg's executables (`tools`) are looked up only when
one of these needs them."""

from __future__ import annotations

import json
import math
import re
import subprocess
import time
from bisect import bisect_left
from collections.abc import Buffer, Callable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
from sanic.log import logger

from .colour_info import (
    ChromaLocation,
    Matrix,
    Primaries,
    Range,
    Transfer,
    VideoMode,
    detect_colour_info,
    pixel_layout,
)
from .ffmpeg import FFMpegEnv
from .video import (
    FrameFormat,
    ReadSettings,
    VideoLoader,
    VideoMetadata,
    _Converters,
    _pyav,
    _seek,
    _UseCliReader,
    frame_index,
)

# Vernier's native planar formats: the Y plane's code values over the maximum code.
_PLANAR = re.compile(
    r"(?:yuvj?(420|422|444|440|411|410)p|gray)(?:(9|10|12|14|16)(le|be))?"
)
KEYFRAME = "Codec keyframe"
REQUESTED = "Requested time"
SCENE = "Scene midpoint"
BT709_WEIGHTS = (0.2126, 0.7152, 0.0722)
# The luma weights of the matrices the linear-light models convert (media.cpp's loadColor).
_MATRIX_WEIGHTS = {
    Matrix.BT709: BT709_WEIGHTS,
    Matrix.SMPTE170M: (0.299, 0.587, 0.114),
    Matrix.BT470BG: (0.299, 0.587, 0.114),
    Matrix.BT2020_NCL: (0.2627, 0.678, 0.0593),
}


def still_plane(image: np.ndarray) -> tuple[np.ndarray, str]:
    """A still image's As stored plane and its provenance: gray as it is, RGB as BT.709-weighted
    code values (chaiNNer's B, G, R order). Transparency is refused, as Vernier refuses it."""
    if image.ndim == 2 or image.shape[2] == 1:
        plane = image.reshape(image.shape[:2]).astype(np.float64)
        return plane, "Unmodified grayscale samples (0..1)."
    if image.shape[2] == 4:
        if (image[:, :, 3] < 1).any():
            raise ValueError(
                "Transparent images require an explicit compositing background."
                " Composite the image onto a background first."
            )
        image = image[:, :, :3]
    b, g, r = (image[:, :, k].astype(np.float64) for k in range(3))
    return (
        BT709_WEIGHTS[0] * r + BT709_WEIGHTS[1] * g + BT709_WEIGHTS[2] * b,
        (
            "BT.709-weighted RGB code values, without gamma conversion or color-profile"
            " transformation. Original luma extraction is preferable for video."
        ),
    )


def linear_provenance(
    domain: str, weights: tuple[float, float, float], assumptions: str
) -> str:
    """media.cpp's linearFrame provenance of a linear-light plane."""
    return (
        f"Per-channel {domain} inverse transfer, then fixed matrix-matched RGB weights"
        f" ({weights[0]:g},{weights[1]:g},{weights[2]:g}). This linear combination commutes"
        " with a common RGB resampler; it is not inverse transfer of encoded luma. Signed"
        " extension preserves negative values; no clipping. " + assumptions
    )


def capped_indices(count: int, cap: int) -> list[int]:
    """media.cpp's cappedIndices: at most `cap` indices spread evenly over `count` candidates in
    their order, with no repeats or padding."""
    if not 1 <= cap <= 200:
        raise ValueError("Frame cap must be between 1 and 200")
    take = min(count, cap)
    return [
        count // 2 if take == 1 else i * (count - 1) // (take - 1) for i in range(take)
    ]


def parse_times(text: str) -> list[float]:
    """Typed times: seconds or [HH:]MM:SS(.fff), separated by commas, semicolons or spaces."""
    times: list[float] = []
    for token in re.split(r"[,;\s]+", text.strip()):
        if not token:
            continue
        try:
            parts = [float(part) for part in token.split(":")]
        except ValueError as error:
            raise ValueError(
                f"Invalid time {token!r}: use seconds or HH:MM:SS"
            ) from error
        if len(parts) > 3 or any(not math.isfinite(p) or p < 0 for p in parts):
            raise ValueError(f"Invalid time {token!r}: use seconds or HH:MM:SS")
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + part
        times.append(seconds)
    return times


@dataclass(frozen=True)
class SamplePoint:
    """A frame candidate (media.hpp's SamplePoint); `frame` is the frame index's number of a
    codec keyframe."""

    seconds: float
    kind: str
    scene_start: float = -1
    scene_end: float = -1
    frame: int | None = None


@dataclass
class SamplingPlan:
    path: str
    mode: str
    duration: float
    candidates: list[SamplePoint] = field(default_factory=list)


def sampling_report(
    plan: SamplingPlan, cap: int, threshold: float, minimum_scene: float
) -> dict[str, object]:
    """report.cpp's samplingReport for one source (vernier.sampling.v1)."""
    candidates = [
        {
            "seconds": p.seconds,
            "kind": p.kind,
            "scene_start": None if p.scene_start < 0 else p.scene_start,
            "scene_end": None if p.scene_end < 0 else p.scene_end,
        }
        for p in plan.candidates
    ]
    selected = capped_indices(len(plan.candidates), cap)
    return {
        "schema": "vernier.sampling.v1",
        "mode": plan.mode,
        "frame_cap": cap,
        "cap_scope": "Whole analysis, across all source files; light-domain tests reuse the selected frames.",
        "selection_rule": "Evenly distributed indices across all discovered candidates in source order; no unreported top-up.",
        "scene_threshold": threshold,
        "minimum_scene_seconds": minimum_scene,
        "discovered": len(plan.candidates),
        "planned": len(selected),
        "selected_candidate_indices": selected,
        "sources": [
            {
                "path": plan.path,
                "duration": plan.duration,
                "mode": plan.mode,
                "discovered": len(plan.candidates),
                "candidates": candidates,
            }
        ],
    }


@dataclass(frozen=True)
class VideoInfo:
    """The first video track as Vernier probes it: size, native pixel format, the format's start
    time and the track's duration (media.cpp's videoDuration), all in seconds."""

    width: int
    height: int
    pix_fmt: str
    start: float
    duration: float


def _indexed_duration(path: Path, time_base: Fraction, start: float) -> float:
    """The first video track's duration by its frame index, for a container that records
    none per track (Matroska, whose duration ends with its longest track): from the
    format's start time to the end of the last frame (its timestamp plus the step before
    it). 0 when the packets cannot number the frames, or number fewer than two."""
    try:
        pts = frame_index(path).pts
    except _UseCliReader:
        return 0.0
    if len(pts) < 2:
        return 0.0
    return float((2 * pts[-1] - pts[-2]) * time_base) - start


def probe(path: Path) -> VideoInfo:
    av = _pyav()
    if av is None:
        raise RuntimeError("Get Native Resolution needs PyAV to read videos.")
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("No video stream found")
        stream = container.streams.video[0]
        start = (container.start_time or 0) / 1_000_000
        duration = 0.0
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        if not (math.isfinite(duration) and duration > 0) and stream.time_base:
            duration = _indexed_duration(path, Fraction(stream.time_base), start)
        if not (math.isfinite(duration) and duration > 0):
            duration = (container.duration or 0) / 1_000_000
            # Matroska's segment duration is an end timestamp, including any initial gap.
            if "matroska" in container.format.name.split(","):
                duration -= start
        context = stream.codec_context
        info = VideoInfo(
            context.width,
            context.height,
            context.format.name if context.format else "",
            start,
            duration if math.isfinite(duration) and duration > 0 else 0.0,
        )
    if not (48 <= info.width <= 8192 and 48 <= info.height <= 8192):
        raise ValueError("Unsupported video dimensions")
    if not info.pix_fmt.startswith(("yuv", "gray")):
        raise ValueError(
            "Video must expose a YUV or grayscale plane. Extract an opaque RGB still for"
            " other formats."
        )
    if not _PLANAR.fullmatch(info.pix_fmt):
        raise ValueError(f"Unsupported native planar pixel format: {info.pix_fmt}")
    return info


def run_ffmpeg(
    arguments: list[str], check: Callable[[], None], timeout: float
) -> bytes:
    """The FFmpeg CLI's stdout. `check` raises to abort: the child is killed first, and so is a
    child past `timeout` seconds."""
    process = subprocess.Popen(
        arguments,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    start = time.monotonic()
    while True:
        try:
            output, errors = process.communicate(timeout=0.05)
            break
        except subprocess.TimeoutExpired:
            try:
                check()
                if time.monotonic() - start > timeout:
                    raise RuntimeError("FFmpeg exceeded its time limit") from None
            except BaseException:
                process.kill()
                process.communicate()
                raise
    if process.returncode != 0:
        raise RuntimeError(
            "FFmpeg failed: " + errors.decode("utf-8", "replace").strip()[-4000:]
        )
    return output


def discover(
    path: Path,
    info: VideoInfo,
    mode: str,
    cap: int,
    times: list[float],
    threshold: float,
    minimum_scene: float,
    tools: Callable[[], FFMpegEnv],
    check: Callable[[], None],
) -> SamplingPlan:
    """media.cpp's discoverSamples for a video. mode: even, manual, keyframes or scenes."""
    capped_indices(0, cap)
    if not (0 < threshold < 1) or not (
        minimum_scene >= 0 and math.isfinite(minimum_scene)
    ):
        raise ValueError("Invalid scene settings")
    plan = SamplingPlan(str(path.resolve()), mode, info.duration)
    if plan.duration <= 0:
        raise ValueError("Frame discovery requires a finite positive video duration")
    points: list[SamplePoint] = []
    if mode == "even":
        points = [
            SamplePoint(plan.duration * (i + 0.5) / cap, REQUESTED) for i in range(cap)
        ]
    elif mode == "manual":
        if not times:
            raise ValueError("Enter at least one timestamp for manual sampling")
        if len(times) > cap:
            raise ValueError("Manual timestamps exceed the frame cap")
        points = [SamplePoint(t, REQUESTED) for t in times]
    elif mode == "keyframes":
        points = _indexed_keyframes(path, info, plan.duration)
        if not points:
            try:
                points = _listed_keyframes(path, info, plan.duration, tools, check)
            except (OSError, RuntimeError, ValueError) as error:
                raise ValueError(
                    f"The codec keyframes could not be listed ({error}). Use Even or"
                    " Scenes sampling."
                ) from error
        check()
    elif mode == "scenes":
        output = run_ffmpeg(
            [
                tools().ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-threads",
                "4",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-dn",
                "-vf",
                f"select='gt(scene,{threshold:.8g})',metadata=mode=print:file=-",
                "-fps_mode",
                "vfr",
                "-f",
                "null",
                "-",
            ],
            check,
            1800,
        )
        boundaries = {0.0}
        for match in re.finditer(
            r"pts_time:([-+0-9.eE]+)", output.decode("utf-8", "replace")
        ):
            t = float(match.group(1))
            if not math.isfinite(t):
                raise ValueError("Invalid scene timestamp")
            if 0 < t < plan.duration:
                boundaries.add(t)
        edges = [*sorted(boundaries), plan.duration]
        plan.candidates = [
            SamplePoint((a + b) / 2, SCENE, a, b)
            for a, b in zip(edges, edges[1:])
            if b - a >= minimum_scene
        ]
        if not plan.candidates:
            raise ValueError("No scenes meet the minimum duration; lower that setting")
        return plan
    else:
        raise ValueError(f"Unknown sampling mode {mode}")
    for point in points:
        if not (math.isfinite(point.seconds) and 0 <= point.seconds < plan.duration):
            raise ValueError("A requested timestamp is outside the video")
    points.sort(key=lambda p: p.seconds)
    for point in points:
        if (
            not plan.candidates
            or abs(point.seconds - plan.candidates[-1].seconds) >= 1e-6
        ):
            plan.candidates.append(point)
    if not plan.candidates:
        raise ValueError(
            "No usable codec keyframes were found. Use Even or Scenes sampling."
        )
    return plan


def _indexed_keyframes(
    path: Path, info: VideoInfo, duration: float
) -> list[SamplePoint]:
    """The codec keyframes within the duration by PyAV's frame index, with their frame
    numbers; none when PyAV cannot index the file (logged)."""
    av = _pyav()
    assert av is not None
    try:
        index = frame_index(path)
        with av.open(str(path)) as container:
            time_base = container.streams.video[0].time_base
    except (_UseCliReader, av.error.FFmpegError) as reason:
        logger.warning("Listing the keyframes of %s with ffprobe: %s", path, reason)
        return []
    start = Fraction(info.start)
    points: list[SamplePoint] = []
    for n in index.keyframes:
        t = float(index.pts[n] * time_base - start)
        if -1e-5 <= t < duration:
            points.append(SamplePoint(max(0.0, t), KEYFRAME, frame=n))
    return points


def _listed_keyframes(
    path: Path,
    info: VideoInfo,
    duration: float,
    tools: Callable[[], FFMpegEnv],
    check: Callable[[], None],
) -> list[SamplePoint]:
    """media.cpp's keyframe discovery: ffprobe decodes the keyframes alone and lists
    their timestamps."""
    output = run_ffmpeg(
        [
            tools().ffprobe,
            "-v",
            "error",
            "-threads",
            "2",
            "-skip_frame",
            "nokey",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=key_frame,best_effort_timestamp_time",
            "-of",
            "json",
            str(path),
        ],
        check,
        1800,
    )
    points: list[SamplePoint] = []
    for frame in json.loads(output).get("frames", []):
        if frame.get("key_frame") != 1:
            continue
        t = float(frame.get("best_effort_timestamp_time", "nan")) - info.start
        if not math.isfinite(t):
            raise ValueError("A keyframe lacks a usable presentation timestamp")
        if -1e-5 <= t < duration:
            points.append(SamplePoint(max(0.0, t), KEYFRAME))
    return points


def _codes(
    data: Buffer, width: int, height: int, stride: int, pix_fmt: str
) -> np.ndarray:
    """The Y plane's code values over the maximum code, with no range conversion."""
    match = _PLANAR.fullmatch(pix_fmt)
    if match is None:
        raise ValueError(f"Unsupported native planar pixel format: {pix_fmt}")
    bits = int(match.group(2) or 8)
    dtype = np.dtype(
        np.uint8 if bits == 8 else ">u2" if match.group(3) == "be" else "<u2"
    )
    codes = np.lib.stride_tricks.as_strided(
        np.frombuffer(data, dtype), (height, width), (stride, dtype.itemsize)
    )
    return codes.astype(np.float64) / ((1 << bits) - 1)


@dataclass(frozen=True)
class ColourModel:
    """What a video's linear-light planes need: Load Video's float RGB format for its colour
    information, the matrix's luma weights, and the assumptions made about missing tags."""

    rgb: FrameFormat
    weights: tuple[float, float, float]
    assumptions: str


def colour_model(
    path: Path,
    info: VideoInfo,
    tools: Callable[[], FFMpegEnv],
    check: Callable[[], None],
) -> ColourModel:
    """media.cpp's loadColor rules over chaiNNer's colour detection: HDR (PQ, HLG) refused,
    the BT.709, BT.601 and BT.2020 non-constant matrices only."""
    output = run_ffmpeg(
        [
            tools().ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        check,
        60,
    )
    (stream,) = json.loads(output)["streams"]
    colour, _ = detect_colour_info(stream)
    if colour.transfer in (Transfer.PQ, Transfer.HLG):
        raise ValueError("Linear-light SDR hypotheses do not support HDR PQ/HLG input")
    weights = _MATRIX_WEIGHTS.get(colour.matrix)
    if weights is None:
        raise ValueError(
            f"Unsupported linear-light RGB conversion matrix: {colour.matrix.value}"
        )
    missing = "".join(
        f"Missing {label}: {value} assumed. "
        for label, name, value in (
            ("matrix", "matrix", colour.matrix.value),
            ("range", "range", colour.range.value),
            ("chroma location", "chroma_location", colour.chroma_location.value),
        )
        if colour.origin_of(name) == "guessed"
    )
    settings = ReadSettings(
        VideoMode.RGB,
        True,
        (Matrix.AUTO, Transfer.AUTO, Primaries.AUTO, Range.AUTO, ChromaLocation.AUTO),
    )
    rgb = VideoLoader.rgb_format(
        VideoMetadata(info.width, info.height, 0.0, 0),
        pixel_layout(info.pix_fmt),
        colour,
        settings,
    )
    return ColourModel(
        rgb,
        weights,
        missing
        + f"YUV matrix {colour.matrix.value}, range {colour.range.value} converted to"
        " full-range encoded RGB by Load Video's float RGB reader (libswscale), with no"
        " spatial resizing or clipping. Chroma reconstruction: lanczos, location"
        f" {colour.chroma_location.value}; subsampling/codec artifacts can limit evidence."
        f" Transfer metadata: {colour.transfer.value}. The selected EOTF is a hypothesis,"
        " not inferred from the tag.",
    )


def decode_frame(
    path: Path,
    info: VideoInfo,
    point: SamplePoint,
    tools: Callable[[], FFMpegEnv],
    check: Callable[[], None],
    rgb: FrameFormat | None = None,
) -> tuple[np.ndarray, str, np.ndarray | None]:
    """The As stored plane of the frame at `point`, its provenance and, given `rgb`, the
    frame in that float RGB format (B, G, R): the first frame at or after the requested
    time, as FFmpeg's input seek gives it (a keyframe candidate: that keyframe)."""
    av = _pyav()
    try:
        if av is None:
            raise _UseCliReader("PyAV is unavailable")
        index = frame_index(path)
    except _UseCliReader as reason:
        if rgb is not None:
            raise ValueError(
                "The light models need Load Video's PyAV reader, which cannot read this"
                f" video: {reason}"
            ) from reason
        return (*_cli_plane(path, info, point, tools().ffmpeg, check), None)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if point.frame is not None:
            n = point.frame
        else:
            # FFmpeg's -ss seconds (microseconds) after the format's start time.
            target = Fraction(round(point.seconds * 1_000_000), 1_000_000) + Fraction(
                info.start
            )
            n = bisect_left(index.pts, math.ceil(target / stream.time_base))
            if n >= len(index.pts):
                raise ValueError("Decoder returned no complete frame at this timestamp")
        frames = _seek(container, stream, path, av, n)
        try:
            frame = next(frames)
        finally:
            frames.close()
        plane = frame.planes[0]
        values = _codes(
            plane, frame.width, frame.height, plane.line_size, frame.format.name
        )
        pix_fmt = frame.format.name
        image = None
        if rgb is not None:
            converters = _Converters(path, stream, [rgb], av)
            try:
                (image,) = converters.values(frame)
            except _UseCliReader as reason:
                raise ValueError(
                    f"The light models cannot read this video's RGB: {reason}"
                ) from reason
            finally:
                converters.free()
            assert isinstance(image, np.ndarray)
    return (
        values,
        (
            f"First plane of the decoder's native {pix_fmt} output, divided by its maximum code"
            " value. No range conversion, clipping or spatial scaling. Decoded by PyAV at frame"
            f" {n} of the file's packet index."
        ),
        image,
    )


def _keyframe_filter(seconds: float, info: VideoInfo) -> str:
    timestamp = seconds + info.start
    return f"select=between(t\\,{timestamp - 0.00001:.8f}\\,{timestamp + 0.00001:.8f})"


def _cli_plane(
    path: Path,
    info: VideoInfo,
    point: SamplePoint,
    ffmpeg: str,
    check: Callable[[], None],
) -> tuple[np.ndarray, str]:
    """media.cpp's loadFrame through the FFmpeg CLI."""
    keyframe = point.kind == KEYFRAME
    arguments = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-threads",
        "2",
    ]
    if keyframe:
        arguments += ["-skip_frame", "nokey", "-copyts"]
    arguments += [
        "-ss",
        f"{max(0.0, point.seconds - (0.001 if keyframe else 0.0)):.6f}",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-fps_mode",
        "passthrough",
    ]
    if keyframe:
        arguments += ["-vf", _keyframe_filter(point.seconds, info)]
    arguments += ["-pix_fmt", info.pix_fmt, "-f", "rawvideo", "pipe:1"]
    data = run_ffmpeg(arguments, check, 60)
    match = _PLANAR.fullmatch(info.pix_fmt)
    assert match is not None
    sample = 2 if match.group(2) else 1
    if len(data) < info.width * info.height * sample:
        raise ValueError("Decoder returned no complete frame at this timestamp")
    values = _codes(
        data[: info.width * info.height * sample],
        info.width,
        info.height,
        info.width * sample,
        info.pix_fmt,
    )
    return values, (
        f"First plane of FFmpeg native {info.pix_fmt} output, divided by its maximum code"
        " value. No range conversion, clipping or spatial scaling. Requested timestamp, not"
        f" a verified frame number. Decoder: {ffmpeg}"
    )
