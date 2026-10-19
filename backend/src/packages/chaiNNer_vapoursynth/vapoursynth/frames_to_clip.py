from __future__ import annotations

import subprocess
import tempfile
import uuid
from enum import Enum
from fractions import Fraction
from pathlib import Path

import numpy as np
from sanic.log import logger

from api import MB, Collector, IteratorInputInfo, NodeContext
from nodes.groups import if_enum_group, required
from nodes.impl.colour_info import (
    ChromaLocation,
    ColourInfo,
    Matrix,
    Primaries,
    Range,
    Transfer,
    detect_colour_info,
    float_to_code,
    option_value,
    subsampling_from_sizes,
)
from nodes.impl.ffmpeg import FFMpegEnv
from nodes.impl.image_utils import to_uint16
from nodes.impl.vapoursynth.colour_props import CHROMA_LOCATION_NAMES
from nodes.impl.vapoursynth.script import VideoClip, clip_var
from nodes.properties.inputs import (
    ColourInfoInput,
    EnumInput,
    ImageInput,
    NumberInput,
    colour_override_inputs,
)
from nodes.properties.outputs import VideoClipOutput
from nodes.utils.utils import get_h_w_c

from ..chains import chain_dir
from . import clips_group

# How many of FFmpeg's last stderr lines an error message keeps.
STDERR_TAIL_LINES = 20
# NUT's codec tag for rawvideo gbrpf32le in current FFmpeg ("G3", 0, 33), which
# BestSource's FFmpeg reads: the integrated FFmpeg 5.1 has none and writes another
# format's tag, so the writer sets it, and -strict unofficial lets it.
GBRPF32_NUT_TAG = "0x21003347"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class ClipSource(Enum):
    FRAMES = "frames"
    PLANES = "planes"


class ClipPrecision(Enum):
    BITS_16 = "16"
    FLOAT_32 = "32"


# The 16-bit planar formats of the subsamplings (log2 width, height) FFV1 writes.
YUV_FORMATS = {(0, 0): "yuv444p16le", (1, 0): "yuv422p16le", (1, 1): "yuv420p16le"}
# Siting that Matroska cannot store: the clip's statement sets it.
UNSTORED_CHROMA_LOCATIONS = (ChromaLocation.BOTTOM_LEFT, ChromaLocation.BOTTOM)


def planes_colour(
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    colour_info: ColourInfo | None,
    choices: tuple[Matrix, Transfer, Primaries, Range, ChromaLocation],
) -> tuple[str, ColourInfo]:
    """The pixel format of a YUV clip of these planes, and the ColourInfo it is tagged
    with: `colour_info` (without one, the guess for planes of these sizes) with the
    non-Auto `choices`."""
    luma, chroma = (y.shape[1], y.shape[0]), (u.shape[1], u.shape[0])
    if v.shape != u.shape:
        raise ValueError(
            f"The U and V planes differ in size: U is {chroma[0]}x{chroma[1]}, V is"
            f" {v.shape[1]}x{v.shape[0]}."
        )
    sx, sy = subsampling_from_sizes(luma, chroma)
    pix_fmt = YUV_FORMATS.get((sx, sy))
    if pix_fmt is None:
        raise ValueError(
            f"Frames to Clip writes 4:4:4, 4:2:2 or 4:2:0 planes, not a Y plane"
            f" {luma[0]}x{luma[1]} with U and V planes {chroma[0]}x{chroma[1]}."
        )
    if luma[0] % (1 << sx) or luma[1] % (1 << sy):
        # BestSource would crop the clip to such a size.
        raise ValueError(
            f"VapourSynth needs the Y plane of 4:{pix_fmt[4]}:{pix_fmt[5]} planes to be"
            f" a multiple of {1 << sx}x{1 << sy} in size, not {luma[0]}x{luma[1]}."
        )
    if colour_info is None:
        stream = {"pix_fmt": pix_fmt, "width": luma[0], "height": luma[1]}
        colour_info, _ = detect_colour_info({**stream, "codec_name": "ffv1"})
    info = colour_info.with_choices(*choices)
    if info.matrix is Matrix.RGB:
        raise ValueError(
            "The Colour Info's matrix is RGB (identity), which a YUV clip cannot have."
            " Set Matrix."
        )
    return pix_fmt, info


def ffv1_options(width: int, height: int, planes: int) -> list[str]:
    """FFmpeg's encoder options for 16-bit frames of this size and plane count:
    lossless FFV1 in 16 slices (level 3), which FFmpeg encodes on up to 16 threads,
    where they are valid, else FFV1's default of one slice.

    The integrated FFmpeg 5.1 refuses 16 slices (4x4) below 4 pixels or chroma samples
    a side, and from 4 to 6 pixels writes slice positions that decoders misread,
    corrupting the frames; from 7 every slice is where it belongs. It also refuses a
    slice of more than 2^27 bits, at 17 bits a sample."""
    slice_samples = -(-width // 4) * -(-height // 4) * planes
    if min(width, height) >= 7 and slice_samples * 17 <= 1 << 27:
        return ["-c:v", "ffv1", "-level", "3", "-slices", "16"]
    return ["-c:v", "ffv1"]


def frame_rate(fps: float) -> Fraction:
    """The clip's frame rate: an NTSC rate (24000/1001 for 23.976) when fps is one to
    the FPS input's 4 decimals, else fps as a fraction."""
    ntsc = Fraction(round(fps * 1.001) * 1000, 1001)
    if abs(fps - ntsc) < 0.0005:
        return ntsc
    return Fraction(fps).limit_denominator(1001)


class RawVideoWriter:
    """FFmpeg encoding raw frames of `pix_fmt` from its stdin into `path`."""

    def __init__(
        self,
        ffmpeg: str,
        path: Path,
        pix_fmt: str,
        size: tuple[int, int],
        rate: Fraction,
        output: list[str],
    ):
        self._stderr = tempfile.TemporaryFile()
        self._process = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                pix_fmt,
                "-s",
                f"{size[0]}x{size[1]}",
                "-framerate",
                f"{rate.numerator}/{rate.denominator}",
                "-i",
                "pipe:",
                *output,
                "-y",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr,
            creationflags=_NO_WINDOW,
        )

    def _stopped(self, returncode: int) -> RuntimeError:
        self._stderr.seek(0)
        lines = self._stderr.read().decode("utf-8", "replace").splitlines()
        message = "\n".join(lines[-STDERR_TAIL_LINES:]).strip()
        text = (
            "FFmpeg stopped while Frames to Clip was writing the video"
            f" (exit code {returncode})"
        )
        return RuntimeError(f"{text}: {message}" if message else f"{text}.")

    def _failure(self) -> RuntimeError | None:
        """After a pipe error: FFmpeg's failure once it has exited with one (a rejected
        option, a full disk close the pipe), else None, and the pipe error stands."""
        try:
            returncode = self._process.wait(5)
        except subprocess.TimeoutExpired:
            return None
        return self._stopped(returncode) if returncode else None

    def write(self, planes: np.ndarray) -> None:
        stdin = self._process.stdin
        assert stdin is not None
        try:
            stdin.write(memoryview(np.ascontiguousarray(planes)).cast("B"))
        except OSError as error:
            failure = self._failure()
            if failure is None:
                raise
            raise failure from error

    def close(self) -> None:
        """Ends the video; raises when FFmpeg failed."""
        stdin = self._process.stdin
        assert stdin is not None
        try:
            stdin.close()
        except OSError as error:
            failure = self._failure()
            if failure is None:
                raise
            raise failure from error
        returncode = self._process.wait()
        if returncode != 0:
            raise self._stopped(returncode)
        self._stderr.close()

    def kill(self) -> None:
        """Stops FFmpeg if it still runs (a run that ended before the last frame)."""
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait()
        self._stderr.close()


@clips_group.register(
    schema_id="chainner:vapoursynth:frames_to_clip",
    name="Frames to Clip",
    description=[
        (
            "Turns a sequence of RGB or gray images, or of Y, U and V planes, into a"
            " clip: the frames are saved as a lossless 16-bit video (FFV1) or as 32-bit"
            " float frames uncompressed, which the clip reads with BestSource."
        ),
        (
            "The video is a temporary file in chaiNNer's storage; the node's next run"
            " replaces it, and a start of chaiNNer deletes it once it is a day old."
        ),
    ],
    icon="BsFilm",
    inputs=[
        EnumInput(
            ClipSource,
            "Input",
            default=ClipSource.FRAMES,
            option_labels={
                ClipSource.FRAMES: "Frames (RGB or gray)",
                ClipSource.PLANES: "Planes (Y, U, V)",
            },
        )
        .with_id(2)
        .with_docs(
            "Planes makes a YUV clip of planes at their own sizes (as Load Video's"
            " Planes mode gives them), 16-bit and tagged with the colour information.",
        ),
        if_enum_group(2, ClipSource.FRAMES)(
            required()(
                ImageInput("Frames", channels=[1, 3]).make_optional().with_id(0)
            ),
            EnumInput(
                ClipPrecision,
                "Precision",
                default=ClipPrecision.BITS_16,
                option_labels={
                    ClipPrecision.BITS_16: "16-bit",
                    ClipPrecision.FLOAT_32: "32-bit float",
                },
            )
            .with_id(3)
            .with_docs(
                "16-bit rounds each value to 1/65535 and clips it to 0-1, in a lossless"
                " compressed video (RGB48 or Gray16 clips).",
                "32-bit float keeps every value as it is, also outside 0-1 (RGBS or"
                " GrayS clips), in an uncompressed file of 12 bytes per pixel (gray"
                " frames too): about 25 MB per 1080p frame.",
            ),
        ),
        if_enum_group(2, ClipSource.PLANES)(
            required()(
                ImageInput("Y", channels=1).make_optional().with_id(4),
                ImageInput("U", channels=1).make_optional().with_id(5),
                ImageInput("V", channels=1).make_optional().with_id(6),
            ),
            ColourInfoInput()
            .make_optional()
            .with_id(7)
            .with_docs(
                "The planes' colour information, which the clip is tagged with (its"
                " frame properties). Without it, Auto guesses from the planes' size."
            ),
            *colour_override_inputs(8),
        ),
        NumberInput("FPS", default=30, min=1, step=1, precision=4).with_id(1),
    ],
    iterator_inputs=IteratorInputInfo(inputs=[0, 4, 5, 6]),
    outputs=[VideoClipOutput("Clip")],
    kind="collector",
    node_context=True,
)
def frames_to_clip_node(
    context: NodeContext,
    source: ClipSource,
    _frames: None,
    precision: ClipPrecision,
    _y: None,
    _u: None,
    _v: None,
    colour_info: ColourInfo | None,
    matrix: Matrix,
    transfer: Transfer,
    primaries: Primaries,
    range_: Range,
    chroma_location: ChromaLocation,
    fps: float,
) -> Collector[tuple[np.ndarray | None, ...], VideoClip]:
    folder = chain_dir(context)
    # The previous run's video and its index, unless a clip in an open vsview still
    # reads them; a later run deletes those.
    for old in folder.iterdir():
        try:
            old.unlink()
        except OSError as error:
            logger.debug("Frames to Clip keeps %s: %s", old, error)

    planes_mode = source is ClipSource.PLANES
    float32 = not planes_mode and precision is ClipPrecision.FLOAT_32
    path = folder / f"frames-{uuid.uuid4().hex[:12]}.{'nut' if float32 else 'mkv'}"
    rate = frame_rate(fps)
    writer: RawVideoWriter | None = None
    # The first iteration's (height, width, channels) per image; every iteration must
    # match them.
    shapes: list[tuple[int, int, int]] = []
    tags: ColourInfo | None = None

    def start(pix_fmt: str, size: tuple[int, int], output: list[str]) -> None:
        nonlocal writer
        ffmpeg = FFMpegEnv.get_integrated(context.storage_dir).ffmpeg
        output = [*output, "-pix_fmt", pix_fmt]
        writer = RawVideoWriter(ffmpeg, path, pix_fmt, size, rate, output)
        context.add_cleanup(writer.kill)

    def check(current: list[tuple[int, int, int]]) -> None:
        if not shapes:
            shapes.extend(current)
        elif current != shapes:
            sizes = ", ".join(f"{w}x{h} with {c} channels" for h, w, c in shapes)
            now = ", ".join(f"{w}x{h} with {c}" for h, w, c in current)
            raise ValueError(
                "Frames to Clip needs images of one size and channel count: the first"
                f" were {sizes}, these are {now}."
            )

    def write_planes(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> None:
        nonlocal tags
        check([get_h_w_c(plane) for plane in (y, u, v)])
        if tags is None:
            choices = (matrix, transfer, primaries, range_, chroma_location)
            pix_fmt, tags = planes_colour(y, u, v, colour_info, choices)
            output = ffv1_options(y.shape[1], y.shape[0], 3)
            for option, value in (
                ("-colorspace", tags.matrix),
                ("-color_trc", tags.transfer),
                ("-color_primaries", tags.primaries),
                ("-color_range", tags.range),
                ("-chroma_sample_location", tags.chroma_location),
            ):
                output += [option, option_value(value)]
            start(pix_fmt, (y.shape[1], y.shape[0]), output)
        assert writer is not None
        for plane, chroma in ((y, False), (u, True), (v, True)):
            writer.write(float_to_code(plane, 16, tags.full_range, chroma))

    def write_frame(frame: np.ndarray) -> None:
        h, w, c = get_h_w_c(frame)
        check([(h, w, c)])
        if writer is None:
            if float32:
                # BestSource reads no gray float format: gray frames go as RGB.
                output = ["-c:v", "rawvideo", "-tag:v", GBRPF32_NUT_TAG]
                start("gbrpf32le", (w, h), [*output, "-strict", "unofficial"])
                logger.warning(
                    "Frames to Clip writes %dx%d 32-bit float frames uncompressed:"
                    " %.1f MB per frame.",
                    w,
                    h,
                    w * h * 12 / MB,
                )
            else:
                # FFV1 in Matroska, lossless. Full range: VapourSynth reads an
                # untagged gray clip as limited.
                pix_fmt = "gray16le" if c == 1 else "gbrp16le"
                start(pix_fmt, (w, h), [*ffv1_options(w, h, c), "-color_range", "pc"])
        assert writer is not None
        if not float32:
            # Clipped to 0-1 and rounded.
            frame = to_uint16(frame)
        if c == 1:
            planes = frame.reshape(1, h, w)
            if float32:
                planes = np.broadcast_to(planes, (3, h, w))
        else:
            # gbrp's plane order: G, B, R of chaiNNer's BGR.
            planes = np.stack((frame[:, :, 1], frame[:, :, 0], frame[:, :, 2]))
        writer.write(planes)

    def on_iterate(values: tuple[np.ndarray | None, ...]) -> None:
        frame, y, u, v = values
        if planes_mode:
            if y is None or u is None or v is None:
                raise ValueError("Frames to Clip's Planes input needs Y, U and V.")
            write_planes(y, u, v)
        else:
            if frame is None:
                raise ValueError("Frames to Clip needs Frames.")
            write_frame(frame)

    def complete() -> VideoClip:
        if writer is None:
            raise ValueError("Frames to Clip got no frames.")
        writer.close()
        var = clip_var(context.node_id)
        # cachemode=3 keeps BestSource's index (written only for long videos) beside
        # the video, so it goes with it; AssumeFPS sets the exact rate the container
        # rounds.
        statement = (
            f"{var} = core.bs.VideoSource(source={str(path)!r}, cachemode=3)"
            f".std.AssumeFPS(fpsnum={rate.numerator}, fpsden={rate.denominator})"
        )
        if float32 and shapes[0][2] == 1:
            statement += ".std.ShufflePlanes(planes=0, colorfamily=vs.GRAY)"
        if tags is not None and tags.chroma_location in UNSTORED_CHROMA_LOCATIONS:
            codes = {name: code for code, name in CHROMA_LOCATION_NAMES.items()}
            location = codes[tags.chroma_location.value]
            statement += f".std.SetFrameProps(_ChromaLocation={location})"
        return VideoClip(
            var=var,
            stmt=statement,
            deps=(),
            imports=frozenset(),
            node_id=context.node_id,
        )

    return Collector(on_iterate=on_iterate, on_complete=complete)
