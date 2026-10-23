from __future__ import annotations

# `x as x` imports: the native mirror reads these names in native/src/video_io.cpp
import os as os
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from subprocess import Popen
from typing import Any, Literal

import ffmpeg as ffmpeg
import numpy as np
from sanic.log import logger as logger

from api import Collector, IteratorInputInfo, KeyInfo, NodeContext
from nodes.groups import Condition, if_enum_group, if_group, required
from nodes.impl.colour_info import (
    MATRIX_LABELS,
    SWSCALE_CHROMA_POSITION,
    ChromaLocation,
    ColourInfo,
    Matrix,
    Primaries,
    Range,
    Transfer,
    float_to_code,
    high_definition,
    option_value,
    pixel_layout,
    subsampling_from_sizes,
    swscale_matrix,
)
from nodes.impl.ffmpeg import FFMpegEnv
from nodes.impl.image_utils import to_uint8, to_uint16, to_uint16_gbr_planes
from nodes.impl.item_window import Prepared, register_phases
from nodes.impl.native_graph import graph
from nodes.impl.native_image_io import cv_save_image
from nodes.impl.plane_resample import Kernel, resample_plane
from nodes.impl.yuv_rgb import (
    guess_colour_info,
    planes_to_rgb,
    resolve_colour_info,
    siting,
)
from nodes.properties.inputs import (
    ColourInfoInput,
    DirectoryInput,
    EnumInput,
    ImageInput,
    RelativePathInput,
    SliderInput,
    TextInput,
    colour_override_inputs,
)
from nodes.properties.inputs.generic_inputs import AudioStreamInput
from nodes.properties.inputs.numeric_inputs import NumberInput
from nodes.utils.utils import get_h_w_c

from .. import video_frames_group


class VideoFormat(Enum):
    MKV = "mkv"
    MP4 = "mp4"
    MOV = "mov"
    WEBM = "webm"
    AVI = "avi"
    GIF = "gif"

    @property
    def ext(self) -> str:
        return self.value

    @property
    def muxer(self) -> str:
        """FFmpeg's muxer for the extension, as FFmpeg picks it from a file name: Save Video
        writes a temporary file ending in .tmp, so it names the muxer."""
        return "matroska" if self == VideoFormat.MKV else self.value

    @property
    def encoders(self) -> tuple[VideoEncoder, ...]:
        return graph().video_container_encoders(globals(), self)


class VideoEncoder(Enum):
    H264 = "libx264"
    H265 = "libx265"
    VP9 = "libvpx-vp9"
    FFV1 = "ffv1"

    @property
    def formats(self) -> tuple[VideoFormat, ...]:
        return graph().video_encoder_formats(globals(), self)

    @property
    def bit_depths(self) -> tuple[int, ...]:
        """The bit depths the encoder takes in chaiNNer's FFmpeg 5.1.2 (ffmpeg -h
        encoder=...), of those Save Video writes, in increasing order."""
        return {
            VideoEncoder.H264: (8, 10),
            VideoEncoder.H265: (8, 10, 12),
            VideoEncoder.VP9: (8, 10, 12),
            VideoEncoder.FFV1: (8, 9, 10, 12, 14, 16),
        }[self]

    @property
    def pixel_formats(self) -> tuple[str, ...]:
        """The YUV and RGB pixel formats the encoder takes in chaiNNer's FFmpeg 5.1.2,
        of those Save Video writes."""
        depths = ["" if depth == 8 else f"{depth}le" for depth in self.bit_depths]
        layouts = ("420", "422", "444")
        formats = [f"yuv{layout}p{depth}" for layout in layouts for depth in depths]
        if self is not VideoEncoder.H264:
            formats += [f"gbrp{depth}" for depth in depths]
        if self is VideoEncoder.FFV1:
            formats.remove("gbrp")  # FFV1 takes 8-bit RGB as bgr0 only
        return tuple(formats)

    @property
    def auto_depth(self) -> int | None:
        """The most bits Bit Depth Auto gives the encoder: what players commonly
        decode (H.264 8, H.265 and VP9 10); FFV1 keeps the source's."""
        return {VideoEncoder.H264: 8, VideoEncoder.H265: 10, VideoEncoder.VP9: 10}.get(
            self
        )


class VideoPreset(Enum):
    ULTRA_FAST = "ultrafast"
    SUPER_FAST = "superfast"
    VERY_FAST = "veryfast"
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    SLOWER = "slower"
    VERY_SLOW = "veryslow"


class AudioSettings(Enum):
    AUTO = "auto"
    COPY = "copy"
    TRANSCODE = "transcode"


class Simplicity(Enum):
    SIMPLE = 0
    ADVANCED = 1


class SimpleVideoFormat(Enum):
    MP4_H264 = "mp4_h264"
    MP4_H265 = "mp4_h265"
    WEBM = "webm"
    GIF = "gif"


class VideoInputMode(Enum):
    RGB = "rgb"
    PLANES = "planes"


class BitDepth(Enum):
    AUTO = "auto"
    BITS_8 = "8"
    BITS_10 = "10"
    BITS_12 = "12"
    BITS_16 = "16"


class SitingCorrection(Enum):
    AUTO = "auto"
    OFF = "off"
    ON = "on"


def get_simple_format(
    simple_video_format: SimpleVideoFormat, quality: int
) -> tuple[VideoFormat, VideoEncoder, VideoPreset, int]:
    return graph().video_simple_format(globals(), simple_video_format, quality)


PARAMETERS: dict[VideoEncoder, list[Literal["preset", "crf"]]] = {
    VideoEncoder.H264: ["preset", "crf"],
    VideoEncoder.H265: ["preset", "crf"],
    VideoEncoder.VP9: ["crf"],
    VideoEncoder.FFV1: [],
}


@dataclass(frozen=True)
class WriteSettings:
    """Save Video's input mode, Colour Info input (None when unconnected), the five
    colour choices (ColourInfo.with_choices's arguments), bit depth and chroma siting
    correction."""

    mode: VideoInputMode
    colour: ColourInfo | None
    choices: tuple[Matrix, Transfer, Primaries, Range, ChromaLocation]
    bit_depth: BitDepth
    siting: SitingCorrection


def rgb_payload(img: np.ndarray) -> np.ndarray:
    """An RGB frame as Save Video pipes it: gbrp16le, the G, B and R planes in 16 bits,
    so FFmpeg's YUV conversion keeps more than 8 bits of the float image."""
    return to_uint16_gbr_planes(img)


def atomic_target(path: str) -> Any:
    """A temporary file in path's folder, renamed onto path by commit()
    (native/include/atomic_file.hpp): Save Video's FFmpeg output and audio mux."""
    return graph().AtomicTarget(path)


def colour_tags(colour: ColourInfo) -> dict[str, str]:
    """`colour` as FFmpeg's colour tag options."""
    return {
        "colorspace": option_value(colour.matrix),
        "color_primaries": option_value(colour.primaries),
        "color_trc": option_value(colour.transfer),
        "color_range": option_value(colour.range),
        "chroma_sample_location": option_value(colour.chroma_location),
    }


def planar_format(colour: ColourInfo) -> str:
    """The planar pixel format of planes coded as `colour` says (4:4:4, 4:2:2 or 4:2:0
    YUV, or RGB for the identity matrix)."""
    depth = "" if colour.bit_depth == 8 else f"{colour.bit_depth}le"
    if colour.matrix is Matrix.RGB:
        return f"gbrp{depth}"
    layout = {(0, 0): "444", (1, 0): "422", (1, 1): "420"}[colour.subsampling]
    return f"yuv{layout}p{depth}"


@dataclass
class Writer:
    container: VideoFormat
    encoder: VideoEncoder | None
    fps: float
    audio: object | None
    audio_settings: AudioSettings
    save_path: str
    output_params: dict[str, str | float]
    global_params: list[str]
    ffmpeg_env: FFMpegEnv
    # The temporary file in save_path's folder that FFmpeg writes (output_params' filename),
    # renamed onto save_path at a successful close and deleted on Stop or a failure.
    target: Any
    out: Popen | None = None
    # Save Video's colour settings; None writes RGB as without them (Auto, 8-bit).
    settings: WriteSettings | None = None
    # In Planes mode, how the planes are coded (from the first frame).
    coding: ColourInfo | None = None
    # The audio mux writes the final file with -movflags faststart, which the encode pass
    # then leaves out (an MP4 or MOV with audio; video_io.cpp save_video).
    faststart: bool = False

    def start(self, width: int, height: int):
        # Create the writer and run process
        graph().video_writer_start(globals(), self, width, height)

    def write_frame(self, frame: object, prepared: Prepared | None = None):
        """Writes a frame: an image, or Save Video's iterated inputs (Image Sequence,
        Y, U, V), of which its Input Mode takes the image or the planes."""
        graph().video_writer_frame(globals(), self, frame, prepared)

    def close(self):
        graph().video_writer_close(globals(), self)

    def _planes(
        self, frame: object
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """The frame's Y, U and V planes in Planes mode, None in RGB mode."""
        settings = self.settings
        if settings is None or settings.mode is VideoInputMode.RGB:
            return None
        assert isinstance(frame, tuple)
        _, y, u, v = frame
        if y is None or u is None or v is None:
            raise ValueError(
                "Save Video's Planes mode needs the Y, U and V planes (gray video, without"
                " U and V, cannot be written)."
            )
        return y, u, v

    @staticmethod
    def _image(frame: object) -> np.ndarray:
        image = frame[0] if isinstance(frame, tuple) else frame
        if image is None:
            raise ValueError("Save Video's RGB mode needs the Image Sequence.")
        assert isinstance(image, np.ndarray)
        return image

    def first_frame(self, frame: object) -> tuple[int, int]:
        """The video's (width, height) from its first frame: the image's, or in Planes
        mode the Y plane's, when the planes' coding is also set. The pixel format is
        checked here, before FFmpeg starts (whose failure to start is only logged)."""
        planes = self._planes(frame)
        if planes is None:
            h, w, _ = get_h_w_c(self._image(frame))
            self._check_pixel_format(w, h)
            return w, h
        settings = self.settings
        assert settings is not None
        y, u, v = planes
        luma, chroma = (y.shape[1], y.shape[0]), (u.shape[1], u.shape[0])
        if u.shape != v.shape:
            raise ValueError(
                f"The U and V planes must have one size, not {chroma[0]}x{chroma[1]} and"
                f" {v.shape[1]}x{v.shape[0]}."
            )
        subsampling = subsampling_from_sizes(luma, chroma)
        if subsampling not in ((0, 0), (1, 0), (1, 1)):
            raise ValueError(
                f"Save Video writes 4:4:4, 4:2:2 and 4:2:0 planes; U and V of"
                f" {chroma[0]}x{chroma[1]} beside a {luma[0]}x{luma[1]} Y are none of them."
            )
        colour = resolve_colour_info(settings.colour, luma, chroma, *settings.choices)
        if colour.matrix is Matrix.RGB and subsampling != (0, 0):
            raise ValueError(
                "RGB-coded planes (G, B, R) cannot be subsampled: use 4:4:4."
            )
        self.coding = replace(colour, bit_depth=self._bit_depth())
        self._check_pixel_format(*luma)
        return luma

    def _bit_depth(self) -> int:
        """Bit Depth, or for Auto the Colour Info's (8 without it) rounded up to one the
        encoder takes (else its deepest), at most the encoder's auto_depth."""
        settings = self.settings
        if settings is not None and settings.bit_depth is not BitDepth.AUTO:
            return int(settings.bit_depth.value)
        if settings is None or settings.colour is None:
            return 8
        depth = max(8, settings.colour.bit_depth)
        encoder = self.encoder
        if encoder is None:
            return depth
        depths = encoder.bit_depths
        depth = next((d for d in depths if d >= depth), depths[-1])
        cap = encoder.auto_depth
        return depth if cap is None else min(depth, cap)

    @property
    def pipe_format(self) -> str:
        """The pixel format frames are piped in: the planes' own; RGB as gbrp16le, so
        a YUV conversion keeps more than 8 bits, but as bgr24 for GIF's palette and RGB
        output of up to 8 bits, which then stay exact (libswscale reduces 16 bits to 8
        by a shift, off by one for half the codes)."""
        if self.coding is not None:
            return planar_format(self.coding)
        params = self.output_params
        output = (
            pixel_layout(str(params.get("pix_fmt", "")))
            if hasattr(params, "get")
            else None
        )
        if self.container is VideoFormat.GIF or (
            output is not None and output.kind == "rgb" and output.bit_depth <= 8
        ):
            return "bgr24"
        return "gbrp16le"

    @property
    def input_options(self) -> dict[str, str]:
        """FFmpeg's options for the piped frames besides their format: in Planes mode the
        colour tags the output gets (_tag: the planes' own, a user's tag winning), as
        FFmpeg 8.1 converts frames whose tags differ from the output's (untagged BT.709
        planes came out with other codes) where the integrated 5.1.2 only tags them, so
        planes are never converted. None for RGB, which the scale filter converts to the
        output's tags."""
        if self.coding is None:
            return {}
        params = self.output_params
        user = params if hasattr(params, "get") else {}
        return {
            key: str(user.get(key, value))
            for key, value in colour_tags(self.coding).items()
        }

    def check_frame(self, frame: object, size: tuple[int, int]) -> None:
        """Check the source geometry before writing even a prepared payload."""
        planes = self._planes(frame)
        if planes is None:
            h, w, _ = get_h_w_c(self._image(frame))
            if (w, h) != size:
                raise ValueError(
                    f"The frames changed size: {w}x{h} after {size[0]}x{size[1]}."
                )
            return
        coding = self.coding
        assert coding is not None
        y, u, v = planes
        luma, chroma = coding.luma_size, coding.chroma_size
        if (
            (y.shape[1], y.shape[0]) != luma
            or (u.shape[1], u.shape[0]) != chroma
            or u.shape != v.shape
        ):
            raise ValueError(
                f"The planes changed size: Y {y.shape[1]}x{y.shape[0]}, U"
                f" {u.shape[1]}x{u.shape[0]} and V {v.shape[1]}x{v.shape[0]} after"
                f" Y {luma[0]}x{luma[1]} and U/V {chroma[0]}x{chroma[1]}."
            )

    def payload(self, frame: object) -> np.ndarray:
        """A frame's bytes in pipe_format."""
        coding = self.coding
        planes = self._planes(frame)
        if coding is None or planes is None:
            if self.pipe_format == "bgr24":
                return to_uint8(self._image(frame), normalized=True)
            return rgb_payload(self._image(frame))
        self.check_frame(frame, coding.luma_size)
        y, u, v = planes
        u, v = self._sited(u, v)
        chroma_coded = coding.matrix is not Matrix.RGB
        codes = [
            float_to_code(
                plane, coding.bit_depth, coding.full_range, chroma_coded and i > 0
            )
            for i, plane in enumerate((y, u, v))
        ]
        return np.concatenate([c.ravel() for c in codes])

    def _sited(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """U and V moved by the chroma siting correction: resized planes keep the source's
        chroma positions only where its siting is centred, so chroma sited elsewhere is
        resampled (Spline36) to where the saved siting places it. Auto corrects when the
        Colour Info says the source was not centre-sited and the chroma grew."""
        settings, coding = self.settings, self.coding
        assert settings is not None and coding is not None
        source = settings.colour
        if settings.siting is SitingCorrection.OFF:
            return u, v
        if source is None:
            if settings.siting is SitingCorrection.ON:
                raise ValueError(
                    "Chroma Siting Correction On needs the Colour Info input: the source's"
                    " plane sizes."
                )
            return u, v
        (width, height), (source_width, source_height) = (
            coding.chroma_size,
            source.chroma_size,
        )
        if settings.siting is SitingCorrection.AUTO and not (
            source.chroma_location is not ChromaLocation.CENTER
            and (width > source_width or height > source_height)
        ):
            return u, v
        _, _, new_left, new_top = siting(coding.chroma_location, coding.subsampling)
        _, _, old_left, old_top = siting(source.chroma_location, source.subsampling)
        # In chroma pixels of the saved planes: where the saved siting expects each
        # sample, less where the source's siting put the sample now there.
        left = new_left - old_left * width / source_width
        top = new_top - old_top * height / source_height
        if left == 0 and top == 0:
            return u, v
        u, v = (
            resample_plane(
                p, width, height, Kernel.SPLINE36, src_left=left, src_top=top
            )
            for p in (u, v)
        )
        return u, v

    def save_frame(self, frame: object, path: Path) -> None:
        """Saves the frame as the video would hold it to the PNG at path, in the video's
        folder, without starting FFmpeg (a run of one chosen frame): the RGB image, or the
        planes, their chroma moved by the siting correction, converted to RGB by their
        coding; in 16 bits when the video would have more than 8, else in 8. A frame the
        video's writer would refuse, or settings it could not start with, raise as they
        would at the video's first frame."""
        width, height = self.first_frame(frame)
        graph().video_writer_check(globals(), self, width, height)
        params = dict(self.output_params)
        self.colour_options(params, width, height)
        planes, coding = self._planes(frame), self.coding
        if planes is None or coding is None:
            img = self._image(frame)
        else:
            y, u, v = planes
            u, v = self._sited(u, v)
            # The matrix leaves values outside 0..1 (planes_to_rgb does not clip).
            img = np.clip(
                planes_to_rgb(
                    y, u, v, coding.matrix, coding.chroma_location, Kernel.SPLINE36
                ),
                0,
                1,
            )
        deep = pixel_layout(str(params["pix_fmt"])).bit_depth > 8
        cv_save_image(
            path,
            to_uint16(img, normalized=True) if deep else to_uint8(img, normalized=True),
            [],
        )

    def colour_options(self, params: dict[str, Any], width: int, height: int) -> None:
        """Save Video's colour options (owner-approved 2026-10-09), in place, before the
        writer starts: GIF gets a palette made for each frame; YUV output carries all its
        colour tags, -chroma_sample_location included, and is converted to match them
        with filtered chroma, exact rounding and explicit chroma positions (libswscale
        otherwise assumes centre siting). A tag the user gives in Additional parameters
        wins and a user -vf runs before the conversion; RGB output and a user
        -filter_complex (or -lavfi) keep FFmpeg's own conversion. Planes are written in
        their own format, without conversion."""
        if "pix_fmt" not in params or {"filter_complex", "lavfi"} & params.keys():
            return
        if self.container is VideoFormat.GIF:
            # FFmpeg would replace yuv420p with the fixed 3-3-2 palette of bgr8.
            if str(params["pix_fmt"]) == "yuv420p":
                params["pix_fmt"] = "pal8"
                _append_filter(
                    params,
                    "split[a][b];[a]palettegen=stats_mode=single[p];[b][p]paletteuse=new=1",
                )
            return
        coding = self.coding
        if coding is not None:
            params["pix_fmt"] = planar_format(coding)
            self._tag(params, coding)
            return
        depth = self._bit_depth()
        if depth > 8 and str(params["pix_fmt"]) == "yuv420p":
            params["pix_fmt"] = f"yuv420p{depth}le"
        if not str(params["pix_fmt"]).startswith(("yuv", "nv")):
            return
        colour = self._rgb_colour(width, height)
        if colour is None:
            self._tag_by_size(params, width, height)
        else:
            self._tag(params, colour)
        self._append_conversion(params)

    def _rgb_colour(self, width: int, height: int) -> ColourInfo | None:
        """The YUV output's colour in RGB mode: the Colour Info input (else the size
        guess) with the choices applied; None without either (the size guess by tags).
        An RGB-coded Colour Info gives its transfer and primaries only."""
        settings = self.settings
        if settings is None:
            return None
        chosen = any(choice.value != "auto" for choice in settings.choices)
        if settings.colour is None and not chosen:
            return None
        guess = guess_colour_info((width, height), (-(-width // 2), -(-height // 2)))
        base = settings.colour or guess
        if base.matrix is Matrix.RGB:
            base = replace(
                base,
                matrix=guess.matrix,
                range=guess.range,
                chroma_location=guess.chroma_location,
            )
        colour = base.with_choices(*settings.choices)
        if colour.matrix in (Matrix.BT2020_CL, Matrix.YCGCO, Matrix.ICTCP, Matrix.RGB):
            raise ValueError(
                f"Save Video's RGB mode cannot convert to the {MATRIX_LABELS[colour.matrix]}"
                " matrix (FFmpeg's scale filter has no such conversion). Convert with RGB"
                " to Planes and save in Planes mode, or choose another Matrix."
            )
        return colour

    @staticmethod
    def _tag(params: dict[str, Any], colour: ColourInfo) -> None:
        """The colour tags, and for H.265 the HDR10 metadata (the mastering display and
        light levels) as x265 writes it; a user's tag or x265 option wins."""
        for key, value in colour_tags(colour).items():
            params.setdefault(key, value)
        hdr = colour.hdr
        if hdr is None or params.get("vcodec") != VideoEncoder.H265.value:
            return
        user = str(params.get("x265-params", ""))
        options = [user] if user else []
        if hdr.mastering_display is not None and "master-display=" not in user:
            options.append(f"master-display={hdr.mastering_display}")
        if (hdr.max_cll, hdr.max_fall) != (None, None) and "max-cll=" not in user:
            options.append(f"max-cll={hdr.max_cll or 0},{hdr.max_fall or 0}")
        if options:
            params["x265-params"] = ":".join(options)

    @staticmethod
    def _tag_by_size(params: dict[str, Any], width: int, height: int) -> None:
        """Without a user matrix, primaries or transfer, the size picks BT.601 or BT.709
        (players disagree on untagged video, upstream chaiNNer #3053). A user matrix picks
        the missing primaries and transfer of its own family (BT.2020: bt2020 primaries
        and BT.709's curve, which H.273 makes BT.2020's at 8 bits); primaries or a
        transfer without a matrix, or a matrix of no family here, add neither. Chroma is
        left-sited, as MPEG-2, H.264, HEVC and VP9 decoders assume."""
        family = None
        if not {"colorspace", "color_primaries", "color_trc"} & params.keys():
            family = "bt709" if high_definition(width, height) else "smpte170m"
        elif "colorspace" in params:
            family = {
                "bt709": "bt709",
                "smpte170m": "smpte170m",
                "bt470bg": "smpte170m",
                "bt2020nc": "bt2020",
                "bt2020_ncl": "bt2020",
                "bt2020c": "bt2020",
                "bt2020_cl": "bt2020",
            }.get(str(params["colorspace"]))
        if family is not None:
            primaries = "bt470bg" if family == "smpte170m" and height == 576 else family
            params.setdefault("colorspace", family)
            params.setdefault("color_primaries", primaries)
            params.setdefault("color_trc", "bt709" if family == "bt2020" else family)
        params.setdefault("color_range", "tv")
        params.setdefault("chroma_sample_location", "left")

    @staticmethod
    def _append_conversion(params: dict[str, Any]) -> None:
        """The scale filter to the final tags: scale's names for the matrices FFmpeg can
        tag (another tag converts as FFmpeg's default), the range and the chroma
        positions."""
        matrix, colour_range = (
            swscale_matrix(str(params.get("colorspace", ""))),
            str(params["color_range"]),
        )
        location = str(params.get("chroma_sample_location", ""))
        positions = {loc.value: pos for loc, pos in SWSCALE_CHROMA_POSITION.items()}
        scale = "scale="
        if matrix is not None:
            scale += f"out_color_matrix={matrix}:"
        if colour_range in ("tv", "mpeg", "pc", "jpeg"):
            scale += f"out_range={colour_range}:"
        if location in positions:
            scale += f"out_h_chr_pos={positions[location][0]}:"
            scale += f"out_v_chr_pos={positions[location][1]}:"
        _append_filter(params, scale + "flags=accurate_rnd+full_chroma_int")

    def _check_pixel_format(self, width: int, height: int) -> None:
        """Refuses a pixel format the encoder cannot take, and warns about 4:4:4, which
        many hardware players cannot play, and about a -pix_fmt Planes mode ignores."""
        if "pix_fmt" not in self.output_params:
            return
        params = dict(self.output_params)
        given = str(params["pix_fmt"])
        self.colour_options(params, width, height)
        fmt = str(params["pix_fmt"])
        if self.coding is not None and given not in ("yuv420p", fmt):
            logger.warning(
                "Save Video writes the planes as %s, not the -pix_fmt %s given.",
                fmt,
                given,
            )
        encoder = next(
            (e for e in VideoEncoder if e.value == params.get("vcodec")), None
        )
        if encoder is None or not fmt.startswith(("yuv4", "gbrp")):
            return
        if fmt not in encoder.pixel_formats:
            raise ValueError(
                f"The {encoder.value} encoder cannot write {fmt}; it writes"
                f" {', '.join(encoder.pixel_formats)}. Choose another Bit Depth or"
                " encoder (FFV1 writes every layout up to 16 bits)."
            )
        if fmt.startswith("yuv444") and encoder is not VideoEncoder.FFV1:
            logger.warning(
                "%s is 4:4:4, which many hardware players cannot play.", self.save_path
            )


def _append_filter(params: dict[str, Any], chain: str) -> None:
    """The user's -vf (or -filter:v, or -filter) runs first, then `chain`: FFmpeg keeps
    only the last of several, so a -filter left beside it would be dropped."""
    filters = [params.pop(key) for key in ("vf", "filter:v", "filter") if key in params]
    params["vf"] = ",".join([*filters, chain])


@dataclass
class VideoCollector(Collector[tuple[np.ndarray | None, ...], None]):
    """Save Video's collector; its commit phase writes prepared frames to `writer`."""

    writer: Writer


class FrameCollector(Collector[tuple[np.ndarray | None, ...], None]):
    """Save Video's collector in a run of one chosen frame (frame_collector)."""


def frame_collector(
    item: int,
    save_dir: Path,
    video_name: str,
    simplicity: Simplicity,
    container: VideoFormat,
    encoder: VideoEncoder,
    video_preset: VideoPreset,
    crf: int,
    additional_parameters: str | None,
    simple_video_format: SimpleVideoFormat,
    quality: int,
    fps: float,
    audio: object | None,
    audio_settings: AudioSettings,
    settings: WriteSettings,
) -> FrameCollector:
    """Save Video in a run of the one chosen frame `item` (Load Video's "Run this frame",
    or In = Out; owner-approved 2026-10-21): the frame goes to `<video name>_f000123.png`
    next to where the video would go, as the video would hold it (Writer.save_frame).
    FFmpeg never starts, the video's file is never touched and the audio is left out.
    Settings the video would refuse raise as they would for the video (its folder, the
    Additional parameters, then its first frame). A second frame is an error: the loop
    runs one item."""
    if simplicity is Simplicity.SIMPLE:
        container, encoder, video_preset, crf = get_simple_format(
            simple_video_format, quality
        )
    video = (save_dir / f"{video_name}.{container.ext}").resolve()
    video.parent.mkdir(parents=True, exist_ok=True)
    params, global_params = graph().video_output_options(
        globals(),
        str(video),
        container,
        encoder,
        video_preset,
        crf,
        fps,
        simplicity,
        additional_parameters,
    )
    if audio is not None and container is not VideoFormat.GIF:
        logger.info(
            "Save Video saves the one chosen frame %d as a PNG, without the audio.",
            item,
        )
    writer = Writer(
        container=container,
        encoder=encoder,
        fps=fps,
        audio=None,
        audio_settings=audio_settings,
        save_path=str(video),
        output_params=params,
        global_params=global_params,
        # Never started (only its checks run), so FFmpeg's paths are never used.
        ffmpeg_env=FFMpegEnv.from_env(),
        target=None,
        settings=settings,
    )
    path = (save_dir / f"{video_name}_f{item:06d}.png").resolve()
    frames = 0

    def iterate(frame: object) -> None:
        nonlocal frames
        frames += 1
        if frames > 1:
            raise RuntimeError(
                f"Save Video got a second frame in a run of the one chosen frame {item}:"
                " the loop's generator gave more than its one item."
            )
        writer.save_frame(frame, path)

    return FrameCollector(on_iterate=iterate, on_complete=lambda: None)


@video_frames_group.register(
    schema_id="chainner:image:save_video",
    name="Save Video",
    description=[
        "Combines an iterable sequence into a video, which it saves to a file.",
        "Uses FFMPEG to write video files.",
        "This iterator is much slower than just using FFMPEG directly, so if you are doing a simple conversion, just use FFMPEG outside chaiNNer instead.",
    ],
    icon="MdVideoCameraBack",
    inputs=[
        if_enum_group(19, VideoInputMode.RGB)(
            required()(ImageInput("Image Sequence", channels=3).make_optional())
        ),
        DirectoryInput(must_exist=False),
        RelativePathInput("Video Name"),
        EnumInput(
            VideoInputMode,
            "Input Mode",
            default=VideoInputMode.RGB,
            option_labels={VideoInputMode.RGB: "RGB", VideoInputMode.PLANES: "Planes"},
            preferred_style="tabs",
        )
        .with_id(19)
        .with_docs(
            "RGB saves RGB images. Planes saves Y, U and V planes (Load Video's Planes"
            " mode) as they are, at the bit depth chosen below: equal sizes are 4:4:4,"
            " U and V of half Y's width 4:2:2, of half its width and height 4:2:0."
        ),
        if_enum_group(19, VideoInputMode.PLANES)(
            required()(
                ImageInput("Y", channels=1).make_optional().with_id(20),
                ImageInput("U", channels=1).make_optional().with_id(21),
                ImageInput("V", channels=1).make_optional().with_id(22),
            )
        ),
        EnumInput(Simplicity, default=Simplicity.SIMPLE, preferred_style="tabs")
        .with_id(16)
        .with_docs(
            "Simple mode offers a more user-friendly interface, while advanced mode allows for more customization of the underlying FFMPEG options."
        ),
        if_enum_group(16, Simplicity.ADVANCED)(
            EnumInput(
                VideoFormat,
                label="Format",
                label_style="inline",
                option_labels={
                    VideoFormat.MKV: "mkv",
                    VideoFormat.MP4: "mp4",
                    VideoFormat.MOV: "mov",
                    VideoFormat.WEBM: "WebM",
                    VideoFormat.AVI: "avi",
                    VideoFormat.GIF: "GIF",
                },
            ).with_id(4),
            EnumInput(
                VideoEncoder,
                label="Encoder",
                label_style="inline",
                option_labels={
                    VideoEncoder.H264: "H.264 (AVC)",
                    VideoEncoder.H265: "H.265 (HEVC)",
                    VideoEncoder.VP9: "VP9",
                    VideoEncoder.FFV1: "FFV1",
                },
                conditions={
                    VideoEncoder.H264: Condition.enum(4, VideoEncoder.H264.formats),
                    VideoEncoder.H265: Condition.enum(4, VideoEncoder.H265.formats),
                    VideoEncoder.VP9: Condition.enum(4, VideoEncoder.VP9.formats),
                    VideoEncoder.FFV1: Condition.enum(4, VideoEncoder.FFV1.formats),
                },
            )
            .with_id(3)
            .wrap_with_conditional_group(),
            if_enum_group(3, (VideoEncoder.H264, VideoEncoder.H265))(
                EnumInput(
                    VideoPreset,
                    label="Preset",
                    label_style="inline",
                    default=VideoPreset.MEDIUM,
                )
                .with_docs(
                    "For more information on presets, see [here](https://trac.ffmpeg.org/wiki/Encode/H.264#Preset)."
                )
                .with_id(8),
            ),
            if_enum_group(3, (VideoEncoder.H264, VideoEncoder.H265, VideoEncoder.VP9))(
                SliderInput(
                    "CRF",
                    min=0,
                    max=51,
                    default=23,
                    ends=("Best", "Worst"),
                )
                .with_docs(
                    "For more information on CRF, see [here](https://trac.ffmpeg.org/wiki/Encode/H.264#crf)."
                )
                .with_id(9),
            ),
            TextInput(
                "Additional parameters",
                multiline=True,
                allow_empty_string=True,
                has_handle=False,
            )
            .make_optional()
            .with_docs(
                "Allow adding extra FFmpeg parameters the same way you would in CLI. [Link to FFmpeg documentation](https://ffmpeg.org/documentation.html).",
                hint=True,
            )
            .with_id(13),
        ),
        if_enum_group(16, Simplicity.SIMPLE)(
            EnumInput(
                SimpleVideoFormat,
                label="Video Format",
                option_labels={
                    SimpleVideoFormat.MP4_H264: "MP4 (H.264/AVC)",
                    SimpleVideoFormat.MP4_H265: "MP4 (H.265/HEVC)",
                    SimpleVideoFormat.WEBM: "WebM",
                    SimpleVideoFormat.GIF: "GIF",
                },
            ).with_id(17),
            if_group(~Condition.enum(17, SimpleVideoFormat.GIF))(
                SliderInput("Quality", min=0, max=100, default=75).with_id(18),
            ),
        ),
        NumberInput("FPS", default=30, min=1, step=1, precision=4).with_id(14),
        if_group(~Condition.enum(4, VideoFormat.GIF))(
            AudioStreamInput().make_optional().with_id(15).suggest(),
            if_group(Condition.type(15, "AudioStream"))(
                EnumInput(
                    AudioSettings,
                    label="Audio",
                    default=AudioSettings.AUTO,
                    conditions={
                        AudioSettings.COPY: ~Condition.enum(4, VideoFormat.WEBM)
                    },
                    label_style="inline",
                )
                .with_docs(
                    "The first audio stream can be discarded, copied or transcoded at 320 kb/s."
                    " Some audio formats are not supported by selected container, thus copying the audio may fail."
                    " Some players may not output the audio stream if its format is not supported."
                    " If it isn't working for you, verify compatibility or use FFMPEG to mux the audio externally."
                )
                .with_id(10)
            ),
        ),
        ColourInfoInput()
        .make_optional()
        .with_id(23)
        .with_docs(
            "The source's colour information (Load Video's Colour Info): the colour"
            " tags, the bit depth for Auto and the source's plane sizes for the chroma"
            " siting correction."
        ),
        *colour_override_inputs(24),
        EnumInput(
            BitDepth,
            "Bit Depth",
            default=BitDepth.AUTO,
            option_labels={
                BitDepth.AUTO: "Auto",
                BitDepth.BITS_8: "8-bit",
                BitDepth.BITS_10: "10-bit",
                BitDepth.BITS_12: "12-bit",
                BitDepth.BITS_16: "16-bit",
            },
        )
        .with_id(29)
        .with_docs(
            "Auto writes the Colour Info's bit depth, rounded up to one the encoder"
            " writes (FFV1: 8, 9, 10, 12, 14 or 16 bits), at most 8 bits for H.264 and 10"
            " for H.265 and VP9 (what players commonly decode), and 8 bits without"
            " Colour Info. 16-bit needs FFV1."
        ),
        if_enum_group(19, VideoInputMode.PLANES)(
            EnumInput(
                SitingCorrection,
                "Chroma Siting Correction",
                default=SitingCorrection.AUTO,
            )
            .with_id(30)
            .with_docs(
                "Planes resized on their own (an upscale) keep the source's chroma"
                " positions only for centre-sited chroma. Auto moves U and V to where"
                " the saved chroma siting places them when the Colour Info says the"
                " source's chroma was not centre-sited and the chroma grew; On always"
                " realigns by the Colour Info's plane sizes, Off never."
            ),
        ),
    ],
    iterator_inputs=IteratorInputInfo(inputs=[0, 20, 21, 22]),
    outputs=[],
    key_info=KeyInfo.enum(4),
    kind="collector",
    side_effects=True,
    node_context=True,
)
def save_video_node(
    node_context: NodeContext,
    _: None,
    save_dir: Path,
    video_name: str,
    mode: VideoInputMode,
    _y: None,
    _u: None,
    _v: None,
    simplicity: Simplicity,
    container: VideoFormat,
    encoder: VideoEncoder,
    video_preset: VideoPreset,
    crf: int,
    additional_parameters: str | None,
    simple_video_format: SimpleVideoFormat,
    quality: int,
    fps: float,
    audio: Any,
    audio_settings: AudioSettings,
    colour_info: ColourInfo | None,
    matrix: Matrix,
    transfer: Transfer,
    primaries: Primaries,
    range_: Range,
    chroma_location: ChromaLocation,
    bit_depth: BitDepth,
    siting_correction: SitingCorrection,
) -> Collector[tuple[np.ndarray | None, ...], None]:
    settings = WriteSettings(
        mode,
        colour_info,
        (matrix, transfer, primaries, range_, chroma_location),
        bit_depth,
        siting_correction,
    )
    chosen = node_context.chosen_item
    if chosen is not None:
        return frame_collector(
            chosen,
            save_dir,
            video_name,
            simplicity,
            container,
            encoder,
            video_preset,
            crf,
            additional_parameters,
            simple_video_format,
            quality,
            fps,
            audio,
            audio_settings,
            settings,
        )
    return graph().video_save(
        globals(),
        node_context,
        _,
        save_dir,
        video_name,
        simplicity,
        container,
        encoder,
        video_preset,
        crf,
        additional_parameters,
        simple_video_format,
        quality,
        fps,
        audio,
        audio_settings,
        settings,
    )


def prepare_video_frame(
    collector: VideoCollector | FrameCollector, inputs: list[np.ndarray | None]
) -> np.ndarray | None:
    """The frame Save Video would pipe for this iteration in RGB mode (SP3-P9), in its
    writer's pipe format (fixed by the container and parameters in RGB mode); in Planes
    mode, without an image, or in a run of one chosen frame (its PNG), None: the node
    runs whole."""
    if isinstance(collector, FrameCollector):
        return None
    writer = collector.writer
    settings = writer.settings
    if inputs[0] is None or (
        settings is not None and settings.mode is VideoInputMode.PLANES
    ):
        return None
    return writer.payload(tuple(inputs))


def commit_video_frame(
    collector: VideoCollector, inputs: list[np.ndarray | None], prepared: Prepared
) -> None:
    """Save Video's iteration with its prepared frame as the pipe payload."""
    collector.writer.write_frame(tuple(inputs), prepared)


register_phases("chainner:image:save_video", prepare_video_frame, commit_video_frame)
