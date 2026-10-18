"""Colour information of video planes, and how planes are stored as chaiNNer images.

A plane is a float32 HxW image at its stored size (4:2:0 1080p: Y 1920x1080, U and V
960x540). Y is range-expanded, 0 = black and 1 = white; U and V are centred on 0.5.
Float keeps values outside 0-1, and `float_to_code` inverts `code_to_float` exactly for
every code, so a plane round trip loses nothing.

The colour values use FFmpeg's names (what ffprobe reports and -colorspace, -color_trc,
-color_primaries, -color_range and -chroma_sample_location take), so a file is tagged
with them directly. Each enum's AUTO member is a dropdown choice only, never a value of a
ColourInfo."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import TypeVar

import numpy as np

from . import native_planes

E = TypeVar("E", bound=Enum)


class Matrix(Enum):
    AUTO = "auto"
    BT709 = "bt709"
    BT470BG = "bt470bg"
    SMPTE170M = "smpte170m"
    BT2020_NCL = "bt2020nc"
    BT2020_CL = "bt2020c"
    SMPTE240M = "smpte240m"
    FCC = "fcc"
    YCGCO = "ycgco"
    ICTCP = "ictcp"
    RGB = "gbr"


class Transfer(Enum):
    AUTO = "auto"
    BT709 = "bt709"
    SMPTE170M = "smpte170m"
    BT470M = "bt470m"
    BT470BG = "bt470bg"
    SMPTE240M = "smpte240m"
    LINEAR = "linear"
    SRGB = "iec61966-2-1"
    XVYCC = "iec61966-2-4"
    BT2020_10 = "bt2020-10"
    BT2020_12 = "bt2020-12"
    PQ = "smpte2084"
    HLG = "arib-std-b67"
    SMPTE428 = "smpte428"


class Primaries(Enum):
    AUTO = "auto"
    BT709 = "bt709"
    BT470M = "bt470m"
    BT470BG = "bt470bg"
    SMPTE170M = "smpte170m"
    SMPTE240M = "smpte240m"
    FILM = "film"
    BT2020 = "bt2020"
    SMPTE428 = "smpte428"
    DCI_P3 = "smpte431"
    DISPLAY_P3 = "smpte432"
    EBU3213 = "jedec-p22"


class Range(Enum):
    AUTO = "auto"
    LIMITED = "tv"
    FULL = "pc"


class ChromaLocation(Enum):
    AUTO = "auto"
    LEFT = "left"
    CENTER = "center"
    TOP_LEFT = "topleft"
    TOP = "top"
    BOTTOM_LEFT = "bottomleft"
    BOTTOM = "bottom"


class VideoMode(Enum):
    """What a video node reads or writes: RGB images, planes, or both."""

    RGB = "rgb"
    PLANES = "planes"
    BOTH = "both"


VIDEO_MODE_LABELS = {
    VideoMode.RGB: "RGB",
    VideoMode.PLANES: "Planes",
    VideoMode.BOTH: "Both",
}

MATRIX_LABELS = {
    Matrix.AUTO: "Auto",
    Matrix.BT709: "BT.709",
    Matrix.BT470BG: "BT.601 (625 lines, PAL)",
    Matrix.SMPTE170M: "BT.601 (525 lines, NTSC)",
    Matrix.BT2020_NCL: "BT.2020 non-constant",
    Matrix.BT2020_CL: "BT.2020 constant",
    Matrix.SMPTE240M: "SMPTE 240M",
    Matrix.FCC: "FCC",
    Matrix.YCGCO: "YCgCo",
    Matrix.ICTCP: "ICtCp",
    Matrix.RGB: "RGB (identity)",
}
TRANSFER_LABELS = {
    Transfer.AUTO: "Auto",
    Transfer.BT709: "BT.709 (BT.1886)",
    Transfer.SMPTE170M: "BT.601 (SMPTE 170M)",
    Transfer.BT470M: "Gamma 2.2 (BT.470 M)",
    Transfer.BT470BG: "Gamma 2.8 (BT.470 BG)",
    Transfer.SMPTE240M: "SMPTE 240M",
    Transfer.LINEAR: "Linear",
    Transfer.SRGB: "sRGB",
    Transfer.XVYCC: "xvYCC (IEC 61966-2-4)",
    Transfer.BT2020_10: "BT.2020 10-bit",
    Transfer.BT2020_12: "BT.2020 12-bit",
    Transfer.PQ: "PQ (SMPTE 2084, HDR10)",
    Transfer.HLG: "HLG (ARIB STD-B67)",
    Transfer.SMPTE428: "SMPTE 428 (DCDM)",
}
PRIMARIES_LABELS = {
    Primaries.AUTO: "Auto",
    Primaries.BT709: "BT.709",
    Primaries.BT470M: "BT.470 M",
    Primaries.BT470BG: "BT.601 (625 lines, PAL)",
    Primaries.SMPTE170M: "BT.601 (525 lines, NTSC)",
    Primaries.SMPTE240M: "SMPTE 240M",
    Primaries.FILM: "Film (C)",
    Primaries.BT2020: "BT.2020",
    Primaries.SMPTE428: "SMPTE 428 (XYZ)",
    Primaries.DCI_P3: "DCI-P3",
    Primaries.DISPLAY_P3: "Display P3",
    Primaries.EBU3213: "EBU 3213-E",
}
RANGE_LABELS = {
    Range.AUTO: "Auto",
    Range.LIMITED: "Limited (TV)",
    Range.FULL: "Full (PC)",
}
CHROMA_LOCATION_LABELS = {
    ChromaLocation.AUTO: "Auto",
    ChromaLocation.LEFT: "Left",
    ChromaLocation.CENTER: "Center",
    ChromaLocation.TOP_LEFT: "Top left",
    ChromaLocation.TOP: "Top",
    ChromaLocation.BOTTOM_LEFT: "Bottom left",
    ChromaLocation.BOTTOM: "Bottom",
}

# libswscale's chroma positions (in_h_chr_pos/in_v_chr_pos and out_*), in 1/256 of a
# luma pixel from the top-left luma sample of the chroma sample's block; it otherwise
# assumes centre siting.
SWSCALE_CHROMA_POSITION = {
    ChromaLocation.LEFT: (0, 128),
    ChromaLocation.CENTER: (128, 128),
    ChromaLocation.TOP_LEFT: (0, 0),
    ChromaLocation.TOP: (128, 0),
    ChromaLocation.BOTTOM_LEFT: (0, 256),
    ChromaLocation.BOTTOM: (128, 256),
}

# The scale filter's in_color_matrix and out_color_matrix names by FFmpeg's matrix names
# (ffprobe's, and -colorspace's aliases); libswscale converts no other matrix, and it
# converts BT.2020 constant luminance with the non-constant coefficients.
_SWSCALE_MATRICES = {
    "bt709": "bt709",
    "bt470bg": "bt470",
    "smpte170m": "smpte170m",
    "fcc": "fcc",
    "smpte240m": "smpte240m",
    "bt2020nc": "bt2020",
    "bt2020_ncl": "bt2020",
    "bt2020c": "bt2020",
    "bt2020_cl": "bt2020",
}


def swscale_matrix(name: str) -> str | None:
    """The scale filter's name for the matrix FFmpeg calls `name`, or None for a matrix
    libswscale does not convert."""
    return _SWSCALE_MATRICES.get(name)


_OPTION_VALUES: dict[Enum, str] = {
    Matrix.RGB: "rgb",
    Transfer.BT470M: "gamma22",
    Transfer.BT470BG: "gamma28",
}


def option_value(value: Enum) -> str:
    """The value FFmpeg's -colorspace, -color_trc, -color_primaries, -color_range and
    -chroma_sample_location take for a ColourInfo field: ffprobe's name, except for
    the three values whose option names differ."""
    return _OPTION_VALUES.get(value, str(value.value))


ORIGINS = ("file", "guessed", "set")
FIELDS = ("matrix", "transfer", "primaries", "range", "chroma_location")


@dataclass(frozen=True)
class HdrInfo:
    """HDR side data: FFmpeg's mastering display string
    ("G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)") and content light levels in cd/m2."""

    mastering_display: str | None = None
    max_cll: int | None = None
    max_fall: int | None = None


@dataclass(frozen=True)
class ColourInfo:
    """How a video's planes are coded. `luma_size` and `chroma_size` (width, height) are
    the plane sizes of the node that issued it, so a later node can tell resized planes
    from the source's (Save Video's chroma siting correction). `origin` gives each of
    FIELDS as "file" (tagged in the file), "guessed" or "set" (an override)."""

    matrix: Matrix
    transfer: Transfer
    primaries: Primaries
    range: Range
    chroma_location: ChromaLocation
    bit_depth: int
    luma_size: tuple[int, int]
    chroma_size: tuple[int, int]
    hdr: HdrInfo | None = None
    origin: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in FIELDS:
            if getattr(self, name).value == "auto":
                raise ValueError(f"A ColourInfo's {name} cannot be Auto.")
        if not 1 <= self.bit_depth <= 16:
            raise ValueError(f"Bit depth {self.bit_depth} is outside 1-16.")
        for name, origin in self.origin:
            if name not in FIELDS or origin not in ORIGINS:
                raise ValueError(f"Invalid origin {name}={origin}.")

    @property
    def subsampling(self) -> tuple[int, int]:
        """log2 of the chroma subsampling (width, height): 4:2:0 is (1, 1)."""
        return subsampling_from_sizes(self.luma_size, self.chroma_size)

    @property
    def full_range(self) -> bool:
        return self.range is Range.FULL

    def origin_of(self, name: str) -> str:
        return dict(self.origin).get(name, "guessed")

    def with_choices(
        self,
        matrix: Matrix = Matrix.AUTO,
        transfer: Transfer = Transfer.AUTO,
        primaries: Primaries = Primaries.AUTO,
        range_: Range = Range.AUTO,
        chroma_location: ChromaLocation = ChromaLocation.AUTO,
    ) -> ColourInfo:
        """This info with every non-Auto choice applied, its origin "set"."""
        chosen = {
            "matrix": matrix,
            "transfer": transfer,
            "primaries": primaries,
            "range": range_,
            "chroma_location": chroma_location,
        }
        changes = {k: v for k, v in chosen.items() if v.value != "auto"}
        if not changes:
            return self
        origin = dict(self.origin)
        origin.update(dict.fromkeys(changes, "set"))
        return replace(self, **changes, origin=tuple(sorted(origin.items())))

    def summary(self) -> str:
        """One line for the node's Colour output, e.g. "Matrix BT.709, transfer BT.709
        (BT.1886), primaries BT.709, limited range, 4:2:0 left-sited, 10-bit (matrix
        guessed)"."""
        sx, sy = self.subsampling
        layout = {(0, 0): "4:4:4", (1, 0): "4:2:2", (1, 1): "4:2:0", (2, 0): "4:1:1"}
        text = (
            f"Matrix {MATRIX_LABELS[self.matrix]}, transfer"
            f" {TRANSFER_LABELS[self.transfer]}, primaries"
            f" {PRIMARIES_LABELS[self.primaries]},"
            f" {RANGE_LABELS[self.range].split()[0].lower()} range,"
            f" {layout.get((sx, sy), f'{1 << sx}x{1 << sy} subsampled')}"
        )
        if (sx, sy) != (0, 0):
            text += f" {CHROMA_LOCATION_LABELS[self.chroma_location].lower()}-sited"
        text += f", {self.bit_depth}-bit"
        notes = [
            f"{name.replace('_', ' ')} {self.origin_of(name)}"
            for name in FIELDS
            if self.origin_of(name) != "file"
        ]
        return text + (f" ({', '.join(notes)})" if notes else "")


def subsampling_from_sizes(
    luma_size: tuple[int, int], chroma_size: tuple[int, int]
) -> tuple[int, int]:
    """log2 (width, height) subsampling of chroma planes `chroma_size` beside a luma plane
    `luma_size` (sizes as (width, height); a subsampled odd size rounds up)."""
    result = []
    for luma, chroma, axis in zip(luma_size, chroma_size, ("width", "height")):
        for shift in range(3):
            if chroma == -(-luma // (1 << shift)):
                result.append(shift)
                break
        else:
            raise ValueError(
                f"A chroma plane {chroma_size[0]}x{chroma_size[1]} does not fit a luma plane"
                f" {luma_size[0]}x{luma_size[1]}: its {axis} must be the luma's, half"
                " or a quarter of it."
            )
    return result[0], result[1]


def chroma_size_for(
    luma_size: tuple[int, int], subsampling: tuple[int, int]
) -> tuple[int, int]:
    return (
        -(-luma_size[0] // (1 << subsampling[0])),
        -(-luma_size[1] // (1 << subsampling[1])),
    )


def _scale(bit_depth: int, full_range: bool, chroma: bool) -> tuple[float, float]:
    """(offset, scale) with plane value = (code - offset) / scale (+ 0.5 for chroma)."""
    if full_range:
        top = float((1 << bit_depth) - 1)
        return (float(1 << (bit_depth - 1)) if chroma else 0.0), top
    step = (
        float(1 << (bit_depth - 8)) if bit_depth >= 8 else 1.0 / (1 << (8 - bit_depth))
    )
    return (128.0 if chroma else 16.0) * step, (224.0 if chroma else 219.0) * step


_CODE_DTYPES = (np.dtype(np.uint8), np.dtype(np.uint16))


def _native_depth(bit_depth: int) -> bool:
    return 8 <= bit_depth <= 16


def code_to_float(
    plane: np.ndarray, bit_depth: int, full_range: bool, chroma: bool
) -> np.ndarray:
    """A plane's stored codes (uint8 or uint16) as float32 plane values."""
    if plane.dtype in _CODE_DTYPES and _native_depth(bit_depth):
        offset, scale = _scale(bit_depth, full_range, chroma)
        return native_planes.codes_to_float(plane, offset, scale, chroma)
    return _numpy_code_to_float(plane, bit_depth, full_range, chroma)


def float_to_code(
    plane: np.ndarray, bit_depth: int, full_range: bool, chroma: bool
) -> np.ndarray:
    """Float32 plane values as codes, rounded to nearest and clipped to the code range:
    uint8 up to 8 bits, else uint16."""
    if _native_depth(bit_depth):
        offset, scale = _scale(bit_depth, full_range, chroma)
        return native_planes.float_to_codes(plane, offset, scale, chroma, bit_depth)
    return _numpy_float_to_code(plane, bit_depth, full_range, chroma)


# The NumPy forms: the reference that the C kernels (native_planes.py) match bit for
# bit, and the path for other dtypes and bit depths.
def _numpy_code_to_float(
    plane: np.ndarray, bit_depth: int, full_range: bool, chroma: bool
) -> np.ndarray:
    offset, scale = _scale(bit_depth, full_range, chroma)
    out = (plane.astype(np.float32) - np.float32(offset)) / np.float32(scale)
    if chroma:
        out += np.float32(0.5)
    return out


def _numpy_float_to_code(
    plane: np.ndarray, bit_depth: int, full_range: bool, chroma: bool
) -> np.ndarray:
    offset, scale = _scale(bit_depth, full_range, chroma)
    values = plane.astype(np.float32)
    if chroma:
        values = values - np.float32(0.5)
    codes = np.rint(values * np.float32(scale) + np.float32(offset))
    np.clip(codes, 0, (1 << bit_depth) - 1, out=codes)
    return codes.astype(np.uint8 if bit_depth <= 8 else np.uint16)


@dataclass(frozen=True)
class PixelLayout:
    """A pixel format's coding: `kind` "yuv", "gray" or "rgb" ("other" for formats
    chaiNNer does not know), its bit depth and chroma subsampling (log2 width, height),
    `planar` the planar format its planes are read in (None when they cannot be: float
    and unknown formats) and `jpeg` for FFmpeg's full-range yuvj formats."""

    kind: str
    bit_depth: int
    subsampling: tuple[int, int]
    planar: str | None
    jpeg: bool = False


_SUBSAMPLING = {
    "444": (0, 0),
    "422": (1, 0),
    "420": (1, 1),
    "440": (0, 1),
    "411": (2, 0),
    "410": (2, 2),
}
# Semi-planar and packed YUV formats (without their le/be suffix): their layout and
# bit depth; their planes are read in the planar format of the same layout and depth.
_INTERLEAVED_YUV = {
    "nv12": ("420", 8),
    "nv21": ("420", 8),
    "nv16": ("422", 8),
    "nv20": ("422", 10),
    "nv24": ("444", 8),
    "nv42": ("444", 8),
    "p010": ("420", 10),
    "p012": ("420", 12),
    "p016": ("420", 16),
    "p210": ("422", 10),
    "p212": ("422", 12),
    "p216": ("422", 16),
    "p410": ("444", 10),
    "p412": ("444", 12),
    "p416": ("444", 16),
    "yuyv422": ("422", 8),
    "uyvy422": ("422", 8),
    "yvyu422": ("422", 8),
    "y210": ("422", 10),
    "y212": ("422", 12),
    "xv30": ("444", 10),
    "xv36": ("444", 12),
    "vuya": ("444", 8),
    "vuyx": ("444", 8),
    "ayuv64": ("444", 16),
    "uyyvyy411": ("411", 8),
}


def _planar_yuv(layout: str, bit_depth: int, jpeg: bool = False) -> str:
    depth = f"{bit_depth}le" if bit_depth > 8 else ""
    return f"yuv{'j' if jpeg else ''}{layout}p{depth}"


def pixel_layout(pix_fmt: str) -> PixelLayout:
    """The coding of an FFmpeg pixel format (ffprobe's pix_fmt)."""
    planar = re.fullmatch(r"yuva?(j?)(4[0-4][0-4])p(\d*)(le|be)?", pix_fmt)
    if planar is not None and planar[2] in _SUBSAMPLING:
        jpeg, layout = planar[1] == "j", planar[2]
        bit_depth = int(planar[3] or 8)
        return PixelLayout(
            "yuv",
            bit_depth,
            _SUBSAMPLING[layout],
            _planar_yuv(layout, bit_depth, jpeg),
            jpeg,
        )
    base = re.sub(r"(le|be)$", "", pix_fmt)
    if base in _INTERLEAVED_YUV:
        layout, bit_depth = _INTERLEAVED_YUV[base]
        return PixelLayout(
            "yuv", bit_depth, _SUBSAMPLING[layout], _planar_yuv(layout, bit_depth)
        )
    gray = re.fullmatch(r"(?:gray|ya)(\d*)(f32)?", base)
    if gray is not None:
        if gray[2]:
            return PixelLayout("gray", 16, (0, 0), None)
        bit_depth = int(gray[1] or 8)
        planar_gray = "gray" if bit_depth == 8 else f"gray{bit_depth}le"
        return PixelLayout("gray", bit_depth, (0, 0), planar_gray)
    if re.match(r"(gbr|rgb|bgr|argb|abgr|0rgb|0bgr|x2rgb|x2bgr|pal8)", base):
        bit_depth = 8
        if "f32" in base or "f16" in base:
            bit_depth = 16
        elif base.startswith("gbr") and re.search(r"\d+$", base):
            bit_depth = int(re.findall(r"\d+$", base)[0])
        elif base.startswith(("rgb48", "bgr48", "rgba64", "bgra64")):
            bit_depth = 16
        elif base.startswith(("x2rgb10", "x2bgr10")):
            bit_depth = 10
        return PixelLayout("rgb", bit_depth, (0, 0), None)
    return PixelLayout("other", 8, (0, 0), None)


def high_definition(width: int, height: int) -> bool:
    """mpv's guess for untagged video: HD (BT.709) when at least 1280 wide or taller
    than 576 lines, else SD (BT.601)."""
    return width >= 1280 or height > 576


def size_guesses(width: int, height: int) -> tuple[Matrix, Primaries]:
    """mpv's and libplacebo's matrix and primaries for untagged YUV of a stored size:
    BT.709 when HD, else BT.601 (PAL's at 576 lines, NTSC's otherwise), and primaries
    by height (576 lines PAL, 480 or 486 NTSC, else BT.709)."""
    if high_definition(width, height):
        return Matrix.BT709, Primaries.BT709
    if height == 576:
        return Matrix.BT470BG, Primaries.BT470BG
    if height in (480, 486):
        return Matrix.SMPTE170M, Primaries.SMPTE170M
    return Matrix.SMPTE170M, Primaries.BT709


_UNTAGGED = (None, "", "unknown", "unspecified", "reserved")
FIELD_INPUTS = {
    "matrix": "Matrix",
    "transfer": "Transfer",
    "primaries": "Primaries",
    "range": "Range",
    "chroma_location": "Chroma Siting",
}


def _fraction(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    numerator, _, denominator = value.partition("/")
    try:
        return int(numerator) / int(denominator or 1)
    except (ValueError, ZeroDivisionError):
        return None


def _hdr_info(side_data: list[dict[str, object]]) -> HdrInfo | None:
    """HDR10 static metadata from ffprobe's side_data_list; the mastering display as
    FFmpeg and x265 write it, chromaticities in 1/50000 and luminance in 1/10000 cd/m2."""
    mastering: str | None = None
    max_cll: int | None = None
    max_fall: int | None = None
    present = False
    for entry in side_data:
        kind = entry.get("side_data_type")
        if kind == "Mastering display metadata":
            present = True
            names = ("green", "blue", "red", "white_point")
            coords = [
                _fraction(entry.get(f"{n}_{axis}")) for n in names for axis in "xy"
            ]
            luminance = [_fraction(entry.get(f"{k}_luminance")) for k in ("max", "min")]
            c = [round(v * 50000) for v in coords if v is not None]
            lum = [round(v * 10000) for v in luminance if v is not None]
            if len(c) == 8 and len(lum) == 2:
                mastering = (
                    f"G({c[0]},{c[1]})B({c[2]},{c[3]})R({c[4]},{c[5]})"
                    f"WP({c[6]},{c[7]})L({lum[0]},{lum[1]})"
                )
        elif kind == "Content light level metadata":
            present = True
            content, average = entry.get("max_content"), entry.get("max_average")
            max_cll = int(content) if isinstance(content, (int, str)) else None
            max_fall = int(average) if isinstance(average, (int, str)) else None
    if not present:
        return None
    return HdrInfo(mastering_display=mastering, max_cll=max_cll, max_fall=max_fall)


def _tag(
    stream: dict[str, object], key: str, name: str, enum: type[E], warnings: list[str]
) -> E | None:
    value = stream.get(key)
    if value in _UNTAGGED:
        return None
    text = {"bt2020_ncl": "bt2020nc", "bt2020_cl": "bt2020c"}.get(
        str(value), str(value)
    )
    for member in enum:
        if member.value == text and member.value != "auto":
            return member
    warnings.append(
        f"The video's {name.replace('_', ' ')} tag {text} is not one chaiNNer converts,"
        f" so it is guessed; set {FIELD_INPUTS[name]} to choose it."
    )
    return None


def detect_colour_info(stream: dict[str, object]) -> tuple[ColourInfo, list[str]]:
    """The ColourInfo of a video stream from its ffprobe entry (codec_name, pix_fmt,
    width, height, the colour tags and side_data_list), and warnings for the node to log
    about tags chaiNNer cannot use and implausible combinations.

    Each field takes the stream's tag (origin "file"), else is guessed as mpv and
    libplacebo guess (origin "guessed"): HDR side data without a transfer tag means PQ,
    without a primaries tag BT.2020; BT.2020 primaries or matrix, or a PQ or HLG
    transfer, mean a BT.2020 non-constant matrix and BT.2020 primaries; RGB formats are
    identity-coded, full range, BT.709 primaries and sRGB; YUV is BT.709 when HD and
    BT.601 otherwise (primaries by height: 576 lines PAL, 480 or 486 NTSC, else BT.709)
    with BT.709's transfer and limited range except the yuvj formats; gray is full range,
    as libswscale reads it; chroma is centre-sited at full range and for MJPEG and
    MPEG-1, else left-sited."""
    layout = pixel_layout(str(stream.get("pix_fmt") or ""))
    width, height = int(str(stream["width"])), int(str(stream["height"]))
    codec = str(stream.get("codec_name") or "")
    warnings: list[str] = []
    matrix = _tag(stream, "color_space", "matrix", Matrix, warnings)
    transfer = _tag(stream, "color_transfer", "transfer", Transfer, warnings)
    primaries = _tag(stream, "color_primaries", "primaries", Primaries, warnings)
    range_ = _tag(stream, "color_range", "range", Range, warnings)
    chroma_location = _tag(
        stream, "chroma_location", "chroma_location", ChromaLocation, warnings
    )
    tagged = (matrix, transfer, primaries, range_, chroma_location)
    origin: dict[str, str] = dict.fromkeys(FIELDS, "guessed")
    origin.update({k: "file" for k, v in zip(FIELDS, tagged) if v is not None})

    side_data = stream.get("side_data_list")
    hdr = _hdr_info(side_data if isinstance(side_data, list) else [])
    if hdr is not None:
        transfer = transfer or Transfer.PQ
        primaries = primaries or Primaries.BT2020
    if (
        primaries is Primaries.BT2020
        or matrix in (Matrix.BT2020_NCL, Matrix.BT2020_CL)
        or transfer in (Transfer.PQ, Transfer.HLG)
    ):
        matrix = matrix or Matrix.BT2020_NCL
        primaries = primaries or Primaries.BT2020
    if layout.kind == "rgb":
        matrix = matrix or Matrix.RGB
        range_ = range_ or Range.FULL
        primaries = primaries or Primaries.BT709
        transfer = transfer or Transfer.SRGB

    hd = high_definition(width, height)
    guessed_matrix, guessed_primaries = size_guesses(width, height)
    matrix = matrix or guessed_matrix
    primaries = primaries or guessed_primaries
    transfer = transfer or Transfer.BT709
    if range_ is None:
        range_ = Range.FULL if layout.jpeg or layout.kind == "gray" else Range.LIMITED
    if chroma_location is None:
        centred = range_ is Range.FULL or codec in ("mjpeg", "mpeg1video")
        chroma_location = ChromaLocation.CENTER if centred else ChromaLocation.LEFT

    if transfer in (Transfer.PQ, Transfer.HLG) and layout.bit_depth <= 8:
        warnings.append(
            f"The video's transfer is {TRANSFER_LABELS[transfer]} at"
            f" {layout.bit_depth} bits, which HDR video never is; if its colours look"
            " wrong, set Transfer."
        )
    for name, value in (("matrix", matrix), ("primaries", primaries)):
        if value in (Primaries.BT2020, Matrix.BT2020_NCL, Matrix.BT2020_CL) and not hd:
            warnings.append(
                f"The video's {name} is BT.2020 at {width}x{height}, which SD video"
                f" rarely is; if its colours look wrong, set {FIELD_INPUTS[name]}."
            )
        if (
            value in (Primaries.BT709, Matrix.BT709)
            and origin[name] == "file"
            and codec == "mpeg2video"
            and height == 576
        ):
            warnings.append(
                f"The video is 576-line MPEG-2 tagged with the BT.709 {name}, which PAL"
                f" video rarely is; if its colours look wrong, set {FIELD_INPUTS[name]}."
            )

    luma_size = (width, height)
    info = ColourInfo(
        matrix=matrix,
        transfer=transfer,
        primaries=primaries,
        range=range_,
        chroma_location=chroma_location,
        bit_depth=layout.bit_depth,
        luma_size=luma_size,
        chroma_size=chroma_size_for(luma_size, layout.subsampling),
        hdr=hdr,
        origin=tuple(sorted(origin.items())),
    )
    return info, warnings
