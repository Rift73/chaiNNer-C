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

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np


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


def code_to_float(
    plane: np.ndarray, bit_depth: int, full_range: bool, chroma: bool
) -> np.ndarray:
    """A plane's stored codes (uint8 or uint16) as float32 plane values."""
    offset, scale = _scale(bit_depth, full_range, chroma)
    out = (plane.astype(np.float32) - np.float32(offset)) / np.float32(scale)
    if chroma:
        out += np.float32(0.5)
    return out


def float_to_code(
    plane: np.ndarray, bit_depth: int, full_range: bool, chroma: bool
) -> np.ndarray:
    """Float32 plane values as codes, rounded to nearest and clipped to the code range:
    uint8 up to 8 bits, else uint16."""
    offset, scale = _scale(bit_depth, full_range, chroma)
    values = plane.astype(np.float32)
    if chroma:
        values = values - np.float32(0.5)
    codes = np.rint(values * np.float32(scale) + np.float32(offset))
    np.clip(codes, 0, (1 << bit_depth) - 1, out=codes)
    return codes.astype(np.uint8 if bit_depth <= 8 else np.uint16)
