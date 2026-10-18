"""VapourSynth's colour frame properties (H.273 codes) and format names under FFmpeg's
names (those of nodes.impl.colour_info): a clip's first frame as the ffprobe entry that
detect_colour_info reads. Nothing here imports VapourSynth."""

from __future__ import annotations

import re

# FFmpeg's names of every H.273 code it knows (libavutil's av_color_*_name); detection
# warns about the names chaiNNer does not convert. 2 (unspecified) is no tag.
MATRIX_NAMES = {
    0: "gbr",
    1: "bt709",
    4: "fcc",
    5: "bt470bg",
    6: "smpte170m",
    7: "smpte240m",
    8: "ycgco",
    9: "bt2020nc",
    10: "bt2020c",
    11: "smpte2085",
    12: "chroma-derived-nc",
    13: "chroma-derived-c",
    14: "ictcp",
}
TRANSFER_NAMES = {
    1: "bt709",
    4: "bt470m",
    5: "bt470bg",
    6: "smpte170m",
    7: "smpte240m",
    8: "linear",
    9: "log100",
    10: "log316",
    11: "iec61966-2-4",
    12: "bt1361e",
    13: "iec61966-2-1",
    14: "bt2020-10",
    15: "bt2020-12",
    16: "smpte2084",
    17: "smpte428",
    18: "arib-std-b67",
}
PRIMARIES_NAMES = {
    1: "bt709",
    4: "bt470m",
    5: "bt470bg",
    6: "smpte170m",
    7: "smpte240m",
    8: "film",
    9: "bt2020",
    10: "smpte428",
    11: "smpte431",
    12: "smpte432",
    22: "jedec-p22",
}
# VapourSynth's _ChromaLocation (FFmpeg's AVChromaLocation minus 1).
CHROMA_LOCATION_NAMES = {
    0: "left",
    1: "center",
    2: "topleft",
    3: "top",
    4: "bottomleft",
    5: "bottom",
}
# The frame property of each ffprobe field. _Range is R81's _ColorRange (which it
# translates), 1 for full range.
PROPS = {
    "color_space": ("_Matrix", MATRIX_NAMES),
    "color_transfer": ("_Transfer", TRANSFER_NAMES),
    "color_primaries": ("_Primaries", PRIMARIES_NAMES),
    "color_range": ("_Range", {0: "tv", 1: "pc"}),
    "chroma_location": ("_ChromaLocation", CHROMA_LOCATION_NAMES),
}
UNSPECIFIED = 2
_FORMAT_NAME = re.compile(r"(Gray|RGB|YUV(\d{3})P)(\d+|[HS])")


def vs_pix_fmt(format_name: str) -> str:
    """FFmpeg's pixel format of a VapourSynth format name ("YUV420P10": yuv420p10le,
    "RGB24": gbrp), "" for none (a clip whose format varies). Float formats (final H
    or S) count as 16-bit, as colour_info.pixel_layout counts FFmpeg's."""
    match = _FORMAT_NAME.fullmatch(format_name)
    if match is None:
        return ""
    family, layout, depth = match[1], match[2], match[3]
    bits = 16 if depth in ("H", "S") else int(depth) // (3 if family == "RGB" else 1)
    suffix = f"{bits}le" if bits > 8 else ""
    if layout is not None:
        return f"yuv{layout}p{suffix}"
    return f"{'gray' if family == 'Gray' else 'gbrp'}{suffix}"


def colour_stream(
    props: dict[str, int], pix_fmt: str, width: int, height: int
) -> dict[str, object]:
    """The ffprobe entry of a clip whose first frame has the colour properties `props`
    (those of PROPS), for detect_colour_info. A code that FFmpeg has no name for keeps
    its number, so detection names it in its warning."""
    stream: dict[str, object] = {"pix_fmt": pix_fmt, "width": width, "height": height}
    for field, (prop, names) in PROPS.items():
        code = props.get(prop)
        if code is None or (code == UNSPECIFIED and prop != "_ChromaLocation"):
            continue
        stream[field] = names.get(code, f"H.273 code {code}")
    return stream
