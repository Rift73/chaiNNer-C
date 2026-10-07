"""C text layout with the installed native FreeType/RAQM glyph engine."""

from __future__ import annotations

import ctypes as ct
from functools import lru_cache

import numpy as np
from PIL import ImageDraw, ImageFont

from .native import check, lib


@lru_cache(maxsize=1)
def _api():
    dll = lib()
    v, n, d, i = ct.c_void_p, ct.c_size_t, ct.c_double, ct.c_int
    signatures = {
        "cn_text_scan": [v, n, v],
        "cn_text_ranges": [v, n, v, n],
        "cn_text_fit": [d, d, d, d, n, v],
        "cn_caption_fit": [d, d, d, v],
        "cn_text_anchor": [d, d, d, d, i, v],
        "cn_text_positions": [v, n, d, d, d, i, v],
    }
    for name, signature in signatures.items():
        function = getattr(dll, name)
        function.argtypes = signature
        function.restype = ct.c_int
    return dll


def scan(text: str) -> tuple[list[str], str]:
    codepoints = np.require(
        np.frombuffer(
            text.encode("utf-32-le", errors="surrogatepass"), dtype=np.uint32
        ),
        requirements=["C", "A"],
    )
    summary = np.empty(3, np.uintp)
    check(
        _api().cn_text_scan(
            codepoints.ctypes.data, codepoints.size, summary.ctypes.data
        )
    )
    count, begin, length = (int(v) for v in summary)
    ranges = np.empty((count, 2), np.uintp)
    check(
        _api().cn_text_ranges(
            codepoints.ctypes.data, codepoints.size, ranges.ctypes.data, count
        )
    )
    return [text[int(b) : int(b + n)] for b, n in ranges], text[begin : begin + length]


def fit(
    width: int, height: int, ref_width: float, ref_height: float, lines: int
) -> int:
    if ref_width == 0 or ref_height == 0:
        # Upstream's own float divisions, in its order, so the interpreter raises its
        # ZeroDivisionError ("division by zero" on CPython 3.14).
        int(width * 100.0 / ref_width)
        int(height * 100.0 / (ref_height * lines))
    result = ct.c_double()
    check(
        _api().cn_text_fit(
            width, height, ref_width, ref_height, lines, ct.byref(result)
        )
    )
    return int(result.value)


def caption_fit(height: int, width: int, measured: float = 0) -> int:
    result = ct.c_double()
    check(_api().cn_caption_fit(height, width, measured, ct.byref(result)))
    return int(result.value)


def anchor(
    width: int, height: int, text_width: float, text_height: float, position: int
) -> tuple[float, float]:
    xy = np.empty(2, np.float64)
    check(
        _api().cn_text_anchor(
            width, height, text_width, text_height, position, xy.ctypes.data
        )
    )
    return float(xy[0]), float(xy[1])


def draw_lines(
    drawing: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    alignment: str,
) -> None:
    if len(lines) == 1:
        drawing.text(xy, lines[0], font=font, anchor="mm", align=alignment, fill=255)
        return
    widths = np.array(
        [drawing.textlength(line, font=font) for line in lines], np.float64
    )
    positions = np.empty((len(lines), 2), np.float64)
    spacing = font.getbbox("A")[3] + 4
    check(
        _api().cn_text_positions(
            widths.ctypes.data,
            len(lines),
            *xy,
            spacing,
            {"left": 0, "center": 1, "right": 2}[alignment],
            positions.ctypes.data,
        )
    )
    # Each public single-line call goes to the same installed native glyph
    # engine. Multiline fitting/positions above are computed in C.
    for line, position in zip(lines, positions, strict=True):
        drawing.text(
            (float(position[0]), float(position[1])),
            line,
            font=font,
            anchor="mm",
            align=alignment,
            fill=255,
        )
