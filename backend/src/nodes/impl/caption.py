from __future__ import annotations

import os
import sys
from contextlib import ExitStack, nullcontext
from enum import Enum

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..utils.utils import get_h_w_c
from . import native_text
from .font_cache import font_lease
from .image_utils import as_target_channels, normalize
from .native_buffers import compose_caption


class CaptionPosition(Enum):
    BOTTOM = "bottom"
    TOP = "top"


def get_font_size(font: ImageFont.FreeTypeFont, text: str) -> tuple[float, float]:
    """Get font [width, height] of the given text"""
    caption_bb = font.getbbox(text)
    font_width = caption_bb[2] - caption_bb[0]
    font_height = caption_bb[3] - caption_bb[1]
    return (font_width, font_height)


def _font_path():
    return os.path.join(
        os.path.dirname(sys.modules["__main__"].__file__),  # type: ignore
        "fonts/Roboto-Light.ttf",  # type: ignore
    )


def get_font(font_size: int):
    return ImageFont.truetype(_font_path(), font_size)


_default_get_font = get_font


def _caption_font_lease(size: int):
    if get_font is not _default_get_font:
        return nullcontext(get_font(size))
    return font_lease(_font_path(), size)


def add_caption(
    img: np.ndarray, caption: str, size: int, position: CaptionPosition
) -> np.ndarray:
    """Add caption with PIL"""
    _, w, c = get_h_w_c(img)
    cap_img = Image.fromarray(np.zeros((size, w), dtype=np.uint8))
    font_size = native_text.caption_fit(size, w)
    with ExitStack() as fonts:
        font = fonts.enter_context(_caption_font_lease(font_size))
        fw, _ = get_font_size(font, caption)
        if fw > w:
            font = fonts.enter_context(
                _caption_font_lease(native_text.caption_fit(size, w, fw))
            )
        d = ImageDraw.Draw(cap_img)
        lines, _ = native_text.scan(caption)
        native_text.draw_lines(d, (w // 2, size // 2), lines, font, "center")
    raster = np.array(cap_img)
    if img.dtype == np.float32 and (img.ndim == 2 or c in (3, 4)):
        if position not in (CaptionPosition.TOP, CaptionPosition.BOTTOM):
            raise ValueError(f"Unknown position {position}")
        return compose_caption(img, raster, position == CaptionPosition.TOP)
    cap_img = normalize(raster)
    cap_img = as_target_channels(cap_img, c)
    if position == CaptionPosition.BOTTOM:
        return np.vstack((img, cap_img))
    elif position == CaptionPosition.TOP:
        return np.vstack((cap_img, img))
    else:
        raise ValueError(f"Unknown position {position}")
