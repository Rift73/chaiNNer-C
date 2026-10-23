"""Frozen upstream adjustments under chaiNNer-C's contract (2026-10-23), where upstream
divides 0 by 0: unpremultiplying keeps a zero channel of a zero-alpha pixel (NaN in
every transparent black pixel), and Color Levels with equal input black and white is a
step, a value at that level taking the output white (NaN there)."""

from __future__ import annotations

import numpy as np


def premultiplied_alpha(module):
    """module's premultiplied_alpha_node with chaiNNer-C's unpremultiply."""
    original = module.premultiplied_alpha_node

    def premultiplied_alpha_node(img, alpha_association):
        result = original(img, alpha_association)
        if alpha_association == module.AlphaAssociation.UNPREMULTIPLY_RGB:
            rgb = img[..., :3]
            kept = (rgb == 0) & (img[..., 3:] == 0)
            result[..., :3] = np.where(kept, rgb, result[..., :3])
        return result

    return premultiplied_alpha_node


def color_levels(module):
    """module's color_levels_node with chaiNNer-C's step for equal input levels."""
    original = module.color_levels_node

    def color_levels_node(img, red, green, blue, alpha, black, white, gamma, low, high):
        result = original(img, red, green, blue, alpha, black, white, gamma, low, high)
        if black != white:
            return result
        source = img if img.ndim == 3 else img[..., None]
        channels = source.shape[2]
        selected = [blue, green, red, alpha][:channels] if channels > 1 else [True]
        top = np.float32(high) - np.float32(low) + np.float32(low)
        at = (source == np.float32(black)) & np.array(selected)
        result[...] = np.where(at, top, result)
        return result

    return color_levels_node
