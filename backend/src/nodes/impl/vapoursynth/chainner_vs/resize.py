"""Scaling by a factor in a VapourSynth script."""

from __future__ import annotations

import importlib
from typing import Any, Literal


def scale_by(
    clip: Any,
    factor: float = 0.5,
    kernel: Literal["Hermite", "Catrom", "Mitchell", "Lanczos", "Bilinear"] = "Hermite",
    linear: bool = True,
) -> Any:
    """The clip scaled by `factor` with a vs-jetpack kernel, in linear light when
    `linear`; width and height are rounded to whole chroma samples."""
    if factor <= 0:
        raise ValueError(f"scale_by: the factor must be above 0, not {factor}")
    kernels = ("Hermite", "Catrom", "Mitchell", "Lanczos", "Bilinear")
    if kernel not in kernels:
        raise ValueError(f"scale_by: kernel {kernel!r} is not one of {kernels}")
    fmt = clip.format
    step_w, step_h = 1 << fmt.subsampling_w, 1 << fmt.subsampling_h
    width = max(1, int(clip.width * factor / step_w + 0.5)) * step_w
    height = max(1, int(clip.height * factor / step_h + 0.5)) * step_h
    scaler = getattr(importlib.import_module("vskernels"), kernel)()
    return scaler.scale(clip, width=width, height=height, linear=linear)
