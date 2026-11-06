"""Grain in a VapourSynth script."""

from __future__ import annotations

import importlib
from typing import Any

# VapourSynth by name, as in upscale.py.
vs: Any = importlib.import_module("vapoursynth")


def poisson_grain(clip: Any, /, *args: Any, **kwargs: Any) -> Any:
    """vs-jetpack's Poisson grain on any clip. The noise plugin's Poisson type has no float
    path: its grain layer stays as the frame's memory held it. So a float clip, as every
    image is, goes through 16 bits, where values outside 0..1 clip."""
    grainer = importlib.import_module("vsdeband").Grainer.POISSON
    if clip.format.sample_type == vs.INTEGER:
        return grainer(clip, *args, **kwargs)
    integer = clip.format.replace(sample_type=vs.INTEGER, bits_per_sample=16)
    grained = grainer(clip.resize.Point(format=integer.id), *args, **kwargs)
    return grained.resize.Point(format=clip.format.id)
