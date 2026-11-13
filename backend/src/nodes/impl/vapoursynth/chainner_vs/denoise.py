"""Denoising in a VapourSynth script."""

from __future__ import annotations

import importlib
from typing import Any

# VapourSynth by name, as in upscale.py.
vs: Any = importlib.import_module("vapoursynth")

# The clips vs-jetpack's BM3D takes besides its own: Search clip and Reference clip.
_BM3D_CLIPS = ("pre", "ref")


def _float32(clip: Any) -> bool:
    return clip.format.sample_type == vs.FLOAT and clip.format.bits_per_sample == 32


def bm3d(clip: Any, /, **kwargs: Any) -> Any:
    """vs-jetpack's BM3D on any clip. Every backend but Old takes only 32-bit float, so
    for those a clip in another format goes to 32-bit float before it and back to its
    own format after it, as Convert Format's Bit depth (vstools.depth) does both ways;
    a Search or Reference clip in another format goes to 32-bit float too. A 32-bit
    float clip, and every clip on the Old backend, go to vs-jetpack's BM3D as they are."""
    vstools = importlib.import_module("vstools")
    vsdenoise = importlib.import_module("vsdenoise")
    backend = kwargs.get("backend", vsdenoise.bm3d.Backend.AUTO)
    if backend.resolve() is vsdenoise.bm3d.Backend.OLD:
        return vsdenoise.bm3d(clip, **kwargs)
    for name in _BM3D_CLIPS:
        other = kwargs.get(name)
        if other is not None and not _float32(other):
            kwargs[name] = vstools.depth(other, 32)
    if _float32(clip):
        return vsdenoise.bm3d(clip, **kwargs)
    denoised = vsdenoise.bm3d(vstools.depth(clip, 32), **kwargs)
    return vstools.depth(denoised, clip)
