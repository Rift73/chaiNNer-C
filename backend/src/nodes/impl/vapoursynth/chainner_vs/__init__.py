"""chaiNNer's helpers for the VapourSynth scripts it writes. The scripts import this
package from chaiNNer's backend/src (the prologue's sys.path line), in their own vspipe
or vsview process; chaiNNer's worker never imports it."""

from __future__ import annotations

from .deinterlace import detelecine
from .denoise import bm3d
from .grain import poisson_grain
from .images import NotFiniteError, image_clip, save_frame
from .ranges import range_fix
from .resize import scale_by
from .upscale import upscale

__all__ = [
    "NotFiniteError",
    "bm3d",
    "detelecine",
    "image_clip",
    "poisson_grain",
    "range_fix",
    "save_frame",
    "scale_by",
    "upscale",
]
