"""chaiNNer's helpers for the VapourSynth scripts it writes. The scripts import this
package from chaiNNer's backend/src (the prologue's sys.path line), in their own vspipe
or vsview process; chaiNNer's worker never imports it."""

from __future__ import annotations

from .grain import poisson_grain
from .images import NotFiniteError, image_clip, save_frame
from .upscale import upscale

__all__ = ["NotFiniteError", "image_clip", "poisson_grain", "save_frame", "upscale"]
