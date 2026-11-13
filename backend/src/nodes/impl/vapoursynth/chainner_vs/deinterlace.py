"""Detelecine in a VapourSynth script."""

from __future__ import annotations

import importlib
from typing import Any


def detelecine(
    clip: Any, tff: bool = True, postprocess: bool = True, cycle: int = 5
) -> Any:
    """Field matching (VFM) then decimation (VDecimate): 3:2 pulldown back to its
    progressive frames, 30000/1001 to 24000/1001 fps at cycle 5. The clip's field order
    is set from `tff`; `postprocess` deinterlaces the frames still combed after
    matching, with NNEDI3 on the CPU (znedi3)."""
    vstools = importlib.import_module("vstools")
    vsdeinterlace = importlib.import_module("vsdeinterlace")
    field_based = vstools.FieldBased.TFF if tff else vstools.FieldBased.BFF
    clip = field_based.apply(clip)
    deinterlace = None
    if postprocess:
        nnedi3 = importlib.import_module("vsaa").NNEDI3
        deinterlace = nnedi3(pscrn=0, tff=tff, double_rate=False).deinterlace
    matched = vsdeinterlace.vfm(clip, postprocess=deinterlace)
    return vsdeinterlace.vdecimate(matched, cycle=cycle)
