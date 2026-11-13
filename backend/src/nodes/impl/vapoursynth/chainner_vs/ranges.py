"""Range fix in a VapourSynth script."""

from __future__ import annotations

import importlib
from typing import Any, Literal

COMPRESS = "Compress (Full to Limited)"
EXPAND = "Expand (Limited to Full)"


def range_fix(
    clip: Any,
    mode: Literal["Compress (Full to Limited)", "Expand (Limited to Full)"] = COMPRESS,
) -> Any:
    """Rescales the clip's values between full and limited range; its range flag stays
    as it is. Float has one range, so a float clip comes back unchanged."""
    if mode not in (COMPRESS, EXPAND):
        raise ValueError(f"range_fix: mode {mode!r} is not {COMPRESS!r} or {EXPAND!r}")
    vstools = importlib.import_module("vstools")
    full, limited = vstools.ColorRange.FULL, vstools.ColorRange.LIMITED
    source, target = (full, limited) if mode == COMPRESS else (limited, full)
    in_low = vstools.get_lowest_values(clip, range_in=source)
    in_peak = vstools.get_peak_values(clip, range_in=source)
    out_low = vstools.get_lowest_values(clip, range_in=target)
    out_peak = vstools.get_peak_values(clip, range_in=target)
    # The arithmetic of the expression is kept as is: other tools write the same one,
    # and a reordering changes rounding.
    return clip.akarin.Expr(
        [
            f"x {in_low[p]} - {out_peak[p]} {out_low[p]} - * {in_peak[p]} {in_low[p]} - / {out_low[p]} +"
            for p in range(clip.format.num_planes)
        ]
    )
