"""The generated VapourSynth family nodes (nodes.impl.vapoursynth.schema_gen), from the
function catalog. Without VapourSynth or a catalog none register; the hand-written
nodes still do."""

from __future__ import annotations

from nodes.impl.vapoursynth.catalog import current_catalog
from nodes.impl.vapoursynth.schema_gen import (
    ANY_FUNCTION,
    FAMILIES,
    LOAD_CLIP,
    register_families,
)

from . import clips_group, filters_group, functions_group

FUNCTIONS = ("User Functions", ANY_FUNCTION)

_catalog = current_catalog()
register_families(_catalog.methods, clips_group, _catalog.enums, [LOAD_CLIP])
register_families(
    _catalog.methods,
    filters_group,
    _catalog.enums,
    [f for f in FAMILIES if f != LOAD_CLIP and f not in FUNCTIONS],
)
register_families(_catalog.methods, functions_group, _catalog.enums, FUNCTIONS)
