"""The catalog's records of VapourSynth functions, shared by the scanner, the type
adapters and the generated nodes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping

_FROM_IMPORT = re.compile(r"from ([\w.]+) import (\w+)")


@dataclass(frozen=True)
class Param:
    name: str
    # The annotation as source text, e.g. "VideoNode | None".
    annotation: str
    # The default's repr, or None when the parameter has no default.
    default: str | None
    # "positional", "keyword" or "var_keyword".
    kind: str


@dataclass(frozen=True)
class Method:
    # Stable id, e.g. "vsdenoise.bm3d"; saved chains refer to it.
    id: str
    family: str
    label: str
    # The callable as source text, e.g. "bm3d" or "NNEDI3(...).deinterlace"; "" for a
    # group (its members have the calls).
    call: str
    # "function", "bound" (a method of a class instance), "plugin", or "group" (one
    # method of many classes or enum members, picked from a dropdown: `members`).
    kind: str
    # The class's constructor parameters, for bound methods.
    init_params: tuple[Param, ...]
    params: tuple[Param, ...]
    doc: str
    # Parameter name -> its help text from the docstring's Args section.
    arg_docs: Mapping[str, str] = field(hash=False)
    # The parameters shown in Simple mode ("class" is a group's dropdown).
    simple: frozenset[str]
    # The import statement the call needs ("import vsaa" for "vsaa.NNEDI3(...)..."); ""
    # for plugins, which the script's `core` reaches.
    import_line: str = ""
    # A folder the script must put on sys.path first (the user folder, for functions
    # of its files), or "".
    search_path: str = ""
    # Parameter name -> plain label, from the overlays.
    labels: Mapping[str, str] = field(default_factory=dict, hash=False)
    # A group's methods, one per class or enum member, in dropdown order.
    members: tuple[Method, ...] = ()
    # A member's dropdown value: its class or enum member name ("Mitchell").
    choice: str = ""
    # A group's preselected member (its choice), or "" for the first.
    default_choice: str = ""
    # Listed first in its node's Method dropdown ("Common"): it has an overlay.
    common: bool = False
    # Parameter name -> the value its input starts with (an overlay's default).
    defaults: Mapping[str, object] = field(default_factory=dict, hash=False)
    # It returns a list of clips (shift_clip_multi), not one clip.
    many: bool = False
    # A function of the user's files or listed modules.
    user: bool = False


def qualified(import_line: str, expr: str) -> tuple[str, str]:
    """An expression on the name an import line imports, module-qualified, and the
    import it then needs: ("from vsdenoise import bm3d", "bm3d.Profile.HIGH") ->
    ("import vsdenoise", "vsdenoise.bm3d.Profile.HIGH"). VapourSynth's own names go
    through the script's `vs` and need none. Generated scripts so bind no imported
    short names, and no two imports of a script can shadow each other. Other import
    lines ("", for plugins) are returned as they are."""
    match = _FROM_IMPORT.fullmatch(import_line)
    if match is None:
        return import_line, expr
    module, name = match.groups()
    if expr != name and not expr.startswith((f"{name}.", f"{name}(")):
        return import_line, expr
    if module == "vapoursynth":
        return "", f"vs.{expr}"
    return f"import {module}", f"{module}.{expr}"
