"""VapourSynth nodes return records of script statements; `compile_script` writes the
chain's records as one .vpy script. Nothing here imports VapourSynth."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# Ends every statement line; a traceback's line number finds its node through it.
NODE_TAG = "  # node:"

# backend/src, put on the script's sys.path for chaiNNer's helpers in the child.
BACKEND_SRC = Path(__file__).resolve().parents[3]

# vsview names the outputs (it has imported itself when it runs a script); else, as in
# vspipe, the clips' own. (vs-jetpack 2.2.6's vstools has no set_output: importing it
# for one cost each vspipe evaluation about 90 ms, only to fail.)
OUTPUTS = """\
if "vsview" in sys.modules:
    from vsview import set_output
else:

    def set_output(clip, index, name):
        clip.set_output(index)
"""


@dataclass(frozen=True)
class VideoClip:
    """A clip: `stmt` is one Python statement that binds `var` (several clips of one
    node may share a statement, e.g. `y, u, v = split(clip)`)."""

    var: str
    stmt: str
    deps: tuple[VideoClip | VSFilter, ...]
    # Whole import statements, e.g. "from vsdenoise import bm3d".
    imports: frozenset[str]
    node_id: str
    # `var` holds a list of clips (shift_clip_multi's), for an input that takes several.
    many: bool = False


def one_clip(clip: VideoClip, where: str) -> str:
    """The variable of a clip that `where` takes alone."""
    if clip.many:
        raise ValueError(
            f"{where} takes one clip, but the node connected to it gives several."
            " Connect that node to an input that takes several clips: Interleave or"
            " Splice (Cut and Splice), Stack (Planes), Expression's clips."
        )
    return clip.var


@dataclass(frozen=True)
class VSFilter:
    """A filter: a callable expression, e.g. `NNEDI3(field=1).deinterlace`, that the
    statement of the node it is wired into uses as an argument."""

    expr: str
    deps: tuple[VideoClip | VSFilter, ...]
    imports: frozenset[str]
    node_id: str


def clip_var(node_id: str, output: int = 0) -> str:
    """The script variable of a node's output: a valid identifier, the same for the
    same node and output (chaiNNer's node ids are UUIDs)."""
    return f"n_{re.sub(r'[^0-9A-Za-z_]', '_', node_id)}_{output}"


def _one_line(stmt: str) -> str:
    # A traceback may name any line of a statement (one spread over lines by pasted
    # text, say), and only the tagged line maps to the node.
    return ast.unparse(ast.parse(stmt)) if "\n" in stmt else stmt


def compile_script(outputs: Sequence[tuple[VideoClip, str]], *, cache_mb: int) -> str:
    """The script of the clips in `outputs` (output i = the i-th (clip, name) pair):
    every statement once, after the statements it depends on, tagged with its node."""
    imports: set[str] = set()
    statements: list[str] = []
    seen: set[tuple[str, str]] = set()
    for clip, _ in outputs:
        # Depth first, iteratively: a long chain must not hit the recursion limit.
        stack: list[tuple[VideoClip | VSFilter, bool]] = [(clip, False)]
        while stack:
            record, deps_done = stack.pop()
            if deps_done:
                if isinstance(record, VideoClip):
                    statements.append(
                        f"{_one_line(record.stmt)}{NODE_TAG}{record.node_id}"
                    )
                continue
            code = record.stmt if isinstance(record, VideoClip) else record.expr
            if (record.node_id, code) in seen:
                continue
            # The id ends its statement's line, in the tag's comment: a line break
            # (from a crafted chain file) would put code after it.
            if not record.node_id.isprintable():
                raise ValueError(
                    f"The chain has a node whose node id {record.node_id!r} cannot be"
                    " written in a script."
                )
            seen.add((record.node_id, code))
            imports.update(record.imports)
            stack.append((record, True))
            stack.extend((dep, False) for dep in reversed(record.deps))

    return "\n".join(
        [
            "import sys",
            "",
            "import vapoursynth as vs",
            "",
            f"sys.path.append({str(BACKEND_SRC)!r})",
            "core = vs.core",
            f"core.max_cache_size = {int(cache_mb)}",
            "",
            # A folder a record needs on sys.path (the user's files) before imports.
            *sorted(imports, key=lambda line: (not line.startswith("sys.path."), line)),
            "",
            *statements,
            "",
            OUTPUTS,
            *(
                f"set_output({one_clip(clip, f'Output {name!r}')}, {index}, {name!r})"
                for index, (clip, name) in enumerate(outputs)
            ),
            "",
        ]
    )


def node_for_line(script: str, line: int) -> str | None:
    """The node whose statement is on the script's 1-based `line`, if any."""
    lines = script.splitlines()
    if not 1 <= line <= len(lines):
        return None
    _, tag, node_id = lines[line - 1].rpartition(NODE_TAG)
    return node_id if tag else None
