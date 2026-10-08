"""Reproduce the installed NCNN optimizer's fixed C++ translation.

optimizer.py stays the unmodified upstream snapshot. CORRECTIONS replaces single
lines of it before translation, and the differential tests run the same corrected
source as their oracle (test_ncnn_graph.py), so the port departs from upstream only
there (native/ARCHITECTURE.md section 7).
"""

import ast
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "tools"))
from graph_codegen_common import Emitter

SOURCE = Path(__file__).with_name("optimizer.py")
OUTPUT = SOURCE.parents[2] / "include" / "ncnn_optimizer_passes.hpp"

# (pass, frozen line, corrected line); each frozen line occurs once in its pass.
CORRECTIONS = (
    # ncnnoptimize fuses MemoryData - Split - BinaryOp from MemoryData layers only.
    # Upstream's inverted test sent any other layer feeding a Split and a two-input
    # BinaryOp into the fusion: a KeyError ('0', '1' or 'data') after a partial
    # mutation.
    (
        "__fuse_memorydata_binaryop",
        'if self.model.layers[i].op_type != "MemoryData":',
        'if self.model.layers[i].op_type == "MemoryData":',
    ),
    # A for/else search starts one step before its first index, so an empty range (the
    # layer is first) ends on the not-found value, as a search that finds nothing does.
    # Upstream started these on the first index (upstream chaiNNer #2397): a first
    # layer overwrote layers[-2] or fused layers[1].
    ("__fuse_binaryop_eltwise", "j0 = 0", "j0 = -1"),
    ("__fuse_binaryop_eltwise", "j1 = 0", "j1 = -1"),
    ("__eliminate_dropout", "j = i - 1", "j = i"),
    ("__eliminate_pooling1x1", "j = i - 1", "j = i"),
    ("__eliminate_split", "j = i - 1", "j = i"),
)


def corrected_source() -> str:
    """optimizer.py with CORRECTIONS applied; line numbers are unchanged."""
    lines = SOURCE.read_text(encoding="utf-8").splitlines(keepends=True)
    cls = next(n for n in ast.parse("".join(lines)).body if isinstance(n, ast.ClassDef))
    passes = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
    for name, frozen, corrected in CORRECTIONS:
        f = passes[name]
        found = [
            k for k in range(f.lineno - 1, f.end_lineno) if lines[k].strip() == frozen
        ]
        assert len(found) == 1, (name, frozen, found)
        lines[found[0]] = lines[found[0]].replace(frozen, corrected)
    return "".join(lines)


def main() -> None:
    emitter = Emitter()
    emitter.line(
        "// Fixed C++ translation of the frozen installed optimizer with the CORRECTIONS"
    )
    emitter.line("// of generate_optimizer_cpp.py. Do not hand-edit.")
    emitter.line("// Source SHA256: " + hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    tree = ast.parse(corrected_source())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    for f in cls.body:
        if isinstance(f, ast.FunctionDef) and f.name != "__init__":
            emitter.function(f)
    OUTPUT.write_text("\n".join(emitter.lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
