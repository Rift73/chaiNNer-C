"""Reproduce the installed NCNN optimizer's fixed C++ translation."""

import ast
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "tools"))
from graph_codegen_common import Emitter

if __name__ == "__main__":
    source = Path(__file__).with_name("optimizer.py")
    emitter = Emitter()
    emitter.line(
        "// Fixed C++ translation of the frozen installed optimizer. Do not hand-edit."
    )
    emitter.line("// Source SHA256: " + hashlib.sha256(source.read_bytes()).hexdigest())
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    for f in cls.body:
        if isinstance(f, ast.FunctionDef) and f.name != "__init__":
            emitter.function(f)
    target = source.parents[2] / "include" / "ncnn_optimizer_passes.hpp"
    target.write_text("\n".join(emitter.lines), encoding="utf-8")
