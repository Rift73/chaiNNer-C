"""backend/src's sources for the structural tests, and server.py's functions compiled
alone: importing server.py would build its Sanic app and parse the process's argv."""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend" / "src"
DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def source(relative: str) -> str:
    return (BACKEND / relative).read_text(encoding="utf-8")


def definition(tree, *path):
    """The definition named path[0] at the top level, then path[1:] nested in it."""
    (node,) = (n for n in tree.body if getattr(n, "name", None) == path[0])
    for name in path[1:]:
        parent = node
        (node,) = (
            n
            for n in ast.walk(parent)
            if n is not parent and isinstance(n, DEFINITIONS) and n.name == name
        )
    assert isinstance(node, DEFINITIONS)
    return node


def compiled(name: str, **names: object) -> Callable:
    """server.py's top-level function name, compiled alone under server.py's future
    import, with names as its module globals."""
    tree = ast.parse(source("server.py"))
    future = tree.body[0]
    assert ast.unparse(future) == "from __future__ import annotations"
    module = ast.Module(body=[future, definition(tree, name)], type_ignores=[])
    server = ModuleType("server")
    server.__dict__.update(names)
    exec(compile(module, str(BACKEND / "server.py"), "exec"), server.__dict__)
    return getattr(server, name)
