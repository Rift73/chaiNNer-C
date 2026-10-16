"""The VapourSynth script compiler: records in, one .vpy script out (no VapourSynth)."""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

from nodes.impl.vapoursynth.runner import script_error
from nodes.impl.vapoursynth.script import (
    VideoClip,
    clip_var,
    compile_script,
    node_for_line,
)


def clip(node_id: str, call: str, *deps: VideoClip, imports: str = "") -> VideoClip:
    var = clip_var(node_id)
    return VideoClip(
        var=var,
        stmt=f"{var} = {call}",
        deps=deps,
        imports=frozenset([imports] if imports else []),
        node_id=node_id,
    )


def statements(script: str) -> list[str]:
    return [line for line in script.splitlines() if "  # node:" in line]


def line_of(script: str, node_id: str) -> int:
    lines = script.splitlines()
    return next(
        i for i, line in enumerate(lines, 1) if line.endswith(f"# node:{node_id}")
    )


def test_clip_var_is_a_stable_identifier():
    node_id = "3f2a9c1e-77b0-4d6e-9a51-0c8d2e4f6a7b"
    assert clip_var(node_id).isidentifier()
    assert clip_var(node_id, 2).isidentifier()
    assert clip_var(node_id) == clip_var(node_id, 0)
    assert len({clip_var(node_id, 0), clip_var(node_id, 1), clip_var("other", 0)}) == 3


def test_compile_orders_dependencies():
    # a -> (b, c) -> d: a diamond; a is reached twice but written once.
    a = clip("a", "core.std.BlankClip(width=64, height=48)")
    b = clip("b", f"core.std.Invert({a.var})", a, imports="from vstools import depth")
    c = clip("c", f"core.std.Transpose({a.var})", a, imports="import vstools")
    d = clip("d", f"core.std.Merge({b.var}, {c.var})", b, c)

    script = compile_script([(d, "Out")], cache_mb=1024)

    assert statements(script) == [
        f"{a.stmt}  # node:a",
        f"{b.stmt}  # node:b",
        f"{c.stmt}  # node:c",
        f"{d.stmt}  # node:d",
    ]
    assert "core.max_cache_size = 1024" in script
    # The records' imports come before the statements, sorted and written once.
    lines = script.splitlines()
    first = lines.index(f"{a.stmt}  # node:a")
    imports = ["from vstools import depth", "import vstools"]
    assert [line for line in lines[:first] if line in imports] == imports
    compile(script, "chain.vpy", "exec")


def test_multi_output_statement_emitted_once():
    src = clip("src", "core.std.BlankClip(format=vs.YUV444P8)")
    names = [clip_var("split", i) for i in range(3)]
    stmt = f"{', '.join(names)} = vstools.split({src.var})"
    y, u, v = (
        VideoClip(
            var=name,
            stmt=stmt,
            deps=(src,),
            imports=frozenset(["import vstools"]),
            node_id="split",
        )
        for name in names
    )

    script = compile_script([(y, "Y"), (u, "U"), (v, "V")], cache_mb=4096)

    assert statements(script) == [f"{src.stmt}  # node:src", f"{stmt}  # node:split"]


def test_traceback_maps_to_node(tmp_path: Path):
    a = clip("a", "core.std.BlankClip(width=64, height=48)")
    b = clip("b", f"core.std.Crop({a.var}, left=100)", a)
    script = compile_script([(b, "Cropped")], cache_mb=4096)
    path = tmp_path / "chain.vpy"
    path.write_text(script, encoding="utf-8")
    line = line_of(script, "b")

    assert node_for_line(script, line) == "b"
    assert node_for_line(script, 1) is None
    assert node_for_line(script, len(script.splitlines()) + 5) is None

    # What vspipe R81 prints when a statement raises during script evaluation.
    stderr = (
        "Critical: Plugin C:\\vs\\plugins\\bm3dcpu.dll uses API 3, which is no longer supported.\n"
        "Script evaluation failed:\n"
        "Python exception: Crop: negative/zero cropping dimensions not allowed\n"
        "\n"
        "Traceback (most recent call last):\n"
        '  File "vapoursynth.pyx", line 4634, in vapoursynth._vpy_evaluate\n'
        f'  File "{path}", line {line}, in <module>\n'
        f"    {b.stmt}  # node:b\n"
        '  File "vapoursynth.pyx", line 4325, in vapoursynth.Function.__call__\n'
        "vapoursynth.Error: Crop: negative/zero cropping dimensions not allowed\n"
    )
    error = script_error(path, stderr)
    assert error.node_id == "b"
    assert "Crop: negative/zero cropping dimensions not allowed" in str(error)
    assert "API 3" not in str(error)

    # A failure outside every node's statement (the outputs) names no node.
    outputs = len(script.splitlines())
    outside = stderr.replace(
        f"line {line}, in <module>", f"line {outputs}, in <module>"
    )
    assert script_error(path, outside).node_id is None
    # Neither does a traceback from another file.
    other = stderr.replace(str(path), str(tmp_path / "other.vpy"))
    assert script_error(path, other).node_id is None


class FakeClip:
    def __init__(self, raw: list[int]) -> None:
        self.raw = raw

    def set_output(self, index: int) -> None:
        self.raw.append(index)


def run_compiled(script: str, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Runs a compiled script against a stand-in VapourSynth; returns the indexes
    set through the clips' own set_output."""
    raw: list[int] = []

    class Namespace:
        def __getattr__(self, _name: str) -> Namespace:
            return self

        def __call__(self, *_args: object, **_kwargs: object) -> FakeClip:
            return FakeClip(raw)

    monkeypatch.setitem(
        sys.modules, "vapoursynth", types.SimpleNamespace(core=Namespace())
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    exec(compile(script, "chain.vpy", "exec"), {"__name__": "__vapoursynth__"})
    return raw


def test_outputs_named(monkeypatch: pytest.MonkeyPatch):
    a = clip("a", "core.std.BlankClip()")
    b = clip("b", f"core.std.Invert({a.var})", a)
    script = compile_script([(a, "Source"), (b, "It's ü")], cache_mb=4096)

    # Outside vsview (vspipe): the clips' own set_output, in output order.
    monkeypatch.delitem(sys.modules, "vsview", raising=False)
    monkeypatch.setitem(sys.modules, "vstools", None)
    assert run_compiled(script, monkeypatch) == [0, 1]

    # Inside vsview: its set_output gets the index and the name.
    named: list[tuple[int, str]] = []
    vsview = types.SimpleNamespace(
        set_output=lambda _clip, index, name: named.append((index, name))
    )
    monkeypatch.setitem(sys.modules, "vsview", vsview)
    assert run_compiled(script, monkeypatch) == []
    assert named == [(0, "Source"), (1, "It's ü")]


def test_prologue_puts_backend_on_the_path():
    script = compile_script([(clip("a", "core.std.BlankClip()"), "A")], cache_mb=4096)
    tree = ast.parse(script)
    appended = [
        ast.literal_eval(node.value.args[0])
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "sys.path.append"
    ]
    backend_src = Path(__file__).resolve().parents[1] / "src"
    assert [Path(p) for p in appended] == [backend_src]


@pytest.mark.parametrize(
    "node_id",
    [
        "a\nimport os",
        "a\rimport os",
        "a\r\nimport os",
        "a\u2028import os",
        "a\x00",
        "a\ud800",
    ],
)
def test_node_id_cannot_end_its_tag(node_id: str):
    """A chain file's node id goes into the comment that tags its statement: one that
    could end that line (and put code after it) is refused, not written."""
    a = clip(node_id, "core.std.BlankClip()")
    with pytest.raises(ValueError, match="node id"):
        compile_script([(a, "A")], cache_mb=64)
    # A dependency's id is checked as well.
    b = clip("b", f"core.std.Invert({a.var})", a)
    with pytest.raises(ValueError, match="node id"):
        compile_script([(b, "B")], cache_mb=64)


def test_search_paths_come_before_imports():
    """A record may need a folder on sys.path (the user's function files): that line
    comes before every import, whatever its sort order."""
    record = VideoClip(
        var="n_a_0",
        stmt="n_a_0 = good.brighten(core.std.BlankClip())",
        deps=(),
        imports=frozenset(["import good", r"sys.path.append('C:\user')"]),
        node_id="a",
    )
    lines = compile_script([(record, "A")], cache_mb=64).splitlines()
    assert lines.index(r"sys.path.append('C:\user')") < lines.index("import good")
