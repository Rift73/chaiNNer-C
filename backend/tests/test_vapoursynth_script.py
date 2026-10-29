"""The VapourSynth script compiler: records in, one .vpy script out (no VapourSynth)."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import tomllib

from nodes.impl.vapoursynth import runner
from nodes.impl.vapoursynth.runner import VSScriptError, script_error
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

    # Outside vsview (vspipe): the clips' own set_output, in output order, without
    # importing vstools (vs-jetpack 2.2.6's has no set_output).
    monkeypatch.delitem(sys.modules, "vsview", raising=False)
    vstools = types.SimpleNamespace(set_output=lambda *_: pytest.fail("vstools used"))
    monkeypatch.setitem(sys.modules, "vstools", vstools)
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


def test_a_crash_without_a_message_names_the_exit_code(tmp_path: Path) -> None:
    # vsrgtools' median_blur crashes vspipe on small chroma planes, with no stderr.
    script = tmp_path / "chain.vpy"
    script.write_text("import vapoursynth as vs\n", encoding="utf-8")
    assert "0xC0000005" in str(script_error(script, "", 3221225477))
    assert str(script_error(script, "", 1)) == (
        "vspipe ended with code 1 and no message (a plugin crashed)."
    )
    assert str(script_error(script, "", 0)) == "vspipe failed without a message."


def test_a_crash_by_a_signal_names_the_signal(tmp_path: Path) -> None:
    # POSIX: Popen reports a child that a signal ended as the signal's number, negated.
    script = tmp_path / "chain.vpy"
    script.write_text("import vapoursynth as vs\n", encoding="utf-8")
    message = str(script_error(script, "", -11))
    assert "signal 11" in message and "no message (a plugin crashed)" in message
    # A message of vspipe's own stays the error.
    assert str(script_error(script, "Script evaluation failed:\n", -11)) == (
        "Script evaluation failed:"
    )


# Linux: how vsscript finds the Python, and how vsview starts.

linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="vsscript's configuration and vsview's launch",
)


@pytest.fixture
def linux_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A Python runtime whose VapourSynth has libvsscript.so in lib64 (a folder the
    configuration names as lib) and whose libpython sits in lib; returns its prefix."""
    prefix = tmp_path / "runtime"
    package = prefix / "lib64" / "python3.14" / "site-packages" / "vapoursynth"
    package.mkdir(parents=True)
    (package / "libvsscript.so").write_bytes(b"")
    (prefix / "lib").mkdir()
    (prefix / "lib" / "libpython3.14.so.1.0").write_bytes(b"")
    python = prefix / "bin" / "python3"
    python.parent.mkdir()
    python.write_bytes(b"")
    config = {"INSTSONAME": "libpython3.14.so.1.0", "LIBDIR": "/install/lib"}
    monkeypatch.setattr(runner.sysconfig, "get_config_var", config.get)
    monkeypatch.setattr(runner, "vspipe", lambda: package / "vspipe")
    monkeypatch.setattr(sys, "prefix", str(prefix))
    monkeypatch.setattr(sys, "base_prefix", str(prefix))
    monkeypatch.setattr(sys, "executable", str(python))
    runner.vsscript_config_home.cache_clear()
    yield prefix
    runner.vsscript_config_home.cache_clear()


def vsscript_line(prefix: Path) -> list[str]:
    toml = prefix / "vsconfig" / "vapoursynth" / "vapoursynth.toml"
    (entry,) = tomllib.loads(toml.read_text(encoding="utf-8")).items()
    return [entry[0], *entry[1]]


@linux_only
def test_vsscript_config_names_this_python_and_follows_it(linux_runtime: Path) -> None:
    python = Path(sys.executable)
    os.utime(python, (1_700_000_007.9, 1_700_000_007.9))

    home = runner.vsscript_config_home()

    assert home == str(linux_runtime / "vsconfig")
    # Keyed by the library with lib64 as lib; the libpython; the mtime in seconds.
    library = (linux_runtime / "lib64").resolve() / "python3.14" / "site-packages"
    assert vsscript_line(linux_runtime) == [
        str(library / "vapoursynth" / "libvsscript.so").replace("/lib64/", "/lib/"),
        str(python),
        str(linux_runtime / "lib" / "libpython3.14.so.1.0"),
        "1700000007",
    ]

    # A Python that was replaced (its mtime) gets its line again; the rest is as it is.
    os.utime(python, (1_700_000_100, 1_700_000_100))
    runner.vsscript_config_home.cache_clear()
    runner.vsscript_config_home()
    assert vsscript_line(linux_runtime)[-1] == "1700000100"
    assert list((linux_runtime / "vsconfig" / "vapoursynth").iterdir()) == [
        linux_runtime / "vsconfig" / "vapoursynth" / "vapoursynth.toml"
    ]


@linux_only
def test_vsscript_config_needs_a_shared_libpython(linux_runtime: Path) -> None:
    (linux_runtime / "lib" / "libpython3.14.so.1.0").unlink()
    with pytest.raises(VSScriptError, match="shared libpython"):
        runner.vsscript_config_home()
    assert not (linux_runtime / "vsconfig").exists()


@linux_only
def test_vs_env_points_vsscript_at_the_runtime_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "vsscript_config_home", lambda: "/runtime/vsconfig")
    monkeypatch.setenv("VAPOURSYNTH_EXTRA_PLUGIN_PATH", "/another/plugins")
    env = runner.vs_env()
    assert env["XDG_CONFIG_HOME"] == "/runtime/vsconfig"
    assert "VAPOURSYNTH_EXTRA_PLUGIN_PATH" not in env


@pytest.fixture
def display(monkeypatch: pytest.MonkeyPatch) -> None:
    """A window system is named; vsview is installed. (A venv's vsscript configuration
    on Windows would start a process.)"""
    for name in runner.DISPLAY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(runner, "vs_env", lambda: dict(os.environ))


def viewer_process(
    monkeypatch: pytest.MonkeyPatch, code: str
) -> list[subprocess.Popen[Any]]:
    """vsview is a Python process running `code`; returns the ones that start."""
    started: list[subprocess.Popen[Any]] = []
    popen = subprocess.Popen

    def start(command: list[str], **kwargs: Any) -> subprocess.Popen[Any]:
        started.append(popen([sys.executable, "-c", code], **kwargs))
        return started[-1]

    monkeypatch.setattr(subprocess, "Popen", start)
    monkeypatch.setattr(runner, "_viewers", {})
    return started


@linux_only
def test_vsview_needs_a_window_system(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in runner.DISPLAY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda _: object())
    with pytest.raises(VSScriptError, match="DISPLAY"):
        runner.launch_preview(tmp_path / "preview.vpy", "view")
    assert not (tmp_path / "vsview.log").exists()


def test_vsview_that_ends_at_once_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, display: None
) -> None:
    # Windows too: a vsview that fails to import, or whose Qt has no platform plugin.
    viewer_process(
        monkeypatch,
        "import sys; print('qt.qpa.plugin: no platform plugin', flush=True); sys.exit(3)",
    )
    with pytest.raises(VSScriptError, match=r"ended right after.*code 3") as error:
        runner.launch_preview(tmp_path / "preview.vpy", "view")
    assert str(tmp_path / "vsview.log") in str(error.value)
    assert str(error.value).endswith(":\nqt.qpa.plugin: no platform plugin")
    assert not (tmp_path / "vsview.pid").exists()


def test_vsview_that_stays_up_is_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, display: None
) -> None:
    started = viewer_process(monkeypatch, "import time; time.sleep(60)")
    monkeypatch.setattr(runner, "VIEWER_GRACE_SECONDS", 0.3)
    try:
        runner.launch_preview(tmp_path / "preview.vpy", "view")
        assert [viewer.poll() for viewer in started] == [None]
    finally:
        for viewer in started:
            viewer.kill()
            viewer.wait()


def test_vsview_of_an_earlier_worker_stays_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, display: None
) -> None:
    started = viewer_process(monkeypatch, "import time; time.sleep(60)")
    monkeypatch.setattr(runner, "VIEWER_GRACE_SECONDS", 0.3)
    script = tmp_path / "preview.vpy"
    try:
        runner.launch_preview(script, "view")
        # A new worker (a restart) knows no viewer; the window is still open.
        monkeypatch.setattr(runner, "_viewers", {})
        runner.launch_preview(script, "view")
        assert len(started) == 1
        # Once the user closes it, the next run opens a new one.
        started[0].kill()
        started[0].wait()
        runner.launch_preview(script, "view")
        assert len(started) == 2
        # Linux reads the start time against a clock that a step or a resume moves.
        pid, recorded = (tmp_path / "vsview.pid").read_text(encoding="utf-8").split()
        moved = float(recorded) + 0.5
        (tmp_path / "vsview.pid").write_text(f"{pid} {moved!r}", encoding="utf-8")
        monkeypatch.setattr(runner, "_viewers", {})
        runner.launch_preview(script, "view")
        assert len(started) == 2
        # A process that took a recorded pid later is not the window.
        (tmp_path / "vsview.pid").write_text(f"{pid} 12.5", encoding="utf-8")
        monkeypatch.setattr(runner, "_viewers", {})
        runner.launch_preview(script, "view")
        assert len(started) == 3
    finally:
        for viewer in started:
            viewer.kill()
            viewer.wait()
