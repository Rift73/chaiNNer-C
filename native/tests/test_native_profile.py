"""CHAINNER_C_PROFILE timer (SP4a): per-name times at the native boundary.

Off, native.lib() is a plain ctypes.CDLL and native_graph.graph() is untouched.
On, every DLL export and eight pyd entries record calls, inclusive, exclusive and
max ns per name, and the worker's /run logs the table as one line.
"""

import ast
import ctypes
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from test_generator_lifecycle import (
    COMMIT_PATH_TIMERS,
    Scenario,
    profiled_run,
    set_window,
    window_summaries,
)

from nodes.impl import native, native_profile
from nodes.impl.item_window import SERIAL_PRODUCER_VARIABLE

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend" / "src"
FIELDS = {"calls", "inclusive_ns", "exclusive_ns", "max_ns"}
CHILD = """
import ctypes, hashlib, json, os, sys
from pathlib import Path

import numpy as np
from nodes.impl import native_profile
from nodes.impl.native import lib
from nodes.impl.native_box import separable_box
from nodes.impl.native_graph import graph


class Stream:  # stands in for api.Generator
    def __init__(self, supplier, length):
        self.supplier = supplier

    def with_fail_fast(self, fail_fast):
        return self


# A real _FileSequenceIterator over two paths; the stub globals load no Pillow
# and read no file.
stub = {
    "get_available_image_formats": lambda: [".png"],
    "list_glob": lambda directory, expression, extensions: [
        Path("x/a.png"), Path("x/b.png")
    ],
    "Generator": Stream,
    "load_image_node": lambda path: (path.stem, str(path.parent), path.name),
    "os": os,
}
stream, _ = graph().file_sequence_load_images(
    stub, Path("x"), False, False, "", False, 0, False
)
items = list(stream.supplier())
image = np.random.default_rng(20261003).random((64, 80, 3), dtype=np.float32)
output = np.ascontiguousarray(separable_box(image, 2.5, 3.5))
print(json.dumps({
    "plain": type(lib()) is ctypes.CDLL,
    "wrapped": hasattr(graph().image_io_read_pil, "__wrapped__"),
    "iter_wrapped": hasattr(graph()._FileSequenceIterator.__next__, "__wrapped__"),
    "items": len(items),
    "digest": hashlib.sha256(output.tobytes()).hexdigest(),
    "table": native_profile.snapshot(),
    "modules": [m for m in ("PIL", "torch") if m in sys.modules],
}))
"""
# A collection inside one of the timer's locked sections can run a finalizer
# that makes a timed call on the same thread (native_neighborhood.py's palette
# plans, native_onnx_runtime.py's sessions). Each hook below makes such a call
# once, deterministically: while an entry is created, under the name being
# created, and while snapshot() reads the table.
REENTRANT_CHILD = """
import json

from nodes.impl import native_profile


class Finalizers(dict):
    def __init__(self):
        super().__init__()
        self.hooks = {"create": "cn_a", "read": "cn_b"}

    def fire(self, hook):
        name = self.hooks.pop(hook, None)
        if name is not None:
            native_profile.timed(name, lambda: None)()

    def setdefault(self, key, default=None):
        self.fire("create")
        return super().setdefault(key, default)

    def __setitem__(self, key, value):
        self.fire("create")
        super().__setitem__(key, value)

    def copy(self):
        self.fire("read")
        return super().copy()

    def items(self):
        for pair in super().items():
            yield pair
            self.fire("read")


native_profile._table = Finalizers()
native_profile.timed("cn_a", lambda: None)()
print(json.dumps([native_profile.snapshot(), native_profile.snapshot()]))
"""
# The conversion bridge: timed as converted_pixels only with the timer on.
BRIDGE_CHILD = """
import json, sys

import numpy as np
from nodes.impl import native_buffers, native_profile

image = (np.arange(4 * 5 * 3) % 256).astype(np.uint8).reshape(4, 5, 3)
native_buffers.converted_pixels(image)  # loads lib(), whose ABI check is timed too
native_profile.reset()
out = native_buffers.converted_pixels(image)
print(json.dumps({
    "wrapped": hasattr(native_buffers.converted_pixels, "__wrapped__"),
    "out": out.tobytes().hex(),
    "table": native_profile.snapshot(),
    "modules": [m for m in ("PIL", "torch") if m in sys.modules],
}))
"""
# With the timer on, lib() attaches the pool counters (untimed) after its ABI
# checks; three 32-chunk no-op dispatches then show in the line's pool deltas.
POOL_CHILD = """
import ctypes, json, sys
from pathlib import Path

from nodes.impl import native, native_profile

dll = native.lib()
native_profile.reset()
noop = dll["cn_parallel_noop"]
noop.argtypes = [ctypes.c_size_t]
noop.restype = ctypes.c_int
statuses = [noop(32) for _ in range(3)]
line = native_profile.log_line()
path = Path(native.__file__).with_name("chainner_native.dll")
capacity = ctypes.CDLL(str(path))["cn_parallel_capacity"]  # untimed
capacity.restype = ctypes.c_uint
print(json.dumps({
    "statuses": statuses,
    "line": json.loads(line[len("native profile: "):]),
    "snapshot": native_profile.snapshot(),
    "capacity": capacity(),
    "modules": [m for m in ("PIL", "torch") if m in sys.modules],
}))
"""


@pytest.fixture(autouse=True)
def empty_table():
    native_profile.reset()
    yield
    native_profile.reset()


def fake_clock(monkeypatch, *ticks):
    """Replace the timer's clock with ticks returned in order."""
    clock = SimpleNamespace(perf_counter_ns=iter(ticks).__next__)
    monkeypatch.setattr(native_profile, "time", clock)


def test_enabled_reads_the_variable_once_and_accepts_only_1(monkeypatch):
    assert native_profile.VARIABLE == "CHAINNER_C_PROFILE"
    try:
        for value, on in (
            ("1", True),
            ("0", False),
            ("true", False),
            (" 1", False),
            ("", False),
            (None, False),
        ):
            if value is None:
                monkeypatch.delenv(native_profile.VARIABLE, raising=False)
            else:
                monkeypatch.setenv(native_profile.VARIABLE, value)
            native_profile.enabled.cache_clear()
            assert native_profile.enabled() is on, value
        monkeypatch.setenv(native_profile.VARIABLE, "1")
        assert native_profile.enabled() is False  # read once: the first read stands
    finally:
        native_profile.enabled.cache_clear()


def test_exclusive_time_subtracts_nested_calls_on_the_same_thread(monkeypatch):
    # perf_counter_ns yields 0, 10, 40, 100: outer 0..100 around inner 10..40
    fake_clock(monkeypatch, 0, 10, 40, 100)
    inner = native_profile.timed("inner", lambda value: value + 1)
    outer = native_profile.timed("outer", lambda value: inner(value) * 2)
    assert outer(1) == 4
    assert native_profile.snapshot() == {
        "outer": {"calls": 1, "inclusive_ns": 100, "exclusive_ns": 70, "max_ns": 70},
        "inner": {"calls": 1, "inclusive_ns": 30, "exclusive_ns": 30, "max_ns": 30},
    }


def test_a_raising_call_is_counted_and_keeps_the_stack_balanced(monkeypatch):
    # outer 0..100 catches the error of inner (10..60), which raises after a
    # timed leaf (20..30); the next outer, 200..300, times inner over 210..260.
    fake_clock(monkeypatch, 0, 10, 20, 30, 60, 100, 200, 210, 260, 300)
    leaf = native_profile.timed("leaf", lambda: None)

    def fail():
        leaf()
        raise ValueError("native failure")

    failing = native_profile.timed("inner", fail)
    passing = native_profile.timed("inner", lambda: None)

    def catch():
        with pytest.raises(ValueError, match="native failure"):
            failing()

    native_profile.timed("outer", catch)()
    native_profile.timed("outer", passing)()
    # Had the raise left inner's entry on the stack, the first outer call would
    # take it (the leaf's 10 plus inner's 50) as its children: 40, not 50.
    assert native_profile.snapshot() == {
        "outer": {"calls": 2, "inclusive_ns": 200, "exclusive_ns": 100, "max_ns": 50},
        "inner": {"calls": 2, "inclusive_ns": 100, "exclusive_ns": 90, "max_ns": 50},
        "leaf": {"calls": 1, "inclusive_ns": 10, "exclusive_ns": 10, "max_ns": 10},
    }


def test_a_call_on_another_thread_is_not_subtracted():
    # outer (main thread) sleeps 50 ms while a thread runs a timed 30 ms call
    inner = native_profile.timed("inner", lambda: time.sleep(0.03))

    def outer():
        worker = threading.Thread(target=inner)
        worker.start()
        time.sleep(0.05)
        worker.join()

    native_profile.timed("outer", outer)()
    table = native_profile.snapshot()
    assert table["inner"]["calls"] == 1
    assert table["outer"]["exclusive_ns"] >= 45_000_000
    assert table["outer"]["exclusive_ns"] == table["outer"]["inclusive_ns"]


def test_record_adds_elapsed_without_nesting(monkeypatch):
    # SP3b Task 7: an await on the loop thread interleaves with the other timed calls
    # there, so record() adds one call of the time it is given (inclusive = exclusive)
    # and nests nothing: the timed call it is made in keeps its exclusive time. It reads
    # no clock: perf_counter_ns yields only outer's 0 and 100.
    fake_clock(monkeypatch, 0, 100)

    def outer():
        native_profile.record("awaited", 30)
        native_profile.record("awaited", 50)

    native_profile.timed("outer", outer)()
    native_profile.record("awaited", 20)
    assert native_profile.snapshot() == {
        "outer": {"calls": 1, "inclusive_ns": 100, "exclusive_ns": 100, "max_ns": 100},
        "awaited": {"calls": 3, "inclusive_ns": 100, "exclusive_ns": 100, "max_ns": 50},
    }


def test_profiled_cdll_times_calls_and_keeps_ctypes_behaviour():
    path = Path(native.__file__).with_name("chainner_native.dll")
    dll = native_profile.ProfiledCDLL(str(path))
    assert isinstance(dll, ctypes.CDLL)
    dll.cn_abi_version.argtypes = []
    dll.cn_abi_version.restype = ctypes.c_int
    assert dll.cn_abi_version() == 2
    # The bridges reach exports by a computed name (native_buffers.py).
    name = "cn_parallel_capacity"
    capacity = getattr(dll, name)
    assert capacity.__name__ == name
    kernel = dll.cn_box_kernel_2d
    kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
    ]
    kernel.restype = ctypes.c_int
    output = np.zeros(9, np.float32)
    assert kernel(native.ptr(output), 9, 1.0, 1.0) == 0
    assert np.array_equal(output, np.full(9, np.float32(1) / np.float32(9)))
    with pytest.raises(ctypes.ArgumentError):
        kernel(native.ptr(output), "9", 1.0, 1.0)
    kernel.restype = None
    assert kernel(native.ptr(output), 9, 1.0, 1.0) is None
    table = native_profile.snapshot()
    assert table["cn_abi_version"]["calls"] == 1
    # The call whose argument conversion raised is counted too.
    assert table["cn_box_kernel_2d"]["calls"] == 3
    assert name not in table


class Items:
    def __init__(self, *values):
        self.values = iter(values)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.values)


class SplitItems(Items):
    """Items that are also produced in two phases (SP3b): describe() names the next
    one as a 1-tuple, and materialize(token) gives it."""

    def describe(self):
        return (next(self.values),)

    def materialize(self, token):
        (value,) = token
        return value


def fake_graph():
    """A module holding the timed entries; its functions and iterator classes."""
    entries = {
        name: (lambda *args, name=name: (name, args))
        for name in native_profile.PYD_FUNCTIONS
    }
    classes = {
        name: type(
            name,
            (SplitItems if name in native_profile.PYD_SPLIT_ITERATORS else Items,),
            {},
        )
        for name in native_profile.PYD_ITERATORS
    }
    module = ModuleType("fake_graph")
    for name, value in {**entries, **classes}.items():
        setattr(module, name, value)
    return module, entries, classes


def test_instrument_graph_wraps_the_named_entries():
    assert native_profile.PYD_FUNCTIONS == (
        "image_io_read_pil",
        "image_io_save_prepare",
        "image_io_save",
        "video_writer_frame",
    )
    assert native_profile.PYD_ITERATORS == ("_FileSequenceIterator", "_VideoFrames")
    # SP3b P8: Load Images' iterator also times its two phases; video keeps __next__.
    assert native_profile.PYD_SPLIT_ITERATORS == ("_FileSequenceIterator",)
    module, entries, classes = fake_graph()
    native_profile.instrument_graph(module)
    for name, original in entries.items():
        entry = getattr(module, name)
        assert entry.__wrapped__ is original
        assert entry(1, 2) == (name, (1, 2))
    for name, iterator in classes.items():
        assert getattr(module, name) is iterator
        assert iterator.__next__.__wrapped__ is Items.__next__
        assert list(iterator(5, 6)) == [5, 6]
    split = classes["_FileSequenceIterator"]
    assert split.describe.__wrapped__ is SplitItems.describe
    assert split.materialize.__wrapped__ is SplitItems.materialize
    items = split(7, 8)
    assert [items.materialize(items.describe()) for _ in range(2)] == [7, 8]
    assert not hasattr(classes["_VideoFrames"], "describe")
    assert not hasattr(Items.__next__, "__wrapped__")
    assert not hasattr(SplitItems.describe, "__wrapped__")
    table = native_profile.snapshot()
    # Each iterator's StopIteration is the third call.
    assert {name: entry["calls"] for name, entry in table.items()} == {
        **dict.fromkeys(native_profile.PYD_FUNCTIONS, 1),
        "_FileSequenceIterator.__next__": 3,
        "_VideoFrames.__next__": 3,
        "_FileSequenceIterator.describe": 2,
        "_FileSequenceIterator.materialize": 2,
    }


def test_instrument_graph_twice_wraps_each_entry_once():
    # Two threads missing graph()'s cache at once would both instrument.
    module, entries, classes = fake_graph()
    native_profile.instrument_graph(module)
    functions = {name: getattr(module, name) for name in entries}
    nexts = {name: iterator.__next__ for name, iterator in classes.items()}
    native_profile.instrument_graph(module)
    for name, function in functions.items():
        assert getattr(module, name) is function
        assert function(1) == (name, (1,))
    for name, iterator in classes.items():
        assert iterator.__next__ is nexts[name]
        assert list(iterator(5)) == [5]
    table = native_profile.snapshot()
    assert {name: entry["calls"] for name, entry in table.items()} == {
        **dict.fromkeys(native_profile.PYD_FUNCTIONS, 1),
        "_FileSequenceIterator.__next__": 2,
        "_VideoFrames.__next__": 2,
    }


def test_instrument_graph_twice_wraps_each_phase_once():
    # P8: the two phases keep __next__'s __wrapped__ idempotence.
    module, _, classes = fake_graph()
    native_profile.instrument_graph(module)
    split = classes["_FileSequenceIterator"]
    phases = (split.describe, split.materialize)
    native_profile.instrument_graph(module)
    assert (split.describe, split.materialize) == phases
    items = split(5)
    assert items.materialize(items.describe()) == 5
    table = native_profile.snapshot()
    assert {name: entry["calls"] for name, entry in table.items()} == {
        "_FileSequenceIterator.describe": 1,
        "_FileSequenceIterator.materialize": 1,
    }


def test_log_line_is_none_when_off_and_one_json_line_when_on(monkeypatch):
    native_profile.timed("cn_b", lambda: None)()
    native_profile.timed("cn_a", lambda: None)()
    monkeypatch.setattr(native_profile, "enabled", lambda: False)
    assert native_profile.log_line() is None
    monkeypatch.setattr(native_profile, "enabled", lambda: True)
    line = native_profile.log_line()
    assert line is not None
    assert line.startswith("native profile: {")
    assert "\n" not in line
    table = native_profile.snapshot()
    assert json.loads(line.removeprefix("native profile: ")) == table
    # Sorted keys, no spaces.
    assert line == "native profile: " + json.dumps(
        table, sort_keys=True, separators=(",", ":")
    )


def child(source, profile=False):
    """The last stdout line, as JSON, of source run from backend/src.

    A hung child fails the test at the timeout instead of blocking the suite.
    """
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() != native_profile.VARIABLE
    }
    if profile:
        environment[native_profile.VARIABLE] = "1"
    try:
        done = subprocess.run(
            [sys.executable, "-B", "-c", source],
            cwd=BACKEND,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("the child interpreter hung")
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.splitlines()[-1])


def test_timer_states_in_fresh_interpreters():
    on, off = child(CHILD, profile=True), child(CHILD)
    assert (on["plain"], on["wrapped"], on["iter_wrapped"]) == (False, True, True)
    assert on["table"]["cn_box_separable"]["calls"] == 1
    # The real pybind11 iterator's two items and its StopIteration.
    assert on["table"]["_FileSequenceIterator.__next__"]["calls"] == 3
    for entry in on["table"].values():
        assert set(entry) == FIELDS
        assert 0 <= entry["max_ns"] <= entry["exclusive_ns"] <= entry["inclusive_ns"]
    assert (off["plain"], off["wrapped"], off["iter_wrapped"]) == (True, False, False)
    assert off["table"] == {}
    assert on["items"] == off["items"] == 2
    assert on["digest"] == off["digest"]
    assert on["modules"] == off["modules"] == []


def test_a_timed_call_made_while_the_lock_is_held_completes_and_is_counted():
    tables = child(REENTRANT_CHILD)
    assert [{name: entry["calls"] for name, entry in t.items()} for t in tables] == [
        {"cn_a": 2, "cn_b": 1}
    ] * 2


def test_conversion_bridge_is_timed_only_with_the_timer_on():
    # Stand-in (v): converted_pixels' exclusive time is the enforce path's glue.
    on, off = child(BRIDGE_CHILD, profile=True), child(BRIDGE_CHILD)
    assert (on["wrapped"], off["wrapped"]) == (True, False)
    assert on["out"] == off["out"]
    bridge = on["table"]["converted_pixels"]
    inner = on["table"]["cn_pixels_convert_checked"]
    assert bridge["calls"] == inner["calls"] == 1
    # The conversion runs nested on the same thread and is subtracted.
    assert bridge["inclusive_ns"] - bridge["exclusive_ns"] == inner["inclusive_ns"]
    assert off["table"] == {}
    assert on["modules"] == off["modules"] == []


def test_profile_line_carries_pool_deltas():
    result = child(POOL_CHILD, profile=True)
    assert result["statuses"] == [0, 0, 0]
    assert native_profile.POOL_COUNTERS == (
        "cn_parallel_dispatches",
        "cn_parallel_wakes",
        "cn_parallel_empty_wakes",
    )
    pool = result["line"]["pool"]
    assert set(pool) == {"dispatches", "wakes", "empty_wakes"}
    assert pool["dispatches"] == 3
    if result["capacity"] > 1:  # one CPU has no helpers to wake
        assert pool["wakes"] >= 1
    assert pool["empty_wakes"] >= 0
    # The counters are read through an untimed library: only the no-ops are timed.
    assert set(result["line"]) == {"pool", "cn_parallel_noop"}
    assert result["line"]["cn_parallel_noop"]["calls"] == 3
    assert "pool" not in result["snapshot"]
    assert result["modules"] == []


class FakeCounter:
    def __init__(self):
        self.argtypes = []
        self.restype = None

    def __call__(self):
        return 7


class LibraryWithoutWakes:
    """Stands in for ctypes.CDLL: every export but cn_parallel_wakes."""

    def __init__(self, path):
        self.path = path

    def __getitem__(self, name):
        if name == "cn_parallel_wakes":
            raise AttributeError(name)
        return FakeCounter()


def test_attach_pool_names_the_dll_when_a_counter_export_is_missing(monkeypatch):
    # Start detached, whatever an earlier lib() in this process attached (review M4).
    monkeypatch.setattr(native_profile, "_pool", None)
    monkeypatch.setattr(native_profile, "_pool_base", ())
    # native_profile opens the library as ctypes.CDLL(path).
    monkeypatch.setattr(ctypes, "CDLL", LibraryWithoutWakes)
    with pytest.raises(
        RuntimeError,
        match=r"Incompatible chaiNNer C kernel ABI at .*: no cn_parallel_wakes export",
    ):
        native_profile.attach_pool("x/chainner_native.dll")
    # Nothing was stored: the line has no pool entry.
    monkeypatch.setattr(native_profile, "enabled", lambda: True)
    native_profile.reset()
    line = native_profile.log_line()
    assert line is not None
    assert "pool" not in json.loads(line.removeprefix("native profile: "))


def test_run_handler_resets_before_and_logs_after_the_executor():
    tree = ast.parse((BACKEND / "server.py").read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "nodes.impl"
        and any(alias.name == "native_profile" for alias in node.names)
        for node in tree.body
    )
    (handler,) = (
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run"
    )
    (block,) = (
        node
        for node in ast.walk(handler)
        if isinstance(node, ast.Try)
        and any(ast.unparse(s) == "await executor.run()" for s in node.body)
    )
    assert ast.unparse(block.body[0]) == "native_profile.reset()"
    # SP4c: the NumPy pool's idle blocks are released before the line, so its
    # "numpy_pool" entry carries the bytes released.
    assert [ast.unparse(s) for s in block.finalbody] == [
        "ctx.executor = None",
        "numpy_pool.release()",
        "line = native_profile.log_line()",
        "if line is not None:\n    logger.info(line)",
    ]


@pytest.mark.parametrize("value", [None, "0"])
def test_profile_off_adds_no_timers(monkeypatch, tmp_path, caplog, value):
    # SP3b Task 7: off (unset, or any value but "1"), the commit path's sites add no
    # record, no clock reading and no line. The K = 3 split Load Images -> lazy Save
    # stand-in passes the two run-time sites (the loop's take_advance, the reading
    # thread's lazy wait), run as the worker's /run runs it; /run's start of the loop
    # thread's CPU time is behind the gate.
    if value is None:
        monkeypatch.delenv(native_profile.VARIABLE, raising=False)
    else:
        monkeypatch.setenv(native_profile.VARIABLE, value)
    set_window(monkeypatch, "3")
    monkeypatch.delenv(SERIAL_PRODUCER_VARIABLE, raising=False)
    added = []
    monkeypatch.setattr(native_profile, "record", lambda *args: added.append(args))
    for clock in ("perf_counter_ns", "thread_time_ns"):
        monkeypatch.setattr(time, clock, lambda clock=clock: added.append(clock) or 0)
    scenario = Scenario(
        monkeypatch, tmp_path, [list(range(6))], source_paths=(), split=True, lazy=True
    )
    native_profile.enabled.cache_clear()
    try:
        with caplog.at_level(logging.INFO, logger="sanic.root"):
            line = profiled_run(scenario)
    finally:
        native_profile.enabled.cache_clear()
    assert added == []
    assert line is None
    assert not set(COMMIT_PATH_TIMERS) & set(native_profile.snapshot())
    # Both run-time sites were passed: the window served slots 1-6 (the last one the
    # end), and the sink read each item's lazy input.
    [summary] = window_summaries(caplog)
    assert summary["items"] == 6
    assert [x[0] for x in scenario.outputs] == list(range(6))
    tree = ast.parse((BACKEND / "server.py").read_text(encoding="utf-8"))
    (handler,) = (
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run"
    )
    (block,) = (
        node
        for node in ast.walk(handler)
        if isinstance(node, ast.Try)
        and any(ast.unparse(s) == "await executor.run()" for s in node.body)
    )
    assert [ast.unparse(s) for s in block.body[:2]] == [
        "native_profile.reset()",
        "if native_profile.enabled():\n    native_profile.start_loop_thread_cpu()",
    ]
