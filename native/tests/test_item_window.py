"""Item window (SP3): read-ahead safety, window sizing and the precompute engine."""

import ast
import asyncio
import concurrent.futures
import functools
import json
import logging
import os
import stat
import subprocess
import sys
import threading
import time
import weakref
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import psutil
import pytest
from PIL import Image
from test_execution_ops import declarations
from test_file_sequence import namespace as file_sequence_namespace
from test_generator_lifecycle import (
    EventCallback,
    PoolCalls,
    Scenario,
    decisions,
    drive,
    make_node,
    run,
    set_window,
    spin,
    stripped,
    window_summaries,
)
from test_image_io import NEW, failing_writes, save_arguments
from test_video_io import FakeFFmpeg
from test_video_io import load as video_module

import process
from api import (
    BaseInput,
    Collector,
    ExecutionOptions,
    Generator,
    InputId,
    IteratorOutputInfo,
    NodeId,
    OutputId,
)
from chain.chain import Chain, Edge, EdgeSource, EdgeTarget
from nodes.impl import item_window
from nodes.impl.item_window import (
    SERIAL_PRODUCER_VARIABLE,
    Candidate,
    ItemWindow,
    SlotGenerator,
    affinity_cpus,
    engine_jobs,
    memory_budget,
    output_nbytes,
    read_ahead_blocker,
    register_phases,
    serial_producer,
    window_size,
)
from nodes.properties.inputs import DirectoryInput, ImageInput, OrderEnum
from nodes.properties.outputs import (
    AudioStreamOutput,
    DirectoryOutput,
    FileNameOutput,
    ImageOutput,
    NumberOutput,
    TextOutput,
)
from nodes.utils.utils import get_h_w_c
from progress_controller import Aborted, ProgressController

ROOT = Path(__file__).resolve().parents[2]
UNDER = "source under output directory: save"


@pytest.mark.parametrize(
    ("files", "directory", "expected"),
    [
        pytest.param(["in/a.png", "in/b.png"], "in", UNDER, id="in-place"),
        pytest.param(
            ["in/a.png", "in/out/a.png"],
            "in/out",
            UNDER,
            id="prior-outputs-under-the-input-tree",
        ),
        pytest.param(["out/in/a.png"], "out", UNDER, id="input-under-the-output-tree"),
        pytest.param(
            ["in/x/../a.png"], "out/../in", UNDER, id="compared-after-resolve"
        ),
        pytest.param(["in/a.png"], "out", None, id="disjoint"),
        pytest.param(["in/a.png"], "in2", None, id="sibling-with-a-common-prefix"),
    ],
)
def test_read_ahead_blocker_compares_resolved_folders(
    tmp_path, files, directory, expected
):
    sources = [(NodeId("g0"), tuple(tmp_path / name for name in files))]
    assert (
        read_ahead_blocker(sources, [(NodeId("save"), tmp_path / directory)])
        == expected
    )


@pytest.mark.skipif(
    os.name != "nt", reason="paths compare case-insensitively on Windows only"
)
def test_read_ahead_blocker_ignores_case_on_windows(tmp_path):
    sources = [(NodeId("g0"), (tmp_path / "In" / "A.png",))]
    assert (
        read_ahead_blocker(
            sources, [(NodeId("save"), Path(str(tmp_path / "in").upper()))]
        )
        == UNDER
    )


@pytest.mark.skipif(os.name != "nt", reason="extended-length paths are Windows only")
@pytest.mark.parametrize("prefixed", ["source", "directory"])
def test_read_ahead_blocker_ignores_extended_length_prefixes(tmp_path, prefixed):
    def extended(path):
        return Path("\\\\?\\" + str(path))

    folder = extended(tmp_path / "in") if prefixed == "source" else tmp_path / "in"
    directory = extended(tmp_path) if prefixed == "directory" else tmp_path
    sources = [(NodeId("g0"), (folder / "a.png",))]
    assert read_ahead_blocker(sources, [(NodeId("save"), directory)]) == UNDER


@pytest.mark.skipif(os.name != "nt", reason="UNC paths are Windows only")
def test_read_ahead_blocker_maps_extended_unc_to_unc(monkeypatch):
    # Resolving a share needs the network; simulated, as for the symlink loop
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: self)
    sources = [(NodeId("g0"), (Path(r"\\server\share\out\in\a.png"),))]
    directory = Path(r"\\?\UNC\server\share\out")
    assert read_ahead_blocker(sources, [(NodeId("save"), directory)]) == UNDER


def test_read_ahead_blocker_needs_declared_sources():
    assert (
        read_ahead_blocker(
            [(NodeId("g0"), ()), (NodeId("g1"), None)], [(NodeId("save"), None)]
        )
        == "sources unknown: g1"
    )


def test_read_ahead_blocker_needs_resolved_directories(tmp_path):
    directories = [(NodeId("copy"), tmp_path), (NodeId("save"), None)]
    assert (
        read_ahead_blocker([(NodeId("g0"), ())], directories)
        == "output directory unresolved: save"
    )


def test_read_ahead_blocker_allows_no_sources_and_no_writers():
    assert read_ahead_blocker([(NodeId("g0"), ()), (NodeId("g1"), ())], []) is None


def test_read_ahead_blocker_resolves_nothing_when_no_file_is_declared():
    # Nothing is read per item, so even an unresolvable directory cannot block.
    assert (
        read_ahead_blocker([(NodeId("g0"), ())], [(NodeId("save"), Path("in\0valid"))])
        is None
    )


@pytest.mark.parametrize("failure", ["nul-in-folder", "symlink-loop"])
def test_read_ahead_blocker_reports_a_path_it_cannot_resolve(
    monkeypatch, tmp_path, failure
):
    if failure == "nul-in-folder":
        # CPython 3.14's non-strict Path.resolve() resolves a folder holding a NUL
        # lexically (3.11 raised ValueError), so it is compared like any other
        # folder: outside the output directory it allows read-ahead, under it it
        # blocks.
        outside = Path("in\0valid") / "a.png"
        assert (
            read_ahead_blocker(
                [(NodeId("g0"), (outside,))], [(NodeId("save"), tmp_path)]
            )
            is None
        )
        under = tmp_path / "in\0valid" / "a.png"
        assert (
            read_ahead_blocker([(NodeId("g0"), (under,))], [(NodeId("save"), tmp_path)])
            == "source under output directory: save"
        )
        return

    # Python 3.11 raises RuntimeError for a loop; simulated, no symlink privilege needed
    def loop(self, strict=False):
        raise RuntimeError(f"Symlink loop from {self!r}")

    monkeypatch.setattr(Path, "resolve", loop)
    source = tmp_path / "in" / "a.png"
    result = read_ahead_blocker(
        [(NodeId("g0"), (source,))], [(NodeId("save"), tmp_path)]
    )
    assert result is not None and result.startswith("path check failed: ")


def test_split_spritesheet_declares_no_per_item_source_files():
    path = (
        ROOT
        / "backend/src/packages/chaiNNer_standard/image/batch_processing/split_spritesheet.py"
    )
    context = {
        "Generator": Generator,
        "get_h_w_c": get_h_w_c,
        "OrderEnum": OrderEnum,
        "np": np,
    }
    node = declarations(path, {"split_spritesheet_node"}, context)[
        "split_spritesheet_node"
    ]
    generator = node(np.zeros((4, 4, 3), np.float32), 2, 2, OrderEnum.ROW_MAJOR)
    assert generator.source_paths == ()


def test_undeclared_generators_have_unknown_sources():
    assert Generator(lambda: iter(()), 0).source_paths is None
    assert Generator.from_iter(lambda: iter(()), 0).source_paths is None


def add_save(scenario, directory):
    """A file writer in g0's group whose DirectoryInput is fed as `directory` says."""
    tmp_path = scenario.directory
    inputs = [
        BaseInput("any", "Image").with_id(0),
        DirectoryInput(must_exist=False).with_id(1),
    ]
    if directory == "collector-static-node":
        collect = make_node(
            "collect",
            "collector",
            lambda _context, _item, _directory: Collector(
                lambda _value: None, lambda: None
            ),
            2,
            0,
        )
        object.__setattr__(collect.data, "inputs", inputs)
        outdir = make_node("outdir", "regularNode", lambda: str(tmp_path / "out"), 0, 1)
        object.__setattr__(outdir.data, "side_effects", False)
        scenario.chain.add_node(collect)
        scenario.chain.add_node(outdir)
        scenario.connect("g0", 0, "collect", 0)
        scenario.connect("outdir", 0, "collect", 1)
        return
    save = make_node("save", "regularNode", lambda _image, _directory: None, 2, 0)
    object.__setattr__(save.data, "inputs", inputs)
    scenario.chain.add_node(save)
    scenario.connect("g0", 0, "save", 0)
    if directory in {"value-in", "value-out"}:
        folder = tmp_path / directory.removeprefix("value-")
        scenario.chain.inputs.set(NodeId("save"), 1, str(folder))
    else:
        scenario.connect("g0", 0 if directory == "generator-item" else 1, "save", 1)


@pytest.mark.parametrize(
    ("override", "cpus", "expected"),
    [
        (None, 32, (8, "auto")),
        (None, 28, (7, "auto")),
        (None, 24, (6, "auto")),
        (None, 20, (6, "auto")),
        (None, 16, (6, "auto")),
        (None, 8, (6, "auto")),
        (None, 1, (6, "auto")),
        (None, 64, (8, "auto")),
        ("1", 32, (1, "forced")),
        ("4", 8, (4, "override")),
        ("0", 32, (8, "auto")),
        ("-2", 32, (8, "auto")),
        ("x", 32, (8, "auto")),
        ("", 32, (8, "auto")),
        (" 2", 32, (8, "auto")),
    ],
)
def test_window_size(monkeypatch, override, cpus, expected):
    set_window(monkeypatch, override)
    assert window_size(cpus) == expected


@pytest.mark.parametrize(("count", "expected"), [(5, 5), (None, 1)])
def test_affinity_cpus_without_an_affinity_api(monkeypatch, count, expected):
    monkeypatch.delattr(psutil.Process, "cpu_affinity")  # as on macOS
    monkeypatch.setattr(os, "cpu_count", lambda: count)
    assert affinity_cpus() == expected


@pytest.mark.parametrize(
    ("override", "sources", "pool_size", "expected"),
    [
        ("1", (), 2, "item window K=1 (forced)"),
        (None, (), 2, "item window K=8 (auto)"),
        ("3", (), 2, "item window K=3 (override)"),
        (None, None, 2, "item window K=1 (sources unknown: g0)"),
        (None, (), None, "item window K=1 (pool size unknown)"),
    ],
)
def test_window_decision_is_logged_once_per_group(
    monkeypatch, tmp_path, caplog, override, sources, pool_size, expected
):
    set_window(monkeypatch, override)
    monkeypatch.setattr(process, "affinity_cpus", lambda: 32)
    scenario = Scenario(monkeypatch, tmp_path, [[0, 1, 2]], source_paths=sources)
    with caplog.at_level(logging.INFO, logger="sanic.root"):
        run(scenario, pool_size=pool_size)
    assert decisions(caplog) == [expected]
    assert [x[0] for x in scenario.outputs] == [0, 1, 2]


@pytest.mark.parametrize(
    ("directory", "expected"),
    [
        ("value-out", "item window K=8 (auto)"),
        ("value-in", "item window K=1 (source under output directory: save)"),
        ("generator-item", "item window K=1 (output directory unresolved: save)"),
        ("generator-static", "item window K=8 (auto)"),
        ("collector-static-node", "item window K=8 (auto)"),
    ],  # not "output directory unresolved: collect"
)
def test_window_decision_follows_the_output_directory(
    monkeypatch, tmp_path, caplog, directory, expected
):
    set_window(monkeypatch, None)
    monkeypatch.setattr(process, "affinity_cpus", lambda: 32)
    files = tuple(tmp_path / "in" / name for name in ("a.png", "b.png", "c.png"))
    scenario = Scenario(monkeypatch, tmp_path, [list(files)], source_paths=files)
    add_save(scenario, directory)
    with caplog.at_level(logging.INFO, logger="sanic.root"):
        run(scenario)
    assert decisions(caplog) == [expected]


def test_window_changes_no_output_or_event(monkeypatch, tmp_path, caplog):
    observed = []
    for window in ("1", "3"):
        set_window(monkeypatch, window)
        scenario = Scenario(
            monkeypatch, tmp_path, [[0, 1, 2, 3]], source_paths=(), intermediate=True
        )
        with caplog.at_level(logging.INFO, logger="sanic.root"):
            run(scenario)
        observed.append(
            (
                scenario.outputs,
                scenario.mapped,
                stripped(scenario.events),
                scenario.broadcasts,
            )
        )
    assert decisions(caplog) == [
        "item window K=1 (forced)",
        "item window K=3 (override)",
    ]
    assert observed[0] == observed[1]


def test_forced_sequential_builds_no_window_and_checks_no_path(monkeypatch, tmp_path):
    set_window(monkeypatch, "1")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("K = 1 must not build a window or check paths")

    monkeypatch.setattr(process, "ItemWindow", forbidden)
    monkeypatch.setattr(process, "read_ahead_blocker", forbidden)
    scenario = Scenario(monkeypatch, tmp_path, [[0, 1, 2]], source_paths=())
    run(scenario)
    assert [x[0] for x in scenario.outputs] == [0, 1, 2]


def test_seam_replays_a_stored_result_in_place_of_the_pool_job(
    monkeypatch, tmp_path, caplog
):
    class StoredWindow(ItemWindow):
        def holds(self, node_id):
            return node_id == "map"

        async def take_node(self, node_id):
            return process.RegularOutput([1000]), 0.25

    set_window(monkeypatch, "1")
    reference = Scenario(
        monkeypatch, tmp_path, [[0, 1, 2]], source_paths=(), intermediate=True
    )
    run(reference)
    caplog.clear()  # the reference's own decision, if the root level lets it through
    set_window(monkeypatch, "3")
    monkeypatch.setattr(process, "ItemWindow", StoredWindow)
    scenario = Scenario(
        monkeypatch, tmp_path, [[0, 1, 2]], source_paths=(), intermediate=True
    )
    with caplog.at_level(logging.INFO, logger="sanic.root"):
        run(scenario)
    assert decisions(caplog) == ["item window K=3 (override)"]
    assert scenario.mapped == [0]  # items 1 and 2 came from the store
    assert [x[0] for x in scenario.outputs] == [7, 1000, 1000]
    times = [
        e["data"]["executionTime"]
        for e in scenario.events
        if e["event"] == "node-finish" and e["data"]["nodeId"] == "map"
    ]
    assert times[1:] == [0.25, 0.25]
    assert stripped(scenario.events) == stripped(reference.events)


def test_server_passes_its_pool_size_to_every_executor():
    tree = ast.parse((ROOT / "backend/src/server.py").read_text(encoding="utf-8"))
    constants = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "POOL_SIZE" for t in n.targets)
    ]
    assert len(constants) == 1 and ast.literal_eval(constants[0].value) == 8
    wiring = {"ThreadPoolExecutor": "max_workers", "Executor": "pool_size"}
    calls = [
        (n, n.func)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id in wiring
    ]
    passed = sorted(
        (
            func.id,
            any(
                k.arg == wiring[func.id]
                and isinstance(k.value, ast.Name)
                and k.value.id == "POOL_SIZE"
                for k in c.keywords
            ),
        )
        for c, func in calls
    )
    assert passed == [
        ("Executor", True),
        ("Executor", True),
        ("ThreadPoolExecutor", True),
    ]


def set_schema(node, schema_id):
    node.schema_id = schema_id
    object.__setattr__(node.data, "schema_id", schema_id)


class Records(logging.Handler):
    """Collects log records; decisions() and window_summaries() read its `records`."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


class Reads:
    """g0's iterator: counts the reads in progress and records each item it yields."""

    def __init__(self, pipeline, source):
        self.pipeline, self.source = pipeline, source

    def __iter__(self):
        return self

    def __next__(self):
        pipeline = self.pipeline
        with pipeline.lock:
            pipeline.reads_active += 1
        try:
            value = next(self.source)
        finally:
            with pipeline.lock:
                pipeline.reads_active -= 1
        if isinstance(value, np.ndarray):
            pipeline.values[len(pipeline.yielded)] = weakref.ref(value)
        pipeline.yielded.append(len(pipeline.yielded))
        return value


class Pipeline:
    """g0 -> nodes[0] -> ... -> nodes[-1] -> save (or the collector `collect`).

    A real Executor runs it. `map` computes value * 10 + 7 and every other node
    value + 1; each node recovers its item from its input. A node is pure (no side
    effects, schema chainner:image:test-<name>) unless it is in `impure`.
    """

    def __init__(
        self,
        monkeypatch,
        tmp_path,
        count,
        *,
        window,
        nodes=("map",),
        impure=(),
        static=False,
        lazy=False,
        skip=(),
        collector=False,
        reader=None,
        fail_map=(),
        fail_fast=True,
    ):
        monkeypatch.setattr(
            process.registry, "get_package", lambda _: SimpleNamespace(id="test")
        )
        self.monkeypatch, self.directory, self.window = monkeypatch, tmp_path, window
        self.skip, self.fail_map, self.static = set(skip), set(fail_map), static
        self.calls, self.yielded, self.saved, self.events = [], [], [], []
        self.cache_after_read, self.cleanup_states, self.cleanup_values = [], [], []
        self.decisions = []
        self._summary: dict[str, Any] | None = None
        self.map_active = self.reads_active = 0
        self.lock = threading.Lock()
        self.map_outputs = {}  # item -> weakref.ref to its map output
        self.values = {}  # item -> weakref.ref to g0's value (an array)
        self.saves = 0  # save (or on_iterate) calls so far: one per item, in order
        self.on_map: Callable[[int], object] | None = None
        self.on_save: Callable[..., object] | None = None
        self.on_event: Callable[[dict], object] | None = None
        self._executor: process.Executor | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self.pool_calls = PoolCalls()
        self.chain = Chain()

        def factory(context):
            context.add_cleanup(self.record_cleanup)
            if reader is not None:
                context.add_cleanup(reader.close)
                source = reader
            else:
                source = (np.full(4, i, np.float32) for i in range(count))
            generator = Generator(lambda: Reads(self, source), count, fail_fast)
            generator.source_paths = ()
            return generator

        self.chain.add_node(make_node("g0", "generator", factory, 0, 1, iterator=True))
        # A node's input is affine[0] * item + affine[1].
        previous, affine = "g0", (1, 0)
        for name in nodes:
            self.add_function(name, previous, affine, impure=name in impure)
            previous = name
            affine = (
                (affine[0] * 10, affine[1] * 10 + 7)
                if name == "map"
                else (affine[0], affine[1] + 1)
            )
        if static:

            def constant():
                self.calls.append(("static", None))
                return np.full(4, 100, np.float32)

            node = make_node("static", "regularNode", constant, 0, 1)
            object.__setattr__(node.data, "side_effects", False)
            set_schema(node, "chainner:image:test-static")
            self.chain.add_node(node)
            self.connect("static", "map", 1)

        def lazy_read(value):
            data = value.value
            self.cache_after_read.append(self.executor.node_cache.has(NodeId("map")))
            return data

        if collector:

            def collect(_context, _item):
                return Collector(self.deliver, lambda: None)

            sink = make_node("collect", "collector", collect, 1, 0)
        else:
            sink = make_node(
                "save",
                "regularNode",
                functools.partial(self.deliver, read=lazy_read if lazy else None),
                1,
                0,
            )
            if lazy:
                inputs = [BaseInput("any", "Input0").with_id(0).make_lazy()]
                object.__setattr__(sink.data, "inputs", inputs)
        self.chain.add_node(sink)
        self.connect(previous, sink.id, 0)

    @property
    def executor(self) -> process.Executor:
        """The running executor (execute() made it)."""
        assert self._executor is not None, "the pipeline has not started"
        return self._executor

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """The running executor's event loop."""
        assert self._loop is not None, "the pipeline has not started"
        return self._loop

    @property
    def task(self) -> asyncio.Task[None]:
        """The task that runs the executor."""
        assert self._task is not None, "the pipeline has not started"
        return self._task

    @property
    def summary(self) -> dict[str, Any]:
        """The run's item-window summary; the run must have logged one."""
        assert self._summary is not None, "the run logged no window summary"
        return self._summary

    def connect(self, source, target, input_id):
        self.chain.add_edge(
            Edge(
                EdgeSource(NodeId(source), OutputId(0)),
                EdgeTarget(NodeId(target), input_id),
            )
        )

    def add_function(self, name, source, affine, *, impure):
        def compute(value, *_static):  # map ignores the static input
            item = round((float(value[0]) - affine[1]) / affine[0])
            self.calls.append((name, item))
            if name != "map":
                return value + 1
            with self.lock:
                self.map_active += 1
            try:
                if self.on_map is not None:
                    self.on_map(item)
                if item in self.fail_map:
                    raise ValueError(f"map failed {item}")
                output = value * 10 + 7
                self.map_outputs[item] = weakref.ref(output)
                return output
            finally:
                with self.lock:
                    self.map_active -= 1

        inputs = 2 if name == "map" and self.static else 1
        node = make_node(name, "regularNode", compute, inputs, 1)
        object.__setattr__(node.data, "side_effects", False)
        set_schema(
            node,
            f"test:item-window:{name}" if impure else f"chainner:image:test-{name}",
        )
        self.chain.add_node(node)
        self.connect(source, name, 0)

    def deliver(self, value, read=None):
        """Save's (or the collector's on_iterate's) body: on_save, then append."""
        index = self.saves
        self.saves += 1
        if self.on_save is not None:
            self.on_save(index)
        if index not in self.skip:
            self.saved.append(float((value if read is None else read(value))[0]))

    def record_cleanup(self):
        """g0's chain cleanup: maps running, reads running and later map outputs
        alive; separately, later generator values alive (cleanup_values)."""
        progress = [
            e["data"]["index"] for e in self.events if e["event"] == "node-progress"
        ]
        last = progress[-1] if progress else 0

        def alive(references):
            return sum(
                1
                for item, reference in list(references.items())
                if item > last and reference() is not None
            )

        self.cleanup_states.append(
            (self.map_active, self.reads_active, alive(self.map_outputs))
        )
        self.cleanup_values.append(alive(self.values))

    def event(self, value):
        self.events.append(value)
        if self.on_event is not None:
            self.on_event(value)

    def run(self, workers=2, timeout=10, pool=None):
        """Run on a pool of `workers` threads (pool_size=workers); return the run's error.

        `pool`, if given, is that pool (of `workers` threads; the run shuts it down),
        else a new ThreadPoolExecutor. The run is bounded twice: asyncio.wait_for, and
        its loop's own thread, which must finish in timeout + 10 s (await_owned_node
        outlasts a cancellation).
        """
        set_window(self.monkeypatch, self.window)
        logger, handler = logging.getLogger("sanic.root"), Records()
        level = logger.level
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        errors = []

        async def body():
            errors.append(await self.execute(workers, timeout, pool))

        try:
            drive(body, timeout + 10)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(level)
        [error] = errors
        self.decisions = decisions(handler)
        summaries = window_summaries(handler)
        assert len(summaries) <= 1
        self._summary = summaries[0] if summaries else None
        return error

    def settle(self, _index=None, own: int | None = 1):
        """From a node on the pool (on_save, on_map): wait until the engine settled,
        the lead a later item's speculation needs (PoolCalls.settle)."""
        self.pool_calls.settle(own)

    async def execute(self, workers, timeout, given):
        self._loop = asyncio.get_running_loop()
        self.pool_calls.install()
        with given or ThreadPoolExecutor(max_workers=workers) as pool:
            self._executor = process.Executor(
                process.ExecutionId("item-window"),
                self.chain,
                False,
                ExecutionOptions({}),
                self._loop,
                EventCallback(self.event),
                pool,
                self.directory,
                pool_size=workers,
            )
            self._task = asyncio.create_task(self._executor.run())
            try:
                await asyncio.wait_for(self._task, timeout)
            except BaseException as error:
                return error  # the run's error, a cancellation or the timeout
        return None


def test_later_items_compute_ahead_and_replay_in_order(monkeypatch, tmp_path):
    reference = Pipeline(monkeypatch, tmp_path, 5, window="1")
    reference.run()
    entered, waited = threading.Event(), []
    pipeline = Pipeline(monkeypatch, tmp_path, 5, window="3")
    pipeline.on_map = lambda index: entered.set() if index == 2 else None
    pipeline.on_save = lambda index: (
        waited.append(entered.wait(5)) if index == 1 else None
    )
    assert pipeline.run() is None
    assert pipeline.decisions == ["item window K=3 (override)"]
    assert waited == [True]  # item 2's map ran while item 1 saved
    # the loop took speculated results from a store
    assert pipeline.summary["replayed"] >= 1
    assert sorted(c for c in pipeline.calls if c[0] == "map") == [
        ("map", i) for i in range(5)
    ]
    assert pipeline.saved == [7.0, 17.0, 27.0, 37.0, 47.0]
    assert stripped(pipeline.events) == stripped(reference.events)


def test_lazy_upstream_replays_inside_its_consumer(monkeypatch, tmp_path):
    runs = {}
    for window in ("1", "3"):
        pipeline = Pipeline(
            monkeypatch, tmp_path, 5, window=window, lazy=True, skip={1, 3}
        )
        # gives the engine a lead; events are unchanged by it
        pipeline.on_save = pipeline.settle
        assert pipeline.run() is None
        runs[window] = pipeline
    # written items' maps replayed inside save's lazy read
    assert runs["3"].summary["replayed"] >= 1
    # item 3's speculated map, never pulled by the skipping save
    assert runs["3"].summary["discarded"] >= 1
    assert stripped(runs["3"].events) == stripped(runs["1"].events)
    assert runs["3"].saved == runs["1"].saved == [7.0, 27.0, 47.0]
    # map's hit consumed by the nested read
    assert runs["3"].cache_after_read == [False, False, False]
    shape = [
        s
        for s in stripped(runs["3"].events)
        if s[0] in ("node-start", "node-finish") and s[1] in ("map", "save")
    ]
    assert shape[-4:] == [
        ("node-start", "save", None),
        ("node-start", "map", None),
        ("node-finish", "map", None),
        ("node-finish", "save", None),
    ]


@pytest.mark.parametrize("mode", ["await", "cancel"])
def test_commit_awaits_an_in_flight_job(monkeypatch, tmp_path, mode):
    entered, release, starts, waited = threading.Event(), threading.Event(), [], []
    pipeline = Pipeline(monkeypatch, tmp_path, 5, window="3")
    pipeline.on_map = lambda index: (
        (entered.set(), release.wait(5)) if index == 2 else None
    )
    pipeline.on_save = lambda index: (
        waited.append(entered.wait(5)) if index == 1 else None
    )

    # The 3rd map start is item 2's replay: its job is still running. The take follows
    # in this loop step, so the job is released only after it began waiting.
    def on_event(event):
        if event["event"] == "node-start" and event["data"]["nodeId"] == "map":
            starts.append(event)
            if len(starts) == 3:
                if mode == "cancel":
                    pipeline.task.cancel()
                pipeline.loop.call_soon(release.set)

    pipeline.on_event = on_event
    error = pipeline.run()
    # the loop waited for item 2's running job
    assert waited == [True] and pipeline.summary["awaited"] >= 1
    assert pipeline.calls.count(("map", 2)) == 1  # awaited, never recomputed
    if mode == "await":
        assert error is None and pipeline.saved == [7.0, 17.0, 27.0, 37.0, 47.0]
    else:
        assert isinstance(error, asyncio.CancelledError)
        assert pipeline.cleanup_states == [(0, 0, 0)]
        # Item 3 was read ahead, and no later item's generator value outlives close().
        assert len(pipeline.yielded) >= 4 and pipeline.cleanup_values == [0]


def diamond(pipeline, through=None):
    """Turn g0 -> a -> save into g0 -> {a, b} -> join -> save; a, b, join are pure.

    a and b compute value + 1; join adds them, so item i saves 2 * i + 2. With
    `through`, b reads a node of that name fed by g0 instead (item i saves 2 * i + 3);
    it has Into Directory's schema id, so it never runs ahead, nor does b.
    """
    chain = pipeline.chain
    chain.remove_edge(chain.edge_to(NodeId("save"), 0))
    source, affine = "g0", (1, 0)
    if through is not None:
        pipeline.add_function(through, "g0", (1, 0), impure=False)
        set_schema(chain.nodes[NodeId(through)], "chainner:utility:into_directory")
        source, affine = through, (1, 1)
    pipeline.add_function("b", source, affine, impure=False)

    def join(x, y):
        pipeline.calls.append(("join", round(float(x[0])) - 1))
        return x + y

    node = make_node("join", "regularNode", join, 2, 1)
    object.__setattr__(node.data, "side_effects", False)
    set_schema(node, "chainner:image:test-join")
    chain.add_node(node)
    pipeline.connect("a", "join", 0)
    pipeline.connect("b", "join", 1)
    pipeline.connect("join", "save", 0)


class FinishedFirst(ThreadPoolExecutor):
    """A pool whose submit() returns once its job finished: every job beats the
    callback wrap_future registers. Python 3.14's asyncio.futures._chain_future then
    sets the asyncio future at once on the loop thread (earlier versions always took a
    loop turn), so the node finishes in the loop step that started it, before a
    parallel-gathered sibling starts: upstream's order, a then b."""

    def submit(self, fn, /, *args, **kwargs):
        future = super().submit(fn, *args, **kwargs)
        future.exception()  # waits for the job; its error stays on the future
        return future


class SiblingGatedPool(ThreadPoolExecutor):
    """A pool slower than any loop step: a job submitted after one of `siblings`
    starts runs only once all of them started (on_event, the loop thread's hook). No
    pool job, nor a replay's round trip, then finishes before a sibling starts."""

    def __init__(self, siblings, **kwargs):
        super().__init__(**kwargs)
        self.siblings, self.started = frozenset(siblings), set()
        self.opened = threading.Event()
        self.opened.set()

    def on_event(self, event):
        node = event["data"].get("nodeId")
        if event["event"] != "node-start" or node not in self.siblings:
            return
        if not self.started:
            self.opened = threading.Event()
        self.started.add(node)
        if frozenset(self.started) == self.siblings:
            self.started.clear()
            self.opened.set()

    def submit(self, fn, /, *args, **kwargs):
        opened = self.opened

        def held():
            opened.wait(5)  # a sibling that never starts fails the shape, not the run
            return fn(*args, **kwargs)

        return super().submit(held)


def item_segments(events):
    """Stripped events, cut after each of g0's progress events: the run's head (g0's
    start and progress 0), then each item's events closed by its progress, and the
    tail (g0's final progress and finish)."""
    segments, current = [], []
    for event in stripped(events):
        current.append(event)
        if event[:2] == ("node-progress", "g0"):
            segments.append(current)
            current = []
    return [*segments, current]


@pytest.mark.parametrize(
    "new_pool",
    [
        pytest.param(ThreadPoolExecutor, id="threads"),
        pytest.param(FinishedFirst, id="finished-first"),
    ],
)
def test_diamond_graph_events_match_k1(monkeypatch, tmp_path, new_pool):
    # How a and b, gathered in parallel, interleave is pool timing at any K (SP2;
    # ARCHITECTURE section 7); upstream awaits them in turn. FinishedFirst forces
    # upstream's order at K = 1.
    runs = {}
    for window in ("1", "3"):
        pipeline = Pipeline(monkeypatch, tmp_path, 6, window=window, nodes=("a",))
        diamond(pipeline)
        # FinishedFirst runs each pool call inside its submit, on the loop's thread,
        # ahead of the commit path; Save cannot wait for that loop, so no lead.
        if new_pool is ThreadPoolExecutor:
            pipeline.on_save = pipeline.settle
        assert pipeline.run(pool=new_pool(max_workers=2)) is None
        runs[window] = pipeline
    assert runs["3"].decisions == ["item window K=3 (override)"]
    assert runs["3"].summary["replayed"] >= 1  # siblings were replayed from stores
    assert runs["3"].saved == runs["1"].saved == [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    # Item order kept, and each item's events, as a multiset, are K = 1's.
    assert [sorted(s) for s in item_segments(runs["3"].events)] == [
        sorted(s) for s in item_segments(runs["1"].events)
    ]
    for pipeline in runs.values():
        for item in item_segments(pipeline.events)[1:7]:
            order = [event[:2] for event in item]
            # Each node starts once, then finishes; an input finishes before its
            # consumer starts.
            for node in ("a", "b", "join", "save"):
                assert [e for e, n in order if n == node] == [
                    "node-start",
                    "node-finish",
                ]
            for source, consumer in (("a", "join"), ("b", "join"), ("join", "save")):
                assert order.index(("node-finish", source)) < order.index(
                    ("node-start", consumer)
                )


@pytest.mark.parametrize(
    "through", [pytest.param(None, id="gathered"), pytest.param("c", id="later-flight")]
)
def test_replayed_sibling_finishes_no_sooner_than_a_pool_job(
    monkeypatch, tmp_path, through
):
    # a's sibling is b, gathered with it, or c, which b reads: c's flight starts a
    # loop step after a's. On a SiblingGatedPool, slower than any loop step, both
    # start before either finishes at K = 1; a replay, one pool round trip, keeps
    # that order. On a real pool the order is pool timing at any K: Python 3.14 can
    # finish a pool job in the loop step that started it (FinishedFirst).
    sibling = through or "b"
    runs = {}
    for window in ("1", "3"):
        pipeline = Pipeline(monkeypatch, tmp_path, 6, window=window, nodes=("a",))
        diamond(pipeline, through=through)
        pool = SiblingGatedPool({"a", sibling}, max_workers=2)
        pipeline.on_event = pool.on_event
        pipeline.on_save = pipeline.settle
        assert pipeline.run(pool=pool) is None
        runs[window] = pipeline
    assert runs["3"].decisions == ["item window K=3 (override)"]
    assert runs["3"].summary["replayed"] >= 1  # a was replayed from stores
    first = 2.0 if through is None else 3.0  # item i saves 2 * i + first
    assert runs["3"].saved == runs["1"].saved == [first + 2 * i for i in range(6)]
    for pipeline in runs.values():
        shape = [
            event
            for event, node, _ in stripped(pipeline.events)
            if node in ("a", sibling) and event in ("node-start", "node-finish")
        ]
        assert shape == ["node-start", "node-start", "node-finish", "node-finish"] * 6
    assert sorted(stripped(runs["3"].events)) == sorted(stripped(runs["1"].events))


def test_out_of_order_finishing_jobs_replay_in_item_order(monkeypatch, tmp_path):
    reference = Pipeline(monkeypatch, tmp_path, 6, window="1")
    reference.run()
    finished, waited, starts = [], [], []
    third, taking_second = threading.Event(), threading.Event()

    # Item 2's job finishes after item 3's, and after the commit path, inside item
    # 2's map, began to take it: it awaits a job in flight, and replays item 3's.
    def slow_second(index):
        if index == 2:
            waited.extend((third.wait(5), taking_second.wait(5)))
        finished.append(index)
        if index == 3:
            third.set()

    def on_event(event):
        if event["event"] == "node-start" and event["data"]["nodeId"] == "map":
            starts.append(event)
            if len(starts) == 3:  # item 2's map, before its take
                taking_second.set()

    pipeline = Pipeline(monkeypatch, tmp_path, 6, window="6")
    pipeline.on_map, pipeline.on_event = slow_second, on_event
    # Item 1's save holds the commit path until item 3's job finished.
    pipeline.on_save = lambda index: (
        waited.append(third.wait(5)) if index == 1 else None
    )
    assert pipeline.run(workers=4) is None  # P = 2: items 2 and 3 run at once
    assert waited == [True, True, True]
    assert pipeline.decisions == ["item window K=6 (override)"]
    assert finished.index(3) < finished.index(2)  # item 3's job finished first
    assert pipeline.summary["awaited"] >= 1 and pipeline.summary["replayed"] >= 1
    assert pipeline.saved == reference.saved == [7.0, 17.0, 27.0, 37.0, 47.0, 57.0]
    assert stripped(pipeline.events) == stripped(reference.events)


@pytest.mark.parametrize(
    ("budget", "bound", "speculates"),
    [
        pytest.param(8, 0, False, id="smaller-than-one-item"),
        pytest.param(70, 1, True, id="two-items"),
        pytest.param(10**9, 2, True, id="unbounded"),
    ],
)
def test_memory_budget_bounds_read_ahead(
    monkeypatch, tmp_path, budget, bound, speculates
):
    # E = 16 (item) + 16 (map) bytes
    monkeypatch.setattr(process, "memory_budget", lambda: budget)
    pipeline = Pipeline(monkeypatch, tmp_path, 6, window="3")
    seen = []
    # once the producer read everything the budget and K admit
    pipeline.on_save = lambda index: (
        pipeline.settle(),
        seen.append(len(pipeline.yielded) - (index + 1)),
    )
    assert pipeline.run() is None  # liveness: the waited-for item is always produced
    assert pipeline.decisions == ["item window K=3 (override)"]
    assert pipeline.saved == [7.0, 17.0, 27.0, 37.0, 47.0, 57.0]
    assert max(seen) <= bound and (max(seen) >= 1) == speculates
    assert (
        (pipeline.summary["ahead"] > 0)
        == (pipeline.summary["replayed"] > 0)
        == speculates
    )


def test_engine_runs_at_most_p_jobs(monkeypatch, tmp_path):
    lock, active, peak, met = threading.Lock(), [0], [0], []
    barrier = threading.Barrier(2, timeout=5)

    def slow_map(index):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        if index in (2, 3):  # both engine jobs: they meet only if two jobs run at once
            try:
                barrier.wait()
                met.append(True)
            except threading.BrokenBarrierError:
                met.append(False)
        # Running until every pool call waits, so all the maps the engine would run
        # at once are counted together.
        pipeline.settle(own=None)
        with lock:
            active[0] -= 1

    pipeline = Pipeline(monkeypatch, tmp_path, 10, window="6")
    pipeline.on_map = slow_map
    assert pipeline.run(workers=4) is None  # P = 4 - 2
    assert met == [True, True]
    # items 2 and 3 were engine jobs
    assert pipeline.summary["replayed"] + pipeline.summary["awaited"] >= 2
    assert peak[0] <= 3  # P engine jobs, plus at most one map of the committing item


def test_waited_for_advance_starts_while_p_jobs_run(monkeypatch, tmp_path):
    """A resumed batch: save skips every item after item 0, so the commit path never
    pulls the map the engine computed ahead. The advance it waits for is its own work
    (spec 4.2) and must not wait for such a job's P slot.

    Rationale, the final review's 30-item probe (in-memory generator, 300 ms pure map,
    a save that skips every item, pool 8): K = 1 0.01 s, K = 6 1.21 s; 0.31 s with the
    waited-for advance exempt from P (close() draining the jobs in flight).
    """
    skip = {1, 2, 3, 4, 5}
    reference = Pipeline(monkeypatch, tmp_path, 6, window="1", lazy=True, skip=skip)
    reference.run(workers=3)
    entered, gate, returned = threading.Event(), threading.Event(), threading.Event()
    waited, during, yielded = [], [], []

    def blocked_map(index):  # item 2's map, the single engine job: it holds P
        if index == 2:
            entered.set()
            gate.wait(5)
            returned.set()

    def save(index):
        if index == 1:
            waited.append(entered.wait(5))
        if not returned.is_set():
            during.append(index)
        if index == 5:
            yielded.append(len(pipeline.yielded))
            gate.set()

    pipeline = Pipeline(monkeypatch, tmp_path, 6, window="6", lazy=True, skip=skip)
    pipeline.on_map, pipeline.on_save = blocked_map, save
    assert pipeline.run(workers=3) is None  # P = 3 - 2
    assert pipeline.decisions == ["item window K=6 (override)"]
    assert waited == [True]  # item 2's map ran while item 1 saved
    # While item 2's map held P, the commit path advanced and skipped items 3 to 5.
    assert during == [0, 1, 2, 3, 4, 5] and yielded[0] >= 6
    # The advances are not jobs; the one job's result was never taken.
    assert pipeline.summary["jobs"] == 1 and pipeline.summary["discarded"] == 1
    # Only the waited-for advance bypasses P: no speculative advance while the map held it.
    assert pipeline.summary["ahead"] <= 2
    assert pipeline.saved == reference.saved == [7.0]
    assert stripped(pipeline.events) == stripped(reference.events)


def test_oldest_item_jobs_go_first(monkeypatch, tmp_path):
    pipeline = Pipeline(monkeypatch, tmp_path, 8, window="6", nodes=("map", "post"))
    pipeline.on_save = pipeline.settle
    assert pipeline.run(workers=3) is None  # P = 1
    # Counters cannot order jobs, so the order comes from the nodes' own entry log. With every map and post of
    # items 2-7 served from a store, all of them were engine jobs, and P = 1 runs them one at a time.
    assert pipeline.summary["replayed"] == 12
    for item in range(2, 6):
        assert pipeline.calls.index(("post", item)) < pipeline.calls.index(
            ("map", item + 1)
        )


@pytest.mark.parametrize(
    ("window", "workers", "budget"),
    [("2", 2, None), ("3", 4, None), ("6", 8, None), ("6", 8, 8)],
)
def test_window_settings_never_change_output(
    monkeypatch, tmp_path, window, workers, budget
):
    reference = Pipeline(
        monkeypatch, tmp_path, 7, window="1", nodes=("map", "post"), lazy=True, skip={2}
    )
    reference.run()
    if budget is not None:
        monkeypatch.setattr(process, "memory_budget", lambda: budget)
    pipeline = Pipeline(
        monkeypatch,
        tmp_path,
        7,
        window=window,
        nodes=("map", "post"),
        lazy=True,
        skip={2},
    )
    pipeline.on_save = pipeline.settle
    assert pipeline.run(workers=workers) is None
    assert pipeline.decisions == [f"item window K={window} (override)"]
    assert pipeline.saved == reference.saved
    assert stripped(pipeline.events) == stripped(reference.events)
    summary = pipeline.summary
    # 8 bytes admit nothing ahead
    assert (summary["replayed"] > 0) == (summary["ahead"] > 0) == (budget is None)


def test_non_pure_iterated_input_is_never_speculated(monkeypatch, tmp_path):
    runs = {}
    for window in ("1", "3"):
        pipeline = Pipeline(
            monkeypatch,
            tmp_path,
            6,
            window=window,
            nodes=("pre", "map", "post"),
            impure={"map"},
        )
        pipeline.on_save = pipeline.settle
        assert pipeline.run() is None
        runs[window] = pipeline
    summary = runs["3"].summary
    assert summary["replayed"] >= 1  # the pure node before it does run ahead
    # Only items 2-5 can be speculated (slot 1 is the loop's when the window opens), and only `pre` qualifies, so
    # at most 4 jobs; a speculated `map` or `post` would push the count past one job per item.
    assert 1 <= summary["jobs"] <= 4
    assert runs["3"].saved == runs["1"].saved
    assert stripped(runs["3"].events) == stripped(runs["1"].events)


def test_static_first_needed_later_runs_once_on_the_commit_path(monkeypatch, tmp_path):
    runs = {}
    for window in ("1", "3"):
        pipeline = Pipeline(
            monkeypatch, tmp_path, 6, window=window, static=True, lazy=True, skip={0, 1}
        )
        pipeline.on_save = pipeline.settle
        assert pipeline.run() is None
        runs[window] = pipeline
    summary = runs["3"].summary
    assert runs["3"].calls.count(("static", None)) == 1
    # Before item 2's save shares `static`, no `map` is eligible; afterwards only items 3-5 can be speculated.
    # A job dispatched too early would add a failed job for item 2 or 3 and push the count past 3.
    assert summary["jobs"] <= 3
    assert summary["replayed"] >= 1  # once it was shared, later items ran ahead
    # its start sits where the sequential run puts it
    assert stripped(runs["3"].events) == stripped(runs["1"].events)


def test_collector_consumes_replayed_items_in_order(monkeypatch, tmp_path):
    runs = {}
    for window in ("1", "3"):
        pipeline = Pipeline(monkeypatch, tmp_path, 6, window=window, collector=True)
        # inside the collector's on_iterate
        pipeline.on_save = pipeline.settle
        assert pipeline.run() is None
        runs[window] = pipeline
    assert runs["3"].saved == runs["1"].saved == [7.0, 17.0, 27.0, 37.0, 47.0, 57.0]
    # the collector's inputs were replayed from stores
    assert runs["3"].summary["replayed"] >= 1
    assert stripped(runs["3"].events) == stripped(runs["1"].events)


@pytest.mark.parametrize(("pool", "expected"), [(8, 6), (4, 2), (3, 1), (2, 1), (1, 1)])
def test_engine_jobs_leave_two_pool_threads(pool, expected):
    assert engine_jobs(pool) == expected


def test_output_nbytes_counts_arrays_and_bytes():
    # A prepared encoding (SP3-P9) counts by its length.
    values = [np.zeros(4, np.float32), 7, "x", None, np.zeros((2, 3), np.uint8)]
    assert output_nbytes([*values, b"12345"]) == 27


def test_memory_budget_is_a_quarter_of_available(monkeypatch):
    monkeypatch.setattr(
        item_window.psutil, "virtual_memory", lambda: SimpleNamespace(available=4000)
    )
    assert memory_budget() == 1000


def test_not_speculated_node_and_its_dependents_never_run_ahead(monkeypatch, tmp_path):
    runs = {}
    for window in ("1", "3"):
        pipeline = Pipeline(
            monkeypatch, tmp_path, 6, window=window, nodes=("into_directory", "map")
        )
        # A stand-in for Into Directory: the exclusion is by schema id (audit section 5).
        set_schema(
            pipeline.chain.nodes[NodeId("into_directory")],
            "chainner:utility:into_directory",
        )
        # a lead in which any eligible node would run ahead
        pipeline.on_save = pipeline.settle
        assert pipeline.run() is None
        runs[window] = pipeline
    assert runs["3"].decisions == ["item window K=3 (override)"]
    summary = runs["3"].summary
    # Later items were read ahead, yet neither the excluded node nor its dependent
    # `map` ran ahead.
    assert summary["ahead"] >= 1
    assert summary["jobs"] == 0
    assert runs["3"].saved == runs["1"].saved == [17.0, 27.0, 37.0, 47.0, 57.0, 67.0]
    assert stripped(runs["3"].events) == stripped(runs["1"].events)


class GuardedReader:
    """A Load Video-style reader of `count` items (arrays, as Pipeline's g0 yields).

    next() raises ValueError("generator already executing") on re-entry, as a running
    generator does, and counts `reads` at entry. The first read of an index from
    `block_from` on calls on_block(index) once, then waits on `gate`. close() records
    and raises the same error while a read runs, and otherwise sets `closed`.
    """

    def __init__(self, count, block_from, gate):
        self.count, self.block_from, self.gate = count, block_from, gate
        self.on_block: Callable[[int], object] | None = None
        self.reads = self.index = 0
        self.reading = self.blocked = self.closed = False
        self.close_errors = []
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            if self.reading:
                raise ValueError("generator already executing")
            self.reading = True
            self.reads += 1
        try:
            index = self.index
            if index >= self.count:
                raise StopIteration
            if index >= self.block_from and not self.blocked:
                self.blocked = True
                assert self.on_block is not None
                self.on_block(index)
                self.gate.wait(5)
            self.index += 1
            return np.full(4, index, np.float32)
        finally:
            with self.lock:
                self.reading = False

    def close(self):
        with self.lock:
            if self.reading:
                error = ValueError("generator already executing")
                self.close_errors.append(error)
                raise error
            self.closed = True


def test_pause_stops_admission(monkeypatch, tmp_path):
    pipeline = Pipeline(monkeypatch, tmp_path, 10, window="6")
    entered, paused, waited, snapshot = threading.Event(), threading.Event(), [], {}

    def map_item(index):  # item 2's map is an engine job in flight when Pause lands
        if index == 2:
            entered.set()
            waited.append(paused.wait(5))

    def pause_at_one(index):
        if index == 1:
            waited.append(entered.wait(5))

            def pause():
                pipeline.executor.pause()
                snapshot.update(calls=len(pipeline.calls), reads=len(pipeline.yielded))
                paused.set()
                pipeline.loop.create_task(check_and_resume())

            pipeline.loop.call_soon_threadsafe(pause)

    async def check_and_resume():
        # The job in flight finished, and the window started all it would.
        await pipeline.pool_calls.settled(0)
        snapshot.update(after=len(pipeline.calls), reads_after=len(pipeline.yielded))
        pipeline.executor.resume()

    pipeline.on_map, pipeline.on_save = map_item, pause_at_one
    assert pipeline.run(workers=3) is None  # P = 1
    assert waited == [True, True]
    assert pipeline.decisions == ["item window K=6 (override)"]
    # no job started while paused (one submitted just before may enter late)
    assert snapshot["after"] <= snapshot["calls"] + 1
    # at most the advance the loop was waiting for
    assert snapshot["reads_after"] <= snapshot["reads"] + 1
    assert pipeline.saved == [float(10 * i + 7) for i in range(10)]


@pytest.mark.parametrize("mode", ["abort", "cancel"])
def test_abort_stops_admission_and_drains_before_cleanup(monkeypatch, tmp_path, mode):
    pipeline = Pipeline(monkeypatch, tmp_path, 10, window="6")
    stopped, running, waited, at_stop = threading.Event(), threading.Event(), [], {}

    def map_item(index):
        if index > 2:  # an engine job: it is still in flight when Stop lands
            running.set()
            if mode == "abort":
                stopped.wait(5)
            else:
                # until the cancelled commit path waits for nothing but this job
                pipeline.settle()

    def stop_inside_item_two(index):
        if index == 2:
            # Item 2's maps are done, so a running map is a later item's engine job.
            waited.append(running.wait(5))

            def stop():
                at_stop.update(calls=len(pipeline.calls), running=pipeline.map_active)
                if mode == "abort":
                    pipeline.executor.kill()
                else:
                    pipeline.task.cancel()
                stopped.set()

            pipeline.loop.call_soon_threadsafe(stop)
            if mode == "abort":
                # the loop is still inside item 2: an unchecked engine would keep
                # admitting once the job in flight finished
                pipeline.settle()

    pipeline.on_map = map_item
    pipeline.on_save = stop_inside_item_two
    error = pipeline.run(workers=3)  # P = 1
    assert waited == [True]
    assert pipeline.decisions == ["item window K=6 (override)"]
    assert isinstance(error, Aborted if mode == "abort" else asyncio.CancelledError)
    # items 0-2: the item in progress completes, as today
    assert len(pipeline.saved) == 3
    # engine work was in flight, so the drain below is not vacuous
    assert at_stop["running"] >= 1
    if mode == "abort":
        # no job starts after Stop (one submitted may enter late)
        assert len(pipeline.calls) <= at_stop["calls"] + 1
    # jobs finished and later stores released first
    assert pipeline.cleanup_states == [(0, 0, 0)]


# Stop lands inside item 1's save. The read returns once the commit path waits in
# close() for nothing else (drain), or while it is still inside item 1 (no new read).
@pytest.mark.parametrize("busy", [False, True])
def test_abort_during_a_blocked_read_starts_nothing_more_and_closes_after_it(
    monkeypatch, tmp_path, busy
):
    gate, saving, stopped = threading.Event(), threading.Event(), threading.Event()
    reader = GuardedReader(8, block_from=2, gate=gate)
    pipeline = Pipeline(monkeypatch, tmp_path, 8, window="3", reader=reader)
    killed = {}

    def on_block(_index):  # inside next() for item 2, in the producer
        saving.wait(5)

        def kill():
            killed.update(reads=reader.reads, at=time.monotonic())
            pipeline.executor.kill()
            stopped.set()
            if busy:
                gate.set()

        pipeline.loop.call_soon_threadsafe(kill)
        if not busy:
            pipeline.settle()  # this read is all the commit path still waits for
            gate.set()

    def save(index):
        if index == 1:
            saving.set()
            stopped.wait(5)
            if busy:
                gate.wait(5)
                pipeline.settle()  # the window starts whatever it would

    reader.on_block = on_block
    pipeline.on_save = save
    error = pipeline.run(workers=4)
    assert pipeline.decisions == ["item window K=3 (override)"]
    assert isinstance(error, Aborted)
    assert reader.reads == killed["reads"]  # no read starts after Stop
    # the cleanup closed it after the read returned
    assert reader.closed and reader.close_errors == []
    assert time.monotonic() - killed["at"] < 3.0  # inside the host's kill budget


def test_abort_between_zipped_reads_stops_the_slot(monkeypatch, tmp_path, caplog):
    set_window(monkeypatch, "3")
    gate, sink_entered, at_kill = threading.Event(), threading.Event(), {}
    scenario = Scenario(
        monkeypatch, tmp_path, [list(range(6)), list(range(10, 16))], source_paths=()
    )
    blocked = []

    def on_yield(node_id, value):
        # The producer is inside slot 2's first read: g0's or g1's, as the group's
        # generator order comes from a set (hash seed).
        if value % 10 == 2 and not blocked:
            blocked.append(node_id)
            # Stop lands while the loop is inside item 1, so closing cannot be what
            # stops the second generator
            sink_entered.wait(5)

            def kill():  # the read continues once Stop landed
                at_kill.update(scenario.advances)
                scenario.executor.kill()
                gate.set()

            scenario.executor.loop.call_soon_threadsafe(kill)
            gate.wait(5)

    def busy_sink(values):
        if values[0] == 1:
            sink_entered.set()
            gate.wait(5)
            scenario.pool_calls.settle()  # the producer does whatever it would

    scenario.on_yield = on_yield
    scenario.on_sink = busy_sink
    with caplog.at_level(logging.INFO, logger="sanic.root"), pytest.raises(Aborted):
        run(scenario)
    assert decisions(caplog) == ["item window K=3 (override)"]
    # The second generator is not read for slot 2 after Stop, as today's suspend()
    # ensures, and no generator is read for a later slot.
    assert dict(scenario.advances) == at_kill


def test_fail_fast_raises_the_earliest_items_stored_error(monkeypatch, tmp_path):
    runs = {}
    for window in ("1", "3"):
        pipeline = Pipeline(monkeypatch, tmp_path, 6, window=window, fail_map={2, 3})
        pipeline.on_save = pipeline.settle
        runs[window] = (pipeline, pipeline.run())
    (one, error_one), (three, error_three) = runs["1"], runs["3"]
    assert str(error_three) == str(error_one) == "map failed 2"
    # no side effect after the failing item
    assert three.saved == one.saved == [7.0, 17.0]
    assert stripped(three.events) == stripped(one.events)
    # Item 2's map is the only store entry the loop takes before it raises; its error
    # came from the store:
    assert (
        three.summary["replayed"] + three.summary["awaited"] == 1
        and three.calls.count(("map", 2)) == 1
    )
    assert ("map", 3) in three.calls  # speculation ran past the failing item


def test_deferred_errors_keep_item_order(monkeypatch, tmp_path):
    runs = {}
    for window in ("1", "3"):
        pipeline = Pipeline(
            monkeypatch, tmp_path, 6, window=window, fail_map={2, 4}, fail_fast=False
        )
        pipeline.on_save = pipeline.settle
        runs[window] = (pipeline, pipeline.run())
    (one, error_one), (three, error_three) = runs["1"], runs["3"]
    assert (
        str(error_three)
        == str(error_one)
        == "Errors occurred during iteration:\n- map failed 2\n- map failed 4"
    )
    assert three.saved == one.saved == [7.0, 17.0, 37.0, 57.0]
    assert stripped(three.events) == stripped(one.events)
    for item in (2, 4):
        # the stored error was raised, never recomputed
        assert three.calls.count(("map", item)) == 1
    # items 2-5 (two errors, two results) all from stores
    assert three.summary["replayed"] + three.summary["awaited"] == 4


def test_dependents_of_a_stored_error_meet_it_at_the_nesting_point(
    monkeypatch, tmp_path
):
    runs = {}
    for window in ("1", "3"):
        pipeline = Pipeline(
            monkeypatch,
            tmp_path,
            5,
            window=window,
            nodes=("map", "post"),
            fail_map={2},
            fail_fast=False,
        )
        pipeline.on_save = pipeline.settle
        pipeline.run()
        runs[window] = pipeline
    assert ("post", 2) not in runs["3"].calls  # never computed ahead or at commit
    assert runs["3"].calls.count(("map", 2)) == 1
    # Item 2's stored error, then map and post of items 3 and 4: five store takes,
    # none recomputed.
    assert runs["3"].summary["replayed"] + runs["3"].summary["awaited"] == 5
    assert stripped(runs["3"].events) == stripped(runs["1"].events)


# The 113 schema ids admitted by pure_cpu_dependency's per-node conditions, each
# audited in native/reports/sp3/thread-safety-audit.md (SP3 Task 4).
AUDITED_SPECULATION_IDS: frozenset[str] = frozenset(
    [
        f"chainner:image:{name}"
        for name in """
        add add_noise add_normals alpha_matting average_color_fix balance_normals
        bilateral_blur blend blur brightness_and_contrast canny_edge_detection caption
        change_colorspace chroma_key clamp color_levels color_transfer combine_rgba
        convert_normal_map create_checkerboard create_color create_colorwheel
        create_gradient create_noise crop crop_content dilate distance_transform dither
        divide edge_detection erode fast_nlmeans fill_alpha flip gamma gaussian_blur
        generate_hash generate_threshold get_bbox get_dims high_pass hue_and_saturation
        image_convolve image_metrics image_statistics inpaint invert lens_blur log2lin
        lut median_blur merge_channels merge_transparency metal_to_specular multiply
        normal_generator normalize_normal_map opacity pad palette_dither
        palette_from_image pick_color pixelate premultiplied_alpha quantize_to_referece
        remove_border resize resize_pixel_art resize_to_side rotate sharpen sharpen_hbf
        shift specular_to_metal split_channels split_transparency stack
        strengthen_normals stretch_contrast text_as_image threshold threshold_adaptive
        z_stack
        """.split()
    ]
    + [
        f"chainner:utility:{name}"
        for name in """
        back_directory color color_from_channels compare derive_seed directory
        directory_to_text execution_number into_directory math math_round note number
        parse_number pass_through percent random_number regex_find regex_replace
        resolutions separate_color switch text text_append text_length text_padding
        text_pattern text_replace text_slice
        """.split()
    ]
)


def make_lazy_lines(expression, bindings):
    """Whether the expression, or a module-level name it uses, calls .make_lazy()."""
    pending, seen = [expression], set()
    while pending:
        node = pending.pop()
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "make_lazy"
            ):
                return True
            if isinstance(sub, ast.Name) and sub.id in bindings and sub.id not in seen:
                seen.add(sub.id)
                pending.append(bindings[sub.id])
    return False


def speculation_eligible_ids():
    """The schema ids pure_cpu_dependency admits per node, by an AST scan of the packages.

    Nothing is imported: each `<group>.register(...)` decorator is read with `ast`.
    """
    ids = set()
    for path in sorted((ROOT / "backend/src/packages").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        bindings = {}  # module-level names an inputs expression may use
        for statement in tree.body:
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        bindings[target.id] = statement.value
            elif (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.value is not None
            ):
                bindings[statement.target.id] = statement.value
            elif isinstance(statement, ast.FunctionDef):
                bindings[statement.name] = statement
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef):
                continue
            for decorator in function.decorator_list:
                if not (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "register"
                ):
                    continue
                keywords = {k.arg: k.value for k in decorator.keywords if k.arg}
                schema = keywords.get("schema_id") or decorator.args[0]

                def flag(name, default, keywords=keywords):
                    value = keywords.get(name)
                    return default if value is None else ast.literal_eval(value)

                schema_id = ast.literal_eval(schema)
                inputs = keywords.get("inputs")
                if (
                    flag("kind", "regularNode") == "regularNode"
                    and flag("side_effects", False) is False
                    and flag("node_context", False) is False
                    and schema_id.startswith(("chainner:image:", "chainner:utility:"))
                    and (inputs is None or not make_lazy_lines(inputs, bindings))
                ):
                    ids.add(schema_id)
    return ids


def test_speculation_eligible_set_is_the_audited_set():
    eligible = speculation_eligible_ids()
    assert eligible == AUDITED_SPECULATION_IDS, (
        f"not audited: {sorted(eligible - AUDITED_SPECULATION_IDS)}; "
        f"audited but no longer eligible: {sorted(AUDITED_SPECULATION_IDS - eligible)}. "
        "A node that pure_cpu_dependency admits may run ahead of earlier items: it needs a "
        "thread-safety audit entry (native/reports/sp3/thread-safety-audit.md) before it "
        "joins AUDITED_SPECULATION_IDS, or an exclusion in process._NOT_SPECULATED."
    )


class Planner:
    """One candidate, `map`, fed by the generator g0; its job returns work(value).

    sizes(value) gives the bytes of an advance value or a job's output (default 0).
    With jobs False, `map` is never eligible: the window runs no node job.
    """

    def __init__(self, work, sizes=lambda _value: 0, *, jobs=True):
        self.work, self.sizes, self.jobs = work, sizes, jobs
        self.candidates = (Candidate(NodeId("map"), frozenset()),)

    def eligible(self, _node_id, _values, _store):
        return self.jobs

    def job(self, _node_id, values, _store):
        value = values[NodeId("g0")]
        return lambda: (self.work(value), 0.0)

    def nbytes(self, value):
        return self.sizes(value)


class ManualPool(concurrent.futures.Executor):
    """Holds each submitted call until the test finishes it; with hold False, runs it at once."""

    def __init__(self):
        self.calls = []
        self.hold = True

    def submit(self, fn, /, *args, **kwargs):
        future = concurrent.futures.Future()
        self.calls.append((functools.partial(fn, *args, **kwargs), future))
        if not self.hold:
            self.finish(-1)
        return future

    def finish(self, index):
        call, future = self.calls[index]
        try:
            future.set_result(call())
        except Exception as error:
            future.set_exception(error)


def new_window(
    pool,
    planner,
    values=(),
    *,
    size=3,
    jobs=1,
    budget=10**9,
    estimate=0,
    generators=None,
    progress=None,
):
    """A window over the inline generator g0 yielding values, then exhausted, or over
    `generators`."""
    if generators is None:
        items = iter(values)
        generators = [
            SlotGenerator(
                NodeId("g0"), lambda: next(items, None), inline=True, fail_fast=True
            )
        ]
    return ItemWindow(
        size=size,
        generators=generators,
        stops=0,
        loop=asyncio.get_running_loop(),
        pool=pool,
        jobs=jobs,
        budget=budget,
        estimate=estimate,
        planner=planner,
        progress=ProgressController() if progress is None else progress,
    )


async def settle(condition, steps=200):
    """Yield to the loop until condition() holds (bounded)."""
    for _ in range(steps):
        if condition():
            return
        await asyncio.sleep(0)
    raise AssertionError("the window never reached the expected state")


@pytest.mark.parametrize("fails", [False, True])
def test_close_awaits_jobs_of_released_slots(fails):
    started, gate = threading.Event(), threading.Event()

    def work(value):
        started.set()
        assert gate.wait(5)
        if fails:
            raise ValueError(f"job failed {value}")
        return value

    async def body():
        with ThreadPoolExecutor(max_workers=2) as pool:
            try:
                window = new_window(pool, Planner(work), [1, 2, 3])  # P = 1
                window.begin(0)
                assert await asyncio.to_thread(started.wait, 5)  # slot 1's job runs
                window.begin(1)
                window.release(1)  # the loop dropped item 1 without taking its result
                closing = asyncio.create_task(window.close())
                await spin()
                assert not closing.done()  # close() waits for the released slot's job
            finally:
                gate.set()
            await asyncio.wait_for(closing, 5)
        assert window.jobs_started == 1
        # its late result, or late error, was dropped, not stored
        assert window.discarded == 1
        assert not window.holds(NodeId("map"))

    drive(body)


@pytest.mark.parametrize("fails", [False, True])
def test_take_node_right_after_job_completion(fails):
    runs = []

    def work(value):
        runs.append(value)
        if fails:
            raise ValueError(f"job failed {value}")
        return value * 10

    async def body():
        pool = ManualPool()
        window = new_window(pool, Planner(work), [1, 2], jobs=2)
        window.begin(0)
        await settle(lambda: pool.calls)  # slot 1's job was submitted
        window.begin(1)
        # The job completes in its worker; one loop step later its future is done,
        # and its done callback runs a step after that.
        pool.finish(0)
        await asyncio.sleep(0)
        pool.hold = False  # a replay's round trip completes at once
        if fails:
            with pytest.raises(ValueError, match="job failed 2"):
                await window.take_node(NodeId("map"))
        else:
            assert await window.take_node(NodeId("map")) == (20, 0.0)
        assert runs == [2]  # a failed job's error is raised, never recomputed
        assert window.replayed + window.awaited == 1
        await window.close()

    drive(body)


def test_take_node_raises_the_error_of_the_job_it_awaits():
    runs, raised = [], []

    def work(value):
        runs.append(value)
        raised.append(ValueError(f"job failed {value}"))
        raise raised[-1]

    async def body():
        pool = ManualPool()
        window = new_window(pool, Planner(work), [1, 2, 3])  # P = 1
        window.begin(0)
        assert await window.take_advance(NodeId("g0")) == 1
        await settle(lambda: pool.calls)  # slot 1's job is held
        window.release(0)
        window.begin(1)
        assert await window.take_advance(NodeId("g0")) == 2
        taking = asyncio.create_task(window.take_node(NodeId("map")))
        await asyncio.sleep(0)  # the take has the job and waits for it
        assert window.awaited == 1 and window.replayed == 0
        pool.finish(0)  # the job fails while the take awaits it
        with pytest.raises(ValueError, match="job failed 2") as caught:
            await asyncio.wait_for(taking, 5)
        assert caught.value is raised[0]  # the job's own error, raised once
        assert runs == [2]  # the node ran once, never recomputed
        pool.hold = False
        for index in range(len(pool.calls)):  # held jobs end, so close() can drain
            if not pool.calls[index][1].done():
                pool.finish(index)
        await window.close()

    drive(body)


@pytest.mark.parametrize("case", ["past-the-end", "released"])
def test_unproducible_slot_raises_instead_of_hanging(case):
    async def body():
        pool = ManualPool()
        if case == "past-the-end":
            window = new_window(pool, Planner(lambda value: value), [7])
            window.begin(0)
            assert await window.take_advance(NodeId("g0")) == 7
            window.release(0)
            window.begin(1)
            assert await window.take_advance(NodeId("g0")) is None  # producer ended
            window.release(1)
            window.begin(2)  # past the end: a bug in the loop or the window
            slot = 2
        else:  # the producer has not ended, but slot 0's outcomes were dropped
            window = new_window(pool, Planner(lambda value: value), [7, 8, 9])
            window.begin(0)
            assert await window.take_advance(NodeId("g0")) == 7
            window.release(0)  # a bug: the loop takes slot 0's advance again
            slot = 0
        with pytest.raises(
            RuntimeError, match=f"item window: slot {slot} cannot be produced"
        ):
            await asyncio.wait_for(window.take_advance(NodeId("g0")), 2)
        pool.hold = False
        for index in range(len(pool.calls)):  # held jobs end, so close() can drain
            if not pool.calls[index][1].done():
                pool.finish(index)
        await window.close()

    drive(body)


@pytest.mark.parametrize(("size", "ahead"), [(1, 3), (100, 2)])
def test_results_taken_from_jobs_raise_the_estimate(size, ahead):
    # Advance values hold 1 byte, each job's output `size` bytes; E starts at 2, B = 50.
    # The commit path catches slot 1's job in flight, so its output never enters a
    # store: only the take itself can raise slot 1's peak, which raises E on release.
    async def body():
        pool = ManualPool()
        planner = Planner(lambda _value: size, sizes=lambda value: value or 0)
        window = new_window(
            pool, planner, [1] * 6, size=2, jobs=2, budget=50, estimate=2
        )
        window.begin(0)
        assert await window.take_advance(NodeId("g0")) == 1
        await settle(lambda: len(pool.calls) == 1)  # slot 1's job is held
        window.release(0)
        window.begin(1)
        assert await window.take_advance(NodeId("g0")) == 1
        await settle(lambda: len(pool.calls) == 2)  # slot 2's job is held too
        taking = asyncio.create_task(window.take_node(NodeId("map")))
        await asyncio.sleep(0)  # the take has the job and waits for it
        pool.finish(0)
        assert await asyncio.wait_for(taking, 5) == (size, 0.0)
        assert window.awaited == 1
        window.release(1)
        window.begin(2)  # slot 3 fits the window; only the budget can refuse it
        assert window.advanced_ahead == ahead
        pool.hold = False
        pool.finish(1)
        await window.close()

    drive(body)


# The two-phase producer (SP3b, spec 4.2-4.4): a serial describe, then each slot's
# materialize as an engine job.


class PhasedSource:
    """g0 as a two-phase generator over `values`.

    describe() returns the next index as its token, None past the end, or raises
    describe_errors[index]; materialize(token) returns that index's value, raising it
    when it is an exception. advance() is the serial producer's call, both at once.
    Each call adds (name, index) to `log` as it starts.
    """

    def __init__(self, values, describe_errors=None):
        self.values, self.describe_errors = values, describe_errors or {}
        self.position = 0
        self.log = []

    def describe(self):
        index = self.position
        self.position += 1
        self.log.append(("describe", index))
        if index in self.describe_errors:
            raise self.describe_errors[index]
        return (index,) if index < len(self.values) else None

    def materialize(self, token):
        (index,) = token
        self.log.append(("materialize", index))
        value = self.values[index]
        if isinstance(value, BaseException):
            raise value
        return value

    def advance(self):
        index = self.position
        self.position += 1
        self.log.append(("advance", index))
        return self.values[index] if index < len(self.values) else None

    def generator(self, *, fail_fast=True, split=True, inline=False, node_id="g0"):
        """This source as a SlotGenerator carrying describe and materialize."""
        return SlotGenerator(
            NodeId(node_id),
            self.advance,
            inline,
            fail_fast,
            describe=self.describe,
            materialize=self.materialize,
            split=split,
        )

    def called(self, name):
        """The indices `name` was called with, in call order."""
        return [index for call, index in self.log if call == name]


class HeldPool(concurrent.futures.Executor):
    """Runs each call as it is submitted, on the loop thread, and holds its result or
    error until the test finishes it by the entry the call added to `log`, as
    ("materialize", 2): a call that runs until then, with no thread involved.

    A call that adds no entry, or whose name is not in `held` (None: every name),
    finishes at once. `running`: the calls submitted and not finished.
    """

    def __init__(self, log, held=None):
        self.log, self.held = log, held
        self.running = {}

    def submit(self, fn, /, *args, **kwargs):
        future = concurrent.futures.Future()
        start = len(self.log)
        try:
            outcome = (fn(*args, **kwargs), None)
        except BaseException as error:  # held, as a pool thread's future holds it
            outcome = (None, error)
        [label] = self.log[start:] or [None]  # one entry per call at most
        self.running[label] = (future, outcome)
        if label is None or (self.held is not None and label[0] not in self.held):
            self.finish(label)
        return future

    def finish(self, label):
        future, (result, error) = self.running.pop(label)
        if error is None:
            future.set_result(result)
        else:
            future.set_exception(error)


class Interrupted(BaseException):
    """Not an Exception, as KeyboardInterrupt is not."""


def no_jobs(sizes=lambda _value: 0):
    """A Planner whose window runs no node job."""
    return Planner(lambda value: value, sizes, jobs=False)


def logged_jobs(source):
    """A planner whose `map` job returns its value and logs ("map", value) as it starts."""
    return Planner(lambda value: source.log.append(("map", value)) or value)


def take(window):
    """The committing slot's g0 outcome, as the loop takes it."""
    return window.take_advance(NodeId("g0"))


async def finish_in_turn(pool, labels):
    """Finish each call once it runs, in this order."""
    for label in labels:
        await settle(lambda label=label: label in pool.running)
        pool.finish(label)


async def close_finishing(window, pool):
    """close(), finishing every call it waits for; nothing starts once it began."""
    closing = asyncio.create_task(window.close())
    await asyncio.sleep(0)  # close() began
    while pool.running:
        for label in list(pool.running):
            pool.finish(label)
        await spin()
    await closing


def test_two_phase_takes_values_in_slot_order_when_materializations_finish_out_of_order():
    # P = 3: slots 1-3 materialize at once and finish in reverse. Each slot's node job
    # starts once its own materialize finished, and the commit path takes the values
    # in slot order (spec 4.2).
    async def body():
        source = PhasedSource([10, 11, 12, 13])
        pool = HeldPool(source.log)
        window = new_window(
            pool, logged_jobs(source), generators=[source.generator()], size=4, jobs=3
        )
        try:
            window.begin(0)
            taking = asyncio.create_task(take(window))
            await finish_in_turn(
                pool,
                [
                    ("describe", 0),
                    ("materialize", 0),
                    ("describe", 1),
                    ("describe", 2),
                    ("describe", 3),
                ],
            )
            assert await taking == 10
            await settle(lambda: ("materialize", 3) in pool.running)
            for slot in (3, 2, 1):
                pool.finish(("materialize", slot))
                await settle(lambda slot=slot: ("map", 10 + slot) in pool.running)
            # A described slot's materialize starts before the next describe (P4).
            assert source.log == [
                ("describe", 0),
                ("materialize", 0),
                ("describe", 1),
                ("materialize", 1),
                ("describe", 2),
                ("materialize", 2),
                ("describe", 3),
                ("materialize", 3),
                ("map", 13),
                ("map", 12),
                ("map", 11),
            ]
            for slot in (1, 2, 3):
                window.release(slot - 1)
                window.begin(slot)
                assert await take(window) == 10 + slot
            assert window.materialized == 4
        finally:
            await close_finishing(window, pool)

    drive(body)


def test_two_phase_error_is_the_same_object_at_its_slot():
    # A materialize error is its slot's outcome, as the same object; fail-fast off, the
    # next slot is still produced.
    failure = ValueError("decode failed 1")

    async def body():
        source = PhasedSource([10, failure, 12])
        pool = HeldPool(source.log, held=())
        window = new_window(
            pool, no_jobs(), generators=[source.generator(fail_fast=False)]
        )
        try:
            taken = []
            for slot in range(4):
                window.begin(slot)
                try:
                    taken.append(await take(window))
                except ValueError as error:
                    taken.append(error)
                window.release(slot)
            assert taken == [10, failure, 12, None] and taken[1] is failure
            assert source.called("materialize") == [0, 1, 2]  # 3: ended at describe
        finally:
            await close_finishing(window, pool)

    drive(body)


def test_fail_fast_later_slot_failing_first_commits_earlier_slot_then_raises():
    # P = 2: slots 1 and 2 materialize at once, and slot 2's fails (fail-fast) before
    # slot 1's finishes. The commit path still takes slot 1's value, then slot 2's
    # error (the same object); nothing is described after slot 2 (spec 4.3).
    failure = ValueError("decode failed 2")

    async def body():
        source = PhasedSource([10, 11, failure, 13, 14])
        pool = HeldPool(source.log)
        window = new_window(
            pool, no_jobs(), generators=[source.generator()], size=4, jobs=2
        )
        try:
            window.begin(0)
            taking = asyncio.create_task(take(window))
            await finish_in_turn(
                pool,
                [("describe", 0), ("materialize", 0), ("describe", 1), ("describe", 2)],
            )
            assert await taking == 10
            await settle(lambda: ("materialize", 2) in pool.running)
            window.release(0)
            window.begin(1)
            taking = asyncio.create_task(take(window))  # waits for slot 1
            await spin()
            # Finished in this order, the loop records slot 2's error before slot 1's
            # value.
            pool.finish(("materialize", 2))
            pool.finish(("materialize", 1))
            assert await taking == 11
            window.release(1)
            window.begin(2)
            with pytest.raises(ValueError, match="decode failed 2") as caught:
                await take(window)
            assert caught.value is failure
            await spin()
            assert source.called("describe") == [0, 1, 2]
            assert window.materialized == 3
        finally:
            await close_finishing(window, pool)

    drive(body)


def test_base_exception_from_materialize_ends_the_sequence_at_its_slot():
    # Fail-fast off: a BaseException from slot 1's materialize is still its outcome, as
    # the same object, and nothing is described after it (spec 4.3). P = 1, so slot 2's
    # describe could start only once slot 1's materialize finished.
    interrupt = Interrupted()

    async def body():
        source = PhasedSource([10, interrupt, 12, 13])
        pool = HeldPool(source.log)
        window = new_window(
            pool, no_jobs(), generators=[source.generator(fail_fast=False)]
        )
        try:
            window.begin(0)
            taking = asyncio.create_task(take(window))
            await finish_in_turn(
                pool,
                [
                    ("describe", 0),
                    ("materialize", 0),
                    ("describe", 1),
                    ("materialize", 1),
                ],
            )
            assert await taking == 10
            await spin()
            assert source.called("describe") == [0, 1]
            window.release(0)
            window.begin(1)
            with pytest.raises(Interrupted) as caught:
                await take(window)
            assert caught.value is interrupt
        finally:
            await close_finishing(window, pool)

    drive(body)


@pytest.mark.parametrize("fail_fast", [False, True])
def test_describe_error_and_exhaustion_are_the_slot_outcome_without_materialize(
    fail_fast,
):
    # Slot 1's describe raises, and slot 3's finds the end. Each is its slot's outcome
    # (the error as the same object) and the slot has no materialize. Fail-fast,
    # nothing is described after the error (spec 4.3).
    failure = ValueError("describe failed 1")

    async def body():
        source = PhasedSource([10, 11, 12], describe_errors={1: failure})
        pool = HeldPool(source.log, held=())
        window = new_window(
            pool, no_jobs(), generators=[source.generator(fail_fast=fail_fast)]
        )
        try:
            taken = []
            for slot in range(2 if fail_fast else 4):
                window.begin(slot)
                try:
                    taken.append(await take(window))
                except ValueError as error:
                    taken.append(error)
                window.release(slot)
            await spin()
            assert taken[1] is failure
            if fail_fast:
                assert taken == [10, failure]
                assert source.called("describe") == [0, 1]
                assert source.called("materialize") == [0]
            else:
                assert taken == [10, failure, 12, None]
                assert source.called("describe") == [0, 1, 2, 3]
                assert source.called("materialize") == [0, 2]
        finally:
            await close_finishing(window, pool)

    drive(body)


def test_described_slot_queued_behind_p_is_still_produced_after_a_later_failure():
    # A window-level sequence: slot 1's describe finishes after the commit path began
    # slot 1 and before it takes it, so its materialize waits for the take while slot
    # 2's describe holds P (P = 1). Slot 2's describe fails (fail-fast), which ends
    # describing; slot 1 is still materialized when taken, and then slot 2's error is
    # raised (spec 4.3: every described slot is materialized unless Stop or close).
    failure = ValueError("describe failed 2")

    async def body():
        source = PhasedSource([10, 11, 12], describe_errors={2: failure})
        pool = HeldPool(source.log)
        window = new_window(pool, no_jobs(), generators=[source.generator()])
        try:
            window.begin(0)
            taking = asyncio.create_task(take(window))
            await finish_in_turn(pool, [("describe", 0), ("materialize", 0)])
            assert await taking == 10
            window.release(0)
            window.begin(1)
            await finish_in_turn(pool, [("describe", 1), ("describe", 2)])
            await spin()
            assert ("materialize", 1) not in source.log  # queued; describing ended
            taking = asyncio.create_task(take(window))
            await finish_in_turn(pool, [("materialize", 1)])
            assert await taking == 11
            window.release(1)
            window.begin(2)
            with pytest.raises(ValueError, match="describe failed 2") as caught:
                await take(window)
            assert caught.value is failure
            assert source.called("describe") == [0, 1, 2]
        finally:
            await close_finishing(window, pool)

    drive(body)


def test_wanted_slot_describe_and_materialize_bypass_p_and_pause():
    # The slot the commit path waits for is described and then materialized while a
    # node job holds P (P = 1), and again while paused (spec 4.2; the resumed-batch
    # regression). A later slot described while paused keeps its token queued until
    # admission continues after the resume.
    async def body():
        source = PhasedSource([10, 11, 12, 13, 14])
        pool = HeldPool(source.log)
        progress = ProgressController()
        window = new_window(
            pool,
            logged_jobs(source),
            generators=[source.generator()],
            size=4,
            progress=progress,
        )
        try:
            window.begin(0)
            taking = asyncio.create_task(take(window))
            await finish_in_turn(
                pool,
                [
                    ("describe", 0),
                    ("materialize", 0),
                    ("describe", 1),
                    ("materialize", 1),
                ],
            )
            assert await taking == 10
            await settle(lambda: ("map", 11) in pool.running)  # slot 1's job holds P
            window.release(0)
            window.begin(1)
            assert await take(window) == 11
            window.release(1)
            window.begin(2)
            taking = asyncio.create_task(take(window))
            await finish_in_turn(pool, [("describe", 2), ("materialize", 2)])
            assert await taking == 12
            window.release(2)
            window.begin(3)
            progress.pause()  # and the job still holds P
            taking = asyncio.create_task(take(window))
            await finish_in_turn(pool, [("describe", 3), ("materialize", 3)])
            assert await taking == 13
            # Resumed, slot 4's describe starts once the job frees P; paused, it
            # finishes.
            progress.resume()
            pool.finish(("map", 11))
            await settle(lambda: ("describe", 4) in pool.running)
            progress.pause()
            pool.finish(("describe", 4))
            await spin()
            assert ("materialize", 4) not in source.log
            progress.resume()
            window.release(3)  # the commit path's next step admits it
            await settle(lambda: ("materialize", 4) in pool.running)
            assert window.jobs_started == 1  # slot 1's job held P throughout
        finally:
            await close_finishing(window, pool)

    drive(body)


def test_describes_stay_within_commit_plus_k_minus_1():
    # K = 3, P = 8; describes finish at once, materializations when taken: the producer
    # describes up to commit + K - 1, never further (spec 4.2).
    async def body():
        source = PhasedSource(list(range(10, 18)))
        pool = HeldPool(source.log, held={"materialize"})
        window = new_window(pool, no_jobs(), generators=[source.generator()], jobs=8)
        try:
            for slot in range(9):
                window.begin(slot)
                taking = asyncio.create_task(take(window))
                last = min(slot + 2, 8)  # 8: the describe that finds the end
                await settle(lambda last=last: ("describe", last) in source.log)
                await spin()  # then nothing further starts
                assert max(source.called("describe")) == last
                if slot < 8:
                    pool.finish(("materialize", slot))
                assert await taking == (10 + slot if slot < 8 else None)
                window.release(slot)
        finally:
            await close_finishing(window, pool)

    drive(body)


@pytest.mark.parametrize(
    ("split", "describe_errors", "before", "after"),
    [
        pytest.param(True, {}, [0, 1], [0, 1, 2], id="split"),
        pytest.param(
            True,
            {1: ValueError("describe failed 1")},
            [0, 1, 2],
            [0, 1, 2],
            id="split-slot-ended-at-describe",
        ),
        pytest.param(False, {}, [0, 1, 2, 3, 4], [0, 1, 2, 3, 4], id="no-split"),
    ],
)
def test_two_phase_memory_counts_2e_until_resolved(
    split, describe_errors, before, after
):
    # E = 10, B = 50, every item 4 bytes; describes finish at once, materializations
    # when the test finishes them. Split, a slot counts 2E from its describe until its
    # outcome is resolved, then max(its bytes, E), and a describe is admitted within
    # B - 2E: two slots, a third once slot 0 is resolved (spec 4.4). A slot ended at
    # describe is resolved there and counts E. Without split a slot counts E: B / E.
    async def body():
        source = PhasedSource([4] * 8, describe_errors=describe_errors)
        pool = HeldPool(source.log, held={"materialize"})
        window = new_window(
            pool,
            no_jobs(sizes=lambda value: value or 0),
            generators=[source.generator(fail_fast=False, split=split)],
            size=8,
            jobs=8,
            budget=50,
            estimate=10,
        )
        try:
            window.begin(0)
            taking = asyncio.create_task(take(window))
            await settle(lambda: ("describe", before[-1]) in source.log)
            await spin()  # then nothing further starts
            assert source.called("describe") == before
            pool.finish(("materialize", 0))
            assert await taking == 4
            await settle(lambda: ("describe", after[-1]) in source.log)
            await spin()
            assert source.called("describe") == after
        finally:
            await close_finishing(window, pool)

    drive(body)


@pytest.mark.parametrize("waiting", ["materialize", "describe"])
def test_stop_abandons_queued_materializations_and_close_drains_running(waiting):
    # Stop lands while the commit path waits for slot 1 (spec 4.3). "materialize"
    # (P = 3): slots 1 and 2 materialize and slot 3 is being described; slot 1's take
    # resolves, slot 3's materialize is abandoned though P has room, and close() waits
    # for slot 2's. "describe" (P = 1): slot 1 is being described; its materialize
    # still starts, and its take resolves. Nothing else starts after Stop.
    async def body():
        source = PhasedSource([10, 11, 12, 13, 14])
        pool = HeldPool(source.log)
        progress = ProgressController()
        window = new_window(
            pool,
            no_jobs(),
            generators=[source.generator()],
            size=4,
            jobs=3 if waiting == "materialize" else 1,
            progress=progress,
        )
        try:
            window.begin(0)
            taking = asyncio.create_task(take(window))
            await finish_in_turn(pool, [("describe", 0), ("materialize", 0)])
            assert await taking == 10
            if waiting == "materialize":
                await finish_in_turn(pool, [("describe", 1), ("describe", 2)])
                await settle(lambda: ("describe", 3) in pool.running)
                assert {("materialize", 1), ("materialize", 2)} <= set(pool.running)
            window.release(0)
            window.begin(1)
            taking = asyncio.create_task(take(window))
            await spin()
            progress.abort()
            stopped = len(source.log)
            if waiting == "materialize":
                pool.finish(("describe", 3))
                await spin()
                pool.finish(("materialize", 1))
                assert await taking == 11
                closing = asyncio.create_task(window.close())
                await spin()
                assert not closing.done()  # slot 2's materialize still runs
                pool.finish(("materialize", 2))
                await asyncio.wait_for(closing, 5)
                assert source.log[stopped:] == []  # slot 3's never started
                assert window.materialized == 3
            else:
                await finish_in_turn(pool, [("describe", 1), ("materialize", 1)])
                assert await taking == 11
                await spin()
                assert source.log[stopped:] == [("materialize", 1)]
        finally:
            await close_finishing(window, pool)

    drive(body)


@pytest.mark.parametrize("order", ["take-then-close", "close-then-take"])
@pytest.mark.parametrize("producer", ["phased", "serial"])
def test_take_once_closing_with_the_slot_in_flight(producer, order):
    # close() begins while slot 0's describe (or serial advance) runs, and slot 0's
    # take starts before or after it. Either way the take waits for that call, which
    # resolves the slot (take_advance's docstring, P10): nothing materializes once
    # closing, so a describe that returns then resolves it with take_advance's
    # RuntimeError, and a serial advance with its value. close() drains the call.
    async def body():
        source = PhasedSource([10, 11])
        pool = HeldPool(source.log)
        generator = (
            source.generator()
            if producer == "phased"
            else SlotGenerator(
                NodeId("g0"), source.advance, inline=False, fail_fast=True
            )
        )
        window = new_window(pool, no_jobs(), generators=[generator])
        label = ("describe" if producer == "phased" else "advance", 0)
        try:
            window.begin(0)
            await settle(lambda: label in pool.running)
            if order == "take-then-close":
                taking = asyncio.create_task(asyncio.wait_for(take(window), 2))
                await spin()  # the take waits for the call in flight
                closing = asyncio.create_task(window.close())
                await asyncio.sleep(0)  # close() began
            else:
                closing = asyncio.create_task(window.close())
                await asyncio.sleep(0)  # close() began, waiting for the call in flight
                taking = asyncio.create_task(asyncio.wait_for(take(window), 2))
            await spin()
            assert not taking.done() and not closing.done()  # both wait for the call
            pool.finish(label)
            if producer == "phased":
                with pytest.raises(
                    RuntimeError, match="item window: slot 0 cannot be produced"
                ):
                    await taking
            else:
                assert await taking == 10
            await asyncio.wait_for(closing, 5)
            assert source.called("materialize") == []
        finally:
            await close_finishing(window, pool)

    drive(body)


@pytest.mark.parametrize("case", ["zipped", "inline", "plain"])
def test_zipped_inline_or_plain_generators_keep_the_serial_producer(case):
    # Only a group of one generator, not inline, that carries describe and materialize
    # has the two-phase producer (P3); the others advance as today (spec 4.2, D3).
    async def body():
        sources = [PhasedSource([10, 11]), PhasedSource([20, 21])]
        if case == "zipped":
            generators = [
                source.generator(node_id=f"g{index}")
                for index, source in enumerate(sources)
            ]
        else:
            del sources[1]
            generators = [
                sources[0].generator(inline=True)
                if case == "inline"
                else SlotGenerator(
                    NodeId("g0"), sources[0].advance, inline=False, fail_fast=True
                )
            ]
        taken = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            window = new_window(pool, no_jobs(), generators=generators)
            for slot in range(3):
                window.begin(slot)
                for generator in generators:
                    taken.append(await window.take_advance(generator.node_id))
                    if taken[-1] is None:
                        break  # the loop's StopIteration
                window.release(slot)
            await window.close()
        assert taken == ([10, 20, 11, 21, None] if case == "zipped" else [10, 11, None])
        assert {name for source in sources for name, _ in source.log} == {"advance"}
        assert window.materialized == 0

    drive(body)


def test_serial_producer_variable_parses_only_one(monkeypatch):
    assert SERIAL_PRODUCER_VARIABLE == "CHAINNER_C_SERIAL_PRODUCER"
    monkeypatch.delenv(SERIAL_PRODUCER_VARIABLE, raising=False)
    assert serial_producer() is False
    for value, expected in [
        ("1", True),
        ("0", False),
        ("", False),
        ("true", False),
        (" 1", False),
        ("01", False),
    ]:
        monkeypatch.setenv(SERIAL_PRODUCER_VARIABLE, value)
        assert serial_producer() is expected, value


# Two-phase side-effect nodes (SP3-P9): a prepare runs ahead, the commit in order.


@pytest.fixture
def phases_registry(monkeypatch):
    """An empty phase registry for one test; its registrations end with it."""
    monkeypatch.setattr(item_window, "_PHASES", {})


class PreparingPlanner(Planner):
    """Planner whose one candidate is save's prepare phase: its job returns work(value)."""

    def __init__(self, work):
        super().__init__(work)
        self.candidates = (Candidate(NodeId("save"), frozenset(), prepare=True),)


FAILED_PREPARE = ValueError("prepare failed")


@pytest.mark.parametrize(
    ("result", "case", "expected"),
    [
        pytest.param(b"encoded", "used", (1, 0, 0), id="stored-and-used"),
        pytest.param(b"encoded", "in-flight", (0, 1, 0), id="awaited-and-used"),
        pytest.param(b"encoded", "unused", (0, 0, 1), id="taken-not-used"),
        pytest.param(None, "used", (0, 0, 1), id="nothing-prepared"),
        pytest.param(FAILED_PREPARE, "used", (0, 0, 1), id="failed"),
        pytest.param(FAILED_PREPARE, "in-flight", (0, 0, 1), id="failed-in-flight"),
    ],
)
def test_take_prepared_counts_each_value_once(result, case, expected):
    # (replayed, awaited, discarded): a used value as take_node counts its result, an
    # unused one (a skipped save), a None result (DDS) or a failed prepare as
    # discarded. A failed prepare is nothing prepared: the node runs whole.
    def work(_value):
        if isinstance(result, Exception):
            raise result
        return result

    async def body():
        pool = ManualPool()
        pool.hold = case == "in-flight"  # else each job runs as it is submitted
        window = new_window(pool, PreparingPlanner(work), [1, 2])
        window.begin(0)
        assert await window.take_advance(NodeId("g0")) == 1
        await settle(lambda: pool.calls)  # slot 1's prepare was submitted
        window.release(0)
        window.begin(1)
        assert await window.take_advance(NodeId("g0")) == 2
        assert not window.holds(NodeId("save"))  # never the node's own result
        taking = asyncio.create_task(window.take_prepared(NodeId("save")))
        if case == "in-flight":
            await asyncio.sleep(0)  # the take has the prepare and waits for it
            pool.finish(0)
        prepared = await asyncio.wait_for(taking, 5)
        if result is None or result is FAILED_PREPARE:
            assert prepared is None  # the node runs whole
        elif case != "unused":
            assert prepared is not None
            assert prepared.result() == result
        window.release(1)
        await window.close()
        assert (window.replayed, window.awaited, window.discarded) == expected

    drive(body)


SAVE_SCHEMA = "test:item-window:save-image"


class SavePipeline(Pipeline):
    """A Pipeline whose save records its whole runs, prepares and commits."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wholes: list[Any] = []
        self.prepares: list[tuple[Any, Any]] = []
        self.commits: list[tuple[Any, Any, Any]] = []


def save_image_pipeline(
    monkeypatch,
    tmp_path,
    window,
    *,
    fmt="PNG",
    skip=False,
    two=(),
    channels=3,
    names=None,
):
    """g0 -> op -> save and g0 -> name -> save, where save is Save Image's own code.

    test_image_io's NEW.save is Save Image's module source: save runs its node function
    whole, or its prepare and commit phases, registered under a test schema id. Each
    is recorded: `wholes` and `prepares` (file name, image received) and `commits`
    (file name, prepared value, lazy image). A whole save or a commit first waits
    until the engine settled: a lead for it. g0 yields item i as a (2, 3, 1) image of
    i / 8; op (pure) gives `channels` channels, 2 for the items in `two`
    (Fortran-ordered for one channel); name (pure) gives the file name item-<i>, or
    names[i]. Files go to out-<window>.
    """
    pipeline = SavePipeline(monkeypatch, tmp_path, 5, window=window)
    directory = tmp_path / f"out-{window}"
    chain = pipeline.chain = Chain()

    def factory(_context):
        images = (np.full((2, 3, 1), i / 8, np.float32) for i in range(5))
        generator = Generator(lambda: images, 5)
        generator.source_paths = ()
        return generator

    chain.add_node(make_node("g0", "generator", factory, 0, 1, iterator=True))

    def item(value):
        return round(float(value.flat[0]) * 8)

    def op(value):
        result = np.repeat(value * 0.5 + 0.25, 2 if item(value) in two else channels, 2)
        return np.asfortranarray(result) if result.shape[2] == 1 else result

    def name(value):
        return (names or {}).get(item(value), f"item-{item(value)}")

    for node_id, function in (("op", op), ("name", name)):
        node = make_node(node_id, "regularNode", function, 1, 1)
        object.__setattr__(node.data, "side_effects", False)
        set_schema(node, f"chainner:image:test-{node_id}")
        chain.add_node(node)
        pipeline.connect("g0", node_id, 0)

    def whole(*inputs):
        pipeline.wholes.append(inputs[3])
        pipeline.settle()
        return NEW.save.save_image_node(*inputs)

    def prepare(inputs):
        pipeline.prepares.append((inputs[3], inputs[0]))
        return NEW.save.prepare_save_image(inputs)

    def commit(inputs, prepared):
        pipeline.commits.append((inputs[3], prepared, inputs[0]))
        pipeline.settle()
        return NEW.save.commit_save_image(inputs, prepared)

    save = make_node("save", "regularNode", whole, 20, 0)
    inputs = [BaseInput("any", f"Input{i}").with_id(i) for i in range(20)]
    inputs[0] = ImageInput().with_id(0).make_lazy()
    inputs[2].make_optional()  # the subdirectory
    object.__setattr__(save.data, "inputs", inputs)
    set_schema(save, SAVE_SCHEMA)
    chain.add_node(save)
    values = save_arguments(
        NEW.save, None, directory, fmt=fmt, skip_existing_files=skip
    )
    for index, value in enumerate(values.values()):
        if index not in (0, 3):
            chain.inputs.set(NodeId("save"), InputId(index), value)
    pipeline.connect("op", "save", 0)
    pipeline.connect("name", "save", 3)
    register_phases(SAVE_SCHEMA, prepare, commit)
    return pipeline, directory


def written(directory):
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


@pytest.mark.usefixtures("phases_registry")
def test_prepared_save_replays_its_upstream_before_writing(monkeypatch, tmp_path):
    counted = []

    def nbytes(values):
        values = list(values)
        counted.extend(values)
        return output_nbytes(values)

    monkeypatch.setattr(process, "output_nbytes", nbytes)
    runs = {}
    for window in ("1", "3"):
        pipeline, directory = save_image_pipeline(monkeypatch, tmp_path, window)
        assert pipeline.run() is None
        runs[window] = (pipeline, written(directory))
    (one, files_one), (three, files_three) = runs["1"], runs["3"]
    assert three.decisions == ["item window K=3 (override)"]
    assert three.summary["replayed"] > 0
    # Later items committed a prepared value taken from a store; K = 1 saves whole.
    assert any(
        prepared is not None and prepared.used for _, prepared, _ in three.commits
    )
    assert one.prepares == one.commits == [] and len(one.wholes) == 5
    assert stripped(three.events) == stripped(one.events)
    shape = [
        s
        for s in stripped(three.events)
        if s[0] in ("node-start", "node-finish") and s[1] in ("op", "save")
    ]
    assert (
        shape
        == [
            ("node-start", "save", None),
            ("node-start", "op", None),
            ("node-finish", "op", None),
            ("node-finish", "save", None),
        ]
        * 5
    )
    assert list(files_one) == [f"item-{i}.png" for i in range(5)]
    assert files_three == files_one
    # prepared bytes count toward the window's in-flight bytes
    assert any(type(value) is bytes and value for value in counted)


@pytest.mark.usefixtures("phases_registry")
def test_prepared_save_skips_without_pulling_its_input(monkeypatch, tmp_path):
    runs = {}
    for window in ("1", "3"):
        pipeline, directory = save_image_pipeline(
            monkeypatch, tmp_path, window, skip=True
        )
        directory.mkdir()
        for i in range(5):
            (directory / f"item-{i}.png").write_bytes(b"existing")
        assert pipeline.run() is None
        assert written(directory) == {f"item-{i}.png": b"existing" for i in range(5)}
        runs[window] = pipeline
    one, three = runs["1"], runs["3"]
    prepared = [value for _, value, _ in three.commits if value is not None]
    assert prepared and not any(value.used for value in prepared)  # all discarded
    assert not any(lazy.has_value for _, _, lazy in three.commits)  # never read
    # each item's op ran ahead and was never taken, as its prepared value
    assert three.summary["discarded"] >= 2 * len(prepared)
    assert not any(e["data"].get("nodeId") == "op" for e in three.events)
    assert stripped(three.events) == stripped(one.events)


@pytest.mark.usefixtures("phases_registry")
@pytest.mark.parametrize("skip", [False, True])
def test_prepare_error_surfaces_where_save_fails_today(monkeypatch, tmp_path, skip):
    # Item 3's op gives 2 channels, which GIF cannot hold: its prepare raises. A failed
    # prepare is nothing prepared, so item 3's save runs whole and fails (or skips)
    # exactly as at K = 1.
    runs = {}
    for window in ("1", "3"):
        pipeline, directory = save_image_pipeline(
            monkeypatch, tmp_path, window, fmt="GIF", skip=skip, two={3}
        )
        if skip:
            directory.mkdir()
            for i in range(5):
                (directory / f"item-{i}.gif").write_bytes(b"existing")
        runs[window] = (pipeline, pipeline.run(), written(directory))
    (one, error_one, files_one), (three, error_three, files_three) = runs.values()
    assert "item-3" in [name for name, _ in three.prepares]  # it ran ahead and raised
    assert "item-3" in three.wholes
    assert "item-3" not in [name for name, _, _ in three.commits]
    assert stripped(three.events) == stripped(one.events)
    assert files_three == files_one
    if skip:
        assert error_one is None and error_three is None
    else:
        assert type(error_one) is type(error_three) is process.NodeExecutionError
        assert str(error_three) == str(error_one)
        assert str(error_one).startswith("Unsupported number of channels.")


@pytest.mark.usefixtures("phases_registry")
def test_prepare_receives_the_image_the_lazy_read_produces(monkeypatch, tmp_path):
    # op gives Fortran-ordered (2, 3, 1) images; Save's ImageInput makes them 2-D. A
    # prepare receives what its commit's lazy read produces: dtype, shape, strides and
    # pixels.
    pipeline, _ = save_image_pipeline(monkeypatch, tmp_path, "3", channels=1)
    assert pipeline.run() is None
    received = dict(pipeline.prepares)
    read = {name: lazy.value for name, prepared, lazy in pipeline.commits if prepared}
    assert read

    def described(value):
        return value.dtype.str, value.shape, value.strides, value.tobytes()

    for name, value in read.items():
        assert value.ndim == 2 and not value.flags.c_contiguous
        assert described(received[name]) == described(value)


@pytest.mark.usefixtures("phases_registry")
@pytest.mark.parametrize("case", ["prepared-nothing", "input-never-stored"])
def test_unprepared_save_runs_whole(monkeypatch, tmp_path, case):
    # The one decision point: without a prepared value (its prepare returned None, as
    # for DDS, or its input is never stored, as behind an impure node) the node runs
    # whole, as at K = 1.
    prepares, commits, runs = [], [], {}

    def prepare(inputs):
        prepares.append(inputs[0])

    for window in ("1", "3"):
        impure = {"map"} if case == "input-never-stored" else set()
        pipeline = Pipeline(
            monkeypatch, tmp_path, 6, window=window, lazy=True, impure=impure
        )
        set_schema(pipeline.chain.nodes[NodeId("save")], "test:item-window:phased")
        register_phases(
            "test:item-window:phased",
            prepare,
            lambda _inputs, prepared: commits.append(prepared),
        )
        pipeline.on_save = pipeline.settle
        assert pipeline.run() is None
        runs[window] = pipeline
    assert commits == []
    assert bool(prepares) == (case == "prepared-nothing")
    assert runs["3"].saved == runs["1"].saved
    assert stripped(runs["3"].events) == stripped(runs["1"].events)


@pytest.mark.usefixtures("phases_registry")
def test_prepared_collector_commits_in_order(monkeypatch, tmp_path):
    # A collector's prepare receives its iterated inputs as on_iterate would; its
    # commit consumes the prepared values in item order.
    commits, runs = [], {}

    def commit(collector, _inputs, prepared):
        commits.append(prepared)
        collector.on_iterate(prepared.result())

    for window in ("1", "3"):
        pipeline = Pipeline(monkeypatch, tmp_path, 6, window=window, collector=True)
        set_schema(pipeline.chain.nodes[NodeId("collect")], "test:item-window:collect")
        register_phases(
            "test:item-window:collect", lambda inputs: inputs[0].copy(), commit
        )
        pipeline.on_save = pipeline.settle
        assert pipeline.run() is None
        runs[window] = pipeline
    assert runs["3"].saved == runs["1"].saved == [7.0, 17.0, 27.0, 37.0, 47.0, 57.0]
    assert commits and all(prepared.used for prepared in commits)
    assert stripped(runs["3"].events) == stripped(runs["1"].events)


def masked(text, directory):
    """text with the output directory, plain or escaped as in a repr, as <out>."""
    for form in (str(directory), repr(str(directory))[1:-1]):
        text = text.replace(form, "<out>")
    return text


def save_runs(monkeypatch, tmp_path, *, existing=(), setup=None, **options):
    """The same save_image_pipeline run at K = 1 and at K = 3, item 3 failing in both.

    Files of the items in `existing` hold b"existing" first; setup(directory) runs
    next. Returns, per run: the pipeline, the error, its message with the output
    directory masked, and the files left.
    """
    runs = []
    for window in ("1", "3"):
        pipeline, directory = save_image_pipeline(
            monkeypatch, tmp_path, window, **options
        )
        directory.mkdir()
        for i in existing:
            (directory / f"item-{i}.{options['fmt'].lower()}").write_bytes(b"existing")
        if setup is not None:
            setup(directory)
        error = pipeline.run()
        assert isinstance(error, process.NodeExecutionError)
        runs.append(
            SimpleNamespace(
                pipeline=pipeline,
                error=error,
                message=masked(str(error), directory),
                files=written(directory),
            )
        )
    return runs


def ran_ahead(pipeline, name):
    """name's save committed a prepared value at K = 3 (its prepare ran ahead)."""
    return any(n == name and p.used for n, p, _ in pipeline.commits)


@pytest.mark.usefixtures("phases_registry")
def test_failed_prepare_over_an_existing_gif_truncates_as_today(monkeypatch, tmp_path):
    # Item 3's GIF encoder fails, in its prepare and in its save. A failed prepare is
    # nothing prepared, so item 3's save runs whole: Image.save opens the existing file
    # ("w+b"), the encoder fails, the file stays truncated, as at K = 1.
    Image.init()
    encode = Image.SAVE["GIF"]
    color = round((3 / 8 * 0.5 + 0.25) * 255)  # item 3's pixel value

    def failing(image, fp, filename):
        if image.getpixel((0, 0))[0] == color:
            raise ValueError("encoder failed")
        return encode(image, fp, filename)

    monkeypatch.setitem(Image.SAVE, "GIF", failing)
    one, three = save_runs(monkeypatch, tmp_path, existing={3}, fmt="GIF")
    assert one.message == three.message == "encoder failed"
    assert one.files == three.files and three.files["item-3.gif"] == b""
    assert "item-3" in [name for name, _ in three.pipeline.prepares]
    assert "item-3" in three.pipeline.wholes and not ran_ahead(three.pipeline, "item-3")
    assert stripped(three.pipeline.events) == stripped(one.pipeline.events)


@pytest.mark.usefixtures("phases_registry")
@pytest.mark.parametrize("existing", [False, True])
def test_prepared_gif_write_failure_leaves_todays_file(monkeypatch, tmp_path, existing):
    # Item 3's file write fails after 4 bytes (a full disk). Its prepared bytes are
    # written with Image.save's file handling: a file the save created is removed, an
    # existing one stays truncated to the partial write, as at K = 1.
    failing_writes(monkeypatch, "item-3.gif")
    one, three = save_runs(
        monkeypatch, tmp_path, existing={3} if existing else (), fmt="GIF"
    )
    assert one.message == three.message == "[Errno 28] No space left on device"
    assert one.files == three.files
    assert (
        (len(three.files["item-3.gif"]) == 4)
        if existing
        else ("item-3.gif" not in three.files)
    )
    assert ran_ahead(three.pipeline, "item-3")
    assert stripped(three.pipeline.events) == stripped(one.pipeline.events)


def read_only_item_3(directory):
    (directory / "item-3.gif").write_bytes(b"existing")
    os.chmod(directory / "item-3.gif", stat.S_IREAD)


def file_named_blocker(directory):
    (directory / "blocker").write_bytes(b"a file, not a folder")


@pytest.mark.usefixtures("phases_registry")
@pytest.mark.parametrize(
    ("setup", "names"),
    [
        pytest.param(read_only_item_3, None, id="permission-error"),
        pytest.param(file_named_blocker, {3: "blocker/item-3"}, id="missing-directory"),
        pytest.param(None, {3: "x" * 300}, id="path-too-long"),
    ],
)
def test_prepared_gif_save_io_errors_match_k1(monkeypatch, tmp_path, setup, names):
    # Pins: errors the path, folder or file raises before any byte is written are the
    # same error, message and files at K = 3 as at K = 1.
    try:
        one, three = save_runs(
            monkeypatch, tmp_path, setup=setup, names=names, fmt="GIF"
        )
    finally:
        for path in tmp_path.rglob("item-3.gif"):
            os.chmod(path, stat.S_IWRITE)
    assert type(one.error.__cause__) is type(three.error.__cause__)
    assert one.message == three.message
    assert one.files == three.files
    assert stripped(three.pipeline.events) == stripped(one.pipeline.events)


# A fresh interpreter: the system msvcp140.dll first (conftest), then pillow_avif (as
# save_image.py imports it), so Pillow's plugin registry is not initialized yet. Imports
# Pillow, cv2, numpy and backend modules, never torch or onnxruntime. argv: repository
# root, case, output folder.
FRESH_PILLOW = r"""
import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

root, case, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
sys.path[:0] = [str(root / "backend/src"), str(root / "native/tests")]

# The system C++ runtime before Pillow's bundled copy, as in the suite and the worker:
# importing conftest loads the system msvcp140.dll on Windows.
import conftest
import pillow_avif
from PIL import Image

result = {"initialized": Image._initialized}

import numpy as np
from test_image_io import NEW, LazyImage, save_arguments


def batch(window, directory):
    import process
    from api import BaseInput, ExecutionOptions, Generator, NodeId
    from chain.chain import Chain, Edge, EdgeSource, EdgeTarget
    from nodes.impl.item_window import register_phases
    from test_generator_lifecycle import PoolCalls, make_node

    process.registry.get_package = lambda _schema: SimpleNamespace(id="test")
    os.environ["CHAINNER_C_ITEM_WINDOW"] = window
    chain = Chain()
    calls = PoolCalls()

    def factory(_context):
        images = (np.full((2, 3, 3), i / 8, np.float32) for i in range(5))
        generator = Generator(lambda: images, 5)
        generator.source_paths = ()
        return generator

    def name(value):
        return f"item-{round(float(value.flat[0]) * 8)}"

    def whole(*inputs):
        if inputs[3] == "item-1":
            calls.settle()  # the engine prepares items 2 and 3 meanwhile
        return NEW.save.save_image_node(*inputs)

    def prepare(inputs):
        try:
            return NEW.save.prepare_save_image(inputs)
        except Exception as error:
            result["prepare errors"].append(repr(error))
            raise

    def commit(inputs, prepared):
        result["committed"].append(inputs[3])
        return NEW.save.commit_save_image(inputs, prepared)

    chain.add_node(make_node("g0", "generator", factory, 0, 1, iterator=True))
    namer = make_node("name", "regularNode", name, 1, 1)
    object.__setattr__(namer.data, "side_effects", False)
    namer.schema_id = "chainner:image:test-name"
    object.__setattr__(namer.data, "schema_id", namer.schema_id)
    chain.add_node(namer)
    save = make_node("save", "regularNode", whole, 20, 0)
    inputs = [BaseInput("any", f"Input{i}").with_id(i) for i in range(20)]
    inputs[0].make_lazy()
    inputs[2].make_optional()
    object.__setattr__(save.data, "inputs", inputs)
    save.schema_id = "test:fresh-pillow:save"
    object.__setattr__(save.data, "schema_id", save.schema_id)
    chain.add_node(save)
    values = save_arguments(NEW.save, None, directory, fmt="GIF", skip_existing_files=True)
    for index, value in enumerate(values.values()):
        if index not in (0, 3):
            chain.inputs.set(NodeId("save"), index, value)
    for source, target, input_id in (("g0", "name", 0), ("g0", "save", 0), ("name", "save", 3)):
        chain.add_edge(Edge(EdgeSource(NodeId(source), 0), EdgeTarget(NodeId(target), input_id)))
    register_phases(save.schema_id, prepare, commit)
    directory.mkdir(parents=True)
    for i in (0, 1):  # a resumed batch: items 0 and 1 are skipped
        (directory / f"item-{i}.gif").write_bytes(b"existing")

    async def main():
        calls.install()
        with ThreadPoolExecutor(max_workers=4) as pool:
            executor = process.Executor(
                "fresh-pillow", chain, False, ExecutionOptions({}), asyncio.get_running_loop(),
                SimpleNamespace(put=lambda _event: None), pool, directory, pool_size=4,
            )
            await executor.run()

    try:
        asyncio.run(main())
        error = None
    except Exception as caught:
        error = repr(caught)
    return error, {p.name: p.read_bytes().hex() for p in sorted(directory.iterdir())}


if case == "gif-batch":
    result.update({"prepare errors": [], "committed": []})
    result["error"], files = batch("3", out / "k3")
    result["files"] = sorted(files)
    result["committed at K = 3"] = list(result["committed"])
    error_k1, files_k1 = batch("1", out / "k1")
    result["equal to K = 1"] = error_k1 is None and files == files_k1
else:
    value = np.random.default_rng(3).uniform(0, 1, (5, 6, 3)).astype(np.float32)
    inputs = save_arguments(NEW.save, value, out / "unused", fmt=case)
    try:
        prepared = NEW.save.prepare_save_image(list(inputs.values()))
    except Exception as error:
        prepared = repr(error)
    NEW.save.save_image_node(**save_arguments(NEW.save, LazyImage(value), out, fmt=case))
    today = (out / f"image.{case.lower()}").read_bytes()
    result["prepared"] = prepared if isinstance(prepared, str) else len(prepared)
    result["equal"] = prepared == today
result["torch"] = "torch" in sys.modules
result["onnxruntime"] = "onnxruntime" in sys.modules
print(json.dumps(result))
"""


@pytest.mark.parametrize("case", ["gif-batch", "GIF", "TGA", "AVIF"])
def test_fresh_pillow_registry_prepares_as_image_save(tmp_path, case):
    # The prepare picks Pillow's plugin as Image.save(path) does, also before anything
    # in the process initialized Pillow's registry: a resumed GIF batch (items 0 and 1
    # skipped) commits prepared bytes equal to K = 1's, and each format's prepared bytes
    # equal today's file.
    environment = {k: v for k, v in os.environ.items() if k != "CHAINNER_C_ITEM_WINDOW"}
    done = subprocess.run(
        [sys.executable, "-B", "-c", FRESH_PILLOW, str(ROOT), case, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        env=environment,
    )
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout.splitlines()[-1])
    assert result["initialized"] == 0  # a fresh registry
    assert not result["torch"] and not result["onnxruntime"]
    if case == "gif-batch":
        assert result["error"] is None and result["prepare errors"] == []
        assert result["files"] == [f"item-{i}.gif" for i in range(5)]
        assert result["committed at K = 3"]  # items committed their prepared bytes
        assert result["equal to K = 1"]
    else:
        assert result["equal"], result["prepared"]


@pytest.mark.parametrize(
    ("module", "prepare", "commit"),
    [
        ("io/save_image.py", "prepare_save_image", "commit_save_image"),
        ("video_frames/save_video.py", "prepare_video_frame", "commit_video_frame"),
    ],
)
def test_save_modules_register_their_phases(module, prepare, commit):
    # A pin (the modules are never imported in tests): the module-level call registers
    # the node's own schema id with the module's prepare and commit functions.
    path = ROOT / "backend/src/packages/chaiNNer_standard/image" / module
    tree = ast.parse(path.read_text(encoding="utf-8"))
    [schema] = [
        ast.literal_eval(keyword.value)
        for function in tree.body
        if isinstance(function, ast.FunctionDef)
        for decorator in function.decorator_list
        if isinstance(decorator, ast.Call)
        for keyword in decorator.keywords
        if keyword.arg == "schema_id"
    ]
    [call] = [
        statement.value
        for statement in tree.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "register_phases"
    ]
    functions_named = []
    for argument in call.args[1:]:
        assert isinstance(argument, ast.Name)
        functions_named.append(argument.id)
    assert [ast.literal_eval(call.args[0]), *functions_named] == [
        schema,
        prepare,
        commit,
    ]
    functions = {f.name for f in tree.body if isinstance(f, ast.FunctionDef)}
    assert {prepare, commit} <= functions


# SP3b in the executor: the group's single generator is described serially and
# materialized as engine jobs (spec 4.2). Load Images names each item in describe() and
# decodes it in materialize(token); any other generator's describe is its read, and its
# materialize the item's output enforce (an image's normalize).


class NormalizeLog(ImageOutput):
    """An ImageOutput whose enforce (an image's normalize) records its thread in
    `threads`, then calls before(value) when it is set."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threads = []
        self.before: Callable[[object], object] | None = None

    def enforce(self, value):
        self.threads.append(threading.get_ident())
        if self.before is not None:
            self.before(value)
        return super().enforce(value)


def image_files(folder, count, corrupt=()):
    """PNG files 0.png to <count - 1>.png in folder, item i a (3, 4, 3) image of i * 20;
    an item in `corrupt` holds bytes that no decoder reads."""
    folder.mkdir()
    for i in range(count):
        path = folder / f"{i}.png"
        if i in corrupt:
            path.write_bytes(b"not an image")
        else:
            Image.fromarray(np.full((3, 4, 3), i * 20, np.uint8)).save(path)


INDEX_TYPE = "min(uint, max(0, IterOutput0.length - 1))"  # Load Images', Load Video's


def generator_outputs(node, outputs, iterated):
    """Give the generator node these outputs, of which `iterated` are its iterator's."""
    object.__setattr__(
        node.data, "outputs", [output.with_id(i) for i, output in enumerate(outputs)]
    )
    object.__setattr__(
        node.data, "iterable_outputs", [IteratorOutputInfo(outputs=iterated)]
    )


class LoaderPipeline(Pipeline):
    """A Pipeline over a loader node whose image output is `normalize`."""

    def __init__(self, monkeypatch, tmp_path, window, normalize: NormalizeLog):
        super().__init__(monkeypatch, tmp_path, 0, window=window)
        self.normalize = normalize


class VideoPipeline(LoaderPipeline):
    """A LoaderPipeline over Load Video, whose FFmpeg is the fake `env`."""

    def __init__(
        self, monkeypatch, tmp_path, window, normalize: NormalizeLog, env: FakeFFmpeg
    ):
        super().__init__(monkeypatch, tmp_path, window, normalize)
        self.env = env


def load_images_pipeline(monkeypatch, tmp_path, window, loader, *, fail_fast=False):
    """Load Images over tmp_path/in -> Save Image: PNG files named by Load Images' Name
    output, in out-<window>.

    Load Images is its node source with load_image_node replaced by loader
    (test_file_sequence's namespace) and with the outputs the node declares; its image
    output is a NormalizeLog (pipeline.normalize). Save is Save Image's own code, run
    whole or in its prepare and commit phases, as in save_image_pipeline; a whole save
    or a commit first calls pipeline.on_save(name) when it is set.
    """
    pipeline = LoaderPipeline(monkeypatch, tmp_path, window, NormalizeLog())
    directory = tmp_path / f"out-{window}"
    chain = pipeline.chain = Chain()
    module = file_sequence_namespace(current=True, loader=loader)
    module["Generator"] = Generator  # the executor's own
    load = make_node("g0", "generator", module["load_images_node"], 7, 5, iterator=True)
    object.__setattr__(load.data, "node_context", False)
    generator_outputs(
        load,
        [
            pipeline.normalize,
            DirectoryOutput("Directory", output_type="Input0"),
            TextOutput("Subdirectory Path"),
            TextOutput("Name"),
            NumberOutput("Index", output_type=INDEX_TYPE),
        ],
        [0, 2, 3, 4],
    )
    chain.add_node(load)
    options = (tmp_path / "in", False, False, "**/*", False, 10, fail_fast)
    for index, value in enumerate(options):
        chain.inputs.set(NodeId("g0"), InputId(index), value)

    def saving(name):
        if pipeline.on_save is not None:
            pipeline.on_save(name)

    def whole(*inputs):
        saving(inputs[3])
        return NEW.save.save_image_node(*inputs)

    def commit(inputs, prepared):
        saving(inputs[3])
        return NEW.save.commit_save_image(inputs, prepared)

    save = make_node("save", "regularNode", whole, 20, 0)
    inputs = [BaseInput("any", f"Input{i}").with_id(i) for i in range(20)]
    inputs[0] = ImageInput().with_id(0).make_lazy()
    inputs[2].make_optional()  # the subdirectory
    object.__setattr__(save.data, "inputs", inputs)
    set_schema(save, SAVE_SCHEMA)
    chain.add_node(save)
    for index, value in enumerate(save_arguments(NEW.save, None, directory).values()):
        if index not in (0, 3):
            chain.inputs.set(NodeId("save"), InputId(index), value)
    for index in (0, 3):  # the image and its name
        chain.add_edge(
            Edge(
                EdgeSource(NodeId("g0"), OutputId(index)),
                EdgeTarget(NodeId("save"), InputId(index)),
            )
        )
    register_phases(SAVE_SCHEMA, NEW.save.prepare_save_image, commit)
    return pipeline, directory


def load_video_pipeline(monkeypatch, tmp_path, window, frames):
    """Load Video -> sink. Load Video is its node source over test_video_io's
    FakeFFmpeg (pipeline.env), which pipes `frames` 2x2 RGB24 frames, frame i all
    i * 40; its frame output is a NormalizeLog (pipeline.normalize). The sink records
    each frame's bytes and index in pipeline.saved."""
    pipeline = VideoPipeline(
        monkeypatch,
        tmp_path,
        window,
        NormalizeLog("Frame", channels=3),
        FakeFFmpeg(chunks=[bytes([i * 40] * 12) for i in range(frames)]),
    )
    video = video_module("native", "video", pipeline.env)
    module = video_module("native", "load_video", pipeline.env)
    module.update(
        VideoLoader=video["VideoLoader"],
        VideoMetadata=video["VideoMetadata"],
        Generator=Generator,  # the executor's own
    )
    chain = pipeline.chain = Chain()
    load = make_node("g0", "generator", module["load_video_node"], 3, 6, iterator=True)
    generator_outputs(
        load,
        [
            pipeline.normalize,
            NumberOutput("Index", output_type=INDEX_TYPE),
            DirectoryOutput("Video Directory", of_input=0),
            FileNameOutput("Name", of_input=0),
            NumberOutput("FPS", output_type="0.."),
            AudioStreamOutput(),
        ],
        [0, 1],
    )
    chain.add_node(load)
    for index, value in enumerate((tmp_path / "video.mkv", False, 0)):
        chain.inputs.set(NodeId("g0"), InputId(index), value)

    def sink(frame, index):
        pipeline.saved.append((frame.tobytes(), index))

    chain.add_node(make_node("sink", "regularNode", sink, 2, 0))
    for index in (0, 1):  # the frame and its index
        chain.add_edge(
            Edge(
                EdgeSource(NodeId("g0"), OutputId(index)),
                EdgeTarget(NodeId("sink"), InputId(index)),
            )
        )
    return pipeline


@pytest.mark.usefixtures("phases_registry")
@pytest.mark.parametrize("fail_fast", [False, True])
def test_load_images_two_phase_equals_k1_with_out_of_order_finishes(
    monkeypatch, tmp_path, fail_fast
):
    # Review Focus 1: two corrupt images among good ones. At K = 3, item 1's decode waits
    # (at most 10 s) until item 2's has finished, so a later decode finishes first: slot
    # 2 is described and materialized while slot 1's decode waits, which the serial
    # producer cannot do. Files, events and the error equal K = 1's. Deferred, the
    # message lists the items in folder order; fail-fast, the first corrupt item's own
    # error object is raised, not a node error (so execution-error's source is None),
    # after item 1's save and before any later one.
    monkeypatch.delenv(SERIAL_PRODUCER_VARIABLE, raising=False)
    image_files(tmp_path / "in", 7, corrupt={2, 4})
    runs = {}
    for window in ("1", "3"):
        finished, raised, second = [], {}, threading.Event()

        def loader(
            path, window=window, finished=finished, raised=raised, second=second
        ):
            index = int(path.stem)
            if window == "3" and index == 1:
                second.wait(10)
            try:
                return NEW.load.load_image_node(path)
            except Exception as failure:
                raised[index] = failure
                raise
            finally:
                finished.append(index)
                if index == 2:
                    second.set()

        pipeline, directory = load_images_pipeline(
            monkeypatch, tmp_path, window, loader, fail_fast=fail_fast
        )
        error = pipeline.run(workers=6, timeout=30)
        runs[window] = SimpleNamespace(
            pipeline=pipeline,
            error=error,
            raised=raised,
            finished=finished,
            files=written(directory),
        )
    one, three = runs["1"], runs["3"]
    assert three.pipeline.decisions == ["item window K=3 (override)"]
    # Slots 1-6 of seven items (slot 7 is the end). Fail-fast, slots 1-3 in practice:
    # 2 if close() began before slot 3's describe returned; 4 if slot 2's failure
    # reached the loop only after slot 1 committed, as begin(2) then admits slot 4.
    assert three.pipeline.summary["materialized"] in ((2, 3, 4) if fail_fast else (6,))
    assert one.finished == sorted(one.finished)
    assert three.finished.index(2) < three.finished.index(1)  # out of order
    assert three.files == one.files
    assert stripped(three.pipeline.events) == stripped(one.pipeline.events)
    if fail_fast:
        assert list(one.files) == ["0.png", "1.png"]
        assert one.error is one.raised[2] and three.error is three.raised[2]
        assert not isinstance(three.error, process.NodeExecutionError)
    else:
        assert list(one.files) == ["0.png", "1.png", "3.png", "5.png", "6.png"]
        errors = "\n- ".join(str(one.raised[i]) for i in (2, 4))  # in folder order
        assert str(three.error) == str(one.error)
        assert str(one.error) == f"Errors occurred during iteration:\n- {errors}"


@pytest.mark.usefixtures("phases_registry")
def test_split_item_without_an_image_fails_at_its_slot_as_today(monkeypatch, tmp_path):
    # The window takes a materialize that returns None as the end (P2), so Load Images'
    # never does: an item whose load gives no image fails at its slot, as at K = 1, and
    # later items go on. Item 2 fails in its image enforce; item 4 in the unpack of
    # load_image_node's result, which the iterator gives as the item's error.
    monkeypatch.delenv(SERIAL_PRODUCER_VARIABLE, raising=False)
    image_files(tmp_path / "in", 6)

    def loader(path):
        if path.stem == "2":
            return None, path.parent, path.stem
        if path.stem == "4":
            return None
        return NEW.load.load_image_node(path)

    runs = {}
    for window in ("1", "3"):
        pipeline, directory = load_images_pipeline(
            monkeypatch, tmp_path, window, loader
        )
        runs[window] = (pipeline, pipeline.run(workers=6), written(directory))
    (one, error_one, files_one), (three, error_three, files_three) = runs.values()
    assert three.decisions == ["item window K=3 (override)"]
    assert three.summary["materialized"] == 5  # items 1-5; slot 6 is the end
    assert list(files_one) == ["0.png", "1.png", "3.png", "5.png"]
    assert files_three == files_one
    assert type(error_three) is type(error_one)
    assert str(error_three) == str(error_one)
    unpack = "cannot unpack non-iterable NoneType object"
    assert str(error_one) == f"Errors occurred during iteration:\n- \n- {unpack}"
    assert stripped(three.events) == stripped(one.events)


@pytest.mark.usefixtures("phases_registry")
def test_load_image_node_runs_once_per_item_never_before_its_describe(
    monkeypatch, tmp_path
):
    # Spec 5: Load Images' decode (load_image_node) runs once per item, in the
    # materialize of a slot already described (item 0's inside __next__, as today), and
    # describes stay within K - 1 slots of the committing one (SP3 4.3).
    monkeypatch.delenv(SERIAL_PRODUCER_VARIABLE, raising=False)
    image_files(tmp_path / "in", 7)
    log, ahead = [], []
    generator, _ = file_sequence_namespace(current=True)["load_images_node"](
        tmp_path / "in", False, False, "", False, 0, False
    )
    iterator = type(generator.supplier())  # Load Images' native iterator
    describe, advance = iterator.describe, iterator.__next__

    def described(self):
        token = describe(self)
        log.append(("describe", token[1]))
        return token

    def advanced(self):
        log.append(("next", None))
        return advance(self)

    def loader(path):
        log.append(("load", int(path.stem)))
        return NEW.load.load_image_node(path)

    def save(name):
        pipeline.settle()  # the producer describes everything K admits
        ahead.append((int(name), sum(kind == "describe" for kind, _ in log)))

    monkeypatch.setattr(iterator, "describe", described)
    monkeypatch.setattr(iterator, "__next__", advanced)
    pipeline, _ = load_images_pipeline(monkeypatch, tmp_path, "3", loader)
    pipeline.on_save = save
    assert pipeline.run(workers=6) is None
    assert pipeline.decisions == ["item window K=3 (override)"]
    assert pipeline.summary["materialized"] == 6  # items 1-6; slot 7 is the end
    assert Counter(i for kind, i in log if kind == "load") == dict.fromkeys(range(7), 1)
    assert log[:2] == [("next", None), ("load", 0)]
    assert log.count(("next", None)) == 1
    for i in range(1, 7):
        assert log.index(("describe", i)) < log.index(("load", i))
    # at item i's save, slots up to i + K - 1 were described, some beyond slot i
    assert all(i <= count <= i + 2 for i, count in ahead)
    assert max(count - i for i, count in ahead) >= 1


@pytest.mark.usefixtures("phases_registry")
@pytest.mark.parametrize("source", ["load-images", "load-video"])
def test_materialize_never_runs_on_the_loop_thread(monkeypatch, tmp_path, source):
    # Spec 4.5: a materialize (the decode, the normalize) runs on the executor pool, as
    # the producer's advance does, never on the event loop's thread.
    monkeypatch.delenv(SERIAL_PRODUCER_VARIABLE, raising=False)
    decodes = []
    if source == "load-images":
        image_files(tmp_path / "in", 6)

        def loader(path):
            decodes.append(threading.get_ident())
            return NEW.load.load_image_node(path)

        pipeline, _ = load_images_pipeline(monkeypatch, tmp_path, "3", loader)
    else:
        pipeline = load_video_pipeline(monkeypatch, tmp_path, "3", 4)
    loop = set()
    pipeline.on_event = lambda _event: loop.add(threading.get_ident())
    assert pipeline.run(workers=6) is None
    # items 1-5 of six images, frames 1-3 of four
    assert pipeline.summary["materialized"] == (5 if source == "load-images" else 3)
    assert len(loop) == 1
    assert len(pipeline.normalize.threads) == (6 if source == "load-images" else 4)
    assert loop.isdisjoint(pipeline.normalize.threads) and loop.isdisjoint(decodes)


def test_video_normalize_leaves_the_producer_and_outputs_are_unchanged(
    monkeypatch, tmp_path
):
    # Load Video has no describe(): its describe is the FFmpeg read (serial, in frame
    # order) and its materialize the frame's normalize (spec 4.2). At K = 3 each window
    # item's normalize waits until the next frame (or the end) was read: the read goes
    # on while a normalize runs, which the serial producer's advance, holding both,
    # does not allow. The frames and events equal K = 1's.
    monkeypatch.delenv(SERIAL_PRODUCER_VARIABLE, raising=False)
    runs = {}
    for window in ("1", "3"):
        pipeline = load_video_pipeline(monkeypatch, tmp_path, window, 4)
        overlapped = []
        if window == "3":

            def reads(env=pipeline.env):
                return sum(event[0] == "read" for event in list(env.events))

            def next_read(frame, reads=reads, overlapped=overlapped):
                item = int(frame.flat[0]) // 40
                if item:  # item 0 is read and normalized in one advance, as today
                    deadline = time.monotonic() + 2
                    while reads() < item + 2 and time.monotonic() < deadline:
                        time.sleep(0.002)
                    overlapped.append((item, reads() >= item + 2))

            pipeline.normalize.before = next_read
        assert pipeline.run(workers=6) is None
        runs[window] = (pipeline, overlapped)
    (one, _), (three, overlapped) = runs.values()
    assert three.decisions == ["item window K=3 (override)"]
    assert three.summary["materialized"] == 3  # frames 1-3; slot 4 is the end
    assert sorted(overlapped) == [(1, True), (2, True), (3, True)]
    assert three.saved == one.saved and len(one.saved) == 4
    assert stripped(three.events) == stripped(one.events)


@pytest.mark.usefixtures("phases_registry")
@pytest.mark.parametrize("value", ["1", "0", ""])
def test_serial_producer_variable_never_changes_output(monkeypatch, tmp_path, value):
    # Spec 4.6: CHAINNER_C_SERIAL_PRODUCER=1 keeps every group on the serial producer,
    # any other value is ignored, and no value changes a file, an event or an error.
    image_files(tmp_path / "in", 6, corrupt={2})
    monkeypatch.setenv(SERIAL_PRODUCER_VARIABLE, value)
    runs = {}
    for window in ("1", "3"):
        pipeline, directory = load_images_pipeline(
            monkeypatch, tmp_path, window, NEW.load.load_image_node
        )
        pipeline.on_save = pipeline.settle
        runs[window] = (pipeline, pipeline.run(workers=6), written(directory))
    (one, error_one, files_one), (three, error_three, files_three) = runs.values()
    assert three.decisions == ["item window K=3 (override)"]
    assert three.summary["ahead"] >= 1  # the lead let either producer read ahead
    assert (three.summary["materialized"] == 0) == (value == "1")
    assert files_three == files_one and len(files_one) == 5
    assert str(error_three) == str(error_one)
    assert str(error_one).startswith("Errors occurred during iteration:\n- ")
    assert stripped(three.events) == stripped(one.events)


# A fresh interpreter (U1): the system C++ runtime first (conftest), then Pillow and its
# AVIF plugin, as load_image.py imports them, and nothing else. argv: repository root,
# case, folder of a.tga, b.ico and serial.bin. "serial" opens serial.bin (TGA bytes)
# alone: Pillow 12 imports only an extension's own plugin, and .bin has none, so that
# first open runs preinit() then init() and builds the whole registry. "imported" runs
# load_image.py's module-level calls, as its import does, then opens both at once, each
# thread's init() held until both first passes missed, then one init() after the other.
PILLOW_REGISTRY = r"""
import ast
import json
import sys
import threading
from pathlib import Path

root, case, folder = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, str(root / "native/tests"))

import conftest
import pillow_avif
from PIL import Image

result = {}
if case == "imported":
    path = root / "backend/src/packages/chaiNNer_standard/image/io/load_image.py"
    calls = [
        statement
        for statement in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
    ]
    exec(compile(ast.Module(body=calls, type_ignores=[]), str(path), "exec"), {"Image": Image})
    result["registry"] = list(Image.ID)
if case == "serial":
    result["opened"] = {"serial.bin": Image.open(folder / "serial.bin").format}
else:
    init, missed, first = Image.init, threading.Barrier(2, timeout=10), threading.Event()
    inits, opened = [], {}

    def held_init():
        inits.append(threading.current_thread().name)
        if missed.wait() == 0:  # both opens missed in their first pass
            try:
                return init()
            finally:
                first.set()
        first.wait(10)  # then the other init() runs
        return init()

    def open_image(name):
        try:
            opened[name] = Image.open(folder / name).format
        except Exception as error:
            opened[name] = type(error).__name__

    Image.init = held_init  # open() calls init() through the module
    threads = [threading.Thread(target=open_image, args=(n,)) for n in ("a.tga", "b.ico")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    Image.init = init
    result.update(opened=opened, inits=len(inits))
result["ID"] = list(Image.ID)
result["modules"] = [m for m in ("cv2", "torch", "onnxruntime") if m in sys.modules]
print(json.dumps(result))
"""


def test_load_image_import_builds_the_pillow_registry_before_parallel_opens(tmp_path):
    # U1: Pillow builds its plugin registry on first use. Under Pillow 9.2 an open()
    # whose first pass missed failed when another thread's init() finished first: its
    # own init() returned 0, and it made no second pass. load_image.py's import builds
    # the registry before any decode, in the order a first serial open builds it.
    # Pillow 12 builds its registry differently and no longer shows that race from a
    # cold start (P4, 2026-10-06: the "cold" demonstration is retired); the fix's own
    # assertions stay.
    for name in ("a.tga", "b.ico"):
        Image.new("RGB", (16, 16), (10, 20, 30)).save(tmp_path / name)
    Image.new("RGB", (16, 16), (10, 20, 30)).save(tmp_path / "serial.bin", format="TGA")

    def child(case):
        arguments = [PILLOW_REGISTRY, str(ROOT), case, str(tmp_path)]
        done = subprocess.run(
            [sys.executable, "-B", "-c", *arguments],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout.splitlines()[-1])

    imported, serial = (child(case) for case in ("imported", "serial"))
    # After the import's calls both open, and no open() calls init().
    assert imported["opened"] == {"a.tga": "TGA", "b.ico": "ICO"}
    assert imported["inits"] == 0
    assert serial["opened"] == {"serial.bin": "TGA"}
    assert imported["registry"] == imported["ID"] == serial["ID"]
    assert imported["modules"] == serial["modules"] == []


def test_load_image_builds_the_pillow_registry_after_its_imports():
    # U1: load_image.py's only module-level calls are Image.preinit(), then
    # Image.init(), after its last import, so pillow_avif has registered AVIF first, as
    # in a serial first open(). The child test above runs them after its own imports,
    # so it would not see them moved above one.
    path = ROOT / "backend/src/packages/chaiNNer_standard/image/io/load_image.py"
    body = ast.parse(path.read_text(encoding="utf-8")).body
    imports = [
        i
        for i, node in enumerate(body)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    calls = [
        (i, ast.unparse(node))
        for i, node in enumerate(body)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert [call for _, call in calls] == ["Image.preinit()", "Image.init()"]
    assert all(i > max(imports) for i, _ in calls)
