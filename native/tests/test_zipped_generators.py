"""Generators whose per-run outputs other nodes read before the generator's own loop starts (a collector's construction,
a regular node, another generator, a loop that runs earlier): both executors keep the generator for its loop and run
its body once. A group's generators, and a loop's sinks and collectors, run in the chain's order on every hash seed.
A loop that needs another loop's collector runs after that loop, whatever the chain's order and its sinks."""

from __future__ import annotations

import gc
import subprocess
import weakref
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from bounded import run_python
from engine_differential import KINDS, execute
from scratch_storage import NO_FFMPEG, STORAGE
from test_engine_scenarios import Builder, assert_same, items
from verify_runtime import edge, make_node

import api
from api import Collector, NodeId
from chain.chain import Chain
from chain.json import JsonNode, parse_json
from chain.optimize import optimize
from nodes.impl.ffmpeg import get_executable_path
from nodes.impl.item_window import ITEM_WINDOW_VARIABLE, register_phases

ROOT = Path(__file__).resolve().parents[2]
FFMPEG = get_executable_path(STORAGE / "ffmpeg")[0]
VALUES: list[object] = [1, 2, 3]
ORDERS = [("first", "second"), ("second", "first")]
NOT_RUN = "A collector was not run before another node needed it."


def reads_its_own_collector(node: str) -> str:
    """The refusal of a node of the numbers loop that also reads its collector, total."""
    return (
        f"Input0 of {node} comes from the items of numbers, one value per item, but"
        f" {node} also depends on total, which has its result only after the loop of"
        " numbers has ended. Connect a value to Input0 that does not change per item."
    )


@pytest.fixture(autouse=True)
def test_package(monkeypatch: pytest.MonkeyPatch):
    # Test nodes belong to no package: both executors read their settings from a stand-in.
    monkeypatch.setattr(
        api.registry, "get_package", lambda _: SimpleNamespace(id="test")
    )


def generator(b: Builder, trace: list[object], name: str, inputs: int = 0) -> None:
    def body(*read: object):
        trace.append(("body", name, *read))
        return items(VALUES), f"{name} per run"

    b.node(name, body, kind="generator", inputs=inputs, outputs=2)


def collector(b: Builder, trace: list[object], name: str, inputs: int = 1) -> None:
    def gather(_: object, *read: object):
        trace.append(("gather", name, *read))
        seen: list[object] = []
        return Collector(seen.append, lambda: trace.append((name, seen)))

    b.node(name, gather, kind="collector", inputs=inputs, outputs=0)


def zipped(
    trace: list[object],
    order: tuple[str, str],
    readers: tuple[str, ...],
    inputs: dict[str, int] | None = None,
) -> Builder:
    """Two generators zipped into a collector whose per-run inputs read the named generators."""
    b = Builder()
    for name in order:
        generator(b, trace, name, (inputs or {}).get(name, 0))
    b.node("pair", lambda x, y: (x, y), inputs=2)
    b.edge("first", 0, "pair", 0)
    b.edge("second", 0, "pair", 1)
    collector(b, trace, "total", 1 + len(readers))
    b.edge("pair", 0, "total", 0)
    for index, reader in enumerate(readers):
        b.edge(reader, 1, "total", 1 + index)
    return b


def reads_first(order: tuple[str, str]) -> Callable[[list[object]], Chain]:
    return lambda trace: zipped(trace, order, ("first",)).chain


def reads_second(order: tuple[str, str]) -> Callable[[list[object]], Chain]:
    return lambda trace: zipped(trace, order, ("second",)).chain


def reads_both(order: tuple[str, str]) -> Callable[[list[object]], Chain]:
    return lambda trace: zipped(trace, order, ("first", "second")).chain


def through_a_regular_node(order: tuple[str, str]) -> Callable[[list[object]], Chain]:
    def scenario(trace: list[object]) -> Chain:
        b = zipped(trace, order, ())
        b.node("label", lambda text: f"{text}!")
        b.edge("second", 1, "label", 0)
        collector(b, trace, "labelled", 2)
        b.edge("pair", 0, "labelled", 0)
        b.edge("label", 0, "labelled", 1)
        return b.chain

    return scenario


def generator_reads_generator(
    order: tuple[str, str],
) -> Callable[[list[object]], Chain]:
    def scenario(trace: list[object]) -> Chain:
        b = zipped(trace, order, (), {"second": 1})
        b.edge("first", 1, "second", 0)
        return b.chain

    return scenario


def read_again_after_the_loop(
    order: tuple[str, str],
) -> Callable[[list[object]], Chain]:
    def scenario(trace: list[object]) -> Chain:
        b = zipped(trace, order, ("second",))
        b.node(
            "after",
            lambda text: trace.append(("after", text)),
            outputs=0,
            side_effects=True,
        )
        b.edge("second", 1, "after", 0)
        return b.chain

    return scenario


def across_loops(order: tuple[str, str]) -> Callable[[list[object]], Chain]:
    """Two loops: each collector reads the other loop's generator, so the later loop's generator is read first."""

    def scenario(trace: list[object]) -> Chain:
        b = Builder()
        for name in order:
            generator(b, trace, name)
        collector(b, trace, "first total", 2)
        b.edge("first", 0, "first total", 0)
        b.edge("second", 1, "first total", 1)
        collector(b, trace, "second total", 2)
        b.edge("second", 0, "second total", 0)
        b.edge("first", 1, "second total", 1)
        return b.chain

    return scenario


@pytest.mark.parametrize("order", ORDERS, ids="-".join)
@pytest.mark.parametrize(
    "scenario",
    [
        reads_first,
        reads_second,
        reads_both,
        through_a_regular_node,
        generator_reads_generator,
        read_again_after_the_loop,
        across_loops,
    ],
)
def test_a_read_generator_still_runs_its_loop(
    scenario: Callable[[tuple[str, str]], Callable[[list[object]], Chain]],
    order: tuple[str, str],
    tmp_path: Path,
):
    build = scenario(order)
    for kind in KINDS:
        trace: list[Any] = []
        result = execute(kind, build(trace), storage=tmp_path / "storage")
        assert result.error is None, (kind, result.error)
        bodies = [entry[1] for entry in trace if entry[0] == "body"]
        assert sorted(bodies) == ["first", "second"], kind
    build.__name__ = scenario.__name__
    assert_same(build, tmp_path)


@pytest.mark.parametrize("order", ORDERS, ids="-".join)
@pytest.mark.parametrize("kind", KINDS)
def test_a_groups_generators_run_in_the_chains_order(
    kind: str, order: tuple[str, str], tmp_path: Path
):
    trace: list[Any] = []
    result = execute(kind, zipped(trace, order, ()).chain, storage=tmp_path)
    assert result.error is None
    assert [entry[1] for entry in trace if entry[0] == "body"] == list(order)
    starts = [
        event["data"]["nodeId"]
        for event in result.events
        if event["event"] == "node-start"
    ]
    assert [node for node in starts if node in order] == list(order)


@pytest.mark.parametrize("kind", KINDS)
def test_a_loops_sinks_and_collectors_run_in_the_chains_order(
    kind: str, tmp_path: Path
):
    trace: list[Any] = []
    b = Builder()
    b.node("numbers", lambda: items(VALUES), kind="generator", inputs=0)
    sinks = [f"sink {index}" for index in range(8)]
    collectors = [f"collector {index}" for index in range(8)]
    for name in sinks:
        b.node(
            name,
            lambda value, name=name: trace.append((name, value)),
            outputs=0,
            side_effects=True,
        )
        b.edge("numbers", 0, name, 0)
    for name in collectors:
        collector(b, trace, name)
        b.edge("numbers", 0, name, 0)
    result = execute(kind, b.chain, storage=tmp_path)
    assert result.error is None
    assert [entry[1] for entry in trace if entry[0] == "gather"] == collectors
    assert [entry[0] for entry in trace if entry[0] in sinks] == sinks * len(VALUES)


def test_the_two_executors_agree_on_a_chain_with_many_sinks(tmp_path: Path):
    def scenario(trace: list[object]) -> Chain:
        b = Builder()
        b.node("numbers", lambda: items(VALUES), kind="generator", inputs=0)
        for index in range(8):
            name = f"sink {index}"
            b.node(
                name,
                lambda value, name=name: trace.append((name, value)),
                outputs=0,
                side_effects=True,
            )
            b.edge("numbers", 0, name, 0)
        return b.chain

    assert_same(scenario, tmp_path)


def counted_loop(
    b: Builder,
    trace: list[object],
    name: str,
    *,
    sinks: int = 1,
    read: str | None = None,
) -> None:
    """A generator with a collector that sums its items and sinks - 1 more sinks; it takes its items from `read`."""

    def body(*values: object):
        trace.append(("body", name, *values))
        return items([*values] or VALUES), f"{name} per run"

    b.node(name, body, kind="generator", inputs=0 if read is None else 1, outputs=2)
    if read is not None:
        b.edge(read, 0, name, 0)

    def gather(_: object):
        seen: list[Any] = []
        return Collector(seen.append, lambda: sum(seen))

    b.node(f"{name} total", gather, kind="collector")
    b.edge(name, 0, f"{name} total", 0)
    for index in range(sinks - 1):
        sink = f"{name} sink {index}"
        b.node(
            sink,
            lambda value, sink=sink: trace.append((sink, value)),
            outputs=0,
            side_effects=True,
        )
        b.edge(name, 0, sink, 0)


def fed_by_a_collector(sinks: int) -> Callable[[list[object]], Chain]:
    def scenario(trace: list[object]) -> Chain:
        b = Builder()
        counted_loop(b, trace, "upstream", sinks=sinks)
        counted_loop(b, trace, "downstream", read="upstream total")
        return b.chain

    return scenario


def two_sink_loop_elsewhere(trace: list[object]) -> Chain:
    b = Builder()
    counted_loop(b, trace, "upstream")
    counted_loop(b, trace, "downstream", read="upstream total")
    counted_loop(b, trace, "unrelated", sinks=2)
    return b.chain


def through_a_regular_reader(trace: list[object]) -> Chain:
    b = Builder()
    counted_loop(b, trace, "upstream", sinks=2)
    b.node("same", lambda value: value)
    b.edge("upstream total", 0, "same", 0)
    counted_loop(b, trace, "downstream", read="same")
    return b.chain


def through_a_per_run_output(trace: list[object]) -> Chain:
    """The middle loop's generator needs the collector; the last one reads that generator's per-run output."""
    b = Builder()
    counted_loop(b, trace, "upstream", sinks=2)
    counted_loop(b, trace, "middle", read="upstream total")

    def body(text: object):
        trace.append(("body", "downstream", text))
        return items(VALUES), "last"

    b.node("downstream", body, kind="generator", outputs=2)
    b.edge("middle", 1, "downstream", 0)
    b.node("after", lambda v: trace.append(("after", v)), outputs=0, side_effects=True)
    b.edge("downstream", 0, "after", 0)
    return b.chain


def read_by_an_iterated_node(trace: list[object]) -> Chain:
    b = Builder()
    counted_loop(b, trace, "upstream", sinks=2)
    b.node("numbers", lambda: items(VALUES), kind="generator", inputs=0)
    b.node("add", lambda x, y: x + y, inputs=2)
    b.edge("numbers", 0, "add", 0)
    b.edge("upstream total", 0, "add", 1)
    b.node("after", lambda v: trace.append(("after", v)), outputs=0, side_effects=True)
    b.edge("add", 0, "after", 0)
    return b.chain


def read_by_a_collectors_construction(trace: list[object]) -> Chain:
    b = Builder()
    counted_loop(b, trace, "upstream", sinks=2)
    b.node("numbers", lambda: items(VALUES), kind="generator", inputs=0)
    collector(b, trace, "reader", 2)
    b.edge("numbers", 0, "reader", 0)
    b.edge("upstream total", 0, "reader", 1)
    return b.chain


def three_loops(trace: list[object]) -> Chain:
    b = Builder()
    counted_loop(b, trace, "upstream", sinks=2)
    counted_loop(b, trace, "middle", sinks=3, read="upstream total")
    counted_loop(b, trace, "downstream", sinks=2, read="middle total")
    return b.chain


@pytest.mark.parametrize("reverse", [False, True], ids=["chain order", "reversed"])
@pytest.mark.parametrize(
    "scenario",
    [
        fed_by_a_collector(1),
        fed_by_a_collector(2),
        fed_by_a_collector(3),
        two_sink_loop_elsewhere,
        through_a_regular_reader,
        through_a_per_run_output,
        read_by_an_iterated_node,
        read_by_a_collectors_construction,
        three_loops,
    ],
)
def test_a_loop_runs_after_the_collector_it_needs(
    scenario: Callable[[list[object]], Chain], reverse: bool, tmp_path: Path
):
    def ordered(trace: list[object]) -> Chain:
        chain = scenario(trace)
        if reverse:
            chain.nodes = dict(reversed(chain.nodes.items()))
        return chain

    for kind in KINDS:
        result = execute(kind, ordered([]), storage=tmp_path / "storage")
        assert result.error is None, (kind, result.error)
    assert_same(ordered, tmp_path)


def test_a_collector_gives_the_later_loop_its_result(tmp_path: Path):
    for kind in KINDS:
        trace: list[Any] = []
        chain = fed_by_a_collector(2)(trace)
        assert execute(kind, chain, storage=tmp_path).error is None, kind
        assert ("body", "downstream", sum(cast("list[int]", VALUES))) in trace, kind
        assert [e[1] for e in trace if e[0] == "body"] == ["upstream", "downstream"]


@pytest.mark.parametrize(
    ("sinks", "expected"),
    [((1, 1, 1, 1), "dcba"), ((1, 2, 1, 1), "abcd")],
)
@pytest.mark.parametrize("kind", KINDS)
def test_loops_that_need_nothing_keep_their_order(
    kind: str, sinks: tuple[int, ...], expected: str, tmp_path: Path
):
    trace: list[Any] = []
    b = Builder()
    for name, count in zip("abcd", sinks, strict=True):
        counted_loop(b, trace, name, sinks=count)
    assert execute(kind, b.chain, storage=tmp_path).error is None
    assert [e[1] for e in trace if e[0] == "body"] == list(expected)


def test_loops_that_need_each_other_report_the_collector(tmp_path: Path):
    def scenario(trace: list[object]) -> Chain:
        b = Builder()
        for name, other in (("a", "b"), ("b", "a")):
            b.node(name, lambda: items(VALUES), kind="generator", inputs=0)
            b.node(
                f"{name} total",
                lambda _: Collector(lambda _: None, lambda: 0),
                kind="collector",
            )
            b.edge(name, 0, f"{name} total", 0)
            b.node(f"{name} add", lambda x, y: x + y, inputs=2)
            b.edge(name, 0, f"{name} add", 0)
            b.edge(f"{other} total", 0, f"{name} add", 1)
            b.node(f"{name} after", trace.append, outputs=0, side_effects=True)
            b.edge(f"{name} add", 0, f"{name} after", 0)
        return b.chain

    # No run reaches the second loop's generator, so the event comparison does not apply.
    assert {
        str(execute(kind, scenario([]), storage=tmp_path).error) for kind in KINDS
    } == {"A collector was not run before another node needed it."}


def unread_collector(_: object, *_read: object):
    return Collector(lambda _: None, lambda: 7)


def phased_sink(b: Builder, name: str, item: str, reads: str) -> None:
    """A sink with a prepare phase over an item and a collector's output."""
    register_phases(
        f"chainner:utility:test_{name}",
        lambda _inputs: 1,
        lambda _enforced, _value: None,
    )
    b.node(name, lambda x, c: None, inputs=2, outputs=0, side_effects=True)
    b.edge(item, 0, name, 0)
    b.edge(reads, 0, name, 1)


def deferred_errors(kind: str, chain: Chain, tmp_path: Path) -> list[str]:
    """The run's error, one line per item: an item's error does not end the loop."""
    error = str(execute(kind, chain, storage=tmp_path).error)
    first, *lines = error.split("\n")
    assert first == "Errors occurred during iteration:", error
    return lines


@pytest.mark.parametrize("order", ORDERS, ids="-".join)
def test_a_phased_sink_reading_a_collector_that_never_ran(
    order: tuple[str, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    def chain() -> Chain:
        b = Builder()
        for name, other in (order, order[::-1]):
            b.node(
                name,
                lambda: items(list(range(8)), fail_fast=False),
                kind="generator",
                inputs=0,
            )
            b.node(f"{name} total", unread_collector, kind="collector")
            b.edge(name, 0, f"{name} total", 0)
            phased_sink(b, f"{name} save", name, f"{other} total")
        return b.chain

    monkeypatch.setenv(ITEM_WINDOW_VARIABLE, "4")
    errors = {kind: deferred_errors(kind, chain(), tmp_path) for kind in KINDS}
    assert errors["native"] == errors["python"] == [f"- {NOT_RUN}"] * 8


class Probe:
    """A value whose life a weak reference shows."""


@pytest.mark.parametrize("reads", [False, True], ids=["no read", "a read"])
def test_the_engine_releases_a_generator_with_its_loop(reads: bool, tmp_path: Path):
    seen: list[weakref.ref[Probe]] = []
    alive: list[bool] = []

    def body():
        probes: list[object] = [Probe(), Probe()]
        seen.append(weakref.ref(cast("Probe", probes[0])))
        return items(probes), "per run"

    def later(_: object):
        gc.collect()
        alive.append(seen[0]() is not None)
        return items(VALUES)

    b = Builder()
    b.node("first", body, kind="generator", inputs=0, outputs=2)
    b.node("total", unread_collector, kind="collector", inputs=2 if reads else 1)
    b.edge("first", 0, "total", 0)
    if reads:
        b.edge("first", 1, "total", 1)
    b.node("later", later, kind="generator")
    b.edge("total", 0, "later", 0)
    b.node("sink", lambda _: None, outputs=0, side_effects=True)
    b.edge("later", 0, "sink", 0)
    assert execute("native", b.chain, storage=tmp_path).error is None
    assert alive == [False]


@pytest.mark.parametrize("kind", KINDS)
def test_a_generator_without_a_loop_is_released_after_its_last_read(
    kind: str, tmp_path: Path
):
    seen: list[weakref.ref[Probe]] = []
    alive: list[bool] = []

    def body():
        probes: list[object] = [Probe()]
        seen.append(weakref.ref(cast("Probe", probes[0])))
        return items(probes), "per run"

    def check(_: object):
        gc.collect()
        alive.append(seen[0]() is not None)

    b = Builder()
    b.node("first", body, kind="generator", inputs=0, outputs=2)
    b.node("reader", lambda text: text, side_effects=True)
    b.edge("first", 1, "reader", 0)
    b.node("after", check, outputs=0, side_effects=True)
    b.edge("reader", 0, "after", 0)
    assert execute(kind, b.chain, storage=tmp_path).error is None
    assert alive == [False]


def test_a_loop_reading_its_own_collector_reports_it(tmp_path: Path):
    def chain() -> Chain:
        b = Builder()
        b.node("numbers", lambda: items(VALUES), kind="generator", inputs=0)
        b.node("value", lambda x: x + 1)
        b.value("value", 0, 41)
        b.node("total", unread_collector, kind="collector", inputs=2)
        b.edge("numbers", 0, "total", 0)
        b.edge("value", 0, "total", 1)
        b.node("sink", lambda x, y: None, inputs=2, outputs=0, side_effects=True)
        b.edge("numbers", 0, "sink", 0)
        b.edge("total", 0, "sink", 1)
        return b.chain

    # Refused before anything runs, on both engines, on the node that reads the item.
    results = [execute(kind, chain(), storage=tmp_path) for kind in KINDS]
    assert [getattr(r.error, "node_id", None) for r in results] == ["sink"] * len(KINDS)
    assert {str(r.error) for r in results} == {reads_its_own_collector("sink")}


def test_a_phased_sink_reading_its_own_collector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    def chain() -> Chain:
        b = Builder()
        b.node(
            "numbers",
            lambda: items(list(range(8)), fail_fast=False),
            kind="generator",
            inputs=0,
        )
        b.node("total", unread_collector, kind="collector")
        b.edge("numbers", 0, "total", 0)
        phased_sink(b, "own save", "numbers", "total")
        return b.chain

    monkeypatch.setenv(ITEM_WINDOW_VARIABLE, "4")
    # Refused before any item, its prepare phase included, rather than failing each item.
    results = [execute(kind, chain(), storage=tmp_path) for kind in KINDS]
    assert [getattr(r.error, "node_id", None) for r in results] == ["own save"] * len(
        KINDS
    )
    assert {str(r.error) for r in results} == {reads_its_own_collector("own save")}


@pytest.mark.parametrize("loaded_first", [True, False])
@pytest.mark.parametrize("reads_numbers", [False, True])
@pytest.mark.parametrize("kind", KINDS)
def test_load_video_zipped_with_a_generator(
    schemas, tmp_path: Path, kind: str, reads_numbers: bool, loaded_first: bool
):
    if not FFMPEG.exists():
        pytest.skip(NO_FFMPEG)
    clip = tmp_path / "clip.mkv"
    subprocess.run(
        [
            str(FFMPEG),
            *("-nostdin", "-v", "error", "-f", "lavfi"),
            *("-i", "testsrc2=size=64x48:rate=30:duration=1"),
            *("-c:v", "ffv1", str(clip)),
        ],
        check=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
    )

    def numbers():
        generator = items(list(range(100)))
        generator.length_is_exact = False
        return generator, "from numbers"

    # Save Video reads Load Video's FPS, audio and colour, and maybe its name from the other generator.
    save = make_node(
        schemas,
        "save",
        "chainner:image:save_video",
        {
            0: None,
            1: str(tmp_path),
            2: None if reads_numbers else "fixed",
            16: 1,
            4: "mkv",
            3: "ffv1",
            14: edge("load", 4),
            15: edge("load", 5),
            23: edge("load", 9),
        },
    )
    load = make_node(schemas, "load", "chainner:image:load_video", {0: str(clip)})
    b = Builder()
    b.chain = parse_json(cast("list[JsonNode]", [load, save]))
    b.node("numbers", numbers, kind="generator", inputs=0, outputs=2)
    b.node("mix", lambda frame, _: frame, inputs=2)
    b.edge("load", 0, "mix", 0)
    b.edge("numbers", 0, "mix", 1)
    b.edge("mix", 0, "save", 0)
    if reads_numbers:
        b.edge("numbers", 1, "save", 2)
    if not loaded_first:
        b.chain.nodes = {
            NodeId("numbers"): b.chain.nodes[NodeId("numbers")],
            **b.chain.nodes,
        }
    optimize(b.chain)
    result = execute(kind, b.chain, storage=STORAGE)
    assert result.error is None
    assert (tmp_path / ("from numbers.mkv" if reads_numbers else "fixed.mkv")).exists()


@pytest.mark.parametrize("reverse", [False, True], ids=["chain order", "reversed"])
@pytest.mark.parametrize("kind", KINDS)
def test_range_reads_the_accumulate_of_a_loop_with_two_sinks(
    schemas, tmp_path: Path, kind: str, reverse: bool
):
    def saved(prefix: str, source: str) -> list[dict[str, Any]]:
        return [
            make_node(
                schemas,
                f"{prefix} name",
                "chainner:utility:text_pattern",
                {0: f"{prefix} {{1}}", 1: edge(source)},
            ),
            make_node(
                schemas,
                f"{prefix} save",
                "chainner:image:save",
                {
                    0: edge("color"),
                    1: str(tmp_path),
                    3: edge(f"{prefix} name"),
                    4: "png",
                },
            ),
        ]

    nodes = [
        make_node(schemas, "color", "chainner:image:create_color", {1: 8, 2: 8}),
        make_node(schemas, "first", "chainner:utility:range", {2: 4}),
        make_node(
            schemas, "sum", "chainner:utility:accumulate", {0: edge("first"), 1: "sum"}
        ),
        *saved("first", "first"),
        make_node(schemas, "second", "chainner:utility:range", {2: edge("sum")}),
        *saved("second", "second"),
    ]
    chain = parse_json(cast("list[JsonNode]", nodes))
    optimize(chain)
    if reverse:
        chain.nodes = dict(reversed(chain.nodes.items()))
    result = execute(kind, chain, storage=STORAGE)
    assert result.error is None
    assert sorted(path.name for path in tmp_path.glob("*.png")) == sorted(
        [*(f"first {i}.png" for i in range(4)), *(f"second {i}.png" for i in range(6))]
    )


@pytest.mark.parametrize("seed", range(1, 7))
def test_on_every_hash_seed(seed: int):
    finished = run_python(
        [
            *("-m", "pytest", "-q", "-p", "no:cacheprovider", str(Path(__file__))),
            *("-k", "not every_hash_seed and not load_video and not accumulate"),
        ],
        ROOT,
        300,
        env={"PYTHONHASHSEED": str(seed)},
    )
    assert finished.returncode == 0, finished.output[-4000:]
