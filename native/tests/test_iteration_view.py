"""Iteration view: a node's item index, lineage source, per-run state and look-ahead
(peek) inside a generator loop, over the item window (process.py, item_window.py)."""

import asyncio
import threading
import time
import weakref

import numpy as np
import pytest
from test_generator_lifecycle import drive, make_node, spin, stripped
from test_item_window import (
    HeldPool,
    ManualPool,
    PhasedSource,
    Pipeline,
    Planner,
    close_finishing,
    finish_in_turn,
    load_video_pipeline,
    new_window,
    set_schema,
    settle,
    take,
)

import process
from api import Aborted, BaseInput, InputId, NodeId
from chain.chain import Edge, EdgeTarget
from nodes.condition import Condition
from nodes.impl.execution_scheduler import LOCKED_DEVICE_NODES
from nodes.impl.item_window import SERIAL_PRODUCER_VARIABLE, ItemWindow
from nodes.properties.inputs import DirectoryInput, ImageInput
from progress_controller import ProgressController

LOOK = "test:iteration-view:look"
G0 = "test:generator-lifecycle:g0"


class Tally:
    """look's state: the items it ran in this loop run."""

    def __init__(self):
        self.items = []


class Looking(Pipeline):
    """Pipeline's g0 -> nodes -> save (or collect), with look before the sink.

    look has a context and runs on the commit path. At each item it records its view:
    index, source, the items its state (a Tally) holds, and inspect(view), by default
    lookahead(0) and peek(0, offset) for offsets 1..depth; then it adds the item to its
    state and passes its input on. `outside` records g0's context.iteration.
    """

    def __init__(self, monkeypatch, tmp_path, count, *, window, depth=3, **options):
        super().__init__(monkeypatch, tmp_path, count, window=window, **options)
        self.depth = depth
        self.looks = []
        self.tallies = []
        self.outside = []
        self.contexts = {}
        sink = "collect" if options.get("collector") else "save"
        edge = self.chain.edge_to(NodeId(sink), InputId(0))
        assert edge is not None
        self.chain.remove_edge(edge)
        node = make_node("look", "regularNode", self.look, 2, 1)
        object.__setattr__(node.data, "side_effects", False)
        object.__setattr__(node.data, "node_context", True)
        set_schema(node, LOOK)
        self.chain.add_node(node)
        self.chain.add_edge(Edge(edge.source, EdgeTarget(NodeId("look"), InputId(0))))
        self.chain.inputs.set(NodeId("look"), InputId(1), "constant")
        self.connect("look", sink, 0)
        g0 = self.chain.nodes[NodeId("g0")]
        factory = g0.data.run

        def generator(context):
            self.outside.append(context.iteration)
            return factory(context)

        object.__setattr__(g0.data, "run", generator)

    def inspect(self, view):
        if not self.depth:
            return {}
        return {
            "lookahead": view.lookahead(0),
            "peeks": [view.peek(0, offset) for offset in range(1, self.depth + 1)],
        }

    def look(self, context, value, _constant):
        self.contexts[context.node_id] = context
        view = context.iteration
        if view.state is None:
            view.state = Tally()
            self.tallies.append(weakref.ref(view.state))
        record = {
            "view": view,
            "index": view.index,
            "source": view.source,
            "state": list(view.state.items),
            "value": value,
        }
        record.update(self.inspect(view))
        view.state.items.append(view.index)
        self.looks.append(record)
        return value

    def context(self, node_id):
        return self.contexts[NodeId(node_id)]


def assert_served(looks, lookahead, *, count=None, missing=()):
    """Each recorded peek is the object look received at index + offset, for offsets
    up to lookahead and items that reached look; None otherwise (past the end, at
    `count`, or an item in `missing`)."""
    values = {record["index"]: record["value"] for record in looks}
    end = len(values) + len(missing) if count is None else count
    for record in looks:
        assert record["lookahead"] == lookahead
        for offset, peeked in enumerate(record["peeks"], 1):
            later = record["index"] + offset
            if offset <= lookahead and later < end and later not in missing:
                assert peeked is values[later], (record["index"], offset)
            else:
                assert peeked is None, (record["index"], offset)


@pytest.mark.parametrize("producer", ["phased", "serial"])
@pytest.mark.parametrize("nodes", [(), ("map",)])
@pytest.mark.parametrize("window", ["2", "4"])
def test_peek_returns_the_values_later_items_deliver(
    monkeypatch, tmp_path, producer, nodes, window
):
    # From the generator, or from map computed ahead; up to K - 1 items ahead, item 0
    # included (the look-ahead decides K there). Outputs and events equal K = 1's,
    # where nothing is served.
    if producer == "serial":
        monkeypatch.setenv(SERIAL_PRODUCER_VARIABLE, "1")
    else:
        monkeypatch.delenv(SERIAL_PRODUCER_VARIABLE, raising=False)
    runs = {}
    for k in ("1", window):
        looking = Looking(monkeypatch, tmp_path, 6, window=k, nodes=nodes, depth=4)
        assert looking.run(workers=4) is None
        runs[k] = looking
    one, many = runs["1"], runs[window]
    assert [record["index"] for record in many.looks] == list(range(6))
    assert {record["source"] for record in many.looks} == {G0}
    assert_served(many.looks, int(window) - 1)
    assert_served(one.looks, 0)
    expected = (
        [i * 10 + 7.0 for i in range(6)] if nodes else [float(i) for i in range(6)]
    )
    assert many.saved == one.saved == expected
    assert stripped(many.events) == stripped(one.events)
    assert many.decisions == [f"item window K={window} (override)"]
    assert many.outside == one.outside == [None]  # a generator sees no iteration


def test_window_decided_at_item_1_when_item_0_does_not_look_ahead(
    monkeypatch, tmp_path
):
    looking = Looking(monkeypatch, tmp_path, 5, window="3")
    standard = looking.inspect
    looking.inspect = lambda view: {} if view.index == 0 else standard(view)
    assert looking.run(workers=4) is None
    assert looking.decisions == ["item window K=3 (override)"]
    assert_served(looking.looks[1:], 2, count=5)


def computed_folder(looking, tmp_path, sources):
    """save's directory comes from `folder`, which save's gather runs after look; g0
    declares `sources` as the files it reads per item."""
    save = looking.chain.nodes[NodeId("save")]
    inputs = [BaseInput("any", "Input0").with_id(0), DirectoryInput(must_exist=False)]
    object.__setattr__(save.data, "inputs", [inputs[0], inputs[1].with_id(1)])
    object.__setattr__(save.data, "run", lambda value, _folder: looking.deliver(value))
    folder = make_node("folder", "regularNode", lambda: tmp_path / "out", 0, 1)
    object.__setattr__(folder.data, "side_effects", False)
    looking.chain.add_node(folder)
    looking.connect("folder", "save", 1)
    g0 = looking.chain.nodes[NodeId("g0")]
    factory = g0.data.run

    def declaring(context):
        generator = factory(context)
        generator.source_paths = sources
        return generator

    object.__setattr__(g0.data, "run", declaring)


@pytest.mark.parametrize("first", [0, 1])
def test_a_look_ahead_at_item_0_sees_a_directory_computed_later_as_unresolved(
    monkeypatch, tmp_path, first
):
    # g0 reads a file per item. A look-ahead during item 0 decides K before `folder`
    # ran: unresolved, K = 1 for the run. Looking ahead from item 1 on, K is decided at
    # item 1 as without a look-ahead.
    looking = Looking(monkeypatch, tmp_path, 4, window="3")
    computed_folder(looking, tmp_path, (tmp_path / "in" / "0.png",))
    standard = looking.inspect
    looking.inspect = lambda view: {} if view.index < first else standard(view)
    assert looking.run(workers=4) is None
    if first == 0:
        assert looking.decisions == [
            "item window K=1 (output directory unresolved: save)"
        ]
        assert_served(looking.looks, 0)
    else:
        assert looking.decisions == ["item window K=3 (override)"]
        assert_served(looking.looks[1:], 2, count=4)
    assert looking.saved == [7.0, 17.0, 27.0, 37.0]


def test_a_failed_early_decision_is_made_again_at_item_1(monkeypatch, tmp_path):
    # The look-ahead at item 0 decides K early and that decision raises (the path
    # check): look fails at item 0, a deferred error, and the loop decides K again at
    # item 1 from the measurement it kept, instead of failing on a bare assertion.
    calls = []
    blocker = process.read_ahead_blocker

    def failing(sources, directories):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("resolve failed")
        return blocker(sources, directories)

    monkeypatch.setattr(process, "read_ahead_blocker", failing)
    looking = Looking(monkeypatch, tmp_path, 4, window="3", fail_fast=False)
    error = looking.run(workers=4)
    assert str(error).startswith("Errors occurred during iteration:"), error
    assert "resolve failed" in str(error)
    assert len(calls) == 2
    assert looking.decisions == ["item window K=3 (override)"]
    assert [record["index"] for record in looking.looks] == [1, 2, 3]
    assert_served(looking.looks, 2, count=4)
    assert looking.saved == [17.0, 27.0, 37.0]


@pytest.mark.parametrize("producer", ["phased", "serial"])
def test_a_directory_computed_later_never_blocks_a_group_reading_no_file_per_item(
    monkeypatch, tmp_path, producer
):
    # g0 reads no file per item, as Load Video: the unresolved folder does not block the
    # window, the look-ahead from item 0 is served, and save receives what it receives
    # at K = 1.
    if producer == "serial":
        monkeypatch.setenv(SERIAL_PRODUCER_VARIABLE, "1")
    else:
        monkeypatch.delenv(SERIAL_PRODUCER_VARIABLE, raising=False)
    runs = {}
    for window in ("1", "3"):
        looking = Looking(monkeypatch, tmp_path, 5, window=window)
        computed_folder(looking, tmp_path, ())
        assert looking.run(workers=4) is None
        runs[window] = looking
    one, three = runs["1"], runs["3"]
    assert three.decisions == ["item window K=3 (override)"]
    assert_served(three.looks, 2)
    assert three.saved == one.saved == [7.0, 17.0, 27.0, 37.0, 47.0]
    assert stripped(three.events) == stripped(one.events)


@pytest.mark.parametrize(("workers", "window"), [(1, "3"), (4, "1")])
def test_no_look_ahead_without_a_second_pool_thread_or_a_window(
    monkeypatch, tmp_path, workers, window
):
    # The waiting node holds a pool thread: with one, nothing is served (and nothing
    # hangs); K = 1 serves nothing either. Index, source and state still work.
    looking = Looking(monkeypatch, tmp_path, 4, window=window)
    assert looking.run(workers=workers) is None
    assert_served(looking.looks, 0)
    assert [record["state"] for record in looking.looks] == [[], [0], [0, 1], [0, 1, 2]]
    assert looking.saved == [7.0, 17.0, 27.0, 37.0]


def test_state_lives_for_one_loop_run_per_node(monkeypatch, tmp_path):
    runs = []
    for _ in range(2):
        looking = Looking(monkeypatch, tmp_path, 4, window="3", depth=0)
        assert looking.run(workers=4) is None
        runs.append(looking)
    for looking in runs:
        views = {record["view"] for record in looking.looks}
        assert len(views) == 1  # one view for the loop run
        assert [record["state"] for record in looking.looks] == [
            [],
            [0],
            [0, 1],
            [0, 1, 2],
        ]
        [view] = views
        assert view.state is None  # dropped at the loop's end
        assert looking.context("look").iteration is None
        assert [tally() for tally in looking.tallies] == [None]
    # never shared between runs
    assert runs[0].looks[0]["view"] is not runs[1].looks[0]["view"]


def test_state_is_per_node(monkeypatch, tmp_path):
    # look -> look2 -> save: two nodes of one loop, two views, two states.
    looking = Looking(monkeypatch, tmp_path, 3, window="3", depth=0)
    edge = looking.chain.edge_to(NodeId("save"), InputId(0))
    assert edge is not None
    looking.chain.remove_edge(edge)
    second = make_node("look2", "regularNode", looking.look, 2, 1)
    object.__setattr__(second.data, "side_effects", False)
    object.__setattr__(second.data, "node_context", True)
    set_schema(second, LOOK)
    looking.chain.add_node(second)
    looking.connect("look", "look2", 0)
    looking.chain.inputs.set(NodeId("look2"), InputId(1), "constant")
    looking.connect("look2", "save", 0)
    assert looking.run(workers=4) is None
    first = [r for r in looking.looks if r["view"] is looking.looks[0]["view"]]
    other = [r for r in looking.looks if r["view"] is not looking.looks[0]["view"]]
    assert [r["state"] for r in first] == [r["state"] for r in other]
    assert [r["state"] for r in first] == [[], [0], [0, 1]]
    assert len(looking.tallies) == 2 and all(t() is None for t in looking.tallies)


@pytest.mark.parametrize("mode", ["fail", "cancel", "stop"])
def test_state_is_dropped_when_the_loop_fails_is_cancelled_or_stopped(
    monkeypatch, tmp_path, mode
):
    looking = Looking(monkeypatch, tmp_path, 6, window="3", depth=1)
    if mode == "fail":

        def fail(index):
            if index == 2:
                raise ValueError("save failed 2")

        looking.on_save = fail
    else:

        def on_event(event):
            if (
                event["event"] == "node-start"
                and event["data"]["nodeId"] == "save"
                and looking.saves == 2
            ):
                if mode == "cancel":
                    looking.task.cancel()
                else:
                    looking.executor.kill()

        looking.on_event = on_event
    error = looking.run(workers=4)
    assert error is not None
    if mode == "cancel":
        assert isinstance(error, asyncio.CancelledError)
    assert [record["index"] for record in looking.looks][:3] == [0, 1, 2]
    view = looking.looks[0]["view"]
    assert view.state is None and looking.context("look").iteration is None
    assert [tally() for tally in looking.tallies] == [None]
    # the window closed after the loop: no map or read ran at the run's cleanup (the
    # records keep the outputs look received and peeked)
    assert [state[:2] for state in looking.cleanup_states] == [(0, 0)]


def test_peek_raises_aborted_once_stopped(monkeypatch, tmp_path):
    looking = Looking(monkeypatch, tmp_path, 5, window="3")
    raised = []
    standard = looking.inspect

    def inspect(view):
        if view.index == 1:
            looking.executor.kill()
            with pytest.raises(Aborted):
                view.peek(0, 1)
            raised.append(view.index)
        return standard(view)

    looking.inspect = inspect
    assert looking.run(workers=4) is not None
    assert raised == [1]
    assert looking.looks[0]["view"].state is None


class Gated:
    """g0's reader: item `gated` waits for `release` once its read began (`reached`)."""

    def __init__(self, count, gated):
        self.items = iter(range(count))
        self.gated = gated
        self.reached, self.release = threading.Event(), threading.Event()

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.items)
        if item == self.gated:
            self.reached.set()
            assert self.release.wait(5)
        return np.full(4, item, np.float32)

    def close(self):
        pass


def test_stop_while_a_peek_waits_ends_it_with_aborted(monkeypatch, tmp_path):
    # Item 1 peeks item 3, whose read is held; Stop comes while the peek waits. The
    # read finishes (liveness), nothing more starts, and the peek raises Aborted.
    reader = Gated(6, 3)
    looking = Looking(monkeypatch, tmp_path, 6, window="3", reader=reader, depth=1)
    raised = []
    standard = looking.inspect
    waiting = threading.Event()
    peeked = ItemWindow._peeked

    def watched(window, slot, node_id):
        decided, value = peeked(window, slot, node_id)
        if slot == 3 and not decided:
            waiting.set()  # the peek of item 3 waits from here
        return decided, value

    monkeypatch.setattr(ItemWindow, "_peeked", watched)

    def stop():
        assert waiting.wait(5) and reader.reached.wait(5)
        looking.executor.kill()
        reader.release.set()

    def inspect(view):
        if view.index == 1:
            threading.Thread(target=stop, daemon=True).start()
            with pytest.raises(Aborted):
                view.peek(0, 2)  # item 3
            raised.append(view.index)
        return standard(view)

    looking.inspect = inspect
    assert looking.run(workers=4) is not None
    assert raised == [1]
    assert looking.cleanup_states == [(0, 0, 0)]


def test_a_peeks_wait_is_not_the_peeking_nodes_time(monkeypatch, tmp_path):
    # Item 1 peeks item 3, whose read is held for 0.5 s after it began: look's time is
    # its own work, not that wait.
    reader = Gated(6, 3)
    looking = Looking(monkeypatch, tmp_path, 6, window="3", reader=reader, depth=2)
    standard = looking.inspect

    def release():
        assert reader.reached.wait(5)
        time.sleep(0.5)
        reader.release.set()

    def inspect(view):
        if view.index == 1:
            threading.Thread(target=release, daemon=True).start()
        return standard(view)

    looking.inspect = inspect
    assert looking.run(workers=4) is None
    times = [
        e["data"]["executionTime"]
        for e in looking.events
        if e["event"] == "node-finish" and e["data"]["nodeId"] == "look"
    ]
    assert len(times) == 6 and max(times) < 0.25


def test_peek_is_served_beyond_the_budget_and_while_paused(monkeypatch, tmp_path):
    # A budget of 0 admits no read-ahead and a pause starts nothing, yet what a peek
    # waits for is produced and computed, as the item the loop waits for is.
    monkeypatch.setattr(process, "memory_budget", lambda: 0)
    looking = Looking(monkeypatch, tmp_path, 5, window="3", nodes=("map",))
    standard = looking.inspect

    def inspect(view):
        if view.index != 1:
            return standard(view)
        looking.executor.pause()
        try:
            return standard(view)
        finally:
            looking.executor.resume()

    looking.inspect = inspect
    assert looking.run(workers=4) is None
    assert_served(looking.looks, 2)
    assert looking.saved == [7.0, 17.0, 27.0, 37.0, 47.0]


@pytest.mark.parametrize("case", ["generator", "map", "fail-fast"])
def test_peek_is_none_for_an_item_that_fails_before_the_input(
    monkeypatch, tmp_path, case
):
    # A deferred generator error at item 2, map failing at item 2, or a fail-fast
    # generator error at item 2: no peek serves item 2; later items are served. Each
    # error surfaces at its item as at K = 1.
    def reader():
        for item in range(6):
            yield (
                ValueError("item 2 failed")
                if item == 2
                else np.full(4, item, np.float32)
            )

    errors = {}
    runs = {}
    for window in ("1", "4"):
        looking = Looking(
            monkeypatch,
            tmp_path,
            6,
            window=window,
            nodes=("map",),
            fail_fast=case == "fail-fast",
            fail_map={2} if case == "map" else (),
            reader=None if case == "map" else Reader(reader()),
        )
        errors[window] = looking.run(workers=4)
        runs[window] = looking
    assert str(errors["4"]) == str(errors["1"])
    many = runs["4"]
    if case == "fail-fast":
        assert [record["index"] for record in many.looks] == [0, 1]
        assert_served(many.looks, 3, count=2, missing={2})
    else:
        assert str(errors["4"]).startswith("Errors occurred during iteration:")
        assert [record["index"] for record in many.looks] == [0, 1, 3, 4, 5]
        assert_served(many.looks, 3, count=6, missing={2})
    assert many.saved == runs["1"].saved


class Reader:
    """A g0 reader over an iterator, closable as Pipeline needs."""

    def __init__(self, items):
        self.items = items

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.items)

    def close(self):
        pass


@pytest.mark.parametrize("through", ["impure", "into-directory", "constant"])
def test_inputs_the_window_does_not_compute_ahead_are_never_served(
    monkeypatch, tmp_path, through
):
    # A node with side effects (not pure CPU either: a node without them runs ahead
    # serially, test_runahead.py), Into Directory (never run ahead), and a constant
    # input: lookahead is 0 and every peek None.
    looking = Looking(
        monkeypatch,
        tmp_path,
        4,
        window="3",
        nodes=("map",),
        impure=("map",) if through == "impure" else (),
    )
    if through == "into-directory":
        set_schema(
            looking.chain.nodes[NodeId("map")], "chainner:utility:into_directory"
        )
    if through == "impure":
        object.__setattr__(
            looking.chain.nodes[NodeId("map")].data, "side_effects", True
        )
    input_id = 1 if through == "constant" else 0
    looking.inspect = lambda view: {
        "lookahead": view.lookahead(input_id),
        "peeks": [view.peek(input_id, offset) for offset in (1, 2)],
    }
    assert looking.run(workers=4) is None
    assert_served(looking.looks, 0)
    assert looking.saved == [7.0, 17.0, 27.0, 37.0]


def test_peek_rejects_bad_offsets_inputs_and_the_loop_thread(monkeypatch, tmp_path):
    looking = Looking(monkeypatch, tmp_path, 3, window="3", depth=0)
    raised = []

    def inspect(view):
        if view.index == 0:
            for call in (lambda: view.peek(0, 0), lambda: view.peek(5, 1)):
                with pytest.raises(ValueError):
                    call()
            raised.append("values")
        return {}

    def on_event(event):  # on the event loop's thread
        if event["event"] == "node-start" and event["data"]["nodeId"] == "save":
            view = looking.looks[-1]["view"]
            for call in (lambda: view.peek(0, 1), lambda: view.lookahead(0)):
                with pytest.raises(RuntimeError, match="look-ahead waits"):
                    call()
            raised.append("loop")

    looking.inspect = inspect
    looking.on_event = on_event
    assert looking.run(workers=4) is None
    assert raised == ["values", "loop", "loop", "loop"]


@pytest.mark.parametrize("nodes", [(), ("map",)])
def test_iterated_by_names_each_input_s_generator(monkeypatch, tmp_path, nodes):
    # look's input 0 comes from g0, directly or through map; input 1 is a constant.
    looking = Looking(monkeypatch, tmp_path, 3, window="1", nodes=nodes, depth=0)
    seen = []

    def inspect(view):
        seen.append((view.iterated_by(0), view.iterated_by(1), view.source))
        with pytest.raises(ValueError):
            view.iterated_by(5)
        return {}

    looking.inspect = inspect
    assert looking.run(workers=4) is None
    assert seen == [(G0, None, G0)] * 3


@pytest.mark.parametrize("schema", sorted(LOCKED_DEVICE_NODES))
def test_a_locked_device_node_computed_ahead_sees_no_iteration_and_is_served(
    monkeypatch, tmp_path, schema
):
    # A locked-device node's id (an Upscale Image: a context, yet computed ahead). The
    # items look's peeks make the window compute ahead see None; the commit path's runs
    # see their item's view. look is served its outputs.
    looking = Looking(monkeypatch, tmp_path, 4, window="3", nodes=("map",), depth=2)
    node = looking.chain.nodes[NodeId("map")]
    set_schema(node, schema)
    object.__setattr__(node.data, "node_context", True)
    seen = {}

    def upscale(context, value):
        view = context.iteration
        seen[round(float(value[0]))] = None if view is None else view.index
        return value * 10 + 7

    object.__setattr__(node.data, "run", upscale)
    assert looking.run(workers=4) is None
    assert seen[0] == 0 and seen[1] is None and seen[2] is None
    assert_served(looking.looks, 2)
    assert looking.saved == [7.0, 17.0, 27.0, 37.0]


def test_load_video_frames_are_peeked_as_delivered(monkeypatch, tmp_path):
    # The real Load Video node over FakeFFmpeg: look's ImageInput receives each frame;
    # its peeks are those very arrays; the frames equal K = 1's.
    monkeypatch.delenv(SERIAL_PRODUCER_VARIABLE, raising=False)
    runs = {}
    for window in ("1", "3"):
        pipeline = load_video_pipeline(monkeypatch, tmp_path, window, 5)
        set_schema(pipeline.chain.nodes[NodeId("g0")], "chainner:image:load_video")
        looks = []

        def look(context, frame, looks=looks):
            view = context.iteration
            looks.append(
                {
                    "index": view.index,
                    "source": view.source,
                    "value": frame,
                    "lookahead": view.lookahead(0),
                    "peeks": [view.peek(0, offset) for offset in (1, 2, 3)],
                }
            )
            return frame

        node = make_node("look", "regularNode", look, 1, 1)
        object.__setattr__(node.data, "inputs", [ImageInput().with_id(0)])
        object.__setattr__(node.data, "side_effects", False)
        object.__setattr__(node.data, "node_context", True)
        set_schema(node, LOOK)
        pipeline.chain.add_node(node)
        edge = pipeline.chain.edge_to(NodeId("sink"), InputId(0))
        assert edge is not None
        pipeline.chain.remove_edge(edge)
        pipeline.chain.add_edge(
            Edge(edge.source, EdgeTarget(NodeId("look"), InputId(0)))
        )
        pipeline.connect("look", "sink", 0)
        assert pipeline.run(workers=6) is None
        runs[window] = (pipeline, looks)
    (one, one_looks), (three, three_looks) = runs.values()
    assert [look["index"] for look in three_looks] == list(range(5))
    assert {look["source"] for look in three_looks} == {"chainner:image:load_video"}
    assert_served(three_looks, 2)
    assert_served(one_looks, 0)
    assert three.saved == one.saved and len(one.saved) == 5
    assert stripped(three.events) == stripped(one.events)


# The window's look-ahead alone (ItemWindow.peek and start).


def test_window_peek_produces_and_computes_beyond_p_budget_and_pause():
    async def body():
        values = [[10], [11], [12], [13]]
        source = PhasedSource(values)
        pool = HeldPool(source.log, held=set())  # every call finishes at once
        progress = ProgressController()
        progress.pause()  # nothing starts but what the commit path waits for
        runs = []
        planner = Planner(lambda value: runs.append(value) or ("map", value))
        window = new_window(
            pool,
            planner,
            generators=[source.generator()],
            size=4,
            budget=0,
            estimate=1,
            progress=progress,
        )
        window.begin(0)
        assert await take(window) is values[0]
        await spin()
        assert source.called("describe") == [0]  # paused: nothing read ahead
        peeked = await window.peek(2, NodeId("g0"))
        assert peeked is values[2]
        assert source.called("describe") == [0, 1, 2]
        assert source.called("materialize") == [0, 2]  # slot 1 is not waited for
        mapped = await window.peek(2, NodeId("map"))
        assert mapped == ("map", values[2]) and runs == [values[2]]
        progress.resume()
        window.release(0)
        window.begin(1)
        assert await take(window) is values[1]
        window.release(1)
        window.begin(2)
        assert await take(window) is peeked  # the same object, produced once
        output, _seconds = await window.take_node(NodeId("map"))
        assert output is mapped and runs == [values[2]]  # computed once
        assert source.called("describe").count(2) == 1
        await close_finishing(window, pool)

    drive(body)


@pytest.mark.parametrize(
    ("case", "slot", "node"),
    [
        ("committing", 0, "g0"),
        ("beyond-k", 3, "g0"),
        ("past-the-end", 2, "g0"),
        ("failed-item", 2, "g0"),
        ("failed-node", 1, "map"),
        ("not-eligible", 1, "map"),
        ("not-a-candidate", 1, "other"),
        ("stopped", 1, "g0"),
    ],
)
def test_window_peek_is_none(case, slot, node):
    async def body():
        values = [[10], [11], ValueError("item 2 failed"), [13]]
        source = PhasedSource(values[:2] if case == "past-the-end" else values)
        pool = HeldPool(source.log, held=set())
        progress = ProgressController()

        def work(value):
            if case == "failed-node":
                raise ValueError("map failed")
            return value

        planner = Planner(work, jobs=case != "not-eligible")
        window = new_window(
            pool,
            planner,
            generators=[source.generator(fail_fast=False)],
            size=3,
            jobs=2,
            progress=progress,
        )
        window.begin(0)
        assert await take(window) == [10]
        if case == "stopped":
            progress.abort()
        assert await window.peek(slot, NodeId(node)) is None
        await close_finishing(window, pool)

    drive(body)


def test_window_peek_waiting_when_closing_begins_gets_none():
    async def body():
        source = PhasedSource([[10], [11], [12]])
        pool = HeldPool(source.log)  # holds every call until the test finishes it
        window = new_window(
            pool, Planner(lambda value: value), generators=[source.generator()], size=3
        )
        window.begin(0)
        taking = asyncio.create_task(take(window))
        await finish_in_turn(pool, [("describe", 0), ("materialize", 0)])
        assert await taking == [10]
        peeking = asyncio.create_task(window.peek(2, NodeId("g0")))
        await settle(lambda: ("describe", 1) in pool.running)
        await spin()
        assert not peeking.done()
        await close_finishing(window, pool)
        assert await peeking is None

    drive(body)


def test_window_start_continues_the_producer_from_the_next_slot():
    # The loop advanced slot 0 itself; start(0) opens the window there, and the first
    # begin (slot 1) keeps the producer where it got: no slot is read twice.
    async def body():
        source = PhasedSource([[10], [11], [12], [13]])
        pool = HeldPool(source.log, held=set())
        window = new_window(
            pool, Planner(lambda value: value), generators=[source.generator()], size=3
        )
        first = source.materialize(source.describe())  # the loop's own advance
        assert first == [10]
        window.start(0)
        await settle(lambda: source.called("describe") == [0, 1, 2])
        window.begin(1)
        assert await take(window) == [11]
        window.release(1)
        window.begin(2)
        assert await take(window) == [12]
        described = source.called("describe")
        assert described[:4] == [0, 1, 2, 3] and len(set(described)) == len(described)
        await close_finishing(window, pool)

    drive(body)


def test_concurrent_peeks_wait_without_spinning(monkeypatch):
    # Two look-aheads wait on held reads; each wakes only when something finishes.
    async def body():
        source = PhasedSource([[10], [11], [12], [13]])
        pool = ManualPool()
        window = new_window(
            pool, Planner(lambda value: value), generators=[source.generator()], size=4
        )
        pumps = []
        pump = window._pump

        def counted():
            pumps.append(None)
            pump()

        monkeypatch.setattr(window, "_pump", counted)
        window.begin(0)
        taking = asyncio.create_task(take(window))
        await settle(lambda: pool.calls)
        pool.finish(0)  # describe 0
        await settle(lambda: len(pool.calls) == 2)
        pool.finish(1)  # materialize 0
        assert await taking == [10]
        first = asyncio.create_task(window.peek(2, NodeId("g0")))
        second = asyncio.create_task(window.peek(3, NodeId("g0")))
        await spin()
        before = len(pumps)
        await spin()
        assert len(pumps) == before  # waiting, not spinning
        pool.hold = False
        for index in range(len(pool.calls)):
            if not pool.calls[index][1].done():
                pool.finish(index)
        while not (first.done() and second.done()):
            for index in range(len(pool.calls)):
                if not pool.calls[index][1].done():
                    pool.finish(index)
            await asyncio.sleep(0)
        assert await first == [12] and await second == [13]
        await window.close()

    drive(body)


def test_iterated_condition_json():
    condition = Condition.iterated(0, "chainner:image:load_video")
    assert condition.to_json() == {
        "kind": "iterated",
        "input": 0,
        "source": "chainner:image:load_video",
    }
    assert (~condition).to_json() == {"kind": "not", "condition": condition.to_json()}
