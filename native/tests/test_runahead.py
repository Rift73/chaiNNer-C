"""Run-ahead of a serial producer: a stateful node the window never speculates (a GPU or
context node) is computed ahead for a look-ahead that waits for it, item by item in
order, once per item, with its own iteration view (process.py, item_window.py). A
look-ahead the read-ahead check switches off is a run warning."""

import pytest
from test_generator_lifecycle import make_node
from test_item_window import set_schema
from test_iteration_view import Looking, assert_served, computed_folder

from api import InputId, NodeId
from chain.chain import Edge, EdgeSource, EdgeTarget


def producing(monkeypatch, tmp_path, count, *, depth, fail=(), **options):
    """Looking's g0 -> map -> look -> save, map a stateful context node that is not
    pure CPU. map records each item it runs (its view's index) and how many items look
    had run then; its output counts the items its state holds."""
    side_effects = options.pop("side_effects", False)
    looking = Looking(
        monkeypatch,
        tmp_path,
        count,
        window="4",
        depth=depth,
        nodes=("map",),
        impure=("map",),
        **options,
    )
    node = looking.chain.nodes[NodeId("map")]
    object.__setattr__(node.data, "node_context", True)
    object.__setattr__(node.data, "side_effects", side_effects)
    ran = []

    def produce(context, value):
        view = context.iteration
        if view.state is None:
            view.state = []
        ran.append((view.index, len(looking.looks)))
        if view.index in fail:
            raise ValueError(f"producer failed {view.index}")
        output = value * 10 + 7 + 1000 * len(view.state)
        view.state.append(view.index)
        return output

    object.__setattr__(node.data, "run", produce)
    return looking, ran


def test_a_serial_producer_runs_ahead_once_per_item_in_order(monkeypatch, tmp_path):
    base, base_ran = producing(monkeypatch, tmp_path, 8, depth=0)
    assert base.run(workers=8) is None
    looking, ran = producing(monkeypatch, tmp_path, 8, depth=3)
    assert looking.run(workers=8) is None
    # Same items, same order, each once; same outputs (they count the shared state).
    assert [index for index, _ in ran] == [index for index, _ in base_ran]
    assert [index for index, _ in ran] == list(range(8))
    assert looking.saved == base.saved
    # Some items ran before look reached them, so the look-ahead was served by them.
    assert any(looks < index for index, looks in ran)
    assert_served(looking.looks, 3)


@pytest.mark.parametrize("fail_fast", [True, False])
def test_an_error_ahead_surfaces_at_its_own_item(monkeypatch, tmp_path, fail_fast):
    base, base_ran = producing(
        monkeypatch, tmp_path, 8, depth=0, fail={5}, fail_fast=fail_fast
    )
    base_error = base.run(workers=8)
    looking, ran = producing(
        monkeypatch, tmp_path, 8, depth=3, fail={5}, fail_fast=fail_fast
    )
    error = looking.run(workers=8)
    assert error is not None and base_error is not None
    assert "producer failed 5" in str(error) and "producer failed 5" in str(base_error)
    assert [index for index, _ in ran] == [index for index, _ in base_ran]
    assert looking.saved == base.saved
    # Failing fast, the run ends at item 5: nothing after it is served. Deferring
    # errors, nothing is served (test_deferred_errors_never_run_a_producer_ahead).
    if fail_fast:
        assert_served(looking.looks, 3, count=5)
    else:
        assert_served(looking.looks, 0)


def test_a_producer_with_side_effects_never_runs_ahead(monkeypatch, tmp_path):
    looking, ran = producing(monkeypatch, tmp_path, 6, depth=3, side_effects=True)
    assert looking.run(workers=8) is None
    assert [index for index, _ in ran] == list(range(6))
    assert all(looks == index for index, looks in ran)
    assert_served(looking.looks, 0)


@pytest.mark.parametrize("workers", [2, 3])
def test_a_pool_too_small_for_the_waits_serves_nothing(monkeypatch, tmp_path, workers):
    # The waiting look, map computed ahead (it may wait in its own look-ahead) and one
    # thread that works: 3. With 2, nothing is served and the run neither hangs nor
    # changes; with 3, everything is.
    base, base_ran = producing(monkeypatch, tmp_path, 6, depth=0)
    assert base.run(workers=workers) is None
    looking, ran = producing(monkeypatch, tmp_path, 6, depth=3)
    assert looking.run(workers=workers) is None
    assert [index for index, _ in ran] == [index for index, _ in base_ran]
    assert looking.saved == base.saved
    assert_served(looking.looks, 3 if workers == 3 else 0)
    if workers == 2:
        assert all(looks == index for index, looks in ran)


def rewired(monkeypatch, tmp_path, depth):
    """producing's chain with fail_fast False, look's input 0 from `bad` (pure, fails at
    item 3) and map moved to look's input 1, which look peeks (depth > 0)."""
    looking, ran = producing(monkeypatch, tmp_path, 6, depth=depth, fail_fast=False)
    chain = looking.chain

    def bad(value):
        if round(float(value[0])) == 3:
            raise ValueError("bad failed 3")
        return value

    node = make_node("bad", "regularNode", bad, 1, 1)
    object.__setattr__(node.data, "side_effects", False)
    set_schema(node, "chainner:image:test-bad")
    chain.add_node(node)
    g0 = chain.nodes[NodeId("g0")]
    chain.add_edge(
        Edge(EdgeSource(g0.id, g0.data.outputs[0].id), EdgeTarget(node.id, InputId(0)))
    )
    edge = chain.edge_to(NodeId("look"), InputId(0))
    assert edge is not None
    chain.remove_edge(edge)
    chain.inputs.set(NodeId("look"), InputId(1), None)
    chain.add_edge(
        Edge(
            EdgeSource(node.id, node.data.outputs[0].id),
            EdgeTarget(NodeId("look"), InputId(0)),
        )
    )
    chain.add_edge(Edge(edge.source, EdgeTarget(NodeId("look"), InputId(1))))
    looking.inspect = lambda view: (
        {"lookahead": view.lookahead(1), "peeks": [view.peek(1, k) for k in (1, 2, 3)]}
        if depth
        else {}
    )
    return looking, ran


def test_deferred_errors_never_run_a_producer_ahead(monkeypatch, tmp_path):
    # look's first input (bad) fails at item 3 before its second (map) is gathered, so
    # in the loop's order map never runs at item 3; run ahead it would have, and its
    # state could not drop the item. Deferring errors, map is never run ahead.
    base, base_ran = rewired(monkeypatch, tmp_path, 0)
    base_error = base.run(workers=8)
    looking, ran = rewired(monkeypatch, tmp_path, 3)
    error = looking.run(workers=8)
    assert "bad failed 3" in str(base_error) and "bad failed 3" in str(error)
    assert ran == base_ran
    assert [index for index, _ in ran] == [0, 1, 2, 4, 5]
    assert looking.saved == base.saved
    assert all(record["lookahead"] == 0 for record in looking.looks)


def saving_into(monkeypatch, tmp_path, folder, depth=3):
    """Looking whose g0 reads a file per item from tmp_path / "in" (as Load Images) and
    whose save writes into `folder`: "computed" by a node save's gather runs after look,
    else the typed-in tmp_path / folder."""
    looking = Looking(monkeypatch, tmp_path, 4, window="3", depth=depth)
    computed_folder(looking, tmp_path, (tmp_path / "in" / "0.png",))
    if folder != "computed":
        looking.chain.remove_node(NodeId("folder"))
        looking.chain.inputs.set(NodeId("save"), InputId(1), str(tmp_path / folder))
    return looking


def run_warnings(looking):
    return [
        e["data"]["message"]
        for e in looking.events
        if e["event"] == "execution-warning"
    ]


@pytest.mark.parametrize(
    ("folder", "decision", "reason"),
    [
        (
            "computed",
            "K=1 (output directory unresolved: save)",
            "save's output folder is computed after look",
        ),
        (
            "in",
            "K=1 (source under output directory: save)",
            "save's output folder may overlap the source folder",
        ),
        ("out", "K=3 (override)", None),
    ],
)
def test_a_look_ahead_the_read_ahead_check_switches_off_is_a_run_warning(
    monkeypatch, tmp_path, folder, decision, reason
):
    # look looks ahead at every item from item 0. When the read-ahead check sets K = 1
    # (save's folder computed after look, or one that holds the source files), the run
    # warns once that look runs without its look-ahead; with K = 3 it does not.
    looking = saving_into(monkeypatch, tmp_path, folder)
    assert looking.run(workers=4) is None
    assert looking.decisions == [f"item window {decision}"]
    assert looking.saved == [7.0, 17.0, 27.0, 37.0]
    if reason is None:
        assert_served(looking.looks, 2)
        assert run_warnings(looking) == []
    else:
        assert_served(looking.looks, 0)
        effect = "look runs without its look-ahead: it uses only the current and earlier frames"
        assert run_warnings(looking) == [f"{reason}, so {effect}."]


def test_a_loop_without_a_look_ahead_has_no_run_warning(monkeypatch, tmp_path):
    # The read-ahead check sets K = 1, but no node looks ahead: nothing to warn of.
    looking = saving_into(monkeypatch, tmp_path, "in", depth=0)
    assert looking.run(workers=4) is None
    assert looking.decisions == [
        "item window K=1 (source under output directory: save)"
    ]
    assert run_warnings(looking) == []
