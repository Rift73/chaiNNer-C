"""Installed/source execution contracts with independent frozen API objects.

Ordered state and branch control are native. Path/range/number primitives remain
their original engines. Note and Execution Number are intentionally trivial.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import itertools
import json
import math
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

import pytest

from nodes.impl.native_graph import graph

ROOT = Path(__file__).resolve().parents[2]
FROZEN = Path(__file__).with_name("reference_execution_ops")
NODE_ROOT = "packages/chaiNNer_standard/utility"
FILES = {
    "directory": "directory/directory_go_into.py",
    "accumulate": "math/accumulate.py",
    "logic": "math/logic_operation.py",
    "conditional": "value/conditional.py",
    "range": "value/range.py",
    "number": "value/execution_number.py",
    "note": "text/note.py",
}
FUNCTIONS = {
    "directory": "directory_go_into_node",
    "accumulate": "accumulate_node",
    "logic": "logic_operation_node",
    "conditional": "conditional_node",
    "range": "range_node",
    "number": "execution_number_node",
    "note": "note_node",
}


def declarations(path, names, context: dict[str, Any]) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    selected: list[ast.stmt] = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = []
            selected.append(node)
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(path), "exec"), context)
    return context


@lru_cache
def frozen_api(variant, module="iter") -> Any:
    path = FROZEN / variant / "api" / (module + ".py")
    key = "_execution_reference_" + variant + "_" + module
    spec = importlib.util.spec_from_file_location(key, path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    sys.modules[key] = result
    spec.loader.exec_module(result)
    return result


def namespace(kind, variant="installed", current=False) -> dict[str, Any]:
    base = ROOT / "backend/src" if current else FROZEN / variant
    api = frozen_api(variant)
    return declarations(
        base / NODE_ROOT / FILES[kind],
        {FUNCTIONS[kind], "Operation", "LogicOperation"},
        {
            "Enum": Enum,
            "Path": Path,
            "graph": graph,
            "Generator": api.Generator,
            "Collector": api.Collector,
        },
    )


def capture(callback):
    try:
        return callback(), None
    except BaseException as error:
        return None, error


def error_key(error):
    if error is None:
        return None
    return (
        type(error),
        error.args,
        error_key(error.__cause__),
        error_key(error.__context__),
        error.__suppress_context__,
    )


def same_value(a, b):
    assert type(a) is type(b)
    if isinstance(a, float):
        assert struct.pack("d", a) == struct.pack("d", b)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=False):
            same_value(x, y)
    else:
        assert a == b


def equivalent(old, new):
    assert error_key(old[1]) == error_key(new[1])
    if old[1] is None:
        same_value(old[0], new[0])


@pytest.mark.parametrize("variant", ["source", "installed"])
@pytest.mark.parametrize("kind", FILES)
def test_frozen_metadata(variant, kind):
    relative = f"{variant}/{NODE_ROOT}/{FILES[kind]}"
    manifest = json.loads((FROZEN / "sources.json").read_text())
    assert (
        hashlib.sha256((FROZEN / relative).read_bytes()).hexdigest()
        == manifest["sha256"][relative]
    )
    old = ast.parse((FROZEN / relative).read_text())
    new = ast.parse((ROOT / "backend/src" / NODE_ROOT / FILES[kind]).read_text())
    a = next(n for n in old.body if isinstance(n, ast.FunctionDef))
    b = next(n for n in new.body if isinstance(n, ast.FunctionDef))
    assert ast.dump(a.args) == ast.dump(b.args)
    assert a.returns is not None and b.returns is not None
    assert ast.dump(a.returns) == ast.dump(b.returns)
    assert [ast.dump(x) for x in a.decorator_list] == [
        ast.dump(x) for x in b.decorator_list
    ]


SEQUENCES = [
    [],
    [0],
    [-0.0],
    [0.0, -0.0],
    [-0.0, 0.0],
    [1, 2, 3, -7],
    [2**150, -(2**100), 37],
    [1e100, 1.0, -1e100, 1.0],
    [1e-300, 1e-30, 1e300],
    [float("nan"), 0.0, -float("inf")],
    [1.0, float("nan"), -1.0],
    [float("inf"), -float("inf"), 0.0],
    [True, False, True],
    [Fraction(1, 3), Fraction(2, 7)],
    [Decimal("1.25"), Decimal("-2.5")],
    [complex(1, 2), complex(2, 1)],
    ["a", 3],
    [None, 2],
]


@pytest.mark.parametrize("variant", ["installed", "source"])
@pytest.mark.parametrize("operation", ["SUM", "PRODUCT", "MINIMUM", "MAXIMUM"])
@pytest.mark.parametrize("values", SEQUENCES)
def test_accumulate_order_and_errors(variant, operation, values):
    collectors = []
    for current in (False, True):
        ns = namespace("accumulate", variant, current)
        collectors.append(ns["accumulate_node"](None, ns["Operation"][operation]))
    for item in values:
        equivalent(*(capture(partial(c.on_iterate, item)) for c in collectors))
        equivalent(*(capture(c.on_complete) for c in collectors))
    equivalent(*(capture(c.on_complete) for c in collectors))
    equivalent(*(capture(c.on_complete) for c in collectors))


def test_accumulate_protocol_state_identity():
    def run(current):
        events = []
        initial = object()

        class Operation:
            @property
            def neutral(self):
                events.append("neutral")
                return initial

            def reduce(self, value, item):
                events.append((value, item))
                if item == "error":
                    raise LookupError("retained state")
                return item

        ns = namespace("accumulate", current=current)
        collector = ns["accumulate_node"](None, Operation())
        assert collector.on_complete() is initial
        marker = object()
        assert collector.on_iterate(marker) is None
        assert collector.on_complete() is marker
        with pytest.raises(LookupError, match="retained state"):
            collector.on_iterate("error")
        assert collector.on_complete() is marker
        assert collector.on_iterate(initial) is None
        assert collector.on_complete() is initial
        assert len(events) == 4

    run(False)
    run(True)


@pytest.mark.parametrize("operation", ["MINIMUM", "MAXIMUM"])
def test_accumulate_tie_identity(operation):
    for current in (False, True):
        ns = namespace("accumulate", current=current)
        c = ns["accumulate_node"](None, ns["Operation"][operation])
        a, b = float("-0.0"), float("0.0")
        c.on_iterate(a)
        c.on_iterate(b)
        assert c.on_complete() is a


class LazyProbe:
    def __init__(self, value=None, error=None):
        self.result = value
        self.error = error
        self.calls = 0

    @property
    def value(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


@pytest.mark.parametrize("variant", ["installed", "source"])
@pytest.mark.parametrize("operation", ["AND", "OR", "XOR", "NOT", "invalid"])
@pytest.mark.parametrize(
    "a,b", itertools.product([False, True, 0, "x", [], [1]], repeat=2)
)
def test_logic_short_circuit(variant, operation, a, b):
    results, calls = [], []
    for current in (False, True):
        ns = namespace("logic", variant, current)
        op = ns["LogicOperation"][operation] if operation != "invalid" else None
        lazy = LazyProbe(b)
        results.append(capture(partial(ns["logic_operation_node"], op, a, lazy)))
        calls.append(lazy.calls)
        if results[-1][1] is None and operation in ("AND", "OR"):
            expected = (a and b) if operation == "AND" else (a or b)
            assert results[-1][0] is expected
    equivalent(*results)
    assert calls[0] == calls[1]


@pytest.mark.parametrize("operation", ["AND", "OR", "XOR", "NOT"])
@pytest.mark.parametrize("a", [False, True])
def test_lazy_real_cached_errors(operation, a):
    for current in (False, True):
        events = []
        failure = LookupError("selected only")

        def factory(events=events, failure=failure):
            events.append(1)
            raise failure

        lazy = frozen_api("installed", "lazy").Lazy(factory)
        ns = namespace("logic", current=current)
        op = ns["LogicOperation"][operation]
        for _ in range(2):
            result = capture(partial(ns["logic_operation_node"], op, a, lazy))
            evaluates = operation == "XOR" or (
                (operation == "AND" and a) or (operation == "OR" and not a)
            )
            if evaluates:
                assert result[1] is failure
            else:
                assert result[1] is None
        assert len(events) == int(evaluates)


@pytest.mark.parametrize("condition", [False, True, 0, 1, [], [1], "", "x"])
def test_conditional_identity_and_untouched_branch(condition):
    marker = object()
    for current in (False, True):
        selected = LazyProbe(marker)
        ignored = LazyProbe(error=AssertionError("not selected"))
        args = (selected, ignored) if condition else (ignored, selected)
        ns = namespace("conditional", current=current)
        assert ns["conditional_node"](condition, *args) is marker
        assert selected.calls == 1 and ignored.calls == 0


@pytest.mark.parametrize("variant", ["installed", "source"])
@pytest.mark.parametrize("start,end", [(-2, 3), (0, 0), (3, -2), (2**150, 2**150 + 3)])
@pytest.mark.parametrize("a,b", itertools.product([False, True], repeat=2))
def test_range_endpoints_reusable(variant, start, end, a, b):
    results = []
    for current in (False, True):
        ns = namespace("range", variant, current)
        result, error = capture(partial(ns["range_node"], start, a, end, b))
        if error is not None:
            results.append((None, error))
            continue
        assert result is not None
        assert result.fail_fast is True and result.metadata is None
        first, second = result.supplier(), result.supplier()
        assert first is not second and iter(first) is first
        values = list(first)
        assert list(second) == values
        assert list(first) == []
        results.append(((result.expected_length, values), None))
    equivalent(*results)


def test_range_declares_no_per_item_source_files():
    assert (
        namespace("range", current=True)["range_node"](0, True, 3, False).source_paths
        == ()
    )


@pytest.mark.parametrize("start,end", [(0.0, 2.0), (None, 2), (0, "3"), (0, math.inf)])
def test_range_deferred_input_errors(start, end):
    results = []
    for current in (False, True):
        ns = namespace("range", current=current)

        def run(ns=ns):
            gen = ns["range_node"](start, True, end, False)
            cursor = gen.supplier()
            first = capture(lambda: next(cursor))
            assert list(cursor) == []
            return gen.expected_length, error_key(first[1])

        results.append(capture(run))
    equivalent(*results)


def test_range_huge_count_and_close():
    for current in (False, True):
        ns = namespace("range", current=current)
        gen = ns["range_node"](0, True, 2**200, False)
        assert gen.expected_length == 2**200
        cursor = gen.supplier()
        assert list(itertools.islice(cursor, 4)) == [0, 1, 2, 3]
        cursor.close()
        assert list(cursor) == []


@pytest.mark.parametrize("action", ["next", "close", "error", "base_error"])
def test_range_callback_errors_and_reentry(action):
    observations = []
    for current in (False, True):
        state = {}

        class Start:
            def __init__(self, state):
                self.state = state

            def __add__(self, index):
                if index:
                    return index
                if action == "next":
                    return next(self.state["iterator"])
                if action == "close":
                    return self.state["iterator"].close()
                if action == "error":
                    raise StopIteration("map error becomes an item")
                raise KeyboardInterrupt("abort mapping")

        class End:
            def __sub__(self, _):
                return 2

        ns = namespace("range", current=current)
        generator = ns["range_node"](Start(state), True, End(), False)
        iterator = generator.supplier()
        state["iterator"] = iterator
        first, error = capture(partial(next, iterator))
        rest = list(iterator)
        observations.append(
            (
                error_key(first) if isinstance(first, Exception) else first,
                error_key(error),
                rest,
            )
        )
        exhausted = capture(partial(next, iterator))[1]
        assert isinstance(exhausted, StopIteration) and exhausted.args == ()
    assert observations[0] == observations[1]


def test_logic_xor_protocol_returns_original_comparison_object():
    marker = object()

    class Value:
        def __ne__(self, _) -> Any:
            return marker

    for current in (False, True):
        ns = namespace("logic", current=current)
        value = LazyProbe(1)
        assert (
            ns["logic_operation_node"](ns["LogicOperation"].XOR, Value(), value)
            is marker
        )
        assert value.calls == 1


@pytest.mark.parametrize(
    "folders", [(), (None,), ("a",), ("a", None, "..", "世界"), ("", "x/../y")]
)
def test_directory_resolves_each_step(tmp_path, folders):
    old, new = namespace("directory"), namespace("directory", current=True)
    root = tmp_path / "nonexistent"
    equivalent(
        capture(lambda: old["directory_go_into_node"](root, *folders)),
        capture(lambda: new["directory_go_into_node"](root, *folders)),
    )
    if not any(x is not None for x in folders):
        assert new["directory_go_into_node"](root, *folders) is root
    assert not root.exists()


def test_directory_protocol_order_and_failure():
    for current in (False, True):
        events = []

        class Directory:
            def __init__(self, events):
                self.events = events

            def __truediv__(self, folder):
                self.events.append(("join", folder))
                return self

            def resolve(self):
                self.events.append("resolve")
                if len(self.events) == 4:
                    raise OSError("resolve failure")
                return self

        ns = namespace("directory", current=current)
        with pytest.raises(OSError, match="resolve failure"):
            ns["directory_go_into_node"](Directory(events), "a", None, "b", "c")
        assert events == [("join", "a"), "resolve", ("join", "b"), "resolve"]


def test_trivial_nodes_have_no_backend_state():
    value = object()
    for current in (False, True):
        assert (
            namespace("number", current=current)["execution_number_node"](value)
            is value
        )
        assert namespace("note", current=current)["note_node"](value, value) is None


def test_independent_concurrent_collectors_and_ranges():
    ns = namespace("accumulate", current=True)
    ranges = namespace("range", current=True)

    def run(seed):
        collector = ns["accumulate_node"](None, ns["Operation"].SUM)
        generator = ranges["range_node"](seed, True, seed + 17, False)
        for value in generator.supplier():
            collector.on_iterate(value)
        return collector.on_complete()

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert list(executor.map(run, range(32))) == [
            sum(range(seed, seed + 17)) for seed in range(32)
        ]
