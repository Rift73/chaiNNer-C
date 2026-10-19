"""Differential harness for the native node entry points (native/src/run/nodes).

assert_native_matches(schema_id, inputs) enforces the raw inputs (the values the UI and upstream nodes send) once
with process.enforce_inputs, then runs on those enforced values
- the node's Python body (node.run) followed by process.enforce_output, as process.run_node does, and
- the native entry point (_chainner_graph.run_native, which returns the outputs before output enforcement) followed
  by the same process.enforce_output,
and asserts identical results: arrays by dtype, shape and C-ordered bytes (NaN payloads and -0 included), floats
by their bits, other values by type and value; errors by Python type name and message (str(e)). Both sides are
compared after output enforcement, so a native node may return the values its Python body returns or their
normalized form.

A node with a NodeContext gets a test context on both sides: the same package settings (a dict), node id and
storage directory, never aborted. A generator's outcome holds its expected length, fail_fast, source paths, static
outputs (enforced) and up to max_items items, each enforced as the executor fills an item's outputs, or the item's
error (a yielded exception, or its enforce error).
"""

from __future__ import annotations

import functools
import importlib
import struct
from collections.abc import Callable, Sequence
from pathlib import Path, PurePath
from typing import Literal

import numpy as np

import api
import process
from nodes.impl.color.color import Color
from nodes.impl.native_graph import graph

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"


@functools.cache
def node_data(schema_id: str) -> api.NodeData:
    """The registered NodeData of schema_id, importing only the module that defines it."""
    pattern = f'schema_id="{schema_id}"'
    for file in sorted((BACKEND / "packages").rglob("*.py")):
        if pattern in file.read_text(encoding="utf-8"):
            importlib.import_module(".".join(file.relative_to(BACKEND).with_suffix("").parts))
            break
    else:
        raise KeyError(f"No node module defines {schema_id}")
    for package in api.registry.packages.values():
        for category in package.categories:
            for group in category.node_groups:
                for node in group.nodes:
                    if node.schema_id == schema_id:
                        return node
    raise KeyError(f"{schema_id} was not registered by its module")


def error_name(error: BaseException) -> str:
    """The exception's type as the engine names it (NodeError.python_type)."""
    kind = type(error)
    return kind.__qualname__ if kind.__module__ == "builtins" else f"{kind.__module__}.{kind.__qualname__}"


def comparable(value: object) -> object:
    """A value's identity for the comparison, recursively, always with its exact type: arrays by dtype, shape,
    writeable flag and C-ordered bytes, floats and NumPy scalars by their bits (-0.0 is not 0.0, a NaN equals the
    same NaN), paths by str() (case-sensitive), colours by their channels, tuples, lists and dicts by their items."""
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    if isinstance(value, np.ndarray):
        return (kind, value.dtype.str, value.shape, value.flags.writeable, value.tobytes())
    if isinstance(value, np.generic):
        return (kind, value.dtype.str, value.tobytes())
    if isinstance(value, float):
        return (kind, struct.pack("<d", value))
    if isinstance(value, PurePath):
        return (kind, str(value))
    if isinstance(value, Color):
        # Channel types too: Color's _norm keeps an int at the bounds (Color.gray(0).value == (0,))
        return (kind, comparable(value.value))
    if isinstance(value, (tuple, list)):
        return (kind, tuple(comparable(item) for item in value))
    if isinstance(value, dict):
        return (kind, tuple((comparable(key), comparable(item)) for key, item in value.items()))
    return (kind, value)


class DifferentialContext(api.NodeContext):
    """The Python body's context in a differential run: never aborted, never paused."""

    def __init__(self, settings: dict[str, object], node_id: api.NodeId, storage_dir: Path) -> None:
        super().__init__()
        self.__settings = api.SettingsParser(settings)
        self.__node_id = node_id
        self.__storage_dir = storage_dir

    @property
    def aborted(self) -> bool:
        return False

    @property
    def paused(self) -> bool:
        return False

    def set_progress(self, progress: float) -> None:
        self.check_aborted()

    @property
    def settings(self) -> api.SettingsParser:
        return self.__settings

    @property
    def node_id(self) -> api.NodeId:
        return self.__node_id

    @property
    def storage_dir(self) -> Path:
        return self.__storage_dir

    def add_cleanup(self, fn: Callable[[], None], after: Literal["node", "chain"] = "chain") -> None:
        raise NotImplementedError("Differential runs register no cleanups")


Outcome = tuple[str, object]


def _error(error: Exception) -> Outcome:
    return "error", (error_name(error), str(error))


def _items(node: api.NodeData, output: process.GeneratorOutput, max_items: int) -> list[Outcome]:
    """The generator's items as process.Executor fills them (__generator_fill_partial_output), or their errors."""
    iterated = node.single_iterable_output.outputs
    items: list[Outcome] = []
    for values in output.generator.supplier():
        if len(items) == max_items:
            break
        if isinstance(values, Exception):
            items.append(_error(values))
            continue
        try:
            if len(iterated) == 1:
                listed = [values]
            else:
                assert isinstance(values, (tuple, list))
                listed = list(values)
            assert len(listed) == len(iterated)
            filled = [o.enforce_(listed.pop(0)) for o in node.outputs if o.id in iterated]
            items.append(("item", [comparable(value) for value in filled]))
        except Exception as error:
            items.append(_error(error))
    return items


def _outcome(node: api.NodeData, run: Callable[[], object], max_items: int) -> Outcome:
    """("outputs", ...), ("generator", ...), ("error", ...), or ("declined", None) when the native entry point
    declined to the Python body (run_engine.hpp RunPython)."""
    try:
        raw = run()
        if node.kind == "generator":
            output = process.enforce_generator_output(raw, node)
            generator = output.generator
            static = [comparable(v) for o, v in zip(node.outputs, output.partial_output) if o.id not in output.info.outputs]
            return "generator", (
                generator.expected_length,
                generator.fail_fast,
                comparable(generator.source_paths),
                static,
                _items(node, output, max_items),
            )
        return "outputs", [comparable(value) for value in process.enforce_output(raw, node).output]
    except graph().NativeDeclined:
        return "declined", None
    except Exception as error:
        return _error(error)


def _shaped(node: api.NodeData, outputs: list[object]) -> object:
    """run_native's output list in the shape node.run returns: None, one value, or a tuple (a generator's
    Generator first, then its static outputs)."""
    if not node.outputs:
        assert outputs == [], f"Expected no outputs, got {len(outputs)}"
        return None
    if node.kind == "generator":
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
    return outputs[0] if len(node.outputs) == 1 else tuple(outputs)


def run_both(
    schema_id: str,
    inputs: Sequence[object],
    *,
    settings: dict[str, object] | None = None,
    storage_dir: Path | None = None,
    max_items: int = 1000,
) -> tuple[Outcome, Outcome]:
    """(python, native) outcomes: ("outputs", comparable enforced outputs), ("generator", (expected length,
    fail_fast, source paths, static outputs, items)) or ("error", (type name, message)). When the native entry point
    declines, native is ("declined", None) and the Python body is not run (python is ("not run", None))."""
    node = node_data(schema_id)
    assert node.kind in ("regularNode", "generator"), f"{schema_id} is a {node.kind}; run_native has no collectors"
    assert schema_id in graph().native_schema_ids(), f"{schema_id} has no native entry point"
    node_id = api.NodeId("native-differential")
    directory = storage_dir if storage_dir is not None else Path.cwd()
    context = DifferentialContext(dict(settings or {}), node_id, directory)
    enforced = process.enforce_inputs(list(inputs), node, node_id, [])
    arguments = [context, *enforced] if node.node_context else enforced
    native = _outcome(
        node,
        lambda: _shaped(
            node,
            graph().run_native(
                schema_id, enforced, settings=dict(settings or {}), node_id=node_id, storage_dir=directory
            ),
        ),
        max_items,
    )
    if native[0] == "declined":
        return ("not run", None), native
    python = _outcome(node, lambda: node.run(*arguments), max_items)
    return python, native


def assert_native_matches(
    schema_id: str,
    inputs: Sequence[object],
    *,
    settings: dict[str, object] | None = None,
    storage_dir: Path | None = None,
    max_items: int = 1000,
    declines: bool = False,
) -> Outcome:
    """Asserts that schema_id's native entry point matches its Python body on inputs; returns the native outcome.
    With declines=True it asserts instead that the entry point declines these inputs to the Python body."""
    python, native = run_both(schema_id, inputs, settings=settings, storage_dir=storage_dir, max_items=max_items)
    if declines:
        assert native[0] == "declined", f"{schema_id}: expected a decline to Python, native gave {native!r:.300}"
        return native
    assert native[0] != "declined", f"{schema_id}: the native entry point declined these inputs to Python"
    assert python[0] == native[0], f"{schema_id}: Python gave {python!r:.300}, native gave {native!r:.300}"
    if python[0] == "outputs":
        python_outputs, native_outputs = python[1], native[1]
        assert isinstance(python_outputs, list) and isinstance(native_outputs, list)
        assert len(python_outputs) == len(native_outputs), f"{schema_id}: output counts differ"
        for index, (e, a) in enumerate(zip(python_outputs, native_outputs)):
            assert e == a, f"{schema_id}: output {index} differs: Python {e!r:.200}, native {a!r:.200}"
    else:
        assert python[1] == native[1], f"{schema_id}: Python {python[1]!r:.300}, native {native[1]!r:.300}"
    return native
