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

Nodes with a NodeContext (node_context=True) are not covered yet (the engine's context arrives with E2).
"""

from __future__ import annotations

import functools
import importlib
import struct
from pathlib import Path

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
    """A value's identity for the comparison: exact bits for arrays, floats and colours."""
    if isinstance(value, np.ndarray):
        return ("ndarray", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, float):
        return ("float", struct.pack("<d", value))
    if isinstance(value, Color):
        return ("Color", tuple(struct.pack("<d", channel) for channel in value.value))
    return (f"{type(value).__module__}.{type(value).__qualname__}", value)


def _outcome(node: api.NodeData, run) -> tuple[str, object]:
    try:
        return "outputs", process.enforce_output(run(), node).output
    except Exception as error:
        return "error", (error_name(error), str(error))


def run_both(schema_id: str, inputs: list[object]) -> tuple[tuple[str, object], tuple[str, object]]:
    """(python, native) outcomes: ("outputs", enforced outputs) or ("error", (type name, message))."""
    node = node_data(schema_id)
    assert not node.node_context, f"{schema_id} takes a NodeContext, which run_native does not provide yet"
    assert node.kind == "regularNode", f"{schema_id} is a {node.kind}; run_native runs regular nodes"
    assert schema_id in graph().native_schema_ids(), f"{schema_id} has no native entry point"
    enforced = process.enforce_inputs(list(inputs), node, "native-differential", [])
    python = _outcome(node, lambda: node.run(*enforced))
    native = _outcome(node, lambda: _single(node, graph().run_native(schema_id, enforced)))
    return python, native


def _single(node: api.NodeData, outputs: list[object]) -> object:
    """run_native's output list in the shape node.run returns (None, one value, or a tuple)."""
    if not node.outputs:
        assert outputs == [], f"Expected no outputs, got {len(outputs)}"
        return None
    return outputs[0] if len(node.outputs) == 1 else tuple(outputs)


def assert_native_matches(schema_id: str, inputs: list[object]) -> tuple[str, object]:
    """Asserts that schema_id's native entry point matches its Python body on inputs; returns the native outcome."""
    python, native = run_both(schema_id, inputs)
    assert python[0] == native[0], f"{schema_id}: Python gave {python}, native gave {native}"
    if python[0] == "error":
        assert python[1] == native[1], f"{schema_id}: errors differ: Python {python[1]}, native {native[1]}"
        return native
    expected = [comparable(value) for value in python[1]]
    actual = [comparable(value) for value in native[1]]
    assert len(expected) == len(actual), f"{schema_id}: {len(expected)} outputs vs {len(actual)}"
    for index, (e, a) in enumerate(zip(expected, actual)):
        assert e == a, f"{schema_id}: output {index} differs: Python {e[:3]}, native {a[:3]}"
    return native
