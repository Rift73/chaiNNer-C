"""Native utility math entry points (native/src/run/nodes/utility_math.cpp) against their Python bodies.

Each operation runs over the cross product of edge operands. An entry point declines to the Python body exactly
where that body raises an error whose text formats a float, returns an int beyond int64, or (pow) gives a subnormal
result, whose error status the C runtime's errno decides; each test asserts both paths occur."""

import sys

import numpy as np
import pytest

from native_differential import assert_native_matches, node_data
from nodes.impl.native_graph import graph
from packages.chaiNNer_standard.utility.math.math import MathOperation
from packages.chaiNNer_standard.utility.math.round import RoundOperation, RoundScale

# Both zeros, ones, halves and ties, int64 and 2**53 edges, the extremes, subnormals, infinities; ints and big ints
# enforce to floats.
OPERANDS = [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -2.5, 3.0, 7.3, -7.3, 10.0, 100.0, 1.0000000000000002,
            0.49999999999999994, 2.0**53 + 1, 9.223372036854775e18, 2.0**63, -(2.0**63), 1e300, -1e308, 1e-300,
            5e-324, float("inf"), float("-inf"), 3, -4, 2**70]
MATH_OPERATIONS = ["add", "sub", "mul", "div", "pow", "log", "max", "min", "mod", "percent"]
INT64 = range(-(2**63), 2**63)


def python_outcome(schema_id: str, inputs: list[object]) -> object:
    """The Python body's result or exception on enforced inputs."""
    try:
        return node_data(schema_id).run(*inputs)
    except Exception as error:
        return error


def subnormal(value: object) -> bool:
    return isinstance(value, float) and value != 0.0 and abs(value) < sys.float_info.min


def math_declines(op: str, a: float, b: float) -> bool:
    result = python_outcome("chainner:utility:math", [MathOperation(op), a, b])
    return (op in ("pow", "log") and isinstance(result, ValueError)) or (op == "pow" and subnormal(result))


@pytest.mark.parametrize("op", MATH_OPERATIONS)
def test_math_matches(op):
    declined = 0
    for a in OPERANDS:
        for b in OPERANDS:
            declines = math_declines(op, float(a), float(b))
            declined += declines
            assert_native_matches("chainner:utility:math", [op, a, b], declines=declines)
    assert declined < len(OPERANDS) ** 2
    assert (declined > 0) == (op in ("pow", "log"))


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("op", MATH_OPERATIONS)
def test_math_matches_on_random_operands(op, seed):
    rng = np.random.default_rng(seed)
    values = np.concatenate([rng.standard_normal(40) * 10.0 ** rng.integers(-5, 6, 40), rng.integers(-50, 50, 20)])
    for a, b in zip(values, rng.permutation(values)):
        declines = math_declines(op, float(a), float(b))
        assert_native_matches("chainner:utility:math", [op, float(a), float(b)], declines=declines)


@pytest.mark.parametrize(
    "op, a, b, error",
    [
        ("div", 1.0, 0.0, ("ZeroDivisionError", "division by zero")),
        ("div", -0.0, -0.0, ("ZeroDivisionError", "division by zero")),
        ("mod", 3.0, 0.0, ("ZeroDivisionError", "division by zero")),
        ("mod", float("inf"), 2.0, ("OverflowError", "cannot convert float infinity to integer")),
        ("mod", float("-inf"), float("inf"), ("ValueError", "cannot convert float NaN to integer")),
        ("log", 1.0, 5.0, ("ZeroDivisionError", "division by zero")),
    ],
)
def test_math_errors(op, a, b, error):
    assert assert_native_matches("chainner:utility:math", [op, a, b]) == ("error", error)


def test_math_unknown_operation_declines():
    with pytest.raises(graph().NativeDeclined):
        graph().run_native("chainner:utility:math", ["bogus", 1.0, 2.0])


@pytest.mark.parametrize("op", [0, 1, 2, 3, 4, 5])
def test_compare_matches(op):
    for left in OPERANDS:
        for right in OPERANDS:
            assert_native_matches("chainner:utility:compare", [op, left, right])


def test_compare_unknown_operation_is_none():
    assert graph().run_native("chainner:utility:compare", [9, 1.0, 1.0]) == [None]


ROUND_INPUTS = OPERANDS + [0.001, 2.5000000000000004, 1e15 + 0.5, 4503599627370495.5, -9.2233720368547748e18,
                           1.7976931348623157e308, 12345.678]
MULTIPLES = [1e-100, 0.1, 0.25, 1, 3, 7.5, 1e300, float("inf"), 0, -5]
POWERS = [float(np.nextafter(1.0, np.inf)), 1.5, 2, 10, 1e300, float("inf"), 0.5]
ROUND_OPERATIONS = ["Round down", "Round up", "Round"]


def round_declines(a: float, operation: str, scale: str, m: float, p: float) -> bool:
    result = python_outcome("chainner:utility:math_round", [a, RoundOperation(operation), RoundScale(scale), m, p])
    return (
        (isinstance(result, int) and result not in INT64)
        or (isinstance(result, ValueError) and str(result).startswith("expected a positive input"))
        or (isinstance(result, OverflowError) and str(result) == "(34, 'Result too large')")
        or subnormal(result)
    )


def enforced_round(a: float, operation: str, scale: str, m: float, p: float) -> tuple[float, str, str, float, float]:
    """Round's inputs as enforce leaves them (the minimums clamp Multiple and Power; every number is a float)."""
    return float(a), operation, scale, max(float(m), 1e-100), max(float(p), float(np.nextafter(1.0, np.inf)))


@pytest.mark.parametrize("operation", ROUND_OPERATIONS)
@pytest.mark.parametrize(
    "scale, values",
    [("Integer", [(1, 2)]), ("Multiple of...", [(m, 2) for m in MULTIPLES]), ("Power of...", [(1, p) for p in POWERS])],
)
def test_round_matches(operation, scale, values):
    declined = cases = 0
    for a in ROUND_INPUTS:
        for m, p in values:
            declines = round_declines(*enforced_round(a, operation, scale, m, p))
            declined += declines
            cases += 1
            assert_native_matches("chainner:utility:math_round", [a, operation, scale, m, p], declines=declines)
    assert declined < cases
    assert (declined > 0) == (scale != "Multiple of...")


@pytest.mark.parametrize(
    "a, scale, error",
    [
        (float("inf"), "Integer", ("OverflowError", "cannot convert float infinity to integer")),
        (float("-inf"), "Multiple of...", ("OverflowError", "cannot convert float infinity to integer")),
        (float("inf"), "Power of...", ("OverflowError", "cannot convert float infinity to integer")),
    ],
)
def test_round_errors(a, scale, error):
    assert assert_native_matches("chainner:utility:math_round", [a, "Round", scale, 1, 2]) == ("error", error)


@pytest.mark.parametrize("inputs", [[1.0, "bogus", "Integer", 1.0, 2.0], [1.0, "Round", "bogus", 1.0, 2.0]])
def test_round_unknown_option_declines(inputs):
    with pytest.raises(graph().NativeDeclined):
        graph().run_native("chainner:utility:math_round", inputs)
