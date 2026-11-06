"""Math Expression's evaluator: right values in floats alone, an unset input as no value at all, and every form of
text it must refuse, cleanly and quickly, whatever the text asks for."""

from __future__ import annotations

import math
import time

import pytest

from nodes.properties.inputs import NumberInput, SliderInput
from packages.chaiNNer_standard.utility import math_group
from packages.chaiNNer_standard.utility.math.math_expression import (
    MAX_EXPRESSION_LENGTH,
    math_expression_node,
)

NODE = next(
    node
    for node in math_group.nodes
    if node.schema_id == "chainner:utility:math_expression"
)


def run(expression: str, **variables: float) -> float:
    values = [variables.get(chr(ord("a") + i)) for i in range(26)]
    return math_expression_node(expression, *values)


@pytest.mark.parametrize(
    "expression,variables,expected",
    [
        ("sqrt(a**2 + b**2) * c / 100", {"a": 3, "b": 4, "c": 50}, 2.5),
        ("a + b * c", {"a": 1, "b": 2, "c": 3}, 7),
        ("(a + b) * c", {"a": 1, "b": 2, "c": 3}, 9),
        ("2 ** 3 ** 2", {}, 512),
        ("-2 ** 2", {}, -4),
        ("7 // 2", {}, 3),
        ("-7 // 2", {}, -4),
        ("7 % 3", {}, 1),
        ("-7 % 3", {}, 2),
        ("7 / 2", {}, 3.5),
        ("+a", {"a": 2}, 2),
        ("abs(-3)", {}, 3),
        ("floor(2.5) + ceil(2.5)", {}, 5),
        ("round(2.5) + round(3.5)", {}, 6),
        ("round(1.234, 2)", {}, 1.23),
        ("round(1.235, a)", {"a": 1}, 1.2),
        ("round(1234.5, -2)", {}, 1200),
        ("round(2.5, 0)", {}, 2),
        ("round(1.5, 1e300)", {}, 1.5),
        ("min(3, 1, 2) + max(3, 1, 2)", {}, 4),
        ("pow(2, 10)", {}, 1024),
        ("log(e)", {}, 1),
        ("log(8, 2)", {}, 3),
        ("log10(1000) + log2(8)", {}, 6),
        ("exp(0)", {}, 1),
        ("degrees(pi)", {}, 180),
        ("radians(180)", {}, math.pi),
        ("sin(0) + cos(0) + tan(0)", {}, 1),
        ("asin(1) + acos(1) + atan(1)", {}, math.pi * 3 / 4),
        ("atan2(1, 1)", {}, math.pi / 4),
        ("sinh(0) + cosh(0) + tanh(0)", {}, 1),
        ("z - a", {"a": 1, "z": 26}, 25),
        ("1_000 + 0x10 + 1e3", {}, 2016),
        (" a + 1 ", {"a": 1}, 2),
        ("(a +\n 1)", {"a": 1}, 2),
    ],
)
def test_values(expression: str, variables: dict[str, float], expected: float):
    result = run(expression, **variables)
    assert type(result) is float
    assert result == pytest.approx(expected, rel=1e-15)


def test_a_constant_expression_needs_no_input():
    assert run("1 + 1") == 2


def test_an_empty_input_has_no_value():
    assert run("a + c", a=1, c=2) == 3
    with pytest.raises(ValueError, match="Name 'b' is not defined"):
        run("a + b", a=1)
    # Zero is a value.
    assert run("a + b", a=1, b=0) == 1


def test_e_is_euler_until_its_input_has_a_value():
    assert run("e") == math.e
    assert run("e", e=2) == 2
    assert run("e", e=0) == 0
    assert run("pi") == math.pi


@pytest.mark.parametrize(
    "expression",
    [
        # Reaching into Python.
        "a.real",
        "(1).__class__",
        "().__class__.__bases__[0].__subclasses__()",
        "abs.__self__",
        "__import__('os')",
        "__builtins__",
        "open('file')",
        "eval('1')",
        "exec('1')",
        "getattr(a, 'real')",
        "(lambda: 1)()",
        "sqrt.__call__(4)",
        "self",
        # Other values than numbers.
        "'text'",
        "b'bytes'",
        "f'{a}'",
        "None",
        "True",
        "False",
        "...",
        "1j",
        "[1, 2]",
        "(1, 2)",
        "{1: 2}",
        "{1}",
        "[1, 2][0]",
        "[x for x in (1, 2)]",
        "(x for x in (1, 2))",
        # Other operators than those it lists.
        "1 < 2",
        "1 == 1",
        "1 and 2",
        "1 or 2",
        "not 1",
        "~1",
        "1 << 2",
        "1 >> 2",
        "1 & 2",
        "1 | 2",
        "2 ^ 2",
        "1 @ 2",
        "1 if 1 else 2",
        "(x := 1)",
        # Calls it must not read loosely.
        "pi()",
        "a(1)",
        "sqrt",
        "sqrt + 1",
        "max(*[1, 2])",
        "max(**{})",
        "max(1, 2, key=abs)",
        "round(1.5, ndigits=1)",
        "sqrt()",
        "sqrt(1, 2)",
        "min()",
        "min(1)",
        "round()",
        "round(1.234, 0.5)",
        "round(1.234, 1e999)",
        "round(1.234, 2, 3)",
        "pow(2, 3, 5)",
        # Not an expression.
        "",
        " ",
        "1 +",
        "(",
        "1 2",
        "a = 1",
        "import os",
        chr(0),
        "1" + chr(0),
        chr(0xD800),
    ],
)
def test_refused(expression: str):
    with pytest.raises(ValueError, match="Error evaluating expression"):
        run(expression, a=1)


@pytest.mark.parametrize(
    "expression",
    [
        "9**9**9",
        "9**9**9**9**9",
        "10**400",
        "2.0**1024",
        "(-10)**401",
        "pow(10, 400)",
        "exp(1000)",
        "cosh(1000)",
        "1/0",
        "1//0",
        "1%0",
        "0**-1",
        "pow(0, -1)",
        "(-8)**(1/3)",
        "pow(-8, 1/3)",
        "sqrt(-1)",
        "log(0)",
        "asin(2)",
        "floor(1e999)",
        "ceil(1e999)",
        "round(1e999)",
        "floor(1e999 - 1e999)",
        "1" + "0" * 400,
        "1" + "0" * 4000,
        "0x" + "f" * 4000,
    ],
)
def test_arithmetic_that_has_no_answer_raises(expression: str):
    start = time.perf_counter()
    with pytest.raises(ValueError, match="Error evaluating expression"):
        run(expression)
    assert time.perf_counter() - start < 1


def test_integers_given_to_it_are_floats_before_a_huge_power():
    # The server's inputs are floats. If an integer ever arrives, 9**1000000000 raises at once
    # instead of computing an integer of a billion digits.
    start = time.perf_counter()
    with pytest.raises(ValueError, match="Error evaluating expression"):
        run("a ** b", a=9, b=1000000000)
    assert time.perf_counter() - start < 1
    assert type(run("a", a=9)) is float


@pytest.mark.parametrize(
    "expression,result",
    [
        ("1e308 * 10", "inf"),
        ("1e999", "inf"),
        ("-1e999", "-inf"),
        ("degrees(1e308)", "inf"),
        ("round(1e999, 2)", "inf"),
        ("a", "inf"),
    ],
)
def test_an_answer_that_is_not_finite_raises(expression: str, result: str):
    with pytest.raises(ValueError, match=f"The result is {result}, not a finite"):
        run(expression, a=math.inf)


@pytest.mark.parametrize(
    "expression",
    [
        "a - a",
        "0 * a",
        "a % 2",
        "a // 2",
        "log(a, a)",
        "max(0, a - a)",
        "max(a - a, 0)",
        "min(0, a - a)",
        "min(a - a, 0)",
        "1 + max(0, a - a, 2)",
    ],
)
def test_a_nan_on_the_way_raises_whatever_the_order(expression: str):
    with pytest.raises(ValueError, match="A step of the expression is not a number"):
        run(expression, a=math.inf)


def test_a_nan_input_raises_whatever_the_order():
    for expression in ("a", "max(0, a)", "max(a, 0)", "min(0, a)", "min(a, 0)"):
        with pytest.raises(ValueError, match="is not a number"):
            run(expression, a=math.nan)


def test_infinities_may_pass_on_the_way_to_a_finite_answer():
    assert run("min(1e999, 2)") == 2
    assert run("1 / 1e999") == 0
    assert run("atan(1e999)") == math.pi / 2


@pytest.mark.parametrize(
    "expression",
    [
        "(" * 150 + "1" + ")" * 150,
        "-" * 300 + "1",
        "1" + "+1" * 300,
        "abs(" * 50 + "1" + ")" * 50,
        "max(" + "1," * 2000 + "1)",
        "1" + " " * (MAX_EXPRESSION_LENGTH - 1),
    ],
)
def test_deep_and_long_within_the_limit_still_compute(expression: str):
    assert type(run(expression)) is float


@pytest.mark.parametrize(
    "expression",
    [
        "(" * 1000 + "1" + ")" * 1000,
        "abs(" * 500 + "1" + ")" * 500,
        "-" * 1500 + "1",
        "1" + "+1" * 1500,
        "1" + "**1" * 1000,
    ],
)
def test_too_deep_raises(expression: str):
    with pytest.raises(ValueError, match="Error evaluating expression"):
        run(expression)


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param("1" + " " * MAX_EXPRESSION_LENGTH, id="one too many"),
        pytest.param("1" + "+1" * 10_000_000, id="20 MB of sums"),
        pytest.param("-" * 100_000_000 + "1", id="100 MB of signs"),
    ],
)
def test_too_long_is_refused_without_parsing_and_without_all_of_its_text(
    expression: str,
):
    start = time.perf_counter()
    with pytest.raises(ValueError, match="longer than 4096 characters") as refused:
        run(expression)
    assert time.perf_counter() - start < 1
    assert len(str(refused.value)) < 400


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param("x" * MAX_EXPRESSION_LENGTH, id="a long name"),
        pytest.param("x" * 4000 + "(1)", id="a long function"),
        pytest.param("1" + "0" * 4095, id="a long number"),
        pytest.param("1 +" + " " * 4000, id="long but invalid"),
    ],
)
def test_an_error_stays_short_within_the_limit(expression: str):
    with pytest.raises(ValueError, match="Error evaluating expression") as refused:
        run(expression)
    assert len(str(refused.value)) < 500


def test_the_schema_labels_and_empties_the_inputs_it_binds():
    expression, *letters = NODE.inputs
    assert expression.label == "Expression"
    assert [i.label for i in letters] == [chr(ord("a") + i) for i in range(26)]
    assert [i.id for i in letters] == list(range(1, 27))
    for number in letters:
        assert number.optional
        assert number.to_dict()["def"] is None
        assert number.enforce_(None) is None
        assert number.enforce_(3) == 3.0
        assert type(number.enforce_(3)) is float


def test_only_the_number_field_can_be_optional():
    optional = NumberInput("N").make_optional()
    assert optional.optional
    assert optional.default is None
    assert optional.to_dict()["def"] is None
    assert optional.enforce_(None) is None
    assert optional.enforce_(2.6) == 3
    with pytest.raises(ValueError, match="cannot be made optional"):
        SliderInput("S").make_optional()


def test_a_required_number_refuses_nothing_new_and_accepts_nothing_new():
    required = NumberInput("N", default=4, min=1, max=9, precision=1)
    assert not required.optional
    assert required.default == 4
    assert required.to_dict()["def"] == 4
    assert required.enforce_(100) == 9.0
    assert required.enforce_(0.04) == 1.0
    with pytest.raises(AssertionError, match="Expected value to exist"):
        required.enforce_(None)
    with pytest.raises(ValueError, match="NaN"):
        required.enforce_(math.nan)
