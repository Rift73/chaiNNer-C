"""Exact scalar/path node contracts against separately frozen original files.

The native code owns application decisions. CPython number parsing/arithmetic,
compiled math, and Path objects remain explicit dependencies. No GPU or timing.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import struct
import sys
import types
import warnings
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from nodes.impl.native_graph import graph

ROOT = Path(__file__).resolve().parents[2]
FROZEN = Path(__file__).with_name("reference_utility_scalar")
NODE_ROOT = "packages/chaiNNer_standard/utility"
FILES = {
    "directory": "directory/directory_go_up.py",
    "math": "math/math.py",
    "round": "math/round.py",
    "parse": "value/parse_number.py",
}
FUNCTIONS = {
    "directory": "directory_go_up_node",
    "math": "math_node",
    "round": "round_node",
    "parse": "parse_number_node",
}


def read_module(path, names, context):
    """Execute only declarations under test; UI decorators remain AST-checked."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    selected: list[ast.stmt] = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    for item in tree.body:
        if isinstance(item, (ast.ClassDef, ast.FunctionDef)) and item.name in names:
            if isinstance(item, ast.FunctionDef):
                item.decorator_list = []
            selected.append(item)
        elif isinstance(item, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names
            for target in item.targets
        ):
            selected.append(item)
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(path), "exec"), context)
    return context


@lru_cache
def namespace(kind, variant="source", current=False):
    base = ROOT / "backend/src" if current else FROZEN / variant
    helper = read_module(
        FROZEN / variant / "nodes/utils/utils.py", {"round_half_up"}, {"math": math}
    )["round_half_up"]
    return read_module(
        base / NODE_ROOT / FILES[kind],
        {
            FUNCTIONS[kind],
            "MathOperation",
            "RoundOperation",
            "RoundScale",
            "_special_mod_numbers",
        },
        {
            "math": math,
            "Enum": Enum,
            "Path": Path,
            "round_half_up": helper,
            "graph": graph,
        },
    )


def capture(callback):
    with warnings.catch_warnings(record=True) as events:
        warnings.simplefilter("always")
        try:
            value, error = callback(), None
        except BaseException as exc:
            value, error = None, exc
    return value, error, [(type(x.message), str(x.message)) for x in events]


def error_signature(error):
    if error is None:
        return None
    return (
        type(error),
        str(error),
        error_signature(error.__cause__),
        error_signature(error.__context__),
        error.__suppress_context__,
    )


def equivalent(expected, actual):
    assert actual[2] == expected[2]
    assert error_signature(actual[1]) == error_signature(expected[1])
    if expected[1] is not None:
        return
    old, new = expected[0], actual[0]
    assert type(new) is type(old)
    if isinstance(old, float):
        assert struct.pack("d", new) == struct.pack("d", old)
    else:
        assert new == old


@pytest.mark.parametrize("variant", ["source", "installed"])
@pytest.mark.parametrize("kind", FILES)
def test_frozen_hash_and_public_contract(variant, kind):
    rel = f"{variant}/{NODE_ROOT}/{FILES[kind]}"
    manifest = json.loads((FROZEN / "sources.json").read_text())
    assert (
        hashlib.sha256((FROZEN / rel).read_bytes()).hexdigest()
        == manifest["sha256"][rel]
    )
    old = ast.parse((FROZEN / rel).read_text())
    new = ast.parse((ROOT / "backend/src" / NODE_ROOT / FILES[kind]).read_text())
    a = next(
        n
        for n in old.body
        if isinstance(n, ast.FunctionDef) and n.name == FUNCTIONS[kind]
    )
    b = next(
        n
        for n in new.body
        if isinstance(n, ast.FunctionDef) and n.name == FUNCTIONS[kind]
    )
    assert ast.dump(a.args) == ast.dump(b.args)
    assert a.returns is not None and b.returns is not None
    assert ast.dump(a.returns) == ast.dump(b.returns)
    assert [ast.dump(d) for d in a.decorator_list] == [
        ast.dump(d) for d in b.decorator_list
    ]
    assert len(b.body) == 1 and isinstance(b.body[0], ast.Return)


PAIR_VALUES = [
    (0.0, 0.0),
    (-0.0, 0.0),
    (0.0, -0.0),
    (-0.0, -0.0),
    (1.0, 2.0),
    (-1.0, 2.0),
    (1.0, -2.0),
    (-1.0, -2.0),
    (2.0, 8.0),
    (8.0, 2.0),
    (-2.0, 0.5),
    (0.0, -1.0),
    (1e308, 2.0),
    (1e-308, 2.0),
    (5e-324, 2.0),
    (float("inf"), 1.0),
    (float("-inf"), 0.0),
    (1.0, float("inf")),
    (float("inf"), float("inf")),
    (float("nan"), 2.0),
    (2.0, float("nan")),
    (True, False),
    (7, 3),
    (-7, 3),
    (7, -3),
    (2**200, 3),
    (-(2**200), 3),
    (Decimal("1.25"), Decimal("2.5")),
    (Fraction(1, 3), Fraction(2, 7)),
    (complex(1, 2), 2),
    (None, 1),
    ("ab", 2),
]


@pytest.mark.parametrize("variant", ["source", "installed"])
@pytest.mark.parametrize("option", list(namespace("math")["MathOperation"].__members__))
@pytest.mark.parametrize("a,b", PAIR_VALUES)
def test_math_values_exact(variant, option, a, b):
    old, new = namespace("math", variant), namespace("math", current=True)
    equivalent(
        capture(lambda: old["math_node"](old["MathOperation"][option], a, b)),
        capture(lambda: new["math_node"](new["MathOperation"][option], a, b)),
    )


@pytest.mark.parametrize("maximum", [True, False])
@pytest.mark.parametrize(
    "a,b", [(0.0, -0.0), (-0.0, 0.0), (float("nan"), 2.0), (2.0, float("nan"))]
)
def test_math_selected_object_identity(maximum, a, b):
    old, new = namespace("math"), namespace("math", current=True)
    mode = "MAXIMUM" if maximum else "MINIMUM"
    expected = old["math_node"](old["MathOperation"][mode], a, b)
    actual = new["math_node"](new["MathOperation"][mode], a, b)
    assert actual is expected


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("same_nan", [False, True])
def test_modulo_nan_tuple_identity(side, same_nan):
    results = []
    for current in (False, True):
        ns = namespace("math", current=current)
        value = ns["_special_mod_numbers"][-1] if same_nan else float("nan")
        args = [2.0, 2.0]
        args[side] = value
        results.append(
            capture(
                lambda ns=ns, args=args: ns["math_node"](
                    ns["MathOperation"].MODULO, *args
                )
            )
        )
    equivalent(*results)


ROUND_VALUES = [
    -2.5,
    -1.5,
    -0.5,
    -0.0,
    0.0,
    0.5,
    1.5,
    2.5,
    math.nextafter(0.5, 0),
    math.nextafter(0.5, 1),
    5e-324,
    1e308,
    2**100,
    float("inf"),
    float("-inf"),
    float("nan"),
    Fraction(3, 2),
    Decimal("1.5"),
    None,
]
ROUND_SCALES = [
    (1.0, 2.0),
    (0.1, 10.0),
    (1e-100, math.nextafter(1.0, math.inf)),
    (0.0, 1.0),
    (-2.0, -2.0),
    (float("inf"), float("inf")),
]


@pytest.mark.parametrize("variant", ["source", "installed"])
@pytest.mark.parametrize("operation", ["FLOOR", "CEILING", "ROUND"])
@pytest.mark.parametrize("scale", ["UNIT", "MULTIPLE", "POWER"])
@pytest.mark.parametrize("a", ROUND_VALUES)
@pytest.mark.parametrize("m,p", ROUND_SCALES)
def test_round_all_branches(variant, operation, scale, a, m, p):
    old, new = namespace("round", variant), namespace("round", current=True)
    equivalent(
        capture(
            lambda: old["round_node"](
                a, old["RoundOperation"][operation], old["RoundScale"][scale], m, p
            )
        ),
        capture(
            lambda: new["round_node"](
                a, new["RoundOperation"][operation], new["RoundScale"][scale], m, p
            )
        ),
    )


@pytest.mark.parametrize(
    "text",
    [
        "0",
        "-0",
        "+123",
        "-123",
        "abc",
        "ABC",
        "10101",
        "0xff",
        "0b10",
        "0o71",
        "1_234",
        "_12",
        "12_",
        "1__2",
        "\uff11\uff12\uff13",
        "١٢٣",
        "\u2003123\u00a0",
        "1.2",
        "1e2",
        "12\x00",
        "",
        "  ",
        "🙂",
        "\ud800",
        b"42",
        bytearray(b"42"),
        42,
        None,
    ],
)
@pytest.mark.parametrize("base", [*range(2, 37), 0, 1, 37, -2, 10.0, None, 2**100])
def test_parse_numbers(text, base):
    old, new = namespace("parse"), namespace("parse", current=True)
    equivalent(
        capture(lambda text=text, base=base: old["parse_number_node"](text, base)),
        capture(lambda text=text, base=base: new["parse_number_node"](text, base)),
    )


def test_parse_arbitrary_precision_and_digit_limit():
    old, new = namespace("parse"), namespace("parse", current=True)
    before = sys.get_int_max_str_digits()
    for text, base in [
        ("f" * 4000, 16),
        ("-" + "1" * 4000, 2),
        ("9" * (max(before, 4300) + 1), 10),
    ]:
        equivalent(
            capture(lambda text=text, base=base: old["parse_number_node"](text, base)),
            capture(lambda text=text, base=base: new["parse_number_node"](text, base)),
        )
    assert sys.get_int_max_str_digits() == before


@pytest.mark.parametrize(
    "path",
    [
        Path("."),
        Path("a/../b"),
        PurePosixPath("/a/é/b"),
        PurePosixPath("a/b"),
        PureWindowsPath("C:/a/🙂/b"),
        PureWindowsPath("C:a/b"),
        PureWindowsPath(r"\\server\share\a\b"),
        PureWindowsPath("C:/"),
    ],
)
@pytest.mark.parametrize("amount", [-2, 0, 1, 2, 8, True, 2.0, None, "2"])
def test_directory_paths(path, amount):
    old, new = namespace("directory"), namespace("directory", current=True)
    expected = capture(lambda: old["directory_go_up_node"](path, amount))
    actual = capture(lambda: new["directory_go_up_node"](path, amount))
    equivalent(expected, actual)
    if type(amount) is int and amount <= 0:
        assert actual[0] is path


def test_directory_parent_protocol_and_large_range():
    def run(current):
        events = []

        class Parent:
            @property
            def parent(self):
                events.append("parent")
                if len(events) == 4:
                    raise LookupError("parent failure")
                return self

        ns = namespace("directory", current=current)
        return capture(lambda: ns["directory_go_up_node"](Parent(), 2**100)), events

    expected, a = run(False)
    actual, b = run(True)
    equivalent(expected, actual)
    assert a == b == ["parent"] * 4


@pytest.mark.parametrize("kind", ["math", "round"])
def test_invalid_enum_and_format_protocol(kind):
    class Unknown:
        def __format__(self, spec):
            assert spec == ""
            return "formatted-operator"

        def __str__(self):
            return "wrong-str"

    old, new = namespace(kind), namespace(kind, current=True)
    if kind == "math":
        calls = [(ns["math_node"], (Unknown(), 1, 2)) for ns in (old, new)]
    else:
        calls = [
            (ns["round_node"], (1, Unknown(), Unknown(), 1, 2)) for ns in (old, new)
        ]
    equivalent(
        *(
            capture(lambda callback=callback, args=args: callback(*args))
            for callback, args in calls
        )
    )
    if kind == "round":
        equivalent(
            *(
                capture(
                    lambda ns=ns: ns["round_node"](
                        1, ns["RoundOperation"].FLOOR, Unknown(), 1, 2
                    )
                )
                for ns in (old, new)
            )
        )


class NotAnException(BaseException):
    pass


@pytest.mark.parametrize("failure", [ValueError, OverflowError, NotAnException])
def test_power_exception_cause_and_handled_state(failure):
    def run(current):
        events = []

        class Power:
            def __pow__(self, other):
                events.append(("pow", other))
                raise failure("original power")

            def __format__(self, spec):
                handled = sys.exc_info()[1]
                events.append(("format", type(handled), str(handled)))
                return "operand"

        ns = namespace("math", current=current)
        try:
            raise LookupError("outer")
        except LookupError as outer:
            result = capture(
                lambda: ns["math_node"](ns["MathOperation"].POWER, Power(), 3)
            )
            assert sys.exc_info()[1] is outer
        return result, events

    expected, old_events = run(False)
    actual, new_events = run(True)
    equivalent(expected, actual)
    assert old_events == new_events


def test_concurrent_independent_nodes():
    def run(current, index):
        m = namespace("math", current=current)
        r = namespace("round", current=current)
        p = namespace("parse", current=current)
        d = namespace("directory", current=current)
        return (
            m["math_node"](m["MathOperation"].MODULO, index - 20, 7),
            r["round_node"](
                index / 7, r["RoundOperation"].ROUND, r["RoundScale"].MULTIPLE, 0.25, 2
            ),
            p["parse_number_node"](f"{index:x}", 16),
            d["directory_go_up_node"](PureWindowsPath("C:/a/b/c"), index % 5),
        )

    expected = [run(False, i) for i in range(80)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        actual = list(pool.map(lambda i: run(True, i), range(80)))
    assert actual == expected


def isolated(kind, current):
    ns = namespace(kind, current=current).copy()
    callback = ns[FUNCTIONS[kind]]
    ns[FUNCTIONS[kind]] = types.FunctionType(callback.__code__, ns)
    helper = ns["round_half_up"]
    ns["round_half_up"] = types.FunctionType(helper.__code__, ns)
    return ns


class TracedNumber:
    def __init__(self, value, events):
        self.value = value
        self.events = events

    def operand(self, value):
        return value.value if isinstance(value, TracedNumber) else value

    def __add__(self, other):
        other = self.operand(other)
        self.events.append(("add", self.value, other))
        return TracedNumber(self.value + other, self.events)

    def __sub__(self, other):
        other = self.operand(other)
        self.events.append(("sub", self.value, other))
        return TracedNumber(self.value - other, self.events)

    def __mul__(self, other):
        other = self.operand(other)
        self.events.append(("mul", self.value, other))
        return TracedNumber(self.value * other, self.events)

    def __rmul__(self, other):
        self.events.append(("rmul", other, self.value))
        return other * self.value

    def __truediv__(self, other):
        other = self.operand(other)
        self.events.append(("div", self.value, other))
        return TracedNumber(self.value / other, self.events)

    def __pow__(self, other):
        other = self.operand(other)
        self.events.append(("pow", self.value, other))
        return self.value**other

    def __mod__(self, other):
        other = self.operand(other)
        self.events.append(("mod", self.value, other))
        return self.value % other

    def __lt__(self, other):
        other = self.operand(other)
        self.events.append(("lt", self.value, other))
        return self.value < other

    def __gt__(self, other):
        other = self.operand(other)
        self.events.append(("gt", self.value, other))
        return self.value > other

    def __float__(self):
        self.events.append(("float", self.value))
        return float(self.value)

    def __floor__(self):
        self.events.append(("floor", self.value))
        return math.floor(self.value)

    def __ceil__(self):
        self.events.append(("ceil", self.value))
        return math.ceil(self.value)


@pytest.mark.parametrize("option", list(namespace("math")["MathOperation"].__members__))
@pytest.mark.parametrize("special", [False, True])
def test_math_numeric_protocol_order(option, special):
    def run(current):
        events = []
        ns = isolated("math", current)
        a, b = TracedNumber(8.5, events), TracedNumber(2.0, events)
        ns["_special_mod_numbers"] = (a,) if special else ()
        value, error, warning = capture(
            lambda: ns["math_node"](ns["MathOperation"][option], a, b)
        )
        if isinstance(value, TracedNumber):
            value = value.value
        return (value, error, warning), events

    expected, first = run(False)
    actual, second = run(True)
    equivalent(expected, actual)
    assert first == second


@pytest.mark.parametrize("operation", ["FLOOR", "CEILING", "ROUND"])
@pytest.mark.parametrize("scale", ["UNIT", "MULTIPLE", "POWER"])
def test_round_numeric_and_primitive_lookup_order(operation, scale):
    def run(current):
        events = []

        class MathPrimitives:
            def __getattr__(self, name):
                events.append(("lookup", name))
                return getattr(math, name)

        ns = isolated("round", current)
        ns["math"] = MathPrimitives()
        result = capture(
            lambda: ns["round_node"](
                TracedNumber(8.5, events),
                ns["RoundOperation"][operation],
                ns["RoundScale"][scale],
                TracedNumber(2.0, events),
                TracedNumber(2.0, events),
            )
        )
        return result, events

    expected, first = run(False)
    actual, second = run(True)
    equivalent(expected, actual)
    assert first == second


@pytest.mark.parametrize("kind", ["math", "round"])
@pytest.mark.parametrize("failure", [None, "comparison", "truth"])
def test_enum_comparison_and_truth_order(kind, failure):
    def run(current):
        events = []

        class Truth:
            def __bool__(self):
                events.append("truth")
                if failure == "truth":
                    raise LookupError("truth failure")
                return False

        class Selector:
            __hash__ = None  # pyright: ignore[reportAssignmentType] -- deliberate: an unhashable fixture; typeshed's unhashable builtins carry the same ignore

            def __eq__(self, other):  # pyright: ignore[reportIncompatibleMethodOverride] -- deliberate contract violation: returns a non-bool truth object, which Python permits
                events.append(other.value)
                if failure == "comparison":
                    raise LookupError("comparison failure")
                return Truth()

            def __format__(self, spec):
                return "selector"

        assert Selector.__hash__ is None
        ns = isolated(kind, current)
        args = (
            (Selector(), 1, 2)
            if kind == "math"
            else (1, Selector(), ns["RoundScale"].UNIT, 1, 2)
        )
        return capture(lambda: ns[FUNCTIONS[kind]](*args)), events

    expected, first = run(False)
    actual, second = run(True)
    equivalent(expected, actual)
    assert first == second


@pytest.mark.parametrize("failure", [False, True])
def test_index_protocols(failure):
    def run(current):
        events = []

        class Index:
            def __index__(self):
                events.append("index")
                if failure:
                    raise LookupError("index failure")
                return 2

        ns = isolated("directory", current)
        directory = capture(lambda: ns["directory_go_up_node"](Path("a/b/c"), Index()))
        ns = isolated("parse", current)
        parsed = capture(lambda: ns["parse_number_node"]("101", Index()))
        return (directory, parsed), events

    expected, first = run(False)
    actual, second = run(True)
    for old, new in zip(expected, actual, strict=True):
        equivalent(old, new)
    assert first == second == ["index", "index"]


def test_power_error_formatting_failure_retains_original_context():
    def run(current):
        class Operand:
            def __pow__(self, other):
                raise ZeroDivisionError("original")

            def __format__(self, spec):
                raise LookupError("format failure")

        ns = isolated("math", current)
        return capture(lambda: ns["math_node"](ns["MathOperation"].POWER, Operand(), 2))

    equivalent(run(False), run(True))


def test_round_half_up_does_not_execute_python_helper():
    ns = isolated("round", True)

    def forbidden(value):
        raise AssertionError("original Python helper executed")

    ns["round_half_up"] = forbidden
    assert (
        ns["round_node"](-1.5, ns["RoundOperation"].ROUND, ns["RoundScale"].UNIT, 1, 2)
        == -1
    )


def test_all_reference_files_remain_frozen():
    manifest = json.loads((FROZEN / "sources.json").read_text())
    assert len(manifest["sha256"]) == 10
    for relative, expected in manifest["sha256"].items():
        assert hashlib.sha256((FROZEN / relative).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("operation", ["ADD", "POWER", "MAXIMUM", "MODULO"])
def test_numeric_subclasses(operation):
    class Integer(int):
        def __add__(self, other):  # pyright: ignore[reportIncompatibleMethodOverride] -- deliberate contract violation: int.__add__ returns a complex, a legal runtime result
            return complex(super().__add__(other), 4)

        def __pow__(self, other, modulo=None):  # pyright: ignore[reportIncompatibleMethodOverride] -- deliberate contract violation: int.__pow__ returns a bool, kept without coercion
            return True

    class Floating(float):
        def __mod__(self, other):  # pyright: ignore[reportIncompatibleMethodOverride] -- deliberate contract violation: float.__mod__ returns a Fraction, a legal runtime result
            return Fraction(3, 7)

    old, new = namespace("math"), namespace("math", current=True)
    for a, b in [(Integer(7), 3), (Floating(-0.0), -2.0), (Floating(7.0), 2.0)]:
        equivalent(
            capture(
                lambda a=a, b=b: old["math_node"](old["MathOperation"][operation], a, b)
            ),
            capture(
                lambda a=a, b=b: new["math_node"](new["MathOperation"][operation], a, b)
            ),
        )


def test_parse_string_and_base_subclasses():
    class Text(str):
        def __str__(self):
            raise AssertionError("must not stringify text")

    class Base(int):
        def __index__(self):
            raise AssertionError("must preserve int-subclass base handling")

    old, new = namespace("parse"), namespace("parse", current=True)
    equivalent(
        capture(lambda: old["parse_number_node"](Text("101"), Base(2))),
        capture(lambda: new["parse_number_node"](Text("101"), Base(2))),
    )


def test_power_accepts_isinstance_class_protocol_and_keeps_identity():
    class Result:
        @property
        def __class__(self):  # pyright: ignore[reportIncompatibleMethodOverride] -- deliberate contract violation: a getter-only __class__ for isinstance's class protocol
            return int

    result = Result()

    class Operand:
        def __pow__(self, other):
            return result

    for current in (False, True):
        ns = namespace("math", current=current)
        assert ns["math_node"](ns["MathOperation"].POWER, Operand(), 2) is result
