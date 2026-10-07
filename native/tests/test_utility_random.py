"""Exact RNG contracts against complete frozen installed/source node bodies.

Only these tests compile the reference AST. Production never executes old bodies.
The retained compiled MT core is checked by its full state after every draw.
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import struct
import types
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nodes.impl.native_graph import graph
from nodes.impl.native_utility_random import (
    derive_seed,
    random_number,
    seed_to_bytes,
)
from nodes.utils.seed import Seed

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).with_name("reference_utility_random")
NODE_ROOT = ROOT / "backend/src/packages/chaiNNer_standard/utility/random"


def load_functions(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = []
    for item in tree.body:
        if isinstance(item, ast.FunctionDef) and item.name in {
            "_to_bytes",
            "derive_seed_node",
            "random_number_node",
        }:
            item.decorator_list = []
            body.append(item)
    namespace = {
        "hashlib": hashlib,
        "struct": struct,
        "Random": random.Random,
        "Seed": Seed,
        "Source": int | float | str | Seed,
        "derive_seed": derive_seed,
        "random_number": random_number,
        "seed_to_bytes": seed_to_bytes,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


ORACLES = [
    load_functions(REFERENCE / kind / "derive_seed.py")
    | load_functions(REFERENCE / kind / "random_number.py")
    for kind in ("installed", "source")
]
CURRENT = load_functions(NODE_ROOT / "derive_seed.py") | load_functions(
    NODE_ROOT / "random_number.py"
)


def exception_snapshot(error: BaseException | None, depth: int = 0):
    if error is None or depth > 5:
        return None
    return (
        type(error).__name__,
        str(error),
        exception_snapshot(error.__cause__, depth + 1),
        exception_snapshot(error.__context__, depth + 1),
        error.__suppress_context__,
    )


def result_snapshot(value: object):
    if isinstance(value, Seed):
        return (type(value).__name__, value.value)
    if isinstance(value, float):
        return (type(value).__name__, struct.pack("d", value))
    return (type(value).__name__, value)


def outcome(
    call: Callable[[], object],
) -> tuple[tuple[str, Any], list[tuple[str, str]]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = ("ok", result_snapshot(call()))
        except Exception as error:
            result = ("error", exception_snapshot(error))
    return result, [(item.category.__name__, str(item.message)) for item in caught]


def check(name: str, *args: object):
    expected = outcome(lambda: ORACLES[0][name](*args))
    assert outcome(lambda: ORACLES[1][name](*args)) == expected
    assert outcome(lambda: CURRENT[name](*args)) == expected
    return expected


@pytest.mark.parametrize("bits", range(257))
@pytest.mark.parametrize("sign", [-1, 1])
def test_signed_integer_serialization_boundaries(bits, sign):
    for delta in (-1, 0, 1):
        value = sign * ((1 << bits) + delta)
        check("_to_bytes", value)
        check("_to_bytes", Seed(value))


@pytest.mark.parametrize(
    "value",
    [
        "",
        "ASCII\0bytes",
        "café",
        "中🙂𐐀",
        "e\u0301",
        "\ud800",
        "\udfff",
        "\ud800\udfff",
        True,
        False,
        0.0,
        -0.0,
        1.0,
        -128.0,
        128.0,
        1.25,
        -0.125,
        2.0**-1074,
        -(2.0**-1074),
        float.fromhex("0x1.fffffffffffffp+1023"),
        float("nan"),
        float("inf"),
        -float("inf"),
        None,
        [],
        {},
        b"123",
        b"abc",
        Path("folder"),
        np.int64(123),
        np.uint64(2**64 - 1),
        np.float32(0.25),
        np.float64(-0.125),
        Decimal("1.125"),
        Fraction(-7, 8),
        Seed(0.5),  # pyright: ignore[reportArgumentType] -- deliberate contract violation: Seed.value is int; the test pins a non-int seed
        Seed("17"),  # pyright: ignore[reportArgumentType] -- deliberate contract violation: Seed.value is int; the test pins a non-int seed
        Seed(None),  # pyright: ignore[reportArgumentType] -- deliberate contract violation: Seed.value is int; the test pins a non-int seed
    ],
)
def test_serialization_values_and_errors(value):
    check("_to_bytes", value)


@pytest.mark.parametrize("seed", [0, -1, 2**32, -(2**64), 2**257 + 73, -(2**8192 + 91)])
@pytest.mark.parametrize(
    "sources",
    [
        (),
        (None,),
        (None, None, None),
        ("",),
        (0,),
        (False,),
        (-0.0,),
        (1.25,),
        ("a", "bc"),
        ("ab", "c"),
        ("é", None, "🙂", -131, 0.75),
        tuple(range(16)),
        (Seed(-(2**521)), "\ud800"),
        (float("nan"),),
        (Path("a"),),
    ],
)
def test_derive_order_serialization_and_noop(seed, sources):
    value = Seed(seed)
    check("derive_seed_node", value, *sources)
    if all(source is None for source in sources):
        assert CURRENT["derive_seed_node"](value, *sources) is value


@pytest.mark.parametrize(
    "seed", [0, 1, -1, 42, 2**32 - 1, 2**64, -(2**64), 2**521 + 1, -(2**8192 + 719)]
)
@pytest.mark.parametrize(
    "bounds",
    [
        (0, 0),
        (0, 1),
        (-1, 1),
        (-83, -17),
        (0, 100),
        (10, 9),
        (-(2**64), 2**64),
        (2**1024, 2**1024 + 257),
        (-(2**4096), 2**4096),
    ]
    + [
        (0, (1 << n) + delta)
        for n in (2, 31, 32, 33, 63, 64, 65, 127)
        for delta in (-2, -1, 0)
    ],
)
def test_random_number_arbitrary_seed_range(seed, bounds):
    check("random_number_node", *bounds, Seed(seed))


@pytest.mark.parametrize(
    "bounds",
    [
        (0.0, 100.0),
        (1.5, 3),
        (1, 2.5),
        (float("nan"), 3),
        (0, float("nan")),
        (float("inf"), 0),
        (0, float("inf")),
        ("1", 3),
        (0, "3"),
        (None, 3),
        (0, None),
        (True, False),
        ([], 3),
        (0, []),
        (1j, 3),
        (0, 1j),
        (np.int64(-3), np.int64(13)),
        (np.float64(1), np.float32(10)),
        (np.float64(1.5), 10),
    ],
)
def test_direct_invalid_ranges_warnings_and_context(bounds):
    check("random_number_node", *bounds, Seed(17))


@pytest.mark.parametrize(
    "seed",
    [
        0,
        -1,
        2**4096 + 91,
        -(2**777),
        True,
        0.125,
        -7.5,
        "",
        "é🙂",
        b"",
        b"bytes\0",
        bytearray(b"test"),
    ],
)
def test_retained_core_state_and_exact_rejection_consumption(seed):
    original = random.Random(seed)
    native = graph().utility_random_core(seed)
    assert native.getstate() == original.getstate()[1]
    for width in (1, 2, 3, 4, 7, 8, 9, 2**32, 2**32 + 1, 2**127 - 1, 2**257):
        assert graph().utility_random_inclusive(
            native, -37, width - 38
        ) == original.randint(-37, width - 38)
        assert native.getstate() == original.getstate()[1]


@pytest.mark.parametrize(
    "seed",
    [
        "abc",
        "é🙂",
        "\ud800",
        b"abc",
        bytearray(b"abc"),
        0.125,
        float("inf"),
        -float("inf"),
        [],
        {},
        object(),
        np.int64(1),
    ],
)
def test_direct_seed_wrapper_compatibility(seed):
    check("random_number_node", -4, 7, Seed(seed))


class Bits:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = []

    def getrandbits(self, width):
        self.calls.append(width)
        return next(self.values)


@pytest.mark.parametrize(
    "width,values",
    [
        (1, [1, 1, 0]),
        (2, [3, 2, 1]),
        (4, [7, 4, 0]),
        (5, [7, 6, 5, 4]),
        (2**72, [2**73 - 1, 2**72, 19]),
    ],
)
def test_control_rejection_trajectory_and_bit_length(width, values):
    a, b = Bits(values), Bits(values)
    # The exact pinned rejection oracle.
    expected = random.Random._randbelow_with_getrandbits(a, width) - 15  # pyright: ignore[reportAttributeAccessIssue] -- CPython's private method, which typeshed does not declare, is the pinned oracle
    assert graph().utility_random_inclusive(b, -15, width - 16) == expected
    assert a.calls == b.calls == [width.bit_length()] * len(values)


def test_global_rng_unchanged_and_concurrent_repeatability():
    state = random.getstate()
    inputs = list(range(128))
    expected = [
        (
            ORACLES[0]["random_number_node"](-(2**79), 2**113, Seed(i)),
            ORACLES[0]["derive_seed_node"](Seed(i), "é", None, i / 7).value,
        )
        for i in inputs
    ]

    def run(i):
        return (
            random_number(-(2**79), 2**113, Seed(i)),
            derive_seed(Seed(i), ("é", None, i / 7)).value,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(run, inputs)) == expected
        assert list(pool.map(run, inputs)) == expected
    assert random.getstate() == state


def test_metadata_and_seed_helper_unchanged():
    for name in ("derive_seed", "random_number"):
        original = ast.parse(
            (REFERENCE / "source" / (name + ".py")).read_text(encoding="utf-8")
        )
        converted = ast.parse((NODE_ROOT / (name + ".py")).read_text(encoding="utf-8"))
        a = next(
            node
            for node in original.body
            if isinstance(node, ast.FunctionDef) and node.name == name + "_node"
        )
        b = next(
            node
            for node in converted.body
            if isinstance(node, ast.FunctionDef) and node.name == name + "_node"
        )
        assert ast.dump(a.args) == ast.dump(b.args)
        assert a.returns is not None and b.returns is not None
        assert ast.dump(a.returns) == ast.dump(b.returns)
        assert [ast.dump(node) for node in a.decorator_list] == [
            ast.dump(node) for node in b.decorator_list
        ]
    # seed.py is upstream's, unchanged. backend/src keeps upstream's GitHub LF, while
    # the frozen copy holds the installed app's bytes (CRLF), so the line endings
    # are compared as LF on both sides.
    assert (ROOT / "backend/src/nodes/utils/seed.py").read_bytes().replace(
        b"\r\n", b"\n"
    ) == (REFERENCE / "source/seed.py").read_bytes().replace(b"\r\n", b"\n")
    for record in json.loads((REFERENCE / "manifest.json").read_text())["files"]:
        assert (
            hashlib.sha256((REFERENCE / record["snapshot"]).read_bytes()).hexdigest()
            == record["sha256"]
        )


def test_no_old_python_random_algorithms(monkeypatch):
    expected = (
        random_number(0, 100, Seed(73)),
        derive_seed(Seed(-97), ("a", 18)).value,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Old Python RNG algorithm invoked")

    for name in ("seed", "randrange", "randint", "_randbelow"):
        monkeypatch.setattr(random.Random, name, forbidden)
    monkeypatch.setattr(Seed, "from_bytes", forbidden)
    assert (
        random_number(0, 100, Seed(73)),
        derive_seed(Seed(-97), ("a", 18)).value,
    ) == expected


def test_direct_object_protocol_order_and_exception_context():
    def attempt(port):
        log = []

        class Numeric:
            __hash__ = None  # pyright: ignore[reportAssignmentType] -- deliberate: an unhashable fixture; typeshed's unhashable builtins carry the same ignore

            def __int__(self):
                log.append("int")
                return 1

            def __eq__(self, other):
                log.append(("eq", other))
                return False

            def __float__(self):
                log.append("float")
                return 0.75

        assert Numeric.__hash__ is None
        function = seed_to_bytes if port else ORACLES[0]["_to_bytes"]
        return outcome(lambda: function(Numeric())), log

    assert attempt(True) == attempt(False)


def test_seed_attribute_precedes_range_errors():
    # Constructing the local RNG occurs before randint evaluates maximum+1.
    for seed in (object(), types.SimpleNamespace(value=[])):
        check("random_number_node", None, None, seed)


def test_no_sources_preserves_any_original_identity():
    for seed in (object(), Seed(-73), None):
        for sources in ((), (None,), (None,) * 20):
            assert CURRENT["derive_seed_node"](seed, *sources) is seed


def test_custom_index_int_and_encode_protocols():
    class Encoded(str):
        def encode(self, *args, **kwargs):
            return b"explicit encoding override"

    class Index:
        def __index__(self):
            return 7

    class IntegralFloat(float):
        def __int__(self):
            return 5

    for value in (Encoded("different"), Index(), IntegralFloat(5.0)):
        check("_to_bytes", value)
    check("random_number_node", Index(), 19, Seed(31))


def test_silent_string_seed_encode_protocol_retained():
    class Encoded(str):
        def encode(self, *args, **kwargs):
            return b"seed bytes override"

    check("random_number_node", 0, 2**129, Seed(Encoded("other")))  # pyright: ignore[reportArgumentType] -- deliberate contract violation: Seed.value is int; the test pins a non-int seed


@pytest.mark.parametrize("kind", [str, int, float, Seed])
def test_isinstance_class_protocol_order(kind):
    def call(port):
        log = []

        class Proxy:
            __hash__ = None  # pyright: ignore[reportAssignmentType] -- deliberate: an unhashable fixture; typeshed's unhashable builtins carry the same ignore

            @property
            def __class__(self):  # pyright: ignore[reportIncompatibleMethodOverride] -- deliberate contract violation: a getter-only __class__ for isinstance's class protocol
                log.append("class")
                return kind

            @property
            def value(self):
                log.append("value")
                return 41

            def encode(self, **kwargs):
                log.append(("encode", kwargs))
                return b"encoded proxy"

            def __int__(self):
                log.append("int")
                return 5

            def __eq__(self, other):
                log.append(("eq", other))
                return True

        assert Proxy.__hash__ is None
        source = Proxy()
        function = seed_to_bytes if port else ORACLES[0]["_to_bytes"]
        return outcome(lambda: function(source)), log

    assert call(True) == call(False)


def test_proxy_string_rng_seed_protocol():
    class Proxy:
        @property
        def __class__(self):  # pyright: ignore[reportIncompatibleMethodOverride] -- deliberate contract violation: a getter-only __class__ for isinstance's class protocol
            return str

        def encode(self):
            return b"string seed proxy"

    check("random_number_node", 0, 2**73, Seed(Proxy()))  # pyright: ignore[reportArgumentType] -- deliberate contract violation: Seed.value is int; the test pins a non-int seed


@pytest.mark.parametrize("minimum", [5, 0, -1, 5.0, 5.5, 2**129])
def test_direct_maximum_add_none_uses_single_bound_range(minimum):
    class Maximum(int):
        def __add__(self, other):  # pyright: ignore[reportIncompatibleMethodOverride] -- deliberate contract violation: int.__add__ returns None to select the single-bound range
            assert other == 1

    check("random_number_node", minimum, Maximum(7), Seed(17))


@pytest.mark.parametrize(
    "failure",
    [TypeError, ValueError, OverflowError, LookupError, KeyboardInterrupt, SystemExit],
)
@pytest.mark.parametrize("outer_context", [False, True])
def test_struct_double_packer_masks_every_float_conversion_error(
    failure, outer_context
):
    class Numeric:
        __hash__ = None  # pyright: ignore[reportAssignmentType] -- deliberate: an unhashable fixture; typeshed's unhashable builtins carry the same ignore

        def __int__(self):
            return 0

        def __eq__(self, other):
            return False

        def __float__(self):
            raise failure("original float conversion failure")

    assert Numeric.__hash__ is None

    def call(namespace):
        try:
            if outer_context:
                raise ArithmeticError("outer handled exception")
        except ArithmeticError:
            return outcome(lambda: namespace["_to_bytes"](Numeric()))
        return outcome(lambda: namespace["_to_bytes"](Numeric()))

    expected = call(ORACLES[0])
    assert expected[0][0] == "error" and expected[0][1][0] == "error"
    assert call(ORACLES[1]) == expected
    assert call(CURRENT) == expected


@pytest.mark.parametrize("where", ["start", "stop"])
def test_deprecation_warning_as_error_preserves_context(where):
    # CPython 3.11's randrange warned (DeprecationWarning) on a float bound; 3.12+
    # randint rejects it through operator.index and warns nothing, so there is no
    # warning to turn into an error: both bodies raise the same TypeError.
    values = (1.0, 3) if where == "start" else (1, 3.0)
    outputs = [
        outcome(lambda node=namespace["random_number_node"]: node(*values, Seed(91)))
        for namespace in (ORACLES[0], CURRENT)
    ]
    assert outputs[0] == outputs[1]
    (status, error), caught = outputs[0]
    assert status == "error" and error[0] == "TypeError" and caught == []
