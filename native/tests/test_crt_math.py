"""The CRT math table of chainner_native.dll (cn_crt_math.h; Consult 6 D-4).

Group-1 mirrors call cn_crt, which the DLL fills from the process's ucrtbase.dll as it
loads; the table holding exactly those exports is checked here. The comparison of each
entry with the build SDK's static UCRT, with the CRT's FMA3 path on and off, is
informational (STATUS's SP5a item): a difference means this machine's ucrtbase.dll moved
away from the static UCRT, so the tracked goldens may lag the live oracle. The
live-NumPy assertions are the contract.
"""

import ctypes as ct
import warnings

import numpy as np
import pytest

from nodes.impl import native
from nodes.impl.native import check, lib

# cn_crt_table's members in order: member, ucrtbase.dll export, arguments, float32.
ENTRIES = (
    ("cos", "cos", 1, False),
    ("sin", "sin", 1, False),
    ("exp", "exp", 1, False),
    ("log", "log", 1, False),
    ("log1p", "log1p", 1, False),
    ("log2", "log2", 1, False),
    ("atan2", "atan2", 2, False),
    ("hypot", "hypot", 2, False),
    ("pow", "pow", 2, False),
    ("cabs", "_cabs", 2, False),
    ("expf", "expf", 1, True),
    ("logf", "logf", 1, True),
    ("powf", "powf", 2, True),
    ("log10f", "log10f", 1, True),
    ("atan2f", "atan2f", 2, True),
    ("hypotf", "_hypotf", 2, True),
    ("asinf", "asinf", 1, True),
    ("cosf", "cosf", 1, True),
    ("sinf", "sinf", 1, True),
)
COUNT = 1 << 14
SPECIALS = (
    0.0,
    -0.0,
    np.inf,
    -np.inf,
    np.nan,
    5e-324,
    -1e-310,
    1e-30,
    0.5,
    1.0,
    -1.0,
    2.0,
    np.pi / 2,
    np.pi,
    2 * np.pi,
    1e22,
    2.0**63,
    -3e38,
)
DOUBLES = ct.POINTER(ct.c_double)


class UcrtMovedWarning(RuntimeWarning):
    """This machine's ucrtbase.dll differs from the build SDK's static UCRT."""


@pytest.fixture(scope="module")
def probe_function():
    function = lib().cn_crt_probe
    function.argtypes = [
        ct.c_char_p,
        ct.c_int,
        DOUBLES,
        DOUBLES,
        ct.c_size_t,
        DOUBLES,
        DOUBLES,
        ct.POINTER(ct.c_int),
    ]
    function.restype = ct.c_int
    return function


@pytest.fixture(scope="module")
def probe(probe_function):
    def run(member, fma3, x, y):
        static, table, used = np.empty(x.size), np.empty(x.size), (ct.c_int * 2)()
        second = None if y is None else y.ctypes.data_as(DOUBLES)
        check(
            probe_function(
                member.encode(),
                fma3,
                x.ctypes.data_as(DOUBLES),
                second,
                x.size,
                static.ctypes.data_as(DOUBLES),
                table.ctypes.data_as(DOUBLES),
                used,
            )
        )
        return static, table, tuple(used)

    return run


def signed(rng, low, high):
    """Random signs times 2**uniform(low, high): every magnitude band of a range."""
    return rng.choice([-1.0, 1.0], COUNT) * np.exp2(rng.uniform(low, high, COUNT))


def probe_set(index: int):
    """The fixed inputs of one entry: its dense range, the bands where the CRT's FMA3
    and SSE2 paths are measured to differ (cos, sin, exp, atan2, log10f, asinf on this
    machine; the counts print with -s), wide magnitudes and the special values."""
    member, _, arguments, single = ENTRIES[index]
    rng = np.random.default_rng(20261006 + index)
    # Binary exponents of the widest finite band: below the format's largest finite value.
    low, high = (-149, 127.99) if single else (-1074, 1023.99)
    near_one = 1 + rng.uniform(-1e-3, 1e-3, COUNT)
    ranges = {
        "cos": [rng.uniform(-32, 32, COUNT), signed(rng, -40, high)],
        "exp": [rng.uniform(-746, 710, COUNT), rng.uniform(-2, 2, COUNT)],
        "log": [np.exp2(rng.uniform(low, high, COUNT)), near_one],
        "log1p": [rng.uniform(-1, 1, COUNT), np.exp2(rng.uniform(-60, high, COUNT))],
        "atan2": [signed(rng, -30, 30), signed(rng, low, high)],
        "pow": [np.exp2(rng.uniform(-12, 12, COUNT)), near_one],
        "expf": [rng.uniform(-104, 89, COUNT), rng.uniform(-2, 2, COUNT)],
        "asinf": [rng.uniform(-1, 1, COUNT), rng.uniform(-1.001, 1.001, COUNT)],
        "cosf": [
            rng.uniform(-32, 32, COUNT),
            rng.uniform(-117436, 117436, COUNT),
            signed(rng, -20, high),
        ],
    }
    family = {
        "sin": "cos",
        "log2": "log",
        "logf": "log",
        "log10f": "log",
        "hypot": "atan2",
        "cabs": "atan2",
        "atan2f": "atan2",
        "hypotf": "atan2",
        "powf": "pow",
        "sinf": "cosf",
    }.get(member, member)
    x = np.concatenate([*ranges[family], SPECIALS])
    if arguments == 1:
        y = None
    elif family == "pow":
        y = np.concatenate(
            [rng.uniform(-60, 60, COUNT), signed(rng, -10, 34), SPECIALS]
        )
    else:
        y = np.concatenate(
            [signed(rng, -30, 30), signed(rng, low, high), SPECIALS[::-1]]
        )
    if single:
        x = x.astype(np.float32).astype(np.float64)
        y = None if y is None else y.astype(np.float32).astype(np.float64)
    return member, x, y


def test_table_holds_the_process_ucrtbase_exports(probe_function, probe):
    table = (ct.c_void_p * len(ENTRIES)).in_dll(lib(), "cn_crt")
    ucrtbase = ct.CDLL("ucrtbase")
    for (member, export, _, _), address in zip(ENTRIES, table, strict=True):
        assert address == ct.cast(ucrtbase[export], ct.c_void_p).value, member
    # The probe knows exactly these members.
    for member, *_ in ENTRIES:
        probe(member, -1, np.ones(1), np.ones(1))
    empty = np.empty(1)
    pointer = empty.ctypes.data_as(DOUBLES)
    used = (ct.c_int * 2)()
    assert probe_function(b"sqrt", -1, pointer, pointer, 1, pointer, pointer, used) == 1


def test_probe_restores_the_fma3_states(probe):
    x = np.ones(1)
    before = probe("exp", -1, x, None)[2]
    for fma3 in (0, 1):
        used = probe("exp", fma3, x, None)[2]
        assert used[0] == used[1] and used[0] in (0, fma3)
        assert probe("exp", -1, x, None)[2] == before


@pytest.mark.parametrize(
    "index", range(len(ENTRIES)), ids=[entry[0] for entry in ENTRIES]
)
def test_static_ucrt_against_ucrtbase(probe, index):
    """Informational: warns, never fails, when the two CRTs differ."""
    member, x, y = probe_set(index)
    paths = {}
    for fma3 in (1, 0):
        static, table, used = probe(member, fma3, x, y)
        moved = np.flatnonzero(static.view(np.uint64) != table.view(np.uint64))
        if moved.size:
            first = moved[0]
            at = f"x={x[first]!r}" + ("" if y is None else f", y={y[first]!r}")
            warnings.warn(
                UcrtMovedWarning(
                    f"{member}, FMA3 {used}: ucrtbase.dll differs from the static UCRT on "
                    f"{moved.size} of {x.size} probe inputs (first {at}: static "
                    f"{static[first]!r}, ucrtbase {table[first]!r}). This machine's UCRT moved; "
                    "the tracked goldens may lag the live oracle, and the live-NumPy "
                    "assertions are the contract."
                ),
                stacklevel=1,
            )
        paths[used[1]] = table
    if len(paths) == 2:
        sensitive = np.count_nonzero(
            paths[1].view(np.uint64) != paths[0].view(np.uint64)
        )
        print(
            f"{member}: {sensitive} of {x.size} probe inputs differ between "
            "ucrtbase.dll's FMA3 and SSE2 paths"
        )


def test_load_failure_names_the_library_error(monkeypatch, tmp_path):
    """A library that does not load: the message carries the loader's error and points to
    the line the library writes to stderr when it refuses to initialize (a missing
    ucrtbase.dll export), not to a rebuild alone. Uncached, so lib() keeps its DLL."""
    broken = tmp_path / "chainner_native.dll"
    broken.write_bytes(b"not a library")
    monkeypatch.setattr(native, "library_path", lambda: broken)
    with pytest.raises(RuntimeError) as raised:
        native.lib.__wrapped__()
    message = str(raised.value)
    assert isinstance(raised.value.__cause__, OSError)
    assert str(broken) in message
    assert str(raised.value.__cause__) in message
    assert "stderr" in message
