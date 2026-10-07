"""ISA probe, CHAINNER_C_ISA override and the cn_isa exports (SP4a, spec 4.3).

The level decides only which implementation runs a kernel's operation sequence;
mirror parameters stay the bit-deciding axis. cn_isa_classify is the probe's
feature rule as a pure export (P8), so every CPU the rule names is tested here.
Tests reach exports through lib()["name"], a fresh function object, and never
set argtypes on the attributes the bridges share.
"""

import ast
import ctypes
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from golden_kernels import load_manifest
from numpy._core._multiarray_umath import __cpu_features__

from nodes.impl import native, native_profile
from nodes.impl.native import lib

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"
GOLDEN = Path(__file__).resolve().parent / "golden"
VARIABLE = "CHAINNER_C_ISA"
# Every feature of both levels: leaf 1 ECX FMA, OSXSAVE, AVX; leaf 7 EBX BMI1,
# AVX2, BMI2, AVX-512 F, DQ, CD, BW, VL; XCR0 SSE, YMM, opmask, ZMM_Hi256, Hi16_ZMM.
FULL = {
    "max_leaf": 7,
    "leaf1_ecx": (1 << 12) | (1 << 27) | (1 << 28),
    "leaf7_ebx": (1 << 3)
    | (1 << 5)
    | (1 << 8)
    | (1 << 16)
    | (1 << 17)
    | (1 << 28)
    | (1 << 30)
    | (1 << 31),
    "xcr0": 0xE6,
}
AVX2_BITS = [
    ("leaf1_ecx", 12),
    ("leaf1_ecx", 27),
    ("leaf1_ecx", 28),
    ("leaf7_ebx", 3),
    ("leaf7_ebx", 5),
    ("leaf7_ebx", 8),
]
AVX512_BITS = [("leaf7_ebx", b) for b in (16, 17, 28, 30, 31)]
CHILD = """
import ctypes, json, sys

from nodes.impl import native, native_profile

gamma = native.lib()["cn_gamma_uses_approximation"]
gamma.argtypes = []
gamma.restype = ctypes.c_int
print(json.dumps({
    "line": native.isa_line(),
    "gamma": gamma(),
    "timed": "cn_isa_get" in native_profile.snapshot(),
    "modules": [m for m in ("PIL", "torch") if m in sys.modules],
}))
"""
# Runs the manifest cases named on stdin ({kernel: [case id]}, default MXCSR only:
# golden_kernels imports torch only for DAZ|FTZ) and reports the ones whose digests
# differ from the manifest's (B3's).
GOLDEN_CHILD = """
import ctypes, json, sys
from pathlib import Path

from nodes.impl import native

NATIVE = Path.cwd().parents[1] / "native"
sys.path.insert(0, str(NATIVE / "tools"))
import golden_kernels

get = native.lib()["cn_isa_get"]
get.argtypes = [ctypes.POINTER(ctypes.c_int)]
get.restype = ctypes.c_int
state = (ctypes.c_int * 3)()
assert get(state) == 0
wanted = json.loads(sys.stdin.read())
ran, mismatched = 0, []
for kernel, ids in wanted.items():
    for case, outputs in golden_kernels.load_manifest(NATIVE / "tests" / "golden" / f"{kernel}.json")[1]:
        if case.id in ids:
            ran += 1
            if golden_kernels.run_case(native.lib(), case) != outputs:
                mismatched.append(case.id)
print(json.dumps({
    "isa": list(state),
    "ran": ran,
    "mismatched": mismatched,
    "modules": [m for m in ("PIL", "torch") if m in sys.modules],
}))
"""


def export(name, argtypes, restype=ctypes.c_int):
    function = lib()[name]
    function.argtypes = argtypes
    function.restype = restype
    return function


def classify(**words):  # positional call: ctypes ignores keyword arguments
    f = export(
        "cn_isa_classify",
        [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint64],
    )
    w = {**FULL, **words}
    return f(w["max_leaf"], w["leaf1_ecx"], w["leaf7_ebx"], w["xcr0"])


def isa_get():
    """(effective, requested, CPU maximum) of this process."""
    state = (ctypes.c_int * 3)()
    assert export("cn_isa_get", [ctypes.POINTER(ctypes.c_int)])(state) == 0
    return tuple(state)


def isa_set(level):
    return export("cn_isa_set", [ctypes.c_int])(level)


def straddling_cases():
    """{kernel: [(case, manifest outputs)]}: the convolution_border cases at fused 8
    and the separable_tiles cases at lanes 8 whose fused boundary (fused_columns, or
    row_width / 8 * 8) is 8 (mod 16), so a 16-lane group straddles it (D15)."""
    found = {}
    for kernel, mirror in (
        ("convolution_border", "fused"),
        ("separable_tiles", "lanes"),
    ):
        for case, outputs in load_manifest(GOLDEN / f"{kernel}.json")[1]:
            args = case.args
            padding = args.get("padding", 0)
            row = (args["w"] + 2 * padding) * args["c"]
            if args[mirror] == 8 and row // 8 * 8 % 16 == 8:
                found.setdefault(kernel, []).append((case, outputs))
    return found


def child(value, profile=False, script=CHILD, stdin=None):
    """The last stdout line, as JSON, of script (CHILD unless given) run from
    backend/src, with stdin as its standard input.

    value None leaves CHAINNER_C_ISA unset. A hung child fails the test at the
    timeout instead of blocking the suite.
    """
    environment = {
        key: item
        for key, item in os.environ.items()
        if key.upper() not in {VARIABLE, native_profile.VARIABLE}
    }
    if value is not None:
        environment[VARIABLE] = value
    if profile:
        environment[native_profile.VARIABLE] = "1"
    try:
        done = subprocess.run(
            [sys.executable, "-B", "-c", script],
            cwd=BACKEND,
            env=environment,
            input=stdin,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("the child interpreter hung")
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.splitlines()[-1])


@pytest.mark.parametrize(
    ("word", "bit", "expected"),
    [*((w, b, 0) for w, b in AVX2_BITS), *((w, b, 1) for w, b in AVX512_BITS)],
)
def test_classify_requires_every_feature_of_the_level(word, bit, expected):
    assert classify() == 2
    assert classify(**{word: FULL[word] & ~(1 << bit)}) == expected


@pytest.mark.parametrize(
    ("xcr0", "expected"),
    [
        (0xE6, 2),
        (0x66, 1),
        (0xA6, 1),
        (0xC6, 1),
        (0x06, 1),
        (0x04, 0),
        (0x02, 0),
        (0, 0),
    ],
)
def test_classify_requires_the_os_state_of_the_level(xcr0, expected):
    assert classify(xcr0=xcr0) == expected


@pytest.mark.parametrize(("max_leaf", "expected"), [(0, 0), (6, 0), (7, 2), (0x24, 2)])
def test_classify_reads_leaf_7_only_if_the_cpu_reports_it(max_leaf, expected):
    assert classify(max_leaf=max_leaf) == expected


def test_cpu_maximum_matches_numpy_features():
    """The probe's CPU maximum against NumPy's own CPUID and XGETBV reading.

    NumPy's AVX2 does not include BMI1 and BMI2: a hypervisor masking them while
    reporting AVX2 is the one case where the probe is right and this test fails.
    """
    avx2 = __cpu_features__["AVX2"] and __cpu_features__["FMA3"]
    avx512 = all(
        __cpu_features__[name]
        for name in ("AVX512F", "AVX512CD", "AVX512BW", "AVX512DQ", "AVX512VL")
    )
    expected = 2 if avx2 and avx512 else 1 if avx2 else 0
    assert isa_get()[2] == expected


def test_isa_get_and_set_cap_at_the_cpu_maximum_and_restore():
    effective, requested, cpu = isa_get()
    try:
        for level in range(3):
            assert isa_set(level) == min(level, cpu)
            assert isa_get() == (min(level, cpu), requested, cpu)
        for invalid in (3, -1):
            before = isa_get()
            assert isa_set(invalid) == -1
            assert isa_get() == before
    finally:
        assert isa_set(effective) == effective
    assert isa_get() == (effective, requested, cpu)


def test_isa_get_refuses_a_null_or_misaligned_buffer():
    get = export("cn_isa_get", [ctypes.c_void_p])
    state = (ctypes.c_int * 4)()
    assert get(None) == 1
    assert get(ctypes.addressof(state) + 1) == 1
    assert list(state) == [0, 0, 0, 0]


@pytest.mark.parametrize(
    "value", [None, "scalar", "avx2", "avx512", "AVX2", "", "avx", " avx2"]
)
def test_environment_override_in_a_child_process(value):
    cpu = isa_get()[2]
    requested = value if value in native.ISA_LEVELS else "auto"
    level = cpu if requested == "auto" else min(native.ISA_LEVELS.index(requested), cpu)
    result = child(value)
    assert result["line"] == (
        f"native isa={native.ISA_LEVELS[level]} "
        f"(requested {requested}, cpu {native.ISA_LEVELS[cpu]})"
    )
    assert result["modules"] == []


def test_avx512_override_in_a_child_process():
    """Review focus 5 (D15): CHAINNER_C_ISA=avx2 caps a child's level (cwd
    backend/src) at avx2: cn_isa_get's effective level is 1 (on a CPU with avx2), the
    straddling cases at the default MXCSR equal B3 there, and the child imports
    neither PIL nor torch."""
    wanted = {
        kernel: [case.id for case, _ in found if case.mxcsr == "default"]
        for kernel, found in straddling_cases().items()
    }
    cpu = isa_get()[2]
    result = child("avx2", script=GOLDEN_CHILD, stdin=json.dumps(wanted))
    assert result["isa"] == [min(1, cpu), 1, cpu]
    assert result["ran"] == sum(map(len, wanted.values())) > 0
    assert result["mismatched"] == []
    assert result["modules"] == []


def test_mirror_probe_ignores_the_override():
    gamma = export("cn_gamma_uses_approximation", [])
    result = child("scalar")
    assert result["line"].startswith("native isa=scalar (requested scalar, cpu ")
    assert result["gamma"] == gamma()
    assert result["modules"] == []


def test_isa_line_with_the_timer_on():
    on, off = child(None, profile=True), child(None)
    assert on["line"] == off["line"]
    assert (on["timed"], off["timed"]) == (True, False)
    assert on["modules"] == off["modules"] == []


def test_isa_line_format():
    assert re.fullmatch(
        r"native isa=(scalar|avx2|avx512) \(requested (auto|scalar|avx2|avx512), "
        r"cpu (scalar|avx2|avx512)\)",
        native.isa_line(),
    )


class Export:
    """A ctypes function stand-in with settable argtypes and restype."""

    def __init__(self, result):
        self.result = result
        self.argtypes = None
        self.restype = None

    def __call__(self):
        return self.result


class OldDll:
    """A DLL of ABI 2 built before the cn_isa exports."""

    def __init__(self, name):
        self.name = name
        self.cn_abi_version = Export(2)


@pytest.mark.parametrize("profile", [False, True])
def test_lib_names_the_dll_when_cn_isa_get_is_missing(monkeypatch, profile):
    monkeypatch.setattr(native_profile, "enabled", lambda: profile)
    monkeypatch.setattr(ctypes, "CDLL", OldDll)
    monkeypatch.setattr(native_profile, "ProfiledCDLL", OldDll)
    with pytest.raises(
        RuntimeError,
        match=r"Incompatible chaiNNer C kernel ABI at .*chainner_native\.dll: "
        r"no cn_isa_get export",
    ):
        native.lib.__wrapped__()


def test_setup_logs_the_isa_line_after_loading_nodes():
    tree = ast.parse((BACKEND / "server.py").read_text(encoding="utf-8"))
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "nodes.impl"
        and [alias.name for alias in node.names]
        == ["native", "native_profile", "numpy_pool"]
        for node in tree.body
    )
    (setup,) = (
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "setup"
    )
    assert [ast.unparse(s) for s in setup.body] == [
        "await import_packages(AppContext.get(sanic_app).config)",
        "logger.info(native.isa_line())",
    ]
