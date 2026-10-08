"""Every type-check directive native/tests carries, recorded (Consults 13-15, D-34).

pyright checks native/tests with no per-rule downgrade. Where a test deliberately
breaks a declared contract, or a stub or pyright cannot express what runs, the line
carries `# pyright: ignore[<rules>] -- <reason>`, and INVENTORY lists it by file,
rules and reason, so no new one arrives unrecorded. No other suppression comment is
allowed: no noqa, no type: ignore, no ruff, isort or fmt directive, and no other
pyright comment. The frozen reference_* oracles are never edited and are not scanned.
"""

from __future__ import annotations

import io
import re
import tokenize
from collections import Counter
from pathlib import Path

TESTS = Path(__file__).parent
DIRECTIVE = re.compile(
    r"#\s*pyright:\s*ignore\[(?P<rules>[^\]]+)\]\s*--\s*(?P<reason>\S.*)$"
)
SUPPRESSION = re.compile(
    r"#\s*(?:noqa\b|type:\s*ignore|ruff:|pyright:|isort:|fmt:\s*(?:off|skip))"
)

MANGLING = "pyright has no model of name mangling outside the class; the test drives the real private method"
UNHASHABLE = "deliberate: an unhashable fixture; typeshed's unhashable builtins carry the same ignore"
CLASS_PROTOCOL = "deliberate contract violation: a getter-only __class__ for isinstance's class protocol"
OVERRIDE = "reportIncompatibleMethodOverride"
OVERLOAD = "reportCallIssue, reportArgumentType"

INVENTORY = Counter(
    {
        # Consult 14 D-29: mangled private reads.
        ("test_execution_cleanup.py", "reportAttributeAccessIssue", MANGLING): 1,
        # D-26 and D-33: protocol fixtures that break typeshed's declarations.
        ("test_utility_scalar.py", "reportAssignmentType", UNHASHABLE): 1,
        (
            "test_utility_scalar.py",
            OVERRIDE,
            "deliberate contract violation: returns a non-bool truth object, which Python permits",
        ): 1,
        (
            "test_utility_scalar.py",
            OVERRIDE,
            "deliberate contract violation: int.__add__ returns a complex, a legal runtime result",
        ): 1,
        (
            "test_utility_scalar.py",
            OVERRIDE,
            "deliberate contract violation: int.__pow__ returns a bool, kept without coercion",
        ): 1,
        (
            "test_utility_scalar.py",
            OVERRIDE,
            "deliberate contract violation: float.__mod__ returns a Fraction, a legal runtime result",
        ): 1,
        ("test_utility_scalar.py", OVERRIDE, CLASS_PROTOCOL): 1,
        # D-32 and D-33.
        (
            "test_utility_random.py",
            "reportArgumentType",
            "deliberate contract violation: Seed.value is int; the test pins a non-int seed",
        ): 5,
        (
            "test_utility_random.py",
            "reportAttributeAccessIssue",
            "CPython's private method, which typeshed does not declare, is the pinned oracle",
        ): 1,
        ("test_utility_random.py", "reportAssignmentType", UNHASHABLE): 3,
        ("test_utility_random.py", OVERRIDE, CLASS_PROTOCOL): 2,
        (
            "test_utility_random.py",
            OVERRIDE,
            "deliberate contract violation: int.__add__ returns None to select the single-bound range",
        ): 1,
        # Consult 15 A: third-party stub gaps.
        (
            "test_palette_complete.py",
            OVERLOAD,
            "OpenCV's generated stub forbids None for the optional bestLabels output that the binding accepts; upstream's palette.py carries the same ignore",
        ): 1,
        (
            "test_palette_complete.py",
            OVERLOAD,
            "OpenCV's generated stub forbids the scalar bounds that the binding accepts; the call only consumes OpenCV's global RNG",
        ): 1,
        (
            "test_image_preparation.py",
            "reportArgumentType",
            "Pillow's stub gives ImagingCore no __iter__, but the mask is a sequence of pixel values; tobytes() packs mode-1 bits and _new() is private API",
        ): 1,
        # Consult 15 B3 and C: deliberate calls outside upstream's signatures.
        (
            "test_geometry.py",
            "reportArgumentType",
            "deliberate: a NumPy float32 angle pins float32 trig parity with upstream; the signature is upstream's",
        ): 2,
        (
            "test_utility_text.py",
            "reportArgumentType",
            "deliberate: an invalid token pins the error path against upstream",
        ): 1,
        (
            "test_utility_text.py",
            "reportArgumentType",
            "deliberate: SimpleNamespace fakes with invalid group data pin the error path against upstream",
        ): 2,
        (
            "test_utility_text.py",
            OVERRIDE,
            "deliberate contract violation: keys() returns a list to pin iteration order",
        ): 1,
        ("test_utility_text.py", OVERRIDE, CLASS_PROTOCOL): 1,
        # D-34: upstream's narrow annotations and deliberately invalid dims.
        (
            "test_onnx_graph.py",
            "reportArgumentType",
            "upstream's annotation says a 4-tuple, but its callers pass ORT's list shapes; the test pins both functions on lists",
        ): 1,
        (
            "test_onnx_graph.py",
            "reportArgumentType",
            "deliberate: a float dim pins the error path against upstream",
        ): 1,
        (
            "test_onnx_graph.py",
            "reportArgumentType",
            "deliberate: a None dim pins the error path against upstream",
        ): 1,
    }
)


def scanned() -> list[Path]:
    return [
        path
        for path in sorted(TESTS.glob("*.py"))
        if not path.name.startswith("reference_")
    ]


def comments(path: Path) -> list[tuple[int, str]]:
    source = io.StringIO(path.read_text(encoding="utf-8"))
    return [
        (token.start[0], token.string)
        for token in tokenize.generate_tokens(source.readline)
        if token.type == tokenize.COMMENT
    ]


def test_every_pyright_directive_is_recorded():
    found = Counter(
        (path.name, match["rules"], match["reason"])
        for path in scanned()
        for _, comment in comments(path)
        if (match := DIRECTIVE.search(comment))
    )
    assert sum(INVENTORY.values()) == 32
    assert found == INVENTORY


def test_no_other_suppression_comment():
    stray = [
        f"{path.name}:{line}: {comment}"
        for path in scanned()
        for line, comment in comments(path)
        if SUPPRESSION.search(comment) and not DIRECTIVE.search(comment)
    ]
    assert stray == []
