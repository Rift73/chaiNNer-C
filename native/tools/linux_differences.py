"""Live comparisons whose reference library is built otherwise on Linux.

A kernel that mirrors NumPy or OpenCV gives the bits of their Windows builds (MSVC),
which are its reference on every system. Their Linux builds (GCC) differ where the
compilers do: OpenCV contracts multiply-adds into FMA in its AVX2 code, and NumPy
converts NaN, infinite and out-of-range floats to integers otherwise, propagates NaN
payloads and signs otherwise and evaluates pocketfft's twiddle angle in extended
precision. On Linux the tests that native/tests/linux_differences.json lists are
skipped with their group's reason (skip_listed(), from the conftests). C-library math
follows the process's libm there (golden_kernels.py twin); a test only lists it where
the reference also moved, so neither side has Windows bits.

"tests" holds the tests whose native results were recorded equal to the Windows run's
while the Linux reference differed, each with its file's group ("files"). Loaded as a
pytest plugin, this module records each test's NumPy array comparisons
(np.testing.assert_array_equal, assert_allclose and assert_equal, np.array_equal; the
native result first) as the SHA-256 of both arrays, to the file LINUX_DIFFERENCES_RECORD
names. Every comparison of a test runs: one that differs is recorded, not raised.
Record the candidates on each system, then list:

  LINUX_DIFFERENCES_RECORD=windows.json python -m pytest -p linux_differences @ids.txt
  LINUX_DIFFERENCES_RECORD=linux.json python -m pytest -p linux_differences @ids.txt
  python native/tools/linux_differences.py list windows.json linux.json

list writes the tests of the files "files" names whose comparisons moved on Linux in
their reference alone. "named" holds, each with its group, the tests checked otherwise:
their native results compared with the Windows run's outside these comparisons (raw
outputs, or the kernel's B3 manifest test passing on Linux).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

LIST = Path(__file__).resolve().parents[1] / "tests" / "linux_differences.json"
RECORD = "LINUX_DIFFERENCES_RECORD"
COMPARISONS = (
    (np.testing, "assert_array_equal"),
    (np.testing, "assert_allclose"),
    (np.testing, "assert_equal"),
    (np, "array_equal"),
)

_current: list[str | None] = [None]
_records: dict[str, list[tuple[str, str]]] = {}


def skip_listed(items: list[pytest.Item]) -> None:
    """Skips the listed tests with the reason of their group (the conftests call it on
    Linux)."""
    data = json.loads(LIST.read_text(encoding="utf-8"))
    groups = dict(data["named"])
    for nodeid in data["tests"]:
        groups.setdefault(nodeid, data["files"][nodeid.split("::", 1)[0]])
    for item in items:
        group = groups.get(item.nodeid)
        if group is not None:
            item.add_marker(pytest.mark.skip(reason=data["groups"][group]))


def _digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = f"{array.dtype.str}{array.shape}".encode()
    if array.dtype.hasobject:
        return hashlib.sha256(header + repr(value).encode()).hexdigest()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def _recorded(module: Any, name: str) -> None:
    compare = getattr(module, name)

    def recording(actual: Any, desired: Any, *args: Any, **kwargs: Any) -> Any:
        nodeid = _current[0]
        if nodeid is None:
            return compare(actual, desired, *args, **kwargs)
        _records.setdefault(nodeid, []).append((_digest(actual), _digest(desired)))
        return True if name == "array_equal" else None

    setattr(module, name, recording)


def pytest_configure(config: pytest.Config) -> None:
    if not os.environ.get(RECORD):
        raise pytest.UsageError(f"the record needs {RECORD}, the file it writes")
    for module, name in COMPARISONS:
        _recorded(module, name)


def pytest_runtest_call(item: pytest.Item) -> None:
    _current[0] = item.nodeid


def pytest_runtest_teardown(item: pytest.Item) -> None:
    _current[0] = None


def pytest_unconfigure(config: pytest.Config) -> None:
    Path(os.environ[RECORD]).write_text(json.dumps(_records), encoding="utf-8")


def moved_references(windows: dict, linux: dict) -> list[str]:
    """The tests whose comparisons moved on Linux, each moved one in its reference
    alone: the native result the Windows run's."""
    found = []
    for nodeid, recorded in linux.items():
        before = windows.get(nodeid, [])
        if len(before) != len(recorded):
            continue
        moved = [
            (tuple(call), tuple(windows_call))
            for call, windows_call in zip(recorded, before, strict=True)
            if call != windows_call
        ]
        if moved and all(call[0] == windows_call[0] for call, windows_call in moved):
            found.append(nodeid)
    return sorted(found)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser(
        "list", help="write the list's tests from two records"
    )
    listing.add_argument("windows", type=Path)
    listing.add_argument("linux", type=Path)
    args = parser.parse_args(argv)
    windows, linux = (
        json.loads(path.read_text(encoding="utf-8"))
        for path in (args.windows, args.linux)
    )
    data = json.loads(LIST.read_text(encoding="utf-8"))
    data["tests"] = [
        nodeid
        for nodeid in moved_references(windows, linux)
        if nodeid.split("::", 1)[0] in data["files"]
    ]
    LIST.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{LIST}: {len(data['tests'])} tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
