"""pytest plugin of run_tests.py: every test phase's outcome and duration, one JSON line
each, written when the run ends to the file that the CHAINNER_C_SHARD_REPORT environment
variable names.

Phases are setup, call and teardown, as pytest reports them, and "collect" for a module
that skipped or failed as a whole. run_tests.py counts a test as failed when its call
failed, as an error when another phase failed, and as skipped when a phase skipped it
(an expected failure included), as JUnit reports do.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

_RECORDS: list[str] = []


def pytest_collectreport(report: pytest.CollectReport) -> None:
    # A module skipped as a whole (importorskip) or failing to import: one "collect"
    # phase, counted as JUnit counts it (a skip, an error)
    if not report.passed:
        record = {
            "nodeid": report.nodeid,
            "when": "collect",
            "outcome": report.outcome,
            "duration": 0.0,
        }
        _RECORDS.append(json.dumps(record))


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    record = {
        "nodeid": report.nodeid,
        "when": report.when,
        "outcome": report.outcome,
        "duration": report.duration,
    }
    _RECORDS.append(json.dumps(record))


def pytest_unconfigure(config: pytest.Config) -> None:
    path = os.environ.get("CHAINNER_C_SHARD_REPORT")
    if path:
        Path(path).write_text("".join(f"{line}\n" for line in _RECORDS), "utf-8")
