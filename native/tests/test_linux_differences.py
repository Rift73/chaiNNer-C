"""linux_differences.py's list: the tests it skips, and a record that must run them."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import linux_differences
import pytest


def listed_item(markers: list[pytest.Mark]) -> list[pytest.Item]:
    data = json.loads(linux_differences.LIST.read_text(encoding="utf-8"))
    item = SimpleNamespace(nodeid=data["tests"][0], add_marker=markers.append)
    return cast("list[pytest.Item]", [item])


def test_a_listed_test_is_skipped(monkeypatch):
    monkeypatch.delenv(linux_differences.RECORD, raising=False)
    markers: list[pytest.Mark] = []
    linux_differences.skip_listed(listed_item(markers))
    assert len(markers) == 1


def test_a_record_runs_the_listed_tests(monkeypatch, tmp_path):
    # Skipped, a listed test would record nothing, and the next `list` would drop it.
    monkeypatch.setenv(linux_differences.RECORD, str(tmp_path / "record.json"))
    markers: list[pytest.Mark] = []
    linux_differences.skip_listed(listed_item(markers))
    assert markers == []
