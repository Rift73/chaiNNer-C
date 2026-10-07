"""The pure parts of native/tools/numpy_pool_growth.py (SP4c Task 3).

The checks themselves start a worker, so the controller runs them at the gate (plan
P9). These tests cover the rules the checks judge by (the plateau, the cap, the band a
level must reach), the guard on every output, the read-only image folder and the
refusal to start with the pool's variables already set.
"""

from __future__ import annotations

import builtins
import io
import os
from pathlib import Path

import numpy_pool_growth as growth
import pytest

MIB = 1 << 20
TOTAL = 300  # items; the first third is index <= 100


def entry(**values):
    """A "numpy_pool" entry holding every counter judge() reads."""
    stats = {
        "requests": 100,
        "hits": 90,
        "realloc_inplace": 5,
        "misses_cold": 5,
        "evictions": 0,
        "dropped": 0,
        "threads": 8,
        "held_peak": 40 * MIB,
        "live_peak": 30 * MIB,
        "cap_bytes": 512 * MIB,
        "releases": 1,
        "released_bytes": 40 * MIB,
    }
    stats.update(values)
    return stats


def run(first_third, after, stats=None):
    """One /run: its entry (entry() unless given) and its samples: the start, two in
    the first third, two after it, the end."""
    samples = [
        {"index": 0, "private": 10 * MIB},
        {"index": 50, "private": first_third - MIB},
        {"index": 100, "private": first_third},
        {"index": 200, "private": after - MIB},
        {"index": 300, "private": after - MIB},
        {"index": "end", "private": after},
    ]
    return {"stats": entry() if stats is None else stats, "samples": samples}


@pytest.mark.parametrize(
    ("first_third", "slack"),
    [
        (500 * MIB, 64 * MIB),  # 10 % is 50 MiB: the 64 MiB floor applies
        (2000 * MIB, 200 * MIB),  # 10 % is 200 MiB: above the floor
    ],
)
def test_plateau_rule(first_third, slack):
    allowed = first_third + slack
    held = growth.judge("video", TOTAL, run(first_third, allowed), False)
    assert held["passed"], held["failures"]
    assert held["private_allowed_mib"] == allowed / MIB
    grown = growth.judge("video", TOTAL, run(first_third, allowed + 1), False)
    assert not grown["passed"]
    assert any("grew after the first third" in f for f in grown["failures"])


def test_cap_rule():
    cap = 16 * MIB
    at_cap = growth.judge(
        "video-cap16",
        TOTAL,
        run(MIB * 500, MIB * 500, entry(held_peak=cap, cap_bytes=cap)),
        True,
    )
    assert at_cap["passed"], at_cap["failures"]
    over = growth.judge(
        "video-cap16",
        TOTAL,
        run(MIB * 500, MIB * 500, entry(held_peak=cap + 1, cap_bytes=cap)),
        True,
    )
    assert not over["passed"]
    assert any("held_peak" in f for f in over["failures"])


def test_a_run_without_the_entry_fails():
    # No "numpy_pool" entry on the profile line: the handler was not installed.
    without = run(MIB * 500, MIB * 500) | {"stats": None}
    missing = growth.judge("images", TOTAL, without, False)
    assert not missing["passed"]
    assert any("numpy_pool" in f for f in missing["failures"])


def test_output_outside_growth_is_refused(tmp_path):
    inside = growth.GROWTH / "run-x" / "images-out"
    assert growth.guard_output(inside) == inside.resolve()
    for path in (
        tmp_path / "out",
        growth.GROWTH,
        growth.GROWTH / ".." / "escape",
        growth.DATASET / "images-out",
    ):
        with pytest.raises(RuntimeError, match="refusing an output outside"):
            growth.guard_output(path)


def fake_schemas():
    labels = {
        "chainner:image:load_images": [
            "Directory",
            "Use WCMatch glob expression",
            "Recursive",
            "WCMatch Glob expression",
            "Use limit",
            "Limit",
            "Stop on first error",
        ],
    }
    return {
        schema: {
            "kind": "regularNode",
            "inputs": [
                {"id": i, "label": label}
                for i, label in enumerate(labels.get(schema, ()))
            ],
        }
        for schema in (
            "chainner:image:load_images",
            "chainner:image:resize",
            "chainner:image:save",
        )
    }


def tree(directory: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(directory)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in directory.rglob("*")
    }


def test_images_dir_defaults_to_the_dataset_and_is_never_written(tmp_path, monkeypatch):
    assert Path(r"F:\Download\New folder\DiverSeg-IP\HR\0") == growth.DATASET
    assert growth.parser().parse_args([]).images_dir == growth.DATASET

    # The images check's graph reads the folder; its only writer saves under growth/.
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for number in range(3):
        (dataset / f"{number}.jpg").write_bytes(b"not decoded here")
    output = growth.GROWTH / "run-x" / "images-out"
    nodes, loader = growth.images_graph(fake_schemas(), dataset, 3, output)
    load = next(n for n in nodes if n["id"] == loader)
    assert load["inputs"][0]["value"] == str(dataset)
    assert (load["inputs"][4]["value"], load["inputs"][5]["value"]) == (True, 3)
    (save,) = (n for n in nodes if n["schemaId"] == "chainner:image:save")
    assert Path(save["inputs"][1]["value"]) == output.resolve()

    # --check counts the folder's JPEGs and opens nothing under it for writing.
    before = tree(dataset)
    opened: list[tuple[str, object]] = []
    real_open, real_os_open = io.open, os.open

    def recording_open(file, mode="r", *args, **kwargs):
        opened.append((str(file), mode))
        return real_open(file, mode, *args, **kwargs)

    def recording_os_open(path, flags, *args, **kwargs):
        opened.append((str(path), flags))
        return real_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(io, "open", recording_open)
    monkeypatch.setattr(builtins, "open", recording_open)
    monkeypatch.setattr(os, "open", recording_os_open)
    monkeypatch.setattr(growth, "require_pinned_tools", lambda: None)
    for variable in (growth.POOL_VARIABLE, growth.PROFILE_VARIABLE):
        monkeypatch.delenv(variable, raising=False)
    argv = [
        "--check",
        "--checks",
        "images",
        "--images",
        "3",
        "--images-dir",
        str(dataset),
    ]
    assert growth.main(argv) == 0

    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

    def writes(mode: object) -> bool:
        if isinstance(mode, str):
            return any(flag in mode for flag in "wax+")
        return isinstance(mode, int) and bool(mode & write_flags)

    under = [
        (path, mode)
        for path, mode in opened
        if Path(path).resolve().is_relative_to(dataset.resolve()) and writes(mode)
    ]
    assert under == []
    assert tree(dataset) == before


@pytest.mark.parametrize("variable", ["CHAINNER_C_NUMPY_POOL", "CHAINNER_C_PROFILE"])
def test_refuses_to_start_with_either_variable_set(variable, monkeypatch, capsys):
    for name in (growth.POOL_VARIABLE, growth.PROFILE_VARIABLE):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, "1")
    with pytest.raises(SystemExit) as stopped:
        growth.main(["--check"])
    assert stopped.value.code == 2
    assert variable in capsys.readouterr().err


def test_a_level_is_the_first_sample_inside_the_band():
    base = 2700 * MIB
    values = iter([base + 40 * MIB, base + 17 * MIB, base + 16 * MIB, base])
    level, taken = growth.settle(lambda: next(values), base, step=0)
    assert level == base + 16 * MIB
    assert taken == [base + 40 * MIB, base + 17 * MIB, base + 16 * MIB]
    level, taken = growth.settle(lambda: base - 17 * MIB, base, step=0)
    assert level is None
    assert len(taken) == growth.SETTLE_SAMPLES == 26  # every 0.2 s for 5 s


def test_rest_is_two_samples_in_a_row_within_two_mib():
    values = iter([100 * MIB, 110 * MIB, 112 * MIB + 1, 113 * MIB, 113 * MIB])
    level, taken = growth.at_rest(lambda: next(values), step=0)
    assert level == 113 * MIB
    assert len(taken) == 4
    climbing = iter(range(0, 100 * 3 * MIB, 3 * MIB))
    level, taken = growth.at_rest(lambda: next(climbing), step=0)
    assert level is None
    assert len(taken) == growth.SETTLE_SAMPLES
