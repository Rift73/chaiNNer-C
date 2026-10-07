from __future__ import annotations

import hashlib
import json
from pathlib import Path

import bench_cases
import bench_fixtures
import pytest

KINDS = {
    "chainner:image:load_images": "generator",
    "chainner:image:load_video": "generator",
    "chainner:image:save_video": "collector",
}


def fake_schemas():
    return {
        schema: {
            "kind": KINDS.get(schema, "regularNode"),
            "inputs": [{"id": i, "def": None} for i in range(24)],
        }
        for schema in bench_cases.ALLOWED
    }


@pytest.mark.parametrize("case", bench_cases.CASES)
def test_every_case_builds_an_allowed_single_generator_graph(case, tmp_path):
    nodes, count, unit = bench_cases.graph(
        fake_schemas(), case, tmp_path, tmp_path / "out"
    )
    assert {n["schemaId"] for n in nodes} <= bench_cases.ALLOWED
    assert sum(n["nodeType"] == "generator" for n in nodes) == 1
    if case.startswith("video-"):
        assert (count, unit) == (bench_cases.VIDEO_FRAMES, "frames/s")
    else:
        assert unit == "images/s"


def test_resize_case_redirects_only_the_output(tmp_path):
    nodes, count, _ = bench_cases.graph(
        fake_schemas(), "resize-real-256", tmp_path, tmp_path / "out"
    )
    save = next(n for n in nodes if n["schemaId"] == "chainner:image:save")
    load = next(n for n in nodes if n["schemaId"] == "chainner:image:load_images")
    assert save["inputs"][1]["value"] == str(tmp_path / "out")
    assert load["inputs"][0]["value"] == r"F:\Download\New folder\DiverSeg-IP\HR\0"
    assert count == 256


def test_unknown_case_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="nope"):
        bench_cases.graph(fake_schemas(), "nope", tmp_path, tmp_path)


def test_tracked_inputs_are_256_distinct_files():
    inputs = bench_fixtures.load_inputs()
    assert len(inputs) == 256
    assert len({item["path"] for item in inputs}) == 256


def test_pinned_output_names_match_inputs_one_to_one():
    names = bench_fixtures.pinned_output_names()
    assert len(names) == 256
    assert all(name.endswith(".png") for name in names)
    assert names == {
        Path(i["path"]).stem + ".png" for i in bench_fixtures.load_inputs()
    }


def test_verify_inputs_names_changed_file(tmp_path):
    good, bad = tmp_path / "a.jpg", tmp_path / "b.jpg"
    good.write_bytes(b"a")
    bad.write_bytes(b"b")
    items = [
        {"path": str(good), "sha256": hashlib.sha256(b"a").hexdigest()},
        {"path": str(bad), "sha256": hashlib.sha256(b"other").hexdigest()},
    ]
    with pytest.raises(RuntimeError, match=r"b\.jpg"):
        bench_fixtures.verify_inputs(items)


def test_verify_inputs_names_missing_file(tmp_path):
    missing = [{"path": str(tmp_path / "gone.jpg"), "sha256": "0" * 64}]
    with pytest.raises(FileNotFoundError, match=r"gone\.jpg"):
        bench_fixtures.verify_inputs(missing)


def test_fixture_pins_cover_every_generated_image_and_the_decoded_video():
    pins = json.loads(bench_fixtures.PINS.read_text(encoding="utf-8"))
    assert list(pins["images"]) == [
        f"images/{i:03}.jpg" for i in range(bench_cases.IMAGE_COUNT)
    ]
    assert all(len(h) == 64 for h in pins["images"].values())
    assert set(pins["moving.mkv"]) == set(bench_fixtures.VIDEO_IDENTITY)
    assert pins["moving.mkv"]["decoded_frame_count"] == bench_cases.VIDEO_FRAMES


def test_check_pins_accepts_matching_values():
    bench_fixtures.check_pins({"images/000.jpg": "a"}, {"images/000.jpg": "a"})


@pytest.mark.parametrize(
    ("actual", "first"),
    [
        ({"images/000.jpg": "a", "images/001.jpg": "x", "images/002.jpg": "y"}, "001"),
        ({"images/000.jpg": "a", "images/002.jpg": "c"}, "001"),
        ({"images/000.jpg": "a", "images/001.jpg": "b"}, "002"),
    ],
)
def test_check_pins_names_first_mismatch(actual, first):
    pinned = {"images/000.jpg": "a", "images/001.jpg": "b", "images/002.jpg": "c"}
    with pytest.raises(RuntimeError, match=rf"images/{first}\.jpg"):
        bench_fixtures.check_pins(actual, pinned)


def test_check_pins_rejects_unpinned_values():
    with pytest.raises(RuntimeError, match=r"images/032\.jpg"):
        bench_fixtures.check_pins(
            {"images/000.jpg": "a", "images/032.jpg": "z"}, {"images/000.jpg": "a"}
        )
