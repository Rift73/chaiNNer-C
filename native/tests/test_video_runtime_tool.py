r"""Static fixture and comparison-policy checks for the video HTTP verifier.

video_runtime/nodes.json is the backend's /nodes as trimmed() keeps it, made
2026-10-06 from native/reports/video-runtime-20261006T133151Z/converted/nodes.json
(sha256 1f4c1342df1333294cfcc07e4f98e7bd55762682ae2190a6439e4c610cb3194b, the
oracle's too, as the verifiers require, and every capture's since 2026-09-20).
When fixture_graphs or one of the schemas it uses changes, run a verifier and
regenerate it from that report, at the repository root:

    $env:PYTHONPATH = 'native\tests;native\tools'
    & $py -B -c "import json, sys, test_video_runtime_tool as t; t.NODES.write_text(json.dumps(t.trimmed(json.load(open(sys.argv[1], encoding='utf-8'))), indent=2) + '\n', encoding='utf-8', newline='\n')" <report>\converted\nodes.json
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import verify_runtime
from test_verify_runtime import LOCK, declared_dependency, declared_metadata, write
from verify_runtime import sse_record
from verify_video_runtime import VIDEO_IDS, compare_runs, fixture_graphs, snapshot

NODES = Path(__file__).with_name("video_runtime") / "nodes.json"


def trimmed(catalog: dict) -> dict:
    """A /nodes response cut to what fixture_graphs reads: the schemas its graphs
    use, in catalog order, each with its kind and its inputs' and outputs' ids
    and labels, and each input's default when the schema gives one."""
    schemas = {node["schemaId"]: node for node in catalog["nodes"]}
    used = {
        node["schemaId"]
        for graph in fixture_graphs(schemas, Path(), Path())
        for node in graph["nodes"]
    }
    return {
        "nodes": [
            {
                "schemaId": node["schemaId"],
                "kind": node["kind"],
                "inputs": [
                    {key: item[key] for key in ("id", "label", "def") if key in item}
                    for item in node["inputs"]
                ],
                "outputs": [
                    {"id": item["id"], "label": item["label"]}
                    for item in node["outputs"]
                ],
            }
            for node in catalog["nodes"]
            if node["schemaId"] in used
        ]
    }


@pytest.fixture(autouse=True)
def lock(tmp_path, monkeypatch):
    """compare_metadata reads test_verify_runtime's LOCK as the project's."""
    write(tmp_path / "native/python-stack.lock.txt", LOCK)
    monkeypatch.setattr(verify_runtime, "PROJECT", tmp_path)


def compare(old, new):
    """compare_runs with the oracle's declarations (declared_metadata) on old's
    side and the port's on new's."""
    oracle, port = declared_metadata()
    return compare_runs({**old, **oracle}, {**new, **port})


def test_the_schema_fixture_is_its_own_trim():
    """NODES holds only the schemas fixture_graphs uses and only the fields it
    reads, so trimmed(), the regeneration step, stays in step with the graphs."""
    catalog = json.loads(NODES.read_text(encoding="utf-8"))
    assert trimmed(catalog) == catalog


def test_video_http_graphs_use_registered_inputs_and_cover_both_nodes(tmp_path):
    catalog = json.loads(NODES.read_text(encoding="utf-8"))
    schemas = {node["schemaId"]: node for node in catalog["nodes"]}
    fixtures = list(
        fixture_graphs(schemas, tmp_path / "fixtures", tmp_path / "outputs")
    )
    assert len(fixtures) == 26
    assert len({f["name"] for f in fixtures}) == 26
    assert sum(f["cancel"] for f in fixtures) == 1
    assert sum(f["expected_error"] for f in fixtures) == 3
    for fixture in fixtures:
        nodes = {n["id"]: n for n in fixture["nodes"]}
        assert VIDEO_IDS <= {n["schemaId"] for n in nodes.values()}
        for node in nodes.values():
            schema = schemas[node["schemaId"]]
            assert len(node["inputs"]) == len(schema["inputs"])
            for value in node["inputs"]:
                if value["type"] == "edge":
                    producer = nodes[value["id"]]
                    assert value["index"] in {
                        item["id"] for item in schemas[producer["schemaId"]]["outputs"]
                    }


def run_fixture() -> dict:
    return {
        "fixtures": [
            {
                "name": "clip",
                "cancel": False,
                "generator_node_ids": [],
                "attempts": [
                    {
                        "http_status": 200,
                        "response": {"type": "success"},
                        "event_kinds": [],
                        "events": [],
                        "final_state": {},
                        "files": {
                            "video.mkv": {
                                "sha256": "different-container-id",
                                "semantic": {"pixels": "exact", "fps": "3/1"},
                            }
                        },
                    }
                ],
            }
        ],
    }


def test_container_bytes_difference_is_disclosed_not_confused_with_pixel_parity():
    old = run_fixture()
    new = copy.deepcopy(old)
    new["fixtures"][0]["attempts"][0]["files"]["video.mkv"]["sha256"] = "other-id"
    comparison = compare(old, new)
    assert comparison["success"]
    assert comparison["fixtures"]["clip"]["encoded_file_bytes_equal"] == {
        "video.mkv": False
    }


@pytest.mark.parametrize("field", ["pixels", "fps"])
def test_pixel_or_fps_difference_fails_comparison(field):
    old = run_fixture()
    new = copy.deepcopy(old)
    new["fixtures"][0]["attempts"][0]["files"]["video.mkv"]["semantic"][field] = (
        "different"
    )
    assert not compare(old, new)["success"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("http_status", 500),
        ("response", {"type": "error"}),
        ("events", [{"event": "chain-start", "data": {"nodes": ["load"]}}]),
    ],
    ids=["http_status", "response", "events"],
)
def test_http_or_sse_difference_fails_comparison(field, value):
    old = run_fixture()
    new = copy.deepcopy(old)
    new["fixtures"][0]["attempts"][0][field] = value
    assert not compare(old, new)["success"]


def started(node):
    return {"event": "node-start", "data": {"nodeId": node}}


def frame(index):
    """Load Video's preview of one frame."""
    return {
        "event": "node-broadcast",
        "data": {
            "nodeId": "load",
            "data": {"0": None},
            "types": {"0": {"value": index}},
            "sequenceTypes": {},
        },
    }


SEQUENCE = {
    "event": "node-broadcast",
    "data": {"nodeId": "load", "data": {}, "types": {}, "sequenceTypes": {"0": 3}},
}
PORT = [started("load"), SEQUENCE, frame(0), frame(1), frame(2), started("save")]


@pytest.mark.parametrize(
    ("baseline", "converted", "equal"),
    [
        # Upstream re-runs Load Video per frame: one more start and broadcast each.
        ([*PORT, *[started("load"), SEQUENCE] * 2], PORT, True),
        (PORT, [*PORT[:3], *PORT[4:]], True),
        (PORT, [*PORT, started("save")], False),
        (PORT, [*PORT[:3], frame(5), *PORT[4:]], False),
        # Consult D-35: the port may end on any frame the oracle previewed.
        (PORT, PORT[:4] + PORT[5:], True),
    ],
    ids=[
        "generator-reruns",
        "dropped-frame-preview",
        "extra-non-generator-start",
        "foreign-frame-preview",
        "dropped-last-frame",
    ],
)
def test_events_compare_under_the_shared_rule(baseline, converted, equal):
    old = run_fixture()
    old["fixtures"][0]["generator_node_ids"] = ["load"]
    new = copy.deepcopy(old)
    for run, events in ((old, baseline), (new, converted)):
        run["fixtures"][0]["attempts"][0].update(sse_record(events, Path("C:/run")))
    comparison = compare(old, new)
    assert comparison["success"] is equal
    assert (comparison["fixtures"]["clip"]["sse_event_mismatches"] == []) is equal


# Load Video's preview of a frame it never decoded, before its last frame's preview
# so the final state is unchanged: only the sub-multiset rule can see it.
FOREIGN = [*PORT[:3], frame(5), *PORT[3:]]


@pytest.mark.parametrize(
    ("first", "second", "equal"),
    [
        (PORT, [*PORT[:3], *PORT[4:]], True),
        (FOREIGN, PORT, False),
        (PORT, FOREIGN, False),
    ],
    ids=["repeat-coalesced", "foreign-preview-first-attempt", "foreign-preview-repeat"],
)
def test_both_port_attempts_compare_under_the_shared_rule(first, second, equal):
    old = run_fixture()
    old["fixtures"][0]["generator_node_ids"] = ["load"]
    old["fixtures"][0]["attempts"][0].update(sse_record(PORT, Path("C:/run")))
    new = copy.deepcopy(old)
    new["fixtures"][0]["attempts"] = [
        {**old["fixtures"][0]["attempts"][0], **sse_record(events, Path("C:/run"))}
        for events in (first, second)
    ]
    comparison = compare(old, new)
    assert comparison["success"] is equal
    if not equal:
        [mismatch] = comparison["fixtures"]["clip"]["sse_event_mismatches"]
        assert mismatch.startswith(
            "repeat: load: broadcast" if second is FOREIGN else "load: broadcast"
        )
    # Consult 8 R-h (4): each attempt's broadcast multiset is in the report.
    frames = [
        sorted(
            (json.loads(key)["types"]["0"]["value"], count)
            for key, count in attempt["load"].items()
            if json.loads(key)["types"]
        )
        for attempt in comparison["fixtures"]["clip"]["broadcast_multisets"]["port"]
    ]
    assert frames == [
        sorted((x["data"]["types"]["0"]["value"], 1) for x in events if x in FRAMES)
        for events in (first, second)
    ]
    [oracle] = comparison["fixtures"]["clip"]["broadcast_multisets"]["oracle"]
    assert len(oracle["load"]) == 4  # The sequence and frames 0, 1 and 2, once each.


FRAMES = [frame(index) for index in range(6)]


def png_snapshot(path: Path, image: np.ndarray, level: int) -> dict:
    """snapshot() of image written as a PNG at a zlib compression level."""
    ok, data = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, level])
    assert ok
    path.write_bytes(data.tobytes())
    return snapshot(path, Path("unused-ffmpeg"), Path("unused-ffprobe"))


def test_png_semantic_is_its_decoded_pixels(tmp_path):
    image = np.random.default_rng(20261006).integers(0, 255, (9, 13, 3), np.uint8)
    # The same pixels, stored uncompressed and at the strongest zlib level.
    stored = png_snapshot(tmp_path / "stored.png", image, 0)
    packed = png_snapshot(tmp_path / "packed.png", image, 9)
    assert stored["sha256"] != packed["sha256"]
    assert stored["semantic"] == packed["semantic"]
    changed = image.copy()
    changed[8, 12, 2] ^= 1  # One bit of the last byte.
    differs = png_snapshot(tmp_path / "changed.png", changed, 9)
    assert differs["semantic"] != packed["semantic"]

    old = run_fixture()
    files = old["fixtures"][0]["attempts"][0]["files"]
    new = copy.deepcopy(old)
    files["0.png"] = stored
    new["fixtures"][0]["attempts"][0]["files"]["0.png"] = packed
    comparison = compare(old, new)
    assert comparison["success"]
    assert comparison["fixtures"]["clip"]["encoded_file_bytes_equal"] == {
        "video.mkv": True,
        "0.png": False,
    }
    new["fixtures"][0]["attempts"][0]["files"]["0.png"] = differs
    assert not compare(old, new)["success"]


def test_other_files_keep_their_byte_hash(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"not a video or PNG")
    record = snapshot(path, Path("unused-ffmpeg"), Path("unused-ffprobe"))
    assert record["semantic"] == record["sha256"]


def test_cancel_requires_no_remaining_owned_children():
    old = run_fixture()
    new = copy.deepcopy(old)
    for value in (old, new):
        value["fixtures"][0]["cancel"] = True
        value["fixtures"][0]["attempts"][0]["remaining_ffmpeg_children"] = {}
    assert compare(old, new)["success"]
    new["fixtures"][0]["attempts"][0]["remaining_ffmpeg_children"] = {
        "123": "owned-ffmpeg.exe"
    }
    assert not compare(old, new)["success"]


def test_metadata_compares_under_the_shared_rule():
    old = run_fixture()
    new = copy.deepcopy(old)
    oracle, port = declared_metadata()
    matched = compare_runs({**old, **oracle}, {**new, **port})
    assert matched["success"]
    assert all(matched["metadata_equal"].values())
    assert len(matched["declaration_changes_applied"]) == 5
    declared_dependency(port, "chaiNNer_pytorch", "torch")["sizeEstimate"] = 1
    unlisted = compare_runs({**old, **oracle}, {**new, **port})
    assert not unlisted["success"]
    assert list(unlisted["metadata_mismatches"]) == ["packages"]
