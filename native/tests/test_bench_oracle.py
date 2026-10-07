from __future__ import annotations

import hashlib
import os
import struct
import threading
import zlib
from pathlib import Path
from typing import cast

import bench_oracle
import cv2
import numpy as np
import pytest
import verify_runtime
import verify_video_runtime

SOURCE = r"F:\fixtures\images"
LABELS = ["Image", "Directory", "Subdirectory Path", "Name", "Index"]
SCHEMAS = {
    "chainner:image:load_images": {
        "kind": "generator",
        "outputs": [{"id": i, "label": label} for i, label in enumerate(LABELS)],
        "iteratorOutputs": [{"id": 0, "outputs": [0, 2, 3, 4]}],
    },
    "chainner:image:save": {"kind": "regularNode", "outputs": []},
}
NODES = [
    {
        "id": "load",
        "schemaId": "chainner:image:load_images",
        "inputs": [{"type": "value", "value": SOURCE}],
    },
    {"id": "save", "schemaId": "chainner:image:save", "inputs": []},
]


def directory_type():
    return {
        "1": {
            "type": "named",
            "name": "Directory",
            "fields": {"path": {"type": "string-literal", "value": SOURCE}},
        }
    }


def stream(count=2, restore=True):
    """A minimal complete Load Images -> Save event log, shaped like real captures."""
    length = {"0": {"type": "named", "name": "Sequence", "fields": {"length": count}}}
    events = [
        {"event": "chain-start", "data": {"nodes": ["save", "load"]}},
        {"event": "node-start", "data": {"nodeId": "load"}},
        {
            "event": "node-progress",
            "data": {
                "nodeId": "load",
                "progress": 0.0,
                "index": 0,
                "total": count,
                "eta": 0,
            },
        },
        {
            "event": "node-broadcast",
            "data": {
                "nodeId": "load",
                "data": {"1": None},
                "types": directory_type(),
                "sequenceTypes": length,
            },
        },
    ]
    for index in range(count):
        events += [
            {
                "event": "node-broadcast",
                "data": {
                    "nodeId": "load",
                    "data": {"4": None},
                    "types": {"4": {"type": "numeric-literal", "value": index}},
                    "sequenceTypes": {},
                },
            },
            {"event": "node-start", "data": {"nodeId": "save"}},
            {"event": "node-finish", "data": {"nodeId": "save", "executionTime": 0.01}},
        ]
    events += [
        {
            "event": "node-progress",
            "data": {
                "nodeId": "load",
                "progress": 1,
                "index": count,
                "total": count,
                "eta": 0,
            },
        },
        {"event": "node-finish", "data": {"nodeId": "load", "executionTime": 0.2}},
    ]
    if restore:
        events.append(
            {
                "event": "node-broadcast",
                "data": {
                    "nodeId": "load",
                    "data": {"1": None},
                    "types": directory_type(),
                    "sequenceTypes": {},
                },
            }
        )
    return events


def test_validate_record_accepts_complete_tail(tmp_path):
    check = bench_oracle.validate_record(NODES, stream(), SCHEMAS, 2, tmp_path)
    assert check["node_id"] == "load"
    assert check["static_restoration_position"] > check["finish_position"]


def test_validate_record_rejects_missing_restoration(tmp_path):
    with pytest.raises(AssertionError, match="restoration"):
        bench_oracle.validate_record(NODES, stream(restore=False), SCHEMAS, 2, tmp_path)


def test_validate_record_rejects_execution_error(tmp_path):
    events = [*stream(), {"event": "execution-error", "data": {"message": "boom"}}]
    with pytest.raises(AssertionError, match="execution error"):
        bench_oracle.validate_record(NODES, events, SCHEMAS, 2, tmp_path)


def test_validate_record_rejects_incomplete_progress(tmp_path):
    with pytest.raises(AssertionError, match="final generator progress"):
        bench_oracle.validate_record(NODES, stream(count=2), SCHEMAS, 3, tmp_path)


class CapturedEvents(verify_runtime.Events):
    """Events over a given list: no SSE connection, the test appends the events."""

    def __init__(self, events):
        self.events = events
        self.error = None


NODE_IDS = {"load", "save"}
BROADCASTERS = {"load"}


def test_wait_for_returns_before_restoration_without_the_parameter(tmp_path):
    # The generator's initial broadcast and its finish satisfy the old condition
    # while the restoration is still in flight; the oracle rejects that capture.
    captured = CapturedEvents(stream(restore=False)).wait_for(
        NODE_IDS, 0, timeout=5, expected_broadcasts=BROADCASTERS
    )
    assert captured[-1] == finish("load", executionTime=0.2)
    with pytest.raises(AssertionError, match="restoration"):
        bench_oracle.validate_record(NODES, captured, SCHEMAS, 2, tmp_path)


def test_wait_for_broadcast_after_finish_waits_for_the_restoration(tmp_path):
    restoration = stream()[-1]
    events = CapturedEvents(stream(restore=False))
    timer = threading.Timer(0.2, events.events.append, [restoration])
    timer.start()
    try:
        captured = events.wait_for(
            NODE_IDS,
            0,
            timeout=5,
            expected_broadcasts=BROADCASTERS,
            broadcast_after_finish=BROADCASTERS,
        )
    finally:
        timer.join()
    assert captured[-1] == restoration
    check = bench_oracle.validate_record(NODES, captured, SCHEMAS, 2, tmp_path)
    assert check["static_restoration_position"] > check["finish_position"]


def test_wait_for_broadcast_after_finish_names_the_missing_restoration():
    events = CapturedEvents(stream(restore=False))
    with pytest.raises(RuntimeError, match=r"broadcast_after_finish=\['load'\]"):
        events.wait_for(
            NODE_IDS,
            0,
            timeout=0.2,
            expected_broadcasts=BROADCASTERS,
            broadcast_after_finish=BROADCASTERS,
        )


def test_wait_for_broadcast_after_finish_counts_only_the_new_events():
    # The restoration of an earlier run is before start and must not count.
    earlier = stream()
    events = CapturedEvents([*earlier, *stream(restore=False)])
    with pytest.raises(RuntimeError, match=r"broadcast_after_finish=\['load'\]"):
        events.wait_for(
            NODE_IDS,
            len(earlier),
            timeout=0.2,
            expected_broadcasts=BROADCASTERS,
            broadcast_after_finish=BROADCASTERS,
        )


def test_sse_contract_normalizes_order_paths_and_times(tmp_path):
    out = tmp_path / "out"
    events = [
        {"event": "chain-start", "data": {"nodes": ["b", "a"]}},
        {"event": "node-finish", "data": {"nodeId": "a", "executionTime": 1.5}},
        {
            "event": "node-broadcast",
            "data": {
                "nodeId": "a",
                "data": {"0": str(out / "x.png")},
                "types": {"0": 1},
                "sequenceTypes": {},
            },
        },
        {
            "event": "node-broadcast",
            "data": {
                "nodeId": "a",
                "data": {"1": 2},
                "types": {},
                "sequenceTypes": None,
            },
        },
    ]
    contract = verify_runtime.sse_contract(events, out)
    assert contract["controls"] == [
        {"event": "chain-start", "data": {"nodes": ["a", "b"]}}
    ]
    finish = next(e for e in contract["final_state"] if e["event"] == "node-finish")
    assert finish == {"event": "node-finish", "data": {"nodeId": "a"}}
    broadcast = next(
        e for e in contract["final_state"] if e["event"] == "node-broadcast"
    )
    assert broadcast["data"]["data"] == {"0": "<run>" + os.sep + "x.png", "1": 2}
    assert broadcast["data"]["types"] == {"0": 1}
    assert broadcast["data"]["sequenceTypes"] == {}


def test_sse_contract_ignores_independent_event_order(tmp_path):
    events = stream()
    swapped = [events[0], events[3], events[1], events[2], *events[4:]]
    contract = verify_runtime.sse_contract
    assert contract(events, tmp_path) == contract(swapped, tmp_path)


def test_sse_contract_rejects_malformed_event(tmp_path):
    with pytest.raises(TypeError, match="Malformed SSE event: 'oops'"):
        # Deliberately not an event dict.
        verify_runtime.sse_contract(cast(list[dict], ["oops"]), tmp_path)


def finish(node_id, **extra):
    return {"event": "node-finish", "data": {"nodeId": node_id, **extra}}


CONTRACT = {
    "controls": [{"event": "chain-start", "data": {"nodes": ["load", "save"]}}],
    "final_state": [finish("load"), finish("save")],
}


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ({**CONTRACT, "controls": []}, "controls"),
        (
            {**CONTRACT, "final_state": [finish("load"), finish("save", x=1)]},
            "('node-finish', 'save')",
        ),
        (
            {**CONTRACT, "final_state": [finish("extra"), *CONTRACT["final_state"]]},
            "('node-finish', 'extra')",
        ),
        ({**CONTRACT, "final_state": [finish("load")]}, "('node-finish', 'save')"),
    ],
)
def test_contract_difference_names_first_difference(actual, expected):
    assert bench_oracle.contract_difference(CONTRACT, actual) == expected


def test_contract_difference_refuses_identical_contracts():
    with pytest.raises(ValueError, match="identical"):
        bench_oracle.contract_difference(CONTRACT, CONTRACT)


def test_png_outputs_hash_only_pngs(tmp_path):
    (tmp_path / "a.png").write_bytes(b"png-a")
    (tmp_path / "note.txt").write_bytes(b"x")
    assert list(bench_oracle.png_outputs(tmp_path)) == ["a.png"]


def write_png(directory: Path, name: str, pixels: np.ndarray, level: int = 3) -> None:
    """Encode pixels as a PNG at a zlib compression level and write it."""
    directory.mkdir(exist_ok=True)
    ok, data = cv2.imencode(".png", pixels, [cv2.IMWRITE_PNG_COMPRESSION, level])
    assert ok
    (directory / name).write_bytes(data.tobytes())


def pixels(shape: tuple[int, ...], dtype: type) -> np.ndarray:
    """Seeded noise, so no encoder can store it trivially."""
    maximum = np.iinfo(dtype).max
    return np.random.default_rng(20261004).integers(0, maximum, shape, dtype=dtype)


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((37, 41, 3), np.uint8),
        ((37, 41, 4), np.uint8),
        ((37, 41), np.uint8),
        ((37, 41, 3), np.uint16),
    ],
    ids=["bgr8", "bgra8", "gray8", "bgr16"],
)
def test_png_outputs_decoded_equal_for_differently_encoded_pngs(tmp_path, shape, dtype):
    image = pixels(shape, dtype)
    # The same pixels, stored uncompressed and at the strongest zlib level.
    write_png(tmp_path / "fast", "000.png", image, level=0)
    write_png(tmp_path / "small", "000.png", image, level=9)
    (tmp_path / "small" / "note.txt").write_bytes(b"x")  # Only PNGs are decoded.
    fast, small = tmp_path / "fast", tmp_path / "small"
    # The default mode compares the file bytes, which differ.
    assert bench_oracle.png_outputs(fast) != bench_oracle.png_outputs(small)
    decoded = bench_oracle.png_outputs(small, "decoded")
    assert bench_oracle.png_outputs(fast, "decoded") == decoded
    assert decoded == {
        "000.png": {
            "shape": list(shape),
            "dtype": np.dtype(dtype).name,
            "pixels_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "chunks": [],  # OpenCV writes no colour or orientation chunk.
        }
    }


def decoded_record(directory: Path) -> dict:
    """The decoded-mode record of directory's 000.png."""
    record = bench_oracle.png_outputs(directory, "decoded")["000.png"]
    assert isinstance(record, dict)
    return record


def test_png_outputs_decoded_compares_shape_dtype_and_every_byte(tmp_path):
    gray = pixels((4, 6), np.uint8)
    write_png(tmp_path / "base", "000.png", gray)
    base = decoded_record(tmp_path / "base")
    changed = gray.copy()
    changed[3, 5] ^= 1  # One bit of the last byte.
    variants = {
        "byte": (changed, {"pixels_sha256"}),
        # The same bytes as two-pixel-wide BGR: only the shape tells them apart.
        "shape": (gray.reshape(4, 2, 3), {"shape"}),
        # The same bytes as 16-bit gray, three samples per row.
        "dtype": (gray.view(np.uint16), {"shape", "dtype"}),
    }
    for name, (variant, expected) in variants.items():
        write_png(tmp_path / name, "000.png", variant)
        record = decoded_record(tmp_path / name)
        assert {key for key in base if record[key] != base[key]} == expected, name


def with_chunk(png: bytes, kind: bytes, payload: bytes) -> bytes:
    """png with one chunk inserted right after IHDR (the signature and IHDR are 33 bytes)."""
    body = kind + payload
    chunk = struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))
    return png[:33] + chunk + png[33:]


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        (b"gAMA", struct.pack(">I", 45455)),
        (b"sRGB", b"\x00"),
        # A big-endian TIFF header and an empty IFD.
        (b"eXIf", b"MM\x00\x2a\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00"),
    ],
    ids=["gAMA", "sRGB", "eXIf"],
)
def test_png_outputs_decoded_compares_colour_and_orientation_chunks(
    tmp_path, kind, payload
):
    write_png(tmp_path / "plain", "000.png", pixels((37, 41, 3), np.uint8))
    plain = (tmp_path / "plain" / "000.png").read_bytes()
    assert plain[12:16] == b"IHDR"
    (tmp_path / "tagged").mkdir()
    (tmp_path / "tagged" / "000.png").write_bytes(with_chunk(plain, kind, payload))
    base = decoded_record(tmp_path / "plain")
    tagged = decoded_record(tmp_path / "tagged")
    assert base["chunks"] == []
    assert tagged["chunks"] == [[kind.decode(), hashlib.sha256(payload).hexdigest()]]
    # The same pixels: only the chunk tells them apart, and that fails a trial.
    assert {key for key in base if tagged[key] != base[key]} == {"chunks"}


def test_png_outputs_decoded_refuses_an_undecodable_png(tmp_path):
    (tmp_path / "000.png").write_bytes(bench_oracle.PNG_SIGNATURE + b"garbage")
    with pytest.raises(RuntimeError, match=r"OpenCV cannot decode .*000\.png"):
        bench_oracle.png_outputs(tmp_path, "decoded")


def test_png_outputs_decoded_refuses_a_file_that_is_not_a_png(tmp_path):
    # OpenCV would decode this JPEG; the name says PNG, so the chunks cannot be read.
    ok, data = cv2.imencode(".jpg", pixels((8, 8, 3), np.uint8))
    assert ok
    (tmp_path / "000.png").write_bytes(data.tobytes())
    with pytest.raises(RuntimeError, match=r"000\.png is not a PNG file"):
        bench_oracle.png_outputs(tmp_path, "decoded")


def test_colour_chunks_refuse_a_truncated_chunk(tmp_path):
    # A whole header announcing 100 payload bytes, then only 8 bytes in all.
    data = bench_oracle.PNG_SIGNATURE + struct.pack(">I4s", 100, b"gAMA") + bytes(8)
    with pytest.raises(RuntimeError, match=r"Truncated PNG chunk b'gAMA'"):
        bench_oracle.colour_chunks(tmp_path / "x.png", data)
    with pytest.raises(RuntimeError, match=r"Truncated PNG chunk header"):
        bench_oracle.colour_chunks(tmp_path / "x.png", data[:12])


def test_png_outputs_refuse_an_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="unknown PNG mode 'pixels'"):
        bench_oracle.png_outputs(tmp_path, "pixels")


def test_bounded_video_oracle_matches_raw_decode(tmp_path):
    clip = tmp_path / "clip.mkv"
    verify_runtime.run_owned(
        [
            str(bench_oracle.FFMPEG),
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=30",
            "-frames:v",
            "12",
            "-an",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "bgr0",
            str(clip),
        ],
        timeout=120,
    )
    bounded = bench_oracle.video_snapshot(
        clip, bench_oracle.FFMPEG, bench_oracle.FFPROBE
    )
    raw = verify_video_runtime.snapshot(clip, bench_oracle.FFMPEG, bench_oracle.FFPROBE)
    assert bounded["semantic"] == raw["semantic"]
    assert bounded["semantic"]["decoded_frame_count"] == 12


def test_video_oracle_rejects_non_video_streams(tmp_path):
    clip = tmp_path / "audio.mkv"
    verify_runtime.run_owned(
        [
            str(bench_oracle.FFMPEG),
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=30",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=mono",
            "-t",
            "0.4",
            "-c:v",
            "ffv1",
            "-c:a",
            "pcm_s16le",
            str(clip),
        ],
        timeout=120,
    )
    with pytest.raises(AssertionError, match="only video streams"):
        bench_oracle.video_snapshot(clip, bench_oracle.FFMPEG, bench_oracle.FFPROBE)


def test_pinned_tools_match_recorded_builds():
    bench_oracle.require_pinned_tools()
    assert Path(bench_oracle.FFPROBE).name == "ffprobe.exe"
