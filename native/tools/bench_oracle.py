"""Exactness and completion oracles for benchmark trials.

Compared trees must produce identical outputs. PNG files are compared by the
SHA-256 of their bytes or, in png_outputs' "decoded" mode, by the pixels OpenCV
reads back (shape, dtype and every byte) plus colour/orientation chunks, so two
encoders of the same pixels and metadata agree. Videos are compared by decoded
BGR24 frames plus selected stream metadata, because container bytes may differ.
SSE logs are reduced to a normalized final UI state (verify_runtime.sse_contract)
and checked for a complete, correctly ordered tail.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import struct
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from verify_runtime import final_broadcasts, run_owned

FFMPEG = Path(r"C:\Executables\ffmpeg-8.1-full_build-shared\bin\ffmpeg.exe")
FFPROBE = FFMPEG.with_name("ffprobe.exe")
TOOL_SHA256 = {
    FFMPEG: "861bdc00650c3cdafb2328a0afdcee26060aed3afc89d77373d570fb763f73f7",
    FFPROBE: "e967bd7865680e2c0244b9247bebe36569c708fc105f4d858d3fb73933502ff4",
}
GENERATORS = frozenset({"chainner:image:load_images", "chainner:image:load_video"})
STREAM_KEYS = frozenset(
    {
        "codec_name",
        "codec_type",
        "profile",
        "pix_fmt",
        "width",
        "height",
        "r_frame_rate",
        "avg_frame_rate",
        "time_base",
        "start_time",
        "duration",
        "nb_frames",
        "sample_rate",
        "channels",
        "channel_layout",
        "bits_per_raw_sample",
    }
)
FORMAT_KEYS = frozenset({"nb_streams", "format_name", "start_time", "duration"})
# png_outputs' modes, each with what it compares, as summary.md names it.
PNG_MODES = {
    "sha256": "SHA-256 of the file bytes",
    "decoded": "decoded pixels (shape, dtype and every byte) plus colour/orientation "
    "chunks",
}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# Ancillary chunks that change how a viewer shows the pixels (colour space,
# significant bits, transparency key, EXIF orientation); cv2.imread applies none.
COLOUR_CHUNKS = frozenset(
    {b"gAMA", b"cHRM", b"sRGB", b"iCCP", b"sBIT", b"tRNS", b"eXIf"}
)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_pinned_tools() -> None:
    """The decoded-video oracle is only exact for the recorded FFmpeg build."""
    for tool, expected in TOOL_SHA256.items():
        if sha256_file(tool) != expected:
            raise RuntimeError(f"Unexpected tool build: {tool}")


def png_outputs(directory: Path, mode: str = "sha256") -> dict[str, str | dict]:
    """Every PNG an image trial wrote, keyed by file name, identified by mode.

    sha256 (the default): the SHA-256 of the file's bytes, so two encoders of the
    same pixels differ. decoded: pixels plus colour/orientation chunks, as
    {"shape", "dtype", "pixels_sha256", "chunks"} (see decoded_png); equal records
    are equal pixels, every byte, with no tolerance, and the same COLOUR_CHUNKS.
    """
    paths = sorted(directory.glob("*.png"))
    if mode == "sha256":
        return {path.name: sha256_file(path) for path in paths}
    if mode == "decoded":
        return {path.name: decoded_png(path) for path in paths}
    raise ValueError(f"unknown PNG mode {mode!r}; one of {', '.join(PNG_MODES)}")


def decoded_png(path: Path) -> dict:
    """One PNG's pixels and colour/orientation chunks.

    The pixels are the array cv2.imread(path, IMREAD_UNCHANGED) returns: its
    shape, dtype and the SHA-256 of its C-order bytes, which with shape and dtype
    fix every sample. "chunks" is colour_chunks() of the file. A file that is not
    a PNG, or that OpenCV cannot decode, raises.
    """
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise RuntimeError(f"{path} is not a PNG file")
    # imread returns None for a file it cannot read; OpenCV's stub leaves that out.
    pixels = cast("np.ndarray | None", cv2.imread(str(path), cv2.IMREAD_UNCHANGED))
    if pixels is None:
        raise RuntimeError(f"OpenCV cannot decode {path}")
    return {
        "shape": list(pixels.shape),
        "dtype": pixels.dtype.name,
        "pixels_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
        "chunks": colour_chunks(path, data),
    }


def colour_chunks(path: Path, data: bytes) -> list[list[str]]:
    """[type, SHA-256 of payload] of each COLOUR_CHUNKS chunk in a PNG, sorted.

    Sorted, because their order does not change how the pixels are shown. A chunk
    that runs past the end of the file raises.
    """
    found: list[list[str]] = []
    offset = len(PNG_SIGNATURE)
    while offset < len(data):
        if offset + 12 > len(data):
            raise RuntimeError(f"Truncated PNG chunk header in {path} at {offset}")
        length, kind = struct.unpack_from(">I4s", data, offset)
        end = offset + 12 + length
        if end > len(data):
            raise RuntimeError(f"Truncated PNG chunk {kind!r} in {path} at {offset}")
        if kind in COLOUR_CHUNKS:
            payload = data[offset + 8 : end - 4]
            found.append([kind.decode("ascii"), hashlib.sha256(payload).hexdigest()])
        if kind == b"IEND":
            break
        offset = end
    return sorted(found)


def video_snapshot(path: Path, ffmpeg: Path, ffprobe: Path) -> dict:
    """Decoded-content identity of a video without loading frames into Python."""
    probe = json.loads(
        run_owned(
            [
                str(ffprobe),
                "-v",
                "error",
                "-count_frames",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ],
            timeout=180,
        )
    )
    streams = [
        {k: v for k, v in stream.items() if k in STREAM_KEYS}
        for stream in probe["streams"]
    ]
    if any(stream["codec_type"] != "video" for stream in streams):
        raise AssertionError("Benchmark videos must contain only video streams")
    decoded = (
        run_owned(
            [
                str(ffmpeg),
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-c:v",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-sws_flags",
                "lanczos+accurate_rnd+full_chroma_int+full_chroma_inp+bitexact",
                "-f",
                "hash",
                "-hash",
                "sha256",
                "pipe:1",
            ],
            timeout=180,
        )
        .decode()
        .strip()
    )
    if not decoded.startswith("SHA256="):
        raise AssertionError(f"Unexpected FFmpeg hash output: {decoded}")
    return {
        "sha256": sha256_file(path),
        "semantic": {
            "streams": streams,
            "format": {k: v for k, v in probe["format"].items() if k in FORMAT_KEYS},
            "decoded_frame_count": int(probe["streams"][0]["nb_read_frames"]),
            "decoded_frames_sha256": decoded.removeprefix("SHA256="),
        },
    }


def contract_difference(expected: dict, actual: dict) -> str:
    """Name where two different sse_contract() results first differ.

    Returns "controls" if the control events differ, otherwise the first
    differing final-state event as (event, nodeId), taken from `actual` unless
    `actual` has no event at that position.
    """
    if expected["controls"] != actual["controls"]:
        return "controls"
    pairs = itertools.zip_longest(expected["final_state"], actual["final_state"])
    for want, got in pairs:
        if want != got:
            event = want if got is None else got
            return repr((event["event"], event["data"]["nodeId"]))
    raise ValueError("SSE contracts are identical")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_record(
    nodes: list[dict], events: list[dict], schemas: dict, count: int, output: Path
) -> dict:
    """Check one generator graph's complete SSE tail; return its position summary."""
    by_id = {node["id"]: node for node in nodes}
    _require(bool(events), "Empty event capture")
    _require(
        not any(e["event"] == "execution-error" for e in events),
        "Captured execution error",
    )
    starts = [e for e in events if e["event"] == "chain-start"]
    _require(len(starts) == 1, "Expected one chain-start boundary")
    _require(set(starts[0]["data"]["nodes"]) == set(by_id), "Wrong chain node set")
    finished = {e["data"]["nodeId"] for e in events if e["event"] == "node-finish"}
    _require(finished == set(by_id), "Missing or foreign node completion")
    broadcasts = {e["data"]["nodeId"] for e in events if e["event"] == "node-broadcast"}
    expected = {n["id"] for n in nodes if schemas[n["schemaId"]]["outputs"]}
    _require(broadcasts == expected, "Missing or foreign node broadcast")
    merged = final_broadcasts(events, output)
    generators = [n for n in nodes if schemas[n["schemaId"]]["kind"] == "generator"]
    _require(len(generators) == 1, "Expected exactly one generator per graph")
    node = generators[0]
    _require(node["schemaId"] in GENERATORS, "Unreviewed generator contract")
    schema = schemas[node["schemaId"]]
    node_id = node["id"]
    outputs = {str(o["id"]): o for o in schema["outputs"]}
    iterable = {str(i) for seq in schema["iteratorOutputs"] for i in seq["outputs"]}
    static = set(outputs) - iterable
    own = [(i, e) for i, e in enumerate(events) if e["data"].get("nodeId") == node_id]
    progress = [(i, e["data"]) for i, e in own if e["event"] == "node-progress"]
    _require(bool(progress), "No generator progress")
    last_progress, data = progress[-1]
    _require(
        data["progress"] == 1 and data["index"] == count and data["total"] == count,
        "Incomplete final generator progress",
    )
    finishes = [i for i, e in own if e["event"] == "node-finish"]
    _require(len(finishes) == 1, "Generator finish must be unique")
    _require(finishes[0] > last_progress, "Generator finish precedes final progress")
    # Final restoration carries only static outputs and no sequence types; the
    # initial broadcast has the same fields but precedes the finish.
    restored = [
        (i, e["data"])
        for i, e in own
        if i > finishes[0]
        and e["event"] == "node-broadcast"
        and set(e["data"].get("data") or {}) == static
        and set(e["data"].get("types") or {}) == static
        and not e["data"].get("sequenceTypes")
    ]
    _require(len(restored) == 1, "Missing unique final static-output restoration")
    index_id = next(k for k, o in outputs.items() if o["label"] == "Index")
    _require(
        merged[node_id]["types"][index_id]
        == {"type": "numeric-literal", "value": count - 1},
        "Last visible item/frame index is incomplete",
    )
    directory_id = next(
        k for k in static if outputs[k]["label"] in {"Directory", "Video Directory"}
    )
    source = Path(node["inputs"][0]["value"])
    expected_directory = (
        source.parent if node["schemaId"].endswith("load_video") else source
    )
    actual = restored[0][1]["types"][directory_id]["fields"]["path"]["value"]
    _require(Path(actual) == expected_directory, "Wrong restored input directory")
    collectors = {}
    for candidate in nodes:
        if schemas[candidate["schemaId"]]["kind"] == "collector":
            positions = [
                i
                for i, e in enumerate(events)
                if e["event"] == "node-finish"
                and e["data"]["nodeId"] == candidate["id"]
            ]
            _require(len(positions) == 1, "Collector finish must be unique")
            _require(
                positions[0] > finishes[0],
                "Collector closed before generator completion",
            )
            collectors[candidate["id"]] = positions[0]
    return {
        "node_id": node_id,
        "event_count": len(events),
        "finish_position": finishes[0],
        "static_restoration_position": restored[0][0],
        "collector_finish_positions": collectors,
    }
