"""Deterministic benchmark inputs derived from the user's read-only dataset.

Every dataset file listed in bench-data/inputs.json is hashed before use.
Generated fixtures must match bench-data/fixtures.json, so every run measures
the same inputs: JPEGs by SHA-256, moving.mkv by its decoded frames (its
container bytes vary with Matroska UIDs and are not pinned).
Nothing is written outside the run's own media directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from bench_baseline import changed_paths
from bench_cases import DATA, IMAGE_COUNT, VIDEO_FRAMES
from bench_oracle import (
    FFMPEG,
    FFPROBE,
    TOOL_SHA256,
    require_pinned_tools,
    sha256_file,
    video_snapshot,
)
from verify_runtime import run_owned

PINS = DATA / "fixtures.json"
VIDEO_IDENTITY = ("decoded_frame_count", "decoded_frames_sha256")


@dataclass(frozen=True)
class Fixtures:
    assets: Path
    record: dict


def load_inputs() -> list[dict]:
    return json.loads((DATA / "inputs.json").read_text(encoding="utf-8"))


def pinned_output_names() -> set[str]:
    """PNG names resize-real-256 must produce: one per pinned input, by stem."""
    return {Path(item["path"]).stem + ".png" for item in load_inputs()}


def verify_inputs(inputs: list[dict]) -> None:
    """Raise naming the first missing or changed dataset file."""
    for item in inputs:
        path = Path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark input missing: {path}")
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Benchmark input changed: {path}")


def check_pins(actual: dict[str, str], pinned: dict[str, str]) -> None:
    """Raise naming the first value that differs from its pin or lacks one."""
    changed = changed_paths(actual, pinned)
    if changed:
        raise RuntimeError(
            f"Generated fixture differs from {PINS.name} at {changed[0]} "
            f"({len(changed)} differ)"
        )


def prepare(media: Path) -> Fixtures:
    """Write images/ and moving.mkv to media/fixtures and check them against the pins.

    Dataset inputs and the FFmpeg builds are verified first.
    """
    inputs = load_inputs()
    verify_inputs(inputs)
    require_pinned_tools()
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    assets = media / "fixtures"
    images = assets / "images"
    images.mkdir(parents=True)
    cv2.ocl.setUseOpenCL(False)
    for index, item in enumerate(inputs[:IMAGE_COUNT]):
        decoded = cv2.imdecode(np.fromfile(item["path"], np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError(f"Cannot decode benchmark input: {item['path']}")
        image = cv2.resize(decoded, (512, 384), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError(f"Cannot encode fixture {index}")
        (images / f"{index:03}.jpg").write_bytes(encoded.tobytes())
    check_pins(
        {f"images/{p.name}": sha256_file(p) for p in sorted(images.iterdir())},
        pins["images"],
    )
    command = [
        str(FFMPEG),
        "-nostdin",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=1280x720:rate=30",
        "-frames:v",
        str(VIDEO_FRAMES),
        "-an",
        "-c:v",
        "ffv1",
        "-pix_fmt",
        "bgr0",
        "-threads",
        "8",
        str(assets / "moving.mkv"),
    ]
    run_owned(command, timeout=600)
    semantic = video_snapshot(assets / "moving.mkv", FFMPEG, FFPROBE)["semantic"]
    decoded = {key: semantic[key] for key in VIDEO_IDENTITY}
    check_pins(
        {f"moving.mkv {key}": str(value) for key, value in decoded.items()},
        {f"moving.mkv {key}": str(value) for key, value in pins["moving.mkv"].items()},
    )
    record = {
        "input_hashes": {item["path"]: item["sha256"] for item in inputs},
        "assets": {
            p.relative_to(assets).as_posix(): sha256_file(p)
            for p in sorted(assets.rglob("*"))
            if p.is_file()
        },
        "decoded_video": decoded,
        "tools": {str(tool): digest for tool, digest in TOOL_SHA256.items()},
        "video_generation": command,
    }
    return Fixtures(assets, record)


def verify_unchanged(fixtures: Fixtures) -> None:
    """Raise if any dataset input or generated asset changed during the run."""
    verify_inputs(
        [{"path": p, "sha256": h} for p, h in fixtures.record["input_hashes"].items()]
    )
    for relative, expected in fixtures.record["assets"].items():
        if sha256_file(fixtures.assets / relative) != expected:
            raise RuntimeError(f"Benchmark fixture changed: {relative}")
