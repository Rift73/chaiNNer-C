"""The native run engine against process.Executor on the bench chains (native/tools/bench_cases.py), each run whole
through both executors in this process on synthetic inputs (native/tools/engine_differential.py): same outcome,
same output bytes, the SSE events as verify_runtime.sse_event_mismatches judges them, and the A6 order invariants
in the engine's events; and no native entry point declines to its Python body on them. resize-real-256's saved request predates the current Save Image inputs, so its chain (Load
Images, Resize 50 % Hermite, Save) is built from the schemas here."""

from __future__ import annotations

import _winapi
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest

import api
import bench_cases
from bench_oracle import FFMPEG
from chain.json import JsonNode, parse_json
from chain.optimize import optimize
from engine_differential import execute, mismatches
from nodes.impl.native_graph import graph
from verify_runtime import edge, make_node

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"
IMAGE_CASES = [case for case in bench_cases.CASES if not case.startswith("video-") and case != "resize-real-256"]
VIDEO_CASES = [case for case in bench_cases.CASES if case.startswith("video-")]


@pytest.fixture(scope="module")
def schemas():
    """The standard package's nodes, registered as the worker registers them; their schemas as /nodes gives them."""
    main = sys.modules["__main__"]
    previous = getattr(main, "__file__", None)
    # Caption loads its font from beside __main__'s file, the app's run.py.
    main.__file__ = str(BACKEND / "run.py")
    importlib.import_module("packages.chaiNNer_standard")
    errors = api.registry.load_nodes(str(BACKEND / "server.py"))
    assert not errors, [e.module for e in errors]
    yield {schema: {"inputs": [i.to_dict() for i in data.inputs], "kind": data.kind} for schema, (data, _) in api.registry.nodes.items()}
    if previous is None:
        del main.__file__
    else:
        main.__file__ = previous


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    """12 smooth-noise JPEGs of odd sizes in images/; with the pinned FFmpeg, a 24-frame clip and a storage folder
    whose ffmpeg/ is that build (a junction, removed link-only), where Save Video finds its integrated FFmpeg."""
    root = tmp_path_factory.mktemp("assets")
    (root / "storage").mkdir()
    images = root / "images"
    images.mkdir()
    rng = np.random.default_rng(73)
    for index in range(12):
        height, width = 61 + 7 * index, 83 + 5 * index
        small = (rng.random((height // 8 + 2, width // 8 + 2, 3)) * 255).astype(np.uint8)
        image = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
        assert cv2.imwrite(str(images / f"{index:03}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if FFMPEG.exists():
        command = [str(FFMPEG), "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30",
                   "-frames:v", "24", "-an", "-c:v", "ffv1", "-pix_fmt", "bgr0", str(root / "moving.mkv")]
        subprocess.run(command, check=True, timeout=120, stdin=subprocess.DEVNULL)
        _winapi.CreateJunction(str(FFMPEG.parent), str(root / "storage" / "ffmpeg"))
    yield root
    if (root / "storage" / "ffmpeg").exists():
        os.rmdir(root / "storage" / "ffmpeg")


def resize_chain(schemas, assets: Path, output: Path) -> list[dict]:
    """resize-real-256's chain: Load Images, Resize 50 % (Hermite), Save as PNG."""
    nodes = [
        make_node(schemas, "load", "chainner:image:load_images", {0: str(assets / "images"), 2: 0, 4: 0, 6: 1}),
        make_node(schemas, "resize", "chainner:image:resize", {0: edge("load"), 1: 0, 2: 50, 3: 1, 4: 1, 5: 5, 6: 0}),
        make_node(
            schemas, "save", "chainner:image:save",
            {0: edge("resize"), 1: str(output), 2: None, 3: edge("load", 3), 4: "png", 15: "u8"},
        ),
    ]
    return nodes


def video_frames_chain(schemas, assets: Path, output: Path) -> list[dict]:
    """video-resize-h264's frames as PNGs: what its writer receives, which the lossy encoder's run-dependent output
    cannot pin."""
    return [
        make_node(schemas, "load", "chainner:image:load_video", {0: str(assets / "moving.mkv"), 1: 0}),
        make_node(schemas, "resize", "chainner:image:resize", {0: edge("load"), 1: 0, 2: 50, 5: 5, 6: False}),
        make_node(
            schemas, "save", "chainner:image:save",
            {0: edge("resize"), 1: str(output), 2: None, 3: edge("load", 1), 4: "png", 15: "u8"},
        ),
    ]


def range_chain(schemas, assets: Path, output: Path) -> list[dict]:
    """A native generator (Range) through native nodes (Math, Create Checkerboard, Text Append) into Save Image."""
    return [
        make_node(schemas, "range", "chainner:utility:range", {0: 1, 1: True, 2: 6, 3: False}),
        make_node(schemas, "size", "chainner:utility:math", {1: "mul", 0: edge("range"), 2: 16}),
        make_node(schemas, "board", "chainner:image:create_checkerboard", {0: edge("size"), 1: edge("size"), 4: 5}),
        make_node(schemas, "name", "chainner:utility:text_append", {0: "-", 1: "board", 2: edge("range")}),
        make_node(
            schemas, "save", "chainner:image:save",
            {0: edge("board"), 1: str(output), 2: None, 3: edge("name"), 4: "png", 15: "u8"},
        ),
    ]


def spritesheet_chain(schemas, assets: Path, output: Path) -> list[dict]:
    """A native generator over an image (Split Spritesheet) into Save Image, named by its index."""
    return [
        make_node(schemas, "board", "chainner:image:create_checkerboard", {0: 90, 1: 64, 4: 7}),
        make_node(schemas, "split", "chainner:image:split_spritesheet", {0: edge("board"), 1: 2, 2: 3}),
        make_node(schemas, "name", "chainner:utility:text_append", {0: "_", 1: "sprite", 2: edge("split", 1)}),
        make_node(
            schemas, "save", "chainner:image:save",
            {0: edge("split"), 1: str(output), 2: None, 3: edge("name"), 4: "png", 15: "u8"},
        ),
    ]


CHAINS = {"resize": resize_chain, "range": range_chain, "spritesheet": spritesheet_chain}


def run_case(schemas, assets: Path, tmp_path: Path, case: str) -> list[str]:
    output = tmp_path / "out"
    storage = assets / "storage"
    results = []
    for kind in ("python", "native"):
        shutil.rmtree(output, ignore_errors=True)
        output.mkdir(parents=True)
        if case in CHAINS:
            nodes = CHAINS[case](schemas, assets, output)
        elif case == "video-frames":
            nodes = video_frames_chain(schemas, assets, output)
        else:
            nodes, _, _ = bench_cases.graph(schemas, case, assets, output)
        chain = parse_json(cast("list[JsonNode]", nodes))
        optimize(chain)
        writers = {
            node.id: (lambda folder=output: sum(1 for p in folder.rglob("*") if p.is_file()))
            for node in chain.nodes.values()
            if node.schema_id == "chainner:image:save"
        }
        declines = graph().native_declines()
        results.append((chain, execute(kind, chain, storage=storage, output=output, writers=writers)))
        assert graph().native_declines() == declines, f"{kind} run declined: {graph().native_declines()}"
    (_, oracle), (chain, engine) = results
    assert oracle.error is None, f"the oracle failed: {oracle.error!r}"
    assert oracle.files, "the oracle wrote nothing"
    return mismatches(case, chain, oracle, engine, tmp_path)


@pytest.mark.parametrize("case", [*CHAINS, *IMAGE_CASES])
def test_image_chain_matches(schemas, assets, tmp_path, case):
    assert run_case(schemas, assets, tmp_path, case) == []


@pytest.mark.parametrize("case", [*VIDEO_CASES, "video-frames"])
def test_video_chain_matches(schemas, assets, tmp_path, case):
    if not (assets / "moving.mkv").exists():
        pytest.skip(f"the pinned FFmpeg ({FFMPEG}) is not installed")
    assert run_case(schemas, assets, tmp_path, case) == []
