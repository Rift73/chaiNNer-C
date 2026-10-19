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
from engine_differential import execute, mismatches, order_violations
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


def test_native_node_error_matches(schemas, assets, tmp_path):
    # A native entry point's NodeError reaches NodeExecutionError through the engine's one exception mapping.
    errors = []
    for kind in ("python", "native"):
        nodes = [
            make_node(schemas, "size", "chainner:utility:math", {0: 1, 1: "div", 2: 0}),
            make_node(schemas, "board", "chainner:image:create_checkerboard", {0: edge("size"), 1: 8, 4: 5}),
            make_node(
                schemas, "save", "chainner:image:save",
                {0: edge("board"), 1: str(tmp_path), 2: None, 3: "x", 4: "png", 15: "u8"},
            ),
        ]
        chain = parse_json(cast("list[JsonNode]", nodes))
        optimize(chain)
        error = execute(kind, chain, storage=assets / "storage").error
        assert error is not None and error.__cause__ is not None, f"{kind}: {error!r}"
        errors.append((type(error).__name__, type(error.__cause__).__name__, str(error), str(error.__cause__)))
    assert errors[0] == errors[1] and errors[0][1] == "ZeroDivisionError"


@pytest.mark.parametrize("case", [*VIDEO_CASES, "video-frames"])
def test_video_chain_matches(schemas, assets, tmp_path, case):
    if not (assets / "moving.mkv").exists():
        pytest.skip(f"the pinned FFmpeg ({FFMPEG}) is not installed")
    assert run_case(schemas, assets, tmp_path, case) == []


@pytest.fixture(scope="module")
def lazy_images(tmp_path_factory):
    """Four PNGs, 83 to 86 pixels wide and 61 to 64 high, that the lazy shapes' comparisons split."""
    folder = tmp_path_factory.mktemp("lazy") / "images"
    folder.mkdir()
    rng = np.random.default_rng(7)
    for index in range(4):
        small = (rng.random((9, 11, 3)) * 255).astype(np.uint8)
        image = cv2.resize(small, (83 + index, 61 + index), interpolation=cv2.INTER_CUBIC)
        assert cv2.imwrite(str(folder / f"{index:03}.png"), image)
    return folder


def lazy_shape(schemas, shape: str, images: Path, output: Path) -> list[dict]:
    """Chains whose lazy inputs are read for some executions of their reader only (or that fail in a lazy reader)."""

    def load(fail_fast: int = 1) -> dict:
        return make_node(schemas, "load", "chainner:image:load_images", {0: str(images), 2: 0, 4: 0, 6: fail_fast})

    def node(node_id: str, schema: str, inputs: dict) -> dict:
        return make_node(schemas, node_id, f"chainner:{schema}", inputs)

    def save(node_id: str, image: dict, name: object, skip: int = 0, folder: Path = output) -> dict:
        return node(node_id, "image:save", {0: image, 1: str(folder), 2: None, 3: name, 4: "png", 15: "u8", 1000: skip})

    def blur(node_id: str, image: dict, radius: float) -> dict:
        return node(node_id, "image:gaussian_blur", {0: image, 1: radius, 2: radius})

    def width_compare(node_id: str, image: dict, width: int) -> list[dict]:
        return [node("dims", "image:get_dims", {0: image}), node(node_id, "utility:compare", {0: 2, 1: edge("dims", 0), 2: width})]

    unwritable = output / "blocker.txt" / "sub"  # a folder under a file cannot be created
    shapes = {
        # Save Image skips the existing 001.png: its lazy image (the blur) is not computed for that item.
        "skip-existing-loop": lambda: [
            load(), blur("blur", edge("load"), 2.0), save("save", edge("blur"), edge("load", 3), skip=1),
        ],
        "conditional-loop": lambda: [
            load(), *width_compare("cmp", edge("load"), 84),
            blur("b1", edge("load"), 1.0), blur("b2", edge("load"), 4.0),
            node("cond", "utility:conditional", {0: edge("cmp"), 1: edge("b1"), 2: edge("b2")}),
            save("save", edge("cond"), edge("load", 3)),
        ],
        # Logic Operation's B is read only when A does not decide the result.
        "logic-conditional-loop": lambda: [
            load(), *width_compare("cmp1", edge("load"), 83),
            node("cmp2", "utility:compare", {0: 3, 1: edge("dims", 1), 2: 64}),
            node("logic", "utility:logic_operation", {0: "and", 1: edge("cmp1"), 2: edge("cmp2")}),
            blur("b1", edge("load"), 1.0),
            node("cond", "utility:conditional", {0: edge("logic"), 1: edge("b1"), 2: edge("load")}),
            save("save", edge("cond"), edge("load", 3)),
        ],
        # The taken branch's crop fails for some items; the loop goes on (Stop on first error off).
        "deferred-lazy-errors": lambda: [
            load(fail_fast=0), *width_compare("cmp", edge("load"), 84),
            node("crop", "image:crop", {0: edge("load"), 1: 0, 2: 70, 3: 70, 4: 70, 5: 70}),
            node("cond", "utility:conditional", {0: edge("cmp"), 1: edge("crop"), 2: edge("load")}),
            save("save", edge("cond"), edge("load", 3)),
        ],
        "conditional-skip-existing": lambda: [
            node("one", "image:load", {0: str(images / "002.png")}), *width_compare("cmp", edge("one"), 84),
            blur("b1", edge("one"), 1.0), blur("b2", edge("one"), 4.0),
            node("cond", "utility:conditional", {0: edge("cmp"), 1: edge("b2"), 2: edge("b1")}),
            save("save", edge("cond"), "single"), save("save2", edge("b2"), "other", skip=1),
        ],
        "unwritable-folder": lambda: [
            node("one", "image:load", {0: str(images / "002.png")}), blur("b1", edge("one"), 1.0),
            save("save", edge("b1"), "x", folder=unwritable),
        ],
        "unwritable-folder-loop": lambda: [
            load(fail_fast=0), blur("b1", edge("load"), 1.0), save("save", edge("b1"), edge("load", 3), folder=unwritable),
        ],
    }
    return shapes[shape]()


# Per shape, lazy-only upstream nodes that the oracle runs for fewer executions than their reader's.
LAZY_SKIPS = {
    "skip-existing-loop": {"blur": "save"},
    "conditional-loop": {"b1": "cond", "b2": "cond"},
    "logic-conditional-loop": {"cmp2": "logic", "b1": "cond"},
    "deferred-lazy-errors": {},
    "conditional-skip-existing": {"b1": "cond"},
    "unwritable-folder": {"b1": "save"},
    "unwritable-folder-loop": {"b1": "save"},
}


@pytest.mark.parametrize("shape", LAZY_SKIPS)
def test_lazy_shapes_match(schemas, lazy_images, tmp_path, shape):
    # Both executors, the engine twice (its second run from the memo), with the order invariants checked on the
    # oracle's events too: a lazy input's node runs inside the reader execution that reads it, not the k-th one.
    output = tmp_path / "out"
    results = []
    for kind in ("python", "native", "native"):
        shutil.rmtree(output, ignore_errors=True)
        output.mkdir(parents=True)
        shutil.copyfile(lazy_images / "001.png", output / "001.png")
        shutil.copyfile(lazy_images / "001.png", output / "other.png")
        (output / "blocker.txt").write_bytes(b"x")
        chain = parse_json(cast("list[JsonNode]", lazy_shape(schemas, shape, lazy_images, output)))
        optimize(chain)
        results.append((chain, execute(kind, chain, storage=tmp_path / "storage", output=output)))
    (chain, oracle), *engines = results
    starts = {}
    for event in oracle.events:
        if event["event"] == "node-start":
            starts[event["data"]["nodeId"]] = starts.get(event["data"]["nodeId"], 0) + 1
    for upstream, reader in LAZY_SKIPS[shape].items():
        assert starts.get(upstream, 0) < starts[reader], (upstream, starts)
    assert order_violations(oracle.events, chain) == []
    for engine_chain, engine in engines:
        assert mismatches(shape, engine_chain, oracle, engine, tmp_path) == []
