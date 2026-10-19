"""The native engine's memo cache (native/local/instant-exec/DESIGN.md amendment A3): its allow-list, hits that send
the events of a computed node (judged against process.Executor by native/tools/engine_differential.py), keys that
follow a source file's identity, and its budget (CHAINNER_C_MEMO_BUDGET_MIB, an entry above an eighth of it not
kept)."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest

import api
import native_executor
from bench_oracle import FFMPEG
from chain.json import JsonNode, parse_json
from chain.optimize import optimize
from engine_differential import execute, mismatches
from nodes.impl.native_graph import graph
from verify_runtime import edge, make_node

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"
PACKAGES = sorted(p.parent.name for p in (BACKEND / "packages").glob("*/__init__.py"))

# The audited Python nodes (deterministic, side-effect free, no file, clock, random, network or GPU reads).
AUDITED = {
    "chainner:image:add_normals",
    "chainner:image:average_color_fix",
    "chainner:image:balance_normals",
    "chainner:image:bilateral_blur",
    "chainner:image:convert_normal_map",
    "chainner:image:generate_hash",
    "chainner:image:metal_to_specular",
    "chainner:image:normal_generator",
    "chainner:image:normalize_normal_map",
    "chainner:image:planes_to_rgb",
    "chainner:image:rgb_to_planes",
    "chainner:image:specular_to_metal",
    "chainner:image:strengthen_normals",
    "chainner:utility:color_from_channels",
    "chainner:utility:derive_seed",
    "chainner:utility:separate_color",
}
# Each generator's per-run (not iterated) outputs, audited as following from its inputs alone: its run is keyed
# (native_executor.MEMO_GENERATORS). A per-run output added to one of them needs a new audit.
KEYED_GENERATORS = {
    "chainner:image:load_dds_images": [],
    "chainner:image:load_images": ["Directory"],
    "chainner:image:split_spritesheet": [],
    "chainner:ncnn:load_models": ["Directory"],
    "chainner:onnx:load_models": ["Directory"],
    "chainner:pytorch:load_models": ["Directory"],
    "chainner:utility:range": [],
    "chainner:vapoursynth:clip_frames": ["FPS", "Colour Info"],
}
# Generators whose per-run outputs are read from a file, so their run has no key.
UNKEYED_GENERATORS = {"chainner:image:load_video"}


@pytest.fixture(scope="module")
def schemas():
    """Every package's nodes, so a node added to any package is held to the allow-list."""
    main = sys.modules["__main__"]
    previous = getattr(main, "__file__", None)
    main.__file__ = str(BACKEND / "run.py")
    for name in PACKAGES:
        importlib.import_module(f"packages.{name}")
    errors = api.registry.load_nodes(str(BACKEND / "server.py"))
    assert not errors, [e.module for e in errors]
    yield {schema: {"inputs": [i.to_dict() for i in data.inputs], "kind": data.kind} for schema, (data, _) in api.registry.nodes.items()}
    if previous is None:
        del main.__file__
    else:
        main.__file__ = previous


def write_images(folder: Path, seed: int, count: int = 5) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for index in range(count):
        small = (rng.random((9, 11, 3)) * 255).astype(np.uint8)
        image = cv2.resize(small, (83 + index, 61 + index), interpolation=cv2.INTER_CUBIC)
        assert cv2.imwrite(str(folder / f"{index:03}.png"), image)


def folder_chain(schemas, images: Path, output: Path) -> list[dict]:
    """Load Images, a native node (Gaussian Blur), an audited Python node (Normal Map Generator), Save Image."""
    return [
        make_node(schemas, "load", "chainner:image:load_images", {0: str(images), 2: 0, 4: 0, 6: 1}),
        make_node(schemas, "blur", "chainner:image:gaussian_blur", {0: edge("load"), 1: 2.5, 2: 1.5}),
        make_node(schemas, "normals", "chainner:image:normal_generator", {0: edge("blur")}),
        make_node(
            schemas, "save", "chainner:image:save",
            {0: edge("normals"), 1: str(output), 2: None, 3: edge("load", 3), 4: "png", 15: "u8"},
        ),
    ]


def single_chain(schemas, image: Path, output: Path) -> list[dict]:
    """Load Image (keyed by its file's identity) into a native node and Save Image."""
    return [
        make_node(schemas, "load", "chainner:image:load", {0: str(image)}),
        make_node(schemas, "blur", "chainner:image:gaussian_blur", {0: edge("load"), 1: 1.5, 2: 1.5}),
        make_node(schemas, "save", "chainner:image:save", {0: edge("blur"), 1: str(output), 2: None, 3: "single", 4: "png", 15: "u8"}),
    ]


def run(kind: str, nodes: list[dict], tmp_path: Path, output: Path):
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    chain = parse_json(cast("list[JsonNode]", nodes))
    optimize(chain)
    return chain, execute(kind, chain, storage=tmp_path / "storage", output=output)


def test_allow_list_is_the_audited_set(schemas):
    assert native_executor.MEMO_PYTHON_NODES == AUDITED
    assert native_executor.MEMO_EXCLUDED_NATIVE == {"chainner:image:add_noise", "chainner:image:create_noise"}
    # Every package's nodes (NVIDIA ModelOpt has none) are in the registry checked below.
    packages = {p.id: p for p in api.registry.packages.values()}
    assert packages.keys() >= set(PACKAGES)
    for package in packages.values():
        loaded = {n.schema_id for c in package.categories for g in c.node_groups for n in g.nodes}
        assert loaded <= api.registry.nodes.keys(), package.id
    native = set(graph().native_schema_ids())
    for schema, (data, _) in api.registry.nodes.items():
        eligible = native_executor.memo_eligible(data, schema in native)
        if schema == native_executor.LOAD_IMAGE:
            assert eligible
        elif data.kind != "regularNode" or data.side_effects or data.node_context:
            assert not eligible, schema
        elif schema in native:
            assert eligible == (schema not in native_executor.MEMO_EXCLUDED_NATIVE), schema
        else:
            assert eligible == (schema in AUDITED), schema


def test_generator_runs_keyed_only_by_their_inputs(schemas):
    # Every generator is decided: keyed with its audited per-run outputs, or not keyed.
    assert native_executor.MEMO_GENERATORS == KEYED_GENERATORS.keys()
    generators = {schema: data for schema, (data, _) in api.registry.nodes.items() if data.kind == "generator"}
    assert generators.keys() == KEYED_GENERATORS.keys() | UNKEYED_GENERATORS
    for schema in KEYED_GENERATORS:
        data = generators[schema]
        per_run = [o.label for o in data.outputs if o.id not in data.single_iterable_output.outputs]
        assert per_run == KEYED_GENERATORS[schema], schema


def video_fps_chain(schemas, video: Path, output: Path) -> list[dict]:
    """Load Video's FPS (a per-run output read from the file), doubled by Math (memoized), names the saved frame."""
    return [
        make_node(schemas, "video", "chainner:image:load_video", {0: str(video), 1: True, 2: 1}),
        make_node(schemas, "math", "chainner:utility:math", {0: edge("video", 4), 1: "mul", 2: 2}),
        make_node(
            schemas, "save", "chainner:image:save",
            {0: edge("video", 0), 1: str(output), 2: None, 3: edge("math", 0), 4: "png", 15: "u8"},
        ),
    ]


def test_a_file_read_per_run_output_is_read_again(schemas, tmp_path):
    # Load Video's run has no key: after the file at its path changes frame rate, Math computes from the new FPS.
    if not FFMPEG.exists():
        pytest.skip(f"the pinned FFmpeg ({FFMPEG}) is not installed")
    video, output = tmp_path / "clip.mkv", tmp_path / "out"

    def write_video(rate: int) -> None:
        command = [str(FFMPEG), "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=64x48:rate={rate}",
                   "-frames:v", "2", "-c:v", "ffv1", str(video)]
        subprocess.run(command, check=True, timeout=120, stdin=subprocess.DEVNULL)

    nodes = video_fps_chain(schemas, video, output)
    write_video(10)
    _, first = run("native", nodes, tmp_path, output)
    assert first.error is None and set(first.files) == {"20.png"}
    write_video(25)
    _, oracle = run("python", nodes, tmp_path, output)
    chain, engine = run("native", nodes, tmp_path, output)
    assert set(oracle.files) == {"50.png"}
    assert engine.files == oracle.files
    assert mismatches("changed frame rate", chain, oracle, engine, tmp_path) == []


def test_an_input_keyed_per_run_output_is_memoized(schemas, tmp_path):
    # Load Images' Directory is its input: its run is keyed, so Text Append on it is a memo hit the second time.
    images, output = tmp_path / "images", tmp_path / "out"
    write_images(images, 7, count=2)
    nodes = [
        make_node(schemas, "load", "chainner:image:load_images", {0: str(images), 2: 0, 4: 0, 6: 1}),
        make_node(schemas, "folder", "chainner:utility:text_append", {0: "-", 1: edge("load", 1), 2: "x"}),
        make_node(schemas, "name", "chainner:utility:text_length", {0: edge("folder")}),
        make_node(schemas, "label", "chainner:utility:text_append", {0: "_", 1: edge("load", 3), 2: edge("name")}),
        make_node(
            schemas, "save", "chainner:image:save",
            {0: edge("load"), 1: str(output), 2: None, 3: edge("label"), 4: "png", 15: "u8"},
        ),
    ]
    _, oracle = run("python", nodes, tmp_path, output)
    run("native", nodes, tmp_path, output)
    before = graph().memo_stats()
    chain, engine = run("native", nodes, tmp_path, output)
    after = graph().memo_stats()
    # Two items' loads and labels, the folder text and its length.
    assert after["hits"] - before["hits"] >= 2 * 2 + 2
    assert engine.files == oracle.files
    assert mismatches("keyed directory", chain, oracle, engine, tmp_path) == []


def test_hits_send_the_events_of_computed_nodes(schemas, tmp_path):
    images, output = tmp_path / "images", tmp_path / "out"
    write_images(images, 1)
    nodes = folder_chain(schemas, images, output)
    _, oracle = run("python", nodes, tmp_path, output)
    _, first = run("native", nodes, tmp_path, output)
    before = graph().memo_stats()
    chain, second = run("native", nodes, tmp_path, output)
    after = graph().memo_stats()
    # Every item's load, blur and normal map come from the memo the second time.
    assert after["hits"] - before["hits"] >= 3 * 5
    assert second.files == first.files == oracle.files
    assert mismatches("memo hits", chain, oracle, second, tmp_path) == []


def test_a_changed_file_is_computed_again(schemas, tmp_path):
    images, output = tmp_path / "images", tmp_path / "out"
    write_images(images, 2)
    nodes = folder_chain(schemas, images, output)
    run("native", nodes, tmp_path, output)
    write_images(images / "changed", 3, count=1)
    os.replace(images / "changed" / "000.png", images / "002.png")
    shutil.rmtree(images / "changed")
    _, oracle = run("python", nodes, tmp_path, output)
    chain, engine = run("native", nodes, tmp_path, output)
    assert mismatches("changed folder", chain, oracle, engine, tmp_path) == []

    image = tmp_path / "single.png"
    write_images(tmp_path / "one", 4, count=1)
    os.replace(tmp_path / "one" / "000.png", image)
    nodes = single_chain(schemas, image, output)
    _, first = run("native", nodes, tmp_path, output)
    write_images(tmp_path / "one", 5, count=1)
    os.replace(tmp_path / "one" / "000.png", image)
    _, oracle = run("python", nodes, tmp_path, output)
    chain, engine = run("native", nodes, tmp_path, output)
    assert engine.files == oracle.files != first.files
    assert mismatches("changed image", chain, oracle, engine, tmp_path) == []


BUDGET = """
import importlib, json, os, sys
from pathlib import Path
sys.path[:0] = [os.environ["TOOLS"]]
sys.modules["__main__"].__file__ = os.path.abspath("run.py")
import api
importlib.import_module("packages.chaiNNer_standard")
api.registry.load_nodes(os.path.abspath("server.py"))
from chain.json import parse_json
from chain.optimize import optimize
from engine_differential import execute
from nodes.impl.native_graph import graph
from verify_runtime import edge, make_node
schemas = {s: {"inputs": [i.to_dict() for i in d.inputs], "kind": d.kind} for s, (d, _) in api.registry.nodes.items()}
work = Path(os.environ["WORK"])
size = int(os.environ["SIZE"])
nodes = [
    make_node(schemas, "board", "chainner:image:create_checkerboard", {0: size, 1: size, 4: 3}),
    make_node(schemas, "blur", "chainner:image:gaussian_blur", {0: edge("board"), 1: 1, 2: 1}),
    make_node(schemas, "save", "chainner:image:save", {0: edge("blur"), 1: str(work), 2: None, 3: "b", 4: "png", 15: "u8"}),
]
for _ in range(2):
    chain = parse_json(nodes)
    optimize(chain)
    assert execute("native", chain, storage=work / "storage").error is None
print(json.dumps(graph().memo_stats()))
"""


@pytest.mark.parametrize(
    "budget, size, cached",
    [("0", 64, False), ("1", 64, True), ("1", 256, False)],
    ids=["off", "small entry kept", "entry above an eighth not kept"],
)
def test_budget(tmp_path, budget, size, cached):
    # A fresh interpreter: the budget is read once per process.
    environment = dict(os.environ, CHAINNER_C_MEMO_BUDGET_MIB=budget, WORK=str(tmp_path), SIZE=str(size),
                       TOOLS=str(BACKEND.parents[1] / "native" / "tools"))
    result = subprocess.run(
        [sys.executable, "-B", "-c", textwrap.dedent(BUDGET)], cwd=BACKEND, env=environment, capture_output=True,
        text=True, timeout=300, check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    stats = __import__("json").loads(result.stdout.strip().splitlines()[-1])
    assert stats["budget"] == int(budget) << 20
    # A 256 x 256 RGBA float32 image is 1 MiB, above an eighth of a 1 MiB budget; a 64 x 64 one fits.
    assert (stats["hits"] > 0) == cached, stats


def test_memo_bytes_count_a_views_whole_base():
    # A view keeps its base alive: a memo entry counts the base, once however many views share it.
    base = np.zeros((100, 10, 3), dtype=np.float32)
    assert native_executor.memo_bytes([base[:5]]) == base.nbytes
    assert native_executor.memo_bytes([base[:5], base[5:10][:, :2]]) == base.nbytes
    assert native_executor.memo_bytes([base[:5], b"1234"]) == base.nbytes + 4


VIEWS = """
import gc, importlib, json, os, sys
from pathlib import Path
import cv2, numpy as np, psutil
sys.path[:0] = [os.environ["TOOLS"]]
sys.modules["__main__"].__file__ = os.path.abspath("run.py")
import api
importlib.import_module("packages.chaiNNer_standard")
api.registry.load_nodes(os.path.abspath("server.py"))
from chain.json import parse_json
from chain.optimize import optimize
from engine_differential import execute
from nodes.impl import numpy_pool
from nodes.impl.native_graph import graph
from verify_runtime import edge, make_node
schemas = {s: {"inputs": [i.to_dict() for i in d.inputs], "kind": d.kind} for s, (d, _) in api.registry.nodes.items()}
work = Path(os.environ["WORK"])
images, output = work / "in", work / "out"
images.mkdir()
output.mkdir()
rng = np.random.default_rng(1)
count = 16
for i in range(count):
    small = (rng.random((33, 33, 3)) * 255).astype(np.uint8)
    cv2.imwrite(str(images / f"{i:02}.png"), cv2.resize(small, (1024, 1024), interpolation=cv2.INTER_CUBIC))
# Crop to 24 full-width rows: a view of the loaded image (native Crop) that would keep it alive in the memo.
crop = {1: 1, 4: 0, 3: 500, 6: 0, 5: 500}
if os.environ["SHAPE"] == "single":
    nodes = []
    for i in range(count):
        nodes.append(make_node(schemas, f"load{i}", "chainner:image:load", {0: str(images / f"{i:02}.png")}))
        nodes.append(make_node(schemas, f"crop{i}", "chainner:image:crop", {0: edge(f"load{i}"), **crop}))
        nodes.append(make_node(schemas, f"save{i}", "chainner:image:save",
                               {0: edge(f"crop{i}"), 1: str(output), 2: None, 3: f"o{i}", 4: "png", 15: "u8"}))
else:
    nodes = [
        make_node(schemas, "load", "chainner:image:load_images", {0: str(images), 2: 0, 4: 0, 6: 1}),
        make_node(schemas, "crop", "chainner:image:crop", {0: edge("load"), **crop}),
        make_node(schemas, "save", "chainner:image:save",
                  {0: edge("crop"), 1: str(output), 2: None, 3: edge("load", 3), 4: "png", 15: "u8"}),
    ]
chain = parse_json(nodes)
optimize(chain)
process = psutil.Process()
numpy_pool.release()
gc.collect()
base = process.memory_info().private
assert execute("native", chain, storage=work / "storage").error is None
gc.collect()
numpy_pool.release()
gc.collect()
print(json.dumps({"held": process.memory_info().private - base, **graph().memo_stats()}))
"""


@pytest.mark.parametrize("shape", ["single", "folder"])
def test_views_keep_the_memo_within_its_budget(tmp_path, shape):
    # A full-width crop is a view of its loaded image: its memo entry counts that image, so what the memo keeps
    # alive stays within the budget (16 images of 12 MiB under a 64 MiB budget: none kept here).
    environment = dict(os.environ, CHAINNER_C_MEMO_BUDGET_MIB="64", WORK=str(tmp_path), SHAPE=shape,
                       TOOLS=str(BACKEND.parents[1] / "native" / "tools"), CUDA_VISIBLE_DEVICES="-1")
    result = subprocess.run(
        [sys.executable, "-B", "-c", textwrap.dedent(VIEWS)], cwd=BACKEND, env=environment, capture_output=True,
        text=True, timeout=600, check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    stats = __import__("json").loads(result.stdout.strip().splitlines()[-1])
    assert stats["bytes"] <= stats["budget"] == 64 << 20
    assert stats["held"] < (64 + 48) << 20, stats


def test_a_file_an_earlier_sink_writes_is_loaded_fresh(schemas, tmp_path):
    # Sink 1 writes x.png, sink 2 loads it: Load Image reads (and keys) the file as it is when it runs, after sink 1,
    # also on a second run whose sink 1 writes other content (the memo then holds the first run's load).
    folder, output = tmp_path / "files", tmp_path / "out"
    folder.mkdir()
    write_images(tmp_path / "old", 6, count=1)
    old = tmp_path / "old" / "000.png"

    def chain_nodes(size: int) -> list[dict]:
        return [
            make_node(schemas, "board", "chainner:image:create_checkerboard", {0: size, 1: size, 4: 3}),
            make_node(schemas, "save_x", "chainner:image:save",
                      {0: edge("board"), 1: str(folder), 2: None, 3: "x", 4: "png", 15: "u8"}),
            make_node(schemas, "load", "chainner:image:load", {0: str(folder / "x.png")}),
            make_node(schemas, "save_y", "chainner:image:save",
                      {0: edge("load"), 1: str(output), 2: None, 3: "y", 4: "png", 15: "u8"}),
        ]

    loaded = []
    for size in (64, 96):
        nodes = chain_nodes(size)
        shutil.copyfile(old, folder / "x.png")  # each run finds the old x.png, which its sink 1 replaces
        _, oracle = run("python", nodes, tmp_path, output)
        shutil.copyfile(old, folder / "x.png")
        chain, engine = run("native", nodes, tmp_path, output)
        assert engine.files == oracle.files
        assert mismatches("write then load", chain, oracle, engine, tmp_path) == []
        loaded.append(engine.files)
    assert loaded[0] != loaded[1]
