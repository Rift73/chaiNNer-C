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
from chain.json import JsonNode, parse_json
from chain.optimize import optimize
from engine_differential import execute, mismatches
from nodes.impl.native_graph import graph
from verify_runtime import edge, make_node

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"

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


@pytest.fixture(scope="module")
def schemas():
    main = sys.modules["__main__"]
    previous = getattr(main, "__file__", None)
    main.__file__ = str(BACKEND / "run.py")
    importlib.import_module("packages.chaiNNer_standard")
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
