"""Is Grayscale's answer steering Conditional, on both executors: each branch that the image picks runs, the other
does not, and a second native run takes the answer from the memo cache."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest
from engine_differential import execute, load_registry_errors, mismatches
from verify_runtime import edge, make_node

import api
from chain.json import JsonNode, parse_json
from chain.optimize import optimize
from nodes.impl.native_graph import graph

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"


@pytest.fixture(scope="module")
def schemas():
    main = sys.modules["__main__"]
    previous = getattr(main, "__file__", None)
    main.__file__ = str(BACKEND / "run.py")
    importlib.import_module("packages.chaiNNer_standard")
    errors = load_registry_errors(BACKEND / "server.py")
    assert not errors, [e.module for e in errors]
    yield {
        schema: {"inputs": [i.to_dict() for i in data.inputs], "kind": data.kind}
        for schema, (data, _) in api.registry.nodes.items()
    }
    if previous is None:
        del main.__file__
    else:
        main.__file__ = previous


def routed_chain(
    schemas, image: Path, tolerance: int, output: Path
) -> list[dict[str, object]]:
    """A grayscale image is inverted, any other flipped; the node of the other branch stays idle."""
    return [
        make_node(schemas, "load", "chainner:image:load", {0: str(image)}),
        make_node(
            schemas,
            "check",
            "chainner:image:is_grayscale",
            {0: edge("load"), 1: tolerance},
        ),
        make_node(schemas, "invert", "chainner:image:invert", {0: edge("load")}),
        make_node(schemas, "flip", "chainner:image:flip", {0: edge("load"), 1: 0}),
        make_node(
            schemas,
            "route",
            "chainner:utility:conditional",
            {0: edge("check"), 1: edge("invert"), 2: edge("flip")},
        ),
        make_node(
            schemas,
            "save",
            "chainner:image:save",
            {
                0: edge("route"),
                1: str(output),
                2: None,
                3: "routed",
                4: "png",
                15: "u8",
            },
        ),
    ]


def picture(channels: int, spread: int) -> np.ndarray:
    """A top-to-bottom ramp whose colour channels lie `spread` levels apart in one pixel."""
    gray = np.tile(np.arange(40, 88, 4, np.uint8)[:, None], (1, 5))
    if channels == 1:
        return gray
    levels = np.dstack([gray] * 3)
    # The blue and green channels only: the red one stays between them.
    levels[3, 2, 0] = levels[3, 2, 2] - spread // 2
    levels[3, 2, 1] = levels[3, 2, 2] + spread - spread // 2
    if channels == 3:
        return levels
    alpha = np.full(gray.shape, 200, np.uint8)
    return np.dstack([levels, alpha])


CASES = [
    # channels, spread, tolerance, grayscale
    (1, 0, 0, True),
    (3, 0, 0, True),
    (4, 0, 0, True),
    (3, 1, 0, False),
    (3, 1, 1, True),
    (3, 20, 10, False),
    (3, 20, 20, True),
    (4, 20, 10, False),
    (4, 20, 20, True),
]


@pytest.mark.parametrize("channels,spread,tolerance,grayscale", CASES)
def test_the_answer_picks_the_branch_on_both_executors(
    schemas,
    tmp_path: Path,
    channels: int,
    spread: int,
    tolerance: int,
    grayscale: bool,
):
    source = picture(channels, spread)
    image, output = tmp_path / "picture.png", tmp_path / "out"
    assert cv2.imwrite(str(image), source)
    nodes = cast("list[JsonNode]", routed_chain(schemas, image, tolerance, output))

    results = {}
    for kind in ("python", "native"):
        output.mkdir()
        chain = parse_json(nodes)
        optimize(chain)
        results[kind] = (
            chain,
            execute(kind, chain, storage=tmp_path / "storage", output=output),
        )
        result = results[kind][1]
        assert result.error is None, kind
        saved = cv2.imread(str(output / "routed.png"), cv2.IMREAD_UNCHANGED)
        colour = source if channels == 1 else source[:, :, :3]
        expected = 255 - colour if grayscale else cv2.flip(colour, 0)
        found = saved if channels == 1 else saved[:, :, :3]
        assert np.array_equal(found, expected), kind
        finished = {
            event["data"]["nodeId"]
            for event in result.events
            if event["event"] == "node-finish"
        }
        assert ("invert" in finished) == grayscale, kind
        assert ("flip" in finished) != grayscale, kind
        for path in output.iterdir():
            path.unlink()
        output.rmdir()

    chain, oracle = results["python"]
    assert results["native"][1].files == oracle.files
    assert mismatches("routed", chain, oracle, results["native"][1], tmp_path) == []


def test_a_second_native_run_reads_the_answer_from_the_memo(schemas, tmp_path: Path):
    image, output = tmp_path / "picture.png", tmp_path / "out"
    assert cv2.imwrite(str(image), picture(3, 0))
    nodes = cast("list[JsonNode]", routed_chain(schemas, image, 0, output))
    chain = parse_json(nodes)
    optimize(chain)
    execute("native", chain, storage=tmp_path / "storage", output=output)
    before = graph().memo_stats()["hits"]
    chain = parse_json(nodes)
    optimize(chain)
    second = execute("native", chain, storage=tmp_path / "storage", output=output)
    assert second.error is None
    # Load Image, Is Grayscale and Invert.
    assert graph().memo_stats()["hits"] - before >= 3


def test_each_item_of_a_folder_gets_its_own_answer_on_both_executors(
    schemas, tmp_path: Path
):
    images, output = tmp_path / "images", tmp_path / "out"
    images.mkdir()
    # Grayscale at each odd name: its answer is the one to change between neighbours.
    sources = {
        "0": picture(3, 20),
        "1": picture(1, 0),
        "2": picture(4, 20),
        "3": picture(3, 0),
        "4": picture(3, 1),
        "5": picture(4, 0),
    }
    for name, source in sources.items():
        assert cv2.imwrite(str(images / f"{name}.png"), source)
    nodes = routed_chain(schemas, images / "0.png", 0, output)
    nodes[0] = make_node(
        schemas,
        "load",
        "chainner:image:load_images",
        {0: str(images), 2: 0, 4: 0, 6: 1},
    )
    nodes[-1]["inputs"][3] = edge("load", 3)
    results = {}
    for kind in ("python", "native", "native"):
        output.mkdir(exist_ok=True)
        chain = parse_json(cast("list[JsonNode]", nodes))
        optimize(chain)
        result = execute(kind, chain, storage=tmp_path / "storage", output=output)
        assert result.error is None, kind
        results[kind] = chain, result
        for name, source in sources.items():
            saved = cv2.imread(str(output / f"{name}.png"), cv2.IMREAD_UNCHANGED)
            colour = source if source.ndim == 2 else source[:, :, :3]
            expected = 255 - colour if int(name) % 2 else cv2.flip(colour, 0)
            found = saved if source.ndim == 2 else saved[:, :, :3]
            assert np.array_equal(found, expected), (kind, name)
    chain, oracle = results["python"]
    assert mismatches("items", chain, oracle, results["native"][1], tmp_path) == []
