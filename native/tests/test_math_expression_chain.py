"""Math Expression through both executors and the memo cache: an empty input arrives as None on each, a value and no
value have different memo keys, and each run's answer names the file saved from it."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest
from engine_differential import execute, mismatches
from verify_runtime import edge, make_node

from chain.json import JsonNode, parse_json
from chain.optimize import optimize
from nodes.impl.native_graph import graph

SCHEMA = "chainner:utility:math_expression"


def named_chain(
    schemas, image: Path, output: Path, values: dict[int, object]
) -> list[JsonNode]:
    """Load Image saved under the expression's answer; a number node can feed the expression."""
    return cast(
        "list[JsonNode]",
        [
            make_node(schemas, "load", "chainner:image:load", {0: str(image)}),
            make_node(schemas, "number", "chainner:utility:number", {0: 7}),
            make_node(schemas, "answer", SCHEMA, values),
            make_node(
                schemas,
                "name",
                "chainner:utility:text_append",
                {0: "_", 1: "answer", 2: edge("answer")},
            ),
            make_node(
                schemas,
                "save",
                "chainner:image:save",
                {
                    0: edge("load"),
                    1: str(output),
                    2: None,
                    3: edge("name"),
                    4: "png",
                    15: "u8",
                },
            ),
        ],
    )


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "picture.png"
    assert cv2.imwrite(str(path), np.full((4, 4, 3), 90, np.uint8))
    return path


def run(kind: str, nodes: list[JsonNode], tmp_path: Path, output: Path):
    for path in output.glob("*"):
        path.unlink()
    output.mkdir(exist_ok=True)
    chain = parse_json(nodes)
    optimize(chain)
    return chain, execute(kind, chain, storage=tmp_path / "storage", output=output)


CASES = [
    # values, the answer's file name
    ({0: "a + b", 1: 2, 2: 3}, "answer_5.png"),
    # b's input is empty beside a value in c; the expression does not read it.
    ({0: "a + c", 1: 2, 3: 3}, "answer_5.png"),
    ({0: "6 * 7"}, "answer_42.png"),
    ({0: "a * 10", 1: edge("number")}, "answer_70.png"),
    ({0: "floor(e)"}, "answer_2.png"),
    ({0: "e", 5: 4}, "answer_4.png"),
]


@pytest.mark.parametrize("values,name", CASES)
def test_empty_inputs_on_both_executors(
    schemas, tmp_path: Path, image: Path, values: dict[int, object], name: str
):
    output = tmp_path / "out"
    nodes = named_chain(schemas, image, output, values)
    _, oracle = run("python", nodes, tmp_path, output)
    assert oracle.error is None
    assert set(oracle.files) == {name}
    chain, engine = run("native", nodes, tmp_path, output)
    assert engine.error is None
    assert engine.files == oracle.files
    assert mismatches("empty inputs", chain, oracle, engine, tmp_path) == []


@pytest.mark.parametrize(
    "expression,message",
    [
        ("a + b", "Name 'a' is not defined"),
        ("9**9**9", "Error evaluating expression '9**9**9'"),
        ("(1).real", "Attribute"),
        ("1 +", "Invalid expression syntax"),
    ],
)
def test_a_refused_expression_is_a_node_error_on_both_executors(
    schemas, tmp_path: Path, image: Path, expression: str, message: str
):
    output = tmp_path / "out"
    nodes = named_chain(schemas, image, output, {0: expression})
    for kind in ("python", "native"):
        _, result = run(kind, nodes, tmp_path, output)
        assert message in str(result.error), kind
        assert result.files == {}, kind


def test_an_infinite_input_from_another_node_on_both_executors(
    schemas, tmp_path: Path, image: Path
):
    # Image Metrics of an image against itself: its PSNR is infinite.
    output = tmp_path / "out"

    def nodes(expression: str) -> list[JsonNode]:
        chain = named_chain(
            schemas, image, output, {0: expression, 1: edge("metrics", 1)}
        )
        chain.append(
            cast(
                "JsonNode",
                make_node(
                    schemas,
                    "metrics",
                    "chainner:image:image_metrics",
                    {0: edge("load"), 1: edge("load")},
                ),
            )
        )
        return chain

    for kind in ("python", "native"):
        _, passed = run(kind, nodes("min(a, 2)"), tmp_path, output)
        assert passed.error is None, kind
        assert set(passed.files) == {"answer_2.png"}, kind
        _, kept = run(kind, nodes("a"), tmp_path, output)
        assert "The result is inf" in str(kept.error), kind
        _, gone = run(kind, nodes("max(a - a, 0)"), tmp_path, output)
        assert "A step of the expression is not a number" in str(gone.error), kind


def test_the_memo_tells_no_value_from_a_value(schemas, tmp_path: Path, image: Path):
    output = tmp_path / "out"
    # Euler's number with e's input empty; the input's own value otherwise, zero included.
    empty = named_chain(schemas, image, output, {0: "floor(e * 10)"})
    zero = named_chain(schemas, image, output, {0: "floor(e * 10)", 5: 0})
    two = named_chain(schemas, image, output, {0: "floor(e * 10)", 5: 2})
    _, first = run("native", empty, tmp_path, output)
    assert first.error is None
    assert set(first.files) == {"answer_27.png"}

    before = graph().memo_stats()["hits"]
    _, again = run("native", empty, tmp_path, output)
    assert graph().memo_stats()["hits"] - before >= 2  # Load Image and Math Expression
    assert again.files == first.files

    _, zeroed = run("native", zero, tmp_path, output)
    assert set(zeroed.files) == {"answer_0.png"}
    _, twice = run("native", two, tmp_path, output)
    assert set(twice.files) == {"answer_20.png"}
    _, back = run("native", empty, tmp_path, output)
    assert back.files == first.files


def test_each_item_of_a_folder_computes_its_own_answer_on_both_executors(
    schemas, tmp_path: Path
):
    images, output = tmp_path / "images", tmp_path / "out"
    images.mkdir()
    for index in range(5):
        assert cv2.imwrite(
            str(images / f"{index}.png"), np.full((4, 4, 3), index, np.uint8)
        )
    nodes = named_chain(
        schemas, images / "0.png", output, {0: "a * a + c", 1: edge("load", 4), 3: 1}
    )
    nodes[0] = make_node(
        schemas,
        "load",
        "chainner:image:load_images",
        {0: str(images), 2: 0, 4: 0, 6: 1},
    )
    expected = {f"answer_{index * index + 1}.png" for index in range(5)}
    _, oracle = run("python", nodes, tmp_path, output)
    assert oracle.error is None
    assert set(oracle.files) == expected
    chain, engine = run("native", nodes, tmp_path, output)
    assert engine.error is None
    assert engine.files == oracle.files
    assert mismatches("items", chain, oracle, engine, tmp_path) == []
