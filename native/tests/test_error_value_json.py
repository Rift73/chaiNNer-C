"""A failed run's reports of its inputs are JSON, whatever the input: JSON has no infinity or NaN, and the renderer that
cannot parse a reply shows no error at all. Image Metrics of an image against itself gives a real infinite PSNR."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest
from engine_differential import execute
from verify_runtime import edge, make_node

from chain.json import JsonNode, parse_json
from chain.optimize import optimize
from nodes.properties.inputs import AnyInput, NumberInput, TextInput


@pytest.mark.parametrize(
    "value,shown",
    [(math.inf, "inf"), (-math.inf, "-inf"), (math.nan, "nan")],
)
@pytest.mark.parametrize(
    "report",
    [
        NumberInput("N", min=None, max=None, precision="unlimited"),
        AnyInput("A"),
        TextInput("T"),
    ],
    ids=["number", "any", "text"],
)
def test_a_value_json_lacks_is_shown_as_text(report, value: float, shown: str):
    reported = report.get_error_value(value)
    assert reported == {"type": "formatted", "formatString": shown}
    assert json.loads(json.dumps(reported, allow_nan=False)) == reported


@pytest.mark.parametrize(
    "value,literal",
    [
        (1.5, 1.5),
        (2, 2),
        (0.0, 0.0),
        (1e308, 1e308),
        ("inf", "inf"),
        (None, None),
        (True, 1),
        (False, 0),
    ],
)
def test_every_other_value_stays_a_literal(value: object, literal: object):
    reported = AnyInput("A").get_error_value(value)
    assert reported == {"type": "literal", "value": literal}


def failed_run(tmp_path: Path, kind: str, nodes: list[dict[str, object]]):
    chain = parse_json(cast("list[JsonNode]", nodes))
    optimize(chain)
    output = tmp_path / "out"
    output.mkdir(exist_ok=True)
    result = execute(kind, chain, storage=tmp_path / "storage", output=output)
    assert result.error is not None, kind
    return result.error


def metrics_chain(schemas, tmp_path: Path) -> list[dict[str, object]]:
    image = tmp_path / "picture.png"
    source = np.random.default_rng(1).random((48, 48, 3)) * 255
    assert cv2.imwrite(str(image), source.astype(np.uint8))
    return [
        make_node(schemas, "load", "chainner:image:load", {0: str(image)}),
        make_node(
            schemas,
            "metrics",
            "chainner:image:image_metrics",
            {0: edge("load"), 1: edge("load")},
        ),
    ]


def name_from(schemas, source: str, output: Path) -> dict[str, object]:
    return make_node(
        schemas,
        "save",
        "chainner:image:save",
        {0: edge("load"), 1: str(output), 2: None, 3: edge(source), 4: "png", 15: "u8"},
    )


@pytest.mark.parametrize("kind", ["python", "native"])
def test_an_infinite_input_in_the_report_of_a_failed_node(
    schemas, tmp_path: Path, kind: str
):
    nodes = [
        *metrics_chain(schemas, tmp_path),
        make_node(
            schemas,
            "text",
            "chainner:utility:text_append",
            {0: "_", 1: "psnr", 2: edge("metrics", 1)},
        ),
        name_from(schemas, "text", tmp_path / "out"),
    ]
    error = failed_run(tmp_path, kind, nodes)
    assert "infinity" in str(error)
    assert error.inputs[2] == {"type": "formatted", "formatString": "inf"}
    assert json.loads(json.dumps(error.inputs, allow_nan=False))


@pytest.mark.parametrize("kind", ["python", "native"])
def test_a_nan_input_in_the_report_of_a_failed_node(schemas, tmp_path: Path, kind: str):
    nodes = [
        *metrics_chain(schemas, tmp_path),
        make_node(
            schemas,
            "nan",
            "chainner:utility:math",
            {0: edge("metrics", 1), 1: "sub", 2: edge("metrics", 1)},
        ),
        make_node(
            schemas,
            "after",
            "chainner:utility:math",
            {0: edge("nan"), 1: "add", 2: 1},
        ),
        make_node(
            schemas,
            "text",
            "chainner:utility:text_append",
            {0: "_", 1: "a", 2: edge("after")},
        ),
        name_from(schemas, "text", tmp_path / "out"),
    ]
    error = failed_run(tmp_path, kind, nodes)
    assert "NaN is not a valid number" in str(error)
    assert error.inputs[0] == {"type": "formatted", "formatString": "nan"}
    assert json.loads(json.dumps(error.inputs, allow_nan=False))
