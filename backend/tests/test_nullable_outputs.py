"""Mode-dependent outputs (BaseOutput.null_unless): typed `null` under the dropdown's
other options, and a None value passes the output's enforce there."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Union, cast

import numpy as np
import pytest

from api import NodeData
from api.node_check import CheckFailedError, eval_type, validate_return_type
from nodes.properties.outputs import ImageOutput, NumberOutput
from process import compute_broadcast, enforce_output


def test_null_unless_types_the_output_null_under_the_other_options():
    output = ImageOutput("Y", channels=1).with_id(6)
    image_type = output.output_type
    output.null_unless(4, "Mode::Planes | Mode::Both")
    assert output.nullable
    assert output.output_type == {
        "type": "match",
        "of": "Input4",
        "arms": [
            {"pattern": "Mode::Planes | Mode::Both", "binding": None, "to": image_type},
            {"pattern": "any", "binding": None, "to": "null"},
        ],
    }


def test_a_nullable_output_accepts_none_and_still_enforces_values():
    output = ImageOutput("Y", channels=1).null_unless(4, "Mode::Planes")
    assert output.enforce_(None) is None
    plane = np.full((4, 6), 0.25, np.float32)
    enforced = output.enforce_(plane)
    assert enforced is not None
    assert enforced.shape == (4, 6)
    with pytest.raises(ValueError, match="supposed to return"):
        output.enforce_(np.zeros((4, 6, 3), np.float32))


def test_other_outputs_still_reject_none():
    with pytest.raises(AssertionError):
        ImageOutput("Frame", channels=3).enforce_(None)


def test_a_node_returns_none_for_its_nullable_outputs():
    node = SimpleNamespace(
        name="Test",
        outputs=[
            NumberOutput("Count"),
            ImageOutput("Y", channels=1).null_unless(0, "Mode::Planes"),
        ],
    )
    # enforce_output reads only a node's name and outputs.
    result = enforce_output((3, None), cast(NodeData, node))
    assert result.output == [3, None]


def test_a_failed_broadcast_is_logged_with_its_traceback(
    caplog: pytest.LogCaptureFixture,
):
    output = NumberOutput("Count").with_id(0)

    def fail(_value: object) -> None:
        raise RuntimeError("no broadcast")

    output.get_broadcast_data = fail
    data, _types = compute_broadcast([3], [output])
    assert data == {}
    assert [str(r.exc_info[1]) for r in caplog.records if r.exc_info] == [
        "no broadcast"
    ]


@pytest.mark.parametrize(
    "annotation",
    [
        "Union[tuple[int, int], tuple[float, float]]",
        "tuple[int, int] | tuple[float, float]",
    ],
)
def test_a_return_type_of_alternative_tuples_is_checked_per_output(annotation: str):
    # Each alternative is one tuple of the node's outputs, whichever way the union is written.
    node = SimpleNamespace(outputs=[NumberOutput("A"), NumberOutput("B")])
    return_type = eval_type(annotation, {"Union": Union})
    validate_return_type(return_type, cast(NodeData, node))
    wrong = eval_type(
        annotation.replace("[float, float]", "[float, float, float]"),
        {"Union": Union},
    )
    with pytest.raises(CheckFailedError, match="same number of arguments"):
        validate_return_type(wrong, cast(NodeData, node))


def test_the_type_check_accepts_none_for_a_nullable_output():
    node = SimpleNamespace(
        outputs=[NumberOutput("A"), NumberOutput("B").null_unless(0, "Mode::B")]
    )
    validate_return_type(eval_type("tuple[int, int | None]", {}), cast(NodeData, node))
    with pytest.raises(CheckFailedError, match="must be a subset"):
        validate_return_type(
            eval_type("tuple[int | None, int]", {}), cast(NodeData, node)
        )
