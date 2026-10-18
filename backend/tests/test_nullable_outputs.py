"""Mode-dependent outputs (BaseOutput.null_unless): typed `null` under the dropdown's
other options, and a None value passes the output's enforce there."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from api import NodeData
from nodes.properties.outputs import ImageOutput, NumberOutput
from process import enforce_output


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
