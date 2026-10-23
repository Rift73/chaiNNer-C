"""A number input with decimal places rounds a connected number as its input conversion
tells the frontend it does (round(Input * 10^p) / 10^p, Navi's round being JavaScript's
Math.round), so the types derived from it are those of the value the node receives.
Rotate's angle (one decimal place) connected to 180.013 turned the image by 180.0 while
the declared size was that of a 180.013-degree turn. tests/common/followOnTypes.test.ts
evaluates the same conversion with Navi on the same values."""

from __future__ import annotations

import math

import pytest

from nodes.properties.inputs import NumberInput, SliderInput

# (value, decimal places, the value the node receives); 0.35 is stored as 0.34999...,
# which Python's round(0.35, 1) took to 0.3 while the frontend's conversion gives 0.4.
ROUNDED = [
    (180.013, 1, 180.0),
    (0.35, 1, 0.4),
    (-0.35, 1, -0.3),
    (2.675, 2, 2.68),
    (0.125, 2, 0.13),
    (12.3456, 3, 12.346),
    (7, 2, 7.0),
]


@pytest.mark.parametrize(("value", "precision", "expected"), ROUNDED)
def test_enforce_rounds_as_the_conversion_does(
    value: float, precision: int, expected: float
):
    number = NumberInput("Number", precision=precision, min=None)
    slider = SliderInput("Slider", precision=precision, min=-1000, max=1000)
    for node_input in (number, slider):
        scale = 10**precision
        assert [c.convert for c in node_input.input_conversions] == [
            f"round(Input * {scale}) / {scale}"
        ]
        enforced = node_input.enforce(value)
        assert type(enforced) is float and enforced == expected


@pytest.mark.parametrize("value", [math.inf, -math.inf, 1e308])
def test_infinite_and_huge_numbers_stay_themselves(value: float):
    assert NumberInput("Ratio", precision=2, min=None).enforce(value) == value


def test_whole_and_unlimited_numbers_keep_their_conversions():
    assert [c.convert for c in NumberInput("N").input_conversions] == ["round(Input)"]
    assert NumberInput("N", precision=0).enforce(2.5) == 3
    unlimited = NumberInput("N", precision="unlimited")
    assert unlimited.input_conversions == []
    assert unlimited.enforce(0.35) == 0.35
