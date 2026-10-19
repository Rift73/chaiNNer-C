"""Native utility value entry points (native/src/run/nodes/utility_value.cpp) against their Python bodies."""

import pytest

from native_differential import assert_native_matches
from nodes.impl.native_graph import graph


@pytest.mark.parametrize(
    "start, start_inclusive, stop, stop_inclusive",
    [
        (0, True, 10, False),
        (0, False, 10, True),
        (-5, True, 5, True),
        (3, True, 3, False),
        (3, False, 3, True),
        (7, True, 3, False),
        (2**62, True, 2**62 + 4, True),
        (-(2**63), True, -(2**63) + 3, False),
    ],
)
def test_range_matches(start, start_inclusive, stop, stop_inclusive):
    assert_native_matches("chainner:utility:range", [start, start_inclusive, stop, stop_inclusive])


def test_range_error_is_an_assertion_error():
    outcome = assert_native_matches("chainner:utility:range", [5, True, 1, False])
    assert outcome == ("error", ("AssertionError", ""))


def test_native_generator_items_are_read_once():
    (generator,) = graph().run_native("chainner:utility:range", [0, True, 3, False])
    assert generator.expected_length == 3 and generator.fail_fast and generator.source_paths == ()
    assert list(generator.supplier()) == [0, 1, 2]
    with pytest.raises(ValueError, match="read once"):
        generator.supplier()
