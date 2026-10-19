"""Native utility value entry points (native/src/run/nodes/utility_value.cpp) against their Python bodies."""

import json
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pytest

from native_differential import assert_native_matches
from nodes.impl.color.color import Color
from nodes.impl.native_graph import graph
from nodes.utils.seed import Seed

# Numbers as the UI and upstream nodes send them: int64 bounds, big ints (they cross as Python objects), both
# zeros, infinities, the smallest denormal, a float that is an integer.
NUMBERS = [0, 1, -1, 7, 2**63 - 1, -(2**63), 2**64, -(10**30), 0.0, -0.0, 0.1, -2.5, 3.0, 1e308, 5e-324,
           float("inf"), float("-inf")]
# Text: empty, ASCII, every UTF-8 length, U+0000, a lone surrogate (crosses as a Python object), numbers (the
# input stringifies them).
TEXTS = ["", "abc", "h\u00e9llo \u4e16\u754c \U0001f600", "a\x00b", "\ud800x", "  spaced  ", 3.0, 2.5, 12]
DIRECTORIES = ["C:/a/b", "C:\\a\\.\\b\\", "relative\\x", "\\\\server\\share\\x", "C:", ".", "\u00e9\u4e16/\U0001f600",
               "C:/\udc80bad", Path("D:/x/y")]
IMAGE = np.linspace(-0.5, 1.5, 3 * 5 * 3, dtype=np.float32).reshape(3, 5, 3)
ANY = [None, 0, 2**70, -0.0, "t\u00e9xt", "\udfff", Path("C:/p"), Color((0.25, 0.5, 0.75)), IMAGE, Seed(5), True,
       [1, 2], (3,)]


@pytest.mark.parametrize("number", NUMBERS)
def test_number_matches(number):
    assert_native_matches("chainner:utility:number", [number])


@pytest.mark.parametrize("percent", [0, 50, 100, 49.5, 50.5, -3, 250, 0.0, -0.0, 99.999])
def test_percent_matches(percent):
    assert_native_matches("chainner:utility:percent", [percent])


@pytest.mark.parametrize("text", TEXTS)
def test_text_matches(text):
    assert_native_matches("chainner:utility:text", [text])


@pytest.mark.parametrize("directory", DIRECTORIES)
def test_directory_matches(directory):
    assert_native_matches("chainner:utility:directory", [directory])


@pytest.mark.parametrize(
    "color",
    [
        Color((0.5,)),
        Color((0.0, 0.25, 1.0)),
        Color((0.1, 0.2, 0.3, 0.4)),
        json.dumps({"kind": "rgba", "values": [255, 0, -3, 0.5]}),
        json.dumps({"kind": "grayscale", "values": [0]}),
    ],
)
def test_color_matches(color):
    assert_native_matches("chainner:utility:color", [color])


@pytest.mark.parametrize("number", [1, 2, 41, 2**63 - 1, 2**64])
def test_execution_number_matches(number):
    assert_native_matches("chainner:utility:execution_number", [number])


@pytest.mark.parametrize("first", ANY[1:])
def test_pass_through_matches(first):
    assert_native_matches("chainner:utility:pass_through", [first, *ANY[:9]])


def test_pass_through_returns_its_inputs_themselves():
    image = IMAGE.copy()
    image.setflags(write=False)
    seed = Seed(9)
    outputs = graph().run_native("chainner:utility:pass_through", [image, seed, None, None, None, None, None, None,
                                                                   None, None])
    # A frozen Seed within int64 crosses as a value (an equal Seed); the image is the same array.
    assert outputs[0] is image and outputs[1] == seed and outputs[2:] == [None] * 8


@pytest.mark.parametrize("index", range(10))
@pytest.mark.parametrize("value", [IMAGE, "x", 2**65, Seed(3), -0.0])
def test_switch_matches(index, value):
    values = [None] * 10
    values[index] = value
    assert_native_matches("chainner:utility:switch", [index, *values])


# Subclasses of the engine's scalar and path types: they cross as themselves (the body returns them unchanged).
SUBCLASSED = [np.float64(-0.0), np.float64(0.1), np.str_("téxt"), PurePosixPath("a/b"), PureWindowsPath("C:/a"),
              np.int64(3)]


def test_pass_through_keeps_subclassed_values():
    inputs = [*SUBCLASSED, None, None, None, None]
    assert_native_matches("chainner:utility:pass_through", inputs)
    outputs = graph().run_native("chainner:utility:pass_through", inputs)
    assert all(output is value for output, value in zip(outputs[: len(SUBCLASSED)], SUBCLASSED, strict=True))


@pytest.mark.parametrize("value", SUBCLASSED)
def test_switch_keeps_subclassed_values(value):
    values = [None] * 10
    values[3] = value
    assert_native_matches("chainner:utility:switch", [3, *values])
    assert graph().run_native("chainner:utility:switch", [3, *values])[0] is value


@pytest.mark.parametrize("index", [0, 4, 9])
def test_switch_unconnected_error(index):
    values: list[object] = list(range(10))
    values[index] = None
    outcome = assert_native_matches("chainner:utility:switch", [index, *values])
    assert outcome == ("error", ("RuntimeError", "Invalid value selected. The selected value should have a connection."))


PRESETS = ["HD_1080", "NTSC MPEG2", "NTSC_DV", "NTSC", "PAL", "PAL MPEG2", "HD_720", "UHD_4K", "UHD2_8K",
           "DCI_2KFLAT", "DCI_2KSCOPE", "DCI_2K1080", "DCI_4KFLAT", "DCI_4KSCOPE", "DCI_4K2160", "SQ256", "SQ512",
           "SQ1024", "SQ2048", "SQ4096", "SQ8192", "CUSTOM"]


def test_resolution_presets_cover_the_enum():
    from packages.chaiNNer_standard.utility.value.resolutions import ResList

    assert sorted(PRESETS) == sorted(member.value for member in ResList)


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("size", [(1920, 1080), (1, 1), (2**64, 3), (7.5, 0.4), (-4, 0)])
def test_resolutions_match(preset, size):
    assert_native_matches("chainner:utility:resolutions", [preset, *size])


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


@pytest.mark.parametrize(
    "start, start_inclusive, stop, stop_inclusive",
    [(2**63, True, 2**63 + 2, False), (0, True, 2**64, False), (2**63 - 2, False, 2**63 - 1, True)],
)
def test_range_beyond_int64_declines(start, start_inclusive, stop, stop_inclusive):
    # Python's integers are unbounded: a bound or an item beyond int64 runs the body.
    assert_native_matches("chainner:utility:range", [start, start_inclusive, stop, stop_inclusive], declines=True)


def test_range_error_is_an_assertion_error():
    outcome = assert_native_matches("chainner:utility:range", [5, True, 1, False])
    assert outcome == ("error", ("AssertionError", ""))


def test_native_generator_items_are_read_once():
    (generator,) = graph().run_native("chainner:utility:range", [0, True, 3, False])
    assert generator.expected_length == 3 and generator.fail_fast and generator.source_paths == ()
    assert list(generator.supplier()) == [0, 1, 2]
    with pytest.raises(ValueError, match="read once"):
        generator.supplier()
