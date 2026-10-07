"""Exact text application layout, native-font integration and ABI boundaries."""

from __future__ import annotations

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import ImageFont
from test_palette_ops import ORIGINAL_CAPTION
from test_text_raster import ORIGINAL, SOURCE, TEXT

from nodes.impl import caption, native_text

TEXTS = [
    "\nfirst\n\nlast\n",
    "\n\n",
    "  \t  ",
    "e\u0301 A\u030a\nfi ffi\nabc",
    # Right-to-left text: Arabic "hello world", then Hebrew "shalom".
    "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645\n\u05e9\u05dc\u05d5\u05dd",
    "漢字\n🙂🚀\nΔ",
    "a\rb\nc\r\nd\u2028e",
    "\ud800\n\udfff",
    "a\x00b\n\x00",
    "W" * 401 + "\nx",
]


def compare_node(text, width, height, alignment, anchor, bold=False, italic=False):
    def call(module):
        return module.text_as_image_node(
            text,
            bold,
            italic,
            getattr(module.TextAlignment, alignment),
            module.Color.bgr((0.17, 0.53, 0.89)),
            width,
            height,
            getattr(module.Anchor, anchor),
        )

    try:
        expected = call(ORIGINAL)
    except (ValueError, ZeroDivisionError, UnicodeError, OSError) as error:
        with pytest.raises(type(error)) as actual_error:
            call(TEXT)
        assert str(actual_error.value) == str(error)
        return None
    actual = call(TEXT)
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    return actual


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("dimensions", [(1, 1), (1, 31), (53, 1), (137, 83)])
@pytest.mark.parametrize(
    "alignment,anchor,bold,italic",
    [
        ("LEFT", "TOP_LEFT", False, False),
        ("CENTER", "CENTER", True, False),
        ("RIGHT", "BOTTOM_RIGHT", False, True),
        ("RIGHT", "TOP_RIGHT", True, True),
    ],
)
def test_text_unicode_blank_lines_and_tiny_images(
    text, dimensions, alignment, anchor, bold, italic
):
    width, height = dimensions
    compare_node(text, width, height, alignment, anchor, bold, italic)


@pytest.mark.parametrize("text", ["", "\n", "a\nb", "ab\ncd", *TEXTS])
def test_codepoint_scan_preserves_python_string_contract(text):
    expected = text.split("\n")
    assert native_text.scan(text) == (expected, max(expected, key=len))


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("size,width", [(1, 1), (13, 7), (41, 137)])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_caption_text_edge_cases(monkeypatch, text, size, width, channels):
    for module in (caption, ORIGINAL_CAPTION):
        monkeypatch.setattr(
            module,
            "get_font",
            lambda value: ImageFont.truetype(
                str(SOURCE / "fonts/Roboto-Light.ttf"), value
            ),
        )
    image = np.linspace(0, 1, width * 3 * channels, dtype=np.float32).reshape(
        (3, width) if channels == 1 else (3, width, channels)
    )
    image.setflags(write=False)
    before = image.tobytes()
    for position in ("TOP", "BOTTOM"):
        try:
            expected = ORIGINAL_CAPTION.add_caption(
                image, text, size, getattr(ORIGINAL_CAPTION.CaptionPosition, position)
            )
        except (ValueError, ZeroDivisionError, UnicodeError, OSError) as error:
            with pytest.raises(type(error)) as actual_error:
                caption.add_caption(
                    image, text, size, getattr(caption.CaptionPosition, position)
                )
            assert str(actual_error.value) == str(error)
        else:
            actual = caption.add_caption(
                image, text, size, getattr(caption.CaptionPosition, position)
            )
            np.testing.assert_array_equal(
                actual.view(np.uint32), expected.view(np.uint32)
            )
    assert image.tobytes() == before


def test_concurrent_text_layout_has_no_shared_mutable_font_state():
    cases = [
        ("first\n\nlast\n", 101, 73, "LEFT", "TOP"),
        # Ends in Arabic "hello".
        ("e\u0301\n🙂\n\u0645\u0631\u062d\u0628\u0627", 151, 93, "RIGHT", "BOTTOM"),
        ("a\nb\nc", 31, 137, "CENTER", "RIGHT"),
    ]
    expected = [compare_node(*case) for case in cases]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda i: compare_node(*cases[i % 3]), range(24)))
    for index, result in enumerate(results):
        np.testing.assert_array_equal(result, expected[index % 3])


def test_raw_text_abi_rejects_invalid_ranges_without_writing():
    api = native_text._api()
    source = np.array([65, 10, 66], np.uint32)
    output = np.full(16, 123, np.uintp)
    before = output.tobytes()
    for address, count, status in [
        (None, 3, 1),
        (source.ctypes.data + 1, 3, 1),
        (ct.c_size_t(-4).value, 3, 2),
        (source.ctypes.data, ct.c_size_t(-1).value, 2),
        (output.ctypes.data, 3, 1),
    ]:
        assert api.cn_text_scan(address, count, output.ctypes.data) == status
        assert output.tobytes() == before
    assert api.cn_text_scan(source.ctypes.data, 3, output.ctypes.data + 1) == 1
    assert api.cn_text_ranges(source.ctypes.data, 3, output.ctypes.data, 3) == 1
    assert api.cn_text_ranges(source.ctypes.data, 3, output.ctypes.data + 1, 2) == 1
    assert api.cn_text_ranges(output.ctypes.data, 3, output.ctypes.data, 1) == 1
    assert output.tobytes() == before
    widths = np.array([3, 1], np.float64)
    for address, count, alignment in [
        (None, 2, 0),
        (widths.ctypes.data + 1, 2, 0),
        (widths.ctypes.data, 0, 0),
        (widths.ctypes.data, 2, 3),
        (output.ctypes.data, 2, 0),
        (ct.c_size_t(-8).value, 2, 0),
    ]:
        assert (
            api.cn_text_positions(
                address, count, 0, 0, 5, alignment, output.ctypes.data
            )
            != 0
        )
        assert output.tobytes() == before
    widths[1] = np.nan
    assert (
        api.cn_text_positions(widths.ctypes.data, 2, 0, 0, 5, 0, output.ctypes.data)
        == 1
    )
    assert output.tobytes() == before
