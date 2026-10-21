"""Extended-range planes stay float until an integer sink saturates them (F26)."""

from __future__ import annotations

import base64
import ctypes as ct

import cv2
import numpy as np
import pytest
from test_buffers import ORIGINAL
from test_image_io import NEW, save_arguments

from api import Lazy
from nodes.impl import image_utils, native_buffers
from nodes.impl.native_graph import graph


@pytest.mark.parametrize("bits", [8, 16])
def test_raw_integer_output_saturates_even_without_normalize(bits):
    # Nine elements exercise an AVX2 block and scalar tail, as well as generic u16.
    source = np.array(
        [-0.0063, -0.0, 0.0, 0.2, 1.0, 1.0115, np.nan, np.inf, -np.inf], np.float32
    )
    out = np.empty(source.shape, dtype=f"uint{bits}")
    events = ct.c_int()
    assert (
        native_buffers._api().cn_pixels_convert_checked(
            source.ctypes.data,
            out.ctypes.data,
            source.size,
            0,
            1 if bits == 8 else 2,
            0,
            ct.byref(events),
        )
        == 0
    )
    top = (1 << bits) - 1
    np.testing.assert_array_equal(out, [0, 0, 0, round(0.2 * top), top, top, 0, top, 0])
    assert (
        events.value == 2
    )  # Quiet NaN keeps the invalid-cast event; infinities saturate.


@pytest.mark.parametrize("bits", [8, 16])
def test_in_range_export_matches_previous_bits(bits):
    image = np.random.default_rng(26).random((17, 23, 3), dtype=np.float32)
    expected = getattr(ORIGINAL, f"to_uint{bits}")(image, normalized=True)
    assert (
        getattr(image_utils, f"to_uint{bits}")(image, normalized=True).tobytes()
        == expected.tobytes()
    )


@pytest.mark.parametrize("bits", [8, 16])
def test_save_png_and_preview_saturate_extended_rgb(bits, tmp_path):
    image = np.array([[[-0.0063, 0.2, 1.0115], [1.0, 0.0, 0.5]]], np.float32)
    before = image.tobytes()
    top = (1 << bits) - 1
    expected = np.round(np.clip(image, 0, 1) * top).astype(f"uint{bits}")
    args = save_arguments(
        NEW.save,
        Lazy.ready(image),
        tmp_path,
        png_color_depth=("PngColorDepth", f"U{bits}"),
        dds_gpu=False,
    )
    graph().run_native("chainner:image:save", list(args.values()))
    saved = cv2.imread(str(tmp_path / "image.png"), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(saved, expected)
    if bits == 8:
        previews = graph().image_io_previews(image)
        assert previews is not None
        data = base64.b64decode(previews[0]["url"].split(",", 1)[1])
        preview = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(preview, expected)
    assert image.tobytes() == before
