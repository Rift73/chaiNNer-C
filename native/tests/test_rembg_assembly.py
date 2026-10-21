"""Fixed-mask CPU checks for RemoveBackground's direct output assembly."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import cv2
import numpy as np
import pytest

from nodes.impl.onnx.session import OnnxSession
from nodes.impl.rembg import bg


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("post", [False, True])
def test_rembg_direct_assembly_bits_and_rgb_snapshot(monkeypatch, count, post):
    image = np.arange(18, dtype=np.float32).reshape(2, 3, 3) / 16
    image.view(np.uint32).flat[:2] = [0x7FC12345, 0x80000000]
    masks = [
        np.array([[0, -0.0, 0.25], [0.5, 0.75, 1]], np.float32) for _ in range(count)
    ]
    for index, mask in enumerate(masks):
        mask.view(np.uint32)[0, 0] = 0x7FC12345 + index
    rgb = bg.cvt_color(image, cv2.COLOR_BGR2RGB)

    def post_process(mask):
        processed = mask.copy()
        processed[1, 0] = 0.125
        return processed

    output_masks = [post_process(mask) if post else mask for mask in masks]
    expected = bg.cvt_color(
        bg.stack([bg.naive_cutout(rgb, mask) for mask in output_masks], "vertical"),
        cv2.COLOR_RGBA2BGRA,
    )
    calls = []
    assemble = bg._assemble

    def predict(snapshot):
        np.testing.assert_array_equal(snapshot.view(np.uint32), rgb.view(np.uint32))
        assert not np.shares_memory(snapshot, image)
        # The old pipeline retains the RGB snapshot even if the caller's BGR
        # buffer changes during prediction.
        image.fill(9)
        return masks

    def tracked_assemble(*args, **kwargs):
        calls.append(1)
        return assemble(*args, **kwargs)

    monkeypatch.setattr(bg, "new_session", lambda _: SimpleNamespace(predict=predict))
    monkeypatch.setattr(bg, "post_process", post_process)
    monkeypatch.setattr(bg, "_assemble", tracked_assemble)
    session = cast(OnnxSession, object())
    actual, returned_mask = bg.remove_bg(image, session, post_process_mask=post)
    assert len(calls) == count
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(
        returned_mask.view(np.uint32), output_masks[-1].view(np.uint32)
    )
    if not post:
        assert returned_mask is masks[-1]


def test_rembg_non_float32_mask_keeps_fallback(monkeypatch):
    image = np.zeros((2, 3, 3), np.float32)
    mask = np.ones((2, 3), np.uint8)
    calls = []
    naive_cutout = bg.naive_cutout

    def tracked_naive(rgb, alpha):
        calls.append(alpha)
        return naive_cutout(rgb, alpha)

    def unexpected_assembly(*args, **kwargs):
        raise AssertionError("Non-float32 masks must retain the existing path")

    monkeypatch.setattr(
        bg, "new_session", lambda _: SimpleNamespace(predict=lambda _: [mask])
    )
    monkeypatch.setattr(bg, "naive_cutout", tracked_naive)
    monkeypatch.setattr(bg, "_assemble", unexpected_assembly)
    # The retained concatenate_channels path rejects non-float32 input today.
    session = cast(OnnxSession, object())
    with pytest.raises(TypeError, match="float32"):
        bg.remove_bg(image, session)
    assert len(calls) == 1 and calls[0] is mask
