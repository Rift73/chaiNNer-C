"""Profile-bounded outer tiling preserves coverage, channel order and cancellation."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch

from api import Aborted, Progress
from nodes.impl.pytorch import vosr2_trt_restore as restoration
from nodes.impl.pytorch.vosr2 import VosrColorCorrection
from nodes.impl.pytorch.vosr2_trt_pipeline import FullSession
from nodes.impl.pytorch.vosr2_trt_restore import LatentStatistics


class RecordingProgress(Progress):
    def __init__(self):
        self.values = []

    @property
    def aborted(self):
        return False

    @property
    def paused(self):
        return False

    def set_progress(self, progress):
        assert 0 <= progress <= 1
        assert not self.values or progress >= self.values[-1]
        self.values.append(progress)


def arguments(image, reference, progress=None, check=lambda: None):
    return (
        cast(LatentStatistics, object()),
        image,
        reference,
        42,
        VosrColorCorrection.WAVELET,
        progress or RecordingProgress(),
        check,
        cast(FullSession, object()),
    )


@pytest.mark.parametrize("shape", [(512, 512), (2176, 3840), (16, 16)])
def test_fitting_images_keep_exact_original_call_and_output(monkeypatch, shape):
    # Expanded views avoid allocating large fixtures; the fast path must not
    # even read or re-pad these inputs before dispatching the established call.
    image = torch.zeros(1, 3, 1, 1).expand(1, 3, *shape)
    reference = np.broadcast_to(np.zeros((1, 1, 3), np.float32), (*shape, 3))
    args = arguments(image, reference)
    expected = object()
    calls = []

    def single(*received):
        assert all(a is b for a, b in zip(received, args, strict=True))
        calls.append(True)
        return expected

    monkeypatch.setattr(restoration, "_restore_tile", single)
    assert restoration.restore(*args) is expected
    assert calls == [True]


@pytest.mark.parametrize(
    "shape", [(1080, 1920), (1920, 1080), (513, 31), (1, 769), (1009, 1)]
)
def test_oversized_frames_cover_every_pixel_without_mutation(monkeypatch, shape):
    # Smaller engine bounds exercise the same multi-tile and minimum-padding
    # behavior without allocating a full 8K CPU fixture for every edge case.
    monkeypatch.setattr(
        restoration,
        "profiles",
        lambda _: {"x": ((1, 3, 256, 256),) * 2 + ((1, 3, 512, 768),)},
    )
    reference = np.random.default_rng(7).random((*shape, 3), dtype=np.float32)
    before = reference.copy()
    image = torch.zeros(1, 3, 1, 1).expand(1, 3, *shape)
    progress = RecordingProgress()
    args = arguments(image, reference, progress)
    calls = []

    def single(statistics, tensor, tile, seed, color, sub, check, session):
        assert statistics is args[0] and session is args[-1]
        assert seed == 42 and color == args[4]
        h, w = tile.shape[:2]
        assert 256 <= h <= 512 and 256 <= w <= 768
        assert h % 16 == w % 16 == 0
        np.testing.assert_array_equal(tensor[0].permute(1, 2, 0), tile * 2 - 1)
        calls.append((h, w))
        sub.set_progress(0.2)
        sub.set_progress(1)
        return np.ascontiguousarray(tile[:, :, ::-1])

    monkeypatch.setattr(restoration, "_restore_tile", single)
    actual = restoration.restore(*args)
    assert len(calls) > 1
    assert actual.shape == (*shape, 3)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, reference[:, :, ::-1], rtol=0, atol=2e-7)
    np.testing.assert_array_equal(reference, before)
    assert progress.values[-1] == 1
    args = (*args[:5], RecordingProgress(), *args[6:])
    np.testing.assert_array_equal(restoration.restore(*args), actual)


@pytest.mark.parametrize("error", [Aborted, RuntimeError])
def test_abort_or_tile_failure_stops_without_returning_partial_output(
    monkeypatch, error
):
    # Actual production width boundary, including a singleton axis.
    reference = np.ones((1, 3841, 3), np.float32)
    image = torch.zeros(1, 3, 1, 3841)
    calls = []

    def single(*args):
        calls.append(True)
        if len(calls) == 2:
            raise error("stop")
        return np.ascontiguousarray(args[2][:, :, ::-1])

    monkeypatch.setattr(restoration, "_restore_tile", single)
    with pytest.raises(error, match="stop"):
        restoration.restore(*arguments(image, reference))
    assert len(calls) == 2


def test_cancel_before_first_tile_does_not_run_model(monkeypatch):
    def check():
        raise Aborted()

    def single(*args):
        pytest.fail("Cancelled restoration ran a tile")

    monkeypatch.setattr(restoration, "_restore_tile", single)
    with pytest.raises(Aborted):
        restoration.restore(
            *arguments(torch.zeros(1, 3, 1, 3841), np.zeros((1, 3841, 3)), check=check)
        )
