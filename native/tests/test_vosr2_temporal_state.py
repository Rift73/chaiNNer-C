"""Persistent history must preserve detail, release stale state and stay bounded."""

import cv2
import numpy as np
import pytest

from nodes.impl.pytorch.vosr2_temporal import (
    Correspondence,
    TemporalObservation,
)
from nodes.impl.pytorch.vosr2_temporal_state import (
    DetailState,
    carry_detail,
    initialize_detail,
)


def noop() -> None:
    pass


def texture() -> np.ndarray:
    rng = np.random.default_rng(31)
    return cv2.GaussianBlur(rng.random((48, 64, 3), dtype=np.float32), (0, 0), 0.8)


def identity(source: np.ndarray) -> Correspondence:
    return Correspondence(
        np.zeros((*source.shape[:2], 2), np.float32),
        np.ones(source.shape[:2], np.float32),
    )


@pytest.mark.parametrize("scale", [1, 2, 4, 8])
def test_single_frame_and_identical_detail_are_exact_at_supported_scales(scale):
    source = texture()
    image = cv2.resize(
        source, (64 * scale, 48 * scale), interpolation=cv2.INTER_LANCZOS4
    )
    image[::3, ::3] += 0.01
    frame = TemporalObservation(image, source, identity(source))
    output, state = initialize_detail([frame], 0.65, noop)
    assert output is image
    output, _ = carry_detail(frame, state, 0.65, noop)
    assert output is image


def test_invalid_regions_and_cuts_receive_no_history():
    source = texture()
    image = cv2.resize(source, (256, 192), interpolation=cv2.INTER_LANCZOS4)
    first = image.copy()
    first[::2, ::2] += 0.04
    _, state = initialize_detail([TemporalObservation(first, source, None)], 0.65, noop)
    match = identity(source)
    match.confidence[12:36, 16:48] = 0
    current = TemporalObservation(image, source, match)
    result, _ = carry_detail(current, state, 0.65, noop)
    np.testing.assert_array_equal(result[48:144, 64:192], image[48:144, 64:192])
    match.confidence.fill(0)
    result, reset = carry_detail(current, state, 0.65, noop)
    assert result is image
    _, fresh = carry_detail(current, None, 0.65, noop)
    np.testing.assert_array_equal(reset.fine, fresh.fine)


def test_larger_subpixel_motion_preserves_the_fresh_fine_pattern():
    source = texture()
    base = cv2.resize(source, (256, 192), interpolation=cv2.INTER_LANCZOS4)
    image = base.copy()
    image[::2, ::2] += 0.04
    first = TemporalObservation(image, source, None)
    _, state = initialize_detail([first], 0.65, noop)
    match = identity(source)
    match.flow[:] = [0.375, 0.25]
    current = TemporalObservation(np.roll(image, 1, axis=1), source, match)
    result, _ = carry_detail(current, state, 0.65, noop)
    np.testing.assert_array_equal(result, current.image)


def test_focus_change_does_not_hold_old_sharp_texture():
    source = texture()
    base = cv2.resize(source, (256, 192), interpolation=cv2.INTER_LANCZOS4)
    pattern = np.zeros_like(base)
    pattern[40:150:3, 40:210:3] = 0.15
    state = None
    ratios = []

    def fine(image):
        return image - cv2.GaussianBlur(image, (0, 0), 0.7)

    for i in range(64):
        detail = cv2.GaussianBlur(pattern, (0, 0), 0.05 + i / 20)
        image = base + detail
        frame = TemporalObservation(image, source, identity(source))
        result, state = carry_detail(frame, state, 0.65, noop)
        ratios.append(float(np.std(fine(result)) / np.std(fine(image))))
        assert np.isfinite(result).all()
    assert max(ratios[-10:]) <= 1.05
    assert min(ratios[-10:]) >= 0.98


def test_injected_history_decays_when_fresh_detail_is_absent():
    source = np.full((48, 64, 3), 0.5, np.float32)
    image = np.full((192, 256, 3), 0.5, np.float32)
    first = image.copy()
    first[40:120:2, 60:180:2] += 0.08
    _, state = initialize_detail([TemporalObservation(first, source, None)], 0.65, noop)
    amplitude = np.max(np.abs(state.fine))
    for _ in range(40):
        _, state = carry_detail(
            TemporalObservation(image, source, identity(source)),
            state,
            0.65,
            noop,
        )
    assert np.max(np.abs(state.fine)) < amplitude * 0.002


def test_consistent_contradictory_detail_replaces_old_history():
    source = np.full((48, 64, 3), 0.5, np.float32)
    base = np.full((192, 256, 3), 0.5, np.float32)
    pattern = np.full_like(base, -0.04)
    pattern[::2, ::2] = 0.04
    first = base + pattern
    _, state = initialize_detail([TemporalObservation(first, source, None)], 0.65, noop)
    current = base - pattern
    result = first
    for _ in range(8):
        result, state = carry_detail(
            TemporalObservation(current, source, identity(source)),
            state,
            0.65,
            noop,
        )
    assert np.mean(np.abs(result - current)) < 0.001


def test_state_shape_and_strength_fail_explicitly():
    source = texture()
    frame = TemporalObservation(source, source, identity(source))
    with pytest.raises(ValueError, match="identical dimensions"):
        carry_detail(frame, DetailState(source[:, :-1], source), 0.65, noop)
    with pytest.raises(ValueError, match="strength"):
        carry_detail(frame, None, float("nan"), noop)
    with pytest.raises(ValueError, match="one to five"):
        initialize_detail([], 0.65, noop)
