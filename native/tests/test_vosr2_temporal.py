"""Temporal reduction must not come from losing motion, history leaks or blur."""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from nodes.impl.native_threshold import channel_mean
from nodes.impl.pytorch.vosr2_temporal import (
    Correspondence,
    RestoredFrame,
    motion_pair,
    remap,
    stabilize,
    validate_image,
)


def noop() -> None:
    pass


def texture() -> np.ndarray:
    rng = np.random.default_rng(8)
    return cv2.GaussianBlur(rng.random((96, 128, 3), dtype=np.float32), (0, 0), 0.8)


def identity(shape: tuple[int, int]) -> Correspondence:
    return Correspondence(np.zeros((*shape, 2), np.float32), np.ones(shape, np.float32))


def test_zero_and_single_frame_preserve_exact_existing_output():
    source = texture()
    restored = cv2.resize(source, (512, 384), interpolation=cv2.INTER_LANCZOS4)
    frame = RestoredFrame.create(source, restored)
    assert stabilize(frame, [], 1, noop) is restored
    assert stabilize(frame, [(frame, identity(source.shape[:2]))], 0, noop) is restored


def test_static_correlated_detail_is_preserved_and_flicker_is_reduced():
    source = texture()
    base = cv2.resize(source, (512, 384), interpolation=cv2.INTER_LANCZOS4)
    detail = np.zeros_like(base)
    detail[::2, ::2] = 0.05
    current = RestoredFrame.create(source, base + detail + 0.015)
    neighbor = RestoredFrame.create(source, base + detail - 0.015)
    match = identity(source.shape[:2])
    result = stabilize(current, [(neighbor, match), (neighbor, match)], 1, noop)
    target = base + detail
    assert (
        np.mean(np.abs(result - target))
        < np.mean(np.abs(current.restored - target)) / 3
    )
    unchanged = stabilize(current, [(current, match), (current, match)], 1, noop)
    np.testing.assert_array_equal(unchanged, current.restored)


def test_flow_direction_and_scale_preserve_a_translating_fine_pattern():
    source = texture()
    base = cv2.resize(source, (512, 384), interpolation=cv2.INTER_LANCZOS4)
    detail = np.zeros_like(base)
    detail[::3, ::3] = 0.08
    current = RestoredFrame.create(source, base + detail)
    shifted = np.roll(source, 3, axis=1)
    other_base = cv2.resize(shifted, (512, 384), interpolation=cv2.INTER_LANCZOS4)
    neighbor = RestoredFrame.create(shifted, other_base + np.roll(detail, 12, axis=1))
    flow = np.zeros((*source.shape[:2], 2), np.float32)
    flow[:, :, 0] = 3
    confidence = np.ones(source.shape[:2], np.float32)
    confidence[:, -5:] = 0
    result = stabilize(current, [(neighbor, Correspondence(flow, confidence))], 1, noop)
    np.testing.assert_allclose(
        result[:, 24:-24], current.restored[:, 24:-24], atol=1e-6
    )


def test_real_dis_flow_tracks_translation_and_rejects_uncovered_border():
    source = texture()
    shifted = np.roll(source, 4, axis=1)
    forward, reverse = motion_pair(source, shifted, noop)
    assert np.median(forward.flow[16:-16, 16:-16, 0]) == pytest.approx(4, abs=0.15)
    assert np.median(reverse.flow[16:-16, 16:-16, 0]) == pytest.approx(-4, abs=0.15)
    assert forward.confidence[16:-16, 16:-16].mean() > 0.7
    assert np.count_nonzero(forward.confidence[:, -4:]) == 0
    aligned = remap(shifted, forward.flow, cv2.INTER_LINEAR)
    assert np.mean(np.abs(aligned[16:-16, 16:-16] - source[16:-16, 16:-16])) < 0.01


def test_cut_is_exact_passthrough_and_occlusion_is_rejected():
    first = np.zeros((96, 128, 3), np.float32)
    second = np.ones_like(first)
    match, reverse = motion_pair(first, second, noop)
    assert not match.confidence.any() and not reverse.confidence.any()
    current = RestoredFrame.create(first, first.copy())
    neighbor = RestoredFrame.create(second, second.copy())
    np.testing.assert_array_equal(
        stabilize(current, [(neighbor, match)], 1, noop), first
    )
    source = texture()
    occluded = source.copy()
    occluded[20:65, 35:85] = 1
    match, _ = motion_pair(source, occluded, noop)
    assert match.confidence[28:57, 43:77].mean() < 0.01


@pytest.mark.parametrize("strength", [-1, 1.1, float("nan"), float("inf")])
def test_invalid_strength_is_rejected(strength):
    source = texture()
    with pytest.raises(ValueError, match="strength"):
        stabilize(RestoredFrame.create(source, source), [], strength, noop)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((20, 20), np.float32),
        np.zeros((20, 20, 4), np.float32),
        np.zeros((20, 20, 3), np.uint8),
        np.zeros((8, 20, 3), np.float32),
        np.full((20, 20, 3), np.nan, np.float32),
    ],
)
def test_invalid_frame_fails_explicitly(image):
    with pytest.raises(ValueError, match="Temporal VOSR"):
        validate_image(image)


def test_mismatched_dimensions_and_too_many_neighbors_fail():
    source = texture()
    with pytest.raises(ValueError, match="identical dimensions"):
        motion_pair(source, source[:, :-1], noop)
    frame = RestoredFrame.create(source, source)
    with pytest.raises(ValueError, match="at most two"):
        stabilize(frame, [(frame, identity(source.shape[:2]))] * 3, 1, noop)


def test_subpixel_alignment_preserves_fine_detail():
    source = texture()
    base = cv2.resize(source, (512, 384), interpolation=cv2.INTER_LANCZOS4)
    current = RestoredFrame.create(source, base)
    neighbors = []
    for offset in (-0.375, 0.375):
        matrix = np.array([[1, 0, offset], [0, 1, 0]], dtype=np.float32)
        shifted = cv2.warpAffine(
            source,
            matrix,
            (128, 96),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        matrix[0, 2] *= 4
        high = cv2.warpAffine(
            base,
            matrix,
            (512, 384),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        other = RestoredFrame.create(shifted, high)
        neighbors.append((other, motion_pair(source, shifted, noop)[0]))
    result = stabilize(current, neighbors, 0.65, noop)
    high_result = result - cv2.GaussianBlur(result, (0, 0), 0.7)
    high_base = base - cv2.GaussianBlur(base, (0, 0), 0.7)
    ratio = np.std(high_result[24:-24, 24:-24]) / np.std(high_base[24:-24, 24:-24])
    assert ratio >= 0.98


def test_contrast_window_does_not_modify_unmatched_pixels():
    source = texture()
    base = cv2.resize(source, (512, 384), interpolation=cv2.INTER_LANCZOS4)
    rng = np.random.default_rng(12)
    first = RestoredFrame.create(
        source, base + rng.normal(0, 0.01, base.shape).astype(np.float32)
    )
    second = RestoredFrame.create(
        source, base + rng.normal(0, 0.01, base.shape).astype(np.float32)
    )
    match = identity(source.shape[:2])
    match.confidence[24:72, 32:96] = 0
    result = stabilize(first, [(second, match)], 0.65, noop)
    np.testing.assert_array_equal(
        result[100:284, 132:380], first.restored[100:284, 132:380]
    )
    assert np.isfinite(result).all()
    assert np.max(np.abs(result - first.restored)) > 0.001


def test_flat_scene_has_no_contrast_gain_singularity_or_added_texture():
    source = np.full((32, 48, 3), 0.5, np.float32)
    base = cv2.resize(source, (192, 128), interpolation=cv2.INTER_LANCZOS4)
    current = RestoredFrame.create(source, base + 0.01)
    neighbor = RestoredFrame.create(source, base - 0.01)
    result = stabilize(current, [(neighbor, identity(source.shape[:2]))], 0.65, noop)
    assert np.isfinite(result).all()
    assert np.std(result) < 1e-6
    assert np.mean(np.abs(result - base)) < 0.01


def test_stationary_fine_consensus_retains_contrast_and_improves_agreement():
    source = texture()
    base = cv2.resize(source, (512, 384), interpolation=cv2.INTER_LANCZOS4)
    rng = np.random.default_rng(20)
    common = rng.normal(0, 0.025, base.shape).astype(np.float32)
    frames = [
        RestoredFrame.create(
            source, base + common + rng.normal(0, 0.012, base.shape).astype(np.float32)
        )
        for _ in range(3)
    ]
    match = identity(source.shape[:2])
    outputs = [
        stabilize(
            frame,
            [(other, match) for other in frames if other is not frame],
            0.65,
            noop,
        )
        for frame in frames
    ]
    sl = np.s_[24:-24, 24:-24]

    def fine(img):
        return (img - cv2.GaussianBlur(img, (0, 0), 0.7))[sl]

    before = np.mean(np.abs(frames[0].restored[sl] - frames[1].restored[sl]))
    after = np.mean(np.abs(outputs[0][sl] - outputs[1][sl]))
    assert after < before * 0.5
    # Contrast is screened separately: an all-zero/blurred result cannot pass.
    assert np.std(fine(outputs[0])) / np.std(fine(frames[0].restored)) >= 0.94
    truth = fine(base + common).ravel()
    assert (
        np.corrcoef(fine(outputs[0]).ravel(), truth)[0, 1]
        > np.corrcoef(fine(frames[0].restored).ravel(), truth)[0, 1]
    )


def test_tiny_strength_remains_close_to_independent_output():
    source = texture()
    current = RestoredFrame.create(source, source + 0.01)
    neighbor = RestoredFrame.create(source, source - 0.01)
    output = stabilize(current, [(neighbor, identity(source.shape[:2]))], 1e-6, noop)
    np.testing.assert_allclose(output, current.restored, atol=1e-6)


# The one-thread stabilize as it was before its work was spread over threads (vosr/integ
# c8a4f267), copied verbatim as the oracle of the threaded one.
def original_stabilize(
    current: RestoredFrame,
    neighbors: Sequence[tuple[RestoredFrame, Correspondence]],
    strength: float,
    check: Callable[[], None],
) -> np.ndarray:
    """Keep the current observation; blend only aligned restoration residuals.

    Coarse residuals retain conservative weights. Fine residuals get stronger
    consensus, with bounded local contrast recovery to avoid averaging away
    detail. Invalid matches contribute nothing. Inputs are independently restored
    frames, never earlier stabilized outputs; support stays at three frames.
    """
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Temporal strength must be between 0 and 1.")
    if len(neighbors) > 2:
        raise ValueError("Temporal stabilization accepts at most two neighbors.")
    check()
    if strength == 0 or not neighbors:
        return current.restored
    height, width = current.restored.shape[:2]
    low = cv2.GaussianBlur(current.residual, (0, 0), 1)
    high = current.residual - low
    delta = np.zeros_like(current.restored)
    denominator = np.ones((height, width, 1), dtype=np.float32)
    low_delta = np.zeros_like(delta)
    high_delta = np.zeros_like(delta)
    high_denominator = np.ones_like(denominator)
    stable_support = np.ones((height, width), dtype=np.float32)
    for neighbor, match in neighbors:
        check()
        if (
            neighbor.source.shape != current.source.shape
            or neighbor.restored.shape != current.restored.shape
        ):
            raise ValueError(
                "All temporal sequence frames must have identical dimensions."
            )
        if not np.any(match.confidence):
            continue
        flow = cv2.resize(match.flow, (width, height), interpolation=cv2.INTER_LINEAR)
        flow *= np.array(
            [width / match.flow.shape[1], height / match.flow.shape[0]],
            dtype=np.float32,
        )
        aligned = remap(neighbor.residual, flow, cv2.INTER_LANCZOS4)
        aligned_low = remap(
            cv2.GaussianBlur(neighbor.residual, (0, 0), 1),
            flow,
            cv2.INTER_LANCZOS4,
        )
        aligned_high = aligned - aligned_low
        disagreement = cv2.GaussianBlur(
            channel_mean(np.abs(aligned - current.residual)), (0, 0), 1
        )
        confidence = cv2.resize(
            match.confidence, (width, height), interpolation=cv2.INTER_LINEAR
        )
        # Fine texture is sensitive to fractional-pixel warping. Use the stronger
        # consensus only on nearly stationary, reliable correspondences; moving
        # regions retain the original conservative fusion. Erosion also protects
        # the neighborhood of an occlusion from the contrast window.
        stable = match.confidence * np.exp(
            -((np.linalg.norm(match.flow, axis=2) / 0.20) ** 4)
        )
        stable = cv2.erode(stable, np.ones((5, 5), np.uint8))
        stable_support = np.minimum(
            stable_support,
            cv2.resize(stable, (width, height), interpolation=cv2.INTER_LINEAR),
        )
        weight = (
            confidence * np.exp(-((disagreement / 0.10) ** 2)) * (strength * 0.5)
        )[:, :, None]
        delta += (aligned - current.residual) * weight
        low_delta += (aligned_low - low) * weight
        denominator += weight
        coarse_disagreement = cv2.GaussianBlur(
            channel_mean(np.abs(aligned_low - low)), (0, 0), 1
        )
        fine_weight = (
            confidence * np.exp(-((coarse_disagreement / 0.03) ** 2)) * strength * 3
        )[:, :, None]
        high_delta += (aligned_high - high) * fine_weight
        high_denominator += fine_weight
    check()
    if not np.any(delta) and not np.any(high_delta) and not np.any(low_delta):
        return current.restored
    fine = high + high_delta / high_denominator
    coarse = current.restored - high + low_delta / denominator
    legacy = current.restored + delta / denominator
    support = stable_support[:, :, None]
    coarse = legacy + (coarse - legacy) * support

    def high_pass(image: np.ndarray) -> np.ndarray:
        return image - cv2.GaussianBlur(image, (0, 0), 0.7)

    def local_mean(image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(channel_mean(image), (0, 0), 4)

    # Solve E[(coarse + gain*fine)^2] = target in the detail band. Include
    # the cross term: normalizing residual energy alone can soften input detail.
    # Spatially varying gains make this a local estimate, not an exact bound.
    fine_band = high_pass(fine * support)
    coarse_band = high_pass(coarse)
    a = local_mean(fine_band * fine_band)
    b = local_mean(fine_band * coarse_band)
    c = local_mean(coarse_band * coarse_band)
    target = np.minimum(
        local_mean(high_pass(legacy) ** 2),
        local_mean(high_pass(current.restored) ** 2),
    )
    gain = np.clip(
        (-b + np.sqrt(np.maximum(b * b + a * (target - c), 0))) / np.maximum(a, 1e-12),
        0,
        1.6,
    )
    gain[a < 1e-10] = 1
    correction = low_delta / denominator + fine * gain[:, :, None] - high
    correction[high_denominator[:, :, 0] == 1] = 0
    conservative = delta / denominator
    correction = conservative + (correction - conservative) * support
    check()
    return np.ascontiguousarray(np.clip(current.restored + correction, 0, 1))


def moving_frames(
    cut: bool = False,
) -> tuple[list[RestoredFrame], list[Correspondence]]:
    """Three frames of a texture moving one pixel per frame, restored at 4x with each
    frame's own noise (the last one another scene with `cut`), and their matches:
    frame 1 to frame 0, then frame 1 to frame 2."""
    rng = np.random.default_rng(12)
    sources = [np.ascontiguousarray(np.roll(texture(), i, axis=1)) for i in range(3)]
    if cut:
        sources[2] = np.ascontiguousarray(1 - sources[2])
    frames = [
        RestoredFrame.create(
            source,
            np.clip(
                cv2.resize(source, (512, 384), interpolation=cv2.INTER_LANCZOS4)
                + rng.normal(0, 0.02, (384, 512, 3)).astype(np.float32),
                0,
                1,
            ),
        )
        for source in sources
    ]
    previous = motion_pair(sources[0], sources[1], noop)[1]
    following = motion_pair(sources[1], sources[2], noop)[0]
    return frames, [previous, following]


@pytest.mark.parametrize("strength", [1, 0.65, 0.2])
@pytest.mark.parametrize("cut", [False, True])
def test_threaded_stabilize_equals_the_one_thread_original_bit_for_bit(strength, cut):
    frames, (previous, following) = moving_frames(cut)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for neighbors in (
            [(frames[0], previous), (frames[2], following)],
            [(frames[0], previous)],
            [(frames[2], following)],
        ):
            expected = original_stabilize(frames[1], neighbors, strength, noop)
            for actual in (
                stabilize(frames[1], neighbors, strength, noop),
                stabilize(frames[1], neighbors, strength, noop, pool),
            ):
                assert actual.dtype == expected.dtype == np.float32
                np.testing.assert_array_equal(actual, expected)
