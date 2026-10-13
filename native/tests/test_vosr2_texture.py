"""Source-guided grain must not follow the neural window or strip grid."""

import cv2
import numpy as np
import pytest
import torch

from nodes.impl.pytorch import vosr2_texture as texture


def grain(height=192, width=384):
    values = np.random.default_rng(67).normal(0.4, 0.018, (height, width))
    return np.repeat(values[:, :, None], 3, axis=2).astype(np.float32)


def apply(source, restored=None, scale=2, check=lambda: None):
    reference = cv2.resize(
        source, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )
    if restored is None:
        restored = np.full_like(reference, 0.4)
    return texture.preserve_texture(
        source,
        reference[:, :, ::-1].copy(),
        restored,
        scale,
        check,
        torch.device("cpu"),
    )


def test_erased_and_retained_windows_converge_to_uniform_source_grain():
    source = grain()
    reference = cv2.resize(source, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    restored = reference.copy()
    restored[:, 384:] = 0.4
    result = apply(source, restored)
    left = result[80:300, 80:300, 0]
    right = result[80:300, 464:684, 0]
    assert 0.8 < right.std() / left.std() < 1.2
    assert right.std() > 0.008  # Do not pass by erasing both sides.
    assert np.std(restored[80:300, 464:684]) < 1e-6


def test_guide_follows_source_coordinates_not_window_grid():
    source = grain(width=800)
    source[40:120, 400:404] += 0.4
    full = texture.texture_guide(source)
    shifted = texture.texture_guide(source[:, 32:])
    assert full is not None and shifted is not None
    np.testing.assert_allclose(full[0][:, 128:-96], shifted[0][:, 96:-96], atol=2e-6)


@pytest.mark.parametrize("kind", ["flat", "gradient", "defocus", "coherent", "noise"])
def test_non_grain_inputs_are_not_treated_as_flat_texture(kind):
    x = np.arange(384, dtype=np.float32)
    if kind == "flat":
        line = np.full_like(x, 0.4)
    elif kind == "gradient":
        line = x / 384
    elif kind == "defocus":
        line = np.sin(x / 24) * 0.2 + 0.4
    elif kind == "coherent":
        line = np.sin(x * 2) * 0.025 + 0.4
    else:
        assert texture.texture_guide(grain() * 12) is None
        return
    source = np.broadcast_to(line[None, :, None], (192, 384, 3)).copy()
    assert texture.texture_guide(source) is None
    value = np.full((384, 768, 3), 0.4, np.float32)
    assert apply(source, value) is value


@pytest.mark.parametrize("chromatic", [False, True])
def test_strong_edges_survive_without_a_texture_free_ring(chromatic):
    source = grain()
    source[:, 192:, 0 if chromatic else slice(None)] += 0.3
    if chromatic:
        source[:, 192:, 1] -= 0.3 * 0.114 / 0.587
    guide = texture.texture_guide(source)
    assert guide is not None
    assert guide[0][:, 172:184].mean() > 0.95
    assert guide[0][:, 200:212].mean() > 0.95
    reference = cv2.resize(source, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    result = apply(source, reference.copy())
    # Matching detail has zero correction, including the edge and its vicinity.
    np.testing.assert_array_equal(result, reference)
    restored = cv2.GaussianBlur(reference, (0, 0), 0.6)
    result = apply(source, restored)
    original_jump = restored[:, 400:430].mean(1) - restored[:, 340:370].mean(1)
    actual_jump = result[:, 400:430].mean(1) - result[:, 340:370].mean(1)
    np.testing.assert_allclose(actual_jump, original_jump, atol=0.005)


def test_bounded_strips_match_whole_image_filter_without_seams(monkeypatch):
    source = grain(height=270)
    reference = cv2.resize(source, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    restored = cv2.GaussianBlur(reference, (0, 0), 3)
    bounded = apply(source, restored)
    sizes = []
    original = texture._detail

    def observe(value, epsilon, radius):
        sizes.append(value.shape[0])
        return original(value, epsilon, radius)

    monkeypatch.setattr(texture, "_detail", observe)
    np.testing.assert_array_equal(apply(source, restored), bounded)
    assert max(sizes) <= 140
    guide = texture.texture_guide(source)
    assert guide is not None
    mask, rms = (cv2.resize(x, reference.shape[1::-1]) for x in guide)
    epsilon = (3 * rms) ** 2 + 0.004**2
    whole = np.clip(
        restored
        + mask[:, :, None]
        * (original(reference, epsilon, 3) - original(restored, epsilon, 3)),
        0,
        1,
    )
    np.testing.assert_allclose(bounded, whole, atol=2e-6, rtol=0)


def test_zero_confidence_pixels_are_exactly_untouched():
    source = grain(width=640)
    source[:, 320:] = 0.25
    reference = cv2.resize(source, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    restored = (
        np.random.default_rng(1).uniform(0, 1, reference.shape).astype(np.float32)
    )
    result = apply(source, restored)
    guide = texture.texture_guide(source)
    assert guide is not None
    mask = cv2.resize(guide[0], reference.shape[1::-1])
    assert np.any(mask == 0)
    np.testing.assert_array_equal(result[mask == 0], restored[mask == 0])


def test_one_window_has_no_guide_or_filter_work(monkeypatch):
    def unexpected(*args):
        pytest.fail("Single window entered texture correction")

    monkeypatch.setattr(texture, "texture_guide", unexpected)
    source = grain(height=128, width=128)
    restored = np.full((256, 256, 3), 0.4, np.float32)
    assert apply(source, restored) is restored


def test_repeatability_input_ownership_and_cancellation():
    source = grain()
    original = source.copy()
    first = apply(source)
    np.testing.assert_array_equal(first, apply(source))
    np.testing.assert_array_equal(source, original)
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError, match="cancelled"):
        apply(source, check=cancel)
    assert calls == 3
    np.testing.assert_array_equal(source, original)


def test_analysis_overlaps_inference_and_workers_are_joined(monkeypatch):
    import threading

    source = grain()
    reference = cv2.resize(source, None, fx=2, fy=2)
    started, release = threading.Event(), threading.Event()
    workers = []

    def analyse(value):
        workers.append(threading.current_thread())
        started.set()
        assert release.wait(3)

    def restore():
        assert started.wait(3), "Analysis was not scheduled before inference"
        release.set()
        return reference

    monkeypatch.setattr(texture, "texture_guide", analyse)
    result = texture.restore_with_texture(
        restore, source, reference, 2, lambda: None, torch.device("cpu")
    )
    assert result is reference
    assert workers and all(not worker.is_alive() for worker in workers)


def test_analysis_failures_propagate_and_do_not_leak_workers(monkeypatch):
    import threading

    workers = []

    def fail(value):
        workers.append(threading.current_thread())
        raise ValueError("analysis failure")

    monkeypatch.setattr(texture, "texture_guide", fail)
    source = grain()
    reference = np.zeros((384, 768, 3), np.float32)
    with pytest.raises(ValueError, match="analysis failure"):
        texture.restore_with_texture(
            lambda: reference, source, reference, 2, lambda: None, torch.device("cpu")
        )
    assert workers and all(not worker.is_alive() for worker in workers)


@pytest.mark.parametrize("shape,radius", [((1, 11), 3), ((2, 3), 6), ((128, 257), 12)])
def test_tensor_filter_matches_opencv_including_tiny_axes(shape, radius):
    # Run the tensor math on CPU in CI; installed GPU runs separately verify
    # the same path on the actual device and full-resolution images.
    rng = np.random.default_rng(17)
    source = rng.normal(0.4, 0.018, (*shape, 3)).astype(np.float32)
    restored = rng.normal(0.4, 0.01, source.shape).astype(np.float32)
    rms = np.full(shape, 0.018, np.float32)
    confidence = rng.uniform(0, 1, shape).astype(np.float32)
    epsilon = (3 * rms) ** 2 + 0.004**2
    expected = confidence[:, :, None] * (
        texture._detail(source, epsilon, radius)
        - texture._detail(restored, epsilon, radius)
    )
    actual = texture._gpu_adjustment(
        source, restored, rms, confidence, radius, torch.device("cpu")
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)
