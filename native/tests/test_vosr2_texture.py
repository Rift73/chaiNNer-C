"""Synthesized flat-area grain must not follow the neural window or strip grid."""

import cv2
import numpy as np
import pytest
import torch

from nodes.impl.pytorch import vosr2_texture as texture

GRAIN = texture.Grain(42)
CPU = torch.device("cpu")


def grain(height=192, width=384):
    values = np.random.default_rng(67).normal(0.4, 0.018, (height, width))
    return np.repeat(values[:, :, None], 3, axis=2).astype(np.float32)


def apply(source, restored=None, scale=2, check=lambda: None, grain=GRAIN):
    if restored is None:
        reference = cv2.resize(
            source, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        restored = np.full_like(reference, 0.4)
    return texture.restore_with_texture(
        lambda: restored, source, scale, grain, check, CPU
    )


def noise(seed, frame, colour, height, width):
    region = np.s_[0:height, 0:width]
    field = texture._noise(seed, frame, colour, region, width, CPU)
    return field[0].permute(1, 2, 0).numpy()


def luma(image):
    return cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_BGR2GRAY)


def test_erased_and_retained_windows_converge_to_uniform_input_strength_grain():
    source = grain()
    reference = cv2.resize(source, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    restored = reference.copy()
    restored[:, 384:] = 0.4
    result = apply(source, restored)
    left = result[80:300, 80:300, 0]
    right = result[80:300, 464:684, 0]
    assert 0.8 < right.std() / left.std() < 1.2
    assert np.std(restored[80:300, 464:684]) < 1e-6
    # The erased side holds the grain alone: GRAIN_GAIN times the input's grain.
    guide = texture.texture_guide(source)
    assert guide is not None
    expected = texture.GRAIN_GAIN * guide.rms[40:150, 232:342].mean()
    assert 0.9 < luma(result[80:300, 464:684]).std() / expected < 1.1


def test_grain_is_fine_and_depends_on_seed_frame_and_position_only():
    colour = (np.eye(3) / np.sqrt((texture.LUMA**2).sum())).astype(np.float32)
    first = noise(7, None, colour, 256, 256)
    np.testing.assert_array_equal(first, noise(7, None, colour, 256, 256))
    grey = luma(first)
    assert abs(grey.mean()) < 0.02
    assert 0.97 < grey.std() < 1.03
    # Softened over about one pixel, with no structure beyond the blur.
    assert 0.2 < np.corrcoef(grey[:, 1:].ravel(), grey[:, :-1].ravel())[0, 1] < 0.6
    assert abs(np.corrcoef(grey[:, 3:].ravel(), grey[:, :-3].ravel())[0, 1]) < 0.03
    for other in ((7, 0), (7, 1), (8, None)):  # two frames' own, another seed's
        changed = luma(noise(*other, colour, 256, 256))
        assert abs(np.corrcoef(grey.ravel(), changed.ravel())[0, 1]) < 0.03


@pytest.mark.parametrize("seed,frame", [(0, None), (42, 41), (7, None)])
def test_grain_is_zero_mean_and_symmetric_for_every_key(seed, frame):
    # The hash keeps 0: seed 0 (the default) and frame seed - 1 have key 0.
    colour = (np.eye(3) / np.sqrt((texture.LUMA**2).sum())).astype(np.float32)
    grey = luma(noise(seed, frame, colour, 512, 512)).astype(np.float64)
    assert abs(grey.mean()) < 0.02
    assert 0.97 < grey.std() < 1.03
    assert abs(((grey - grey.mean()) ** 3).mean() / grey.std() ** 3) < 0.05


def test_video_grain_changes_only_where_the_input_does():
    previous = grain()
    source = previous.copy()
    # New film grain in the right half; the left half is a still background.
    source[:, 192:] = np.random.default_rng(5).normal(0.4, 0.018, (192, 192, 1))
    still = apply(source)
    video = apply(source, grain=texture.Grain(42, 9, previous))
    np.testing.assert_array_equal(video[:, :300], still[:, :300])
    right = np.s_[40:-40, 440:-40]
    fine_still = luma(still[right]) - 0.4
    fine_video = luma(video[right]) - 0.4
    assert abs(np.corrcoef(fine_still.ravel(), fine_video.ravel())[0, 1]) < 0.1
    assert 0.9 < fine_video.std() / fine_still.std() < 1.1


def test_codec_flicker_is_not_a_change():
    source = grain()
    previous = source.copy()
    flicker = np.random.default_rng(3).random(source.shape[:2]) < 0.03
    previous[flicker] += 1 / 255
    guide = texture.texture_guide(source, previous)
    assert guide is not None and guide.change is not None
    assert guide.change.max() == 0
    alone = texture.texture_guide(source)
    assert alone is not None and alone.change is None


@pytest.mark.parametrize("channel", [None, 1])
def test_grain_takes_the_input_grain_colours(channel):
    source = grain()
    if channel is not None:  # grain in green only
        source[:, :, [0, 2]] = 0.4
    guide = texture.texture_guide(source)
    assert guide is not None
    values = noise(42, None, guide.colour, 128, 128).reshape(-1, 3)
    spread = values.std(axis=0)
    if channel is None:
        np.testing.assert_allclose(spread, spread.mean(), rtol=0.01)
        assert np.corrcoef(values.T).min() > 0.999
    else:
        assert spread[[0, 2]].max() < 0.05 * spread[1]
    assert 0.97 < luma(values.reshape(128, 128, 3)).std() < 1.03


def test_guide_follows_source_coordinates_not_window_grid():
    source = grain(width=800)
    source[40:120, 400:404] += 0.4
    full = texture.texture_guide(source)
    shifted = texture.texture_guide(source[:, 32:])
    assert full is not None and shifted is not None
    np.testing.assert_allclose(
        full.confidence[:, 128:-96], shifted.confidence[:, 96:-96], atol=2e-6
    )


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
    assert guide.confidence[:, 172:184].mean() > 0.95
    assert guide.confidence[:, 200:212].mean() > 0.95
    reference = cv2.resize(source, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    restored = cv2.GaussianBlur(reference, (0, 0), 0.6)
    result = apply(source, restored)
    original_jump = restored[:, 400:430].mean((0, 1)) - restored[:, 340:370].mean(
        (0, 1)
    )
    actual_jump = result[:, 400:430].mean((0, 1)) - result[:, 340:370].mean((0, 1))
    np.testing.assert_allclose(actual_jump, original_jump, atol=0.002)

    def fine(columns):
        band = luma(result) - cv2.GaussianBlur(luma(result), (0, 0), 2)
        return band[20:-20, columns].std()

    assert fine(np.s_[344:368]) > 0.7 * fine(np.s_[100:300])


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
    height, width = reference.shape[:2]
    mask, rms = (cv2.resize(x, (width, height)) for x in (guide.confidence, guide.rms))
    epsilon = (3 * rms) ** 2 + 0.004**2
    donor = (
        texture.GRAIN_GAIN
        * rms[:, :, None]
        * noise(42, None, guide.colour, height, width)
    )
    whole = np.clip(
        restored + mask[:, :, None] * (donor - original(restored, epsilon, 3)), 0, 1
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
    mask = cv2.resize(guide.confidence, reference.shape[1::-1])
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
    assert not np.array_equal(first, apply(source, grain=texture.Grain(43)))
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


def test_temporal_restorations_get_no_grain(monkeypatch):
    def unexpected(*args):
        pytest.fail("A temporal restoration was analysed")

    monkeypatch.setattr(texture, "texture_guide", unexpected)
    restored = np.full((384, 768, 3), 0.4, np.float32)
    result = texture.restore_with_texture(
        lambda: restored, grain(), 2, None, lambda: None, CPU
    )
    assert result is restored


def test_analysis_overlaps_inference_and_workers_are_joined(monkeypatch):
    import threading

    source = grain()
    reference = cv2.resize(source, None, fx=2, fy=2)
    started, release = threading.Event(), threading.Event()
    workers = []

    def analyse(*args):
        workers.append(threading.current_thread())
        started.set()
        assert release.wait(3)

    def restore():
        assert started.wait(3), "Analysis was not scheduled before inference"
        release.set()
        return reference

    monkeypatch.setattr(texture, "texture_guide", analyse)
    result = texture.restore_with_texture(restore, source, 2, GRAIN, lambda: None, CPU)
    assert result is reference
    assert workers and all(not worker.is_alive() for worker in workers)


def test_analysis_failures_propagate_and_do_not_leak_workers(monkeypatch):
    import threading

    workers = []

    def fail(*args):
        workers.append(threading.current_thread())
        raise ValueError("analysis failure")

    monkeypatch.setattr(texture, "texture_guide", fail)
    source = grain()
    reference = np.zeros((384, 768, 3), np.float32)
    with pytest.raises(ValueError, match="analysis failure"):
        texture.restore_with_texture(
            lambda: reference, source, 2, GRAIN, lambda: None, CPU
        )
    assert workers and all(not worker.is_alive() for worker in workers)


@pytest.mark.parametrize("shape,radius", [((1, 11), 3), ((2, 3), 6), ((128, 257), 12)])
def test_tensor_filter_matches_opencv_including_tiny_axes(shape, radius):
    # Run the tensor math on CPU in CI; installed GPU runs separately verify
    # the same path on the actual device and full-resolution images.
    rng = np.random.default_rng(17)
    restored = rng.normal(0.4, 0.01, (*shape, 3)).astype(np.float32)
    rms = np.full(shape, 0.018, np.float32)
    confidence = rng.uniform(0, 1, shape).astype(np.float32)
    epsilon = (3 * rms) ** 2 + 0.004**2
    colour = np.eye(3, dtype=np.float32)
    donor = texture.GRAIN_GAIN * rms[:, :, None] * noise(42, None, colour, *shape)
    expected = confidence[:, :, None] * (
        donor - texture._detail(restored, epsilon, radius)
    )
    region = np.s_[0 : shape[0], 0 : shape[1]]
    actual = texture._gpu_adjustment(
        restored,
        rms,
        confidence,
        texture._noise(42, None, colour, region, shape[1], CPU),
        radius,
        CPU,
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)
