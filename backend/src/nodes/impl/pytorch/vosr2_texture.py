"""Keep flat fine texture consistent across independent restoration windows.

The input alone selects low-contrast, incoherent grain. An edge-aware detail
swap retains that grain without replacing restored contours or defocused areas.
This heuristic cannot distinguish intentional grain from photographic noise.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from torch.nn import functional

from nodes.impl import numpy_pool


def _ramp(value: np.ndarray, low: float, high: float) -> np.ndarray:
    fraction = np.clip((value - low) / (high - low), 0, 1)
    return fraction * fraction * (3 - 2 * fraction)


def texture_guide(source: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Estimate grain confidence and RMS at original source resolution (BGR)."""
    grey = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

    def blur(value: np.ndarray, sigma: float = 6) -> np.ndarray:
        return cv2.GaussianBlur(value, (0, 0), sigma)

    dx = cv2.Sobel(source, cv2.CV_32F, 1, 0) / 8
    dy = cv2.Sobel(source, cv2.CV_32F, 0, 1) / 8
    edge = np.sqrt(np.max(dx * dx + dy * dy, axis=2))
    weight = 1 - cv2.dilate(_ramp(edge, 0.03, 0.06), np.ones((5, 5), np.uint8))
    denominator = np.maximum(blur(weight), 1e-8)

    def mean(value: np.ndarray) -> np.ndarray:
        return blur(value * weight) / denominator

    detail = grey - blur(grey, 1.5)
    rms = np.sqrt(np.maximum(mean(detail * detail), 0))
    base = blur(grey, 3)
    dx = cv2.Sobel(base, cv2.CV_32F, 1, 0) / 8
    dy = cv2.Sobel(base, cv2.CV_32F, 0, 1) / 8
    ratio = np.sqrt(np.maximum(mean(dx * dx + dy * dy), 0)) / np.maximum(rms, 1e-8)
    dx = cv2.Sobel(detail, cv2.CV_32F, 1, 0) / 8
    dy = cv2.Sobel(detail, cv2.CV_32F, 0, 1) / 8
    xx, yy, xy = mean(dx * dx), mean(dy * dy), mean(dx * dy)
    coherence = np.sqrt((xx - yy) ** 2 + 4 * xy * xy) / np.maximum(xx + yy, 1e-8)
    confidence = (
        _ramp(rms, 0.004, 0.010)
        * (1 - _ramp(ratio, 0.12, 0.25))
        * (1 - _ramp(coherence, 0.2, 0.35))
        * (1 - _ramp(rms, 0.04, 0.08))
    )
    if not np.any(confidence):
        return None
    # Expand the estimate across nearby edges, not an eroded edge exclusion:
    # the guided filter protects the edge itself without leaving a bare ring.
    confidence = blur(
        cv2.dilate(confidence, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))),
        3,
    )
    confidence[confidence < 0.02] = 0
    return (confidence, rms) if np.any(confidence) else None


def _detail(value: np.ndarray, epsilon: np.ndarray, radius: int) -> np.ndarray:
    def box(x: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(x, -1, (2 * radius + 1,) * 2)

    mean = box(value)
    variance = np.maximum(box(value * value) - mean * mean, 0)
    coefficient = variance / (variance + epsilon[:, :, None])
    return value - (box(coefficient) * value + box(mean - coefficient * mean))


def _gpu_adjustment(
    source: np.ndarray,
    restored: np.ndarray,
    rms: np.ndarray,
    confidence: np.ndarray,
    radius: int,
    device: torch.device,
) -> np.ndarray:
    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(value)).to(device)

    epsilon = (3 * tensor(rms)[None, None]) ** 2 + 0.004**2

    def box(value: torch.Tensor) -> torch.Tensor:
        # Repeat REFLECT_101 also for axes smaller than the filter radius.
        for axis in (2, 3):
            size = value.shape[axis]
            indices = torch.arange(-radius, size + radius, device=device)
            if size == 1:
                indices.zero_()
            else:
                indices = indices.remainder(2 * (size - 1))
                indices = torch.minimum(indices, 2 * (size - 1) - indices)
            value = value.index_select(axis, indices)
        return functional.avg_pool2d(value, 2 * radius + 1, 1)

    def detail(value: np.ndarray) -> torch.Tensor:
        image = tensor(value.transpose(2, 0, 1))[None]
        mean = box(image)
        variance = (box(image * image) - mean * mean).clamp_min_(0)
        coefficient = variance / (variance + epsilon)
        return image - (box(coefficient) * image + box(mean - coefficient * mean))

    correction = tensor(confidence)[None, None] * (detail(source) - detail(restored))
    return correction[0].permute(1, 2, 0).contiguous().cpu().numpy()


def preserve_texture(
    source: np.ndarray,
    reference: np.ndarray,
    restored: np.ndarray,
    scale: int,
    check: Callable[[], None],
    device: torch.device,
) -> np.ndarray:
    """Correct BGR output using the RGB resize reference, after colour correction.

    All neural backends and outer tiles share this final source-derived pass.
    Filtering runs on bounded strips, using CUDA when the node already does.
    CPU nodes use OpenCV. No model transfer or extra inference is needed.
    A 2-radius halo makes the strips equivalent to whole-image guided filtering.
    Pixels with zero confidence retain the exact original float32 value.
    """
    height, width = restored.shape[:2]
    if height <= 512 and width <= 512:
        return restored
    check()
    guide = texture_guide(source)
    return _preserve_texture(reference, restored, scale, guide, check, device)


def restore_with_texture(
    restore: Callable[[], np.ndarray],
    source: np.ndarray,
    reference: np.ndarray,
    scale: int,
    check: Callable[[], None],
    device: torch.device,
) -> np.ndarray:
    """Overlap the CPU-only source analysis with neural inference.

    The scoped worker is joined on success, cancellation and failure. It only
    reads this call's source; no image-dependent state survives into later runs.
    """
    if max(reference.shape[:2]) <= 512:
        return restore()
    check()
    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="vosr-texture",
        initializer=numpy_pool.install,
    ) as pool:
        pending = pool.submit(texture_guide, source)
        restored = restore()
        check()
        guide = pending.result()
    return _preserve_texture(reference, restored, scale, guide, check, device)


def _preserve_texture(
    reference: np.ndarray,
    restored: np.ndarray,
    scale: int,
    guide: tuple[np.ndarray, np.ndarray] | None,
    check: Callable[[], None],
    device: torch.device,
) -> np.ndarray:
    if guide is None:
        return restored
    height, width = restored.shape[:2]
    confidence, rms = guide
    confidence = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_LINEAR)
    rms = cv2.resize(rms, (width, height), interpolation=cv2.INTER_LINEAR)
    radius = max(1, round(1.5 * scale))
    halo = 2 * radius
    result = restored.copy()
    for top in range(0, height, 128):
        check()
        bottom = min(top + 128, height)
        columns = np.flatnonzero(np.any(confidence[top:bottom] > 0, axis=0))
        if not len(columns):
            continue
        left, right = int(columns[0]), int(columns[-1]) + 1
        y0, y1 = max(0, top - halo), min(height, bottom + halo)
        x0, x1 = max(0, left - halo), min(width, right + halo)
        region = np.s_[y0:y1, x0:x1]
        crop = np.s_[top - y0 : bottom - y0, left - x0 : right - x0]
        target = np.s_[top:bottom, left:right]
        if device.type == "cuda":
            adjustment = _gpu_adjustment(
                reference[region][:, :, ::-1],
                restored[region],
                rms[region],
                confidence[region],
                radius,
                device,
            )[crop]
        else:
            epsilon = (3 * rms[region]) ** 2 + 0.004**2
            source_detail = _detail(reference[region][:, :, ::-1], epsilon, radius)
            output_detail = _detail(restored[region], epsilon, radius)
            adjustment = (
                confidence[target][:, :, None] * (source_detail - output_detail)[crop]
            )
        result[target] = np.clip(restored[target] + adjustment, 0, 1)
    return result
