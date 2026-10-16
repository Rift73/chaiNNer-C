"""Give flat grainy areas one fine, consistent grain across restoration windows.

Independent windows keep or erase the input's grain differently and add a regular
16px token-grid pattern. The input alone selects low-contrast, incoherent grain;
there an edge-aware swap replaces the output's finest detail band with grain
synthesized at the output resolution, as strong as the input's local grain and in
its colours. In a video, the areas whose fine texture changed since the previous
frame get that frame's own grain and the rest the seed's, so still areas keep one
grain (after a change, from the next still frame). Restored contours and defocused
areas keep the model's output. This
heuristic cannot distinguish intentional grain from photographic noise.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple, TypeVar

import cv2
import numpy as np
import torch
from torch.nn import functional

from nodes.impl import numpy_pool

# The grain's luma RMS relative to the input's local grain RMS. Chosen by eye on the
# owner's 2x sample (2026-10-13): 1.4 times the strength of the input grain enlarged
# with bicubic, which measured 1.13 to 1.16 times the input grain at 2x and 4x.
GRAIN_GAIN = 1.16
# The grain's softness: a Gaussian of sigma 0.6 output pixels.
GRAIN_KERNEL = cv2.getGaussianKernel(5, 0.6).astype(np.float32)[:, 0]
# BGR weights of cv2.COLOR_BGR2GRAY.
LUMA = np.array([0.114, 0.587, 0.299])

# Unsigned 32-bit values, in int64 tensors or Python ints.
Word = TypeVar("Word", torch.Tensor, int)


class Grain(NamedTuple):
    """The grain an output gets; `seed` chooses it. In a video, the areas whose fine
    texture changed since the `previous` frame's input get grain of their own `frame`,
    and the rest keep the seed's, so a still background does not shimmer."""

    seed: int
    frame: int = 0
    previous: np.ndarray | None = None


class TextureGuide(NamedTuple):
    """The input's flat grain, at source resolution."""

    confidence: np.ndarray
    rms: np.ndarray  # local luma RMS of the finest detail
    # Mixes independent unit noise (BGR) into the grain's colours, with unit luma variance.
    colour: np.ndarray
    # The share of the grain that is new this frame; None without a previous frame.
    change: np.ndarray | None


def _ramp(value: np.ndarray, low: float, high: float) -> np.ndarray:
    fraction = np.clip((value - low) / (high - low), 0, 1)
    return fraction * fraction * (3 - 2 * fraction)


def texture_guide(
    source: np.ndarray, previous: np.ndarray | None = None
) -> TextureGuide | None:
    """Estimate grain confidence, RMS and colour at original source resolution (BGR),
    and with the `previous` frame's input, where the fine texture changed."""
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
    share = (confidence * weight).reshape(-1, 1)
    if not np.any(share):
        return None
    # The grain's colours: its BGR covariance where it was detected, edges excluded.
    detail_bgr = (source - blur(source, 1.5)).reshape(-1, 3)
    covariance = (detail_bgr * share).T @ detail_bgr / share.sum()
    values, vectors = np.linalg.eigh(covariance)
    colour = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
    colour /= np.sqrt(max(LUMA @ covariance @ LUMA, 1e-12))
    # Expand the estimate across nearby edges, not an eroded edge exclusion:
    # the guided filter protects the edge itself without leaving a bare ring.
    confidence = blur(
        cv2.dilate(confidence, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))),
        3,
    )
    confidence[confidence < 0.02] = 0
    if not np.any(confidence):
        return None
    change = None
    if previous is not None:
        # 0 where the fine texture is the previous frame's, 1 where it changed as much
        # as independent grain does (sqrt(2) times its RMS): film grain, motion, cuts.
        grey = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        moved = detail - (grey - blur(grey, 1.5))
        moved = np.sqrt(blur(moved * moved, 3)) / np.maximum(rms, 1e-8)
        change = _ramp(moved, 0.2, 0.6)
    return TextureGuide(confidence, rms, colour.astype(np.float32), change)


def _detail(value: np.ndarray, epsilon: np.ndarray, radius: int) -> np.ndarray:
    def box(x: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(x, -1, (2 * radius + 1,) * 2)

    mean = box(value)
    variance = np.maximum(box(value * value) - mean * mean, 0)
    coefficient = variance / (variance + epsilon[:, :, None])
    return value - (box(coefficient) * value + box(mean - coefficient * mean))


def _reflect(value: torch.Tensor, radius: int) -> torch.Tensor:
    """Pad the last two axes by `radius` as OpenCV's REFLECT_101 does, repeating the
    reflection also for axes smaller than the radius."""
    for axis in (2, 3):
        size = value.shape[axis]
        indices = torch.arange(-radius, size + radius, device=value.device)
        if size == 1:
            indices.zero_()
        else:
            indices = indices.remainder(2 * (size - 1))
            indices = torch.minimum(indices, 2 * (size - 1) - indices)
        value = value.index_select(axis, indices)
    return value


def _times(x: Word, factor: int) -> Word:
    # x * factor modulo 2**32, in halves so that no product reaches 2**49.
    low = x * (factor & 0xFFFF)
    return (low + (((x * (factor >> 16)) & 0xFFFF) << 16)) & 0xFFFFFFFF


def _hash(x: Word) -> Word:
    """lowbias32 (C. Wellons) of 32-bit values, as int64 tensors or Python ints."""
    x = _times(x ^ (x >> 16), 0x7FEB352D)
    x = _times(x ^ (x >> 15), 0x846CA68B)
    return x ^ (x >> 16)


def _noise(
    seed: int,
    frame: int | None,
    colour: np.ndarray,
    region: tuple[slice, slice],
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Grain (1, 3, h, w) of unit luma variance for `region` of an output `width` wide:
    the seed's own, or with `frame`, that frame's.

    Every noise value is a hash of the seed, the frame and its pixel's position, so the
    strips agree with the whole image. The region's halo covers the blur.
    """
    key = _hash(seed & 0xFFFFFFFF)
    if frame is not None:
        key = _hash(key ^ _hash((frame + 1) & 0xFFFFFFFF))
    rows, columns = region
    y = torch.arange(rows.start, rows.stop, device=device)[None, :, None]
    x = torch.arange(columns.start, columns.stop, device=device)[None, None, :]
    c = torch.arange(3, device=device)[:, None, None]
    index = _hash(((y * width + x) * 3 + c) & 0xFFFFFFFF)
    first = (_hash(index ^ key).float() + 0.5) / 2**32
    # Its own stream also for key 0, which the hash keeps: equal inputs would skew the
    # values.
    second = _hash(index ^ key ^ 0x9E3779B9).float() / 2**32
    # Box-Muller: independent standard normal values.
    normal = torch.sqrt(-2 * torch.log(first)) * torch.cos(math.tau * second)
    taps = len(GRAIN_KERNEL)

    def blur(value: torch.Tensor, axis: int) -> torch.Tensor:
        # Float32 sums: CUDA convolutions may round to TF32.
        size = value.shape[axis] - taps + 1
        result = value.narrow(axis, 0, size) * float(GRAIN_KERNEL[0])
        for i in range(1, taps):
            result += value.narrow(axis, i, size) * float(GRAIN_KERNEL[i])
        return result

    normal = blur(blur(_reflect(normal[:, None], taps // 2), 2), 3)
    # The blur keeps sum(kernel**2) of the standard deviation.
    normal = normal[:, 0] / float(np.square(GRAIN_KERNEL).sum())
    mix = torch.from_numpy(colour).to(device)
    return torch.einsum("ij,jhw->ihw", mix, normal)[None]


def _gpu_adjustment(
    restored: np.ndarray,
    rms: np.ndarray,
    confidence: np.ndarray,
    grain: torch.Tensor,
    radius: int,
    device: torch.device,
) -> np.ndarray:
    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(value)).to(device)

    strength = tensor(rms)[None, None]
    epsilon = (3 * strength) ** 2 + 0.004**2

    def box(value: torch.Tensor) -> torch.Tensor:
        return functional.avg_pool2d(_reflect(value, radius), 2 * radius + 1, 1)

    image = tensor(restored.transpose(2, 0, 1))[None]
    mean = box(image)
    variance = (box(image * image) - mean * mean).clamp_min_(0)
    coefficient = variance / (variance + epsilon)
    detail = image - (box(coefficient) * image + box(mean - coefficient * mean))
    correction = tensor(confidence)[None, None] * (
        GRAIN_GAIN * strength * grain - detail
    )
    return correction[0].permute(1, 2, 0).contiguous().cpu().numpy()


def restore_with_texture(
    restore: Callable[[], np.ndarray],
    source: np.ndarray,
    scale: int,
    grain: Grain | None,
    check: Callable[[], None],
    device: torch.device,
) -> np.ndarray:
    """Give the BGR output of `restore` (`scale` times `source`) its grain, the CPU-only
    source analysis overlapping `restore`: a restoration or a temporal output.

    All neural backends, outer tiles and temporal outputs share this final pass;
    without `grain` (a temporal restoration, whose output gets its grain after
    stabilization) only `restore` runs. Filtering runs on bounded strips, using CUDA
    when the node already does; CPU nodes use OpenCV. A 2-radius halo makes the strips
    equivalent to whole-image filtering. Pixels with zero confidence retain the exact
    original float32 value. The scoped worker is joined on success, cancellation and
    failure. It only reads this call's frames; no image-dependent state survives into
    later runs.
    """
    if grain is None or max(source.shape[:2]) * scale <= 512:
        return restore()
    check()
    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="vosr-texture",
        initializer=numpy_pool.install,
    ) as pool:
        pending = pool.submit(texture_guide, source, grain.previous)
        restored = restore()
        check()
        guide = pending.result()
    return _add_grain(restored, scale, guide, grain, check, device)


def _add_grain(
    restored: np.ndarray,
    scale: int,
    guide: TextureGuide | None,
    grain: Grain,
    check: Callable[[], None],
    device: torch.device,
) -> np.ndarray:
    if guide is None:
        return restored
    height, width = restored.shape[:2]
    confidence = cv2.resize(
        guide.confidence, (width, height), interpolation=cv2.INTER_LINEAR
    )
    rms = cv2.resize(guide.rms, (width, height), interpolation=cv2.INTER_LINEAR)
    change = guide.change
    if change is not None:
        change = cv2.resize(change, (width, height), interpolation=cv2.INTER_LINEAR)
    # Other nodes (CPU, and MPS or DirectML, whose tensors NumPy cannot read) filter
    # with OpenCV and make the grain on the CPU.
    noise_device = device if device.type == "cuda" else torch.device("cpu")
    radius = max(1, round(1.5 * scale))
    # Also wider than the grain's blur.
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
        noise = _noise(grain.seed, None, guide.colour, region, width, noise_device)
        if change is not None and np.any(change[region]):
            # The two fields are independent, so the mix keeps unit variance.
            new = torch.from_numpy(np.ascontiguousarray(change[region]))
            new = new.to(noise_device)
            frame = _noise(
                grain.seed, grain.frame, guide.colour, region, width, noise_device
            )
            noise = (1 - new.square()).sqrt() * noise + new * frame
        if device.type == "cuda":
            adjustment = _gpu_adjustment(
                restored[region],
                rms[region],
                confidence[region],
                noise,
                radius,
                device,
            )[crop]
        else:
            epsilon = (3 * rms[region]) ** 2 + 0.004**2
            donor = (
                GRAIN_GAIN * rms[region][:, :, None] * noise[0].permute(1, 2, 0).numpy()
            )
            output_detail = _detail(restored[region], epsilon, radius)
            adjustment = confidence[target][:, :, None] * (donor - output_detail)[crop]
        result[target] = np.clip(restored[target] + adjustment, 0, 1)
    return result
