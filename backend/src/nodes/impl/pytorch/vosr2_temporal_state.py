"""Bounded, motion-tracked detail history for temporal VOSR's Persistent Detail."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from nodes.impl.native_threshold import channel_mean

from .vosr2_temporal import (
    Correspondence,
    TemporalObservation,
    motion_pair,
    remap,
)


def _base(frame: TemporalObservation) -> np.ndarray:
    height, width = frame.image.shape[:2]
    return cv2.resize(frame.source, (width, height), interpolation=cv2.INTER_LANCZOS4)


def _fine(residual: np.ndarray) -> np.ndarray:
    return residual - cv2.GaussianBlur(residual, (0, 0), 1.5)


def _upscale(
    match: Correspondence, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width, height = size
    flow = cv2.resize(match.flow, size, interpolation=cv2.INTER_LINEAR)
    flow *= np.array(
        [width / match.flow.shape[1], height / match.flow.shape[0]], np.float32
    )
    confidence = cv2.resize(match.confidence, size, interpolation=cv2.INTER_LINEAR)
    stable = match.confidence * np.exp(
        -((np.linalg.norm(match.flow, axis=2) / 0.20) ** 4)
    )
    stable = cv2.erode(stable, np.ones((5, 5), np.uint8))
    return flow, confidence, cv2.resize(stable, size, interpolation=cv2.INTER_LINEAR)


def _contrast(
    coarse: np.ndarray,
    fine: np.ndarray,
    reference: np.ndarray,
    support: np.ndarray,
    minimum: float,
) -> np.ndarray:
    """Match total-image detail energy, including its coarse/fine cross term."""

    def high_pass(image: np.ndarray) -> np.ndarray:
        return image - cv2.GaussianBlur(image, (0, 0), 0.7)

    def local_mean(image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(channel_mean(image), (0, 0), 4)

    f = high_pass(fine)
    c = high_pass(coarse)
    a = local_mean(f * f)
    b = local_mean(f * c)
    target = local_mean(high_pass(reference) ** 2)
    gain = np.clip(
        (-b + np.sqrt(np.maximum(b * b + a * (target - local_mean(c * c)), 0)))
        / np.maximum(a, 1e-12),
        minimum,
        1.6,
    )
    gain[(a < 1e-10) | (support == 0)] = 1
    output = np.ascontiguousarray(np.clip(coarse + fine * gain[:, :, None], 0, 1))
    output[support == 0] = reference[support == 0]
    return output


@dataclass(frozen=True)
class DetailState:
    fine: np.ndarray
    base: np.ndarray


def initialize_detail(
    frames: Sequence[TemporalObservation],
    strength: float,
    check: Callable[[], None],
) -> tuple[np.ndarray, DetailState]:
    """Seed from a small consensus rather than perpetuating the first model guess."""
    if not 1 <= len(frames) <= 5:
        raise ValueError("Persistent detail initialization needs one to five frames.")
    first = frames[0]
    first_base = _base(first)
    if len(frames) == 1 or strength == 0:
        return first.image, DetailState(_fine(first.image - first_base), first_base)
    reference = frames[len(frames) // 2]
    height, width = reference.image.shape[:2]
    reference_base = _base(reference)
    residual = reference.image - reference_base
    matches = []
    for frame in frames:
        check()
        if frame.image.shape != reference.image.shape:
            raise ValueError(
                "All temporal sequence frames must have identical dimensions."
            )
        matches.append(motion_pair(reference.source, frame.source, check)[0])

    def sample(frame: TemporalObservation, match: Correspondence) -> np.ndarray:
        check()
        if not np.any(match.confidence):
            return residual
        flow, _, support = _upscale(match, (width, height))
        aligned = remap(frame.image - _base(frame), flow, cv2.INTER_LANCZOS4)
        return residual + (aligned - residual) * support[:, :, None]

    # Two passes avoid stacking five output-size RGB buffers plus their costs.
    # The medoid chooses an observed local texture; it does not average incompatible
    # fine patterns. Only the small input-resolution correspondences are cached.
    centroid = np.zeros_like(reference.image)
    for frame, match in zip(frames, matches, strict=True):
        centroid += sample(frame, match)
    centroid /= len(frames)
    best = np.full((height, width), np.inf, np.float32)
    canonical = np.empty_like(reference.image)
    for frame, match in zip(frames, matches, strict=True):
        value = sample(frame, match)
        cost = cv2.GaussianBlur(channel_mean((value - centroid) ** 2), (0, 0), 4)
        selected = cost < best
        canonical[selected] = value[selected]
        np.minimum(best, cost, out=best)
    del matches, centroid, best, value, cost, selected
    reference_fine = canonical - cv2.GaussianBlur(canonical, (0, 0), 1)
    restored = _contrast(
        reference_base + canonical - reference_fine,
        reference_fine,
        reference.image,
        np.ones((height, width), np.float32),
        1,
    )
    match, _ = motion_pair(first.source, reference.source, check)
    if not np.any(match.confidence):
        return first.image, DetailState(_fine(first.image - first_base), first_base)
    flow, _, support = _upscale(match, (width, height))
    aligned = remap(restored - reference_base, flow, cv2.INTER_LANCZOS4)
    difference = aligned - (first.image - first_base)
    disagreement = cv2.GaussianBlur(
        channel_mean(np.abs(cv2.GaussianBlur(difference, (0, 0), 3))), (0, 0), 2
    )
    support *= np.exp(-((disagreement / 0.05) ** 2))
    support[support < np.finfo(np.float32).eps] = 0
    if not np.any(difference) or not np.any(support):
        return first.image, DetailState(_fine(first.image - first_base), first_base)
    estimate = (
        first.image + difference * (support * strength / (strength + 0.115))[:, :, None]
    )
    fine = estimate - first_base
    fine -= cv2.GaussianBlur(fine, (0, 0), 1)
    output = _contrast(estimate - fine, fine, first.image, support, 1)
    check()
    return output, DetailState(_fine(output - first_base), first_base)


def carry_detail(
    frame: TemporalObservation,
    state: DetailState | None,
    strength: float,
    check: Callable[[], None],
) -> tuple[np.ndarray, DetailState]:
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Temporal strength must be between 0 and 1.")
    check()
    base = _base(frame)
    fresh = _fine(frame.image - base)
    match = frame.previous_match
    if state is None or match is None or strength == 0 or not np.any(match.confidence):
        return frame.image, DetailState(fresh, base)
    if state.fine.shape != frame.image.shape or state.base.shape != frame.image.shape:
        raise ValueError("All temporal sequence frames must have identical dimensions.")
    height, width = frame.image.shape[:2]
    flow, confidence, stable = _upscale(match, (width, height))
    held = remap(state.fine, flow, cv2.INTER_LANCZOS4)
    change = np.max(np.abs(remap(state.base, flow, cv2.INTER_LINEAR) - base), axis=2)
    # Confidence follows the observations, not the model's changing guesses.
    trust = confidence * np.exp(
        -((cv2.GaussianBlur(change, (0, 0), 1) / (6 / 255)) ** 2)
    )
    trust *= stable
    trust = np.minimum(
        cv2.GaussianBlur(trust, (0, 0), 2), 0.95 * min(1, strength / 0.65)
    )
    trust[(confidence == 0) | (trust < np.finfo(np.float32).eps)] = 0
    # Energy normalization alone can lock a texture whose phase contradicts every
    # new observation. Reject that history instead of amplifying it indefinitely.
    agreement = cv2.GaussianBlur(np.sum(held * fresh, axis=2), (0, 0), 4)
    trust[agreement <= 0] = 0
    if not np.any(trust) or np.array_equal(held, fresh):
        return frame.image, DetailState(fresh, base)
    mixed = trust[:, :, None] * held + (1 - trust[:, :, None]) * fresh

    def energy(value: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(np.sum(value * value, axis=2), (0, 0), 4) + 1e-9

    # Two-sided gain lets held sharp texture fade during a genuine focus change.
    # At least 5% is a fresh observation; no history image or age list grows.
    carried = (
        mixed * np.clip(np.sqrt(energy(fresh) / energy(mixed)), 0.8, 1.25)[:, :, None]
    )
    carried[trust == 0] = fresh[trust == 0]
    output = _contrast(frame.image - fresh, carried, frame.image, trust, 0)
    check()
    # Display contrast correction never feeds back into the detail state.
    return output, DetailState(carried, base)
