"""Motion-compensated stabilization of independently restored video frames."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import TypeVar

import cv2
import numpy as np

from nodes.impl.native_threshold import channel_mean

T = TypeVar("T")


class TemporalMode(Enum):
    THREE_FRAMES = 0
    PERSISTENT_DETAIL = 1


@dataclass(frozen=True)
class Correspondence:
    flow: np.ndarray
    confidence: np.ndarray


@lru_cache(maxsize=2)
def coordinates(height: int, width: int) -> np.ndarray:
    y, x = np.mgrid[:height, :width].astype(np.float32)
    grid = np.stack((x, y), axis=2)
    grid.setflags(write=False)
    return grid


def remap(image: np.ndarray, flow: np.ndarray, interpolation: int) -> np.ndarray:
    mapped = coordinates(*flow.shape[:2]) + flow
    return cv2.remap(
        image,
        mapped[:, :, 0],
        mapped[:, :, 1],
        interpolation,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def validate_image(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.float32:
        raise ValueError("Temporal VOSR requires float32 RGB images without alpha.")
    if min(image.shape[:2]) < 16 or max(image.shape[:2]) >= 32767:
        raise ValueError("Temporal VOSR image dimensions must be 16 to 32766 pixels.")
    if not np.isfinite(image).all():
        raise ValueError("Temporal VOSR input contains non-finite values.")


def motion_pair(
    first: np.ndarray, second: np.ndarray, check: Callable[[], None]
) -> tuple[Correspondence, Correspondence]:
    """Both directions share one consistency check; flow is in input pixels."""
    validate_image(first)
    validate_image(second)
    if first.shape != second.shape:
        raise ValueError("All temporal sequence frames must have identical dimensions.")
    check()
    height, width = first.shape[:2]
    gray = [
        np.rint(np.clip(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 1) * 255).astype(
            np.uint8
        )
        for img in (first, second)
    ]
    flows = []
    for source, target in (gray, gray[::-1]):
        check()
        algorithm = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        algorithm.setFinestScale(0)
        algorithm.setGradientDescentIterations(32)
        algorithm.setVariationalRefinementIterations(8)
        initial = np.zeros((height, width, 2), np.float32)
        flows.append(algorithm.calc(source, target, initial))
    result = []
    for source, target, flow, reverse in (
        (first, second, flows[0], flows[1]),
        (second, first, flows[1], flows[0]),
    ):
        check()
        mapped = coordinates(height, width) + flow
        consistency = np.linalg.norm(
            flow + remap(reverse, flow, cv2.INTER_LINEAR), axis=2
        )
        difference = np.max(
            np.abs(source - remap(target, flow, cv2.INTER_LINEAR)), axis=2
        )
        valid = (
            (mapped[:, :, 0] >= 1)
            & (mapped[:, :, 0] < width - 1)
            & (mapped[:, :, 1] >= 1)
            & (mapped[:, :, 1] < height - 1)
        )
        tolerance = np.where(np.linalg.norm(flow, axis=2) > 0.5, 0.25, 0.5)
        confidence = np.exp(
            -((consistency / tolerance) ** 2) - (difference / 0.04) ** 2
        ).astype(np.float32)
        confidence *= valid
        confidence = cv2.erode(confidence, np.ones((3, 3), np.uint8))
        # Cuts and large unmatched transitions contribute exactly nothing.
        if np.median(difference) > 0.12 or np.mean(confidence > 0.25) < 0.70:
            confidence.fill(0)
        result.append(Correspondence(flow, confidence))
    return result[0], result[1]


@dataclass(frozen=True)
class RestoredFrame:
    source: np.ndarray
    restored: np.ndarray
    residual: np.ndarray

    @staticmethod
    def create(source: np.ndarray, restored: np.ndarray) -> RestoredFrame:
        validate_image(source)
        validate_image(restored)
        height, width = restored.shape[:2]
        base = cv2.resize(source, (width, height), interpolation=cv2.INTER_LANCZOS4)
        return RestoredFrame(source, restored, restored - base)


def _start(pool: Executor | None, task: Callable[[], T]) -> Callable[[], T]:
    """Run task on the pool, or now without one; calling the result waits for it."""
    if pool is None:
        value = task()
        return lambda: value
    return pool.submit(task).result


def _high_pass(image: np.ndarray) -> np.ndarray:
    return image - cv2.GaussianBlur(image, (0, 0), 0.7)


def _local_mean(image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(channel_mean(image), (0, 0), 4)


def _align(
    neighbor: RestoredFrame, match: Correspondence, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """The neighbour's residual and its coarse band, warped onto the current frame."""
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
    return aligned, aligned_low


@dataclass(frozen=True)
class _Contribution:
    """One neighbour's terms of stabilize's sums, before they are added."""

    stable: np.ndarray
    weight: np.ndarray
    delta: np.ndarray
    low_delta: np.ndarray
    fine_weight: np.ndarray
    high_delta: np.ndarray


def _contribution(
    residual: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    match: Correspondence,
    aligned: np.ndarray,
    aligned_low: np.ndarray,
    strength: float,
) -> _Contribution:
    height, width = residual.shape[:2]
    aligned_high = aligned - aligned_low
    difference = aligned - residual
    disagreement = cv2.GaussianBlur(channel_mean(np.abs(difference)), (0, 0), 1)
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
    weight = (confidence * np.exp(-((disagreement / 0.10) ** 2)) * (strength * 0.5))[
        :, :, None
    ]
    low_difference = aligned_low - low
    coarse_disagreement = cv2.GaussianBlur(
        channel_mean(np.abs(low_difference)), (0, 0), 1
    )
    fine_weight = (
        confidence * np.exp(-((coarse_disagreement / 0.03) ** 2)) * strength * 3
    )[:, :, None]
    return _Contribution(
        cv2.resize(stable, (width, height), interpolation=cv2.INTER_LINEAR),
        weight,
        difference * weight,
        low_difference * weight,
        fine_weight,
        (aligned_high - high) * fine_weight,
    )


def stabilize(
    current: RestoredFrame,
    neighbors: Sequence[tuple[RestoredFrame, Correspondence]],
    strength: float,
    check: Callable[[], None],
    pool: Executor | None = None,
) -> np.ndarray:
    """Keep the current observation; blend only aligned restoration residuals.

    Coarse residuals retain conservative weights. Fine residuals get stronger
    consensus, with bounded local contrast recovery to avoid averaging away
    detail. Invalid matches contribute nothing. Inputs are independently restored
    frames, never earlier stabilized outputs; support stays at three frames.

    With a pool, the work that does not depend on other work runs on its threads: each
    neighbour's terms (while the next neighbour is warped), the quotients and the
    detail bands. The warps stay on this thread: OpenCV already spreads one over every
    core, and two at once are slower. Every value is the same expression of the same
    inputs and the sums add the neighbours in order, so the output equals the
    one-thread result bit for bit. The pool's tasks never wait for one another.
    """
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Temporal strength must be between 0 and 1.")
    if len(neighbors) > 2:
        raise ValueError("Temporal stabilization accepts at most two neighbors.")
    check()
    if strength == 0 or not neighbors:
        return current.restored
    height, width = current.restored.shape[:2]
    restored_energy = _start(
        pool, lambda: _local_mean(_high_pass(current.restored) ** 2)
    )
    low = cv2.GaussianBlur(current.residual, (0, 0), 1)
    high = current.residual - low
    pending: list[Callable[[], _Contribution]] = []
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
        aligned, aligned_low = _align(neighbor, match, width, height)
        pending.append(
            _start(
                pool,
                lambda match=match, aligned=aligned, aligned_low=aligned_low: (
                    _contribution(
                        current.residual,
                        low,
                        high,
                        match,
                        aligned,
                        aligned_low,
                        strength,
                    )
                ),
            )
        )
    delta = np.zeros_like(current.restored)
    denominator = np.ones((height, width, 1), dtype=np.float32)
    low_delta = np.zeros_like(delta)
    high_delta = np.zeros_like(delta)
    high_denominator = np.ones_like(denominator)
    stable_support = np.ones((height, width), dtype=np.float32)
    for wait in pending:
        contribution = wait()
        stable_support = np.minimum(stable_support, contribution.stable)
        delta += contribution.delta
        low_delta += contribution.low_delta
        denominator += contribution.weight
        high_delta += contribution.high_delta
        high_denominator += contribution.fine_weight
    check()
    if not np.any(delta) and not np.any(high_delta) and not np.any(low_delta):
        restored_energy()  # settle the pool's task before returning
        return current.restored
    # Each quotient serves two expressions below; it is the same value in both.
    conservative = _start(pool, lambda: delta / denominator)
    low_correction = _start(pool, lambda: low_delta / denominator)
    fine = high + high_delta / high_denominator
    coarse = current.restored - high + low_correction()
    legacy = current.restored + conservative()
    support = stable_support[:, :, None]
    coarse = legacy + (coarse - legacy) * support

    # Solve E[(coarse + gain*fine)^2] = target in the detail band. Include
    # the cross term: normalizing residual energy alone can soften input detail.
    # Spatially varying gains make this a local estimate, not an exact bound.
    legacy_energy = _start(pool, lambda: _local_mean(_high_pass(legacy) ** 2))
    fine_pass = _start(pool, lambda: _high_pass(fine * support))
    coarse_band = _high_pass(coarse)
    c = _local_mean(coarse_band * coarse_band)
    fine_band = fine_pass()
    fine_energy = _start(pool, lambda: _local_mean(fine_band * fine_band))
    b = _local_mean(fine_band * coarse_band)
    target = np.minimum(legacy_energy(), restored_energy())
    a = fine_energy()
    gain = np.clip(
        (-b + np.sqrt(np.maximum(b * b + a * (target - c), 0))) / np.maximum(a, 1e-12),
        0,
        1.6,
    )
    gain[a < 1e-10] = 1
    correction = low_correction() + fine * gain[:, :, None] - high
    correction[high_denominator[:, :, 0] == 1] = 0
    correction = conservative() + (correction - conservative()) * support
    check()
    return np.ascontiguousarray(np.clip(current.restored + correction, 0, 1))


@dataclass(frozen=True)
class TemporalObservation:
    """A frame's stabilized output, its input and its match to the previous frame."""

    image: np.ndarray
    source: np.ndarray
    previous_match: Correspondence | None
