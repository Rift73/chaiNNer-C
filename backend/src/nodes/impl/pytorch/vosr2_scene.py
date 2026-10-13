"""Input-only scene routing shared by PyTorch and TensorRT restoration."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def pad_image(image: torch.Tensor) -> torch.Tensor:
    """Reflect small axes up to a trained window; singleton axes repeat."""
    height, width = image.shape[-2:]
    if height >= 512 and width >= 512:
        return image

    def indices(length: int) -> torch.Tensor:
        target = torch.arange(max(512, length))
        if length == 1:
            return torch.zeros_like(target)
        phase = target % (2 * length - 2)
        return torch.minimum(phase, 2 * length - 2 - phase)

    return image.index_select(2, indices(height)).index_select(3, indices(width))


def detail_energy(image: torch.Tensor) -> np.ndarray:
    rgb = image[0].permute(1, 2, 0).numpy() * 0.5 + 0.5
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return np.square(grey - cv2.GaussianBlur(grey, (0, 0), 2))


def context_position(image: torch.Tensor) -> tuple[int, int]:
    """Select a 512-high, 64-wide context strip on the original patch phase.

    This is an experimental input sharpness heuristic, not a defocus classifier.
    Integral sums avoid rescanning the pixels for each overlapping candidate.
    """
    return _context_position(detail_energy(image))


def _context_position(energy: np.ndarray) -> tuple[int, int]:
    integral = cv2.integral(energy)
    height, width = energy.shape
    ys = np.arange(0, height - 511, 16)
    xs = np.arange(0, width - 63, 16)
    sums = (
        integral[np.ix_(ys + 512, xs + 64)]
        - integral[np.ix_(ys, xs + 64)]
        - integral[np.ix_(ys + 512, xs)]
        + integral[np.ix_(ys, xs)]
    )
    # Stable ties choose the last row/column, matching the research max tuple.
    flat = sums.size - 1 - int(np.argmax(sums.ravel()[::-1]))
    row, col = divmod(flat, len(xs))
    return int(ys[row]), int(xs[col])


def context_strengths(
    image: torch.Tensor, grid: list[tuple[int, int]], anchor: tuple[int, int]
) -> list[float]:
    """Measure input activity; never use restored/reference images for routing.

    A local window at >=25% of the strongest strip's mean detail energy keeps
    its unmodified prediction. At <=5%, scene context contributes fully, with
    a continuous transition between. These calibrated inference thresholds
    cannot distinguish every low-contrast texture from intentional defocus.
    """
    return _context_strengths(detail_energy(image), grid, anchor)


def _context_strengths(
    energy: np.ndarray, grid: list[tuple[int, int]], anchor: tuple[int, int]
) -> list[float]:
    ay, ax = anchor
    reference_energy = float(energy[ay : ay + 512, ax : ax + 64].mean())
    if reference_energy <= 1e-12:
        return [0.0] * len(grid)
    return [
        float(
            np.clip(
                (
                    0.25
                    - energy[y * 8 : y * 8 + 512, x * 8 : x * 8 + 512].mean()
                    / reference_energy
                )
                / 0.20,
                0,
                1,
            )
        )
        for y, x in grid
    ]


def scene_routing(
    image: torch.Tensor, grid: list[tuple[int, int]]
) -> tuple[tuple[int, int], list[float]]:
    if len(grid) == 1:
        # A single trained window uses no scene-context prediction.
        return (0, 0), [0.0]
    energy = detail_energy(image)
    anchor = _context_position(energy)
    return anchor, _context_strengths(energy, grid, anchor)
