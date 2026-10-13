"""DiT window layout and input-only scene routing, shared by PyTorch and TensorRT
restoration."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, cast

import cv2
import numpy as np
import torch

from .vosr2 import gaussian_weights, tile_positions

# DiT windows are 64 x 64 latents (512 px). Neighbours overlap by at least the regular
# grid's 8 latents, so each hand-over stays as far from a window's weak edges, and by at
# most 24, except the last window of a row or column, which ends on the image's edge.
WINDOW = 64
MIN_STEP, MAX_STEP = 40, 56
# A face window replaces the blend over its middle: out to SEAM_REACH - FEATHER / 2
# latents from its centre it alone counts (400 px across); its share then falls smoothly
# to 0 by SEAM_REACH + FEATHER / 2, half at SEAM_REACH (448 px across). That fall is
# gentler than the hand-over between two grid windows.
SEAM_REACH, FEATHER = 28, 6
# Latents along each edge of a face window outside the part it holds alone.
_EDGE = math.ceil((WINDOW - 1) / 2 - (SEAM_REACH - FEATHER / 2))
# Face windows: at most a fifth as many as the layout's windows.
SEAM_SHARE = 0.2
# A new shot starts where a frame's 1/16-size luma differs from the previous frame's by
# this mean (a cut). Gradual change, such as a pan, keeps the layout: re-planning
# mid-shot would redraw every window's invented detail at once.
SHOT_CHANGE = 0.08

Grid = list[tuple[int, int]]


class Layout(NamedTuple):
    """DiT windows (latent top-left corners): the grid, blended by Gaussian weights, and
    the face windows, each laid over that blend with `face_feather`."""

    grid: Grid
    faces: Grid

    @property
    def windows(self) -> Grid:
        """Every window, in the order they are restored: the grid, then the faces."""
        return self.grid + self.faces


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


def face_feather() -> torch.Tensor:
    """A face window's share of the blend over its 64 x 64 latents: 1 in its middle,
    falling as a raised cosine over FEATHER latents to 0 near its edges."""
    distance = (torch.arange(WINDOW, dtype=torch.float32) - (WINDOW - 1) / 2).abs()
    fall = ((SEAM_REACH + FEATHER / 2 - distance) / FEATHER).clamp(0, 1)
    share = 0.5 - 0.5 * torch.cos(torch.pi * fall)
    return (share[:, None] * share[None, :])[None, None]


def detail_energy(image: torch.Tensor) -> np.ndarray:
    rgb = image[0].permute(1, 2, 0).numpy() * 0.5 + 0.5
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return np.square(grey - cv2.GaussianBlur(grey, (0, 0), 2))


def _context_position(energy: np.ndarray) -> tuple[int, int]:
    """Select a 512-high, 64-wide context strip on the original patch phase.

    This is an experimental input sharpness heuristic, not a defocus classifier.
    Integral sums avoid rescanning the pixels for each overlapping candidate.
    """
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


def _context_strengths(
    energy: np.ndarray, grid: Grid, anchor: tuple[int, int]
) -> list[float]:
    """Measure input activity; never use restored/reference images for routing.

    A local window at >=25% of the strongest strip's mean detail energy keeps
    its unmodified prediction. At <=5%, scene context contributes fully, with
    a continuous transition between. These calibrated inference thresholds
    cannot distinguish every low-contrast texture from intentional defocus.
    """
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


def window_positions(cost: np.ndarray, count: int) -> list[int]:
    """`count` window starts, from 0 to len(cost) - WINDOW, whose hand-overs lie on the
    cheapest entries of `cost`.

    Two neighbours' Gaussian blend weights cross at the midpoint of their centres. Steps
    stay between MIN_STEP and MAX_STEP; the last one, onto the end, may be shorter, as in
    the regular grid. A slight preference for even steps settles ties, so a featureless
    axis keeps a near-regular layout.
    """
    end = len(cost) - WINDOW
    if count == 1:
        return [0]
    even = end / (count - 1)
    tie = 1e-3 * max(float(cost.mean()), 1e-12)
    best = np.full((count, end + 1), np.inf)
    back = np.zeros((count, end + 1), dtype=np.int64)
    best[0, 0] = 0
    for k in range(1, count):
        for step in range(1 if k == count - 1 else MIN_STEP, MAX_STEP + 1):
            if step > end:
                break
            start = np.arange(end + 1 - step)
            value = (
                best[k - 1, start]
                + cost[start + WINDOW // 2 + step // 2]
                + tie * (step - even) ** 2
            )
            target = best[k, step:]
            better = value < target
            target[better] = value[better]
            back[k, step:][better] = step
    if not np.isfinite(best[-1, end]):
        raise ValueError(f"No {count} windows cover {len(cost)} latents.")
    positions = [end]
    for k in range(count - 1, 0, -1):
        positions.append(positions[-1] - int(back[k, positions[-1]]))
    return positions[::-1]


class Face(NamedTuple):
    """A face's box in prepared-image pixels."""

    x: float
    y: float
    width: float
    height: float


_DETECTORS: dict[Path, cv2.FaceDetectorYN] = {}
_DETECTOR_LOCK = threading.Lock()


def find_faces(image: torch.Tensor, model: Path) -> list[Face]:
    """Faces in a padded RGB image (-1 to 1), found by YuNet on a quarter-size copy (it
    finds faces of about 10 to 300 pixels there)."""
    rgb = image[0].permute(1, 2, 0).numpy() * 0.5 + 0.5
    size = (max(1, rgb.shape[1] // 4), max(1, rgb.shape[0] // 4))
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    bgr = np.rint(np.clip(small[:, :, ::-1], 0, 1) * 255).astype(np.uint8)
    with _DETECTOR_LOCK:
        detector = _DETECTORS.get(model)
        if detector is None:
            detector = cv2.FaceDetectorYN.create(str(model), "", size, 0.6, 0.3, 50)
            _DETECTORS[model] = detector
        detector.setInputSize(size)
        # OpenCV returns None, not an empty array, when it finds no face.
        found = cast(np.ndarray | None, detector.detect(bgr)[1])
    if found is None:
        return []
    return [Face(*(float(v) * 4 for v in row[:4])) for row in found]


def _owners(grid: Grid, height: int, width: int) -> np.ndarray:
    """Per latent, the index of the window whose blend weight dominates it."""
    weights = gaussian_weights(WINDOW)[0, 0].numpy()
    best = np.full((height, width), -1.0, np.float32)
    owner = np.zeros((height, width), np.int32)
    for index, (y, x) in enumerate(grid):
        region = np.s_[y : y + WINDOW, x : x + WINDOW]
        better = weights > best[region]
        best[region][better] = weights[better]
        owner[region][better] = index
    return owner


def face_windows(faces: list[Face], grid: Grid, height: int, width: int) -> Grid:
    """Windows on the faces whose middle (their central half) the layout splits between
    windows, largest first: each centred on its face, where it holds that middle
    alone, and 2 * SEAM_REACH latents from the others; at most SEAM_SHARE of the
    layout's windows. A face whose middle is wider than that (over 800 px) keeps the
    layout's hand-overs."""
    owner = _owners(grid, height, width)
    centres: list[tuple[float, float]] = []
    windows: Grid = []
    for face in sorted(faces, key=lambda f: f.width * f.height, reverse=True):
        if len(windows) == int(SEAM_SHARE * len(grid)):
            break
        top, bottom = (
            int(np.clip(edge((face.y + face.height * part / 4) / 8), 0, height))
            for edge, part in ((math.floor, 1), (math.ceil, 3))
        )
        left, right = (
            int(np.clip(edge((face.x + face.width * part / 4) / 8), 0, width))
            for edge, part in ((math.floor, 1), (math.ceil, 3))
        )
        middle = owner[top:bottom, left:right]
        if middle.size == 0 or np.all(middle == middle.flat[0]):
            continue
        cy, cx = (face.y + face.height / 2) / 8, (face.x + face.width / 2) / 8
        y = int(np.clip(round(cy) - WINDOW // 2, 0, height - WINDOW))
        x = int(np.clip(round(cx) - WINDOW // 2, 0, width - WINDOW))
        holds = (
            y + _EDGE <= top
            and bottom <= y + WINDOW - _EDGE
            and x + _EDGE <= left
            and right <= x + WINDOW - _EDGE
        )
        if holds and all(
            max(abs(cy - b), abs(cx - a)) >= 2 * SEAM_REACH for b, a in centres
        ):
            centres.append((cy, cx))
            windows.append((y, x))
    return windows


def scene_windows(
    energy: np.ndarray, height: int, width: int, faces: list[Face] | None = None
) -> Layout:
    """Windows over height x width latents: as many as the regular 8-latent-overlap grid,
    placed so that their hand-overs run through the least detailed parts of the input,
    plus a window on each face that a hand-over still splits.

    Rows first, against each latent row's detail across the image; then each row's
    windows against the detail inside that row, so neighbouring rows' hand-overs need
    not line up. Neighbouring windows invent different amounts of detail, which shows
    most on faces: a face window gives a face's middle, up to 400 px across, one
    window's detail.
    """
    rms = cv2.resize(
        np.sqrt(energy), (width, height), interpolation=cv2.INTER_AREA
    ).astype(np.float64)
    per_row = len(tile_positions(width, WINDOW, 8))
    rows = window_positions(rms.sum(axis=1), len(tile_positions(height, WINDOW, 8)))
    grid = [
        (y, x)
        for y in rows
        for x in window_positions(rms[y : y + WINDOW].sum(axis=0), per_row)
    ]
    return Layout(grid, face_windows(faces or [], grid, height, width))


class WindowPlan:
    """A restoration's window layouts. `faces` is the YuNet model that finds faces for
    face windows. A video keeps one plan: layouts are planned at a shot's first frame and
    kept until a cut, so that seams stay put within a shot.

    The restorations of one video run one at a time, in frame order. Each calls
    `begin_frame` with its input, then `layout` once per restored tile, in the same order
    every frame (outer tiles each keep their own layout).
    """

    def __init__(self, faces: Path | None = None) -> None:
        self.faces = faces
        self.shape: tuple[int, ...] | None = None
        self.previous: np.ndarray | None = None
        self.layouts: list[Layout] = []
        self.calls = 0

    def begin_frame(self, source: np.ndarray) -> None:
        grey = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(
            grey,
            (max(1, grey.shape[1] // 16), max(1, grey.shape[0] // 16)),
            interpolation=cv2.INTER_AREA,
        )
        if (
            self.previous is None
            or source.shape != self.shape
            or float(np.abs(small - self.previous).mean()) > SHOT_CHANGE
        ):
            self.layouts = []
        self.shape = source.shape
        self.previous = small
        self.calls = 0

    def layout(self, plan: Callable[[], Layout]) -> Layout:
        if self.calls == len(self.layouts):
            self.layouts.append(plan())
        layout = self.layouts[self.calls]
        self.calls += 1
        return layout


def scene_routing(
    image: torch.Tensor, windows: WindowPlan | None = None
) -> tuple[Layout, tuple[int, int], list[float]]:
    """The DiT window layout of a padded image, its context strip and each window's
    context strength. `windows` carries the face model and keeps a video shot's layout."""
    energy = detail_energy(image)
    height, width = image.shape[-2] // 8, image.shape[-1] // 8

    def plan() -> Layout:
        model = None if windows is None else windows.faces
        faces = [] if model is None else find_faces(image, model)
        return scene_windows(energy, height, width, faces)

    layout = plan() if windows is None else windows.layout(plan)
    if len(layout.windows) == 1:
        # A single trained window uses no scene-context prediction.
        return layout, (0, 0), [0.0]
    anchor = _context_position(energy)
    return layout, anchor, _context_strengths(energy, layout.windows, anchor)
