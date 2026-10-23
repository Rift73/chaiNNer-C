"""Get Native Resolution's kernel decision (native/src/getnative_kernel.cpp) through the session:
least-squares evidence per frame on its leading grid, the paired rule across the frames on the
consensus grid (a preset is identified from 7 frames on), the one-frame rule for an image, and
the node carrying the decided kernel. Frames: natively sharp 960x540 planes upscaled to 1240x698
(the x1.29 of real anime releases) with Vernier's forward operator and a preset, in 8 bits."""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import cache

import numpy as np
import pytest
from test_getnative import natively_sharp
from test_getnative_node import Context, node, run

from nodes.impl import native_getnative as gn

NATIVE, OBSERVED = (960, 540), (1240, 698)
CATMULL_ROM = "Bicubic B=0 C=0.5"


def cubic(b: float, c: float) -> Callable[[np.ndarray], np.ndarray]:
    def value(x: np.ndarray) -> np.ndarray:
        x = np.abs(x)
        near = (12 - 9 * b - 6 * c) * x**3 + (-18 + 12 * b + 6 * c) * x**2 + 6 - 2 * b
        far = (
            (-b - 6 * c) * x**3
            + (6 * b + 30 * c) * x**2
            + (-12 * b - 48 * c) * x
            + 8 * b
            + 24 * c
        )
        return np.where(x < 1, near / 6, np.where(x < 2, far / 6, 0.0))

    return value


def lanczos2(x: np.ndarray) -> np.ndarray:
    return np.where(np.abs(x) < 2, np.sinc(x) * np.sinc(x / 2), 0.0)


# Vernier's presets as (kernel, radius), by Vernier's names.
PRESETS = {
    "Bilinear": (lambda x: np.maximum(0.0, 1 - np.abs(x)), 1),
    CATMULL_ROM: (cubic(0, 0.5), 2),
    "Lanczos 2": (lanczos2, 2),
    "Mitchell": (cubic(1 / 3, 1 / 3), 2),
}


def forward(plane: np.ndarray, size: int, axis: int, preset: str) -> np.ndarray:
    """Vernier's Resampler::forward along `axis` at integer geometry: the preset's taps around
    each observed pixel centre, clamped border, weights normalized per row."""
    kernel, radius = PRESETS[preset]
    n = plane.shape[axis]
    matrix = np.zeros((size, n))
    for j in range(size):
        pos = (j + 0.5) * n / size - 0.5
        taps = np.arange(
            int(np.floor(pos)) - radius + 1, int(np.floor(pos)) + radius + 1
        )
        weights = kernel(pos - taps)
        np.add.at(matrix[j], np.clip(taps, 0, n - 1), weights / weights.sum())
    return np.moveaxis(np.tensordot(matrix, np.moveaxis(plane, axis, 0), 1), 0, axis)


@cache
def planted(preset: str, seed: int, native: tuple[int, int] = NATIVE) -> np.ndarray:
    """An 8-bit upscale of a natively sharp plane to OBSERVED."""
    up = forward(natively_sharp(*native, seed=seed), OBSERVED[0], 1, preset)
    up = forward(up, OBSERVED[1], 0, preset)
    return np.round(np.clip(up, 0, 1) * 255) / 255


def summary(planes: list[np.ndarray], kernels: int = 0) -> dict:
    session = gn.Session(gn.SearchOptions(500, 600, threads=8, kernels=kernels))
    for k, plane in enumerate(planes):
        session.add_frame(plane, gn.FrameFacts("", "Still image", f"plane {k}"))
    return json.loads(session.summary(len(planes), len(planes), 1, None))


def frames(preset: str, count: int) -> list[np.ndarray]:
    return [planted(preset, seed) for seed in range(1, count + 1)]


def test_seven_bilinear_frames_identify_bilinear():
    result = summary(frames("Bilinear", 7))
    assert result["kernel_decision"] == {
        "status": "supported_model",
        "band": ["Bilinear"],
        "carried": "Bilinear",
        "rule": "cross_frame",
        "frames": 7,
    }
    assert result["selection"]["kernel"] == "Bilinear"
    assert (
        result["selection"]["rule"]
        == "kernel decision supported_model (cross_frame rule)"
    )


def test_seven_catmull_rom_frames_identify_catmull_rom():
    decision = summary(frames(CATMULL_ROM, 7))["kernel_decision"]
    assert (decision["status"], decision["carried"]) == ("supported_model", CATMULL_ROM)


def test_lanczos2_and_catmull_rom_stay_a_band():
    # The two kernels' responses are too close at x1.29 to tell apart: a band, carrying the first
    # of the smallest support.
    result = summary(frames("Lanczos 2", 7))
    decision = result["kernel_decision"]
    assert decision["status"] == "ambiguous_band"
    assert decision["band"] == [CATMULL_ROM, "Lanczos 2"]
    assert (decision["carried"], decision["rule"]) == (CATMULL_ROM, "cross_frame_band")
    assert result["selection"]["kernel"] == CATMULL_ROM


def test_fewer_than_seven_frames_never_identify_a_kernel():
    decision = summary(frames("Bilinear", 6))["kernel_decision"]
    assert (decision["status"], decision["rule"]) == ("unresolved", "parsimony")
    assert decision["carried"] == "Bilinear"


def test_one_frame_uses_the_single_frame_rule():
    for preset in ("Bilinear", CATMULL_ROM):
        decision = summary(frames(preset, 1))["kernel_decision"]
        assert decision == {
            "status": "supported_model",
            "band": [preset],
            "carried": preset,
            "rule": "single_frame",
            "frames": 1,
        }


def test_a_single_tested_preset_is_not_identified():
    decision = summary(frames("Bilinear", 7), kernels=1)["kernel_decision"]
    assert (decision["status"], decision["rule"]) == ("unresolved", "single_preset")


def test_frames_off_the_consensus_grid_are_left_out():
    other = planted("Bilinear", 8, native=(900, 506))
    result = summary([*frames("Bilinear", 7), other])
    assert result["kernel_decision"]["frames"] == 7
    assert result["kernel_decision"]["status"] == "supported_model"


def test_an_inconclusive_frame_has_no_evidence():
    session = gn.Session(gn.SearchOptions(500, 600, threads=8))
    session.add_frame(
        np.full(OBSERVED[::-1], 0.25), gn.FrameFacts("", "Still image", "flat")
    )
    report = json.loads(session.report(0))
    evidence = report["kernel_evidence"]
    assert evidence["measured"] is False
    assert evidence["reason"] == report["verdict"]
    decision = json.loads(session.summary(1, 1, 1, None))["kernel_decision"]
    assert decision["status"] == "not_measured"


def test_the_free_bicubic_is_never_carried():
    # Mitchell with Kernels = Bicubic: Catmull-Rom is the one preset tried and the free
    # bicubic fits far better, so the frame's own recommendation is the free bicubic.
    session = gn.Session(gn.SearchOptions(500, 600, threads=8, kernels=2))
    session.add_frame(planted("Mitchell", 1), gn.FrameFacts("", "Still image", "plane"))
    selection = json.loads(session.summary(1, 1, 1, None))["selection"]
    assert selection["kernel"] == CATMULL_ROM
    assert selection["rule"] == (
        "the best-scoring preset (no preset is within 8 % of the free bicubic)"
    )
    free = selection["free_bicubic"]
    assert free["advantage"] > 0.08
    # Candidate (1-based) carries the free bicubic instead
    parameters = json.loads(session.parameters(0, free["rank"] - 1))
    assert parameters["kernel"]["family"] == "bicubic"
    assert parameters["kernel"]["params"] == pytest.approx(
        {"b": free["b"], "c": free["c"]}
    )
    assert free["b"] == pytest.approx(1 / 3, abs=0.1)
    assert free["c"] == pytest.approx(1 / 3, abs=0.1)


def test_the_node_says_the_free_bicubic_fits_better(tmp_path):
    rgb = np.dstack([planted("Mitchell", 1)] * 3).astype(np.float32)
    outputs = run(
        Context(tmp_path), image=rgb, kernels=node.NativeResolutionKernels.BICUBIC
    )
    width, height, b, c, *_ = outputs
    kernel, status = outputs[9], outputs[11]
    assert (width, height, kernel, b, c) == (960, 540, CATMULL_ROM, 0.0, 0.5)
    free = json.loads(outputs[12])["selection"]["free_bicubic"]
    assert (
        f"The free bicubic (b {free['b']:.3f}, c {free['c']:.3f}) fits"
        f" {free['advantage']:.1%} better than {CATMULL_ROM}: Candidate {free['rank']}"
        " carries it." in status
    )


def test_the_node_carries_the_decided_kernel(tmp_path):
    rgb = np.dstack([planted(CATMULL_ROM, seed) for seed in (1, 2, 3)]).astype(
        np.float32
    )
    width, height, b, c, _, _, _, _, _, kernel, _, status, *_ = run(
        Context(tmp_path), image=rgb
    )
    assert (width, height, kernel, b, c) == (960, 540, CATMULL_ROM, 0.0, 0.5)
    assert "Kernel: kernel decision supported_model (single_frame rule)." in status
    assert (
        "Kernel decision: supported_model by the single_frame rule over 1 frame(s)"
        in status
    )
