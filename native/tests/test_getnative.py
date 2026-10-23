"""Get Native Resolution's analysis core (native/src/getnative_ops.cpp over the vendored Vernier 0.2
core): the same plane gives Vernier 0.2's nativeprobe.exe verdicts, grids, kernels and scores, also
a video frame decoded by getnative_media.py where nativeprobe decodes it itself; the baseline and
AVX2 units give identical JSON; a cancel stops an analysis promptly."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from test_isa_dispatch import isa_get, isa_set

from nodes.impl import getnative_media as media
from nodes.impl import native_getnative as gn
from nodes.impl.ffmpeg import FFMpegEnv

# Vernier 0.2's command-line probe, the oracle (an FMA/PGO build, so scores agree to 1e-6): the
# nativeprobe.exe of its release folder, named by CHAINNER_C_NATIVEPROBE; without it the parity cases skip.
NATIVEPROBE = Path(os.environ.get("CHAINNER_C_NATIVEPROBE", "nativeprobe-not-set"))
WORKERS = 8
# A 1080p BD remux episode (read only) whose stored frames carry a known native grid, named by
# CHAINNER_C_GETNATIVE_EPISODE; without it the episode case skips.
EPISODE = Path(os.environ.get("CHAINNER_C_GETNATIVE_EPISODE", "episode-not-set"))
FFMPEG, FFPROBE = shutil.which("ffmpeg"), shutil.which("ffprobe")


def kernel_bilinear(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, 1 - np.abs(x))


def forward(
    plane: np.ndarray, size: int, axis: int, extent: float = 0, shift: float = 0
):
    """Vernier's Resampler::forward along `axis`: native samples onto `size` observed pixel
    centres, bilinear, clamped border, weights normalized per row."""
    n = plane.shape[axis]
    extent = extent or n
    matrix = np.zeros((size, n))
    for j in range(size):
        pos = (j + 0.5) * extent / size - 0.5 + (n - extent) * 0.5 + shift
        first = int(np.floor(pos))
        taps = np.arange(first, first + 2)
        weights = kernel_bilinear(pos - taps)
        np.add.at(matrix[j], np.clip(taps, 0, n - 1), weights / weights.sum())
    return np.moveaxis(np.tensordot(matrix, np.moveaxis(plane, axis, 0), 1), 0, axis)


def natively_sharp(width: int, height: int, seed: int = 1) -> np.ndarray:
    """Hard-edged rectangles: a source with detail at its own sampling rate."""
    rng = np.random.default_rng(seed)
    plane = np.full((height, width), 0.5)
    for _ in range(400):
        x, y = rng.integers(0, width), rng.integers(0, height)
        w, h = rng.integers(4, 120, size=2)
        plane[y : y + h, x : x + w] = rng.random()
    return plane


def planted(
    native: tuple[int, int],
    observed: tuple[int, int],
    extent: tuple[float, float] = (0, 0),
    shift: tuple[float, float] = (0, 0),
) -> np.ndarray:
    """A bilinear upscale of a natively sharp plane, as float32 values (what a PFM holds)."""
    up = forward(natively_sharp(*native), observed[0], 1, extent[0], shift[0])
    up = forward(up, observed[1], 0, extent[1], shift[1])
    return up.astype(np.float32).astype(np.float64)


CASES = {
    "planted-720p": lambda: planted((1280, 720), (1920, 1080)),
    "fractional-shift": lambda: planted(
        (1280, 720), (1920, 1080), (1279.5, 719.5), (0.25, -0.2)
    ),
    "flat": lambda: np.full((1080, 1920), 0.25),
}


def write_pfm(path: Path, plane: np.ndarray) -> None:
    """Little-endian grayscale PFM, rows bottom to top."""
    height, width = plane.shape
    with path.open("wb") as file:
        file.write(f"Pf\n{width} {height}\n-1.0\n".encode())
        file.write(np.ascontiguousarray(plane[::-1], "<f4").tobytes())


def analyze(
    plane: np.ndarray, min_height: int = 500, max_height: int = 1000
) -> tuple[gn.Session, dict]:
    session = gn.Session(gn.SearchOptions(min_height, max_height, threads=WORKERS))
    session.add_frame(plane, gn.FrameFacts("", "Still image", "test plane"))
    return session, json.loads(session.report(0))


def oracle(tmp_path: Path, plane: np.ndarray) -> dict:
    source, out = tmp_path / "plane.pfm", tmp_path / "oracle.json"
    write_pfm(source, plane)
    subprocess.run(
        [NATIVEPROBE, source, "--cpu", "all", "--threads", str(WORKERS), "--out", out],
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
    )
    return json.loads(out.read_text(encoding="utf-8"))


def assert_same_analysis(ours: dict, theirs: dict) -> None:
    for key in (
        "verdict",
        "grid_status",
        "kernel_status",
        "recommended_rank",
        "competing_grids",
        "kernel_axes_and_splits_agree",
    ):
        assert ours[key] == theirs[key], key
    assert len(ours["candidates"]) == len(theirs["candidates"])
    for a, b in zip(ours["candidates"], theirs["candidates"], strict=True):
        for axis in ("x", "y"):
            assert a[axis]["samples"] == b[axis]["samples"]
            assert a[axis]["kernel"] == b[axis]["kernel"]
            assert (
                a[axis]["fractional_supported_heuristically"]
                == (b[axis]["fractional_supported_heuristically"])
            )
        assert a["continuous_approximation"] == b["continuous_approximation"]
        assert a["score"] == pytest.approx(b["score"], rel=1e-6, abs=1e-30)
    if not ours["candidates"]:
        assert ours["parameters"] == theirs["parameters"] == {}
        return
    assert ours["parameters"]["native"] == theirs["parameters"]["native"]
    for key in ("extent_x", "extent_y", "offset_x", "offset_y"):
        assert ours["parameters"]["geometry"][key] == pytest.approx(
            theirs["parameters"]["geometry"][key], rel=1e-6, abs=1e-9
        )


@pytest.mark.skipif(
    not NATIVEPROBE.is_file(), reason="needs Vernier 0.2's nativeprobe.exe"
)
@pytest.mark.parametrize("case", CASES)
def test_matches_nativeprobe(tmp_path, case):
    plane = CASES[case]()
    _, ours = analyze(plane)
    assert_same_analysis(ours, oracle(tmp_path, plane))


@pytest.mark.skipif(
    not (NATIVEPROBE.is_file() and EPISODE.is_file() and FFMPEG and FFPROBE),
    reason="needs nativeprobe.exe, the episode and ffmpeg/ffprobe on PATH",
)
def test_matches_nativeprobe_on_a_video_frame(tmp_path):
    assert FFMPEG and FFPROBE
    info = media.probe(EPISODE)
    point = media.SamplePoint(300.0, media.REQUESTED)
    tools = FFMpegEnv(FFMPEG, FFPROBE)
    plane, _, _ = media.decode_frame(EPISODE, info, point, lambda: tools, no_abort)
    _, ours = analyze(plane)
    out = tmp_path / "oracle.json"
    subprocess.run(
        [
            NATIVEPROBE,
            EPISODE,
            "--time",
            "300",
            "--ffmpeg",
            FFMPEG,
            "--ffprobe",
            FFPROBE,
            "--cpu",
            "all",
            "--threads",
            str(WORKERS),
            "--out",
            out,
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
    )
    assert_same_analysis(ours, json.loads(out.read_text(encoding="utf-8")))


def no_abort() -> None:
    pass


def test_planted_upscale_found():
    session, report = analyze(CASES["planted-720p"]())
    assert report["verdict"] == "Supported candidate"
    assert report["parameters"]["native"] == {"width": 1280, "height": 720}
    summary = json.loads(session.summary(1, 1, 1, None))
    assert summary["selection"]["kernel"] == "Bilinear"
    assert summary["selection"]["consensus_grid"] == [1280, 720]


def scrub(text: str) -> str:
    """A report without its run's timestamp and timings."""
    report = json.loads(text)
    for key in ("created_utc", "analysis_seconds"):
        report.pop(key)
    return json.dumps(report, sort_keys=True)


@pytest.mark.skipif(isa_get()[2] < 1, reason="the CPU has no AVX2 level")
def test_isa_units_identical():
    plane = CASES["fractional-shift"]()
    level = isa_get()[0]
    try:
        isa_set(0)
        base = scrub(analyze(plane)[0].report(0))
        isa_set(1)
        avx2 = scrub(analyze(plane)[0].report(0))
    finally:
        isa_set(level)
    assert base == avx2


def test_frame_facts_reach_the_report():
    # The facts cross the C interface as cn_getnative_frame: each lands in its field.
    session = gn.Session(gn.SearchOptions(250, 500, threads=WORKERS))
    facts = gn.FrameFacts(
        "clip.mkv",
        "Codec keyframe · 1.250 s",
        "plane provenance",
        "linear-srgb",
        "assumed BT.709",
        1.25,
        (0.299, 0.587, 0.114),
        True,
    )
    session.add_frame(planted((640, 360), (960, 540)), facts)
    report = json.loads(session.report(0))
    weights = json.loads(session.parameters(0))["linear_analysis_weights"]
    assert weights == [0.299, 0.587, 0.114]
    assert {key: report[key] for key in ("source", "sample", "provenance")} == {
        "source": "clip.mkv",
        "sample": "Codec keyframe · 1.250 s",
        "provenance": "plane provenance",
    }
    assert (report["light_domain"], report["color_assumptions"]) == (
        "linear-srgb",
        "assumed BT.709",
    )
    assert report["requested_seconds"] == 1.25
    assert report["keyframes_only_decode"] is True


def test_text_that_is_not_unicode_round_trips():
    # A file name with bytes that are not UTF-8 reaches Python as lone surrogates (Linux,
    # surrogateescape); Windows names can hold them too. The reports keep them as JSON
    # escapes, so the text stays valid UTF-8 and loads back unchanged.
    session = gn.Session(gn.SearchOptions(500, 600, threads=WORKERS))
    source = "/clips/episode\udcff\udc80.mkv"
    session.add_frame(
        np.full((698, 1240), 0.25),
        gn.FrameFacts(source, "Requested time \ud800 1.000 s", "provenance \udfff"),
    )
    report = session.report(0)
    assert "\\udcff\\udc80" in report
    assert json.loads(report)["source"] == source
    summary = json.loads(session.summary(1, 0, 1, None, f"1 of 1: {source}"))
    assert summary["error"] == f"1 of 1: {source}"
    (analysis,) = summary["analyses"]
    assert analysis["sample"] == "Requested time \ud800 1.000 s"


def test_cancel_stops_promptly():
    session = gn.Session(gn.SearchOptions(500, 1000, threads=WORKERS))
    plane = CASES["planted-720p"]()
    threading.Timer(0.3, session.cancel).start()
    start = time.perf_counter()
    with pytest.raises(RuntimeError, match="cancelled"):
        session.add_frame(plane, gn.FrameFacts("", "Still image", "test plane"))
    assert time.perf_counter() - start < 2


def test_bounds_below_16_refused():
    with pytest.raises(ValueError, match="at least 16 pixels"):
        analyze(CASES["planted-720p"](), min_height=8, max_height=1000)
