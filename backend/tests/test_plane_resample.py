"""resample_plane (nodes/impl/plane_resample.py), checked against zimg (VapourSynth's
resize). The zimg cases need vapoursynth (run this file in the VapourSynth dev venv);
elsewhere they are skipped."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from nodes.impl.plane_resample import KERNEL_LABELS, Kernel, resample_plane

# (input width, input height, resample_plane's keyword arguments)
CASES: dict[str, tuple[int, int, dict[str, Any]]] = {
    "up": (37, 23, {"width": 80, "height": 50}),
    "down": (37, 23, {"width": 15, "height": 9}),
    "down_far": (64, 61, {"width": 5, "height": 4}),
    "tiny_up": (3, 2, {"width": 16, "height": 11}),
    "shifted": (37, 23, {"width": 37, "height": 23, "src_left": 0.25, "src_top": -0.5}),
    "chroma_left": (19, 12, {"width": 37, "height": 23, "src_left": 0.25}),
    "windowed": (
        37,
        23,
        {
            "width": 41,
            "height": 30,
            "src_left": 3.3,
            "src_top": 1.7,
            "src_width": 20.5,
            "src_height": 12.25,
        },
    ),
    "windowed_down": (
        40,
        30,
        {"width": 9, "height": 7, "src_left": -2.5, "src_width": 44.0},
    ),
    "one_axis": (37, 23, {"width": 52, "height": 23}),
}


def random_plane(width: int, height: int) -> np.ndarray:
    # Values well outside 0-1: nothing may clip.
    rng = np.random.default_rng(width * 1000 + height)
    return (rng.standard_normal((height, width)) * 2 + 0.5).astype(np.float32)


def zimg_resize(plane: np.ndarray, kernel: Kernel, **args: Any) -> np.ndarray:
    vs = pytest.importorskip("vapoursynth")
    h, w = plane.shape
    blank = vs.core.std.BlankClip(format=vs.GRAYS, width=w, height=h, length=1)

    def fill(n: int, f: Any) -> Any:
        frame = f.copy()
        np.asarray(frame[0])[:] = plane
        return frame

    clip = blank.std.ModifyFrame(blank, fill)
    resized = getattr(vs.core.resize, kernel.name.title())(clip, **args)
    return np.asarray(resized.get_frame(0)[0]).copy()


@pytest.mark.parametrize("kernel", list(Kernel))
@pytest.mark.parametrize("case", list(CASES))
def test_matches_zimg(kernel: Kernel, case: str):
    w, h, args = CASES[case]
    plane = random_plane(w, h)
    expected = zimg_resize(plane, kernel, **args)
    actual = resample_plane(plane, kernel=kernel, **args)
    assert actual.dtype == np.float32
    assert actual.shape == expected.shape == (args["height"], args["width"])
    assert np.abs(actual - expected).max() <= 1e-5


@pytest.mark.parametrize("kernel", list(Kernel))
def test_constant_planes_stay_constant(kernel: Kernel):
    plane = np.full((23, 37), -2.5, np.float32)
    for size in [(80, 50), (11, 7)]:
        out = resample_plane(plane, *size, kernel=kernel, src_left=0.3)
        assert np.allclose(out, -2.5, rtol=0, atol=1e-5)


def test_unchanged_size_is_a_copy():
    plane = random_plane(37, 23)
    out = resample_plane(plane, 37, 23, Kernel.BICUBIC)
    assert out is not plane and np.array_equal(out, plane)


def test_every_kernel_has_a_label():
    assert set(KERNEL_LABELS) == set(Kernel)


def test_invalid_arguments():
    plane = random_plane(8, 8)
    with pytest.raises(ValueError, match="2 dimensions"):
        resample_plane(plane[..., None], 4, 4)
    with pytest.raises(ValueError, match="0x4"):
        resample_plane(plane, 0, 4)
    with pytest.raises(ValueError, match="empty"):
        resample_plane(plane, 4, 4, src_width=0)
