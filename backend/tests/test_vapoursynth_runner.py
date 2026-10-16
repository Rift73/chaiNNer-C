"""vspipe runs of compiled scripts: clip info, frames as float32 BGR, errors on nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nodes.impl.vapoursynth.runner import (
    ClipInfo,
    VSScriptError,
    clip_info,
    read_frames,
)
from nodes.impl.vapoursynth.script import VideoClip, clip_var, compile_script

# VapourSynth lives in native\local\vapoursynth\venv, not in the runtime; the modules
# above never import it. The references below are computed in this process.
vs = pytest.importorskip("vapoursynth")


def clip(node_id: str, call: str, *deps: VideoClip) -> VideoClip:
    var = clip_var(node_id)
    return VideoClip(
        var=var, stmt=f"{var} = {call}", deps=deps, imports=frozenset(), node_id=node_id
    )


def write(tmp_path: Path, *outputs: VideoClip) -> Path:
    path = tmp_path / "chain.vpy"
    script = compile_script(
        [(o, f"Output {i}") for i, o in enumerate(outputs)], cache_mb=256
    )
    path.write_text(script, encoding="utf-8")
    return path


def planes(frame: Any) -> list[np.ndarray]:
    """The planes of a vapoursynth.VideoFrame."""
    return [np.asarray(frame[i]) for i in range(frame.format.num_planes)]


def test_clip_info_blankclip(tmp_path: Path):
    blank = clip("a", "core.std.BlankClip(width=64, height=48, length=5)")
    gray = clip(
        "b",
        "core.std.BlankClip(width=32, height=16, length=7, format=vs.GRAY16, fpsnum=30000, fpsden=1001)",
    )
    path = write(tmp_path, blank, gray)

    assert clip_info(path, 0) == ClipInfo(
        width=64, height=48, num_frames=5, fps_num=24, fps_den=1, format_name="RGB24"
    )
    assert clip_info(path, 1) == ClipInfo(
        width=32,
        height=16,
        num_frames=7,
        fps_num=30000,
        fps_den=1001,
        format_name="Gray16",
    )


def test_read_frames_matches_numpy(tmp_path: Path):
    blank = clip(
        "a", "core.std.BlankClip(width=16, height=8, length=4, format=vs.RGBS)"
    )
    gradient = clip(
        "b", f'core.akarin.Expr({blank.var}, ["X 16 /", "Y 8 /", "N 4 /"])', blank
    )
    path = write(tmp_path, gradient)

    frames = list(read_frames(path, 0, 1, 4))

    y, x = np.mgrid[0:8, 0:16].astype(np.float32)
    assert len(frames) == 3
    for n, frame in enumerate(frames, 1):
        assert frame.dtype == np.float32
        assert frame.shape == (8, 16, 3)
        assert frame.flags.c_contiguous
        expected = np.stack([np.full_like(x, n / 4), y / 8, x / 16], axis=2)  # B, G, R
        np.testing.assert_array_equal(frame, expected)


def test_read_frames_gray_is_single_channel(tmp_path: Path):
    gray = clip(
        "a",
        "core.std.BlankClip(width=16, height=8, length=2, format=vs.GRAY8, color=[235])",
    )
    path = write(tmp_path, gray)

    [frame] = read_frames(path, 0, 0, 1)

    reference = vs.core.std.BlankClip(
        width=16, height=8, length=2, format=vs.GRAY8, color=[235]
    )
    [expected] = planes(reference.resize.Bicubic(format=vs.GRAYS).get_frame(0))
    assert frame.shape == (8, 16, 1)
    np.testing.assert_array_equal(frame[:, :, 0], expected)


def test_read_frames_yuv_uses_matrix(tmp_path: Path):
    # 48 lines would default to 170m; the _Matrix frame property (709) must win.
    width, height, exprs = 64, 48, ["X 64 /", "Y 48 / 0.5 -", "0.5 X 64 / -"]
    yuv = clip(
        "a",
        f"core.std.SetFrameProps(core.std.BlankClip(width={width}, height={height},"
        f" length=2, format=vs.YUV444PS).akarin.Expr({exprs!r}), _Matrix=1)",
    )
    path = write(tmp_path, yuv)

    [frame] = read_frames(path, 0, 0, 1)

    core = vs.core
    blank = core.std.BlankClip(width=width, height=height, length=2, format=vs.YUV444PS)
    reference = core.std.SetFrameProps(blank.akarin.Expr(exprs), _Matrix=1)
    r, g, b = planes(
        core.resize.Bicubic(reference, format=vs.RGBS, matrix_in=1).get_frame(0)
    )
    np.testing.assert_array_equal(frame, np.stack([b, g, r], axis=2))
    # The test tells the matrices apart: 170m gives other red values.
    no_prop = core.std.RemoveFrameProps(reference, "_Matrix")
    [other_r, _, _] = planes(
        core.resize.Bicubic(no_prop, format=vs.RGBS, matrix_in=6).get_frame(0)
    )
    assert not np.array_equal(frame[:, :, 2], other_r)


def test_vspipe_error_names_node(tmp_path: Path):
    blank = clip("a", "core.std.BlankClip(width=64, height=48, length=3)")
    crop = clip("b", f"core.std.Crop({blank.var}, left=100)", blank)
    path = write(tmp_path, crop)

    with pytest.raises(VSScriptError, match="negative/zero cropping") as error:
        clip_info(path, 0)
    assert error.value.node_id == "b"

    # A failure while frames are rendered (a callback raising) names its node too.
    failing = clip(
        "c",
        f"core.std.ModifyFrame({blank.var}, {blank.var}, lambda n, f: 1 / 0)",
        blank,
    )
    path = write(tmp_path, failing)
    with pytest.raises(VSScriptError, match="ZeroDivisionError") as error:
        list(read_frames(path, 0, 0, 3))
    assert error.value.node_id == "c"
