"""Get Native Resolution (the node and nodes/impl/getnative_media.py): Vernier's frame cap and
sampling on generated clips, the chain to Descale on a planted upscale, abort during analysis and
during scene discovery, the automatic search range, the Export Folder and permuted layouts."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from test_getnative import forward, natively_sharp

from api import Aborted, NodeContext, NodeId, SettingsParser
from nodes.impl import getnative_media as media
from nodes.impl import video
from nodes.impl.ffmpeg import FFMpegEnv
from nodes.impl.linear_light import LightTransfer
from packages.chaiNNer_standard.image_dimension.resize.descale import (
    DescaleBorder,
    DescaleKernel,
    descale_node,
)
from packages.chaiNNer_standard.image_dimension.utility import (
    get_native_resolution as node,
)

FFMPEG, FFPROBE = shutil.which("ffmpeg"), shutil.which("ffprobe")
needs_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None, reason="needs ffmpeg and ffprobe on PATH"
)
S = node.NativeResolutionSource
M = node.NativeResolutionSampling


class Context(NodeContext):
    """The node context the node reads: abort (after `abort_after` seconds), progress and the
    storage folder. Every other member raises AttributeError, as on a context without it."""

    def __init__(self, storage: Path, abort_after: float | None = None):
        self._storage = storage
        self.abort_after = abort_after
        self.start = time.monotonic()

    @property
    def aborted(self) -> bool:
        return (
            self.abort_after is not None
            and time.monotonic() - self.start > self.abort_after
        )

    @property
    def paused(self) -> bool:
        return False

    def set_progress(self, progress: float) -> None:
        self.check_aborted()

    @property
    def storage_dir(self) -> Path:
        return self._storage

    @property
    def settings(self) -> SettingsParser:
        raise AttributeError("settings")

    @property
    def node_id(self) -> NodeId:
        raise AttributeError("node_id")

    def add_cleanup(
        self, fn: Callable[[], None], after: Literal["node", "chain"] = "chain"
    ) -> None:
        raise AttributeError("add_cleanup")


def path_tools() -> FFMpegEnv:
    assert FFMPEG is not None and FFPROBE is not None
    return FFMpegEnv(FFMPEG, FFPROBE)


def no_tools() -> FFMpegEnv:
    raise AssertionError("FFmpeg's executables are not needed here")


def run(
    context: Context,
    *,
    source: node.NativeResolutionSource = S.IMAGE,
    image: np.ndarray | None = None,
    video_path: Path | None = None,
    sampling: node.NativeResolutionSampling = M.KEYFRAMES,
    max_frames: int = 50,
    gamma24: bool = False,
    srgb: bool = False,
    kernels: node.NativeResolutionKernels = node.NativeResolutionKernels.ALL,
    export_folder: Path | None = None,
):
    """The node with the given inputs and these defaults (Workers 8)."""
    return node.get_native_resolution_node(
        context,
        source=source,
        image=image,
        video_path=video_path,
        sampling=sampling,
        max_frames=max_frames,
        scene_threshold=0.3,
        minimum_shot=0.5,
        times="",
        gamma24=gamma24,
        srgb=srgb,
        bt709=False,
        gamma22=False,
        min_height=0,
        max_height=0,
        min_width=0,
        max_width=0,
        kernels=kernels,
        lines=24,
        refinement_lines=128,
        refine=True,
        workers=8,
        candidate=0,
        export_folder=export_folder,
    )


@pytest.mark.parametrize(
    "count, cap, expected",
    [
        (3, 50, [0, 1, 2]),
        (10, 1, [5]),
        (100, 4, [0, 33, 66, 99]),
        (7, 7, list(range(7))),
        (0, 5, []),
    ],
)
def test_capped_indices(count, cap, expected):
    assert media.capped_indices(count, cap) == expected


@pytest.mark.parametrize("cap", [0, 201])
def test_cap_bounds(cap):
    with pytest.raises(ValueError, match="between 1 and 200"):
        media.capped_indices(10, cap)


def test_parse_times():
    assert media.parse_times("90, 5:00 0:12:30.5;1:00:00") == [90, 300, 750.5, 3600]
    with pytest.raises(ValueError, match="Invalid time"):
        media.parse_times("1:xx")


def clip(path: Path, sources: list[str], seconds: float) -> Path:
    """An x264 clip of the lavfi sources one after another (a cut between each), each
    `seconds` long, a keyframe every second and none at the cuts."""
    assert FFMPEG is not None
    inputs: list[str] = []
    for source in sources:
        inputs += [
            "-f",
            "lavfi",
            "-i",
            f"{source}=size=640x360:rate=24:duration={seconds}",
        ]
    joined = "".join(f"[{k}:v]" for k in range(len(sources)))
    subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *inputs,
            "-filter_complex",
            f"{joined}concat=n={len(sources)}:v=1[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-g",
            "24",
            "-keyint_min",
            "24",
            "-sc_threshold",
            "0",
            str(path),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    return path


@pytest.fixture(scope="module")
def cuts(tmp_path_factory) -> Path:
    return clip(
        tmp_path_factory.mktemp("clips") / "cuts.mp4",
        ["testsrc2", "smptebars", "rgbtestsrc"],
        2,
    )


@pytest.fixture(scope="module")
def three_keyframes(tmp_path_factory) -> Path:
    return clip(tmp_path_factory.mktemp("clips") / "three.mp4", ["testsrc2"], 3)


@pytest.fixture(scope="module")
def longer_audio(tmp_path_factory) -> Path:
    """10 s of video with 11 s of AAC in Matroska, which records no duration per track:
    the container's duration ends with the audio."""
    assert FFMPEG is not None
    path = tmp_path_factory.mktemp("clips") / "longer_audio.mkv"
    subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=24:duration=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=11",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    return path


@needs_ffmpeg
def test_even_samples_span_the_video_track_not_a_longer_audio_track(longer_audio):
    info = media.probe(longer_audio)
    assert info.duration == pytest.approx(10.0, abs=0.05)
    plan = media.discover(
        longer_audio, info, "even", 50, [], 0.3, 0.5, no_tools, lambda: None
    )
    assert len(plan.candidates) == 50
    for point in plan.candidates:
        media.decode_frame(longer_audio, info, point, no_tools, lambda: None)


@needs_ffmpeg
def test_frames_that_cannot_be_decoded_are_skipped_logged_and_counted(
    tmp_path, longer_audio, monkeypatch, caplog
):
    decode = media.decode_frame

    def failing(path, info, point, *args):
        if point.seconds > 7:
            raise ValueError("Decoder returned no complete frame at this timestamp")
        return decode(path, info, point, *args)

    monkeypatch.setattr(media, "decode_frame", failing)
    monkeypatch.setattr(node.FFMpegEnv, "get_integrated", lambda _: no_tools())
    with caplog.at_level(logging.ERROR):
        outputs = run(
            Context(tmp_path),
            source=S.VIDEO,
            video_path=longer_audio,
            sampling=M.EVEN,
            max_frames=10,
        )

    # Even bins at the middle of each tenth of the video track: the three past 7 s fail
    duration = media.probe(longer_audio).duration
    failed = [f"{duration * (k + 0.5) / 10:.3f} s" for k in (7, 8, 9)]
    summary = json.loads(outputs[12])
    assert summary["planned_frames"] == 10
    assert summary["completed_frames_all_domains"] == summary["analyzed_frames"] == 7
    assert summary["partial"] is True
    message = "ValueError: Decoder returned no complete frame at this timestamp"
    assert summary["error"] == "3 of 10 frames could not be decoded: " + "; ".join(
        f"{seconds}: {message}" for seconds in failed
    )
    assert "3 of 10 frames could not be decoded" in outputs[11]
    skipped = [r for r in caplog.records if "could not be decoded" in r.message]
    assert len(skipped) == 3 and all(r.exc_info for r in skipped)


@needs_ffmpeg
def test_a_video_whose_frames_all_fail_to_decode_is_an_error(
    tmp_path, longer_audio, monkeypatch
):
    def broken(*_args):
        raise ValueError("Decoder returned no complete frame at this timestamp")

    monkeypatch.setattr(media, "decode_frame", broken)
    with pytest.raises(ValueError, match="None of the 4 frames could be decoded"):
        run(
            Context(tmp_path),
            source=S.VIDEO,
            video_path=longer_audio,
            sampling=M.EVEN,
            max_frames=4,
        )


@needs_ffmpeg
def test_keyframes_are_discovered(three_keyframes):
    info = media.probe(three_keyframes)
    plan = media.discover(
        three_keyframes, info, "keyframes", 50, [], 0.3, 0.5, no_tools, lambda: None
    )
    assert [p.seconds for p in plan.candidates] == pytest.approx([0, 1, 2])
    assert [p.frame for p in plan.candidates] == [0, 24, 48]


def unindexed(_path, _check=None):
    raise video._UseCliReader("a frame's packet has no timestamp")


def without_keyframes(path, check=None):
    return video.FrameIndex(video.frame_index(path, check).pts, ())


@needs_ffmpeg
@pytest.mark.parametrize("pyav", [unindexed, without_keyframes])
def test_keyframes_pyav_cannot_list_come_from_ffprobe(
    three_keyframes, monkeypatch, pyav
):
    info = media.probe(three_keyframes)
    monkeypatch.setattr(media, "frame_index", pyav)
    plan = media.discover(
        three_keyframes, info, "keyframes", 50, [], 0.3, 0.5, path_tools, lambda: None
    )
    assert [p.seconds for p in plan.candidates] == pytest.approx([0, 1, 2])
    assert {(p.kind, p.frame) for p in plan.candidates} == {(media.KEYFRAME, None)}


@needs_ffmpeg
def test_keyframes_from_ffprobe_are_analyzed(tmp_path, three_keyframes, monkeypatch):
    monkeypatch.setattr(media, "frame_index", unindexed)
    monkeypatch.setattr(node.FFMpegEnv, "get_integrated", lambda _: path_tools())
    summary = json.loads(
        run(Context(tmp_path), source=S.VIDEO, video_path=three_keyframes)[12]
    )
    assert summary["analyzed_frames"] == 3 and summary["partial"] is False
    assert [a["requested_seconds"] for a in summary["analyses"]] == pytest.approx(
        [0, 1, 2]
    )


@needs_ffmpeg
def test_keyframes_neither_pyav_nor_ffprobe_can_list_are_a_clear_error(
    tmp_path, three_keyframes, monkeypatch
):
    info = media.probe(three_keyframes)
    monkeypatch.setattr(media, "frame_index", unindexed)

    def broken_tools() -> FFMpegEnv:
        assert FFMPEG is not None
        return FFMpegEnv(FFMPEG, str(tmp_path / "missing" / "ffprobe.exe"))

    with pytest.raises(ValueError, match="Use Even or Scenes sampling") as raised:
        media.discover(
            three_keyframes,
            info,
            "keyframes",
            50,
            [],
            0.3,
            0.5,
            broken_tools,
            lambda: None,
        )
    assert "could not be listed" in str(raised.value)
    assert isinstance(raised.value.__cause__, OSError)


@needs_ffmpeg
def test_scene_midpoints_avoid_cuts(cuts):
    info = media.probe(cuts)
    plan = media.discover(
        cuts, info, "scenes", 50, [], 0.3, 0.5, path_tools, lambda: None
    )
    assert [p.seconds for p in plan.candidates] == pytest.approx([1, 3, 5], abs=0.05)
    for point in plan.candidates:
        assert point.scene_end - point.scene_start >= 0.5
        assert min(abs(point.seconds - cut) for cut in (2, 4)) > 0.5


@needs_ffmpeg
def test_fewer_candidates_than_the_cap(tmp_path, three_keyframes, monkeypatch):
    monkeypatch.setattr(node.FFMpegEnv, "get_integrated", lambda _: no_tools())
    *_, report, _, _, descaled = run(
        Context(tmp_path), source=S.VIDEO, video_path=three_keyframes
    )
    summary = json.loads(report)
    assert summary["planned_frames"] == summary["analyzed_frames"] == 3
    assert [a["requested_seconds"] for a in summary["analyses"]] == pytest.approx(
        [0, 1, 2]
    )
    assert summary["sampling"]["discovered"] == 3
    assert descaled is None


@needs_ffmpeg
def test_cli_fallback_reads_the_same_planes(three_keyframes, monkeypatch):
    info = media.probe(three_keyframes)
    points = [
        media.SamplePoint(1.0, media.KEYFRAME, frame=24),
        media.SamplePoint(1.5, media.REQUESTED),
    ]
    decoded = [
        media.decode_frame(three_keyframes, info, p, no_tools, lambda: None)[0]
        for p in points
    ]

    def unindexed(_path, _check=None):
        raise video._UseCliReader("test")

    monkeypatch.setattr(media, "frame_index", unindexed)
    for point, plane in zip(points, decoded, strict=True):
        cli, provenance, rgb = media.decode_frame(
            three_keyframes, info, point, path_tools, lambda: None
        )
        assert provenance.startswith("First plane of FFmpeg native yuv420p output")
        assert rgb is None
        assert np.array_equal(cli, plane)


@needs_ffmpeg
def test_video_light_models(tmp_path, three_keyframes, monkeypatch):
    monkeypatch.setattr(node.FFMpegEnv, "get_integrated", lambda _: path_tools())
    outputs = run(
        Context(tmp_path), source=S.VIDEO, video_path=three_keyframes, gamma24=True
    )
    summary = json.loads(outputs[12])
    domains = [a["light_domain"] for a in summary["analyses"]]
    assert domains == ["encoded", "linear-gamma24"] * 3
    assert summary["requested_domains_per_frame"] == 2
    assert summary["paired_grid_consensus"]
    assert "Gamma 2.4 model:" in outputs[11]
    assert summary["selection"] is None or (
        summary["analyses"][summary["selection"]["analysis_id"]]["light_domain"]
        == "encoded"
    )


def many_packets(path: Path, frames: int) -> Path:
    """A clip of `frames` small MPEG-4 frames written by PyAV, one packet each."""
    av = video._pyav()
    assert av is not None
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=24)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        image = np.zeros((64, 64, 3), np.uint8)
        for n in range(frames):
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = n
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return path


def test_stop_ends_the_packet_index_scan(tmp_path):
    """Reading one frame (a typed time) indexes the file's packets first, about 7 s for
    a 1.4 GB file: Stop ends the scan at its next check, and the cut-short scan is not
    kept."""
    path = many_packets(tmp_path / "long.mp4", 3000)
    info = media.probe(path)
    checks = 0

    def stop() -> None:
        nonlocal checks
        checks += 1
        raise Aborted()

    with pytest.raises(Aborted):
        media.decode_frame(
            path, info, media.SamplePoint(60.0, media.REQUESTED), no_tools, stop
        )
    # the first check, after 256 of the 3000 packets, ended it
    assert checks == 1
    assert len(video.frame_index(path).pts) == 3000


def rotated(path: Path, degrees: int) -> Path:
    """`path`'s stream copied (not re-encoded) with a display matrix of `degrees`."""
    assert FFMPEG is not None
    copy = path.with_name(f"{path.stem}-rotated{degrees}.mp4")
    subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-display_rotation",
            str(degrees),
            "-i",
            str(path),
            "-c",
            "copy",
            str(copy),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    return copy


@needs_ffmpeg
@pytest.mark.parametrize("degrees", [90, 180, 270])
def test_light_models_read_a_rotated_video_as_stored(three_keyframes, degrees):
    """The display matrix only says how to show the frames: the light models' RGB is the
    stored raster the As stored plane comes from (a quarter turn was refused, a half
    turn came upside down)."""
    point = media.SamplePoint(1.0, media.KEYFRAME, frame=24)
    decoded = []
    for path in (three_keyframes, rotated(three_keyframes, degrees)):
        info = media.probe(path)
        colour = media.colour_model(path, info, path_tools, lambda: None)
        decoded.append(
            media.decode_frame(path, info, point, path_tools, lambda: None, colour.rgb)
        )
    (plane, _, rgb), (rotated_plane, _, rotated_rgb) = decoded
    assert rgb is not None and rotated_rgb is not None
    assert np.array_equal(rotated_plane, plane)
    assert np.array_equal(rotated_rgb, rgb)


@needs_ffmpeg
def test_video_light_models_on_a_rotated_video(tmp_path, three_keyframes, monkeypatch):
    monkeypatch.setattr(node.FFMpegEnv, "get_integrated", lambda _: path_tools())
    outputs = run(
        Context(tmp_path),
        source=S.VIDEO,
        video_path=rotated(three_keyframes, 90),
        gamma24=True,
    )
    summary = json.loads(outputs[12])
    domains = [a["light_domain"] for a in summary["analyses"]]
    assert domains == ["encoded", "linear-gamma24"] * 3


@needs_ffmpeg
def test_abort_during_scene_discovery_stops_ffmpeg(monkeypatch):
    assert FFMPEG is not None
    started: list[subprocess.Popen] = []
    popen = subprocess.Popen

    def recording(*args, **kwargs):
        started.append(popen(*args, **kwargs))
        return started[-1]

    monkeypatch.setattr(media.subprocess, "Popen", recording)
    context = Context(Path("."), abort_after=0.5)
    start = time.monotonic()
    with pytest.raises(Aborted):
        media.run_ffmpeg(
            [
                FFMPEG,
                "-hide_banner",
                "-nostdin",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=1920x1080:duration=3600",
                "-vf",
                "select='gt(scene,0.3)'",
                "-f",
                "null",
                "-",
            ],
            context.check_aborted,
            600,
        )
    assert time.monotonic() - start < 2
    assert started and started[0].poll() is not None


def planted_image() -> tuple[np.ndarray, np.ndarray]:
    """A 1280x720 natively sharp RGB source and its bilinear upscale to 1920x1080."""
    source = np.dstack([natively_sharp(1280, 720, seed) for seed in (1, 2, 3)])
    upscaled = forward(forward(source, 1920, 1), 1080, 0)
    return source.astype(np.float32), upscaled.astype(np.float32)


def test_chain_to_descale_reproduces_the_source(tmp_path):
    source, upscaled = planted_image()
    (
        width,
        height,
        b,
        c,
        taps,
        left,
        top,
        src_w,
        src_h,
        kernel,
        light,
        status,
        _,
        height_plot,
        width_plot,
        descaled,
    ) = run(Context(tmp_path), image=upscaled)
    assert (width, height, kernel, light) == (1280, 720, "Bilinear", "As stored")
    assert status.startswith("SUPPORTED CANDIDATE: 1280x720, Bilinear.")
    assert height_plot.shape == width_plot.shape == (256, 1024, 3)
    chained = descale_node(
        upscaled,
        width,
        height,
        DescaleKernel.BILINEAR,
        b,
        c,
        taps,
        1.0,
        left,
        top,
        src_w,
        src_h,
        DescaleBorder.EXTEND,
        False,
        LightTransfer.GAMMA_24,
    )
    assert descaled is not None
    assert np.array_equal(chained, descaled)
    assert np.abs(chained - source)[8:-8, 8:-8].max() <= 1e-3


def test_permuted_layout_gives_the_same_result(tmp_path):
    _, upscaled = planted_image()
    permuted = np.ascontiguousarray(upscaled.transpose(2, 0, 1)).transpose(1, 2, 0)
    c_layout = run(Context(tmp_path), image=upscaled)
    other = run(Context(tmp_path), image=permuted)
    assert c_layout[:12] == other[:12]
    for a, b in zip(c_layout[13:], other[13:], strict=True):
        assert a is not None and b is not None
        assert np.array_equal(a, b)


def test_abort_mid_analysis(tmp_path):
    _, upscaled = planted_image()
    start = time.monotonic()
    with pytest.raises(Aborted):
        run(Context(tmp_path, abort_after=0.3), image=upscaled)
    assert time.monotonic() - start < 2.3


@pytest.mark.parametrize(
    "height, expected", [(1080, (500, 1000)), (720, (333, 667)), (576, (267, 533))]
)
def test_automatic_range_scales(height, expected):
    assert node.search_bounds(height, 0, 0) == expected


def test_bounds_below_16_are_refused(tmp_path):
    image = np.random.default_rng(1).random((30, 64, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="start at 16 or more"):
        run(Context(tmp_path), image=image)


def test_export_folder(tmp_path):
    _, upscaled = planted_image()
    run(Context(tmp_path), image=upscaled, export_folder=tmp_path)
    (folder,) = tmp_path.glob("Vernier-*")
    assert sorted(p.name for p in folder.iterdir()) == [
        "analysis.json",
        "curves.csv",
        "parameters.json",
        "summary.json",
    ]
    parameters = json.loads((folder / "parameters.json").read_text(encoding="utf-8"))
    assert parameters["schema"] == "vernier.parameters.v3"
    assert parameters["native"] == {"width": 1280, "height": 720}


def test_linear_models_add_evidence_only(tmp_path):
    _, upscaled = planted_image()
    outputs = run(Context(tmp_path), image=upscaled, srgb=True)
    summary = json.loads(outputs[12])
    assert {a["light_domain"] for a in summary["analyses"]} == {
        "encoded",
        "linear-srgb",
    }
    assert outputs[:2] == (1280, 720)
    assert "sRGB model:" in outputs[11]
