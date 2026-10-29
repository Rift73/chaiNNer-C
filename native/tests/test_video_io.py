"""Frozen installed video contracts and task-owned CPU FFmpeg fixtures.

Reference AST execution is test-only. Production contains no reference evaluator.
Native cleanup deliberately terminates abandoned owned children, and successful
writer completion/mux is idempotent; comparisons keep these corrections explicit.
Save Video's audio mux compares with INTENDED_MUX instead of the installed nightly
(upstream chaiNNer #3331).
"""

from __future__ import annotations

import ast
import copy
import errno
import gc
import hashlib
import io
import itertools
import json
import math
import os
import runpy
import shutil
import subprocess
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path, PureWindowsPath
from typing import Any, Literal

import ffmpeg
import numpy as np
import pytest

from api import Generator
from nodes.impl import colour_info
from nodes.impl.image_utils import to_uint8, to_uint16, to_uint16_gbr_planes
from nodes.impl.native_framework_images import planes_to_interleaved
from nodes.impl.native_graph import graph
from nodes.impl.yuv_rgb import resolve_colour_info
from nodes.utils.utils import file_identity, get_h_w_c, split_file_path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).with_name("reference_video")
NODE = "packages/chaiNNer_standard/image/video_frames/"
API = runpy.run_path(str(REFERENCE / "installed/api/iter.py"))
MODULES = []


def snap(value):
    if isinstance(value, Enum):
        return (type(value).__name__, value.name)
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes(), value.flags.writeable)
    if isinstance(value, BaseException):
        return (type(value).__name__, str(value))
    if isinstance(value, (list, tuple)):
        return tuple(snap(x) for x in value)
    if isinstance(value, dict):
        return tuple((k, snap(v)) for k, v in value.items())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, FakeStream):
        return ("stream", value.label)
    return value


def outcome(call):
    try:
        return "ok", snap(call())
    except BaseException as error:
        return "error", snap(error)


class Logger:
    def __init__(self, events):
        self.events = events

    def __getattr__(self, name):
        def write(*args, **kwargs):
            self.events.append((name, snap(args), snap(kwargs)))

        return write


class ReadPipe(io.BufferedIOBase):
    def __init__(self, env, chunks):
        self.env, self.chunks = env, list(chunks)

    def read(self, n: int | None = -1):
        self.env.record("read", n)
        if not self.chunks:
            return b""
        value = self.chunks.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    def close(self):
        if not self.closed:
            self.env.record("stdout.close")
        super().close()


class WritePipe(io.BufferedIOBase):
    def __init__(self, env):
        self.env = env

    def write(self, data):
        self.env.record("write", data)
        return len(data)

    def close(self):
        if not self.closed:
            self.env.record("stdin.close")
        super().close()


class FakeProcess:
    def __init__(self, env, reading=False):
        self.env = env
        if not reading:
            self.stdout = None
        elif env.stdout is None:
            self.stdout = ReadPipe(env, env.chunks)
        else:
            self.stdout = env.stdout
        self.stdin = None if reading else WritePipe(env)
        self.stderr = None
        self.returncode = None
        env.processes.append(self)

    def __enter__(self):
        self.env.record("enter")
        return self

    def __exit__(self, kind, value, trace):
        self.env.record("exit", None if kind is None else kind.__name__, snap(value))
        for pipe in (self.stdin, self.stdout, self.stderr):
            if pipe is not None:
                pipe.close()
        self.wait()
        return self.env.suppress

    def poll(self):
        self.env.record("poll")
        return self.returncode

    def terminate(self):
        self.env.record("terminate")
        self.returncode = -15

    def kill(self):
        self.env.record("kill")
        self.returncode = -9

    def wait(self, *args, **kwargs):
        self.env.record("wait", args, kwargs)
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode


class FakeStream:
    def __init__(self, env, label):
        self.env, self.label = env, label
        # ffmpeg.input(path).audio keeps its file in its input node's kwargs.
        self.node = types.SimpleNamespace(kwargs={"filename": label + "-source.mkv"})

    @property
    def audio(self):
        self.env.record("audio", self)
        return FakeStream(self.env, "audio")

    def __str__(self):
        return self.label

    def __getitem__(self, selector):
        assert selector == "v:0"
        return self

    def output(self, *args, **kwargs):
        self.env.record("output", (self, *args), kwargs)
        return FakeStream(self.env, "output")

    def overwrite_output(self):
        self.env.record("overwrite")
        return self

    def global_args(self, *args):
        self.env.record("global_args", args)
        return self

    def run_async(self, **kwargs):
        self.env.record("run_async", kwargs)
        if self.env.process_override is not None:
            return self.env.process_override
        return FakeProcess(self.env, reading=bool(kwargs.get("pipe_stdout")))


class FakeFFmpeg:
    Error = ffmpeg.Error

    def __init__(self, probe=None, chunks=(), failures=None, stdout=None):
        self.events, self.processes = [], []
        self.probe_result = probe if probe is not None else valid_probe()
        self.chunks, self.failures = list(chunks), failures or {}
        # A real pipe reader for the reading process instead of the chunks: readinto
        # is then CPython's BufferedReader.readinto, as on Popen's stdout.
        self.stdout = stdout
        self.suppress = False
        # A real owned Popen for run_async to return instead of a FakeProcess.
        self.process_override: subprocess.Popen[bytes] | None = None
        self.targets = 0

    def record(self, name, *args):
        self.events.append((name, *(snap(a) for a in args)))
        error = self.failures.get(name)
        if error is not None:
            raise error

    def probe(self, *args, **kwargs):
        self.record("probe", args, kwargs)
        return self.probe_result

    def input(self, *args, **kwargs):
        self.record("input", args, kwargs)
        return FakeStream(self, "input")

    def output(self, *args, **kwargs):
        self.record("output", args, kwargs)
        return FakeStream(self, "mux")

    def run(self, *args, **kwargs):
        self.record("run", args, kwargs)


class FakeTarget:
    """Save Video's temporary file with the fakes (m): named after its path and a count,
    its commit and abandon recorded."""

    def __init__(self, env, path):
        self.env, self.temp = env, f"{path}.{env.targets}.tmp"
        env.targets += 1

    def check(self):
        """The real target's check opens the path the rename will replace; here every
        path can be replaced (test_save_video_atomic.py tests a held one)."""

    def commit(self):
        self.env.record("commit", self.temp)
        return True

    def abandon(self):
        self.env.record("abandon", self.temp)


def real_target(path):
    return graph().AtomicTarget(path)


def valid_probe() -> dict[str, Any]:
    return {
        "format": {"duration": "2"},
        "streams": [
            {"codec_type": "audio"},
            {
                "codec_type": "video",
                "width": "2",
                "height": 2,
                "r_frame_rate": "30000/1001",
                "nb_frames": "4",
            },
            {"codec_type": "video", "width": 1000},
        ],
    }


class Context:
    storage_dir = Path("isolated/storage")
    chosen_item = None  # every item runs, so Save Video writes the video
    node_id = "load-video"  # Load Video's Preview reader is the node's

    def __init__(self):
        self.cleanups = []

    def add_cleanup(self, callback, after="chain"):
        assert after == "chain"
        self.cleanups.append(callback)

    def cleanup(self):
        callbacks, self.cleanups = self.cleanups, []
        for callback in callbacks:
            callback()


# Save Video's audio mux departs from both frozen references (approved fix for
# upstream chaiNNer #3331; ARCHITECTURE section 7). The "intended" oracle is the
# frozen source snapshot, upstream v0.25.1's mux, with this block replaced. Against
# upstream: (a) the options come first, so a WebM Copy raises before the mux as in
# the installed nightly; (b) FFmpeg logs errors only, so its stderr is the error; (c)
# Auto transcodes as Transcode does when the container cannot hold a copy; (d) the
# temporary file never stays behind; (e) the command is always chaiNNer's FFmpeg
# (upstream's PATH fallback is dead: Writer requires ffmpeg_env); (f) audio the mux
# cannot carry raises instead of logging, and only a source without audio (probed
# after a failure) saves the video without it; (g) os.replace swaps the files
# atomically, so a failed swap keeps the video; (h) each Opus stream gets min(320k,
# 256k x channels), libopus's per-channel ceiling, so mono into WebM works
# (approved). (m) Atomic writes (instant-exec A4): FFmpeg and the mux each write a
# temporary file of the path (atomic_target, the muxer named as the extension picks it);
# a successful close commits the finished one, a failure or Stop abandons them. (n)
# instant-exec gate R5a: with audio, an MP4's or MOV's faststart (the node's, not a
# user's -movflags) goes to the mux, which writes the final file, instead of the encode
# pass; a source without audio then gets its video alone copied with faststart. (o)
# Auto transcodes PCM and FLAC audio for MP4 before the mux: FFmpeg 6.1 and later copy
# them into MP4, which many players cannot play, where 5.1.2 refused the copy.
INTENDED_MUX = (
    """        if self.audio is not None:
            video_path = self.save_path
            if not os.path.exists(video_path):
                logger.error(f"Video file not found at {video_path}")
                return

            base, ext = os.path.splitext(video_path)
            audio_video_path = f"{base}_av{ext}"

            # Default and auto -> copy
            output_params = {
                "vcodec": "copy",
                "acodec": "copy",
            }

            try:
                logger.debug(f"Attempting to process audio from: {self.audio}")

                # Create video stream
                video_stream = ffmpeg.input(str(video_path))

                # Handle WebM specific settings
                if self.container == VideoFormat.WEBM:
                    if self.audio_settings in (
                        AudioSettings.TRANSCODE,
                        AudioSettings.AUTO,
                    ):
                        output_params["acodec"] = "libopus"
                        output_params["b:a"] = "320k"
                    else:
                        raise ValueError(f"WebM does not support {self.audio_settings}")
                elif self.audio_settings == AudioSettings.TRANSCODE:
                    output_params["acodec"] = "aac"
                    output_params["b:a"] = "320k"

                # Use the ffmpeg environment from the class
                cmd = (
                    self.ffmpeg_env.ffmpeg if hasattr(self, "ffmpeg_env") else "ffmpeg"
                )

                output_video = ffmpeg.output(
                    video_stream,  # video first
                    self.audio,  # audio second (already an input stream)
                    str(audio_video_path),
                    **output_params,
                ).overwrite_output()

                try:
                    ffmpeg.run(
                        output_video, cmd=cmd, capture_stdout=True, capture_stderr=True
                    )
                except ffmpeg.Error as e:
                    error_msg = e.stderr.decode() if hasattr(e, "stderr") else str(e)
                    logger.error(f"FFmpeg error: {error_msg}")
                    return

                if os.path.exists(audio_video_path):
                    os.remove(video_path)
                    os.rename(audio_video_path, video_path)
                else:
                    logger.error(
                        f"Expected output file not created: {audio_video_path}"
                    )

            except Exception as e:
                error_msg = f"Failed to copy audio to video: {e!s}"
                logger.warning(error_msg)

                if os.path.exists(audio_video_path):
                    try:
                        os.remove(audio_video_path)
                    except Exception as cleanup_error:
                        logger.warning(
                            f"Failed to cleanup temporary file: {cleanup_error!s}"
                        )
""",
    """        if self.audio is not None:

            def audio_streams():
                source = self.audio.node.kwargs["filename"]
                probe = ffmpeg.probe(source, cmd=self.ffmpeg_env.ffprobe)
                return [s for s in probe["streams"] if s.get("codec_type") == "audio"]

            def options(settings):
                params = {
                    "vcodec": "copy",
                    "acodec": "copy",
                    "format": self.container.muxer,  # (m)
                }
                if self.container == VideoFormat.WEBM:
                    if settings in (AudioSettings.TRANSCODE, AudioSettings.AUTO):
                        params["acodec"] = "libopus"
                        params["b:a"] = "320k"
                        for index, stream in enumerate(audio_streams()):  # (h)
                            if stream.get("channels") == 1:
                                params[f"b:a:{index}"] = "256k"
                    else:
                        raise ValueError(f"WebM does not support {settings}")
                elif settings == AudioSettings.TRANSCODE:
                    params["acodec"] = "aac"
                    params["b:a"] = "320k"
                if self.faststart:  # (n)
                    params["movflags"] = "faststart"
                params["loglevel"] = "error"  # (b)
                return params

            settings = self.audio_settings
            if settings == AudioSettings.AUTO and self.container == VideoFormat.MP4:  # (o)
                for stream in audio_streams():
                    codec = stream.get("codec_name", "")
                    if codec.startswith("pcm_") or codec == "flac":
                        logger.info(
                            f"Auto transcodes the {codec} audio: players often cannot"
                            " play it in an .mp4 file"
                        )
                        settings = AudioSettings.TRANSCODE
                        break
            output_params = options(settings)  # (a)
            transcode_on_failure = (  # (c)
                settings == AudioSettings.AUTO and output_params["acodec"] == "copy"
            )
            video_path = self.save_path
            if not os.path.exists(self.target.temp):  # (m)
                logger.error(f"Video file not found at {video_path}")
                return self.target

            av = atomic_target(video_path)  # (m)
            try:
                logger.debug(f"Attempting to process audio from: {self.audio}")
                video_stream = ffmpeg.input(self.target.temp)
                muxed = False
                while True:
                    output_video = ffmpeg.output(
                        video_stream, self.audio, av.temp, **output_params
                    ).overwrite_output()
                    try:
                        ffmpeg.run(
                            output_video,
                            cmd=self.ffmpeg_env.ffmpeg,  # (e)
                            capture_stdout=True,
                            capture_stderr=True,
                        )
                        muxed = True
                        break
                    except ffmpeg.Error as e:  # (f)
                        if e.stderr is None:
                            message = str(e)
                        else:
                            message = e.stderr.decode("utf-8", "replace").strip()
                        streams = audio_streams()
                        if not streams:
                            source = self.audio.node.kwargs["filename"]
                            logger.warning(
                                "The audio source has no audio stream, so the video"
                                f" is saved without audio: {source}"
                            )
                            if self.faststart:  # (n)
                                alone = ffmpeg.output(
                                    video_stream,
                                    av.temp,
                                    vcodec="copy",
                                    format=self.container.muxer,
                                    movflags="faststart",
                                    loglevel="error",
                                ).overwrite_output()
                                ffmpeg.run(
                                    alone,
                                    cmd=self.ffmpeg_env.ffmpeg,
                                    capture_stdout=True,
                                    capture_stderr=True,
                                )
                                muxed = True
                            break
                        codec = streams[0].get("codec_name", "unknown")
                        container = self.container.value
                        if transcode_on_failure:
                            logger.info(
                                f"Auto transcodes the {codec} audio, which the"
                                f" .{container} file cannot hold as a copy: {message}"
                            )
                        elif output_params["acodec"] == "copy":
                            raise RuntimeError(
                                f"Save Video could not copy the {codec} audio into the"
                                f" .{container} file. Set Audio to Auto or Transcode"
                                f" to re-encode it. FFmpeg: {message}"
                            )
                        else:
                            raise RuntimeError(
                                f"Save Video could not transcode the {codec} audio to"
                                f" {output_params['acodec']} for the .{container}"
                                f" file. FFmpeg: {message}"
                            )
                    output_params = options(AudioSettings.TRANSCODE)
                    transcode_on_failure = False
                if not muxed:
                    av.abandon()  # (d)
                    return self.target
                if os.path.exists(av.temp):  # (g), (m)
                    self.target.abandon()
                    return av
                raise RuntimeError(f"Expected output file not created: {av.temp}")
            except BaseException:
                av.abandon()  # (d)
                raise
        return self.target  # (m)
""",
)


# Save Video's colours depart from both frozen references too (approved fix for
# upstream chaiNNer #3053; ARCHITECTURE section 7): (i) YUV output other than GIF gets
# all four colour tags, BT.601 for SD and BT.709 for HD (width >= 1280 or height >
# 576), the user's tags winning, and a scale filter converting to the final matrix and
# range after the user's -vf (-filter:v, -filter); (k) GIF with the node's yuv420p (which
# FFmpeg replaces with the fixed bgr8 palette) is written as pal8 with a palette made for
# each frame, after the user's -vf; a user pixel format, or a user -filter_complex (-lavfi),
# which FFmpeg refuses beside a -vf, keeps FFmpeg's conversion. (p) A setparams after the
# scale puts the matrix, primaries and transfer tags on the frames too: FFmpeg 7 and later
# take them from the frames over -colorspace, -color_primaries and -color_trc.
INTENDED_COLOURS = [
    (
        """@dataclass
class Writer:
""",
        """def colour_options(container, params, width, height):  # (i), (k)
    if not isinstance(params, dict):
        return params  # The caller's ** raises its own TypeError.
    if "pix_fmt" not in params or {"filter_complex", "lavfi"} & params.keys():
        return params
    if container == VideoFormat.GIF:
        if str(params["pix_fmt"]) != "yuv420p":
            return params
        params["pix_fmt"] = "pal8"
        filters = [params.pop(key) for key in ("vf", "filter:v", "filter") if key in params]
        palette = "split[a][b];[a]palettegen=stats_mode=single[p];[b][p]paletteuse=new=1"
        params["vf"] = ",".join([*filters, palette])
        return params
    if not str(params["pix_fmt"]).startswith(("yuv", "nv")):
        return params
    family = None
    if not {"colorspace", "color_primaries", "color_trc"} & params.keys():
        family = "bt709" if width >= 1280 or height > 576 else "smpte170m"
    elif "colorspace" in params:
        family = {
            "bt709": "bt709",
            "smpte170m": "smpte170m",
            "bt470bg": "smpte170m",
            "bt2020nc": "bt2020",
            "bt2020_ncl": "bt2020",
            "bt2020c": "bt2020",
            "bt2020_cl": "bt2020",
        }.get(str(params["colorspace"]))
    if family is not None:
        primaries = family
        if family == "smpte170m" and height == 576:
            primaries = "bt470bg"
        params.setdefault("colorspace", family)
        params.setdefault("color_primaries", primaries)
        params.setdefault("color_trc", "bt709" if family == "bt2020" else family)
    params.setdefault("color_range", "tv")
    params.setdefault("chroma_sample_location", "left")
    matrices = {
        "bt709": "bt709",
        "smpte170m": "smpte170m",
        "bt470bg": "bt470",
        "fcc": "fcc",
        "smpte240m": "smpte240m",
        "bt2020nc": "bt2020",
        "bt2020_ncl": "bt2020",
        "bt2020c": "bt2020",
        "bt2020_cl": "bt2020",
    }
    matrix, colour_range = str(params.get("colorspace", "")), str(params["color_range"])
    positions = {
        "left": (0, 128),
        "center": (128, 128),
        "topleft": (0, 0),
        "top": (128, 0),
        "bottomleft": (0, 256),
        "bottom": (128, 256),
    }
    location = str(params.get("chroma_sample_location", ""))
    filters = [params.pop(key) for key in ("vf", "filter:v", "filter") if key in params]
    scale = "scale="
    if matrix in matrices:
        scale += f"out_color_matrix={matrices[matrix]}:"
    if colour_range in ("tv", "mpeg", "pc", "jpeg"):
        scale += f"out_range={colour_range}:"
    if location in positions:
        scale += f"out_h_chr_pos={positions[location][0]}:"
        scale += f"out_v_chr_pos={positions[location][1]}:"
    params["vf"] = ",".join([*filters, scale + "flags=accurate_rnd+full_chroma_int"])
    names = {
        "colorspace": (
            "bt709", "bt470bg", "smpte170m", "bt2020nc", "bt2020c", "smpte240m", "fcc",
            "ycgco", "ictcp", "gbr",
        ),
        "color_primaries": (
            "bt709", "bt470m", "bt470bg", "smpte170m", "smpte240m", "film", "bt2020",
            "smpte428", "smpte431", "smpte432", "jedec-p22",
        ),
        "color_trc": (
            "bt709", "smpte170m", "bt470m", "bt470bg", "smpte240m", "linear",
            "iec61966-2-1", "iec61966-2-4", "bt2020-10", "bt2020-12", "smpte2084",
            "arib-std-b67", "smpte428",
        ),
    }
    tags = [f"{key}={params[key]}" for key in names if str(params.get(key)) in names[key]]
    if tags:  # (p)
        params["vf"] += ",setparams=" + ":".join(tags)
    return params


@dataclass
class Writer:
""",
    ),
    (
        """                    .output(**self.output_params, loglevel="error")""",
        """                    .output(
                        **colour_options(
                            self.container,
                            {**self.output_params, "loglevel": "error"}
                            if isinstance(self.output_params, dict)
                            else self.output_params,
                            width,
                            height,
                        )
                    )""",
    ),
]


# (l) RGB frames are piped as gbrp16le (16-bit G, B and R planes), so FFmpeg's YUV
# conversion keeps more than 8 bits of the float image; GIF and RGB output of up to 8
# bits keep bgr24, which stays exact there.
INTENDED_RGB16 = [
    (
        """@dataclass
class Writer:
""",
        """def rgb_pipe(container, params):  # (l)
    fmt = str(params.get("pix_fmt", "")) if hasattr(params, "get") else None
    output = None if fmt is None else pixel_layout(fmt)
    if container == VideoFormat.GIF or (
        output is not None and output.kind == "rgb" and output.bit_depth <= 8
    ):
        return "bgr24"
    return "gbrp16le"


@dataclass
class Writer:
""",
    ),
    (
        """                        pix_fmt="bgr24",
""",
        """                        pix_fmt=rgb_pipe(self.container, self.output_params),
""",
    ),
    (
        """        out_frame = to_uint8(img, normalized=True)
""",
        """        if rgb_pipe(self.container, self.output_params) == "bgr24":
            out_frame = to_uint8(img, normalized=True)
        else:
            out_frame = np.ascontiguousarray(
                to_uint16(img, normalized=True).transpose(2, 0, 1)[[1, 0, 2]]
            )
""",
    ),
]


# (j) The encoder's stderr is piped (native code drains its tail on a thread; the fakes
# have no stderr pipe), and a dead-pipe error (EPIPE, or EINVAL as Windows reports it)
# once FFmpeg has exited names FFmpeg and its exit code (upstream chaiNNer #3109).
INTENDED_ENCODER_ERRORS = [
    (
        """                        pipe_stdin=True, pipe_stdout=False, cmd=self.ffmpeg_env.ffmpeg
""",
        """                        pipe_stdin=True,
                        pipe_stdout=False,
                        pipe_stderr=True,
                        cmd=self.ffmpeg_env.ffmpeg,
""",
    ),
    (
        """            self.out.stdin.write(out_frame.tobytes())
""",
        """            try:
                self.out.stdin.write(out_frame.tobytes())
            except OSError as error:
                encoder_stopped(self.out, error)
                raise
""",
    ),
    (
        """            if self.out.stdin is not None:
                self.out.stdin.close()
            self.out.wait()
""",
        """            if self.out.stdin is not None:
                try:
                    self.out.stdin.close()
                except OSError as error:
                    encoder_stopped(self.out, error)
                    raise
            self.out.wait()
""",
    ),
    (
        """@dataclass
class Writer:
""",
        """def encoder_stopped(out, error):  # (j)
    import errno
    import subprocess

    if error.errno not in (errno.EPIPE, errno.EINVAL):
        return  # Not the OS's dead-pipe error.
    try:
        code = out.wait(timeout=5)
    except subprocess.TimeoutExpired:
        return
    raise RuntimeError(
        f"FFmpeg stopped while Save Video was writing the video (exit code {code})"
    ) from error


@dataclass
class Writer:
""",
    ),
]


# (m) Atomic writes outside the mux: the Writer's target, the node creating it, and close()
# committing the finished video or abandoning the target on a failure.
INTENDED_ATOMIC = [
    (
        """    def ext(self) -> str:
        return self.value
""",
        """    def ext(self) -> str:
        return self.value

    @property
    def muxer(self) -> str:
        return "matroska" if self == VideoFormat.MKV else self.value
""",
    ),
    (
        """    ffmpeg_env: FFMpegEnv
    out: Popen | None = None
""",
        """    ffmpeg_env: FFMpegEnv
    target: Any
    out: Popen | None = None
""",
    ),
    (
        """    def close(self):
""",
        """    def close(self):
        try:
            finished = self._close()
            if os.path.exists(finished.temp):
                finished.commit()
        except BaseException:
            self.target.abandon()
            raise

    def _close(self):
""",
    ),
    (
        """    save_path.parent.mkdir(parents=True, exist_ok=True)
""",
        """    save_path.parent.mkdir(parents=True, exist_ok=True)
    target = atomic_target(str(save_path))
""",
    ),
    (
        """        "filename": str(save_path),
""",
        """        "filename": target.temp,
        "format": container.muxer,
""",
    ),
    (
        """        ffmpeg_env=FFMpegEnv.get_integrated(node_context.storage_dir),
""",
        """        ffmpeg_env=FFMpegEnv.get_integrated(node_context.storage_dir),
        target=target,
""",
    ),
]


# (n): the encode pass of an MP4 or MOV with audio leaves the node's faststart to the mux.
INTENDED_FASTSTART = [
    (
        """    out: Popen | None = None
""",
        """    out: Popen | None = None
    faststart: bool = False
""",
    ),
    (
        """    if container == VideoFormat.GIF:
        audio = None
""",
        """    if container == VideoFormat.GIF:
        audio = None
    faststart = (
        audio is not None
        and output_params["movflags"] == "faststart"
        and container in (VideoFormat.MP4, VideoFormat.MOV)
    )
    if faststart:
        del output_params["movflags"]
""",
    ),
    (
        """        target=target,
""",
        """        target=target,
        faststart=faststart,
""",
    ),
]


COLOUR_NAMES = (
    "MATRIX_LABELS SWSCALE_CHROMA_POSITION VIDEO_MODE_LABELS ChromaLocation ColourInfo"
    " Matrix PixelLayout Primaries Range Transfer VideoMode code_to_float"
    " detect_colour_info float_to_code high_definition option_value pixel_layout"
    " subsampling_from_sizes swscale_matrix"
).split()


def preview_stub(node_id, loader, n):
    """video.py's preview_frame keeps each node's reader in module state, which the
    namespaces here lack: a black frame of the video's size, read from nowhere."""
    metadata = vars(loader)["metadata"]
    return np.zeros((metadata.height, metadata.width, 3), np.float32)


def link_loader(g, vg):
    """Load Video's namespace with the video module's names it imports; this
    repository's Preview reads through preview_stub."""
    for name in ("VideoLoader", "VideoMetadata", "ReadSettings"):
        if name in vg:
            g[name] = vg[name]
    if "load_preview" in g:
        g["preview_frame"] = preview_stub


def run_load_video(g, context, path, first=None, last=None):
    """This repository's Load Video with Frames All, or a Range first..last, Seek 0, Mode
    RGB, RGB Precision "8-bit (as before)" and every colour choice Auto, which read as the
    installed node did."""
    auto = (
        colour_info.Matrix.AUTO,
        colour_info.Transfer.AUTO,
        colour_info.Primaries.AUTO,
        colour_info.Range.AUTO,
        colour_info.ChromaLocation.AUTO,
    )
    frames = g["FramesMode"]
    return g["load_video_node"](
        context,
        path,
        0,
        frames.ALL if first is None else frames.RANGE,
        first or 0,
        last or 0,
        colour_info.VideoMode.RGB,
        g["RgbPrecision"].EIGHT_BIT,
        *auto,
    )


def call_load_video(kind, g, context, path, use_limit, limit):
    """Load Video's node as the installed node is called. This repository's node reads
    a limit as the Range 0..limit - 1 that old chains migrate to (src/common/migrations.ts
    loadVideoFrames); its extra outputs (Colour Info, Preview, Frame Count) are dropped."""
    if kind == "installed":
        return g["load_video_node"](context, path, use_limit, limit)
    if use_limit:
        return run_load_video(g, context, path, 0, limit - 1)[:5]
    return run_load_video(g, context, path)[:5]


def installed_item(item):
    """An item of this repository's Load Video as the installed node yields it: its
    Y, U and V (None in RGB mode) dropped."""
    if isinstance(item, tuple) and len(item) == 5:
        assert item[2:] == (None, None, None)
        return item[:2]
    return item


def installed_metadata(metadata):
    """This repository's LoadedVideo wraps the installed node's VideoMetadata."""
    return getattr(metadata, "metadata", metadata)


def load(kind, component, env=None):
    rel = "nodes/impl/video.py" if component == "video" else NODE + component + ".py"
    path = (
        ROOT / "backend/src" / rel
        if kind.startswith("native")
        else REFERENCE / ("source" if kind == "intended" else kind) / rel
    )
    namespace = types.ModuleType(f"video_test_{len(MODULES)}")
    MODULES.append(namespace)
    sys.modules[namespace.__name__] = namespace
    g = namespace.__dict__
    env = env or FakeFFmpeg()
    g.update(
        {
            "graph": graph,
            "dataclass": dataclass,
            "Enum": Enum,
            "Path": Path,
            "np": np,
            "os": os,
            "math": math,
            "itertools": itertools,
            "ffmpeg": env,
            "logger": Logger(env.events),
            "BufferedIOBase": io.BufferedIOBase,
            "subprocess": types.SimpleNamespace(Popen=FakeProcess),
            "Popen": FakeProcess,
            "Any": Any,
            "Literal": Literal,
            "get_h_w_c": get_h_w_c,
            "to_uint8": to_uint8,
            "to_uint16": to_uint16,
            "to_uint16_gbr_planes": to_uint16_gbr_planes,
            "pixel_layout": colour_info.pixel_layout,
            "split_file_path": split_file_path,
            "Generator": API["Generator"],
            "Collector": API["Collector"],
            "NodeContext": Context,
            "FFMpegEnv": types.SimpleNamespace(
                get_integrated=lambda _: types.SimpleNamespace(
                    ffmpeg="integrated-ffmpeg", ffprobe="integrated-ffprobe"
                )
            ),
        }
    )
    if kind.startswith("native"):
        # The names this repository's video modules import from colour_info.py, and
        # video.py's file_identity and planes_to_interleaved.
        g.update({name: getattr(colour_info, name) for name in COLOUR_NAMES})
        g["file_identity"] = file_identity
        g["planes_to_interleaved"] = planes_to_interleaved
        g["replace"] = replace
        g["resolve_colour_info"] = resolve_colour_info
        # Load Video's generator carries its frame range (with_range), which the
        # installed api lacks.
        g["Generator"] = Generator
    text = path.read_text(encoding="utf-8")
    if kind == "intended" and component == "save_video":
        for frozen, corrected in [
            INTENDED_MUX,
            *INTENDED_COLOURS,
            *INTENDED_ENCODER_ERRORS,
            *INTENDED_RGB16,
            *INTENDED_ATOMIC,
            *INTENDED_FASTSTART,
        ]:
            assert text.count(frozen) == 1
            text = text.replace(frozen, corrected)
    tree = ast.parse(text)
    body: list[ast.stmt] = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    for item in tree.body:
        if isinstance(item, (ast.ClassDef, ast.FunctionDef)):
            if isinstance(item, ast.FunctionDef):
                item.decorator_list = []
                if item.name == "get_item_types":
                    continue
            body.append(item)
        elif (
            isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and item.target.id == "PARAMETERS"
        ):
            body.append(item)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            str(path),
            "exec",
        ),
        g,
    )
    # The fakes write no file, so Save Video's temporary files are fakes too (m).
    g["atomic_target"] = lambda path: FakeTarget(env, path)
    return g


def writer(
    g,
    *,
    container="MKV",
    encoder: str | None = "FFV1",
    audio=None,
    audio_settings="AUTO",
    out=None,
    faststart=False,
):
    save_path = "owned/video." + container.lower()
    # The temporary file of the installed writer's own output file (m).
    target = (
        {"target": g["atomic_target"](save_path)}
        if "target" in g["Writer"].__dataclass_fields__
        else {}
    )
    return g["Writer"](
        container=g["VideoFormat"][container],
        encoder=None if encoder is None else g["VideoEncoder"][encoder],
        fps=30,
        audio=audio,
        audio_settings=g["AudioSettings"][audio_settings],
        save_path=save_path,
        output_params={"filename": "owned/video.mkv", "vcodec": "ffv1"},
        global_params=["-nostdin"],
        ffmpeg_env=types.SimpleNamespace(
            ffmpeg="integrated-ffmpeg", ffprobe="integrated-ffprobe"
        ),
        out=out,
        **target,
        **({"faststart": True} if faststart else {}),
    )


def mock_os(env, exists=True):
    def remove(path):
        env.record("remove", path)

    def rename(a, b):
        env.record("rename", a, b)

    def replace(a, b):
        env.record("replace", a, b)

    def path_exists(path):
        env.record("exists", path)
        return exists

    return types.SimpleNamespace(
        path=types.SimpleNamespace(splitext=os.path.splitext, exists=path_exists),
        remove=remove,
        rename=rename,
        replace=replace,
    )


def registration(path: Path, name: str) -> tuple[ast.FunctionDef, dict[str, ast.expr]]:
    tree = ast.parse(path.read_text())
    node = next(
        x
        for x in tree.body
        if isinstance(x, ast.FunctionDef) and x.name == name + "_node"
    )
    (decorator,) = node.decorator_list
    assert isinstance(decorator, ast.Call)
    return node, {k.arg: k.value for k in decorator.keywords if k.arg is not None}


def unwrapped(item: ast.expr) -> str:
    """An input or output as the installed node registers it: without the groups, the
    null_unless(...) or the make_optional() (its Mode) this repository puts around it."""
    while isinstance(item, ast.Call):
        if isinstance(item.func, ast.Attribute) and item.func.attr in (
            "null_unless",
            "make_optional",
        ):
            item = item.func.value
        elif isinstance(item.func, ast.Call) and len(item.args) == 1:
            item = item.args[0]
        else:
            break
    return ast.dump(item)


def label(item: ast.expr) -> object:
    """The label an input or output is made with: its class's first argument."""
    for node in ast.walk(item):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            return node.args[0].value
    return None


# Sanctioned changes of an installed registration, by node: its keywords, inputs and
# outputs (by label) and parameters. Load Video's frame range (native/local/video-seek):
# Frames, In and Out replace Use limit and Limit (old chains migrate,
# src/common/migrations.ts loadVideoFrames), Index is the frame's number in the video,
# and the description tells of the seek bar, the Range and one-frame runs.
REPLACED = {
    "load_video": {
        "keywords": {"description"},
        "inputs": {"Use limit", "Limit"},
        "outputs": {"Index"},
        "parameters": {"use_limit", "limit"},
    },
}


def test_reference_hashes_and_registration_contracts():
    for entry in json.loads((REFERENCE / "manifest.json").read_text())["files"]:
        assert (
            hashlib.sha256(
                (REFERENCE / PureWindowsPath(entry["snapshot"]).as_posix()).read_bytes()
            ).hexdigest()
            == entry["sha256"]
        )
    # Old chains keep working: the installed registration with inputs and outputs
    # added (Load Video's Mode and planes, Save Video's), every installed one kept in
    # order with its id, and the installed parameters in order; but for REPLACED.
    appended = {"inputs", "outputs", "iterator_inputs", "iterator_outputs"}
    for name in ("load_video", "save_video"):
        replaced: dict[str, set[str]] = REPLACED.get(name, {})
        old, old_keywords = registration(
            REFERENCE / "installed" / (NODE + name + ".py"), name
        )
        new, new_keywords = registration(
            ROOT / "backend/src" / (NODE + name + ".py"), name
        )
        for key in old_keywords.keys() | new_keywords.keys():
            if key not in appended and key not in replaced.get("keywords", ()):
                assert ast.dump(old_keywords[key]) == ast.dump(new_keywords[key]), key
        for key in ("inputs", "outputs"):
            old_items = old_keywords[key]
            new_items = new_keywords[key]
            assert isinstance(old_items, ast.List) and isinstance(new_items, ast.List)
            remaining = iter([unwrapped(x) for x in new_items.elts])
            for position, item in enumerate(old_items.elts):
                if label(item) in replaced.get(key, ()):
                    # A replaced input is gone; a replaced output keeps its place (id).
                    assert key == "inputs" or label(new_items.elts[position]) == label(
                        item
                    )
                    continue
                assert unwrapped(item) in remaining, (name, key, ast.unparse(item))
        new_arguments = iter([ast.dump(a) for a in new.args.args])
        for argument in old.args.args:
            if argument.arg not in replaced.get("parameters", ()):
                assert ast.dump(argument) in new_arguments, (name, argument.arg)


PROBE_CASES = [
    valid_probe(),
    {"streams": []},
    {"format": {}, "streams": []},
    {"format": {}},
    {"format": {}, "streams": [{}]},
]
for field in ("width", "height", "r_frame_rate", "nb_frames"):
    for value in (None, "", "bad", 0, -2, 1.75, "30000/1001", "1/0", "3", "inf", "nan"):
        probe = valid_probe()
        probe["streams"][1][field] = value
        PROBE_CASES.append(probe)
for stream_duration in (None, "1.001", "bad", "inf", "nan", -2):
    for format_duration in (None, "2.3", "bad"):
        probe = valid_probe()
        probe["streams"][1].pop("nb_frames")
        probe["streams"][1]["duration"] = stream_duration
        probe["format"]["duration"] = format_duration
        PROBE_CASES.append(probe)


@pytest.mark.parametrize("probe", PROBE_CASES)
def test_metadata_frozen(probe):
    results = []
    for kind in ("installed", "native"):
        env = FakeFFmpeg(copy.deepcopy(probe))
        g = load(kind, "video", env)
        result = outcome(
            lambda g=g: vars(
                g["VideoMetadata"].from_file(
                    Path("x.mkv"), types.SimpleNamespace(ffprobe="probe-bin")
                )
            )
        )
        results.append((result, env.events))
    assert results[0] == results[1]


@pytest.mark.parametrize("format_name", ["MP4_H264", "MP4_H265", "WEBM", "GIF"])
@pytest.mark.parametrize(
    "quality",
    [
        *range(101),
        -1,
        101,
        20.1,
        35.9,
        49.99,
        95.0001,
        float("nan"),
        float("inf"),
        None,
        "75",
    ],
)
def test_simple_quality_all_boundaries(format_name, quality):
    results = []
    for kind in ("installed", "native"):
        g = load(kind, "save_video")
        results.append(
            outcome(
                lambda g=g: g["get_simple_format"](
                    g["SimpleVideoFormat"][format_name], quality
                )
            )
        )
    assert results[0] == results[1]


@pytest.mark.parametrize("enum_name", ["VideoEncoder", "VideoFormat"])
def test_enum_compatibility(enum_name):
    values = []
    for kind in ("installed", "native"):
        g = load(kind, "save_video")
        values.append(
            [
                snap(
                    getattr(v, "formats" if enum_name == "VideoEncoder" else "encoders")
                )
                for v in g[enum_name]
            ]
        )
    assert values[0] == values[1]


@pytest.mark.parametrize(
    "chunks",
    [
        [],
        [b""],
        [bytes(range(12)), bytes(reversed(range(12)))],
        [b"1"],
        [b"123456"],
        [OSError("read failed")],
    ],
)
def test_reader_lazy_frames_and_errors(chunks, monkeypatch):
    # The FFmpeg CLI reader's own semantics: keep PyAV's in-process reader out.
    monkeypatch.setenv("CHAINNER_C_VIDEO_READER", "cli")
    results = []
    for kind in ("installed", "native"):
        env = FakeFFmpeg(chunks=chunks)
        g = load(kind, "video", env)
        obj = g["VideoLoader"](
            Path("input.mkv"),
            types.SimpleNamespace(ffmpeg="reader-bin", ffprobe="probe-bin"),
        )
        it = obj.stream_frames()
        assert [e[0] for e in env.events] == ["probe"]
        result = outcome(lambda it=it: list(it))
        results.append((result, env.events.copy()))
        assert not env.processes or env.processes[0].returncode is not None
    assert results[0] == results[1]


@pytest.mark.parametrize(
    "step", ["input", "output", "run_async", "enter", "read", "exit", "wait"]
)
def test_reader_injected_errors(step):
    results = []
    for kind in ("installed", "native"):
        env = FakeFFmpeg(chunks=[bytes(12)], failures={step: OSError(step)})
        g = load(kind, "video", env)
        obj = g["VideoLoader"](
            Path("input.mkv"),
            types.SimpleNamespace(ffmpeg="reader-bin", ffprobe="probe-bin"),
        )
        result = outcome(lambda obj=obj: list(obj.stream_frames()))
        results.append(result)
        env.failures.clear()
        for proc in env.processes:
            proc.__exit__(None, None, None)
    assert results[0] == results[1]


def test_reader_abandon_close_is_owned_idempotent():
    env = FakeFFmpeg(chunks=[bytes(12)] * 5)
    g = load("native", "video", env)
    obj = g["VideoLoader"](
        Path("input.mkv"),
        types.SimpleNamespace(ffmpeg="reader-bin", ffprobe="probe-bin"),
    )
    unused = obj.stream_frames()
    unused.close()
    assert not env.processes
    it = obj.stream_frames()
    frame = next(it)
    assert not frame.flags.writeable
    it.close()
    events = env.events.copy()
    it.close()
    assert env.events == events
    assert sum(e[0] == "terminate" for e in env.events) == 1
    assert env.processes[0].returncode is not None
    assert frame.tobytes() == bytes(12)
    with pytest.raises(StopIteration):
        next(it)


def test_frame_is_a_read_only_view_with_todays_flags():
    # D8 (SP4c 4.7): a frame is read into a fresh NumPy buffer, which the NumPy pool
    # can serve, and returned as a read-only view of it with today's frombuffer flags.
    data = bytes(range(12))
    env = FakeFFmpeg(chunks=[data])
    g = load("native", "video", env)
    loader = g["VideoLoader"](Path("x"), types.SimpleNamespace(ffmpeg="f", ffprobe="p"))
    frame = next(loader.stream_frames())
    today = np.frombuffer(data, np.uint8).reshape(2, 2, 3)
    assert frame.tobytes() == today.tobytes()
    assert (frame.shape, frame.dtype) == (today.shape, today.dtype)
    flags = ("C_CONTIGUOUS", "OWNDATA", "WRITEABLE", "ALIGNED")
    values = [frame.flags[flag] for flag in flags]
    assert values == [today.flags[flag] for flag in flags]
    assert values == [True, False, False, True]
    for array in (frame, today):  # neither can be made writeable again
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag to True"):
            array.setflags(write=True)
    # Today's base is frombuffer's array over the bytes, which owns no data (NumPy
    # stops collapsing a view chain at the bytes); D8's is the buffer that owns it.
    assert type(frame.base) is np.ndarray
    assert (frame.base.dtype, frame.base.shape) == (np.uint8, (12,))
    assert frame.base.flags.owndata


def test_probes_are_kept_per_file_version_ffprobe_and_options(tmp_path):
    # ffprobe runs once per file version (path, size, times), ffprobe and options: a
    # file previewed and read again is not probed again until it changes. Each call
    # gets its own copy; a file that cannot be stat'ed and a failed probe are not kept.
    path = tmp_path / "clip.mkv"
    path.write_bytes(b"video")
    env = FakeFFmpeg()
    g = load("native", "video", env)
    ffprobe = str(tmp_path / "ffprobe.exe")
    tools = types.SimpleNamespace(ffmpeg="f", ffprobe=ffprobe)

    def probes():
        return sum(e[0] == "probe" for e in env.events)

    first = g["VideoMetadata"].from_file(path, tools)
    loader = g["VideoLoader"](path, tools)
    assert probes() == 1 and loader.metadata == first
    assert env.events[0] == ("probe", (str(path),), (("cmd", ffprobe),))
    loader.video_stream["width"] = 999  # the caller's own copy
    assert g["VideoMetadata"].from_file(path, tools) == first and probes() == 1
    # The first frame's probe (HDR10 metadata) is kept as well, apart.
    frame = {"select_streams": "v:0", "read_intervals": "%+#1", "show_frames": None}
    video_probe = graph().video_probe
    assert video_probe(g, str(path), ffprobe, **frame) == valid_probe()
    assert video_probe(g, str(path), ffprobe, **frame) == valid_probe()
    assert probes() == 2
    assert env.events[-1] == ("probe", (str(path),), snap({"cmd": ffprobe, **frame}))
    # Another ffprobe, a changed file, a missing file and a failed probe probe again.
    g["VideoMetadata"].from_file(path, types.SimpleNamespace(ffmpeg="f", ffprobe="p"))
    assert probes() == 3
    info = path.stat()
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
    g["VideoMetadata"].from_file(path, tools)
    g["VideoMetadata"].from_file(path, tools)
    assert probes() == 4
    for _ in range(2):
        g["VideoMetadata"].from_file(tmp_path / "missing.mkv", tools)
    assert probes() == 6
    path.write_bytes(b"other video")
    env.failures["probe"] = OSError("ffprobe failed")
    with pytest.raises(OSError, match="ffprobe failed"):
        g["VideoMetadata"].from_file(path, tools)
    env.failures.clear()
    g["VideoMetadata"].from_file(path, tools)
    assert probes() == 8


def test_a_same_size_rewrite_with_its_time_put_back_is_probed_again(tmp_path):
    # Windows' st_ctime_ns is the creation time, so the file's version includes its
    # change time, which reading the file leaves and any write or time change moves.
    path = tmp_path / "clip.mkv"
    path.write_bytes(b"video")
    env = FakeFFmpeg()
    g = load("native", "video", env)
    tools = types.SimpleNamespace(ffmpeg="f", ffprobe="p")
    g["VideoMetadata"].from_file(path, tools)
    before = file_identity(path)
    assert path.read_bytes() == b"video" and file_identity(path) == before
    info = path.stat()
    time.sleep(0.05)  # past the file system clock's tick
    with path.open("r+b") as file:
        file.write(b"other")
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
    after = path.stat()
    assert (after.st_size, after.st_mtime_ns) == (info.st_size, info.st_mtime_ns)
    assert file_identity(path) != before
    g["VideoMetadata"].from_file(path, tools)
    assert sum(e[0] == "probe" for e in env.events) == 2


def test_the_options_cannot_name_another_ffprobe(tmp_path):
    path = tmp_path / "clip.mkv"
    path.write_bytes(b"video")
    env = FakeFFmpeg()
    g = load("native", "video", env)
    with pytest.raises(TypeError, match="multiple values for keyword argument 'cmd'"):
        graph().video_probe(g, str(path), "p", cmd="other")
    assert env.events == []


def test_concurrent_probes_each_get_the_answer(tmp_path):
    env = FakeFFmpeg()
    g = load("native", "video", env)
    tools = types.SimpleNamespace(ffmpeg="f", ffprobe="p")
    paths = [tmp_path / f"clip{index}.mkv" for index in range(8)]
    for path in paths:
        path.write_bytes(b"video")
    record = env.probe

    def slow_probe(*args, **kwargs):
        time.sleep(0.005)  # probes overlap
        return record(*args, **kwargs)

    env.probe = slow_probe
    expected = g["VideoMetadata"].from_file(tmp_path / "missing.mkv", tools)

    def read(offset):
        return [
            g["VideoMetadata"].from_file(paths[(offset + index) % 8], tools)
            for index in range(24)
        ]

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(read, range(8)))
    assert all(metadata == expected for result in results for metadata in result)
    probed = sum(e[0] == "probe" for e in env.events)
    assert 1 + 8 <= probed <= 1 + 8 * 8
    for path in paths:
        assert g["VideoMetadata"].from_file(path, tools) == expected
    assert sum(e[0] == "probe" for e in env.events) == probed


def test_the_64_newest_probes_are_kept(tmp_path):
    env = FakeFFmpeg()
    g = load("native", "video", env)
    tools = types.SimpleNamespace(ffmpeg="f", ffprobe="p")
    paths = [tmp_path / f"clip{index}.mkv" for index in range(65)]
    for path in paths:
        path.write_bytes(b"video")

    def probed(indices):
        before = sum(e[0] == "probe" for e in env.events)
        for index in indices:
            g["VideoMetadata"].from_file(paths[index], tools)
        return sum(e[0] == "probe" for e in env.events) - before

    assert probed(range(64)) == 64  # these 64 now fill the cache
    assert probed(range(64)) == 0
    assert probed([64]) == 1  # evicts the oldest, the first file
    assert probed([0]) == 1  # evicts the second
    assert probed(range(2, 65)) == 0
    assert probed([1]) == 1


@pytest.mark.parametrize("short", [b"1", b"123456", bytes(11)])
def test_short_read_raises_todays_error(short):
    env = FakeFFmpeg(chunks=[short])
    g = load("native", "video", env)
    loader = g["VideoLoader"](Path("x"), types.SimpleNamespace(ffmpeg="f", ffprobe="p"))
    with pytest.raises(ValueError) as native:
        next(loader.stream_frames())
    with pytest.raises(ValueError) as today:
        np.frombuffer(short, np.uint8).reshape([2, 2, 3])
    assert type(native.value) is type(today.value)
    assert str(native.value) == str(today.value)
    assert env.processes[0].stdout.closed


def test_end_of_stream_stops_as_today():
    env = FakeFFmpeg(chunks=[bytes(12)])
    g = load("native", "video", env)
    loader = g["VideoLoader"](Path("x"), types.SimpleNamespace(ffmpeg="f", ffprobe="p"))
    it = loader.stream_frames()
    next(it)
    with pytest.raises(StopIteration):
        next(it)  # a 0-byte read
    assert [e for e in env.events if e[0] in ("read", "debug", "exit")] == [
        ("read", 12),
        ("read", 12),
        ("debug", ("Can't receive frame (stream end?). Exiting ...",), ()),
        ("exit", None, None),
    ]
    assert env.processes[0].returncode is not None
    with pytest.raises(StopIteration):
        next(it)


@pytest.mark.parametrize("ending", ["eof", "stop", "range"])
def test_reader_nonzero_exit_only_fails_at_natural_eof(tmp_path, monkeypatch, ending):
    monkeypatch.setenv("CHAINNER_C_VIDEO_READER", "cli")
    env = FakeFFmpeg(chunks=[bytes(12)])
    vg = load("native", "video", env)
    loader = vg["VideoLoader"](
        Path("input.mkv"), types.SimpleNamespace(ffmpeg="f", ffprobe="p")
    )
    frames = loader.stream_frames()
    if ending == "range":
        frames = vg["_first"](frames, 1)
    g = load("native", "save_video", env)
    g["os"] = mock_os(env)
    context = Context()
    collector = g["save_video_node"](**node_args(g, context, tmp_path))
    try:
        collector.on_iterate(next(frames))
        decoder = env.processes[0]
        decoder.returncode = 23
        if ending == "eof":
            with pytest.raises(RuntimeError, match=r"Load Video.*exit code 23"):
                for frame in frames:
                    collector.on_iterate(frame)
                collector.on_complete()
        elif ending == "range":
            with pytest.raises(StopIteration):
                next(frames)
            collector.on_complete()
        else:
            frames.close()
    finally:
        frames.close()
        context.cleanup()
    assert decoder.returncode == 23
    assert decoder.stdout.closed
    assert sum(event[0] == "write" for event in env.events) == 1
    assert any(event[0] == "commit" for event in env.events) == (ending == "range")
    if ending != "range":
        assert any(event[0] == "abandon" for event in env.events)


class CountingRaw(io.FileIO):
    """A pipe's raw reader that records what each raw read asked for and returned."""

    def __init__(self, fd):
        super().__init__(fd, "rb")
        self.requests, self.counts = [], []
        self.returned = threading.Event()

    def readinto(self, buffer):
        self.requests.append(memoryview(buffer).nbytes)
        count = super().readinto(buffer)
        self.counts.append(count)
        self.returned.set()
        return count


def test_frames_from_a_pipe_written_in_parts_arrive_whole():
    # D8 reads a frame with one readinto on Popen's stdout, an io.BufferedReader over
    # a pipe. Like read(n), it must return the whole frame although the pipe delivers
    # it in parts. The first raw read gets a part only; a frame (9216 bytes) is larger
    # than the reader's 8192-byte buffer, so readinto reads both into the frame and
    # through its buffer. The pipe then ends inside a third frame.
    width, height = 64, 48
    size = width * height * 3
    data = np.random.default_rng(0).bytes(2 * size + size // 3)
    frames, tail = [data[:size], data[size : 2 * size]], data[2 * size :]
    read_fd, write_fd = os.pipe()
    raw = CountingRaw(read_fd)
    reader = io.BufferedReader(raw)

    def write():
        try:
            for start in range(0, len(data), 700):
                part = data[start : start + 700]
                while part:
                    part = part[os.write(write_fd, part) :]
                if start == 0:
                    raw.returned.wait(timeout=10)  # until a raw read returned this part
                time.sleep(0.001)
        finally:
            os.close(write_fd)

    probe = {
        "format": {"duration": "1"},
        "streams": [
            {
                "codec_type": "video",
                "width": width,
                "height": height,
                "r_frame_rate": "30/1",
                "nb_frames": "3",
            }
        ],
    }
    env = FakeFFmpeg(probe=probe, stdout=reader)
    g = load("native", "video", env)
    it = g["VideoLoader"](
        Path("x"), types.SimpleNamespace(ffmpeg="f", ffprobe="p")
    ).stream_frames()
    thread = threading.Thread(target=write)
    thread.start()
    try:
        got = [next(it).tobytes() for _ in frames]
        short = outcome(lambda: next(it))
    finally:
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert 0 < raw.counts[0] < size
    assert got == frames
    shape = [height, width, 3]
    assert short == outcome(lambda: np.frombuffer(tail, np.uint8).reshape(shape))
    assert reader.closed


def test_large_frames_are_read_in_slices_of_at_most_1_mib():
    # A Windows pipe read asked for a whole large frame is several times slower per
    # byte, so a frame is read in slices of at most 1 MiB, each filled as the whole
    # frame was. Frames of 1.6 MiB (two slices) arrive whole, and the pipe ending
    # inside a third frame's second slice gives today's error for the bytes read.
    width, height = 1024, 560
    size = width * height * 3
    data = np.random.default_rng(1).bytes(2 * size + size * 3 // 4)
    frames, tail = [data[:size], data[size : 2 * size]], data[2 * size :]
    read_fd, write_fd = os.pipe()
    raw = CountingRaw(read_fd)
    reader = io.BufferedReader(raw)

    def write():
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(write_fd, view[: 1 << 16]) :]
        finally:
            os.close(write_fd)

    probe = {
        "format": {"duration": "1"},
        "streams": [
            {
                "codec_type": "video",
                "width": width,
                "height": height,
                "r_frame_rate": "30/1",
                "nb_frames": "3",
            }
        ],
    }
    env = FakeFFmpeg(probe=probe, stdout=reader)
    g = load("native", "video", env)
    it = g["VideoLoader"](
        Path("x"), types.SimpleNamespace(ffmpeg="f", ffprobe="p")
    ).stream_frames()
    thread = threading.Thread(target=write)
    thread.start()
    try:
        got = [next(it).tobytes() for _ in frames]
        short = outcome(lambda: next(it))
    finally:
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert size > 1 << 20
    assert max(raw.requests) <= 1 << 20
    assert got == frames
    shape = [height, width, 3]
    assert short == outcome(lambda: np.frombuffer(tail, np.uint8).reshape(shape))
    assert reader.closed


@pytest.mark.parametrize("container", ["MKV", "MP4", "MOV", "WEBM", "AVI", "GIF"])
@pytest.mark.parametrize("audio_setting", ["AUTO", "COPY", "TRANSCODE"])
def test_audio_policies_exact(container, audio_setting):
    # The audio mux follows INTENDED_MUX, not the installed nightly (#3331).
    results = []
    for kind in ("intended", "native"):
        env = FakeFFmpeg()
        g = load(kind, "save_video", env)
        g["os"] = mock_os(env)
        w = writer(
            g,
            container=container,
            audio=FakeStream(env, "audio"),
            audio_settings=audio_setting,
        )
        results.append((outcome(w.close), env.events.copy()))
    assert results[0] == results[1]
    result, events = results[1]
    if result[0] == "ok":
        mux = next(e for e in events if e[0] == "output")
        assert mux[1][:2] == (("stream", "input"), ("stream", "audio"))
        run = next(e for e in events if e[0] == "run")
        assert ("cmd", "integrated-ffmpeg") in run[2]
    else:
        assert (container, audio_setting) == ("WEBM", "COPY")


RUN_ERROR = ffmpeg.Error("ffmpeg", b"out", b"error details")


@pytest.mark.parametrize(
    "step", ["input", "output", "overwrite", "run", "probe", "commit"]
)
@pytest.mark.parametrize("error", [OSError("failure"), RUN_ERROR])
@pytest.mark.parametrize("setting", ["AUTO", "COPY", "TRANSCODE"])
def test_audio_errors_exact(step, error, setting):
    # INTENDED_MUX (#3331), (m): every failure reaches the caller after the temporary
    # files are abandoned (the mux's within it), never the path. The probe runs only
    # after a failed mux, so it fails with one.
    failures = {step: error} if step != "probe" else {"run": RUN_ERROR, step: error}
    results = []
    for kind in ("intended", "native"):
        env = FakeFFmpeg(failures=failures)
        g = load(kind, "save_video", env)
        g["os"] = mock_os(env)
        w = writer(g, audio=FakeStream(env, "audio"), audio_settings=setting)
        results.append((outcome(w.close), env.events.copy()))
    assert results[0] == results[1]
    assert results[1][0][0] == "error"
    assert results[1][1][-1] == ("abandon", "owned/video.mkv.0.tmp")
    if step != "commit":
        assert results[1][1][-2] == ("abandon", "owned/video.mkv.1.tmp")


class FailFirstMux(FakeFFmpeg):
    """FFmpeg whose first mux fails as it does for audio the container cannot hold."""

    def run(self, *args, **kwargs):
        first = not any(e[0] == "run" for e in self.events)
        self.record("run", args, kwargs)
        if first:
            raise ffmpeg.Error("ffmpeg", b"", b"Could not find tag for codec\n")


@pytest.mark.parametrize("setting", ["AUTO", "COPY", "TRANSCODE"])
@pytest.mark.parametrize("source_audio", [True, False])
def test_failed_copy_transcodes_on_auto_and_raises_otherwise(setting, source_audio):
    # #3331: Auto retries once with Transcode's options; Copy and Transcode raise;
    # a source without audio has nothing to lose, so its video is kept quietly.
    probe = valid_probe() if source_audio else {"streams": [{"codec_type": "video"}]}
    results = []
    for kind in ("intended", "native"):
        env = FailFirstMux(probe=probe)
        g = load(kind, "save_video", env)
        g["os"] = mock_os(env)
        w = writer(
            g, container="MP4", audio=FakeStream(env, "audio"), audio_settings=setting
        )
        results.append((outcome(w.close), env.events.copy()))
    assert results[0] == results[1]
    result, events = results[1]
    options = [dict(e[2]) for e in events if e[0] == "output"]
    probes = [e for e in events if e[0] == "probe"]
    assert probes[0] == (
        "probe",
        ("audio-source.mkv",),
        (("cmd", "integrated-ffprobe"),),
    )
    if not source_audio:  # (m): the video's own temporary file is committed.
        assert result == ("ok", None) and len(options) == 1
        assert events[-1] == ("commit", "owned/video.mp4.0.tmp")
        assert any(e[0] == "warning" for e in events)
    elif setting == "AUTO":
        assert result == ("ok", None)
        assert [o["acodec"] for o in options] == ["copy", "aac"]
        assert options[1]["b:a"] == "320k"
        assert events[-1] == ("commit", "owned/video.mp4.1.tmp")
    else:
        assert result[0] == "error" and result[1][0] == "RuntimeError"
        assert "the unknown audio" in result[1][1]
        assert ("Set Audio to Auto or Transcode" in result[1][1]) == (setting == "COPY")
        assert len(options) == 1


@pytest.mark.parametrize(
    ("container", "codec", "acodec"),
    [
        ("MP4", "pcm_s16le", "aac"),
        ("MP4", "flac", "aac"),
        ("MP4", "aac", "copy"),
        ("MOV", "pcm_s16le", "copy"),
    ],
)
def test_auto_transcodes_pcm_and_flac_for_mp4(container, codec, acodec):
    # (o): one mux, with Transcode's options for PCM and FLAC into MP4 only.
    probe = {"streams": [{"codec_type": "audio", "codec_name": codec}]}
    results = []
    for kind in ("intended", "native"):
        env = FakeFFmpeg(probe=probe)
        g = load(kind, "save_video", env)
        g["os"] = mock_os(env)
        w = writer(
            g,
            container=container,
            audio=FakeStream(env, "audio"),
            audio_settings="AUTO",
        )
        results.append((outcome(w.close), env.events.copy()))
    assert results[0] == results[1]
    result, events = results[1]
    assert result == ("ok", None)
    assert [dict(e[2])["acodec"] for e in events if e[0] == "output"] == [acodec]


@pytest.mark.parametrize("source_audio", [True, False])
def test_faststart_moved_to_the_mux_writes_the_final_file_with_it(source_audio):
    # (n): the mux writes the final MP4 with the faststart the encode pass left out; a
    # source without audio gets its video alone copied with it.
    probe = valid_probe() if source_audio else {"streams": [{"codec_type": "video"}]}
    results = []
    for kind in ("intended", "native"):
        env = FakeFFmpeg(probe=probe) if source_audio else FailFirstMux(probe=probe)
        g = load(kind, "save_video", env)
        g["os"] = mock_os(env)
        w = writer(g, container="MP4", audio=FakeStream(env, "audio"), faststart=True)
        results.append((outcome(w.close), env.events.copy()))
    assert results[0] == results[1]
    result, events = results[1]
    assert result == ("ok", None)
    outputs = [e for e in events if e[0] == "output"]
    assert all(dict(e[2])["movflags"] == "faststart" for e in outputs)
    assert events[-1] == ("commit", "owned/video.mp4.1.tmp")
    if not source_audio:
        assert outputs[-1][1:] == (
            (("stream", "input"), "owned/video.mp4.1.tmp"),
            snap(
                {
                    "vcodec": "copy",
                    "format": "mp4",
                    "movflags": "faststart",
                    "loglevel": "error",
                }
            ),
        )


@pytest.mark.parametrize(
    "first,last,cut",
    [
        (None, None, {}),
        (0, 3, {"t": 4}),
        (0, 2, {"t": 3}),
        (0, 0, {"t": 1}),
        (1, 2, {"ss": 1, "t": 2}),
        (3, 3, {"ss": 3, "t": 1}),
        (0, 9, {"t": 10}),
    ],
)
def test_a_range_cuts_the_audio(first, last, cut):
    # instant-exec gate R7: the mux reads the source's audio only for as long as the
    # frames read last, from the first one's time: the input's -ss In / fps (from the
    # start, none) and -t (Out - In + 1) / fps. The probe's count (4) is an estimate here
    # (the fake file has no frame index), so an Out past it cuts at Out: audio past the
    # end is not there to copy. All keeps the whole audio, as before. (A limit, which
    # this Range replaces, kept it whole when it left no frame out.)
    env = FakeFFmpeg()
    vg = load("native", "video", env)
    g = load("native", "load_video", env)
    link_loader(g, vg)
    path = Path("folder/test.mkv")
    run_load_video(g, Context(), path, first, last)
    expected = {key: frames / (30000 / 1001) for key, frames in cut.items()}
    assert [e for e in env.events if e[0] == "input"] == [
        ("input", snap((path,)), snap(expected))
    ]


@pytest.mark.parametrize("channels", [None, 1, 2, 6])
@pytest.mark.parametrize("setting", ["AUTO", "TRANSCODE"])
def test_webm_opus_bitrate_follows_the_channel_count(channels, setting):
    # (h): min(320k, 256k x channels); a stream without a channel count keeps 320k.
    probe = valid_probe()
    if channels is not None:
        probe["streams"][0]["channels"] = channels
    results = []
    for kind in ("intended", "native"):
        env = FakeFFmpeg(probe=probe)
        g = load(kind, "save_video", env)
        g["os"] = mock_os(env)
        w = writer(
            g, container="WEBM", audio=FakeStream(env, "audio"), audio_settings=setting
        )
        results.append((outcome(w.close), env.events.copy()))
    assert results[0] == results[1]
    mux = dict(next(e for e in results[1][1] if e[0] == "output")[2])
    assert mux["b:a"] == "320k"
    assert mux.get("b:a:0") == ("256k" if channels == 1 else None)


def test_webm_opus_bitrate_is_per_audio_stream():
    # (h): of a 5.1 + mono source, only the mono stream gets 256k.
    probe = valid_probe()
    probe["streams"][0]["channels"] = 6
    probe["streams"].append({**probe["streams"][0], "channels": 1})
    results = []
    for kind in ("intended", "native"):
        env = FakeFFmpeg(probe=probe)
        g = load(kind, "save_video", env)
        g["os"] = mock_os(env)
        w = writer(g, container="WEBM", audio=FakeStream(env, "audio"))
        results.append((outcome(w.close), env.events.copy()))
    assert results[0] == results[1]
    mux = dict(next(e for e in results[1][1] if e[0] == "output")[2])
    assert (mux["b:a"], mux.get("b:a:0"), mux["b:a:1"]) == ("320k", None, "256k")


@pytest.mark.parametrize("encoder", [None, "H264", "H265", "VP9", "FFV1"])
@pytest.mark.parametrize(
    "shape", [(2, 2, 3), (1, 2, 3), (2, 1, 3), (0, 2, 3), (2, 2), (2, 2, 4)]
)
def test_writer_frames_and_even_contract(encoder, shape):
    results = []
    for kind in ("intended", "native"):
        env = FakeFFmpeg()
        g = load(kind, "save_video", env)
        w = writer(g, encoder=encoder)
        image = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
        result = outcome(lambda w=w, image=image: w.write_frame(image))
        if w.out is not None:
            w.close()
        results.append((result, env.events.copy()))
    assert results[0] == results[1]


@pytest.mark.parametrize("layout", ["contiguous", "readonly", "reverse", "fortran"])
def test_writer_foreign_layouts(layout):
    image = np.arange(36, dtype=np.float32).reshape(3, 4, 3) / 35
    if layout == "readonly":
        image.flags.writeable = False
    elif layout == "reverse":
        image = image[::-1, ::-1]
    elif layout == "fortran":
        image = np.asfortranarray(image)
    before = image.tobytes()
    results = []
    for kind in ("intended", "native"):
        env = FakeFFmpeg()
        g = load(kind, "save_video", env)
        w = writer(g)
        w.write_frame(image)
        w.close()
        results.append(env.events.copy())
    assert results[0] == results[1]
    assert image.tobytes() == before


@pytest.mark.parametrize(
    "step",
    [
        "input",
        "output",
        "overwrite",
        "global_args",
        "run_async",
        "write",
        "stdin.close",
        "wait",
    ],
)
def test_writer_failure_contract(step):
    results = []
    for kind in ("intended", "native"):
        env = FakeFFmpeg(failures={step: BrokenPipeError(errno.EPIPE, step)})
        g = load(kind, "save_video", env)
        w = writer(g)
        result = outcome(
            lambda w=w: (w.write_frame(np.zeros((2, 2, 3), np.float32)), w.close())
        )
        results.append((result, env.events.copy()))
        env.failures.clear()
        if w.out is not None:
            w.close()
    assert results[0] == results[1]
    # (j): a pipe error once FFmpeg has exited names FFmpeg (upstream chaiNNer #3109).
    if step in ("write", "stdin.close"):
        assert results[1][0][1][0] == "RuntimeError"
        assert "FFmpeg stopped" in results[1][0][1][1]


def test_successful_close_mux_once_deliberate_correction():
    env = FakeFFmpeg()
    g = load("native", "save_video", env)
    g["os"] = mock_os(env)
    w = writer(g, audio=FakeStream(env, "audio"))
    w.write_frame(np.zeros((2, 2, 3), np.float32))
    w.close()
    expected = env.events.copy()
    w.close()
    w._native_video_resource.abort()
    assert env.events == expected
    assert sum(e[0] == "run" for e in expected) == 1


def installed_writer_arguments(kwargs):
    """Save Video's Writer arguments as the installed node passes them: without the
    FFmpeg environment and colour settings, FFmpeg writing the path itself instead of
    the target's temporary file with the extension's muxer named (m), and the faststart
    the mux takes (n) in the encode pass's parameters."""
    kwargs = {k: v for k, v in kwargs.items() if k not in ("ffmpeg_env", "settings")}
    target = kwargs.pop("target", None)
    if target is not None:
        params = dict(kwargs["output_params"])
        assert params.pop("filename") == target.temp
        assert params.pop("format") == kwargs["container"].muxer
        kwargs["output_params"] = {"filename": kwargs["save_path"], **params}
    if kwargs.pop("faststart", False):
        params = {}
        for key, value in kwargs["output_params"].items():
            params[key] = value
            if key == "r":
                params["movflags"] = "faststart"
        kwargs["output_params"] = params
    return kwargs


def node_args(g, context, directory, **changes):
    kwargs = {
        "node_context": context,
        "_": None,
        "save_dir": directory,
        "video_name": "clip",
        "simplicity": g["Simplicity"].ADVANCED,
        "container": g["VideoFormat"].MKV,
        "encoder": g["VideoEncoder"].FFV1,
        "video_preset": g["VideoPreset"].MEDIUM,
        "crf": 23,
        "additional_parameters": None,
        "simple_video_format": g["SimpleVideoFormat"].MP4_H264,
        "quality": 75,
        "fps": 30,
        "audio": None,
        "audio_settings": g["AudioSettings"].AUTO,
    }
    if "VideoInputMode" in g:
        kwargs.update(
            mode=g["VideoInputMode"].RGB,
            _y=None,
            _u=None,
            _v=None,
            colour_info=None,
            matrix=colour_info.Matrix.AUTO,
            transfer=colour_info.Transfer.AUTO,
            primaries=colour_info.Primaries.AUTO,
            range_=colour_info.Range.AUTO,
            chroma_location=colour_info.ChromaLocation.AUTO,
            bit_depth=g["BitDepth"].AUTO,
            siting_correction=g["SitingCorrection"].AUTO,
        )
    kwargs.update(changes)
    return kwargs


@pytest.mark.parametrize(
    "additional",
    [
        None,
        "",
        "  ",
        "-nostdin",
        "-threads 1",
        "-pix_fmt bgr0 -threads 1",
        "-foo a b",
        "-filename bad",
        "-vcodec h264",
        "-crf 0",
        "-preset fast",
        "-c:v libx264",
        "-filename2 bad",
        "trailing",
        "-foo -bar 2",
        "-x\t2\n-y 3",
    ],
)
def test_additional_argument_policy(tmp_path, additional):
    def run(kind):
        env = FakeFFmpeg()
        g = load(kind, "save_video", env)
        captured = []
        original_writer = g["Writer"]

        def capture(**kwargs):
            captured.append(snap(installed_writer_arguments(kwargs)))
            return original_writer(**kwargs)

        g["Writer"] = capture
        result = outcome(
            lambda: bool(
                g["save_video_node"](
                    **node_args(
                        g, Context(), tmp_path, additional_parameters=additional
                    )
                )
            )
        )
        return result, captured

    results = []
    for kind in ("installed", "native"):
        results.append(run(kind))
    assert results[0] == results[1]


@pytest.mark.parametrize("container", ["MKV", "MP4", "MOV", "WEBM", "AVI", "GIF"])
@pytest.mark.parametrize("encoder", ["H264", "H265", "VP9", "FFV1"])
def test_node_parameters_all_container_encoder_pairs(tmp_path, container, encoder):
    # The node's audio mux follows INTENDED_MUX (#3331); the rest equals installed.
    results = []
    for kind in ("intended", "native"):
        env = FakeFFmpeg()
        g = load(kind, "save_video", env)
        c = g["save_video_node"](
            **node_args(
                g,
                Context(),
                tmp_path,
                container=g["VideoFormat"][container],
                encoder=g["VideoEncoder"][encoder],
                audio=FakeStream(env, "audio"),
            )
        )
        c.on_iterate(np.zeros((2, 2, 3), np.float32))
        g["os"] = mock_os(env)
        result = outcome(c.on_complete)
        results.append((result, env.events.copy()))
    assert results[0] == results[1]


@pytest.mark.parametrize("container", ["MP4", "GIF"])
@pytest.mark.parametrize(
    "additional",
    [
        None,
        "-vf hflip",
        "-colorspace bt709",
        "-color_range pc -filter:v vflip",
        "-filter hflip",
        "-filter_complex hflip",
        "-lavfi hflip",
        "-colorspace rgb",
        "-colorspace bt2020nc",
        "-color_primaries bt2020",
        "-pix_fmt bgr0",
    ],
)
@pytest.mark.parametrize(
    "height,width", [(480, 720), (576, 720), (534, 1280), (578, 1024)]
)
def test_colour_options_follow_size_and_user_tags(
    tmp_path, container, additional, height, width
):
    # (i): BT.601 for SD, BT.709 for HD, all four tags; user tags win; RGB output keeps
    # FFmpeg's own conversion. (k): GIF gets a palette per frame after the user's filter
    # and no tags; a user pixel format keeps FFmpeg's conversion.
    results = []
    for kind in ("intended", "native"):
        env = FakeFFmpeg()
        g = load(kind, "save_video", env)
        c = g["save_video_node"](
            **node_args(
                g,
                Context(),
                tmp_path,
                container=g["VideoFormat"][container],
                encoder=g["VideoEncoder"]["H264"],
                additional_parameters=additional,
            )
        )
        c.on_iterate(np.zeros((height, width, 3), np.float32))
        output = dict(next(e for e in env.events if e[0] == "output")[2])
        results.append((output, env.events.copy()))
    assert results[0] == results[1]
    output = results[1][0]
    assert "filter" not in output and "filter:v" not in output
    if additional in ("-filter_complex hflip", "-lavfi hflip"):
        # FFmpeg refuses a -vf beside a complex filtergraph: its own conversion stays.
        assert "vf" not in output and "color_primaries" not in output
        assert output["pix_fmt"] == "yuv420p"
        return
    if container == "GIF" and additional != "-pix_fmt bgr0":
        assert "color_trc" not in output
        assert output["pix_fmt"] == "pal8"
        user_filter = {
            "-vf hflip": "hflip,",
            "-color_range pc -filter:v vflip": "vflip,",
            "-filter hflip": "hflip,",
        }
        assert output["vf"] == user_filter.get(str(additional), "") + (
            "split[a][b];[a]palettegen=stats_mode=single[p];[b][p]paletteuse=new=1"
        )
        return
    if container == "GIF" or additional == "-pix_fmt bgr0":
        assert "color_primaries" not in output
        assert not output.get("vf", "").endswith("full_chroma_int")
        return
    hd = width >= 1280 or height > 576
    expected = "bt709" if hd or additional == "-colorspace bt709" else "smpte170m"
    if additional == "-colorspace bt2020nc":
        # A user matrix picks its own family's missing tags, whatever the size.
        assert (output["color_primaries"], output["color_trc"]) == ("bt2020", "bt709")
    elif additional in ("-colorspace rgb", "-color_primaries bt2020"):
        # No family to take them from: none is added.
        assert len({"colorspace", "color_primaries", "color_trc"} & output.keys()) == 1
    else:
        assert output["colorspace"] == expected
        assert {"color_primaries", "color_trc"} <= output.keys()
    assert "color_range" in output
    # (p): the matrix, primaries and transfer go on the frames too, after the conversion.
    keys = ("colorspace", "color_primaries", "color_trc")
    tags = [f"{k}={output[k]}" for k in keys if k in output and output[k] != "rgb"]
    suffix = ",setparams=" + ":".join(tags) if tags else ""
    assert output["vf"].endswith("flags=accurate_rnd+full_chroma_int" + suffix)


def test_writer_context_abort_no_mux_and_reaps_after_pipe_error(tmp_path):
    env = FakeFFmpeg()
    g = load("native", "save_video", env)
    context = Context()
    c = g["save_video_node"](
        **node_args(g, context, tmp_path, audio=FakeStream(env, "audio"))
    )
    assert not env.processes
    c.on_iterate(np.zeros((2, 2, 3), np.float32))
    env.failures["stdin.close"] = BrokenPipeError("closed during abort")
    with pytest.raises(BrokenPipeError):
        context.cleanup()
    assert env.processes[0].returncode is not None
    assert not any(e[0] == "run" for e in env.events)
    assert any(e[0] == "wait" for e in env.events)
    env.failures.clear()
    env.processes[0].stdin.close()


# A limit reads as the Range 0..limit - 1 (call_load_video), Out an upper bound as the
# limit was. Limits below 1 (which the UI never allowed) make In after Out, an error now:
# test_a_range_s_frames_and_errors.
@pytest.mark.parametrize(
    "use_limit,limit", [(False, 2), (True, 1), (True, 2), (True, 3), (True, 10)]
)
def test_loader_node_index_limit_audio_and_supplier_reuse(use_limit, limit):
    results = []
    for kind in ("installed", "native"):
        env = FakeFFmpeg(chunks=[bytes([i] * 12) for i in range(3)])
        vg = load(kind, "video", env)
        g = load(kind, "load_video", env)
        link_loader(g, vg)
        context = Context()
        generated, directory, name, fps, audio = call_load_video(
            kind, g, context, Path("folder/test.mkv"), use_limit, limit
        )
        expected_length = 4
        if use_limit:
            # The migrated Range is 0..limit - 1, so Out - In + 1 is limit.
            expected_length = limit if kind == "native" else min(limit, 4)
        assert generated.expected_length == expected_length
        first = [installed_item(x) for x in generated.supplier()]
        second = [installed_item(x) for x in generated.supplier()]
        assert snap(first) == snap(second)
        results.append(
            (
                vars(installed_metadata(generated.metadata)),
                snap(first),
                directory,
                name,
                fps,
                audio.label,
            )
        )
        context.cleanup()
        assert all(p.returncode is not None for p in env.processes)
    assert results[0] == results[1]


def frames_read(items):
    return [(item[0].tobytes()[0], item[1]) for item in items]


def past_the_end(n, count):
    return (
        "error",
        ("ValueError", f"frame {n} is past the end (the video has {count} frames)"),
    )


# The probe counts 4 frames and the pipe gives 3, as the CLI's renumbered frames can.
@pytest.mark.parametrize(
    "first,last,length,expected",
    [
        # Index counts from In, the item's number in the video.
        (1, 2, 2, ("ok", ((1, 1), (2, 2)))),
        (1, 1, 1, ("ok", ((1, 1),))),
        # Checked at the first item, so the node still reports its Frame Count.
        (2, 1, 0, ("error", ("ValueError", "In (frame 2) is after Out (frame 1)."))),
        # Out is an upper bound, as the limit was: past the probe's count, or past the
        # pipe's end, the video is read to its end; requested length is an upper bound.
        (0, 9, 10, ("ok", ((0, 0), (1, 1), (2, 2)))),
        (1, 3, 3, ("ok", ((1, 1), (2, 2)))),
        # In past the end fails, naming the frames there are.
        (3, 5, 3, past_the_end(3, 3)),
        (5, 6, 2, past_the_end(5, 3)),
    ],
)
def test_a_range_s_frames_and_errors(first, last, length, expected):
    env = FakeFFmpeg(chunks=[bytes([i] * 12) for i in range(3)])
    vg = load("native", "video", env)
    g = load("native", "load_video", env)
    link_loader(g, vg)
    context = Context()
    outputs = run_load_video(g, context, Path("folder/test.mkv"), first, last)
    generated = outputs[0]
    assert outputs[-1] == 4  # Frame Count: the probe's
    assert generated.expected_length == length
    assert generated.index_base == first
    assert generated.chosen_item == (first if first == last else None)
    assert outcome(lambda: frames_read(generated.supplier())) == expected
    context.cleanup()
    assert all(p.returncode is not None for p in env.processes)


def test_a_range_s_context_frames_are_clipped_by_eof_not_an_estimate():
    # The probe's estimate is 4. All requested lookahead remains available, and
    # the actual pipe ends after 3 frames without error.
    env = FakeFFmpeg(chunks=[bytes([i] * 12) for i in range(3)])
    vg = load("native", "video", env)
    g = load("native", "load_video", env)
    link_loader(g, vg)
    context = Context()
    generated = run_load_video(g, context, Path("folder/test.mkv"), 1, 2)[0]
    assert generated.context is not None
    items, before, after = generated.context(5, 5)
    assert (before, after) == (1, 5)
    assert frames_read(items) == [(0, 0), (1, 1), (2, 2)]
    # In past the pipe's end, behind context frames: the In error when they end.
    generated = run_load_video(g, context, Path("folder/test.mkv"), 3, 5)[0]
    items, before, after = generated.context(2, 0)
    assert (before, after) == (2, 0)
    assert outcome(lambda: frames_read(items)) == past_the_end(3, 3)
    # All reads every frame: no context frames.
    whole = run_load_video(g, context, Path("folder/test.mkv"))[0]
    assert whole.index_base == 0 and whole.chosen_item is None
    assert whole.context is not None
    items, before, after = whole.context(5, 5)
    assert (before, after) == (0, 0)
    context.cleanup()
    assert all(p.returncode is not None for p in env.processes)


@pytest.mark.parametrize("first,last", [(95, 99), (100, 104)])
def test_underestimated_count_keeps_range_and_temporal_lookahead(first, last):
    probe = valid_probe()
    probe["streams"][1]["nb_frames"] = "100"
    env = FakeFFmpeg(probe=probe, chunks=[bytes([i] * 12) for i in range(150)])
    vg = load("native", "video", env)
    g = load("native", "load_video", env)
    link_loader(g, vg)
    context = Context()
    generated = run_load_video(g, context, Path("folder/test.mkv"), first, last)[0]
    assert generated.expected_length == 5
    assert generated.context is not None
    items, before, after = generated.context(3, 3)
    assert (before, after) == (3, 3)
    assert [item[1] for item in items] == list(range(first - 3, last + 4))
    context.cleanup()
    assert all(p.returncode is not None for p in env.processes)


def test_parallel_independent_readers():
    def run(index):
        env = FakeFFmpeg(chunks=[bytes([index] * 12)] * 3)
        g = load("native", "video", env)
        reader = g["VideoLoader"](
            Path("x.mkv"), types.SimpleNamespace(ffmpeg="f", ffprobe="p")
        )
        result = list(reader.stream_frames())
        assert all(p.returncode is not None for p in env.processes)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(run, range(12)))
    for i, images in enumerate(outputs):
        assert len(images) == 3
        assert all(image.tobytes() == bytes([i] * 12) for image in images)


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def real_module(kind, component):
    g = load(kind, component)
    g.update(
        ffmpeg=ffmpeg,
        os=os,
        subprocess=subprocess,
        Popen=subprocess.Popen,
        atomic_target=real_target,
    )
    return g


def decode(kind, path):
    g = real_module(kind, "video")
    reader = g["VideoLoader"](
        path, types.SimpleNamespace(ffmpeg=FFMPEG, ffprobe=FFPROBE)
    )
    return reader.metadata, list(reader.stream_frames())


@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="CPU FFmpeg/ffprobe unavailable")
def test_real_cpu_lossless_decode_encode_and_audio(tmp_path):
    assert FFMPEG is not None and FFPROBE is not None
    frames = [
        np.arange(16 * 18 * 3, dtype=np.uint8).reshape(16, 18, 3),
        np.full((16, 18, 3), 73, np.uint8),
        np.flip(np.arange(16 * 18 * 3, dtype=np.uint8).reshape(16, 18, 3), 1),
    ]
    outputs = []
    owned = []
    try:
        for kind in ("installed", "native"):
            g = real_module(kind, "save_video")
            path = tmp_path / (kind + ".mkv")
            w = writer(g)
            w.save_path = str(path)
            w.ffmpeg_env = types.SimpleNamespace(ffmpeg=FFMPEG)
            w.output_params = {
                "filename": str(path),
                "vcodec": "ffv1",
                "pix_fmt": "bgr0",
                "r": 3,
                "threads": 1,
            }
            if kind == "native":  # (m): FFmpeg writes the target's temporary file.
                w.target = real_target(str(path))
                w.output_params.update(filename=w.target.temp, format="matroska")
            w.fps = 3
            for frame in frames:
                w.write_frame(frame.astype(np.float32) / 255)
                if w.out not in owned:
                    owned.append(w.out)
            w.close()
            if kind == "native":  # FFmpeg's stderr pipe is closed once drained.
                assert w.out.stderr.closed
            assert w.out.poll() == 0
            before = decode("installed", path)
            native = decode("native", path)
            assert vars(before[0]) == vars(native[0])
            assert (
                [x.tobytes() for x in before[1]]
                == [x.tobytes() for x in native[1]]
                == [x.tobytes() for x in frames]
            )
            outputs.append((vars(before[0]), [x.tobytes() for x in before[1]]))
        assert outputs[0] == outputs[1]
        # Owned synthetic PCM source: no device capture, network, GPU, or user file.
        audio_path = tmp_path / "tone.wav"
        command = [
            FFMPEG,
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=8000:duration=1",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(audio_path),
        ]
        subprocess.run(
            command, check=True, stdin=subprocess.DEVNULL, capture_output=True
        )
        # The audio mux follows INTENDED_MUX (#3331): video first, PATH-free.
        muxed = []
        for kind, video in (("intended", "installed"), ("native", "native")):
            g = real_module(kind, "save_video")
            path = tmp_path / (kind + "-audio.mkv")
            w = writer(g, audio=ffmpeg.input(str(audio_path)).audio)
            w.save_path = str(path)
            # (m): the video FFmpeg wrote is the target's temporary file.
            w.target = real_target(str(path))
            shutil.copyfile(tmp_path / (video + ".mkv"), w.target.temp)
            w.ffmpeg_env = types.SimpleNamespace(ffmpeg=FFMPEG, ffprobe=FFPROBE)
            w.close()
            probe = ffmpeg.probe(str(path), cmd=FFPROBE)
            streams = [
                (
                    x["codec_type"],
                    x["codec_name"],
                    x.get("width"),
                    x.get("height"),
                    x.get("sample_rate"),
                )
                for x in probe["streams"]
            ]
            raw_audio, _ = (
                ffmpeg.input(str(path))
                .output("pipe:", format="s16le", acodec="pcm_s16le", loglevel="error")
                .run(cmd=FFMPEG, capture_stdout=True, capture_stderr=True)
            )
            muxed.append(
                (streams, raw_audio, [x.tobytes() for x in decode("native", path)[1]])
            )
            assert not list(tmp_path.glob("*.chainner-c-*.tmp"))
        assert muxed[0] == muxed[1]
        assert [x[0] for x in muxed[0][0]] == ["video", "audio"]
    finally:
        for process in owned:
            if process.poll() is None:
                process.terminate()
            process.wait()
        gc.collect()
    assert all(p.poll() is not None for p in owned)


@pytest.mark.skipif(not FFMPEG, reason="CPU FFmpeg unavailable")
def test_real_encoder_that_exits_early_is_named_with_its_message(tmp_path):
    # x264 rejects the preset when its encoder opens, at the first frame, and FFmpeg
    # exits; later writes fail on the dead pipe (upstream chaiNNer #3109).
    g = real_module("native", "save_video")
    w = writer(g, encoder="H264")
    w.ffmpeg_env = types.SimpleNamespace(ffmpeg=FFMPEG)
    w.save_path = str(tmp_path / "early.mkv")
    w.output_params = {
        "filename": w.save_path,
        "vcodec": "libx264",
        "preset": "no-such-preset",
    }
    try:
        with pytest.raises(RuntimeError, match=r"FFmpeg stopped") as raised:
            for _ in range(64):
                w.write_frame(np.zeros((256, 256, 3), np.float32))
            w.close()
        message = str(raised.value)
        assert f"(exit code {w.out.returncode})" in message
        assert w.out.returncode != 0
        assert "no-such-preset" in message
        assert isinstance(raised.value.__cause__, OSError)
    finally:
        w._native_video_resource.abort()


@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="CPU FFmpeg/ffprobe unavailable")
@pytest.mark.parametrize("container", ["MKV", "GIF"])
@pytest.mark.parametrize("key", ["filter", "filter_complex", "lavfi"])
def test_real_user_filter_options_reach_ffmpeg(tmp_path, container, key):
    # A user -filter beside the colour -vf was dropped (FFmpeg keeps the last), and a
    # user -filter_complex/-lavfi beside it made FFmpeg exit; both work as in upstream.
    assert FFPROBE is not None
    g = real_module("native", "save_video")
    w = writer(g, container=container, encoder="H264" if container == "MKV" else None)
    w.ffmpeg_env = types.SimpleNamespace(ffmpeg=FFMPEG)
    w.save_path = str(tmp_path / ("user." + container.lower()))
    w.output_params = {
        "filename": w.save_path,
        "vcodec": "libx264" if container == "MKV" else "gif",
        "pix_fmt": "yuv420p",
        key: "scale=32:16",
    }
    try:
        for _ in range(4):
            w.write_frame(np.full((64, 64, 3), 0.5, np.float32))
        w.close()
    finally:
        w._native_video_resource.abort()
    size = subprocess.run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
        ]
        + [w.save_path],
        capture_output=True,
        check=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    assert size.stdout.strip() == "32,16"


@pytest.mark.parametrize("exists", [False, True])
def test_missing_audio_video_exists_policy(exists):
    # INTENDED_MUX (#3331): no video file, so no audio to lose; upstream's error log.
    results = []
    for kind in ("intended", "native"):
        env = FakeFFmpeg()
        g = load(kind, "save_video", env)
        g["os"] = mock_os(env, exists=exists)
        results.append(
            (
                outcome(writer(g, audio=FakeStream(env, "audio")).close),
                env.events.copy(),
            )
        )
    assert results[0] == results[1]


@pytest.mark.parametrize("setting", ["AUTO", "COPY", "TRANSCODE"])
def test_audio_cleanup_nested_errors_and_ffmpeg_stderr(setting):
    # Captured stderr=None, so the message is str(error); a cleanup failure is
    # logged and the mux's own error still reaches the caller (INTENDED_MUX, #3331).
    results = []
    for kind in ("intended", "native"):
        env = FakeFFmpeg(
            failures={
                "run": ffmpeg.Error("ffmpeg", None, None),
                "remove": PermissionError("locked"),
            }
        )
        g = load(kind, "save_video", env)
        g["os"] = mock_os(env)
        results.append(
            (
                outcome(
                    writer(
                        g, audio=FakeStream(env, "audio"), audio_settings=setting
                    ).close
                ),
                env.events.copy(),
            )
        )
    assert results[0] == results[1]


def test_reader_exit_failure_still_reaps_owned_child():
    env = FakeFFmpeg(
        chunks=[bytes(12)], failures={"exit": OSError("context close failed")}
    )
    g = load("native", "video", env)
    obj = g["VideoLoader"](
        Path("input.mkv"),
        types.SimpleNamespace(ffmpeg="reader-bin", ffprobe="probe-bin"),
    )
    with pytest.raises(OSError, match="context close failed"):
        list(obj.stream_frames())
    assert env.processes[0].returncode is not None
    assert env.processes[0].stdout.closed
    assert any(e[0] == "terminate" for e in env.events)


def test_context_manager_suppression_and_closed_generator(monkeypatch):
    # The FFmpeg CLI reader's own semantics: keep PyAV's in-process reader out.
    monkeypatch.setenv("CHAINNER_C_VIDEO_READER", "cli")
    results = []
    for kind in ("installed", "native"):
        env = FakeFFmpeg(chunks=[b"1"])
        env.suppress = True
        g = load(kind, "video", env)
        obj = g["VideoLoader"](
            Path("x"), types.SimpleNamespace(ffmpeg="f", ffprobe="p")
        )
        it = obj.stream_frames()
        results.append(
            (
                outcome(lambda it=it: list(it)),
                outcome(lambda it=it: next(it)),
                env.events.copy(),
            )
        )
    assert results[0] == results[1]


def test_writer_zero_frames_and_abort_before_start(tmp_path):
    env = FakeFFmpeg()
    g = load("native", "save_video", env)
    context = Context()
    c = g["save_video_node"](**node_args(g, context, tmp_path))
    c.on_complete()
    context.cleanup()
    assert env.events == []
    assert not env.processes


def test_writer_external_process_not_owned():
    env = FakeFFmpeg()
    g = load("native", "save_video", env)
    external = FakeProcess(env)
    w = writer(g, out=external)
    # A direct caller-supplied process is closed only by explicit Writer.close,
    # never by native owned-process resource cleanup/destruction.
    graph().video_writer_start(g, w, 2, 2)
    graph().video_writer_close(g, writer(g))
    del w
    gc.collect()
    assert external.returncode is None
    assert not any(e[0] == "terminate" for e in env.events)
    external.__exit__(None, None, None)


def test_writer_duplicate_log_level_keyword_original_error():
    results = []
    for kind in ("installed", "native"):
        env = FakeFFmpeg()
        g = load(kind, "save_video", env)
        w = writer(g)
        w.output_params["loglevel"] = "quiet"
        events = env.events
        results.append((outcome(lambda w=w: w.start(2, 2)), events.copy()))
    # (l) This repository pipes RGB as gbrp16le; the rest equals installed.
    pipe = ("pix_fmt", "gbrp16le")
    native = [
        (*event[:2], tuple(("pix_fmt", "bgr24") if x == pipe else x for x in event[2]))
        if event[0] == "input"
        else event
        for event in results[1][1]
    ]
    assert results[0] == (results[1][0], native)


def test_reader_supplier_creation_remains_lazy():
    env = FakeFFmpeg(chunks=[bytes(12)])
    vg = load("native", "video", env)
    g = load("native", "load_video", env)
    link_loader(g, vg)
    calls = []
    original = vg["VideoLoader"].stream_frames

    def tracked(loader):
        calls.append("stream_frames")
        return original(loader)

    vg["VideoLoader"].stream_frames = tracked
    context = Context()
    gen = call_load_video("native", g, context, Path("x"), False, 1)[0]
    unused = gen.supplier()
    assert not calls
    unused.close()
    it = gen.supplier()
    assert not calls
    next(it)
    assert calls == ["stream_frames"]
    context.cleanup()
    assert all(p.returncode is not None for p in env.processes)


def test_loader_declares_no_per_item_source_files():
    env = FakeFFmpeg(chunks=[bytes(12)])
    vg = load("native", "video", env)
    g = load("native", "load_video", env)
    link_loader(g, vg)
    context = Context()
    assert (
        call_load_video("native", g, context, Path("x"), False, 1)[0].source_paths == ()
    )
    context.cleanup()


REAL_ENCODERS = [
    ("MKV", "H264"),
    ("MKV", "H265"),
    ("MKV", "VP9"),
    ("MKV", "FFV1"),
    ("MP4", "H264"),
    ("MP4", "H265"),
    ("MP4", "VP9"),
    ("MOV", "H264"),
    ("MOV", "H265"),
    ("WEBM", "VP9"),
    ("AVI", "H264"),
    ("GIF", "H264"),
]


@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="CPU FFmpeg/ffprobe unavailable")
@pytest.mark.parametrize("container,encoder", REAL_ENCODERS)
def test_real_cpu_all_exposed_encoder_container_pairs(tmp_path, container, encoder):
    images = [
        np.arange(24 * 32 * 3, dtype=np.float32).reshape(24, 32, 3) % 255 / 255,
        np.full((24, 32, 3), 0.5, np.float32),
    ]
    outputs = []
    for kind in ("intended", "native"):
        directory = tmp_path / kind
        g = real_module(kind, "save_video")
        g["FFMpegEnv"] = types.SimpleNamespace(
            get_integrated=lambda _: types.SimpleNamespace(ffmpeg=FFMPEG)
        )
        context = Context()
        params = "-threads 1"
        if encoder == "H265":
            params += " -x265-params pools=none:frame-threads=1:log-level=error"
        c = g["save_video_node"](
            **node_args(
                g,
                context,
                directory,
                container=g["VideoFormat"][container],
                encoder=g["VideoEncoder"][encoder],
                video_preset=g["VideoPreset"].ULTRA_FAST,
                fps=2,
                additional_parameters=params,
            )
        )
        try:
            for image in images:
                c.on_iterate(image)
            c.on_complete()
        finally:
            context.cleanup()
        path = directory / ("clip." + container.lower())
        metadata, decoded = decode("native", path)
        outputs.append((vars(metadata), [x.tobytes() for x in decoded]))
        assert len(decoded) == 2 if container != "GIF" else len(decoded) > 0
    if container == "GIF":
        # (k): the per-frame palette departs from the installed reference's bgr8; the
        # colours are judged in backend/tests/test_save_video_colours.py.
        assert outputs[0][0] == outputs[1][0]
        assert len(outputs[0][1]) == len(outputs[1][1])
        return
    assert outputs[0] == outputs[1]


@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="CPU FFmpeg/ffprobe unavailable")
def test_real_cpu_abandoned_reader_and_writer_owned_processes(tmp_path, monkeypatch):
    # The FFmpeg CLI reader's own semantics: keep PyAV's in-process reader out.
    monkeypatch.setenv("CHAINNER_C_VIDEO_READER", "cli")
    owned = []
    original_popen = subprocess.Popen

    def record(*args, **kwargs):
        p = original_popen(*args, **kwargs)
        owned.append(p)
        return p

    # Patch only ffmpeg's process creation entrypoint while retaining the real
    # Popen type used by loader assertions. No unrelated process is touched.
    monkeypatch.setattr(
        ffmpeg._run,
        "subprocess",
        types.SimpleNamespace(Popen=record, PIPE=subprocess.PIPE),
    )
    source = tmp_path / "source.mkv"
    g = real_module("native", "save_video")
    w = writer(g)
    w.save_path = str(source)
    w.ffmpeg_env = types.SimpleNamespace(ffmpeg=FFMPEG)
    w.output_params = {
        "filename": str(source),
        "vcodec": "ffv1",
        "pix_fmt": "bgr0",
        "threads": 1,
    }
    try:
        for i in range(20):
            w.write_frame(np.full((128, 128, 3), i / 20, np.float32))
        w.close()
        vg = real_module("native", "video")
        loader = vg["VideoLoader"](
            source, types.SimpleNamespace(ffmpeg=FFMPEG, ffprobe=FFPROBE)
        )
        it = loader.stream_frames()
        frame = next(it)
        it.close()
        assert frame.shape == (128, 128, 3)
        assert all(p.poll() is not None for p in owned)
        aborted = writer(g, audio=ffmpeg.input(str(source)).audio)
        aborted.save_path = str(tmp_path / "aborted.mkv")
        aborted.output_params = {
            "filename": aborted.save_path,
            "vcodec": "ffv1",
            "threads": 1,
        }
        aborted.ffmpeg_env = types.SimpleNamespace(ffmpeg=FFMPEG)
        aborted.write_frame(np.zeros((32, 32, 3), np.float32))
        aborted._native_video_resource.abort()
        assert not (tmp_path / "aborted_av.mkv").exists()
        assert all(p.poll() is not None for p in owned)
    finally:
        for p in owned:
            if p.poll() is None:
                p.terminate()
            p.wait()
    assert owned


def test_writer_empty_close_followed_by_first_frame_preserves_owned_lifecycle():
    results = []
    for kind in ("intended", "native"):
        env = FakeFFmpeg()
        g = load(kind, "save_video", env)
        w = writer(g)
        w.close()
        w.write_frame(np.zeros((2, 2, 3), np.float32))
        w.close()
        assert all(p.returncode is not None for p in env.processes)
        results.append(env.events.copy())
    assert results[0] == results[1]


@pytest.mark.parametrize("started", [False, True])
def test_aborted_collector_never_muxes_even_if_completed_later(tmp_path, started):
    env = FakeFFmpeg()
    g = load("native", "save_video", env)
    g["os"] = mock_os(env)
    context = Context()
    collector = g["save_video_node"](
        **node_args(g, context, tmp_path, audio=FakeStream(env, "audio"))
    )
    if started:
        collector.on_iterate(np.zeros((2, 2, 3), np.float32))
    context.cleanup()
    collector.on_complete()
    assert not any(e[0] == "run" for e in env.events)
    assert all(p.returncode is not None for p in env.processes)
    if not started:
        with pytest.raises(RuntimeError, match="Video writer was aborted"):
            collector.on_iterate(np.zeros((2, 2, 3), np.float32))
        assert not env.processes


@pytest.mark.parametrize("simple", ["MP4_H264", "MP4_H265", "WEBM", "GIF"])
def test_simple_node_overrides_and_gif_audio_suppression(tmp_path, simple):
    def run(kind):
        env = FakeFFmpeg()
        g = load(kind, "save_video", env)
        captured = []
        original = g["Writer"]

        def capture(**kwargs):
            captured.append(snap(installed_writer_arguments(kwargs)))
            return original(**kwargs)

        g["Writer"] = capture
        g["save_video_node"](
            **node_args(
                g,
                Context(),
                tmp_path,
                simplicity=g["Simplicity"].SIMPLE,
                simple_video_format=g["SimpleVideoFormat"][simple],
                quality=96,
                audio=FakeStream(env, "audio"),
                additional_parameters="-filename forbidden",
            )
        )
        return captured

    results = []
    for kind in ("installed", "native"):
        results.append(run(kind))
    assert results[0] == results[1]


def test_writer_nonzero_exit_code_is_an_error():
    # Owner-approved 2026-10-09: a non-zero exit at a normal close fails the run with
    # FFmpeg's exit code before the audio mux, where upstream leaves it unchecked and
    # muxes; (m) the temporary file FFmpeg wrote is abandoned, the path left untouched.
    results = []
    for kind in ("intended", "native"):
        env = FakeFFmpeg()
        g = load(kind, "save_video", env)
        g["os"] = mock_os(env)
        p = FakeProcess(env)
        p.returncode = 23
        w = writer(g, out=p, audio=FakeStream(env, "audio"))
        results.append((outcome(w.close), env.events.copy()))
    assert results[0][0] == ("ok", None)
    assert any(e[0] == "run" for e in results[0][1])
    assert results[1][0] == (
        "error",
        (
            "RuntimeError",
            "FFmpeg stopped while Save Video was writing the video (exit code 23)",
        ),
    )
    # The same flush and wait, then the error: no mux, no commit, the target abandoned.
    *flushed, abandoned = results[1][1]
    assert flushed == results[0][1][: len(flushed)]
    assert abandoned == ("abandon", "owned/video.mkv.0.tmp")
    assert not any(e[0] in ("run", "commit") for e in results[1][1])


def test_reader_remaining_frames_metadata_can_differ():
    env = FakeFFmpeg(chunks=[bytes([i] * 12) for i in range(6)])
    env.probe_result["streams"][1]["nb_frames"] = "1"
    g = load("native", "video", env)
    loader = g["VideoLoader"](Path("x"), types.SimpleNamespace(ffmpeg="f", ffprobe="p"))
    assert loader.metadata.frame_count == 1
    assert len(list(loader.stream_frames())) == 6


def test_reader_reentrant_next_and_close_match_python_generator():
    def run(kind):
        env = FakeFFmpeg(chunks=[bytes(12)])
        g = load(kind, "video", env)
        loader = g["VideoLoader"](
            Path("x"), types.SimpleNamespace(ffmpeg="f", ffprobe="p")
        )
        it = loader.stream_frames()
        old_record = env.record
        reentrant = []

        def record(name, *args):
            if name == "read":
                reentrant.append(outcome(lambda: next(it)))
                reentrant.append(outcome(it.close))
            return old_record(name, *args)

        env.record = record
        assert len(list(it)) == 1
        return reentrant

    results = []
    for kind in ("installed", "native"):
        results.append(run(kind))
    assert results[0] == results[1]


@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="CPU FFmpeg/ffprobe unavailable")
@pytest.mark.parametrize(
    "container,encoder,setting,channels",
    [
        ("MKV", "FFV1", "TRANSCODE", 2),
        ("WEBM", "VP9", "AUTO", 2),
        ("WEBM", "VP9", "TRANSCODE", 2),
        ("WEBM", "VP9", "AUTO", 1),
    ],
)
def test_real_cpu_audio_transcode_policies(
    tmp_path, container, encoder, setting, channels
):
    import wave

    assert FFMPEG is not None and FFPROBE is not None
    audio_path = tmp_path / "owned.wav"
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(np.repeat(np.arange(8000, dtype=np.int16), channels).tobytes())
    # The audio mux follows INTENDED_MUX (#3331): video first.
    outputs = []
    for kind in ("intended", "native"):
        g = real_module(kind, "save_video")
        g["FFMpegEnv"] = types.SimpleNamespace(
            get_integrated=lambda _: types.SimpleNamespace(
                ffmpeg=FFMPEG, ffprobe=FFPROBE
            )
        )
        context = Context()
        c = g["save_video_node"](
            **node_args(
                g,
                context,
                tmp_path / kind,
                container=g["VideoFormat"][container],
                encoder=g["VideoEncoder"][encoder],
                fps=2,
                additional_parameters="-threads 1",
                audio=ffmpeg.input(str(audio_path)).audio,
                audio_settings=g["AudioSettings"][setting],
            )
        )
        try:
            for value in (0.25, 0.75):
                c.on_iterate(np.full((24, 32, 3), value, np.float32))
            result = outcome(c.on_complete)
        finally:
            context.cleanup()
        path = tmp_path / kind / ("clip." + container.lower())
        probe = ffmpeg.probe(str(path), cmd=FFPROBE)
        streams = [
            (
                x["codec_type"],
                x["codec_name"],
                x.get("sample_rate"),
                x.get("r_frame_rate"),
            )
            for x in probe["streams"]
        ]
        # Mono into WebM works: 256 kb/s Opus, libopus's ceiling for one channel.
        assert result == ("ok", None)
        assert [item[0] for item in streams] == ["video", "audio"]
        audio, _ = (
            ffmpeg.input(str(path))
            .output("pipe:", format="s16le", acodec="pcm_s16le", loglevel="error")
            .run(cmd=FFMPEG, capture_stdout=True, capture_stderr=True)
        )
        assert not path.with_name(path.stem + "_av" + path.suffix).exists()
        outputs.append(
            (
                result[0],
                streams,
                audio,
                [x.tobytes() for x in decode("native", path)[1]],
            )
        )
    assert outputs[0] == outputs[1]


def test_indexed_reader_reentrant_close_does_not_poison_iteration():
    def run(kind):
        env = FakeFFmpeg(chunks=[bytes(12), bytes([1] * 12)])
        vg = load(kind, "video", env)
        g = load(kind, "load_video", env)
        link_loader(g, vg)
        context = Context()
        it = call_load_video(kind, g, context, Path("x"), False, 1)[0].supplier()
        old_record = env.record
        reentrant = []

        def record(name, *args):
            if name == "read":
                reentrant.append(outcome(lambda: next(it)))
                reentrant.append(outcome(it.close))
            return old_record(name, *args)

        env.record = record
        result = (snap([installed_item(x) for x in it]), reentrant)
        context.cleanup()
        return result

    results = []
    for kind in ("installed", "native"):
        results.append(run(kind))
    assert results[0] == results[1]


@pytest.mark.parametrize("field", ["output_params", "global_params"])
@pytest.mark.parametrize("value", [None, 2, [], [("filename", "x")], {1: "bad"}, "str"])
def test_writer_keyword_and_star_parameter_protocol(field, value):
    results = []
    for kind in ("intended", "native"):
        env = FakeFFmpeg()
        g = load(kind, "save_video", env)
        w = writer(g)
        setattr(w, field, value)
        result = outcome(lambda w=w: w.start(2, 2))
        results.append((result, env.events.copy()))
        if w.out is not None:
            w.close()
    assert results[0] == results[1]


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("phase", ["eof", "close", "limit"])
@pytest.mark.parametrize("event", ["exit", "stdout.close", "wait"])
def test_reader_cleanup_reentry_remains_running(indexed, phase, event):
    def run(kind):
        env = FakeFFmpeg(chunks=[bytes(12), bytes([1] * 12)])
        vg = load(kind, "video", env)
        context = Context()
        if indexed:
            g = load(kind, "load_video", env)
            link_loader(g, vg)
            it = call_load_video(kind, g, context, Path("x"), phase == "limit", 1)[
                0
            ].supplier()
        else:
            it = vg["VideoLoader"](
                Path("x"), types.SimpleNamespace(ffmpeg="f", ffprobe="p")
            ).stream_frames()
        reentry = []
        old_record = env.record

        def record(name, *args):
            if name == event:
                reentry.append(outcome(lambda: next(it)))
                reentry.append(outcome(it.close))
            return old_record(name, *args)

        env.record = record
        if phase == "close":
            next(it)
            it.close()
        else:
            list(it)
        # Explicit cancellation now terminates rather than entering Popen.__exit__;
        # compare the reentry events that exist in both ownership policies.
        if (
            kind == "native"
            and (phase == "close" or (indexed and phase == "limit"))
            and event == "exit"
        ):
            assert not reentry
        else:
            # CPython 3.14's gen_close finishes a generator suspended at a yield
            # outside any try block before clearing its frame, so closing the indexed
            # supplier releases the reader with the supplier already finished; any
            # other cleanup runs while its generator is still executing.
            pair = (
                [("error", ("StopIteration", "")), ("ok", None)]
                if indexed and phase == "close"
                else [("error", ("ValueError", "generator already executing"))] * 2
            )
            assert reentry and reentry == pair * (len(reentry) // 2)
        result = outcome(lambda: next(it))
        context.cleanup()
        return result

    results = []
    for kind in ("installed", "native"):
        results.append(run(kind))
    assert results[0] == results[1]


@pytest.mark.skipif(not FFMPEG, reason="CPU FFmpeg unavailable")
def test_abort_ends_the_encoder_and_deletes_only_its_temporary_file(tmp_path):
    # Stop (the chain's cleanup aborts the writer): FFmpeg still gets the buffered EOF and
    # exits normally, as the installed writer's abandonment let it, and never muxes; (m)
    # the video it finished is the temporary file, which is deleted, so the path never
    # appears. Foreign files, other processes' temporary names included, stay.
    image = np.arange(24 * 32 * 3, dtype=np.float32).reshape(24, 32, 3) % 255 / 255
    path = tmp_path / "native.mkv"
    foreign = {
        tmp_path / "native.mkv.chainner-c-1-0.tmp": b"a",
        tmp_path / "x.tmp": b"b",
    }
    for name, data in foreign.items():
        name.write_bytes(data)

    class Spy:
        """The target, noting whether its file existed when abandoned."""

        def __init__(self, target):
            self.target, self.temp, self.existed = target, target.temp, None

        def abandon(self):
            self.existed = Path(self.temp).is_file()
            self.target.abandon()

    g = real_module("native", "save_video")
    w = writer(g, audio=ffmpeg.input("never-open-this-audio-on-abort.wav").audio)
    w.save_path = str(path)
    w.target = Spy(real_target(str(path)))
    w.ffmpeg_env = types.SimpleNamespace(ffmpeg=FFMPEG)
    w.output_params = {
        "filename": w.target.temp,
        "format": "matroska",
        "vcodec": "ffv1",
        "pix_fmt": "bgr0",
        "r": 3,
        "threads": 1,
    }
    w.fps = 3
    try:
        w.write_frame(image)
        w._native_video_resource.abort()
        w.close()  # Aborted completion must still never start muxing.
        assert w.out.poll() == 0
        assert w.target.existed is True
        assert {p: p.read_bytes() for p in tmp_path.iterdir()} == foreign
    finally:
        if w.out is not None and w.out.poll() is None:
            w.out.kill()
            w.out.wait(timeout=10)


@pytest.mark.parametrize("terminate_fails", [False, True])
def test_writer_abnormal_wait_falls_back_to_termination_and_joins(terminate_fails):
    env = FakeFFmpeg()
    g = load("native", "save_video", env)
    w = writer(g, audio=FakeStream(env, "audio"))
    w.write_frame(np.zeros((2, 2, 3), np.float32))
    p = w.out
    original_wait = p.wait

    def stalled_wait(*args, **kwargs):
        if p.returncode is None:
            raise subprocess.TimeoutExpired("owned", kwargs.get("timeout", 0))
        return original_wait(*args, **kwargs)

    p.wait = stalled_wait
    if terminate_fails:

        def denied():
            raise OSError("owned terminate denied")

        p.terminate = denied
        with pytest.raises(OSError, match="owned terminate denied"):
            w._native_video_resource.abort()
    else:
        w._native_video_resource.abort()
    assert p.stdin.closed
    assert p.returncode is not None
    assert not any(e[0] == "run" for e in env.events)
    expected = env.events.copy()
    w._native_video_resource.abort()
    assert env.events == expected


@pytest.mark.parametrize(
    "denied",
    [
        "none",
        "terminate",
        pytest.param(
            "both",
            marks=pytest.mark.skipif(
                sys.platform != "win32",
                reason="with both stops refused, Windows joins the close through "
                "CancelSynchronousIo, which POSIX has no counterpart of",
            ),
        ),
    ],
)
def test_real_stalled_pipe_close_is_joined_even_when_termination_errors(
    denied, monkeypatch
):
    # This child owns no user resource and deliberately does not drain stdin.
    # Its 7200-byte pending BufferedWriter flush exceeds the anonymous pipe's
    # capacity, exercising the native close deadline, not an encoder benchmark.
    p = subprocess.Popen(
        [sys.executable, "-B", "-c", "import time; time.sleep(60)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    original_kill = p.kill
    original_pipe = p.stdin
    assert original_pipe is not None
    if sys.platform == "linux":
        import fcntl

        # A Windows anonymous pipe's 4 KiB: Linux's 64 KiB would take the flush whole
        fcntl.fcntl(original_pipe.fileno(), fcntl.F_SETPIPE_SZ, 4096)
    finished = threading.Event()
    entered = threading.Event()

    def close():
        entered.set()
        try:
            original_pipe.close()
        finally:
            finished.set()

    monkeypatch.setattr(
        p, "stdin", types.SimpleNamespace(write=original_pipe.write, close=close)
    )

    def terminate_denied():
        raise OSError("owned termination denied")

    if denied in {"terminate", "both"}:
        p.terminate = terminate_denied
    if denied == "both":
        p.kill = terminate_denied
    try:
        env = FakeFFmpeg()
        env.process_override = p
        g = load("native", "save_video", env)
        w = writer(g, audio=FakeStream(env, "audio"))
        w.write_frame(np.zeros((2, 1200, 3), np.float32))
        with pytest.raises(
            OSError, match="owned termination denied" if denied != "none" else None
        ):
            w._native_video_resource.abort()
        assert entered.is_set() and finished.is_set()
        assert original_pipe.closed
        assert not any(e[0] == "run" for e in env.events)
        if denied != "both":
            assert p.poll() is not None
        else:
            # Both public stop operations were deliberately rejected. The
            # close worker still joined via CancelSynchronousIo; report the
            # error, and let this fixture's original owned handle stop its child.
            assert p.poll() is None
    finally:
        original_kill()
        p.wait(timeout=10)
        original_pipe.close()


def test_finalization_cleanup_avoids_creating_gil_acquiring_worker(monkeypatch):
    env = FakeFFmpeg()
    g = load("native", "save_video", env)
    w = writer(g)
    w.write_frame(np.zeros((2, 2, 3), np.float32))
    calls = []
    original_close = w.out.stdin.close

    def close():
        calls.append(threading.get_ident())
        original_close()

    w.out.stdin.close = close
    with monkeypatch.context() as patch:
        patch.setattr(sys, "is_finalizing", lambda: True)
        w._native_video_resource.abort()
    assert calls == [threading.get_ident()]
    assert any(event[0] == "terminate" for event in env.events)
    assert w.out.returncode is not None


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="the close cancels a stalled write with CancelSynchronousIo on Windows; "
    "elsewhere ending the reader breaks the pipe",
)
def test_stalled_pipe_cancellation_covers_io_start_after_first_cancel(monkeypatch):
    # Hold the close before its real OS write until both public stop calls fail.
    # The initial cancellation then finds no pending synchronous I/O. A fixture
    # watchdog rescues the old one-shot implementation without leaving a child
    # or close thread behind; successful cleanup must never need that rescue.
    p = subprocess.Popen(
        [sys.executable, "-B", "-c", "import time; time.sleep(60)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    original_kill, original_wait, original_pipe = p.kill, p.wait, p.stdin
    assert original_pipe is not None
    enter_io, finished, rescued, stop_watchdog = (threading.Event() for _ in range(4))
    timers = []

    def close():
        assert enter_io.wait(20), "fixture did not release the real pipe flush"
        try:
            original_pipe.close()
        finally:
            finished.set()

    def rescue():
        if not stop_watchdog.wait(3):
            rescued.set()
            original_kill()

    def denied():
        raise OSError("owned termination denied")

    def kill_denied():
        # The event delay is fault injection, not a latency assertion. Until
        # released, CancelSynchronousIo cannot find a WriteFile to cancel.
        timer = threading.Timer(0.1, enter_io.set)
        timer.start()
        timers.append(timer)
        watchdog = threading.Thread(target=rescue)
        watchdog.start()
        timers.append(watchdog)
        denied()

    def wait(*args, **kwargs):
        if p.poll() is None:
            raise subprocess.TimeoutExpired("owned", kwargs.get("timeout", 0))
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(
        p, "stdin", types.SimpleNamespace(write=original_pipe.write, close=close)
    )
    p.terminate, p.kill, p.wait = denied, kill_denied, wait
    try:
        env = FakeFFmpeg()
        env.process_override = p
        g = load("native", "save_video", env)
        w = writer(g)
        w.write_frame(np.zeros((2, 1200, 3), np.float32))
        with pytest.raises(OSError, match="owned termination denied"):
            w._native_video_resource.abort()
        stop_watchdog.set()
        assert finished.is_set() and original_pipe.closed
        assert not rescued.is_set(), "one-shot cancellation missed later pipe I/O"
        assert p.poll() is None
    finally:
        stop_watchdog.set()
        enter_io.set()
        original_kill()
        original_wait(timeout=10)
        for thread in timers:
            thread.join(timeout=5)
            assert not thread.is_alive()
        original_pipe.close()


def test_prepared_frames_write_todays_payload():
    # A prepared frame (SP3-P9) is only the payload: start-up on the first frame and
    # every pipe write are today's, for a first and a later frame.
    from nodes.impl.item_window import Prepared

    frames = [
        np.linspace(0, 1, 12, dtype=np.float32).reshape(2, 2, 3),
        np.full((2, 2, 3), 0.25, np.float32),
    ]

    def run(prepared):
        env = FakeFFmpeg()
        g = load("native", "save_video", env)
        converted = []
        payload = g["rgb_payload"]
        g["rgb_payload"] = lambda value: converted.append(value) or payload(value)
        w = writer(g)
        holders = []
        for frame in frames:
            if prepared:
                holders.append(Prepared(payload(frame)))
                w.write_frame(frame, holders[-1])
            else:
                w.write_frame(frame)
        w.close()
        assert len(converted) == (0 if prepared else 2)
        assert all(holder.used for holder in holders)
        return env.events.copy()

    results = []
    for prepared in (False, True):
        results.append(run(prepared))
    assert results[0] == results[1]


@pytest.mark.parametrize(
    ("mode", "prepared"),
    [
        ("rgb", False),
        ("rgb", True),
        ("rgb8", False),
        ("rgb8", True),
        ("y", False),
        ("u", False),
        ("v", False),
    ],
)
def test_writer_rejects_changed_geometry_before_writing(tmp_path, mode, prepared):
    from nodes.impl.item_window import Prepared

    env = FakeFFmpeg()
    g = load("native", "save_video", env)
    context = Context()
    rgb = mode in {"rgb", "rgb8"}
    collector = g["save_video_node"](
        **node_args(
            g,
            context,
            tmp_path,
            mode=g["VideoInputMode"].RGB if rgb else g["VideoInputMode"].PLANES,
            additional_parameters="-pix_fmt bgr0" if mode == "rgb8" else None,
        )
    )
    if rgb:
        first = [np.zeros((2, 4, 3), np.float32), None, None, None]
        changed = [np.zeros((4, 2, 3), np.float32), None, None, None]
    else:
        first = [None, np.zeros((4, 8), np.float32)]
        first += [np.zeros((2, 4), np.float32) for _ in range(2)]
        changed = first.copy()
        index = {"y": 1, "u": 2, "v": 3}[mode]
        plane = first[index]
        assert plane is not None
        changed[index] = plane.T.copy()
    holder = None
    try:
        if prepared:
            holder = Prepared(g["prepare_video_frame"](collector, first))
            g["commit_video_frame"](collector, first, holder)
            holder = Prepared(g["prepare_video_frame"](collector, changed))
        else:
            collector.on_iterate(tuple(first))
        writes = [event for event in env.events if event[0] == "write"]
        assert len(writes) == 1
        with pytest.raises(ValueError, match="changed size"):
            if prepared:
                assert holder is not None
                g["commit_video_frame"](collector, changed, holder)
            else:
                collector.on_iterate(tuple(changed))
        assert [event for event in env.events if event[0] == "write"] == writes
    finally:
        context.cleanup()
    assert any(event[0] == "abandon" for event in env.events)
    assert not any(event[0] in {"run", "commit"} for event in env.events)


@pytest.mark.parametrize(
    ("container", "additional", "pipe"),
    [
        ("MKV", None, "gbrp16le"),
        ("GIF", None, "bgr24"),  # the palette's input
        ("MKV", "-pix_fmt bgr0", "bgr24"),  # 8-bit RGB output
    ],
)
def test_prepared_collector_commit_writes_as_on_iterate(
    tmp_path, container, additional, pipe
):
    # Save Video's collector carries its writer: its prepare makes the payload in the
    # writer's pipe format, the commit writes it through the writer as on_iterate
    # writes the frame, and on_iterate keeps its one-frame contract.
    from nodes.impl.item_window import Prepared

    frame = np.linspace(0, 1, 12, dtype=np.float32).reshape(2, 2, 3)
    results = []
    for prepared in (False, True):
        env = FakeFFmpeg()
        g = load("native", "save_video", env)
        arguments = node_args(
            g,
            Context(),
            tmp_path,
            container=g["VideoFormat"][container],
            additional_parameters=additional,
        )
        collector = g["save_video_node"](**arguments)
        assert isinstance(collector, g["VideoCollector"])
        assert isinstance(collector.writer, g["Writer"])
        assert collector.writer.pipe_format == pipe
        item = (frame, None, None, None)  # Image Sequence, Y, U, V
        holder = Prepared(g["prepare_video_frame"](collector, list(item)))
        assert isinstance(holder.value, np.ndarray)
        assert holder.value.dtype == (np.uint8 if pipe == "bgr24" else np.uint16)
        with pytest.raises(TypeError):
            collector.on_iterate(item, holder)  # one argument, as Collector declares
        if prepared:
            g["commit_video_frame"](collector, list(item), holder)
        else:
            collector.on_iterate(item)
        results.append(env.events.copy())
        collector.on_complete()
        assert holder.used == prepared
    assert results[0] == results[1]


def test_planes_mode_prepares_nothing(tmp_path):
    # Planes are coded as the first frame's commit decides, so the node runs whole,
    # also with an image left connected.
    g = load("native", "save_video", FakeFFmpeg())
    arguments = node_args(g, Context(), tmp_path, mode=g["VideoInputMode"].PLANES)
    collector = g["save_video_node"](**arguments)
    plane, image = np.full((2, 2), 0.5, np.float32), np.zeros((2, 2, 3), np.float32)
    for first in (None, image):
        assert g["prepare_video_frame"](collector, [first, plane, plane, plane]) is None
