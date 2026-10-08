"""Frozen installed video contracts and task-owned CPU FFmpeg fixtures.

Reference AST execution is test-only. Production contains no reference evaluator.
Native cleanup deliberately terminates abandoned owned children, and successful
writer completion/mux is idempotent; comparisons keep these corrections explicit.
"""

from __future__ import annotations

import ast
import copy
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
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import ffmpeg
import numpy as np
import pytest

from nodes.impl.image_utils import to_uint8
from nodes.impl.native_graph import graph
from nodes.utils.utils import get_h_w_c, split_file_path

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

    @property
    def audio(self):
        self.env.record("audio", self)
        return FakeStream(self.env, "audio")

    def __str__(self):
        return self.label

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

    def __init__(self):
        self.cleanups = []

    def add_cleanup(self, callback, after="chain"):
        assert after == "chain"
        self.cleanups.append(callback)

    def cleanup(self):
        callbacks, self.cleanups = self.cleanups, []
        for callback in callbacks:
            callback()


def load(kind, component, env=None):
    rel = "nodes/impl/video.py" if component == "video" else NODE + component + ".py"
    path = (
        ROOT / "backend/src" / rel
        if kind.startswith("native")
        else REFERENCE / kind / rel
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
    tree = ast.parse(path.read_text(encoding="utf-8"))
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
    return g


def writer(
    g,
    *,
    container="MKV",
    encoder: str | None = "FFV1",
    audio=None,
    audio_settings="AUTO",
    out=None,
):
    return g["Writer"](
        container=g["VideoFormat"][container],
        encoder=None if encoder is None else g["VideoEncoder"][encoder],
        fps=30,
        audio=audio,
        audio_settings=g["AudioSettings"][audio_settings],
        save_path="owned/video." + container.lower(),
        output_params={"filename": "owned/video.mkv", "vcodec": "ffv1"},
        global_params=["-nostdin"],
        ffmpeg_env=types.SimpleNamespace(ffmpeg="integrated-ffmpeg"),
        out=out,
    )


def mock_os(env, exists=True):
    def remove(path):
        env.record("remove", path)

    def rename(a, b):
        env.record("rename", a, b)

    def path_exists(path):
        env.record("exists", path)
        return exists

    return types.SimpleNamespace(
        path=types.SimpleNamespace(splitext=os.path.splitext, exists=path_exists),
        remove=remove,
        rename=rename,
    )


def test_reference_hashes_and_registration_contracts():
    for entry in json.loads((REFERENCE / "manifest.json").read_text())["files"]:
        assert (
            hashlib.sha256((REFERENCE / entry["snapshot"]).read_bytes()).hexdigest()
            == entry["sha256"]
        )
    for name in ("load_video", "save_video"):
        old = ast.parse((REFERENCE / "installed" / (NODE + name + ".py")).read_text())
        new = ast.parse((ROOT / "backend/src" / (NODE + name + ".py")).read_text())
        for tree in (old, new):
            node = next(
                x
                for x in tree.body
                if isinstance(x, ast.FunctionDef) and x.name == name + "_node"
            )
            node.body = [ast.Pass()]
            if tree is old:
                expected = ast.dump(node)
            else:
                assert ast.dump(node) == expected


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


class CountingRaw(io.FileIO):
    """A pipe's raw reader that records what each raw read returned."""

    def __init__(self, fd):
        super().__init__(fd, "rb")
        self.counts = []
        self.returned = threading.Event()

    def readinto(self, buffer):
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


@pytest.mark.parametrize("container", ["MKV", "MP4", "MOV", "WEBM", "AVI", "GIF"])
@pytest.mark.parametrize("audio_setting", ["AUTO", "COPY", "TRANSCODE"])
def test_audio_policies_exact(container, audio_setting):
    results = []
    for kind in ("installed", "native"):
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


@pytest.mark.parametrize(
    "step", ["input", "output", "overwrite", "run", "remove", "rename"]
)
@pytest.mark.parametrize(
    "error", [OSError("failure"), ffmpeg.Error("ffmpeg", b"out", b"error details")]
)
def test_audio_errors_exact(step, error):
    results = []
    for kind in ("installed", "native"):
        env = FakeFFmpeg(failures={step: error})
        g = load(kind, "save_video", env)
        g["os"] = mock_os(env)
        w = writer(g, audio=FakeStream(env, "audio"))
        results.append((outcome(w.close), env.events.copy()))
    assert results[0] == results[1]


@pytest.mark.parametrize("encoder", [None, "H264", "H265", "VP9", "FFV1"])
@pytest.mark.parametrize(
    "shape", [(2, 2, 3), (1, 2, 3), (2, 1, 3), (0, 2, 3), (2, 2), (2, 2, 4)]
)
def test_writer_frames_and_even_contract(encoder, shape):
    results = []
    for kind in ("installed", "native"):
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
    for kind in ("installed", "native"):
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
    for kind in ("installed", "native"):
        env = FakeFFmpeg(failures={step: BrokenPipeError(step)})
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
            captured.append(
                snap({k: v for k, v in kwargs.items() if k != "ffmpeg_env"})
            )
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
    results = []
    for kind in ("installed", "native"):
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


@pytest.mark.parametrize(
    "use_limit,limit",
    [(False, 2), (True, 0), (True, -1), (True, 1), (True, 2), (True, 10)],
)
def test_loader_node_index_limit_audio_and_supplier_reuse(use_limit, limit):
    results = []
    for kind in ("installed", "native"):
        env = FakeFFmpeg(chunks=[bytes([i] * 12) for i in range(3)])
        vg = load(kind, "video", env)
        g = load(kind, "load_video", env)
        g["VideoLoader"], g["VideoMetadata"] = vg["VideoLoader"], vg["VideoMetadata"]
        context = Context()
        generated, directory, name, fps, audio = g["load_video_node"](
            context, Path("folder/test.mkv"), use_limit, limit
        )
        first = list(generated.supplier())
        second = list(generated.supplier())
        assert snap(first) == snap(second)
        results.append(
            (
                generated.expected_length,
                vars(generated.metadata),
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
    g.update(ffmpeg=ffmpeg, os=os, subprocess=subprocess, Popen=subprocess.Popen)
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
            w.fps = 3
            for frame in frames:
                w.write_frame(frame.astype(np.float32) / 255)
                if w.out not in owned:
                    owned.append(w.out)
            w.close()
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
        muxed = []
        for kind in ("installed", "native"):
            g = real_module(kind, "save_video")
            path = tmp_path / (kind + "-audio.mkv")
            shutil.copyfile(tmp_path / (kind + ".mkv"), path)
            w = writer(g, audio=ffmpeg.input(str(audio_path)).audio)
            w.save_path = str(path)
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
            assert not path.with_name(path.stem + "_av.mkv").exists()
        assert muxed[0] == muxed[1]
        assert [x[0] for x in muxed[0][0]] == ["audio", "video"]
    finally:
        for process in owned:
            if process.poll() is None:
                process.terminate()
            process.wait()
        gc.collect()
    assert all(p.poll() is not None for p in owned)


@pytest.mark.parametrize("exists", [False, True])
def test_missing_audio_video_exists_policy(exists):
    results = []
    for kind in ("installed", "native"):
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
    # Captured stderr=None; the installed policy catches only once. Retain its
    # nested error policy and cleanup exception handling.
    results = []
    for kind in ("installed", "native"):
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
    graph().video_writer_close_installed(g, writer(g))
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
        results.append((outcome(lambda w=w: w.start(2, 2)), env.events.copy()))
    assert results[0] == results[1]


def test_reader_supplier_creation_remains_lazy():
    env = FakeFFmpeg(chunks=[bytes(12)])
    vg = load("native", "video", env)
    g = load("native", "load_video", env)
    g["VideoLoader"], g["VideoMetadata"] = vg["VideoLoader"], vg["VideoMetadata"]
    calls = []
    original = vg["VideoLoader"].stream_frames

    def tracked(loader):
        calls.append("stream_frames")
        return original(loader)

    vg["VideoLoader"].stream_frames = tracked
    context = Context()
    gen = g["load_video_node"](context, Path("x"), False, 1)[0]
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
    g["VideoLoader"], g["VideoMetadata"] = vg["VideoLoader"], vg["VideoMetadata"]
    context = Context()
    assert g["load_video_node"](context, Path("x"), False, 1)[0].source_paths == ()
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
    for kind in ("installed", "native"):
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
    for kind in ("installed", "native"):
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
            captured.append(
                snap({k: v for k, v in kwargs.items() if k != "ffmpeg_env"})
            )
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


def test_writer_nonzero_exit_code_original_policy():
    results = []
    for kind in ("installed", "native"):
        env = FakeFFmpeg()
        g = load(kind, "save_video", env)
        p = FakeProcess(env)
        p.returncode = 23
        results.append((outcome(writer(g, out=p).close), env.events.copy()))
    assert results[0] == results[1]


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
    outputs = []
    for kind in ("installed", "native"):
        g = real_module(kind, "save_video")
        g["FFMpegEnv"] = types.SimpleNamespace(
            get_integrated=lambda _: types.SimpleNamespace(ffmpeg=FFMPEG)
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
            c.on_complete()
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
        if channels == 1 and container == "WEBM":
            # The installed fixed 320 kb/s Opus setting rejects mono input.
            # Both paths must catch mux failure and preserve video without audio.
            assert [item[0] for item in streams] == ["video"]
            audio = None
        else:
            assert [item[0] for item in streams] == ["audio", "video"]
            audio, _ = (
                ffmpeg.input(str(path))
                .output("pipe:", format="s16le", acodec="pcm_s16le", loglevel="error")
                .run(cmd=FFMPEG, capture_stdout=True, capture_stderr=True)
            )
        assert not path.with_name(path.stem + "_av" + path.suffix).exists()
        outputs.append(
            (streams, audio, [x.tobytes() for x in decode("native", path)[1]])
        )
    assert outputs[0] == outputs[1]


def test_indexed_reader_reentrant_close_does_not_poison_iteration():
    def run(kind):
        env = FakeFFmpeg(chunks=[bytes(12), bytes([1] * 12)])
        vg = load(kind, "video", env)
        g = load(kind, "load_video", env)
        g["VideoLoader"], g["VideoMetadata"] = vg["VideoLoader"], vg["VideoMetadata"]
        context = Context()
        it = g["load_video_node"](context, Path("x"), False, 1)[0].supplier()
        old_record = env.record
        reentrant = []

        def record(name, *args):
            if name == "read":
                reentrant.append(outcome(lambda: next(it)))
                reentrant.append(outcome(it.close))
            return old_record(name, *args)

        env.record = record
        result = (snap(list(it)), reentrant)
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
    for kind in ("installed", "native"):
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
            g["VideoLoader"], g["VideoMetadata"] = (
                vg["VideoLoader"],
                vg["VideoMetadata"],
            )
            it = g["load_video_node"](context, Path("x"), phase == "limit", 1)[
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


@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="CPU FFmpeg/ffprobe unavailable")
def test_abort_preserves_original_buffered_eof_partial_video_without_audio(tmp_path):
    images = [np.arange(24 * 32 * 3, dtype=np.float32).reshape(24, 32, 3) % 255 / 255]
    outputs = []
    for kind in ("installed", "native"):
        g = real_module(kind, "save_video")
        path = tmp_path / (kind + ".mkv")
        w = writer(g, audio=ffmpeg.input("never-open-this-audio-on-abort.wav").audio)
        w.save_path = str(path)
        w.ffmpeg_env = types.SimpleNamespace(ffmpeg=FFMPEG)
        w.output_params = {
            "filename": str(path),
            "vcodec": "ffv1",
            "pix_fmt": "bgr0",
            "r": 3,
            "threads": 1,
        }
        w.fps = 3
        try:
            w.write_frame(images[0])
            if kind == "installed":
                # Windows reference writer abandonment releases its buffered
                # pipe via EOF; do not call Writer.close(), which would mux.
                w.out.stdin.close()
                w.out.wait(timeout=10)
            else:
                w._native_video_resource.abort()
                w.close()  # Aborted completion must still never start muxing.
            assert w.out.poll() == 0
            assert path.is_file()
            metadata, frames = decode("installed", path)
            outputs.append((vars(metadata), [f.tobytes() for f in frames]))
            assert len(frames) == 1
            assert not path.with_name(path.stem + "_av.mkv").exists()
        finally:
            if w.out is not None and w.out.poll() is None:
                w.out.kill()
                w.out.wait(timeout=10)
    assert outputs[0] == outputs[1]


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


@pytest.mark.parametrize("denied", ["none", "terminate", "both"])
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
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    original_kill = p.kill
    original_pipe = p.stdin
    assert original_pipe is not None
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
        creationflags=subprocess.CREATE_NO_WINDOW,
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
    # A prepared uint8 frame (SP3-P9) is only the payload: start-up on the first frame
    # and every pipe write are today's, for a first and a later frame.
    from nodes.impl.item_window import Prepared

    frames = [
        np.linspace(0, 1, 12, dtype=np.float32).reshape(2, 2, 3),
        np.full((2, 2, 3), 0.25, np.float32),
    ]

    def run(prepared):
        env = FakeFFmpeg()
        g = load("native", "save_video", env)
        converted = []
        g["to_uint8"] = lambda value, **options: (
            converted.append(value) or to_uint8(value, **options)
        )
        w = writer(g)
        holders = []
        for frame in frames:
            if prepared:
                holders.append(Prepared(to_uint8(frame, normalized=True)))
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


def test_prepared_collector_commit_writes_as_on_iterate(tmp_path):
    # Save Video's collector carries its writer: the commit writes a prepared frame
    # through it, and on_iterate keeps its one-frame contract.
    from nodes.impl.item_window import Prepared

    frame = np.linspace(0, 1, 12, dtype=np.float32).reshape(2, 2, 3)
    results = []
    for prepared in (False, True):
        env = FakeFFmpeg()
        g = load("native", "save_video", env)
        collector = g["save_video_node"](**node_args(g, Context(), tmp_path))
        assert isinstance(collector, g["VideoCollector"])
        assert isinstance(collector.writer, g["Writer"])
        holder = Prepared(to_uint8(frame, normalized=True))
        with pytest.raises(TypeError):
            collector.on_iterate(frame, holder)  # one argument, as Collector declares
        if prepared:
            g["commit_video_frame"](collector, [frame], holder)
        else:
            collector.on_iterate(frame)
        results.append(env.events.copy())
        collector.on_complete()
        assert holder.used == prepared
    assert results[0] == results[1]
