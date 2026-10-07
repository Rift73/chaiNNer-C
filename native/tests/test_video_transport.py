"""Video byte transport: allocation ownership, short writes and CPU parity."""

from __future__ import annotations

import gc
import io
import threading
import types
import weakref
from concurrent.futures import ThreadPoolExecutor

import ffmpeg
import numpy as np
import pytest
from test_video_io import (
    FFMPEG,
    FFPROBE,
    FakeFFmpeg,
    decode,
    load,
    real_module,
    to_uint8,
    writer,
)


class Sink(io.BufferedIOBase):
    def __init__(self, callback):
        self.callback = callback

    def write(self, data):
        return self.callback(data)


def make_writer(callback):
    env = FakeFFmpeg()
    module = load("native", "save_video", env)
    result = writer(module, out=types.SimpleNamespace(stdin=Sink(callback)))
    return module, result


@pytest.mark.parametrize(
    "layout", ["contiguous", "readonly", "reverse", "fortran", "unaligned"]
)
@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
def test_writer_owns_readonly_zero_copy_quantized_buffer(layout, dtype):
    image = np.arange(24 * 32 * 3, dtype=dtype).reshape(24, 32, 3)
    if np.issubdtype(dtype, np.floating):
        image /= 255
        image[0, 0] = [np.nan, np.inf, -np.inf]
    if layout == "readonly":
        image.flags.writeable = False
    elif layout == "reverse":
        image = image[::-1, ::-1]
    elif layout == "fortran":
        image = np.asfortranarray(image)
    elif layout == "unaligned":
        foreign = np.ndarray(
            image.shape, dtype=dtype, buffer=bytearray(image.nbytes + 1), offset=1
        )
        foreign[:] = image
        image = foreign
    with np.errstate(invalid="ignore"):
        expected = to_uint8(image, normalized=True).tobytes()
    before = image.tobytes()
    owner = []
    observed = []

    def quantize(value, *, normalized):
        result = to_uint8(value, normalized=normalized)
        owner.append(weakref.ref(result))
        return result

    def consume(data):
        if owner[-1]().flags.c_contiguous:
            assert isinstance(data, memoryview)
            assert data.ndim == 1 and data.format == "B" and data.readonly
            assert np.shares_memory(np.frombuffer(data, np.uint8), owner[-1]())
        else:
            assert isinstance(data, bytes)
        gc.collect()
        assert owner[-1]() is not None
        if isinstance(data, memoryview):
            with pytest.raises(TypeError):
                data[0] = 0
        observed.append(bytes(data))
        return len(data)

    module, result = make_writer(consume)
    module["to_uint8"] = quantize
    with np.errstate(invalid="ignore"):
        result.write_frame(image)
    assert observed == [expected]
    assert len(owner) == 1
    gc.collect()
    assert owner[0]() is None
    assert image.tobytes() == before


@pytest.mark.parametrize("chunks", [[1], [2, 3, 5], [11, 1], [12], [4, 4, 4]])
def test_short_writes_deliver_exact_frame_once(chunks):
    accepted = bytearray()
    calls = []
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)

    def consume(data):
        count = min(chunks[len(calls) % len(chunks)], len(data))
        calls.append(bytes(data))
        accepted.extend(data[:count])
        return count

    _, result = make_writer(consume)
    result.write_frame(image)
    assert accepted == image.tobytes()
    assert sum(len(item) for item in calls) >= len(accepted)
    assert calls[-1] == image.tobytes()[-len(calls[-1]) :]


@pytest.mark.parametrize(
    "returned,error",
    [
        (None, BlockingIOError),
        (0, BlockingIOError),
        (-1, OSError),
        (13, OSError),
        (2**100, OverflowError),
        (1.5, TypeError),
    ],
)
def test_invalid_write_result_fails_without_retry_spin(returned, error):
    calls = []

    def consume(data):
        calls.append(bytes(data))
        return returned

    _, result = make_writer(consume)
    with pytest.raises(error):
        result.write_frame(np.zeros((2, 2, 3), np.float32))
    assert len(calls) == 1


def test_exception_after_partial_write_is_preserved():
    failure = BrokenPipeError("owned transport failure")
    calls = []

    def consume(data):
        calls.append(bytes(data))
        if len(calls) == 1:
            return 5
        raise failure

    _, result = make_writer(consume)
    with pytest.raises(BrokenPipeError) as caught:
        result.write_frame(np.arange(12, dtype=np.uint8).reshape(2, 2, 3))
    assert caught.value is failure
    assert calls == [bytes(range(12)), bytes(range(5, 12))]


@pytest.mark.parametrize("kind", ["subclass", "strided", "fortran", "empty"])
def test_legacy_custom_and_noncontiguous_quantizer_bytes_are_preserved(kind):
    class Custom(np.ndarray):
        def tobytes(self, *args, **kwargs):
            calls.append("tobytes")
            return b"custom payload"

    calls = []
    quantized = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    if kind == "subclass":
        quantized = quantized.view(Custom)
        expected = b"custom payload"
    elif kind == "strided":
        quantized = quantized[:, ::2]
        expected = quantized.tobytes()
    elif kind == "fortran":
        quantized = np.asfortranarray(quantized)
        expected = quantized.tobytes()
    else:
        quantized = quantized[:0]
        expected = b""
    received = []

    def consume(data):
        assert isinstance(data, bytes)
        received.append(data)
        return len(data)

    module, result = make_writer(consume)
    module["to_uint8"] = lambda *args, **kwargs: quantized
    result.write_frame(np.zeros((2, 2, 3), np.float32))
    assert received == [expected]
    assert calls == (["tobytes"] if kind == "subclass" else [])


def test_sequential_writer_calls_can_move_between_workers():
    thread_ids, received = [], []

    def consume(data):
        thread_ids.append(threading.get_ident())
        received.append(bytes(data))
        return len(data)

    _, result = make_writer(consume)
    with ThreadPoolExecutor(max_workers=1) as a, ThreadPoolExecutor(max_workers=1) as b:
        a.submit(result.write_frame, np.full((2, 2, 3), 0.25, np.float32)).result()
        b.submit(result.write_frame, np.full((2, 2, 3), 0.75, np.float32)).result()
    assert thread_ids[0] != thread_ids[1]
    assert received == [bytes([64] * 12), bytes([191] * 12)]


def test_independent_writers_concurrent_strided_frames():
    def run(index):
        image = np.full((128, 256, 3), index / 15, np.float32)[:, ::-1]
        image.flags.writeable = False
        received = []

        def consume(data):
            received.append(bytes(data))
            return len(data)

        _, result = make_writer(consume)
        result.write_frame(image)
        assert received == [to_uint8(image, normalized=True).tobytes()]
        assert not image.flags.writeable
        return index

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(run, range(16))) == list(range(16))


def test_reader_view_keeps_each_immutable_frame_alive_without_array_copy():
    from pathlib import Path

    chunks = [bytes(range(12)), bytes(range(12, 24))]
    env = FakeFFmpeg(chunks=chunks)
    module = load("native", "video", env)
    loader = module["VideoLoader"](
        Path("owned.mkv"), types.SimpleNamespace(ffmpeg="f", ffprobe="p")
    )
    frames = list(loader.stream_frames())
    assert len(frames) == 2
    # D8 (SP4c): each frame views the fresh buffer it was read into, no longer the
    # read's bytes; there is still no array copy after the read. Every frame is checked
    # after all were read: a live frame keeps its content and its own buffer.
    for frame, source in zip(frames, chunks, strict=True):
        assert np.shares_memory(frame, frame.base)
        assert frame.base.flags.owndata
        assert (frame.base.dtype, frame.base.shape) == (np.uint8, (len(source),))
        assert not frame.flags.writeable
        assert frame.tobytes() == source
        with pytest.raises(ValueError):
            frame.setflags(write=True)
    assert not np.shares_memory(frames[0], frames[1])
    assert sum(event[0] == "probe" for event in env.events) == 1
    assert sum(event[0] == "run_async" for event in env.events) == 1
    assert all(process.poll() == 0 for process in env.processes)


@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="CPU FFmpeg unavailable")
def test_real_ffv1_transport_matches_direct_ffmpeg_settings_and_frames(tmp_path):
    frames = [
        np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3),
        np.full((24, 32, 3), 79, np.uint8),
    ]
    direct, native = tmp_path / "direct.mkv", tmp_path / "native.mkv"
    options = {"vcodec": "ffv1", "pix_fmt": "bgr0", "r": 3, "threads": 1}
    (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="bgr24",
            s="32x24",
            r=3,
            loglevel="error",
        )
        .output(str(direct), **options, loglevel="error")
        .overwrite_output()
        .global_args("-nostdin")
        .run(
            cmd=FFMPEG,
            input=b"".join(frame.tobytes() for frame in frames),
            capture_stdout=True,
            capture_stderr=True,
        )
    )
    module = real_module("native", "save_video")
    result = writer(module)
    result.save_path = str(native)
    result.ffmpeg_env = types.SimpleNamespace(ffmpeg=FFMPEG)
    result.output_params = {"filename": str(native), **options}
    result.fps = 3
    try:
        for frame in frames:
            result.write_frame(frame.astype(np.float32) / 255)
        result.close()
        assert result.out.poll() == 0
    finally:
        result._native_video_resource.abort()  # owned process cleanup
    first, second = decode("native", direct), decode("native", native)
    assert vars(first[0]) == vars(second[0])
    assert [frame.tobytes() for frame in first[1]] == [
        frame.tobytes() for frame in frames
    ]
    assert [frame.tobytes() for frame in second[1]] == [
        frame.tobytes() for frame in frames
    ]
