"""A TensorRT session gives its engine reference back when its setup fails, and its
close frees everything after a CUDA error; items share it at once, each with its own
tiles. TensorRT and cuda-python are stand-ins: no GPU is used."""

from __future__ import annotations

import ctypes
import importlib
import sys
import threading
import time
import weakref
from collections.abc import Callable
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from nodes.impl.tensorrt.model import (
    DeserializedEngine,
    TensorRTEngine,
    TensorRTEngineInfo,
)
from nodes.impl.upscale.tile_report import TileReport
from util import waited_seconds

ILLEGAL_ADDRESS = 700
OUT_OF_MEMORY = 2


class _Handle(str):
    """A stream or event handle: its name, and an integer as TensorRT takes a stream."""

    def __int__(self) -> int:
        return 0


class _Cudart(ModuleType):
    """cuda.bindings.runtime: every call succeeds and is recorded, until `fail` names
    it (call name -> successful calls before it fails with `fail_code`). Device and
    pinned memory are host buffers, and copies between them are memmoves. A device buffer
    freed with a copy on the stream fails the test. An allocation refused leaves the
    thread's last error, as CUDA's does, until cudaGetLastError takes it."""

    def __init__(self) -> None:
        super().__init__("cuda.bindings.runtime")
        self.__dict__.update(
            cudaEventDisableTiming=2,
            cudaStreamNonBlocking=1,
            cudaStreamCaptureMode=SimpleNamespace(cudaStreamCaptureModeThreadLocal=0),
            cudaError_t=SimpleNamespace(
                cudaSuccess=0, cudaErrorMemoryAllocation=OUT_OF_MEMORY
            ),
            cudaMemcpyKind=SimpleNamespace(
                cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2
            ),
        )
        self.fail: dict[str, int] = {}
        self.fail_code = ILLEGAL_ADDRESS
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.memory: list[ctypes.Array[ctypes.c_char]] = []
        # The card: a cudaMalloc past this many bytes in use finds no memory.
        self.capacity: int | None = None
        self.allocated: dict[int, int] = {}
        self.stream_busy = False
        # Called in a thread's first stream wait after its allocation found no memory.
        self.after_no_memory: Callable[[], object] | None = None
        self._thread = threading.local()

    def __getattr__(self, name: str) -> Callable[..., tuple[Any, ...]]:
        if not name.startswith("cuda"):
            raise AttributeError(name)
        return lambda *args: self._call(name, args)

    def _call(self, name: str, args: tuple[Any, ...]) -> tuple[Any, ...]:
        count = sum(1 for called, _ in self.calls if called == name)
        self.calls.append((name, args))
        created = {
            "cudaStreamCreateWithFlags": "stream",
            "cudaEventCreateWithFlags": "event",
            "cudaStreamEndCapture": "graph",
            "cudaGraphInstantiate": "graph_exec",
        }
        if name in self.fail and count >= self.fail[name]:
            return (self.fail_code, None) if name in created else (self.fail_code,)
        if name in ("cudaGetErrorName", "cudaGetErrorString"):
            if args[0] == OUT_OF_MEMORY:
                return 0, b"cudaErrorMemoryAllocation"
            return 0, b"cudaErrorIllegalAddress"
        if name in created:
            return 0, _Handle(f"{created[name]}{count}")
        if name == "cudaGetLastError":
            left = self.last_error()
            self._thread.last_error = 0
            return (left,)
        if name == "cudaStreamSynchronize":
            self.stream_busy = False
            if getattr(self._thread, "no_memory", False):
                self._thread.no_memory = False
                if self.after_no_memory is not None:
                    self.after_no_memory()
        if (
            name == "cudaMalloc"
            and self.capacity is not None
            and sum(self.allocated.values()) + args[0] > self.capacity
        ):
            self._thread.no_memory = True
            self._thread.last_error = OUT_OF_MEMORY
            return OUT_OF_MEMORY, None
        if name in ("cudaMalloc", "cudaMallocHost"):
            self.memory.append(ctypes.create_string_buffer(args[0]))
            address = ctypes.addressof(self.memory[-1])
            if name == "cudaMalloc":
                self.allocated[address] = args[0]
            return 0, address
        if name == "cudaFree":
            assert not self.stream_busy, (
                "a device buffer freed under a tile on the stream"
            )
            del self.allocated[args[0]]
        if name == "cudaMemcpyAsync":
            ctypes.memmove(args[0], args[1], args[2])
            self.stream_busy = True
        return (0,)

    def last_error(self) -> int:
        return getattr(self._thread, "last_error", 0)

    def fail_once(self, name: str, code: int, at: int = 0) -> None:
        """`name` fails once with `code`, at its `at`th call from now."""
        seen = [0]

        def call(*args: Any) -> tuple[Any, ...]:
            seen[0] += 1
            if seen[0] - 1 == at:
                self.calls.append((name, args))
                return (code,)
            return self._call(name, args)

        self.__dict__[name] = call

    def destroyed(self) -> list[str]:
        return [
            args[0]
            for name, args in self.calls
            if name in ("cudaEventDestroy", "cudaStreamDestroy")
        ]


class _Context:
    pass


class _CudaEngine:
    num_io_tensors = 2

    def __init__(self) -> None:
        self.contexts: list[weakref.ref[_Context]] = []

    def get_tensor_name(self, index: int) -> str:
        return ("input", "output")[index]

    def get_tensor_mode(self, name: str) -> str:
        return "INPUT" if name == "input" else "OUTPUT"

    def get_tensor_dtype(self, _name: str) -> str:
        return "FLOAT"

    def get_tensor_profile_shape(self, _name: str, _profile: int) -> tuple:
        return (1, 3, 64, 64), (1, 3, 128, 128), (1, 3, 256, 256)

    def create_execution_context(self, _strategy: object) -> _Context:
        context = _Context()
        self.contexts.append(weakref.ref(context))
        return context


class _Loaded(DeserializedEngine):
    """Records, at the last release, whether every context was gone by then."""

    def __init__(self, cuda_engine: _CudaEngine) -> None:
        super().__init__(object(), cuda_engine, 0)
        self.contexts_gone_first: bool | None = None

    def release(self) -> None:
        if self._refs == 1:
            self.contexts_gone_first = all(
                ref() is None for ref in self.engine.contexts
            )
        super().release()


def _fake_tensorrt() -> ModuleType:
    trt = ModuleType("tensorrt")
    severity = SimpleNamespace(INTERNAL_ERROR=0, ERROR=1, WARNING=2, INFO=3, VERBOSE=4)
    trt.__dict__.update(
        __version__="11",
        DataType=SimpleNamespace(FLOAT="FLOAT", HALF="HALF", BF16="BF16"),
        nptype=lambda _dtype: np.float32,
        ExecutionContextAllocationStrategy=SimpleNamespace(USER_MANAGED=1),
        TensorIOMode=SimpleNamespace(INPUT="INPUT", OUTPUT="OUTPUT"),
        ILogger=type("ILogger", (), {"Severity": severity}),
    )
    return trt


@pytest.fixture
def cudart() -> _Cudart:
    return _Cudart()


@pytest.fixture
def inference(monkeypatch: pytest.MonkeyPatch, cudart: _Cudart) -> ModuleType:
    cuda = ModuleType("cuda")
    bindings = ModuleType("cuda.bindings")
    cuda.__dict__["bindings"] = bindings
    bindings.__dict__["runtime"] = cudart
    for name, module in (
        ("tensorrt", _fake_tensorrt()),
        ("cuda", cuda),
        ("cuda.bindings", bindings),
        ("cuda.bindings.runtime", cudart),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    # Import against the stand-ins; undoing the patches drops these imports again.
    package = importlib.import_module("nodes.impl.tensorrt")
    for name in ("inference", "engine_info", "memory", "trt_logger"):
        monkeypatch.setitem(sys.modules, f"{package.__name__}.{name}", None)
        del sys.modules[f"{package.__name__}.{name}"]
        monkeypatch.setattr(package, name, None, raising=False)
    return importlib.import_module("nodes.impl.tensorrt.inference")


def _engine() -> tuple[TensorRTEngine, _Loaded]:
    loaded = _Loaded(_CudaEngine())
    shape = (1, 3, 64, 64)
    info = TensorRTEngineInfo(
        "fp32", 3, 3, 2, "sm_120", "11", True, shape, shape, shape
    )
    return TensorRTEngine(b"engine", info, loaded), loaded


@pytest.mark.parametrize(
    ("failing", "destroyed"),
    [
        ({"cudaStreamCreateWithFlags": 0}, []),
        # the second event fails: the first event and the stream go
        ({"cudaEventCreateWithFlags": 1}, ["event0", "stream0"]),
    ],
)
def test_a_failed_setup_gives_the_engine_back(
    inference: ModuleType,
    cudart: _Cudart,
    failing: dict[str, int],
    destroyed: list[str],
):
    engine, loaded = _engine()
    cudart.fail = failing

    with pytest.raises(RuntimeError, match="cudaErrorIllegalAddress"):
        inference.TensorRTSession(engine, 0)

    assert cudart.destroyed() == destroyed
    engine.close()  # Load Engine's own reference: the last one
    assert loaded.engine is None
    assert loaded.contexts_gone_first is True


def test_close_after_a_cuda_error_still_frees_everything(
    inference: ModuleType, cudart: _Cudart
):
    engine, loaded = _engine()
    session = inference.TensorRTSession(engine, 0)
    # a sticky error: every later call fails too
    cudart.fail = {
        "cudaStreamSynchronize": 0,
        "cudaEventDestroy": 0,
        "cudaStreamDestroy": 0,
    }

    with pytest.raises(RuntimeError, match="cudaErrorIllegalAddress"):
        session.close()

    assert cudart.destroyed() == ["event0", "event1", "stream0"]
    # the first error is the one raised: its name was read right after the sync
    first = next(
        i for i, (name, _) in enumerate(cudart.calls) if name == "cudaGetErrorName"
    )
    assert cudart.calls[first - 1][0] == "cudaStreamSynchronize"
    engine.close()
    assert loaded.engine is None
    assert loaded.contexts_gone_first is True


class _Weighting(_Context):
    """An engine's execution: 2x nearest, its output channel k times k + 1, so the
    order the planes arrive in shows in the result."""

    def __init__(self) -> None:
        self.addresses: dict[str, int] = {}
        self.shape = (1, 1, 1, 1)

    def set_input_shape(self, _name: str, shape: tuple[int, ...]) -> bool:
        self.shape = shape
        return True

    def get_tensor_shape(self, _name: str) -> tuple[int, ...]:
        n, c, h, w = self.shape
        return n, c, 2 * h, 2 * w

    def update_device_memory_size_for_shapes(self) -> int:
        return 0

    def set_tensor_address(self, name: str, address: int) -> None:
        self.addresses[name] = address

    def execute_async_v3(self, _stream: int) -> bool:
        _, c, h, w = self.shape
        source = (ctypes.c_float * (c * h * w)).from_address(self.addresses["input"])
        planes = np.frombuffer(source, np.float32).reshape(c, h, w)
        weights = np.arange(1, c + 1, dtype=np.float32)[:, None, None]
        result = planes.repeat(2, 1).repeat(2, 2) * weights
        target = (ctypes.c_float * result.size).from_address(self.addresses["output"])
        np.frombuffer(target, np.float32)[:] = result.reshape(-1)
        return True


class _WeightingEngine(_CudaEngine):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

    def get_tensor_profile_shape(self, _name: str, _profile: int) -> tuple:
        c = self.channels
        return (1, c, 64, 64), (1, c, 128, 128), (1, c, 256, 256)

    def create_execution_context(self, _strategy: object) -> _Context:
        context = _Weighting()
        self.contexts.append(weakref.ref(context))
        return context


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_planes_reach_the_engine_in_rgb_order_and_return_bgr(
    inference: ModuleType, channels: int
):
    # The engine takes and gives RGB(A) planes, as the ONNX it was built from, and as
    # ONNX's and PyTorch's nodes feed it: an RGBA engine's red and blue were swapped.
    from nodes.impl.tensorrt.tiling import tiled_upscale

    loaded = DeserializedEngine(object(), _WeightingEngine(channels), 0)
    shape = (1, channels, 64, 64)
    info = TensorRTEngineInfo(
        "fp32", channels, channels, 2, "sm_120", "11", True, shape, shape, shape
    )
    session = inference.TensorRTSession(TensorRTEngine(b"engine", info, loaded), 0)
    image = np.random.default_rng(31).random((40, 50, channels), np.float32)
    out = tiled_upscale(image, 2, session.bounds, session)
    session.close()

    # BGR(A) image -> RGB(A) planes -> channel k times k + 1 -> BGR(A) result
    order = [2, 1, 0, 3][:channels] if channels > 1 else [0]
    weights = np.empty(channels, np.float32)
    weights[order] = np.arange(1, channels + 1, dtype=np.float32)
    expected = image.repeat(2, 0).repeat(2, 1) * weights
    np.testing.assert_array_equal(out, expected)


class _Bfloat(_Weighting):
    """A bfloat16 engine's execution: 2x nearest, its output channel k bits + k."""

    def execute_async_v3(self, _stream: int) -> bool:
        _, c, h, w = self.shape
        source = (ctypes.c_uint16 * (c * h * w)).from_address(self.addresses["input"])
        planes = np.frombuffer(source, np.uint16).reshape(c, h, w)
        offsets = np.arange(c, dtype=np.uint16)[:, None, None]
        result = planes.repeat(2, 1).repeat(2, 2) + offsets
        target = (ctypes.c_uint16 * result.size).from_address(self.addresses["output"])
        np.frombuffer(target, np.uint16)[:] = result.reshape(-1)
        return True


class _BfloatEngine(_WeightingEngine):
    def get_tensor_dtype(self, _name: str) -> str:
        return "BF16"

    def create_execution_context(self, _strategy: object) -> _Context:
        context = _Bfloat()
        self.contexts.append(weakref.ref(context))
        return context


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_a_bfloat_engines_planes_are_rgb_and_round_to_even(
    inference: ModuleType, channels: int
):
    from nodes.impl.tensorrt.tiling import tiled_upscale

    loaded = DeserializedEngine(object(), _BfloatEngine(channels), 0)
    shape = (1, channels, 64, 64)
    info = TensorRTEngineInfo(
        "bf16", channels, channels, 2, "sm_120", "11", True, shape, shape, shape
    )
    session = inference.TensorRTSession(TensorRTEngine(b"engine", info, loaded), 0)
    image = np.random.default_rng(33).random((40, 50, channels), np.float32)
    out = tiled_upscale(image, 2, session.bounds, session)
    session.close()

    # to bfloat16 bits (ties to even), RGB(A) planes, bits + the plane's index, BGR(A)
    bits = image.view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    order = [2, 1, 0, 3][:channels] if channels > 1 else [0]
    offsets = np.empty(channels, np.uint32)
    offsets[order] = np.arange(channels, dtype=np.uint32)
    expected = ((rounded + offsets) << 16).repeat(2, 0).repeat(2, 1)
    np.testing.assert_array_equal(out.view(np.uint32), expected)


def _shared_session(inference: ModuleType, cudart: _Cudart) -> Any:
    """A 2x session on RGB tiles of 64 to 256 pixels, its captures failing (the stand-in
    has no CUDA graphs), which keeps it on the engine's own enqueue."""
    cudart.fail = {"cudaStreamBeginCapture": 0}
    loaded = DeserializedEngine(object(), _WeightingEngine(3), 0)
    shape = (1, 3, 64, 64)
    info = TensorRTEngineInfo(
        "fp32", 3, 3, 2, "sm_120", "11", True, shape, shape, (1, 3, 256, 256)
    )
    return inference.TensorRTSession(TensorRTEngine(b"engine", info, loaded), 0)


def _doubled(image: np.ndarray) -> np.ndarray:
    """_Weighting's output of a BGR image, without clipping."""
    return image.repeat(2, 0).repeat(2, 1) * np.array([3, 2, 1], np.float32)


def _on_thread(work: Any) -> threading.Thread:
    thread = threading.Thread(target=work)
    thread.start()
    return thread


def _assert_both_slots_free(session: Any) -> None:
    tile = np.zeros((64, 64, 3), np.float32)

    def two_tiles() -> None:
        session.submit(tile)
        session.submit(tile)
        session.collect(128, 128)
        session.collect(128, 128)

    thread = _on_thread(two_tiles)
    thread.join(10)
    assert not thread.is_alive(), "a tile kept its slot"


def test_each_thread_collects_its_own_tile(inference: ModuleType, cudart: _Cudart):
    session = _shared_session(inference, cudart)
    rng = np.random.default_rng(40)
    first, second = rng.random((2, 64, 64, 3), np.float32)
    first_in, second_done = threading.Event(), threading.Event()
    results: dict[str, np.ndarray] = {}

    def early():
        session.submit(first)
        first_in.set()
        assert second_done.wait(10)
        results["first"] = session.collect(128, 128)

    def late():
        assert first_in.wait(10)
        session.submit(second)
        results["second"] = session.collect(128, 128)
        second_done.set()

    threads = [_on_thread(early), _on_thread(late)]
    for thread in threads:
        thread.join(10)
    session.close()
    np.testing.assert_array_equal(results["first"], _doubled(first))
    np.testing.assert_array_equal(results["second"], _doubled(second))


def test_a_third_tile_waits_for_a_slot(inference: ModuleType, cudart: _Cudart):
    session = _shared_session(inference, cudart)
    tile = np.random.default_rng(41).random((64, 64, 3), np.float32)
    session.submit(tile)
    other_in, go = threading.Event(), threading.Event()

    def other():
        session.submit(tile)
        other_in.set()
        assert go.wait(10)
        session.collect(128, 128)

    third_in = threading.Event()

    def third():
        session.submit(tile)
        third_in.set()
        session.collect(128, 128)

    holder = _on_thread(other)
    assert other_in.wait(10)
    waiter = _on_thread(third)
    assert not third_in.wait(0.2)
    session.collect(128, 128)  # gives its slot back
    assert third_in.wait(10)
    go.set()
    for thread in (holder, waiter):
        thread.join(10)
        assert not thread.is_alive()
    _assert_both_slots_free(session)
    session.close()


def test_images_of_one_and_of_many_tiles_share_a_session(
    inference: ModuleType, cudart: _Cudart
):
    from nodes.impl.tensorrt.tiling import tiled_upscale

    session = _shared_session(inference, cudart)
    rng = np.random.default_rng(42)
    # One tile, one tile padded, and several tiles of two shapes.
    images = [
        rng.random(shape, np.float32)
        for shape in ((64, 64, 3), (40, 50, 3), (500, 300, 3), (256, 700, 3))
    ]
    expected = [tiled_upscale(i, 2, session.bounds, session, clip=True) for i in images]
    failures: list[object] = []

    def work(start: int) -> None:
        try:
            for i in range(start, start + 12):
                image = i % len(images)
                out = session.upscale(images[image], 2)
                if out.tobytes() != expected[image].tobytes():
                    failures.append(image)
        except Exception as e:
            failures.append(e)
            raise

    threads = [_on_thread(lambda start=start: work(start)) for start in range(6)]
    for thread in threads:
        thread.join(60)
        assert not thread.is_alive(), "the images deadlocked"
    assert failures == []
    _assert_both_slots_free(session)
    session.close()


def test_an_image_of_several_tiles_can_ask_for_balanced_tiles(
    inference: ModuleType, cudart: _Cudart
):
    from nodes.impl.tensorrt.tiling import tiled_upscale

    session = _shared_session(inference, cudart)
    image = np.random.default_rng(44).random((300, 700, 3), np.float32)
    expected = tiled_upscale(
        image, 2, session.bounds, session, clip=True, balanced=True
    )
    out = session.upscale(image, 2, balanced=True)
    assert out.tobytes() == expected.tobytes()
    assert session.context.shape == (1, 3, 192, 256)
    session.upscale(image, 2)
    assert session.context.shape == (1, 3, 256, 256)
    small = image[:64, :64]
    assert (
        session.upscale(small, 2, balanced=True).tobytes()
        == session.upscale(small, 2).tobytes()
    )
    _assert_both_slots_free(session)
    session.close()


class _StopAt:
    """A progress that stops the image at its nth check."""

    def __init__(self, checks: int) -> None:
        self.checks = checks

    def check_aborted(self) -> None:
        self.checks -= 1
        if not self.checks:
            raise InterruptedError("stopped")

    def set_progress(self, _progress: float) -> None:
        pass


def test_a_stopped_image_gives_its_slots_back(inference: ModuleType, cudart: _Cudart):
    session = _shared_session(inference, cudart)
    rng = np.random.default_rng(43)
    tiled, small = rng.random((2, 700, 700, 3), np.float32)
    with pytest.raises(InterruptedError):
        session.upscale(tiled, 2, _StopAt(2))  # a tile in flight
    _assert_both_slots_free(session)
    # An item that shares the session goes on, and its first tile is its own.
    out = session.upscale(small[:64, :64], 2)
    np.testing.assert_array_equal(out, np.clip(_doubled(small[:64, :64]), 0, 1))
    session.close()


def test_the_tile_ahead_is_not_this_nodes_time(inference: ModuleType, cudart: _Cudart):
    session = _shared_session(inference, cudart)
    tile = np.random.default_rng(44).random((64, 64, 3), np.float32)
    slow = _Handle("event0")  # the first slot's, as the stand-in names them
    delay = 0.1

    def synchronize(event: object) -> tuple[int]:
        if event == slow:
            time.sleep(delay)
        return (0,)

    cudart.__dict__["cudaEventSynchronize"] = synchronize
    ahead_in, go = threading.Event(), threading.Event()

    def ahead():
        session.submit(tile)
        ahead_in.set()
        assert go.wait(10)
        before = waited_seconds()
        session.collect(128, 128)
        # Its own wait for the GPU is its own time.
        assert waited_seconds() == before

    first = _on_thread(ahead)
    assert ahead_in.wait(10)
    session.submit(tile)  # on the stream behind `ahead`
    before = waited_seconds()
    session.collect(128, 128)
    assert waited_seconds() - before >= delay * 0.9
    go.set()
    first.join(10)
    session.close()


def test_a_repeated_shape_sets_the_context_once(inference: ModuleType, cudart: _Cudart):
    session = _shared_session(inference, cudart)
    rng = np.random.default_rng(45)
    shapes: list[tuple[int, ...]] = []
    context = session.context
    set_shape = context.set_input_shape

    def record(name: str, shape: tuple[int, ...]) -> bool:
        # Nothing runs on the context while it changes.
        assert cudart.calls[-1][0] == "cudaStreamSynchronize"
        shapes.append(shape)
        return set_shape(name, shape)

    context.set_input_shape = record
    for side in (64, 64, 128, 128, 64):
        tile = rng.random((side, side, 3), np.float32)
        session.submit(tile)
        out = session.collect(2 * side, 2 * side)
        np.testing.assert_array_equal(out, _doubled(tile))
    assert [shape[2] for shape in shapes] == [64, 128, 64]
    session.close()


def test_the_stream_is_non_blocking_so_the_legacy_stream_never_waits_for_a_capture(
    inference: ModuleType, cudart: _Cudart
):
    session = _shared_session(inference, cudart)
    assert (
        "cudaStreamCreateWithFlags",
        (cudart.cudaStreamNonBlocking,),
    ) in cudart.calls
    session.close()


def test_a_failed_growth_fails_its_item_and_leaves_no_freed_buffer_in_use(
    inference: ModuleType, cudart: _Cudart
):
    session = _shared_session(inference, cudart)
    rng = np.random.default_rng(46)
    small = rng.random((64, 64, 3), np.float32)
    large = rng.random((256, 256, 3), np.float32)
    expected = np.clip(_doubled(small), 0, 1)
    np.testing.assert_array_equal(session.upscale(small, 2), expected)
    allocated = sum(1 for name, _ in cudart.calls if name == "cudaMalloc")
    cudart.fail["cudaMalloc"] = allocated  # the card is full
    with pytest.raises(RuntimeError, match="cudaErrorIllegalAddress"):
        session.upscale(large, 2)
    del cudart.fail["cudaMalloc"]
    start = len(cudart.calls)
    # An item that fits goes on, on buffers made for it.
    np.testing.assert_array_equal(session.upscale(small, 2), expected)
    freed = {args[0] for name, args in cudart.calls if name == "cudaFree"}
    used = [
        args[:2] for name, args in cudart.calls[start:] if name == "cudaMemcpyAsync"
    ]
    assert freed and used and not freed.intersection(a for pair in used for a in pair)
    session.close()


def test_a_failed_enqueue_ends_the_session_for_every_later_tile(
    inference: ModuleType, cudart: _Cudart
):
    # An enqueue after a failed one can crash inside TensorRT.
    session = _shared_session(inference, cudart)
    tile = np.random.default_rng(47).random((64, 64, 3), np.float32)
    session.upscale(tile, 2)
    context = session.context
    enqueues = [0]

    def fails(_stream: int) -> bool:
        enqueues[0] += 1
        return False

    context.execute_async_v3 = fails
    with pytest.raises(RuntimeError, match="failed to enqueue"):
        session.upscale(tile, 2)
    with pytest.raises(RuntimeError, match="failed on another image"):
        session.upscale(tile, 2)
    assert enqueues == [1]
    session.close()


def test_a_failed_copy_ends_the_session_and_waits_for_the_stream_before_its_slot(
    inference: ModuleType, cudart: _Cudart
):
    session = _shared_session(inference, cudart)
    tile = np.random.default_rng(48).random((64, 64, 3), np.float32)
    session.upscale(tile, 2)
    copies = sum(1 for name, _ in cudart.calls if name == "cudaMemcpyAsync")
    cudart.fail["cudaMemcpyAsync"] = copies + 1  # the tile's output copy
    with pytest.raises(RuntimeError, match="cudaErrorIllegalAddress"):
        session.upscale(tile, 2)
    assert cudart.calls[-1][0] == "cudaStreamSynchronize"
    del cudart.fail["cudaMemcpyAsync"]
    with pytest.raises(RuntimeError, match="failed on another image"):
        session.upscale(tile, 2)
    session.close()


class _Activations(_Weighting):
    """An engine whose activation memory grows with its tile, or is that of its largest
    size at any tile, and is checked at each execution: it must be the memory set last,
    still allocated, and enough."""

    PER_PIXEL = 100

    def __init__(self, cudart: _Cudart, reserved: int | None) -> None:
        super().__init__()
        self.cudart = cudart
        self.reserved = reserved
        self.memory = (0, 0)

    def update_device_memory_size_for_shapes(self) -> int:
        if self.reserved is not None:
            return self.reserved * self.reserved * self.PER_PIXEL
        return self.shape[2] * self.shape[3] * self.PER_PIXEL

    def set_device_memory(self, address: int, nbytes: int) -> None:
        self.memory = (address, nbytes)

    def execute_async_v3(self, _stream: int) -> bool:
        # TensorRT's enqueue fails on an error CUDA has left on this thread.
        if self.cudart.last_error():
            return False
        address, nbytes = self.memory
        assert self.cudart.allocated.get(address) == nbytes
        assert nbytes >= self.update_device_memory_size_for_shapes()
        return super().execute_async_v3(_stream)


class _ActivationsEngine(_WeightingEngine):
    def __init__(self, cudart: _Cudart, sizes: tuple[int, int], reserves: bool) -> None:
        super().__init__(3)
        self.cudart = cudart
        self.sizes = sizes
        self.reserves = reserves

    def get_tensor_profile_shape(self, _name: str, _profile: int) -> tuple:
        smallest, largest = self.sizes
        return (
            (1, 3, smallest, smallest),
            (1, 3, largest, largest),
            (1, 3, largest, largest),
        )

    def create_execution_context(self, _strategy: object) -> _Context:
        context = _Activations(self.cudart, self.sizes[1] if self.reserves else None)
        self.contexts.append(weakref.ref(context))
        return context


def _memory_session(
    inference: ModuleType,
    cudart: _Cudart,
    sizes: tuple[int, int] = (64, 256),
    reserves: bool = False,
) -> Any:
    """A 2x session on RGB tiles of `sizes` pixels, whose activations take device memory:
    by the tile, or the largest size's at any tile."""
    cudart.fail = {"cudaStreamBeginCapture": 0}
    loaded = DeserializedEngine(
        object(), _ActivationsEngine(cudart, sizes, reserves), 0
    )
    shape = (1, 3, sizes[0], sizes[0])
    info = TensorRTEngineInfo(
        "fp32", 3, 3, 2, "sm_120", "11", True, shape, shape, (1, 3, sizes[1], sizes[1])
    )
    return inference.TensorRTSession(TensorRTEngine(b"engine", info, loaded), 0)


def _tile_memory(size: int) -> int:
    """The device bytes of a square tile: activations, the 3-channel float32 input, and
    its 2x output."""
    return size * size * (_Activations.PER_PIXEL + 3 * 4 + 4 * 3 * 4)


def _no_memory_count(cudart: _Cudart, start: int = 0) -> int:
    """The allocations since call `start` that found no memory: each error's name is
    read once, and nothing else fails here."""
    return sum(1 for name, _ in cudart.calls[start:] if name == "cudaGetErrorName")


def test_out_of_memory_on_a_dynamic_engine_goes_on_in_smaller_tiles(
    inference: ModuleType, cudart: _Cudart
):
    image = np.random.default_rng(50).random((230, 300, 3), np.float32)
    control = _memory_session(inference, cudart, (64, 128))
    expected = control.upscale(image, 2)
    control.close()

    session = _memory_session(inference, cudart)
    cudart.capacity = _tile_memory(128)  # 256 px tiles do not fit
    report = TileReport()
    np.testing.assert_array_equal(session.upscale(image, 2, None, report), expected)
    assert report.text == "128x128 px"
    assert _no_memory_count(cudart) == 1

    # The next image of the run goes straight to the tiles that fit.
    start = len(cudart.calls)
    np.testing.assert_array_equal(session.upscale(image, 2), expected)
    assert _no_memory_count(cudart, start) == 0

    # The next run starts from the largest tiles: here the card has room by then.
    cudart.capacity = None
    session.use_largest_tiles()
    report = TileReport()
    session.upscale(image, 2, None, report)
    assert report.text == "256x256 px"
    session.close()
    assert not cudart.allocated


def test_a_refused_allocation_leaves_no_error_for_this_threads_next_enqueue(
    inference: ModuleType, cudart: _Cudart
):
    # TensorRT's enqueue reads CUDA's per-thread last error, which a refusal sets: the
    # retry's first enqueue failed on it, "cudaError 2: out of memory", and that ended
    # the session.
    session = _memory_session(inference, cudart)
    image = np.random.default_rng(65).random((256, 256, 3), np.float32) * 0.3
    cudart.capacity = _tile_memory(128)
    report = TileReport()
    out = session.upscale(image, 2, None, report)
    assert report.text == "128x128 px"
    assert cudart.last_error() == 0
    np.testing.assert_allclose(out, np.clip(_doubled(image), 0, 1), atol=3e-6)
    session.close()


def test_a_fixed_engines_message_leaves_this_thread_able_to_enqueue(
    inference: ModuleType, cudart: _Cudart
):
    session = _memory_session(inference, cudart, (256, 256))
    image = np.random.default_rng(66).random((256, 256, 3), np.float32) * 0.3
    cudart.capacity = 1000
    with pytest.raises(RuntimeError, match="Fixed engine"):
        session.upscale(image, 2)
    cudart.capacity = None
    out = session.upscale(image, 2)
    np.testing.assert_allclose(out, np.clip(_doubled(image), 0, 1), atol=3e-6)
    session.close()


def test_the_tile_size_of_an_image_that_needs_no_retry(
    inference: ModuleType, cudart: _Cudart
):
    session = _memory_session(inference, cudart)
    rng = np.random.default_rng(60)
    report = TileReport()
    session.upscale(rng.random((100, 90, 3), np.float32), 2, None, report)
    assert report.text == "Whole image"
    session.upscale(rng.random((100, 300, 3), np.float32), 2, None, report)
    assert report.text == "256x128 px"
    assert _no_memory_count(cudart) == 0
    session.close()


def test_each_lack_of_memory_halves_the_tiles_until_they_fit(
    inference: ModuleType, cudart: _Cudart
):
    session = _memory_session(inference, cudart, (64, 512))
    image = np.random.default_rng(51).random((512, 512, 3), np.float32)
    cudart.capacity = _tile_memory(128)
    report = TileReport()
    session.upscale(image, 2, None, report)
    assert report.text == "128x128 px"
    assert _no_memory_count(cudart) == 2  # 512 and 256
    session.close()


def test_a_tile_of_192_px_goes_to_half_not_a_third(
    inference: ModuleType, cudart: _Cudart
):
    # 192 // 2 = 96, rounded down to 64 where 128 fits: 81 tiles instead of 4.
    session = _memory_session(inference, cudart)
    image = np.random.default_rng(71).random((192, 192, 3), np.float32)
    cudart.capacity = _tile_memory(128)
    report = TileReport()
    session.upscale(image, 2, None, report)
    assert report.text == "128x128 px"
    assert _no_memory_count(cudart) == 1
    session.close()


def test_a_small_image_shrinks_from_the_tile_it_used_not_the_engines_largest(
    inference: ModuleType, cudart: _Cudart
):
    session = _memory_session(inference, cudart, (64, 1024))
    image = np.random.default_rng(52).random((256, 256, 3), np.float32)
    cudart.capacity = _tile_memory(128)
    report = TileReport()
    session.upscale(image, 2, None, report)
    assert report.text == "128x128 px"
    assert _no_memory_count(cudart) == 1
    session.close()


def test_a_buffer_that_fails_after_the_activations_does_not_keep_them(
    inference: ModuleType, cudart: _Cudart
):
    # The activations and the input fit and the output does not. Kept, they would leave
    # the smaller tile's output no room either, and the tiles would end up too small.
    session = _memory_session(inference, cudart)
    image = np.random.default_rng(53).random((256, 256, 3), np.float32)
    cudart.capacity = 256 * 256 * (_Activations.PER_PIXEL + 3 * 4) + 1
    report = TileReport()
    session.upscale(image, 2, None, report)
    assert report.text == "128x128 px"
    assert _no_memory_count(cudart) == 1
    session.close()


def test_a_fixed_engine_out_of_memory_says_what_to_do(
    inference: ModuleType, cudart: _Cudart
):
    session = _memory_session(inference, cudart, (256, 256))
    image = np.random.default_rng(54).random((100, 100, 3), np.float32)
    cudart.capacity = 1000
    with pytest.raises(
        RuntimeError, match=r"256x256 px.*Fixed engine.*Dynamic"
    ) as raised:
        session.upscale(image, 2)
    assert isinstance(raised.value.__cause__, inference.CudaOutOfMemory)
    assert _no_memory_count(cudart) == 1
    session.close()


def test_a_dynamic_engine_at_its_smallest_tile_says_so(
    inference: ModuleType, cudart: _Cudart
):
    session = _memory_session(inference, cudart, (64, 128))
    image = np.random.default_rng(55).random((128, 128, 3), np.float32)
    cudart.capacity = 1000
    with pytest.raises(RuntimeError, match=r"64x64 px, its smallest tile size"):
        session.upscale(image, 2)
    assert _no_memory_count(cudart) == 2  # 128 and 64
    session.close()


def test_an_engine_that_reserves_its_largest_size_says_smaller_tiles_do_not_help(
    inference: ModuleType, cudart: _Cudart
):
    session = _memory_session(inference, cudart, reserves=True)
    image = np.random.default_rng(63).random((256, 256, 3), np.float32)
    cudart.capacity = 1000
    with pytest.raises(RuntimeError, match=r"takes the memory of its largest size"):
        session.upscale(image, 2)
    # An item that was beside it can still go on, once the card has room.
    cudart.capacity = None
    session.use_largest_tiles()
    assert session.upscale(image, 2).shape == (512, 512, 3)
    session.close()


def test_an_engine_that_reserves_its_largest_size_goes_on_when_buffers_were_short(
    inference: ModuleType, cudart: _Cudart
):
    # The activations fit; the 256 px tile's output does not, and the 128 px tile's does.
    session = _memory_session(inference, cudart, reserves=True)
    image = np.random.default_rng(64).random((256, 256, 3), np.float32)
    cudart.capacity = 256 * 256 * _Activations.PER_PIXEL + 128 * 128 * (
        3 * 4 + 4 * 3 * 4
    )
    report = TileReport()
    session.upscale(image, 2, None, report)
    assert report.text == "128x128 px"
    assert _no_memory_count(cudart) == 1
    session.close()


def test_another_items_tile_in_flight_keeps_its_result(
    inference: ModuleType, cudart: _Cudart
):
    session = _memory_session(inference, cudart)
    rng = np.random.default_rng(56)
    small = rng.random((64, 64, 3), np.float32)
    large = rng.random((256, 256, 3), np.float32)
    session.submit(small)
    cudart.capacity = _tile_memory(128)
    results: list[np.ndarray] = []
    profile = session.bounds
    thread = _on_thread(lambda: results.append(session.upscale(large, 2)))
    deadline = time.monotonic() + 10
    while session.bounds is profile and time.monotonic() < deadline:
        time.sleep(0.001)
    assert session.bounds.max_h == 128

    np.testing.assert_array_equal(session.collect(128, 128), _doubled(small))
    thread.join(10)
    assert not thread.is_alive()
    assert results[0].shape == (512, 512, 3)
    _assert_both_slots_free(session)
    session.close()


def test_a_tile_enqueued_after_the_failure_ends_before_the_buffers_go(
    inference: ModuleType, cudart: _Cudart
):
    # Another item's tile reaches the stream between the failure and the release of the
    # device buffers (the stand-in fails a free with a tile on the stream). It has the
    # shape of the retry's tiles, which must not take the freed buffers for its own.
    session = _memory_session(inference, cudart)
    rng = np.random.default_rng(61)
    small = rng.random((128, 128, 3), np.float32)
    large = rng.random((256, 256, 3), np.float32)
    submitted = threading.Event()
    results: list[np.ndarray] = []

    def other_item() -> None:
        session.submit(small)
        submitted.set()
        results.append(session.collect(256, 256))

    def enqueue_the_other_item() -> None:
        thread = _on_thread(other_item)
        assert submitted.wait(10)
        threads.append(thread)

    threads: list[threading.Thread] = []
    cudart.after_no_memory = enqueue_the_other_item
    cudart.capacity = _tile_memory(128)
    report = TileReport()
    session.upscale(large, 2, None, report)
    assert report.text == "128x128 px"
    threads[0].join(10)
    assert not threads[0].is_alive()
    np.testing.assert_array_equal(results[0], _doubled(small))
    session.close()


def test_two_items_out_of_memory_at_once_shrink_the_tiles_once(
    inference: ModuleType, cudart: _Cudart
):
    session = _memory_session(inference, cudart)
    image = np.random.default_rng(57).random((256, 256, 3), np.float32)
    cudart.capacity = _tile_memory(128)
    # Both have failed before either goes on to choose smaller tiles.
    cudart.after_no_memory = threading.Barrier(2, timeout=10).wait
    reports = [TileReport(), TileReport()]
    threads = [
        _on_thread(lambda report=report: session.upscale(image, 2, None, report))
        for report in reports
    ]
    for thread in threads:
        thread.join(20)
        assert not thread.is_alive()
    assert _no_memory_count(cudart) == 2
    assert [report.text for report in reports] == ["128x128 px", "128x128 px"]
    session.close()


def test_a_larger_tile_that_failed_before_the_choice_does_not_undo_it(
    inference: ModuleType, cudart: _Cudart
):
    # A 128 px tile and a 256 px tile have both failed. The small one chooses 64 px; the
    # large one goes on after that, in the tiles chosen, instead of asking for 128 px.
    session = _memory_session(inference, cudart)
    rng = np.random.default_rng(62)
    profile = session.bounds
    reports = [TileReport(), TileReport()]
    small = threading.Thread(
        target=session.upscale,
        args=(rng.random((128, 128, 3), np.float32), 2, None, reports[0]),
    )
    large = threading.Thread(
        target=session.upscale,
        args=(rng.random((256, 256, 3), np.float32), 2, None, reports[1]),
    )
    both_failed = threading.Barrier(2, timeout=10)

    def after_no_memory() -> None:
        both_failed.wait()
        deadline = time.monotonic() + 10
        while (
            threading.current_thread() is large
            and session.bounds is profile
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)

    cudart.after_no_memory = after_no_memory
    cudart.capacity = _tile_memory(64)
    small.start()
    large.start()
    for thread in (small, large):
        thread.join(20)
        assert not thread.is_alive()
    assert session.bounds.max_h == 64
    assert _no_memory_count(cudart) == 2
    assert [report.text for report in reports] == ["64x64 px", "64x64 px"]
    session.close()


def test_stop_during_the_smaller_tiles_gives_the_slots_back(
    inference: ModuleType, cudart: _Cudart
):
    session = _memory_session(inference, cudart)
    image = np.random.default_rng(58).random((256, 256, 3), np.float32)
    cudart.capacity = _tile_memory(128)
    with pytest.raises(InterruptedError):
        # The failed whole image, the first smaller tile, then Stop with it in flight
        session.upscale(image, 2, _StopAt(3))
    assert session.bounds.max_h == 128
    _assert_both_slots_free(session)
    session.close()


@pytest.mark.parametrize(
    "wait",
    [
        pytest.param(0, id="the shape's wait"),
        pytest.param(1, id="the growth's wait"),
    ],
)
def test_a_memory_error_from_a_stream_wait_is_a_failed_stream(
    inference: ModuleType, cudart: _Cudart, wait: int
):
    # A wait reports what a launch on the stream met: not an allocation for this tile.
    session = _memory_session(inference, cudart)
    rng = np.random.default_rng(67)
    session.upscale(rng.random((128, 128, 3), np.float32), 2)
    profile = session.bounds
    cudart.fail_once("cudaStreamSynchronize", OUT_OF_MEMORY, at=wait)
    with pytest.raises(inference.CudaOutOfMemory):
        session.upscale(rng.random((256, 256, 3), np.float32), 2)
    assert session.bounds is profile
    with pytest.raises(RuntimeError, match="failed on another image"):
        session.upscale(rng.random((128, 128, 3), np.float32), 2)
    session.close()


def test_a_wait_that_fails_after_a_refused_allocation_stops_the_retry(
    inference: ModuleType, cudart: _Cudart
):
    # The refused tile's own wait, unchecked so that it cannot hide the refusal, finds
    # that the stream has failed.
    session = _memory_session(inference, cudart)
    rng = np.random.default_rng(68)
    session.upscale(rng.random((128, 128, 3), np.float32), 2)
    profile = session.bounds
    cudart.capacity = _tile_memory(128)
    cudart.fail_once("cudaStreamSynchronize", ILLEGAL_ADDRESS, at=2)
    with pytest.raises(inference.CudaOutOfMemory):
        session.upscale(rng.random((256, 256, 3), np.float32), 2)
    assert session.bounds is profile
    cudart.capacity = None
    with pytest.raises(RuntimeError, match="failed on another image"):
        session.upscale(rng.random((128, 128, 3), np.float32), 2)
    session.close()


def test_a_tile_on_the_stream_that_failed_stops_the_retry(
    inference: ModuleType, cudart: _Cudart
):
    # An image's first tile has failed on the stream when its second finds no memory for
    # its pinned buffer: the retry would go on to the failed stream.
    session = _memory_session(inference, cudart)
    image = np.random.default_rng(69).random((300, 300, 3), np.float32)
    profile = session.bounds
    # Each slot's input and output: the second tile's input is the third.
    cudart.fail_once("cudaMallocHost", OUT_OF_MEMORY, at=2)
    cudart.fail_once("cudaEventSynchronize", ILLEGAL_ADDRESS)
    with pytest.raises(inference.CudaOutOfMemory):
        session.upscale(image, 2)
    assert session.bounds is profile
    with pytest.raises(RuntimeError, match="failed on another image"):
        session.upscale(image, 2)
    session.close()


def test_a_wait_that_fails_when_the_buffers_go_ends_the_session(
    inference: ModuleType, cudart: _Cudart
):
    session = _memory_session(inference, cudart)
    rng = np.random.default_rng(70)
    session.upscale(rng.random((128, 128, 3), np.float32), 2)
    cudart.capacity = _tile_memory(128)
    # The shape's, the growth's and the refused tile's waits pass; the release's does not.
    cudart.fail_once("cudaStreamSynchronize", ILLEGAL_ADDRESS, at=3)
    with pytest.raises(RuntimeError, match="cudaErrorIllegalAddress"):
        session.upscale(rng.random((256, 256, 3), np.float32), 2)
    cudart.capacity = None
    with pytest.raises(RuntimeError, match="failed on another image"):
        session.upscale(rng.random((128, 128, 3), np.float32), 2)
    session.close()


def test_out_of_memory_in_an_enqueue_ends_the_session_without_a_retry(
    inference: ModuleType, cudart: _Cudart
):
    # Unlike an allocation for a tile, this leaves the stream in an unknown state.
    session = _memory_session(inference, cudart)
    image = np.random.default_rng(59).random((256, 256, 3), np.float32)
    profile = session.bounds
    cudart.fail_code = OUT_OF_MEMORY
    cudart.fail["cudaMemcpyAsync"] = 0
    with pytest.raises(inference.CudaOutOfMemory):
        session.upscale(image, 2)
    assert session.bounds is profile
    del cudart.fail["cudaMemcpyAsync"]
    with pytest.raises(RuntimeError, match="failed on another image"):
        session.upscale(image, 2)
    session.close()


def _capturing_session(inference: ModuleType, cudart: _Cudart) -> Any:
    session = _shared_session(inference, cudart)
    del cudart.fail["cudaStreamBeginCapture"]
    return session


def test_an_enqueue_refused_inside_a_capture_leaves_the_enqueue_in_use(
    inference: ModuleType, cudart: _Cudart
):
    # Another thread's stream synchronize makes TensorRT refuse an enqueue being captured,
    # and the context goes on bit for bit. VapourSynth's session is never replaced: a
    # session failed here would raise for every later frame.
    session = _capturing_session(inference, cudart)
    tile = np.random.default_rng(49).random((64, 64, 3), np.float32)
    session.upscale(tile, 2)  # seen once: no capture yet
    context = session.context
    run = context.execute_async_v3
    enqueues: list[bool] = []

    def refuses_in_capture(stream: int) -> bool:
        ok = len(enqueues) != 1
        enqueues.append(ok)
        return run(stream) if ok else False

    context.execute_async_v3 = refuses_in_capture
    expected = np.clip(_doubled(tile), 0, 1)
    for _ in range(3):
        np.testing.assert_array_equal(session.upscale(tile, 2), expected)
    assert enqueues == [True, False, True, True]
    names = [name for name, _ in cudart.calls]
    assert "cudaStreamEndCapture" in names and "cudaGetLastError" in names
    assert names.count("cudaStreamBeginCapture") == 1
    session.close()


def test_a_capture_invalidated_from_outside_leaves_the_enqueue_in_use(
    inference: ModuleType, cudart: _Cudart
):
    session = _capturing_session(inference, cudart)
    tile = np.random.default_rng(50).random((64, 64, 3), np.float32)
    session.upscale(tile, 2)
    cudart.fail["cudaStreamEndCapture"] = 0
    expected = np.clip(_doubled(tile), 0, 1)
    for _ in range(3):
        np.testing.assert_array_equal(session.upscale(tile, 2), expected)
    # One capture was tried, and the session goes on without one; the error it left on
    # this thread is taken, or the next enqueue fails on it.
    names = [name for name, _ in cudart.calls]
    assert names.count("cudaStreamBeginCapture") == 1
    assert "cudaGetLastError" in names
    session.close()


def test_an_enqueue_that_raises_inside_a_capture_ends_the_capture(
    inference: ModuleType, cudart: _Cudart
):
    session = _capturing_session(inference, cudart)
    tile = np.random.default_rng(51).random((64, 64, 3), np.float32)
    session.upscale(tile, 2)
    context = session.context
    run = context.execute_async_v3
    enqueues = [0]

    def raises_in_capture(stream: int) -> bool:
        enqueues[0] += 1
        if enqueues[0] == 2:
            raise ValueError("inside the capture")
        return run(stream)

    context.execute_async_v3 = raises_in_capture
    with pytest.raises(ValueError, match="inside the capture"):
        session.upscale(tile, 2)
    assert "cudaStreamEndCapture" in [name for name, _ in cudart.calls]
    with pytest.raises(RuntimeError, match="failed on another image"):
        session.upscale(tile, 2)
    session.close()


def test_a_capture_and_a_new_context_take_turns_across_engines(
    inference: ModuleType, cudart: _Cudart
):
    lock = inference.gpu_setup_lock
    held_during: list[bool] = []
    session = _capturing_session(inference, cudart)
    tile = np.random.default_rng(52).random((64, 64, 3), np.float32)
    session.upscale(tile, 2)

    def begin(_stream: object, _mode: object) -> tuple[int]:
        held_during.append(lock.locked())
        return (0,)

    cudart.__dict__["cudaStreamBeginCapture"] = begin
    session.upscale(tile, 2)
    assert held_during == [True]
    session.close()

    created_under_lock: list[bool] = []
    engine = _WeightingEngine(3)
    real = engine.create_execution_context

    def create(strategy: object) -> _Context:
        created_under_lock.append(lock.locked())
        return real(strategy)

    engine.create_execution_context = create
    loaded = DeserializedEngine(object(), engine, 0)
    shape = (1, 3, 64, 64)
    info = TensorRTEngineInfo(
        "fp32", 3, 3, 2, "sm_120", "11", True, shape, shape, shape
    )
    inference.TensorRTSession(TensorRTEngine(b"engine", info, loaded), 0).close()
    assert created_under_lock == [True]


class _RecordsItsEnd(_Weighting):
    """A context that records whether the setup lock is held when it is destroyed."""

    lock: Any = None
    ends: Any = None

    def __del__(self) -> None:
        self.ends.append(self.lock.locked())


class _RecordingEngine(_WeightingEngine):
    def get_tensor_shape(self, _name: str) -> tuple[int, ...]:
        return (1, self.channels, -1, -1)

    def create_execution_context(self, _strategy: object) -> _Context:
        return _RecordsItsEnd()


def test_destroying_a_context_takes_turns_with_captures(
    inference: ModuleType, cudart: _Cudart, monkeypatch: pytest.MonkeyPatch
):
    # With another engine's sessions closing beside it, a capture failed in 3 of 3 runs
    # on the RTX 5090, and a session without graphs runs slower for the rest of its life.
    ends: list[bool] = []
    monkeypatch.setattr(_RecordsItsEnd, "lock", inference.gpu_setup_lock)
    monkeypatch.setattr(_RecordsItsEnd, "ends", ends)
    loaded = DeserializedEngine(object(), _RecordingEngine(3), 0)
    shape = (1, 3, 64, 64)
    info = TensorRTEngineInfo(
        "fp32", 3, 3, 2, "sm_120", "11", True, shape, shape, shape
    )
    inference.TensorRTSession(TensorRTEngine(b"engine", info, loaded), 0).close()
    assert ends == [True]

    ends.clear()
    cudart.__dict__["cudaGetDeviceProperties"] = lambda _device: (
        0,
        SimpleNamespace(major=12, minor=0),
    )
    engine_info = importlib.import_module("nodes.impl.tensorrt.engine_info")
    engine_info.read_engine_info(DeserializedEngine(object(), _RecordingEngine(3), 0))
    assert ends == [True]


def test_a_failed_growth_leaves_no_error_for_the_threads_next_image(
    inference: ModuleType, cudart: _Cudart
):
    # CUDA keeps the error for this thread, and TensorRT's next enqueue reads it: the
    # session failed, so the item beside it got "failed on another image".
    session = _shared_session(inference, cudart)
    rng = np.random.default_rng(53)
    small = rng.random((64, 64, 3), np.float32)
    large = rng.random((256, 256, 3), np.float32)
    expected = np.clip(_doubled(small), 0, 1)
    np.testing.assert_array_equal(session.upscale(small, 2), expected)
    left: list[int] = []

    def out_of_memory(_size: int) -> tuple[int, None]:
        left.append(ILLEGAL_ADDRESS)
        return ILLEGAL_ADDRESS, None

    def take_error() -> tuple[int]:
        code = left[-1] if left else 0
        left.clear()
        return (code,)

    context = session.context
    run = context.execute_async_v3
    context.execute_async_v3 = lambda stream: not left and run(stream)
    cudart.__dict__["cudaGetLastError"] = take_error
    cudart.__dict__["cudaMalloc"] = out_of_memory
    with pytest.raises(RuntimeError, match="cudaErrorIllegalAddress"):
        session.upscale(large, 2)
    del cudart.__dict__["cudaMalloc"]
    np.testing.assert_array_equal(session.upscale(small, 2), expected)
    session.close()


def test_a_shape_changes_wait_for_another_tile_is_not_this_nodes_time(
    inference: ModuleType, cudart: _Cudart
):
    session = _shared_session(inference, cudart)
    delay = 0.1

    def synchronize(_stream: object) -> tuple[int]:
        time.sleep(delay)
        cudart.stream_busy = False
        return (0,)

    cudart.__dict__["cudaStreamSynchronize"] = synchronize
    before = waited_seconds()
    session.submit(np.zeros((64, 64, 3), np.float32))
    assert waited_seconds() - before >= delay * 0.9
    session.collect(128, 128)
    session.close()


def test_an_images_second_tile_goes_before_items_of_one_tile(
    inference: ModuleType, cudart: _Cudart
):
    session = _shared_session(inference, cudart)
    tile = np.zeros((64, 64, 3), np.float32)
    holder_in, holder_go = threading.Event(), threading.Event()
    first_in, second_in, first_go = (threading.Event() for _ in range(3))
    single_in = threading.Event()

    def holder() -> None:
        session.submit(tile)
        holder_in.set()
        assert holder_go.wait(10)
        session.collect(128, 128)

    def tiled() -> None:
        session.submit(tile)
        first_in.set()
        assert first_go.wait(10)
        session.submit(tile)  # its second tile in flight
        second_in.set()
        session.collect(128, 128)
        session.collect(128, 128)

    def single() -> None:
        session.submit(tile)
        single_in.set()
        session.collect(128, 128)

    threads = [_on_thread(holder)]
    assert holder_in.wait(10)
    threads.append(_on_thread(tiled))
    assert first_in.wait(10)
    threads.append(_on_thread(single))
    time.sleep(0.05)  # the item of one tile waits first
    first_go.set()
    time.sleep(0.05)
    holder_go.set()  # one slot comes free
    assert second_in.wait(10)
    for thread in threads:
        thread.join(10)
        assert not thread.is_alive()
    assert single_in.is_set()
    session.close()


def test_a_grey_image_reaches_a_one_channel_engine(inference: ModuleType):
    # chaiNNer holds a grey image as HxW, which the session takes as one plane.
    from nodes.impl.tensorrt.tiling import tiled_upscale

    loaded = DeserializedEngine(object(), _WeightingEngine(1), 0)
    shape = (1, 1, 64, 64)
    info = TensorRTEngineInfo(
        "fp32", 1, 1, 2, "sm_120", "11", True, shape, shape, shape
    )
    session = inference.TensorRTSession(TensorRTEngine(b"engine", info, loaded), 0)
    image = np.random.default_rng(32).random((40, 50), np.float32)
    out = tiled_upscale(image, 2, session.bounds, session)
    session.close()
    np.testing.assert_array_equal(out, image.repeat(2, 0).repeat(2, 1))


def test_before_clip_sees_a_sessions_image_before_its_clip(
    inference: ModuleType, cudart: _Cudart
):
    session = _shared_session(inference, cudart)
    seen: list[np.ndarray] = []
    image = np.random.default_rng(1).random((64, 64, 3), np.float32)
    out = session.upscale(image, 2, before_clip=lambda raw: seen.append(raw.copy()))
    assert len(seen) == 1
    assert seen[0].shape == out.shape


def test_only_an_engine_of_fp32_in_and_fp16_out_takes_opaque_as_same(
    inference: ModuleType,
):
    takes = inference.TensorRTSession.takes_fp32_into_fp16.fget
    fp16, fp32 = np.dtype(np.float16), np.dtype(np.float32)
    for storage_in, storage_out, expected in (
        (fp32, fp16, True),
        (fp16, fp16, False),
        (fp32, fp32, False),
        (np.dtype(np.uint16), np.dtype(np.uint16), False),
    ):
        session = SimpleNamespace(
            _in=SimpleNamespace(storage=storage_in),
            _out=SimpleNamespace(storage=storage_out),
        )
        assert takes(session) is expected


@pytest.mark.parametrize(("takes", "runs"), [(False, 24), (True, 15)])
def test_an_fp32_into_fp16_engine_reuses_a_dark_opaque_tile(
    inference: ModuleType,
    cudart: _Cudart,
    monkeypatch: pytest.MonkeyPatch,
    takes: bool,
    runs: int,
):
    # An image with a hole, of colours whose white background loses its lowest bits
    from nodes.impl.upscale.convenient_upscale import convenient_upscale

    monkeypatch.setattr(inference.TensorRTSession, "takes_fp32_into_fp16", takes)
    session = _shared_session(inference, cudart)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (500, 700, 4)).astype(np.float32) / 255
    img[:, :, 3] = 1
    img[250:258, 350:358, 3] = 0
    submitted: list[tuple[int, ...]] = []
    submit = session.submit

    def counted(tile: np.ndarray) -> None:
        submitted.append(tile.shape)
        submit(tile)

    monkeypatch.setattr(session, "submit", counted)
    convenient_upscale(img, 3, 3, lambda i: session.upscale(i, 2), False, clip=False)
    session.close()
    assert len(submitted) == runs
