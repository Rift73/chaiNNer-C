"""The GPU's thread and the CPU's for the PyTorch, ONNX and NCNN Upscale Image nodes,
which may run ahead of earlier items (execution_scheduler.LOCKED_DEVICE_NODES). Each
runs its whole upscale on the thread of the device it runs on, on the run's driver's
turn or computed ahead, as these nodes ran on the driver alone before: one upscale on a
device at a time, so two never share a device's memory and a tile size estimated from
the free memory (NCNN's automatic mode) sees what one upscale at a time sees; and one
calling thread, as the frameworks keep state per calling thread for the thread's life,
which the engine's workers (one per CPU) would each make: ONNX Runtime's GPU providers
a context (cuDNN and cuBLAS handles and workspace, about 66 MiB a thread with the CUDA
provider), PyTorch and NCNN on the CPU an OpenMP thread team. One thread for every GPU:
NCNN's Vulkan device and PyTorch's or ONNX Runtime's CUDA device may be one GPU under
two indexes."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident
from typing import TypeVar

from util import add_waiting, waited_seconds, waiting

from . import numpy_pool

T = TypeVar("T")

# They run node code, so they install the NumPy pool handler.
_threads = {
    gpu: ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="gpu-upscale" if gpu else "cpu-upscale",
        initializer=numpy_pool.install,
    )
    for gpu in (True, False)
}
# Each device thread's identity, once it ran a call.
_idents: dict[bool, int] = {}


def on_device_thread(gpu: bool, call: Callable[[], T]) -> T:
    """`call()` on the GPU's thread (or the CPU's), after the calls before it; at once
    when called there. The wait for the thread is `waiting`, as `held`'s for a lock, and
    so are the waits inside `call()` (a shared session's or model's lock)."""
    if _idents.get(gpu) == get_ident():
        return call()
    started = Event()
    waited: list[float] = []

    def run() -> T:
        _idents[gpu] = get_ident()
        started.set()
        before = waited_seconds()
        try:
            return call()
        finally:
            waited.append(waited_seconds() - before)

    future = _threads[gpu].submit(run)
    # Also when the call never runs (a broken pool): result() then raises its error.
    future.add_done_callback(lambda _: started.set())
    with waiting():
        started.wait()
    try:
        return future.result()
    finally:
        add_waiting(sum(waited))
        # The error's traceback holds this frame, so the frame drops the future, which
        # holds the error.
        del future
