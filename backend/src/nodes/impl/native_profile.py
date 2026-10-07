"""CHAINNER_C_PROFILE: per-name call times at the native boundary.

The variable is read once; only "1" turns the timer on. Off, native.lib() is a
plain ctypes.CDLL and native_graph.graph() is untouched, so no call pays for
it. On, every DLL export and the pyd entries named below record, per name:
calls; inclusive ns (elapsed); exclusive ns (elapsed minus the timed calls
nested in it on the same thread); max ns (the largest exclusive time of one
call). A call that raises is recorded too. The worker's /run resets the table
before its executor and logs it after, as one line. A recorded time includes
re-acquiring the GIL after the call. The timer adds about 0.74 us to a call
(829 ns against 89 ns for cn_abi_version, measured on Python 3.11.5).

native.lib() also attaches the DLL's pool counters (attach_pool): the line then
carries "pool", the counters' deltas since reset(); it is not a timed name.
numpy_pool.install() attaches the NumPy pool's counters (attach_numpy_pool) at the
first install that installs the handler: the line then carries "numpy_pool", the
pyd's numpy_pool_stats(), which reset() zeroes; it is not a timed name either.
native_buffers times its conversion bridge as "converted_pixels" and its freeze
of a node's fresh output as "freeze_normalized": the exclusive time of each is
the bridge's Python around cn_pixels_convert_checked.

The item window's commit path (SP3b) adds three names. "window.take_advance" is
the commit path's await of take_advance, once per generator per slot (once per
slot on a two-phase window, which has one generator): the wait for the
producer's outcome, including take_advance's own synchronous work (setting
_wanted and pumping, for a slot not yet resolved) and every loop callback that
runs while it is suspended (it does not suspend for a resolved slot).
"loop.thread_cpu" is the event loop thread's CPU time (time.thread_time_ns) from
start_loop_thread_cpu() to log_line(), one call per run. record() adds both
outside the nesting, since other timed calls run on that thread meanwhile.
"lazy.resolve" is a reading thread's blocking wait for a lazy input's coroutine,
measured nested in the reading node's timed call. It starts at the wait: the
submission (call_soon_threadsafe, cleanup.detach()) is outside it, in the
reader's exclusive time. So it is a lower bound of the hop to the loop thread
plus what the coroutine awaits, near zero when the loop finished the coroutine
before the wait began.
"""

from __future__ import annotations

import ctypes
import functools
import json
import os
import threading
import time
from collections.abc import Callable
from types import ModuleType
from typing import ParamSpec, TypeVar

VARIABLE = "CHAINNER_C_PROFILE"
# Python-visible pyd entries; swap_channels and the fpng repack run inside them.
PYD_FUNCTIONS = (
    "image_io_read_pil",
    "image_io_save_prepare",
    "image_io_save",
    "video_writer_frame",
)
# Iterator classes whose __next__ is timed as "<class>.__next__".
PYD_ITERATORS = ("_FileSequenceIterator", "_VideoFrames")
# Of those, the ones whose items the item window also produces in two phases (SP3b),
# timed as "<class>.describe" and "<class>.materialize".
PYD_SPLIT_ITERATORS = ("_FileSequenceIterator",)
# The pool's counters (parallel.h), logged as the line's "pool" deltas.
POOL_COUNTERS = (
    "cn_parallel_dispatches",
    "cn_parallel_wakes",
    "cn_parallel_empty_wakes",
)
_POOL_KEYS = ("dispatches", "wakes", "empty_wakes")

P = ParamSpec("P")
T = TypeVar("T")


class _Frames(threading.local):
    def __init__(self) -> None:
        # One accumulator per timed call in flight on this thread: the elapsed
        # time of the timed calls nested in it.
        self.children: list[int] = []


_frames = _Frames()
# A collection triggered inside a locked section can run a finalizer that makes
# a timed call on the same thread (native_neighborhood.py, native_onnx_runtime.py),
# so the lock is reentrant and no locked section iterates the live table.
_lock = threading.RLock()
# name -> [calls, inclusive_ns, exclusive_ns, max_ns]
_table: dict[str, list[int]] = {}
# The attached counters (untimed) and their values at the last reset(); None
# until attach_pool succeeds.
_pool: tuple[Callable[[], int], ...] | None = None
_pool_base: tuple[int, ...] = ()
# The NumPy pool's (stats, reset), untimed; None until attach_numpy_pool.
_numpy_pool: tuple[Callable[[], dict[str, object]], Callable[[], None]] | None = None
# The loop thread's CPU time at start_loop_thread_cpu(); None before it, after
# a reset(), and once log_line() has recorded it.
_loop_cpu: int | None = None


@functools.lru_cache(maxsize=1)
def enabled() -> bool:
    return os.environ.get(VARIABLE) == "1"


def _add(name: str, elapsed: int, exclusive: int) -> None:
    """Count one call of name: elapsed ns inclusive, exclusive ns exclusive."""
    with _lock:
        entry = _table.get(name)
        if entry is None:
            # A finalizer run while the new list is allocated may have
            # created this entry; setdefault keeps its call.
            entry = _table.setdefault(name, [0, 0, 0, 0])
        entry[0] += 1
        entry[1] += elapsed
        entry[2] += exclusive
        entry[3] = max(entry[3], exclusive)


def measure(name: str, call: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
    """call(*args, **kwargs), timed as one call of name, nested on this thread."""
    children = _frames.children
    children.append(0)
    start = time.perf_counter_ns()
    try:
        return call(*args, **kwargs)
    finally:
        elapsed = time.perf_counter_ns() - start
        exclusive = elapsed - children.pop()
        if children:
            children[-1] += elapsed
        _add(name, elapsed, exclusive)


def record(name: str, elapsed_ns: int) -> None:
    """One call of elapsed_ns under name, its inclusive and exclusive time alike.

    It is outside the nesting: it neither subtracts the timed calls made on this
    thread meanwhile nor is subtracted from the one it is made in. For a wait on
    the event loop's thread, where other timed calls run while it is suspended.
    """
    _add(name, elapsed_ns, elapsed_ns)


def timed(name: str, call: Callable[P, T]) -> Callable[P, T]:
    @functools.wraps(call)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return measure(name, call, *args, **kwargs)

    return wrapper


class ProfiledCDLL(ctypes.CDLL):
    """A CDLL whose exports record every call under their name."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

        # CDLL.__init__ sets _FuncPtr on the instance, so a class attribute
        # would never be used; a subclass must restate _flags_ (else TypeError)
        # and _restype_. In the class body, self is still the library.
        class _TimedFuncPtr(self._FuncPtr):
            _flags_ = self._func_flags_
            _restype_ = self._func_restype_

            def __call__(self, *args: object, **kwargs: object) -> object:
                return measure(self.__name__, super().__call__, *args, **kwargs)

        self._FuncPtr = _TimedFuncPtr


def instrument_graph(module: ModuleType) -> None:
    """Time the pyd entries of module; an entry already timed is left as it is.

    The lock keeps two threads that miss graph()'s cache from both wrapping an entry.
    """
    with _lock:
        for name in PYD_FUNCTIONS:
            function = getattr(module, name)
            if not hasattr(function, "__wrapped__"):
                setattr(module, name, timed(name, function))
        for name in PYD_ITERATORS:
            iterator = getattr(module, name)
            if not hasattr(iterator.__next__, "__wrapped__"):
                iterator.__next__ = timed(f"{name}.__next__", iterator.__next__)
        for name in PYD_SPLIT_ITERATORS:
            iterator = getattr(module, name)
            for phase in ("describe", "materialize"):
                method = getattr(iterator, phase)
                if not hasattr(method, "__wrapped__"):
                    setattr(iterator, phase, timed(f"{name}.{phase}", method))


def attach_pool(path: str) -> None:
    """Read the pool counters of the DLL at path from now on, untimed.

    All three exports are resolved before any state is stored; a missing one
    raises the ABI error naming the DLL. The baseline is their current value.
    """
    library = ctypes.CDLL(path)
    counters = []
    for name in POOL_COUNTERS:
        try:
            counter = library[name]
        except AttributeError as error:
            raise RuntimeError(
                f"Incompatible chaiNNer C kernel ABI at {path}: no {name} export"
            ) from error
        counter.argtypes = []
        counter.restype = ctypes.c_uint64
        counters.append(counter)
    global _pool, _pool_base
    with _lock:
        _pool = tuple(counters)
        _pool_base = tuple(counter() for counter in _pool)


def attach_numpy_pool(
    stats: Callable[[], dict[str, object]], reset: Callable[[], None]
) -> None:
    """Log stats() as the line's "numpy_pool" from now on, untimed; reset() calls
    reset, which zeroes the counters that stats() reads."""
    global _numpy_pool
    with _lock:
        _numpy_pool = (stats, reset)


def reset() -> None:
    global _pool_base, _loop_cpu
    with _lock:
        _table.clear()
        if _pool is not None:
            _pool_base = tuple(counter() for counter in _pool)
        if _numpy_pool is not None:
            _numpy_pool[1]()
        _loop_cpu = None


def start_loop_thread_cpu() -> None:
    """Read the calling thread's CPU time; the next log_line() records the time
    since, read on its own thread, as "loop.thread_cpu". The worker's /run calls
    both on the event loop's thread, after reset().

    Windows advances thread time in ticks of about 15.6 ms, so it is a total for
    the run, not a time per item.
    """
    global _loop_cpu
    _loop_cpu = time.thread_time_ns()


def snapshot() -> dict[str, dict[str, int]]:
    with _lock:
        table = _table.copy()
        return {
            name: {
                "calls": calls,
                "inclusive_ns": inclusive,
                "exclusive_ns": exclusive,
                "max_ns": longest,
            }
            for name, (calls, inclusive, exclusive, longest) in table.items()
        }


def log_line() -> str | None:
    """The table as one line; with the pool attached, plus "pool": the counters'
    deltas since reset() ({"dispatches", "wakes", "empty_wakes"}); with the NumPy
    pool attached, plus "numpy_pool": its stats(). After start_loop_thread_cpu(), it
    first records "loop.thread_cpu"."""
    global _loop_cpu
    if not enabled():
        return None
    if _loop_cpu is not None:
        record("loop.thread_cpu", time.thread_time_ns() - _loop_cpu)
        _loop_cpu = None
    line: dict[str, object] = dict(snapshot())
    with _lock:
        if _pool is not None:
            now = (counter() for counter in _pool)
            line["pool"] = {
                key: value - base
                for key, value, base in zip(_POOL_KEYS, now, _pool_base, strict=True)
            }
        if _numpy_pool is not None:
            line["numpy_pool"] = _numpy_pool[0]()
    table = json.dumps(line, sort_keys=True, separators=(",", ":"))
    return f"native profile: {table}"
