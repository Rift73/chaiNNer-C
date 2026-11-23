"""VOSR's worker threads, which live as long as the process, and one call's jobs on them.

On Windows a thread that runs PyTorch's CPU operators gets a team of OpenMP threads, and
the team outlives the thread: ending the thread strands its team until the process ends.
So threads made and ended per frame, per image, per video or per run would each leave a
team behind; VOSR makes each of its pools once per process and never ends it.

Calls share a pool's threads, also at the same time: a stopped run's node can go on beside
the next run for a moment, and a single node's run does not wait for a running chain. That
is safe because each call ends its own jobs only, and no job waits for another job of its
own pool: the waits go one way, from a node's thread to vosr-restore to vosr-windows, and
from a node's thread to vosr-temporal. A call's job may wait behind another call's jobs,
never for them to wait for it.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor, wait
from threading import Lock
from typing import Any, ParamSpec, TypeVar

from nodes.impl import numpy_pool

P = ParamSpec("P")
T = TypeVar("T")

_pools: dict[str, tuple[ThreadPoolExecutor, int]] = {}
_pools_lock = Lock()


def process_threads(name: str, workers: int) -> ThreadPoolExecutor:
    """The process's pool of `workers` threads named `name`, made at the first call and
    never shut down (see the module's note). A pool asked for again with another number
    of workers is a programming error."""
    with _pools_lock:
        known = _pools.get(name)
        if known is None:
            pool = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix=name,
                initializer=numpy_pool.install,
            )
            _pools[name] = (pool, workers)
            return pool
        pool, made = known
        if made != workers:
            raise ValueError(
                f"The {name} threads are {made}, not the {workers} asked for."
            )
        return pool


class Jobs(Executor):
    """One call's jobs on the process's threads. The call's end waits for every one of
    them, so that no job of a call runs on into the next call, video or run that the
    threads serve; with `cancel`, the jobs not yet started never run.

    A temporal step's item always ends with `cancel`: after a success nothing of it is
    left to start, and after a failure nothing more may start. The scene windows' calls
    never cancel: each job runs to its end, as their pool's shutdown let it."""

    def __init__(self, executor: ThreadPoolExecutor):
        self.executor = executor
        self.futures: list[Future[Any]] = []

    def submit(
        self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        future = self.executor.submit(fn, *args, **kwargs)
        self.futures.append(future)
        return future

    def end(self, cancel: bool) -> None:
        """Wait for every job; with `cancel`, the jobs not yet started never run."""
        if cancel:
            for future in self.futures:
                future.cancel()
        wait(self.futures)
