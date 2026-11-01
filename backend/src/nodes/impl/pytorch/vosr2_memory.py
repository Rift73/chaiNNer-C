"""Scoped VOSR CUDA allocation admission and total-device memory checks."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import RLock

import torch

from api import Progress
from gpu import MemoryUsage, get_vram_usage
from util import waiting

from .vosr2 import MemoryPlan, VosrMemoryMode

_BUDGET_LOCK = RLock()
_HEADROOM = 800_000_000


@contextmanager
def vram_budget(
    device: torch.device,
    mode: VosrMemoryMode,
    progress: Progress,
    application_limit_gib: int = 0,
    prepare: Callable[[], None] | None = None,
) -> Iterator[Callable[[], None]]:
    """Preserve stricter allocator limits and restore state on every exit.

    The quota reserves space for non-allocator CUDA use and checks external
    allocations at suspension points. Other applications can allocate at any
    instant; this cannot impose a device-wide quota on those applications.
    VOSR calls are serialized while changing process-wide allocator state.
    """
    if device.type != "cuda":
        yield progress.suspend
        return
    # Another VOSR call's turn is not this node's time.
    with waiting():
        while not _BUDGET_LOCK.acquire(timeout=0.05):
            progress.suspend()
    try:
        if prepare is not None:
            prepare()
        with torch.cuda.device(device):
            index = torch.cuda.current_device()

        def current_usage() -> MemoryUsage:
            usage = get_vram_usage(index)
            if usage is None:
                raise RuntimeError(
                    "VOSR memory profiles require working GPU memory monitoring."
                )
            return usage

        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            usage = current_usage()
            profile = MemoryPlan.for_mode(mode).limit_bytes
            limit = usage.total if profile is None else min(profile, usage.total)
            external = max(0, usage.used - torch.cuda.memory_reserved(device))
            previous = torch.cuda.get_per_process_memory_fraction(device)
            total = torch.cuda.get_device_properties(device).total_memory
            cap = min(limit - external - _HEADROOM, int(previous * total))
            if application_limit_gib:
                cap = min(cap, application_limit_gib * 1024**3)
            advice = (
                "Close other GPU workloads, select Default GPU memory or reduce the output size."
                if mode == VosrMemoryMode.LOW
                else "Close other GPU workloads or reduce the output size."
            )
            if cap <= torch.cuda.memory_allocated(device) + 128 * 1024**2:
                raise RuntimeError(
                    f"Not enough free GPU memory for VOSR {mode.value} mode. {advice}"
                )

            def check() -> None:
                progress.suspend()
                if current_usage().used >= limit - 100_000_000:
                    raise RuntimeError(
                        f"GPU memory reached the VOSR {mode.value} safety boundary. {advice}"
                    )

            try:
                torch.cuda.set_per_process_memory_fraction(cap / total, device)
                check()
                yield check
            except torch.OutOfMemoryError as error:
                raise RuntimeError(
                    f"VOSR exhausted its {mode.value} GPU allocation budget. {advice}"
                ) from error
            finally:
                torch.cuda.set_per_process_memory_fraction(previous, device)
    finally:
        _BUDGET_LOCK.release()
