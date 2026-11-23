"""Scoped VOSR CUDA allocation admission and total-device memory checks."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import RLock

import torch
from sanic.log import logger

from api import Progress
from gpu import MemoryUsage, get_vram_usage
from util import waiting

from .vosr2 import MemoryPlan, VosrMemoryMode

_BUDGET_LOCK = RLock()
HEADROOM = 800_000_000
# Whether the budget in force is Low's limit less the device's other use and HEADROOM
# (another bound may be smaller); set under _BUDGET_LOCK.
_cap_from_limit = False


def available_memory(device: torch.device) -> float:
    """What the allocator may still hand out on `device` under the current budget."""
    total = torch.cuda.get_device_properties(device).total_memory
    return total * torch.cuda.get_per_process_memory_fraction(
        device
    ) - torch.cuda.memory_allocated(device)


def _advice(mode: VosrMemoryMode) -> str:
    return (
        "Close other GPU workloads, select Default GPU memory or reduce the output size."
        if mode == VosrMemoryMode.LOW
        else "Close other GPU workloads or reduce the output size."
    )


def short_of_memory(mode: VosrMemoryMode, figures: str = "") -> str:
    """The refusal when the device leaves `mode` too little memory to start; `figures`
    (a sentence) says by how much."""
    return f"Not enough free GPU memory for VOSR {mode.value} mode.{figures} {_advice(mode)}"


def in_use(usage: MemoryUsage) -> int:
    """The device's use as nvidia-smi and Task Manager show it: without the driver's own
    reserve, which NVML's older reading counts as used."""
    return usage.used - usage.reserved


def chain_share(device: torch.device, own: int) -> int:
    """What this process's PyTorch memory other than VOSR's `own` kept allocations holds
    on `device`: the chain's other GPU nodes with the blocks PyTorch keeps cached for them,
    and, during an image, VOSR's own working memory of that image."""
    return max(0, torch.cuda.memory_reserved(device) - own)


def held_figures(usage: MemoryUsage, chain: int | None) -> str:
    """A refusal's sentence on what holds the device: its use as nvidia-smi shows it, and
    the other nodes of the chain's share of it when known and not negligible."""
    held = (
        f", {chain / 1e9:.1f} GB of it held by this chain outside VOSR's loaded model"
        if chain is not None and chain >= 50_000_000
        else ""
    )
    return f" The GPU reports {in_use(usage) / 1e9:.1f} GB in use{held}."


def cap_from_limit() -> bool:
    """Whether the budget in force was set by its mode's limit, the device's other use and
    HEADROOM, and not by a smaller bound (the application's GPU budget, a stricter
    allocator limit)."""
    return _cap_from_limit


def log_refusal(
    device: torch.device,
    reason: str,
    reading: Callable[[], MemoryUsage],
    external: int,
    limit: int,
    cap: int,
) -> None:
    """One line with every reading a refusal of VOSR's budget was decided on: the
    driver's count (`reading`) and CUDA's own, which need not agree. A failure to read
    them is logged in full and never replaces the refusal itself."""
    try:
        usage = reading()
        free, total = torch.cuda.mem_get_info(device)
        mib = 1024**2
        logger.warning(
            "VOSR memory refusal (%s), MiB: NVML used %d (driver reserve %d) of %d; "
            "CUDA free %d of %d; torch reserved %d, allocated %d; external %d; "
            "limit %d; cap %d",
            reason,
            usage.used // mib,
            usage.reserved // mib,
            usage.total // mib,
            free // mib,
            total // mib,
            torch.cuda.memory_reserved(device) // mib,
            torch.cuda.memory_allocated(device) // mib,
            external // mib,
            limit // mib,
            cap // mib,
        )
    except Exception:
        logger.exception(
            "VOSR could not read the memory figures of its refusal (%s)", reason
        )


def unreserved_memory(device: torch.device) -> float:
    """What the allocator may still reserve on `device` under the current budget, after
    giving back its unused blocks: a new block this large fits, whatever the free space
    inside the reserved ones."""
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(device).total_memory
    return total * torch.cuda.get_per_process_memory_fraction(
        device
    ) - torch.cuda.memory_reserved(device)


@contextmanager
def vram_budget(
    device: torch.device,
    mode: VosrMemoryMode,
    progress: Progress,
    application_limit_gib: int = 0,
    prepare: Callable[[], None] | None = None,
    shrink: Callable[[], bool] | None = None,
    own: Callable[[], int] | None = None,
) -> Iterator[Callable[[], None]]:
    """Preserve stricter allocator limits and restore state on every exit.

    The quota reserves space for non-allocator CUDA use and checks external
    allocations at suspension points. Other applications can allocate at any
    instant; this cannot impose a device-wide quota on those applications.
    VOSR calls are serialized while changing process-wide allocator state.

    `shrink`, when this call would be refused (too little budget, or the device's use
    already at the limit's boundary), gives back resident weights, True when it did:
    the budget is then measured again. `own`, the bytes of PyTorch's allocations that
    are VOSR's, lets a refusal say what the chain's other nodes hold.
    """
    global _cap_from_limit
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
            profile = MemoryPlan.for_mode(mode).limit_bytes
            previous = torch.cuda.get_per_process_memory_fraction(device)
            total = torch.cuda.get_device_properties(device).total_memory

            def measure() -> tuple[MemoryUsage, int, int, int, int]:
                """The device's use, the limit on it, other use, the cap, and the cap
                the limit alone sets."""
                torch.cuda.empty_cache()
                usage = current_usage()
                # A profile limits the use as users read it, without the driver's own
                # reserve; never more than the device has. Default: the whole device,
                # the reserve still counted as used.
                limit = (
                    usage.total
                    if profile is None
                    else min(profile + usage.reserved, usage.total)
                )
                external = max(0, usage.used - torch.cuda.memory_reserved(device))
                from_limit = limit - external - HEADROOM
                cap = min(from_limit, int(previous * total))
                if application_limit_gib:
                    cap = min(cap, application_limit_gib * 1024**3)
                return usage, limit, external, cap, from_limit

            usage, limit, external, cap, from_limit = measure()
            if (
                shrink is not None
                and (
                    cap <= torch.cuda.memory_allocated(device) + 128 * 1024**2
                    or usage.used >= limit - 100_000_000
                )
                and shrink()
            ):
                usage, limit, external, cap, from_limit = measure()
            _cap_from_limit = profile is not None and cap == from_limit
            advice = _advice(mode)

            def figures(reading: MemoryUsage) -> str:
                return held_figures(
                    reading, None if own is None else chain_share(device, own())
                )

            if cap <= torch.cuda.memory_allocated(device) + 128 * 1024**2:
                log_refusal(
                    device,
                    "not enough free memory",
                    lambda: usage,
                    external,
                    limit,
                    cap,
                )
                raise RuntimeError(short_of_memory(mode, figures(usage)))

            def check() -> None:
                progress.suspend()
                now = current_usage()
                if now.used >= limit - 100_000_000:
                    log_refusal(
                        device, "safety boundary", lambda: now, external, limit, cap
                    )
                    raise RuntimeError(
                        f"GPU memory reached the VOSR {mode.value} safety boundary."
                        f"{figures(now)} {advice}"
                    )

            try:
                torch.cuda.set_per_process_memory_fraction(cap / total, device)
                check()
                yield check
            except torch.OutOfMemoryError as error:
                log_refusal(
                    device, "budget exhausted", current_usage, external, limit, cap
                )
                raise RuntimeError(
                    f"VOSR exhausted its {mode.value} GPU allocation budget."
                    f"{figures(usage)} {advice}"
                ) from error
            finally:
                torch.cuda.set_per_process_memory_fraction(previous, device)
    finally:
        _cap_from_limit = False
        _BUDGET_LOCK.release()
