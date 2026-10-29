"""PyTorch's per-process CUDA memory fraction, which scopes on different threads cap at
once: Upscale Image's VRAM offload (on the GPU thread, nodes/impl/device_thread.py) and
VOSR's allocation budget (on the run's driver, its stages on their own threads). Each
scope's cap holds while the scope is active: a device's fraction is the smallest active
cap, set again whenever a scope starts, lowers its cap or ends, and the fraction from
before the first scope once none is left. Each restoring the value it read at its start
left a scope's cap behind when two interleaved. No lock is held while a scope is active
(VOSR looks at later frames inside its budget, which may wait on an upscale)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock

import torch


@dataclass(eq=False)
class _Cap:
    fraction: float
    lowerable: bool


_lock = Lock()
_active: dict[torch.device, list[_Cap]] = {}
_outside: dict[torch.device, float] = {}


def _key(device: torch.device) -> torch.device:
    """The device with its index: torch.device("cuda") is the current one, as in
    torch.cuda's own functions (only a device without an index equals it)."""
    if device != torch.device(device.type):
        return device
    return torch.device(device.type, torch.cuda.current_device())


def _apply(device: torch.device) -> None:
    """The smallest active cap as the device's fraction (_lock held)."""
    fraction = min(cap.fraction for cap in _active[device])
    torch.cuda.set_per_process_memory_fraction(fraction, device)


@contextmanager
def capped(
    device: torch.device, fraction: float, lowerable: bool = False
) -> Iterator[None]:
    """Caps PyTorch's allocations on the CUDA device to `fraction` of its memory while
    inside, or lower where another active scope caps it lower. A lowerable scope (VOSR's
    budget, one at a time) is the one `lower` lowers."""
    device = _key(device)
    cap = _Cap(fraction, lowerable)
    with _lock:
        if device not in _active:
            _outside[device] = torch.cuda.get_per_process_memory_fraction(device)
            _active[device] = []
        active = _active[device]
        active.append(cap)
        try:
            _apply(device)
        except BaseException:
            # The fraction stays as it was: the cap leaves, and with the last the device.
            active.remove(cap)
            if not active:
                del _active[device], _outside[device]
            raise
    try:
        yield
    finally:
        with _lock:
            active.remove(cap)
            if active:
                _apply(device)
            else:
                del _active[device]
                torch.cuda.set_per_process_memory_fraction(_outside.pop(device), device)


def lower(device: torch.device, fraction: float) -> None:
    """Lowers the cap of the device's active lowerable scope to `fraction` (VOSR's
    stages, which admit their weights inside its budget, on any thread); without one, the
    device's fraction itself, which the caller then restores."""
    device = _key(device)
    with _lock:
        caps = [cap for cap in _active.get(device, []) if cap.lowerable]
        if not caps:
            torch.cuda.set_per_process_memory_fraction(fraction, device)
            return
        caps[-1].fraction = min(caps[-1].fraction, fraction)
        _apply(device)
