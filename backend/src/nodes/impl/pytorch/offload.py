"""A PyTorch forward larger than the GPU's memory, without the driver's overflow.

Windows places the allocations past the GPU's memory in shared system memory and the GPU reads
them over PCIe at every access (a 1x ESRGAN at 4000 px: over 7 minutes against 9 s). Here
PyTorch is capped below the free memory instead, and the activations unused for the longest go
to pinned RAM on a copy stream while the GPU computes; one an op needs comes back first. The
ops and their data are unchanged, so the output is the one the whole forward gives on a GPU
large enough: no tiles. Models whose peak holds long-idle activations (DUAL: 42 % of its peak
is unused for 200+ ops) gain most; one op's own tensors and cuDNN's workspace must still fit.
"""

from __future__ import annotations

import weakref
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.utils._pytree as pytree
from sanic.log import logger
from torch._ops import OpOverload
from torch.utils._python_dispatch import TorchDispatchMode

from . import memory_cap

# Left free besides PyTorch's memory: the CUDA context, cuDNN's and cuBLAS's handles, other
# programs' growth and Windows' own (a forward capped 1.5 GiB below the free memory ran at the
# speed of an overflow).
HEADROOM = 3 * 2**30
# Kept free within the cap for an op's new tensors and cuDNN's workspace, which cuDNN does not
# fail for: it takes a slower algorithm.
WORKSPACE = 2 * 2**30
# Activations smaller than this stay on the GPU.
MIN_BYTES = 32 * 2**20
# Smaller inputs never offload: their forward is short, and the per-op check would show.
MIN_PIXELS = 2_000_000

# Set once an activation moved to RAM: its pinned RAM stays PyTorch's until released.
_pinned = False

# Per model, the most PyTorch allocated during a forward per input pixel (learned).
_bytes_per_pixel: weakref.WeakKeyDictionary[torch.nn.Module, float] = (
    weakref.WeakKeyDictionary()
)


class OffloadMode(TorchDispatchMode):
    def __init__(self, device: torch.device, cap: int) -> None:
        super().__init__()
        self.device = device
        self.limit = cap - WORKSPACE
        self.copies = torch.cuda.Stream(device)
        # GPU activations, least recently used first; and the ones in RAM.
        self.resident: OrderedDict[int, weakref.ref[torch.Tensor]] = OrderedDict()
        self.moved: dict[int, weakref.ref[torch.Tensor]] = {}
        # Per moved tensor, the end of its copy to RAM (its way back waits for that copy only).
        self.copied: dict[int, torch.cuda.Event] = {}
        # Live tensors per GPU storage: one shared with a view or another tensor never moves
        # (moving one would leave the others on the old memory).
        self.sharers: dict[int, int] = {}
        # Per tensor, the count its death takes back (detached when it moves to RAM).
        self.shares: dict[int, weakref.finalize] = {}
        self.out_bytes = 0
        self.back_bytes = 0

    def _forget(self, key: int) -> None:
        self.resident.pop(key, None)
        self.moved.pop(key, None)
        self.copied.pop(key, None)
        self.shares.pop(key, None)

    def _unshare(self, storage: int) -> None:
        count = self.sharers.get(storage, 0) - 1
        if count > 0:
            self.sharers[storage] = count
        else:
            self.sharers.pop(storage, None)

    def _track(self, t: torch.Tensor, new: bool) -> None:
        if t.device != self.device:
            return
        storage = t.untyped_storage()
        if new:
            pointer = storage.data_ptr()
            self.sharers[pointer] = self.sharers.get(pointer, 0) + 1
            self.shares[id(t)] = weakref.finalize(t, self._unshare, pointer)
        if storage.nbytes() >= MIN_BYTES:
            key = id(t)
            if key not in self.resident:
                self.resident[key] = weakref.ref(
                    t, lambda _ref, key=key: self._forget(key)
                )
            self.resident.move_to_end(key)

    def _bring_back(self, t: torch.Tensor) -> None:
        copied = self.copied.pop(id(t), None)
        if copied is not None:
            torch.cuda.current_stream(self.device).wait_event(copied)
        t.data = t.data.to(self.device, non_blocking=True)
        self.moved.pop(id(t), None)
        self.back_bytes += t.untyped_storage().nbytes()
        self._track(t, new=True)

    def _move_out(self, keep: set[int]) -> bool:
        """Moves the least recently used activation outside `keep` to RAM; False when none."""
        for key, ref in list(self.resident.items()):
            t = ref()
            if t is None or key in keep:
                continue
            if self.sharers.get(t.untyped_storage().data_ptr(), 0) != 1:
                continue
            global _pinned
            _pinned = True
            host = torch.empty_like(t, device="cpu", pin_memory=True)
            self.copies.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.copies):
                host.copy_(t, non_blocking=True)
            copied = torch.cuda.Event()
            copied.record(self.copies)
            self.copied[key] = copied
            # Its GPU memory is reused only once the copy read it.
            t.record_stream(self.copies)
            self.out_bytes += t.untyped_storage().nbytes()
            share = self.shares.pop(key, None)
            if share is not None:
                share.detach()
            self._unshare(t.untyped_storage().data_ptr())
            t.data = host
            self.resident.pop(key, None)
            self.moved[key] = ref
            return True
        return False

    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        kwargs = kwargs or {}
        inputs = [
            a for a in pytree.tree_leaves((args, kwargs)) if isinstance(a, torch.Tensor)
        ]
        keep = {id(t) for t in inputs}
        while torch.cuda.memory_allocated(self.device) > self.limit:
            if not self._move_out(keep):
                break
        while True:
            try:
                for t in inputs:
                    if id(t) in self.moved:
                        self._bring_back(t)
                out = func(*args, **kwargs)
                break
            except torch.OutOfMemoryError:
                if not self._move_out(keep):
                    raise
        for t in inputs:
            if id(t) in self.resident:
                self.resident.move_to_end(id(t))
        known = {id(t) for t in inputs}
        for t in pytree.tree_leaves(out):
            if isinstance(t, torch.Tensor):
                if id(t) not in known and t.device == self.device:
                    pointer = t.untyped_storage().data_ptr()
                    if pointer not in self.sharers and any(
                        i.untyped_storage().data_ptr() == pointer for i in inputs
                    ):
                        # A view of a tensor from before the forward: that one shares it.
                        self.sharers[pointer] = 1
                self._track(t, new=id(t) not in known)
        return out


@contextmanager
def offloaded(module: torch.nn.Module, x: torch.Tensor) -> Iterator[None]:
    """Runs a forward of `module` on `x` (a CUDA batch, NCHW) with its activations moved to
    RAM when they would not fit in the GPU's free memory; as is otherwise."""
    pixels = x.shape[-2] * x.shape[-1]
    device = x.device
    if device.type != "cuda" or pixels < MIN_PIXELS:
        yield
        return
    free, total = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    previous = torch.cuda.get_per_process_memory_fraction(device)
    cap = min(free + reserved - HEADROOM, int(previous * total))
    if cap <= WORKSPACE:
        raise torch.OutOfMemoryError(
            "CUDA memory is insufficient for activation offload and its workspace."
        )
    known = _bytes_per_pixel.get(module)
    torch.cuda.reset_peak_memory_stats(device)
    mode = None
    if known is None or allocated + known * pixels > cap - WORKSPACE:
        mode = OffloadMode(device, cap)
    try:
        if mode is None:
            yield
        else:
            with memory_cap.capped(device, cap / total), mode:
                yield
    finally:
        peak = torch.cuda.max_memory_allocated(device) - allocated
        if mode is None or mode.out_bytes == 0:
            # A forward that moved nothing saw its whole peak.
            _bytes_per_pixel[module] = max(peak / pixels, known or 0.0)
        if mode is not None and mode.out_bytes:
            logger.info(
                "VRAM offload: %.1f GiB to RAM and %.1f GiB back (cap %.1f GiB)",
                mode.out_bytes / 2**30,
                mode.back_bytes / 2**30,
                cap / 2**30,
            )


def release_pinned_memory() -> None:
    """Gives the RAM pinned for moved activations back to Windows (2026-10-20: when a
    run ends, so an idle worker holds no locked RAM; a run's later images reuse it). PyTorch
    keeps it until the process ends otherwise, and 2.14 frees it only through this internal
    call (the runtime pins torch)."""
    global _pinned
    if _pinned:
        _pinned = False
        # Only GPU builds of torch have it, and only a GPU forward pins.
        from torch._C import _host_emptyCache

        _host_emptyCache()
