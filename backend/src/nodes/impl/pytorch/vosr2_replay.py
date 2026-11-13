"""PyTorch model calls of fixed shapes, replayed as CUDA graphs.

A DiT prediction issues about 3,400 small GPU operations from Python, and issuing them
took 58 of its 60 ms (RTX 5090). A graph holds the same kernels, so the same
bits, and starts them all with one call.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from sanic.log import logger


@dataclass
class _Captured:
    inputs: tuple[torch.Tensor, ...]
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor


class Replays:
    """`function` of tensors, one graph per set of input shapes.

    A shape's first call runs plainly, which also readies cuBLAS and cuDNN for it; its
    second is captured. A capture that runs out of memory (a graph's tensors take blocks of
    their own) leaves that shape plain. Inputs may be on any device; an output is borrowed
    until the next call with these shapes. A graph holds the addresses of the weights `function`
    uses: drop this object before they move.

    A capture that CUDA rejects leaves the process's GPU libraries unusable, so a file at
    `no_graphs` then keeps later launches from trying again, and this one must restart.
    AMD GPUs (ROCm's PyTorch) always run plainly.
    """

    def __init__(
        self,
        function: Callable[..., torch.Tensor],
        device: torch.device,
        no_graphs: Path,
    ):
        self.function = function
        self.device = device
        self.no_graphs = no_graphs
        self.plain = bool(torch.version.hip) or no_graphs.exists()
        # A graph's outputs stay allocated; its other tensors may share their memory with
        # those of graphs that never run at the same time.
        self.pool = None if self.plain else torch.cuda.graph_pool_handle()
        self.seen: set[tuple[torch.Size, ...]] = set()
        self.uncaptured: set[tuple[torch.Size, ...]] = set()
        self.captured: dict[tuple[torch.Size, ...], _Captured] = {}
        self.stream = None if self.plain else torch.cuda.Stream(device=device)

    def _plain(self, values: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.function(*(value.to(self.device) for value in values))

    def _capture(self, values: tuple[torch.Tensor, ...]) -> _Captured | None:
        inputs = tuple(
            torch.empty(value.shape, device=self.device, dtype=value.dtype)
            for value in values
        )
        # What torch.cuda.graph does, without its garbage collection. A graph's tensors take
        # new blocks: emptying the cache gives its unused ones back to the budget.
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.stream(self.stream):
                # Only this thread's calls can spoil the capture.
                graph.capture_begin(self.pool, capture_error_mode="thread_local")
                try:
                    output = self.function(*inputs)
                finally:
                    graph.capture_end()
        except torch.OutOfMemoryError:
            # Raised by the allocator before CUDA sees the allocation: the process is fine.
            logger.info("VOSR runs without a CUDA graph: its capture found no memory.")
            del graph
            torch.cuda.empty_cache()
            return None
        except RuntimeError as error:
            logger.exception("VOSR could not capture a CUDA graph on this GPU.")
            self.no_graphs.touch()
            raise RuntimeError(
                "VOSR could not capture a CUDA graph on this GPU. Restart chaiNNer-C: "
                "it will then run VOSR without one."
            ) from error
        return _Captured(inputs, graph, output)

    def close(self) -> None:
        """Free the graphs now, not when the garbage collector finds them."""
        self.captured.clear()

    def __call__(self, *values: torch.Tensor) -> torch.Tensor:
        key = tuple(value.shape for value in values)
        if self.plain or key not in self.seen or key in self.uncaptured:
            self.seen.add(key)
            return self._plain(values)
        entry = self.captured.get(key)
        if entry is None:
            entry = self._capture(values)
            if entry is None:
                self.uncaptured.add(key)
                return self._plain(values)
            self.captured[key] = entry
        for target, value in zip(entry.inputs, values, strict=True):
            target.copy_(value)
        entry.graph.replay()
        return entry.output
