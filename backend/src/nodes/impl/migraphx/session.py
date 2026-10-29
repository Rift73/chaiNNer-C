"""The MIGraphX runner on programs, for MIGraphX's Upscale Image node: the counterpart of the
TensorRT runner's session (tensorrt/runner.py), with the same tiles, blend weights and channel
rules (tiling.py's and convenient_upscale's). The runner uploads planes as stored and
converts, tiles and blends them on the GPU. Programs and sessions outlive a run
(tensorrt/cache.py)."""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager
from functools import lru_cache
from types import ModuleType

import numpy as np
from sanic.log import logger

from api import NodeId

from ..tensorrt.cache import SharedCache
from . import drft, dual
from .model import MIGraphXProgram
from .runner import load_migraphx

# The node upscales one image at a time (its session lock), its tiles back to back on the
# GPU, so one stream serves it.
NODE_STREAMS = 1


@lru_cache(maxsize=1)
def native() -> ModuleType:
    """The runner's binding with the DUAL and DRFT custom ops registered (each route's
    load_ops, a no-op for a build without its library), as every process that loads
    programs needs them before the first."""
    module = load_migraphx()
    dual.load_ops()
    drft.load_ops()
    return module


class RunnerSession:
    """The runner on one program, for the Upscale Image node: it upscales a whole image."""

    def __init__(self, program: MIGraphXProgram, gpu_index: int) -> None:
        self.runner = native().Runner(program.bytes, gpu_index, NODE_STREAMS)
        self._warned = False
        # One image at a time: the item window may run the node for several items at
        # once (execution_scheduler.LOCKED_DEVICE_NODES).
        self.lock = threading.Lock()

    def upscale(self, img: np.ndarray) -> np.ndarray:
        """A float32 BGR or gray image (HxW, HxWx1 or HxWx3) upscaled, as tiled_upscale
        makes it from an image with the program's input channels."""
        img = img.astype(np.float32, copy=False)
        runner = self.runner
        h, w = img.shape[:2]
        planes = 1 if img.ndim == 2 else img.shape[2]
        # The runner's result planes: the program's output channels when they differ from
        # its input's, else as many as came in.
        out_c = (
            runner.output_channels
            if runner.input_channels != runner.output_channels
            else planes
        )
        size = (h * runner.scale, w * runner.scale)
        out = np.empty(
            size if img.ndim == 2 and out_c == 1 else (*size, out_c), dtype=np.float32
        )
        # Planes in R, G, B order: a BGR image's channel views, uploaded as one block.
        inputs = [img] if img.ndim == 2 else [img[:, :, c] for c in range(planes)[::-1]]
        outputs = [out] if out.ndim == 2 else [out[:, :, c] for c in range(out_c)[::-1]]
        runner.upscale(inputs, outputs)
        failure = runner.capture_failure()
        if failure and not self._warned:
            self._warned = True
            logger.warning("%s", failure)
        return out

    def close(self) -> None:
        self.runner.close()


_sessions: SharedCache[RunnerSession] = SharedCache(RunnerSession.close)

# Load Program's programs, read once and kept across runs until no node uses them, so
# that Upscale Image keeps its session (keyed by the program): a program holds only bytes,
# so closing one frees nothing.
programs: SharedCache[MIGraphXProgram] = SharedCache(lambda _program: None)


def use_runner_session(
    program: MIGraphXProgram, gpu_index: int, node_id: NodeId
) -> AbstractContextManager[RunnerSession]:
    """The runner for a program on a GPU, kept across runs for `node_id`
    (tensorrt/cache.py)."""
    return _sessions.use(
        (program, gpu_index), node_id, lambda: RunnerSession(program, gpu_index)
    )
