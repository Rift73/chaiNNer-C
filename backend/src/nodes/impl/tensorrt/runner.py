"""chaiNNer-C's native TensorRT runner (native/trt_runner) for AI Upscale's TensorRT backend
and TensorRT's Upscale Image node. AI Upscale uses it by default (its VapourSynth plugin);
CHAINNER_C_TRT_RUNNER=0 keeps its Python bridge (chainner_vs/upscale.py). The Upscale Image
node keeps the Python session (inference.py) unless CHAINNER_C_TRT_RUNNER=1 turns the runner
on for it too. A build without the runner's binaries uses the Python paths, with one log line.

The runner uploads planes as stored and converts, tiles and blends them on the GPU, one CUDA
stream per execution context. Its tiles, blend weights and channel rules are tiling.py's and
convenient_upscale's, so an image comes out as the Python session makes it.

This module does not import TensorRT until a runner is made."""

from __future__ import annotations

import os
import threading
from contextlib import AbstractContextManager
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType

import numpy as np
from sanic.log import logger

from api import NodeId

from .cache import SharedCache
from .drft import load_plugins
from .model import TensorRTEngine

RUNNER_VARIABLE = "CHAINNER_C_TRT_RUNNER"
MODULE = "nodes.impl.tensorrt._chainner_trt"
PLUGIN = Path(__file__).with_name("chainner_trt_vs.dll")
# The node upscales one image at a time (its session lock), its tiles back to back on the
# GPU, so one execution context serves it.
NODE_STREAMS = 1
# Binaries reported missing, each logged once per process.
_reported: set[str] = set()


def _built(present: bool, name: str) -> bool:
    """Whether a runner binary is in this build; logs its absence once."""
    if not present and name not in _reported:
        _reported.add(name)
        logger.warning(
            "The TensorRT runner's %s is not in this build: TensorRT runs on the Python"
            " path.",
            name,
        )
    return present


def runner_enabled() -> bool:
    """Whether the Upscale Image node uses the runner: when CHAINNER_C_TRT_RUNNER is "1"
    and the build has the runner's module."""
    return os.environ.get(RUNNER_VARIABLE) == "1" and _built(
        find_spec(MODULE) is not None, "module _chainner_trt.pyd"
    )


def runner_plugin() -> Path | None:
    """AI Upscale's runner plugin (namespace chainner_trt), or None for the Python bridge:
    when CHAINNER_C_TRT_RUNNER is "0", or the build lacks the plugin."""
    if os.environ.get(RUNNER_VARIABLE) == "0":
        return None
    return (
        PLUGIN
        if _built(PLUGIN.is_file(), f"VapourSynth plugin {PLUGIN.name}")
        else None
    )


def load_tensorrt() -> None:
    """Loads TensorRT's DLLs (tensorrt_libs), which the runner's delay-loaded imports find,
    and registers the DRFT plugins (drft.load_plugins) for the engines the runner loads."""
    import tensorrt

    if not tensorrt.__version__.startswith("11."):
        raise RuntimeError("The TensorRT runner requires TensorRT 11.")
    load_plugins()


@lru_cache(maxsize=1)
def native() -> ModuleType:
    """The runner's binding, _chainner_trt."""
    load_tensorrt()
    return import_module(MODULE)


class RunnerSession:
    """The runner on one engine, for the Upscale Image node (in tiled_upscale's place: it
    upscales a whole image)."""

    def __init__(self, engine: TensorRTEngine, gpu_index: int) -> None:
        self.runner = native().Runner(engine.bytes, gpu_index, NODE_STREAMS)
        self._warned = False
        # One image at a time, as TensorRTSession.lock.
        self.lock = threading.Lock()

    def upscale(self, img: np.ndarray) -> np.ndarray:
        """A float32 BGR or gray image (HxW, HxWx1 or HxWx3) upscaled, as tiled_upscale
        makes it from an image with the engine's input channels."""
        img = img.astype(np.float32, copy=False)
        runner = self.runner
        h, w = img.shape[:2]
        planes = 1 if img.ndim == 2 else img.shape[2]
        # The runner's result planes: the engine's output channels when they differ from
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


def use_runner_session(
    engine: TensorRTEngine, gpu_index: int, node_id: NodeId
) -> AbstractContextManager[RunnerSession]:
    """The runner for an engine on a GPU, kept across runs for `node_id` (see `cache`)."""
    return _sessions.use(
        (engine, gpu_index), node_id, lambda: RunnerSession(engine, gpu_index)
    )
