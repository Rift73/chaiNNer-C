"""chaiNNer-C's native TensorRT runner (native/trt_runner) for TensorRT's Upscale Image node and
AI Upscale's TensorRT backend, behind a hidden switch that stays off until a GPU session
verifies it: CHAINNER_C_TRT_RUNNER=1 turns it on; any other value, or none, keeps the Python
session (inference.py).

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
from pathlib import Path
from types import ModuleType

import numpy as np
from sanic.log import logger

from api import NodeId

from .cache import SharedCache
from .model import TensorRTEngine

RUNNER_VARIABLE = "CHAINNER_C_TRT_RUNNER"
# The node upscales one image at a time (its session lock), its tiles back to back on the
# GPU, so one execution context serves it.
NODE_STREAMS = 1


def runner_enabled() -> bool:
    """Whether the runner replaces the Python session: only the value "1" asks for it."""
    return os.environ.get(RUNNER_VARIABLE) == "1"


def _missing(name: str) -> RuntimeError:
    return RuntimeError(
        f"The TensorRT runner's {name} is missing: this build was made without it. Unset"
        f" {RUNNER_VARIABLE}, or build it with native/Build.ps1 -Define"
        " CHAINNER_C_TENSORRT_ROOT=<TensorRT 11 SDK>,CHAINNER_C_CUDA_ROOT=<CUDA 13 toolkit>"
        " (and CHAINNER_C_VAPOURSYNTH_INCLUDE=<VapourSynth headers> for the plugin)."
    )


def load_tensorrt() -> None:
    """Loads TensorRT's DLLs (tensorrt_libs), which the runner's delay-loaded imports find."""
    import tensorrt

    if not tensorrt.__version__.startswith("11."):
        raise RuntimeError("The TensorRT runner requires TensorRT 11.")


@lru_cache(maxsize=1)
def native() -> ModuleType:
    """The runner's binding, _chainner_trt."""
    load_tensorrt()
    try:
        return import_module("nodes.impl.tensorrt._chainner_trt")
    except ModuleNotFoundError as e:
        raise _missing("module _chainner_trt.pyd") from e


def plugin_path() -> Path:
    """The runner's VapourSynth plugin (namespace chainner_trt)."""
    path = Path(__file__).with_name("chainner_trt_vs.dll")
    if not path.is_file():
        raise _missing(f"VapourSynth plugin {path.name}")
    return path


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
