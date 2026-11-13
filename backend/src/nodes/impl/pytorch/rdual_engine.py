"""Upscale Image's Use TensorRT Engine for a temporal model (RDUAL) on NVIDIA GPUs: the
model's engine set for the frame's size (tensorrt/rdual.py, built once and kept as the
image route keeps its engines), run by its session (tensorrt/rdual_session.py) as the
temporal path's executor. A set that cannot be built, or fails on a frame, leaves the
model on PyTorch for the rest of the chain run, with a warning that says why; that frame
runs on PyTorch at once, its planner starting cold (another executor's key).

This module must not import TensorRT at its top: the PyTorch package loads without it."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from threading import Lock
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import numpy as np
import torch
from sanic.log import logger
from spandrel import ImageModelDescriptor

from api import Aborted, NodeContext

from ..upscale.basic_upscale import PAD_SIZE, PaddingType
from ..upscale.temporal_window import Fit, temporal_upscale
from ..upscale.tile_report import TileReport
from .auto_engine import cache_folder, hold_engine, release_engine
from .rdual_window import RDUALWindows

if TYPE_CHECKING:
    from ..tensorrt.model import TensorRTEngine

_lock = Lock()
# The models of this chain run whose engine set could not be built or failed.
_failed: WeakKeyDictionary[ImageModelDescriptor, str] = WeakKeyDictionary()
# Each model's sets by (height, width, GPU, Engine Cache folder), while the model lives:
# a set's key hashes the model's weights, which took about 45 ms a frame at 1488x837.
_sets: WeakKeyDictionary[
    ImageModelDescriptor, dict[tuple[int, int, int, str], TensorRTEngine]
] = WeakKeyDictionary()


def _forget(model: ImageModelDescriptor) -> None:
    with _lock:
        _failed.pop(model, None)


def _on_pytorch(context: NodeContext, model: ImageModelDescriptor, reason: str) -> None:
    """Keep the model on PyTorch for this chain run, once saying why (an empty `reason`
    says nothing)."""
    with _lock:
        _failed[model] = reason
    context.add_cleanup(partial(_forget, model), after="chain")
    if reason:
        context.warn(
            f"Upscale Image runs this {model.architecture.name} model on PyTorch, not"
            f" through a TensorRT engine: {reason}"
        )


def temporal_upscale_through_engine(
    context: NodeContext,
    img: np.ndarray,
    model: ImageModelDescriptor,
    device: torch.device,
    windows: Callable[[], RDUALWindows],
    gpu_index: int,
    report: TileReport,
    padding: PaddingType,
    fit: Fit,
) -> np.ndarray:
    """`img` on the temporal path through the model's engine set, or on PyTorch's
    executor `windows()` (with `fit`) when it has none or the set fails: only then is
    it made, which moves the model to `device`."""
    from packages.chaiNNer_tensorrt.settings import ENGINE_CACHE_DIRECTORY

    from ..tensorrt import rdual
    from ..tensorrt.rdual_session import temporal_engine_upscale

    def pytorch() -> np.ndarray:
        release_engine(context.node_id)
        return temporal_upscale(context, img, windows(), report, padding, fit)

    with _lock:
        failed = model in _failed
    if failed:
        return pytorch()
    reason = rdual.unavailable(model, device)
    if reason is not None:
        _on_pytorch(context, model, reason)
        return pytorch()
    height, width = img.shape[:2]
    if padding != PaddingType.NONE:
        height, width = height + 2 * PAD_SIZE, width + 2 * PAD_SIZE
    folder = cache_folder(
        context, "chaiNNer_tensorrt", "engine_cache", ENGINE_CACHE_DIRECTORY
    )
    where = (height, width, gpu_index, str(folder))
    with _lock:
        engine = _sets.get(model, {}).get(where)
    try:
        if engine is None:
            # TensorRT's Timing Cache setting, as Build Engine reads it (None: off)
            timing = context.settings.of_package(
                "chaiNNer_tensorrt"
            ).get_cache_location("timing_cache")
            engine = rdual.engine_set(
                model, height, width, gpu_index, where[3], context, timing
            )
            with _lock:
                _sets.setdefault(model, {})[where] = engine
    except Aborted:
        raise
    except Exception as e:
        logger.exception("Use TensorRT Engine: no engine set for this RDUAL model")
        _on_pytorch(context, model, f"{type(e).__name__}: {e}")
        return pytorch()
    hold_engine(context.node_id)
    try:
        return temporal_engine_upscale(context, img, engine, gpu_index, report, padding)
    except Aborted:
        raise
    except Exception as e:
        logger.exception("Use TensorRT Engine: the RDUAL engine set failed on a frame")
        _on_pytorch(context, model, f"{type(e).__name__}: {e}")
        return pytorch()


__all__ = ["temporal_upscale_through_engine"]
