"""Temporal Fix's Use TensorRT Engine on NVIDIA GPUs: the strength's engine set for the
frame's size (tensorrt/temporal_fix.py; built once and kept as Upscale Image's temporal
sets are), run by its session (tensorrt/temporal_fix_session.py). A set that cannot be
built, or fails on a frame, leaves the node on PyTorch for the rest of the chain run,
with a warning that says why, as rdual_engine.py leaves a temporal model; that frame
runs on PyTorch at once.

This module must not import TensorRT at its top: the PyTorch package loads without it."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from threading import Lock
from typing import TYPE_CHECKING

import torch
from sanic.log import logger

from api import Aborted, NodeContext

from .auto_engine import (
    cache_folder,
    hold_engine,
    migraphx_here,
    no_engine_here,
    release_engine,
)
from .utils import Output

if TYPE_CHECKING:
    from ..tensorrt.model import TensorRTEngine

_lock = Lock()
# The nodes of this chain run whose engine set could not be built or failed.
_failed: dict[str, str] = {}
# The sets by (strength, height, width, GPU, Engine Cache folder): a set's key hashes the
# strength's weights.
_sets: dict[tuple[float, int, int, int, str], TensorRTEngine] = {}


def _forget(node_id: str) -> None:
    with _lock:
        _failed.pop(node_id, None)


def _on_pytorch(context: NodeContext, reason: str) -> None:
    """Keep the node on PyTorch for this chain run, once saying why (an empty `reason`
    says nothing)."""
    with _lock:
        _failed[context.node_id] = reason
    context.add_cleanup(partial(_forget, context.node_id), after="chain")
    if reason:
        context.warn(
            f"Temporal Fix runs on PyTorch, not through a TensorRT engine: {reason}"
        )


@dataclass(frozen=True)
class Window:
    """An item's window for the engine route: the loop run it belongs to (`owner`), the
    item, its frames (gathered before the route runs) and, when the loop reads one frame
    further, the next item's (`ahead()`, None at the clip's end); `ahead` is None when
    the route runs each window on its own."""

    owner: object
    index: int
    frames: Sequence[torch.Tensor]
    ahead: Callable[[], Sequence[torch.Tensor] | None] | None


class ReadAheadError(Exception):
    """An error while the next item's window was read: the reader's, not the engine's;
    its cause is that error."""


def unavailable(device: torch.device) -> str | None:
    """Why this machine has no engine set, or None if it may ("" for a machine without
    NVIDIA's TensorRT, which is no surprise worth a warning)."""
    reason = no_engine_here(device)
    if reason is not None:
        return reason
    if migraphx_here():
        return ""
    return None


def fix_through_engine(
    context: NodeContext,
    window: Window,
    strength: float,
    size: tuple[int, int],
    state: Callable[[], OrderedDict[str, torch.Tensor]],
    device: torch.device,
    gpu_index: int,
    pytorch: Callable[[], Output],
) -> Output:
    """The item's window (RGB CHW FP16 frames on `device`, at the network's even size)
    fixed by the strength's engine set for frames of `size` (height, width), or by
    `pytorch()` when there is none or it fails. `state()` gives the strength's weights
    for a build. With `window.ahead`, the next item's window is submitted before this
    one's result is collected (TemporalFixSession.pipelined)."""
    from packages.chaiNNer_tensorrt.settings import ENGINE_CACHE_DIRECTORY

    from ..tensorrt import temporal_fix, temporal_fix_session

    def on_pytorch() -> Output:
        release_engine(context.node_id)
        return pytorch()

    with _lock:
        failed = context.node_id in _failed
    if failed:
        return on_pytorch()
    reason = unavailable(device)
    if reason is not None:
        _on_pytorch(context, reason)
        return on_pytorch()
    height, width = size
    folder = cache_folder(
        context, "chaiNNer_tensorrt", "engine_cache", ENGINE_CACHE_DIRECTORY
    )
    where = (round(float(strength), 2), height, width, gpu_index, str(folder))
    with _lock:
        engine = _sets.get(where)
    try:
        if engine is None:
            # TensorRT's Timing Cache setting, as Build Engine reads it (None: off)
            timing = context.settings.of_package(
                "chaiNNer_tensorrt"
            ).get_cache_location("timing_cache")
            engine = temporal_fix.engine_set(
                state(), height, width, gpu_index, where[4], context, timing
            )
            with _lock:
                _sets[where] = engine
    except Aborted:
        raise
    except Exception as e:
        logger.exception("Use TensorRT Engine: no Temporal Fix engine set")
        _on_pytorch(context, f"{type(e).__name__}: {e}")
        return on_pytorch()
    hold_engine(context.node_id)
    ahead = window.ahead

    def read_ahead() -> Sequence[torch.Tensor] | None:
        assert ahead is not None
        try:
            return ahead()
        except Aborted:
            raise
        except Exception as e:
            raise ReadAheadError from e

    try:
        with temporal_fix_session.use_session(
            engine, gpu_index, context.node_id
        ) as session:
            if ahead is None:
                return session.run(window.frames)
            return session.pipelined(
                window.owner, window.index, lambda: window.frames, read_ahead
            )
    except Aborted:
        raise
    except ReadAheadError as e:
        upstream = e.__cause__
    except Exception as e:
        logger.exception("Use TensorRT Engine: the Temporal Fix engine set failed")
        _on_pytorch(context, f"{type(e).__name__}: {e}")
        return on_pytorch()
    # Not the engine's: the upstream error itself, as the PyTorch route raises it, with
    # its own cause and context (raised outside the handler, which would replace them)
    assert upstream is not None
    raise upstream


__all__ = ["Window", "fix_through_engine"]
