"""Temporal VOSR in a Load Video loop: one output per item, each frame restored once.

The loop runs Upscale Image (VOSR) once per frame, in order. A TemporalStep lives in the
node's iteration state for one loop run and keeps what later outputs still need: the
restorations of the previous, current and later frames (later frames come from the item
window's look-ahead), the motion between neighbours, the observations of a scene start and
Persistent Detail's detail state. With the look-ahead, output i is the frame i that the old
generator pipeline (the Temporal Sequence node) yielded for the same restorations.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Literal

import numpy as np

from api import Aborted, NodeContext, NodeId
from api.settings import SettingsParser
from nodes.impl import numpy_pool

from .vosr2_temporal import (
    Correspondence,
    RestoredFrame,
    TemporalMode,
    TemporalObservation,
    motion_pair,
    stabilize,
    validate_image,
)
from .vosr2_temporal_state import DetailState, carry_detail, initialize_detail

# Persistent Detail starts a scene from up to five observations; the fifth is stabilized
# with the frame after it, so a scene start reads five frames ahead.
SCENE_START = 5

Restore = Callable[[np.ndarray, NodeContext], np.ndarray]
Peek = Callable[[int], object | None]
Pair = tuple[Correspondence, Correspondence]
# The CPU work's threads per item, besides the node's own. At 2560x1440, stabilize took
# 536 ms on one thread, 458 ms with one more, 384 ms with two, and no less with three
# or four.
CPU_THREADS = 2


def frames_ahead(method: TemporalMode) -> int:
    """The frames ahead of the current one that the method uses."""
    return SCENE_START if method == TemporalMode.PERSISTENT_DETAIL else 1


class _RestoreContext(NodeContext):
    """The item's context for its one background restoration, stopped with the item."""

    def __init__(self, parent: NodeContext, stopped: Event):
        self.parent = parent
        self.stopped = stopped

    @property
    def aborted(self) -> bool:
        return self.stopped.is_set() or self.parent.aborted

    @property
    def paused(self) -> bool:
        return self.parent.paused

    @property
    def settings(self) -> SettingsParser:
        return self.parent.settings

    @property
    def node_id(self) -> NodeId:
        return self.parent.node_id

    @property
    def storage_dir(self) -> Path:
        return self.parent.storage_dir

    def set_progress(self, progress: float) -> None:
        self.suspend()
        self.parent.set_progress(progress)

    def add_cleanup(
        self, fn: Callable[[], None], after: Literal["node", "chain"] = "chain"
    ) -> None:
        self.parent.add_cleanup(fn, after)


@dataclass(frozen=True)
class _Item:
    """One item's call: the frame it outputs, its look-ahead and its restorer. `pool`
    restores one later frame (overlap); `cpu` runs the stabilization's independent CPU
    work and the motion pairs whose frames are known."""

    index: int
    image: np.ndarray
    context: NodeContext
    peek: Peek
    restore: Restore
    pool: ThreadPoolExecutor | None
    cpu: ThreadPoolExecutor
    background: NodeContext


@dataclass
class _Sequence:
    """What later outputs of a sequence still need. One starts with the loop run, and
    again after an item that never reached the node."""

    shape: tuple[int, ...] | None = None
    end: int | None = None  # the first frame known not to reach the node
    frames: dict[int, RestoredFrame | Future[RestoredFrame]] = field(
        default_factory=dict
    )
    pairs: dict[int, Pair | Future[Pair]] = field(default_factory=dict)
    observations: dict[int, TemporalObservation] = field(default_factory=dict)
    detail: DetailState | None = None


def _is_cut(observation: TemporalObservation) -> bool:
    match = observation.previous_match
    return match is not None and not np.any(match.confidence)


class TemporalStep:
    """One Load Video loop run of temporal VOSR, kept in the node's iteration state.

    Three Frames stabilizes frame i's restoration with those of frames i - 1 and i + 1;
    Persistent Detail carries motion-tracked fine detail across those outputs and starts
    each scene (the first frame, a detected cut) from a consensus of up to five of them.
    `lookahead` is how far the item window's peek reaches. When it is shorter than the
    method needs, the step uses past frames only: Three Frames stabilizes with frame i - 1
    alone, and Persistent Detail starts each scene from its first frame alone.

    With `overlap`, one later frame's restoration runs on a worker thread while the current
    output is computed on the CPU; it always finishes before the item's output is returned.
    The CPU work itself uses CPU_THREADS more threads per item, also joined before the
    output is returned: each motion pair starts once both frames' inputs are taken, and
    stabilize runs its independent parts in parallel; the outputs are the same bit for bit.
    """

    def __init__(
        self,
        method: TemporalMode,
        strength: float,
        lookahead: int,
        overlap: bool,
        inputs: tuple[object, ...],
    ):
        if not np.isfinite(strength) or not 0 < strength <= 1:
            raise ValueError("Temporal strength must be above 0 and at most 1.")
        self.method = method
        self.strength = strength
        self.ahead = lookahead if lookahead >= frames_ahead(method) else 0
        self.overlap = overlap
        self.inputs = inputs
        self.index: int | None = None
        self.sequence = _Sequence()

    def frame(
        self,
        context: NodeContext,
        index: int,
        image: np.ndarray,
        peek: Peek,
        restore: Restore,
        inputs: tuple[object, ...],
    ) -> np.ndarray:
        """Item `index`'s output. `peek(offset)` is the image of item index + offset, or
        None when it does not reach the node; `restore` is the node's own restoration."""
        if inputs != self.inputs:
            raise ValueError(
                "Temporal VOSR needs the same model and settings for every frame of a video."
            )
        if self.index is not None and index != self.index + 1:
            # An item failed before it reached the node: a new sequence starts here.
            self.sequence = _Sequence()
        self.index = index
        stopped = Event()
        pool = (
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="vosr-restore",
                initializer=numpy_pool.install,
            )
            if self.overlap
            else None
        )
        cpu = ThreadPoolExecutor(
            max_workers=CPU_THREADS,
            thread_name_prefix="vosr-temporal",
            initializer=numpy_pool.install,
        )
        item = _Item(
            index,
            image,
            context,
            peek,
            restore,
            pool,
            cpu,
            _RestoreContext(context, stopped),
        )
        try:
            if self.method == TemporalMode.PERSISTENT_DETAIL:
                output = self._persistent(item)
            else:
                output = self._observation(item, index).image
            # No restoration may still run when downstream nodes receive the frame,
            # and no CPU work: the later items take the finished motion pairs.
            wait([f for f in self.sequence.frames.values() if isinstance(f, Future)])
            wait([f for f in self.sequence.pairs.values() if isinstance(f, Future)])
            self.sequence.observations.pop(index, None)
            return output
        finally:
            stopped.set()
            cpu.shutdown(wait=True, cancel_futures=True)
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
            # A restoration or pair that never started or that the item's stop ended (the
            # item failed) is made again if needed: its Aborted is not a later item's.
            for jobs in (self.sequence.frames, self.sequence.pairs):
                for k, job in list(jobs.items()):
                    if isinstance(job, Future) and (
                        job.cancelled() or isinstance(job.exception(), Aborted)
                    ):
                        del jobs[k]

    def _persistent(self, item: _Item) -> np.ndarray:
        observation = self._observation(item, item.index)
        check = item.context.suspend
        if self.sequence.detail is None or _is_cut(observation):
            scene = [observation]
            for k in range(item.index + 1, item.index + SCENE_START):
                if self._restored(item, k) is None:
                    break
                following = self._observation(item, k)
                if _is_cut(following):
                    break
                scene.append(following)
            output, self.sequence.detail = initialize_detail(
                scene, self.strength, check
            )
            return output
        output, self.sequence.detail = carry_detail(
            observation, self.sequence.detail, self.strength, check
        )
        return output

    def _observation(self, item: _Item, k: int) -> TemporalObservation:
        """Frame k's Three Frames output, with its input and match to frame k - 1."""
        observation = self.sequence.observations.get(k)
        if observation is not None:
            return observation
        current = self._restored(item, k)
        assert current is not None, f"Frame {k} has no restoration"
        neighbors: list[tuple[RestoredFrame, Correspondence]] = []
        previous = self._restored(item, k - 1)
        previous_match = None
        if previous is not None:
            previous_match = self._pair(item, k, previous, current)[1]
            neighbors.append((previous, previous_match))
        following = self._restored(item, k + 1)
        if following is not None:
            neighbors.append(
                (following, self._pair(item, k + 1, current, following)[0])
            )
        image = stabilize(
            current, neighbors, self.strength, item.context.suspend, item.cpu
        )
        observation = TemporalObservation(image, current.source, previous_match)
        self.sequence.observations[k] = observation
        # Frame k - 1 and its match were last needed here.
        self.sequence.frames.pop(k - 1, None)
        self.sequence.pairs.pop(k, None)
        return observation

    def _pair(
        self, item: _Item, k: int, first: RestoredFrame, second: RestoredFrame
    ) -> Pair:
        """The matches of frames k - 1 and k, in both directions."""
        pair = self.sequence.pairs.get(k)
        if isinstance(pair, Future):
            pair = pair.result()
            self.sequence.pairs[k] = pair
        elif pair is None:
            pair = motion_pair(first.source, second.source, item.context.suspend)
            self.sequence.pairs[k] = pair
        return pair

    def _start_pair(self, item: _Item, k: int, image: np.ndarray) -> None:
        """Start the matches of frames k - 1 and k on the CPU threads once frame k's image
        is taken, so that they run during its restoration or the current frame's CPU
        work. They need the frames' inputs only, the same ones _pair would use."""
        previous = self.sequence.frames.get(k - 1)
        if k in self.sequence.pairs or not isinstance(previous, RestoredFrame):
            return
        self.sequence.pairs[k] = item.cpu.submit(
            motion_pair, previous.source, image, item.background.suspend
        )

    def _restored(self, item: _Item, k: int) -> RestoredFrame | None:
        """Frame k's restoration, made once; None when frame k does not reach the node
        (before the sequence, released, beyond the look-ahead or the sequence's end)."""
        frame = self.sequence.frames.get(k)
        if isinstance(frame, Future):
            frame = frame.result()
            self.sequence.frames[k] = frame
        elif frame is None:
            image = self._image(item, k)
            if image is None:
                return None
            self._check(image)
            # Frames are taken in order and the one in flight is the next one, so two
            # restorations never share the device.
            assert not any(
                isinstance(f, Future) and not f.done()
                for f in self.sequence.frames.values()
            ), f"Frame {k} is restored while another restoration runs"
            self._start_pair(item, k, image)
            frame = RestoredFrame.create(image, item.restore(image, item.context))
            self.sequence.frames[k] = frame
        if item.pool is not None:
            self._prefetch(item, k + 1)
        return frame

    def _prefetch(self, item: _Item, k: int) -> None:
        """Start frame k's restoration in the background when no restoration runs."""
        assert item.pool is not None
        if k in self.sequence.frames or any(
            isinstance(f, Future) for f in self.sequence.frames.values()
        ):
            return
        image = self._image(item, k)
        if image is None:
            return
        self._check(image)
        self.sequence.frames[k] = item.pool.submit(
            lambda: RestoredFrame.create(image, item.restore(image, item.background))
        )
        self._start_pair(item, k, image)

    def _image(self, item: _Item, k: int) -> np.ndarray | None:
        if k == item.index:
            return item.image
        if k < item.index or k - item.index > self.ahead:
            return None
        if self.sequence.end is not None and k >= self.sequence.end:
            return None
        image = item.peek(k - item.index)
        if image is None:  # the video ended, or that item fails at its own turn
            self.sequence.end = k
            return None
        assert isinstance(image, np.ndarray), "Peeked frame is not an image"
        return image

    def _check(self, image: np.ndarray) -> None:
        validate_image(image)
        if self.sequence.shape is not None and image.shape != self.sequence.shape:
            raise ValueError(
                "All temporal sequence frames must have identical dimensions."
            )
        self.sequence.shape = image.shape
