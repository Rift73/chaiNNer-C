"""Upscale Image's temporal path: which frames a temporal model (RDUAL) runs together, and
the state it carries from one window to the next.

RDUAL reads the frames around each one through two sweeps, a reverse one for the frames
after it and a forward one that carries a state from frame to frame. Inside a loop the
node sees one item at a time. At an item no earlier window kept, it runs a window of this
frame and the frames it can read ahead, carrying the state of the window before; the
window keeps the outputs of its frames that have LOOK_AHEAD frames after them in it (all
of them at the video's end) for the items that follow. The frames after those are its
look-ahead, read by the reverse sweep only; the next window starts at the first of them,
so every frame is upscaled once.

The planner and `temporal_upscale` are free of any framework, generic over the frame, the
output and the state: an executor (`Executor`, PyTorch's or a TensorRT session) plugs in
without a change to them."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

import numpy as np
from sanic.log import logger

from api import NodeContext
from nodes.utils.utils import Padding, get_h_w_c

from ..image_utils import as_target_channels, create_border
from ..native_framework_shared import assemble_alpha
from ..resize import ResizeFilter, resize
from .basic_upscale import PAD_SIZE, PaddingType
from .tile_report import TileReport
from .tiler import NoTiling, Tiler

F = TypeVar("F")
O = TypeVar("O")
S = TypeVar("S")

# Frames before the first wanted one (Run this frame, a Range's start) that run only to
# build the state, and frames after a window's last kept one that its reverse sweep reads.
# A cold start lacks the past; more frames ahead changed no output measurably.
WARM_UP = 5
LOOK_AHEAD = 1

F_contra = TypeVar("F_contra", contravariant=True)
O_co = TypeVar("O_co", covariant=True)

# The input of the loop's frames, which the node reads ahead.
IMAGE_INPUT = 1


class Run(Protocol[F_contra, O_co, S]):
    """A window's run: `run(frames, keep, past, first) -> (outputs, state)`.

    - `frames`: the window's frames in order, the item's own first, at least `keep`.
    - `keep`: how many of the first frames get outputs (1 or more); the frames after them
      are the look-ahead, which the reverse sweep reads and which give no output.
    - `past`: None to start cold, as at a clip's start; else `(state, frame)`, the state
      an earlier call returned and that call's frame `frames[keep - 1]`, the frame just
      before this window's first, which links the state to it.
    - `first`: the loop index of `frames[0]`, so an executor may reuse what it computed
      for a frame an earlier window also read (the look-ahead frames), by index.
    - returns `keep` outputs, one per kept frame in order, which the planner holds until
      their items take them, and the state after frame `keep - 1` for the next window.

    Raising leaves the planner without a state: the next window starts cold."""

    def __call__(
        self,
        frames: Sequence[F_contra],
        keep: int,
        past: tuple[S, F_contra] | None,
        first: int,
        /,
    ) -> tuple[Sequence[O_co], S]: ...


class Executor(Protocol[F, O, S]):
    """A temporal model's runtime. `convert` turns a 3-channel BGR float32 frame, already
    padded, into a model frame; `run` is a window's `Run`; `to_image` turns an output
    into a 3-channel BGR float32 image in 0..1. `key` names the runtime: a change (a
    fallback to another runtime mid-run, another model) starts the next window cold.

    Optional: `with_tile(tile) -> Executor`, the executor running each window in tiles of
    `tile` (width, height) inside its own call, with a key of its own (its state holds
    one state per tile, so a change of tiles starts cold); without it, the fit offers no
    tiles. Optional: `fitting_ahead(shape, ahead) -> int`, the frames read ahead (at most
    `ahead`) whose window of frames of `shape` (height, width, after padding) fits the
    device now in this executor's mode and tiles, or -1 when not even a window of one
    frame fits. A
    runtime that knows or estimates its memory says so before the first window: the
    temporal path chooses its first step by it and caps each window's frames ahead by it,
    and running out of memory stays the second line of defence (`Fit`)."""

    @property
    def key(self) -> Hashable: ...

    def convert(self, frame: np.ndarray, /) -> F: ...

    def run(
        self, frames: Sequence[F], keep: int, past: tuple[S, F] | None, first: int, /
    ) -> tuple[Sequence[O], S]: ...

    def to_image(self, output: O, /) -> np.ndarray: ...


class WindowPlanner(Generic[F, O, S]):
    """Per loop run: the outputs of the last window not yet taken, and the state it
    carries with the index and frame it was carried from. `shape` and `key` are the
    frames' and the executor's it plans for."""

    def __init__(self, shape: tuple[int, ...], key: Hashable, look_ahead: int):
        self.shape = shape
        self.key = key
        self.look_ahead = look_ahead
        self.past: tuple[int, S, F] | None = None
        self.outputs: dict[int, O] = {}

    def output(
        self,
        index: int,
        current: Callable[[], F],
        peek: Callable[[int], F | None],
        ahead: int,
        run: Run[F, O, S],
    ) -> O:
        """The output of item `index`. `current` converts its frame and `peek(k)` the
        frame k items ahead, None past the video's end; `ahead` is how many items ahead
        can be read.

        An item the last window keeps takes its output from that window. Otherwise a
        new window holds this frame and every frame read ahead, and keeps this item and
        the ones after it with `look_ahead` frames after them, at least this one, or all
        of them when the window reached the video's end. It carries the last window's
        state when that window's last kept item is the one just before; a missing
        earlier item (one this node did not run for), a failed window or an item asked
        again after it took its output starts cold."""
        self.__window(index, current, peek, ahead, run)
        return self.outputs.pop(index)

    def skip(
        self,
        index: int,
        current: Callable[[], F],
        peek: Callable[[int], F | None],
        ahead: int,
        run: Run[F, O, S],
    ) -> None:
        """Item `index` without its output, which nobody reads (a warm-up item): its
        window runs as `output` runs it, since the state needs it, but the output is
        dropped, so every output read is the same as without skips."""
        self.__window(index, current, peek, ahead, run)
        del self.outputs[index]

    def __window(
        self,
        index: int,
        current: Callable[[], F],
        peek: Callable[[int], F | None],
        ahead: int,
        run: Run[F, O, S],
    ) -> None:
        if index in self.outputs:
            return
        # Until the window succeeds the planner holds nothing: after a failure the next
        # item starts cold.
        past, self.past, self.outputs = self.past, None, {}
        frames = [current()]
        ended = False
        for offset in range(1, ahead + 1):
            later = peek(offset)
            if later is None:
                ended = True
                break
            frames.append(later)
        keep = len(frames) if ended else max(1, len(frames) - self.look_ahead)
        carried = None
        if past is not None and past[0] == index - 1:
            carried = (past[1], past[2])
        outputs, state = run(frames, keep, carried, index)
        if len(outputs) != keep:
            raise ValueError(
                f"a window keeping {keep} frames gave {len(outputs)} outputs"
            )
        self.past = (index + keep - 1, state, frames[keep - 1])
        self.outputs = {index + k: output for k, output in enumerate(outputs)}


@dataclass(frozen=True)
class Fit:
    """How the temporal path fits a frame into the device's memory, from the node's
    Tile Size: `tiler` gives the first tile size and splits it (the whole frame when the
    tile covers it), and `shorten` (Auto) first shortens the window, to LOOK_AHEAD + 1
    frames read ahead and then to none, before any tile, since a shorter window costs
    only speed and tiles cost seams at their borders. `lighter` are the executor's
    lighter memory modes, (a name, a function that makes the executor run in it), each
    lighter than the one before; at each tile size they come first, with the full
    window, since they keep the frame and the window and cost only time. Past-only frames
    (no frames ahead) come before tiles: measured closer to the full window than tiles.
    `out_of_memory(error)` says whether `error` is one that the next step may avoid, and
    `release()` frees what the failed attempt left cached, once nothing of it is
    referenced. `tiles` False keeps the whole frame at every step (an executor without
    `with_tile`); tiles stop at `min_tile`. When the last step still runs out of memory,
    `exhausted`, if any, is the error raised instead."""

    tiler: Tiler
    shorten: bool
    out_of_memory: Callable[[Exception], bool]
    release: Callable[[], None]
    lighter: Sequence[tuple[str, Callable[[Any], Any]]] = ()
    tiles: bool = True
    exhausted: str = ""
    min_tile: int = 128

    def steps(
        self, width: int, height: int, ahead: int
    ) -> list[tuple[int, tuple[int, int] | None, int]]:
        """(frames read ahead, tile or None for the whole frame, memory mode: 0 the
        executor's own, k its k-th lighter one), in the order tried."""
        modes = [(ahead, m) for m in range(len(self.lighter) + 1)]
        if self.shorten:
            modes += [
                (a, len(self.lighter))
                for a in sorted({LOOK_AHEAD + 1, 0}, reverse=True)
                if a < ahead
            ]
        tile = tuple(self.tiler.starting_tile_size(width, height, 3))
        tiles: list[tuple[int, int] | None] = []
        while True:
            whole = tile[0] >= width and tile[1] >= height
            tiles.append(None if whole else (tile[0], tile[1]))
            if isinstance(self.tiler, NoTiling) or not self.tiles:
                break
            smaller = tuple(self.tiler.split((tile[0], tile[1])))
            if smaller == tile or min(smaller) < self.min_tile:
                break
            tile = smaller
        return [(a, t, m) for t in tiles for a, m in modes]


class _TemporalRun:
    """Per loop run: the steps of the fit, the one in use (kept for the rest of the run
    once a step had to be left), and the planner of the executor it uses."""

    def __init__(self, shape: tuple[int, ...], key: Hashable, steps: list) -> None:
        self.shape = shape
        self.key = key
        self.steps = steps
        self.at = 0
        self.planner: WindowPlanner | None = None


def temporal_upscale(
    context: NodeContext,
    img: np.ndarray,
    executor: Executor[F, O, S],
    report: TileReport,
    padding: PaddingType = PaddingType.NONE,
    fit: Fit | None = None,
) -> np.ndarray:
    """`img` upscaled by a temporal model, with the frames around it inside a loop.

    The model sees RGB: a grayscale image runs as RGB and comes back as grayscale, and an
    image with alpha runs its RGB while the alpha is resized (Lanczos). `padding` pads
    every frame, peeked ones included, and the output is cropped by the scaled padding.
    Outside a loop the image is a window of one frame. Inside one, the planner in the
    loop's state slot runs the windows; it restarts cold when the frame shape or the
    executor's key (a tile size included) changes. An item whose output nobody reads still runs
    its window (the state needs it) and returns `img` unchanged.

    With a `fit`, an attempt that runs out of memory moves to the fit's next step and
    tries the item again: a lighter memory mode or a shorter window carries the state on,
    a tile size starts cold. The step is logged once and kept for the rest of the loop
    run. An executor's `fitting_ahead` chooses the first step before any window runs (the
    first whose frames ahead fit), and caps the frames read ahead before each window."""
    view = context.iteration
    ahead = 0 if view is None else view.lookahead(IMAGE_INPUT)
    height, width = img.shape[:2]
    if padding != PaddingType.NONE:
        height, width = height + 2 * PAD_SIZE, width + 2 * PAD_SIZE
    run = None if view is None else view.state
    if (
        not isinstance(run, _TemporalRun)
        or run.shape != img.shape
        or run.key != executor.key
    ):
        steps: list[tuple[int, tuple[int, int] | None, int]] = (
            [(ahead, None, 0)]
            if fit is None
            else [
                step
                for step in fit.steps(width, height, ahead)
                if step[1] is None or hasattr(executor, "with_tile")
            ]
        )
        run = _TemporalRun(img.shape, executor.key, steps)
        if fit is not None:
            run.at = _first_fitting(executor, fit, steps, height, width)
            if run.at:
                _log_step(
                    "the temporal model's window would not fit;", fit, steps[run.at]
                )
        if view is not None:
            view.state = run
    while True:
        step_ahead, tile, mode = run.steps[run.at]
        base = executor if fit is None else _step_executor(executor, fit, tile, mode)
        fitting = getattr(base, "fitting_ahead", None)
        if fitting is not None:
            step_ahead = min(step_ahead, max(0, fitting((height, width), step_ahead)))
        if run.planner is None or run.planner.key != base.key:
            run.planner = WindowPlanner(img.shape, base.key, LOOK_AHEAD)
        planner = run.planner
        kept = planner.past, dict(planner.outputs)
        try:
            out = _item(context, img, base, planner, step_ahead, padding)
        except Exception as error:
            if fit is None or not fit.out_of_memory(error):
                raise
            if run.at + 1 == len(run.steps):
                if fit.exhausted:
                    raise RuntimeError(fit.exhausted) from error
                raise
        else:
            if tile is None:
                report.record(*img.shape[:2], whole=True)
            else:
                report.record(min(tile[1], height), min(tile[0], width), whole=False)
            return out
        # Out of the handler, the error and its traceback, which held the failed
        # attempt's tensors, are gone: what they left cached can be released.
        fit.release()
        planner.past, planner.outputs = kept
        run.at += 1
        _log_step("the temporal model ran out of memory;", fit, run.steps[run.at])


def _first_fitting(
    executor: Executor[F, O, S],
    fit: Fit,
    steps: list[tuple[int, tuple[int, int] | None, int]],
    height: int,
    width: int,
) -> int:
    """The first step whose frames ahead fit by the executor's `fitting_ahead`, before
    any window runs; 0 when the executor cannot tell, and the last step when none fits."""
    for at, (ahead, tile, mode) in enumerate(steps):
        fitting = getattr(
            _step_executor(executor, fit, tile, mode), "fitting_ahead", None
        )
        if fitting is None:
            return 0
        if fitting((height, width), ahead) >= ahead:
            return at
    return len(steps) - 1


def _step_executor(
    executor: Executor[F, O, S], fit: Fit, tile: tuple[int, int] | None, mode: int
) -> Executor[F, O, S]:
    """`executor` in a step's memory mode and tiles."""
    base = executor if mode == 0 else fit.lighter[mode - 1][1](executor)
    tiling = getattr(base, "with_tile", None)
    return base if tile is None or tiling is None else tiling(tile)


def _log_step(
    why: str, fit: Fit, step: tuple[int, tuple[int, int] | None, int]
) -> None:
    ahead, tile, mode = step
    logger.info(
        "Upscale Image: %s it now reads %d frames ahead, %s, %s",
        why,
        ahead,
        "the whole frame" if tile is None else f"in tiles of {tile[0]}x{tile[1]} px",
        "in its own memory mode" if mode == 0 else fit.lighter[mode - 1][0],
    )


def _item(
    context: NodeContext,
    img: np.ndarray,
    executor: Executor[F, O, S],
    planner: WindowPlanner[F, O, S],
    ahead: int,
    padding: PaddingType,
) -> np.ndarray:
    """One attempt at `img`'s output with `executor`, reading `ahead` frames ahead."""

    def model_frame(frame: np.ndarray) -> F:
        rgb = as_target_channels(frame[:, :, :3] if frame.ndim == 3 else frame, 3)
        if padding != PaddingType.NONE:
            rgb = create_border(rgb, padding.to_border_type(), Padding.all(PAD_SIZE))
        return executor.convert(rgb)

    def image(output: O) -> np.ndarray:
        out = executor.to_image(output)
        h, _w, c = get_h_w_c(img)
        if padding != PaddingType.NONE:
            crop = PAD_SIZE * (out.shape[0] // (h + 2 * PAD_SIZE))
            out = out[crop:-crop, crop:-crop]
        if c == 1:
            return as_target_channels(out, 1, True)
        if c == 4:
            alpha = resize(
                img[:, :, 3], (out.shape[1], out.shape[0]), ResizeFilter.LANCZOS
            )
            return assemble_alpha(out, np.clip(alpha.reshape(out.shape[:2]), 0, 1))
        return out

    view = context.iteration
    if view is None:
        return image(executor.run([model_frame(img)], 1, None, 0)[0][0])

    def peek(offset: int) -> F | None:
        frame = view.peek(IMAGE_INPUT, offset)
        if isinstance(frame, np.ndarray) and frame.shape == img.shape:
            return model_frame(frame)
        return None

    if view.unread:
        # Nothing is saved or shown from this item: the input passes the output's checks.
        planner.skip(view.index, lambda: model_frame(img), peek, ahead, executor.run)
        return img
    return image(
        planner.output(view.index, lambda: model_frame(img), peek, ahead, executor.run)
    )
