"""Upscale Image's temporal path, free of any framework: which frames each window holds,
which outputs it keeps and which state it carries (the planner, with a stand-in executor),
and `temporal_upscale`'s rules (a stand-in executor over NumPy images)."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from api import NodeContext
from nodes.impl.resize import ResizeFilter, resize
from nodes.impl.upscale import temporal_window
from nodes.impl.upscale.auto_split_tiles import (
    ESTIMATE,
    MAX_TILE_SIZE,
    NO_TILING,
    TileSize,
    parse_tile_size_input,
)
from nodes.impl.upscale.basic_upscale import PAD_SIZE, PaddingType
from nodes.impl.upscale.temporal_window import (
    LOOK_AHEAD,
    Fit,
    WindowPlanner,
    temporal_upscale,
)
from nodes.impl.upscale.tile_report import TileReport
from nodes.impl.upscale.tiler import MaxTileSize

# A stand-in window: (its frames, how many it keeps, the window whose state it carried).
Window = tuple[list[int], int, int | None]


def stand_in_run(windows: list[Window], fail=()):
    """An executor over frames that stand for their own index: each output names the
    window that made it and its frame, and the state names its window."""

    def run(frames, keep, past, first):
        # The window's first frame is the loop index it gets.
        assert first == frames[0]
        carried = None
        if past is not None:
            state, frame = past
            # The frame that links the state is the one just before the window.
            assert frame == frames[0] - 1
            carried = state
        windows.append((list(frames), keep, carried))
        if len(windows) - 1 in fail:
            raise RuntimeError("out of memory")
        return [(len(windows) - 1, f) for f in frames[:keep]], len(windows) - 1

    return run


def peeker(n, frames):
    def peek(k):
        return n + k if n + k < frames else None

    return peek


def run_planner(frames, ahead, look_ahead=3, missing=(), unread=()):
    """Every item read's output, as (the window that made it, frame), and the windows;
    `missing` items are not run, `unread` ones skipped."""
    planner = WindowPlanner[int, tuple[int, int], int]((1,), "stand-in", look_ahead)
    windows: list[Window] = []
    run = stand_in_run(windows)
    outputs = {}
    for n in range(frames):
        if n in missing:
            continue
        if n in unread:
            planner.skip(n, lambda n=n: n, peeker(n, frames), ahead, run)
        else:
            outputs[n] = planner.output(n, lambda n=n: n, peeker(n, frames), ahead, run)
        assert all(i > n or i in missing for i in planner.outputs)
    return outputs, windows


def test_a_window_keeps_the_frames_with_look_ahead_after_them_and_carries_its_state():
    outputs, windows = run_planner(12, ahead=5)
    assert windows == [
        ([0, 1, 2, 3, 4, 5], 3, None),
        ([3, 4, 5, 6, 7, 8], 3, 0),
        ([6, 7, 8, 9, 10, 11], 3, 1),
        # The video's end: every frame is kept.
        ([9, 10, 11], 3, 2),
    ]
    assert outputs == {n: (n // 3, n) for n in range(12)}


@pytest.mark.parametrize("frames", [1, 4, 7, 12, 30])
@pytest.mark.parametrize("ahead", [3, 5, 7])
@pytest.mark.parametrize("look_ahead", [0, 1, 3])
def test_every_frame_is_upscaled_once_with_its_look_ahead(frames, ahead, look_ahead):
    outputs, windows = run_planner(frames, ahead, look_ahead)
    kept = [f for window, keep, _ in windows for f in window[:keep]]
    assert kept == list(range(frames))
    for n in range(frames):
        made_by, frame = outputs[n]
        window = windows[made_by][0]
        assert frame == n
        assert len(window) - 1 - window.index(n) >= min(look_ahead, frames - 1 - n)
    # Every window after the first carries the state of the one before it.
    assert [carried for _, _, carried in windows] == [None, *range(len(windows) - 1)]
    assert all(len(window) <= ahead + 1 for window, _, _ in windows)
    assert len(windows) <= -(-frames // max(1, ahead - look_ahead + 1))


def test_without_frames_ahead_each_frame_has_its_past_only():
    outputs, windows = run_planner(6, ahead=0)
    assert outputs == {n: (n, n) for n in range(6)}
    assert windows == [([n], 1, n - 1 if n else None) for n in range(6)]


def test_fewer_frames_ahead_than_the_look_ahead_keep_one_frame_per_window():
    _, windows = run_planner(5, ahead=2)
    assert windows == [
        ([0, 1, 2], 1, None),
        ([1, 2, 3], 1, 0),
        ([2, 3, 4], 1, 1),
        ([3, 4], 2, 2),
    ]


def test_a_missing_item_starts_cold():
    _, windows = run_planner(10, ahead=0, missing={4})
    assert windows[4] == ([5], 1, None)
    assert windows[5] == ([6], 1, 4)
    # An item missing inside a window's kept frames does not break the state: the
    # window made it.
    outputs, windows = run_planner(12, ahead=5, missing={1, 4})
    assert windows[:2] == [([0, 1, 2, 3, 4, 5], 3, None), ([3, 4, 5, 6, 7, 8], 3, 0)]
    assert 5 in outputs
    # A missing item that would have started a window leaves the next one cold.
    _, windows = run_planner(12, ahead=5, missing={3})
    assert windows[1] == ([4, 5, 6, 7, 8, 9], 3, None)


@pytest.mark.parametrize("unread", [1, 3, 4, 7])
@pytest.mark.parametrize("ahead", [0, 5])
def test_skipped_items_still_run_their_windows_for_the_state(unread, ahead):
    full, full_windows = run_planner(12, ahead)
    outputs, windows = run_planner(12, ahead, unread=range(unread))
    assert windows == full_windows
    assert outputs == {n: full[n] for n in range(unread, 12)}


def test_an_item_asked_again_gets_a_cold_window_of_its_own():
    planner = WindowPlanner[int, tuple[int, int], int]((1,), "stand-in", 3)
    windows: list[Window] = []
    run = stand_in_run(windows)
    outputs = [planner.output(0, lambda: 0, lambda k: k, 4, run) for _ in range(2)]
    assert outputs == [(0, 0), (1, 0)]
    assert windows == [([0, 1, 2, 3, 4], 2, None)] * 2


def test_a_failed_window_leaves_the_next_item_a_cold_one():
    planner = WindowPlanner[int, tuple[int, int], int]((1,), "stand-in", 3)
    windows: list[Window] = []
    run = stand_in_run(windows, fail={1})
    assert planner.output(0, lambda: 0, peeker(0, 12), 4, run) == (0, 0)
    assert planner.output(1, lambda: 1, peeker(1, 12), 4, run) == (0, 1)
    with pytest.raises(RuntimeError):
        planner.output(2, lambda: 2, peeker(2, 12), 4, run)
    assert planner.output(3, lambda: 3, peeker(3, 12), 4, run) == (2, 3)
    assert windows == [
        ([0, 1, 2, 3, 4], 2, None),
        ([2, 3, 4, 5, 6], 2, 0),
        ([3, 4, 5, 6, 7], 2, None),
    ]


def test_a_window_must_give_one_output_per_kept_frame():
    planner = WindowPlanner[int, int, int]((1,), "stand-in", 3)
    with pytest.raises(ValueError, match="a window keeping 2 frames gave 1 outputs"):
        planner.output(0, lambda: 0, lambda k: k, 4, lambda f, k, p, i: (f[:1], 0))
    # Nothing is kept from it: the next item starts cold.
    assert planner.past is None and planner.outputs == {}


# temporal_upscale: a stand-in executor over NumPy frames, a stand-in iteration view.


class Doubling:
    """Each output is its frame at twice the size; it records each window as (first,
    frames, keep, carried) and checks the frames' shape."""

    def __init__(self, key="a", shape=None):
        self.key = key
        self.shape = shape
        self.windows: list[tuple[int, int, int, bool]] = []

    def convert(self, frame):
        assert frame.ndim == 3 and frame.shape[2] == 3
        if self.shape is not None:
            assert frame.shape == self.shape
        return frame

    def run(self, frames, keep, past, first):
        self.windows.append((first, len(frames), keep, past is not None))
        return [f.repeat(2, 0).repeat(2, 1) for f in frames[:keep]], len(self.windows)

    def to_image(self, output):
        return output


class View:
    """The loop as the node sees it: items in order, `ahead` items peeked."""

    def __init__(self, frames, ahead):
        self.frames, self.ahead = frames, ahead
        self.index, self.state, self.unread = 0, None, False

    def peek(self, _input, offset):
        i = self.index + offset
        if offset <= self.ahead and i < len(self.frames):
            return self.frames[i]
        return None

    def lookahead(self, _input):
        return self.ahead


def in_loop(view, executor, index, padding=PaddingType.NONE, unread=False):
    view.index, view.unread = index, unread
    context = cast(NodeContext, SimpleNamespace(iteration=view))
    return temporal_upscale(
        context, view.frames[index], executor, TileReport(), padding
    )


def images(count, shape=(6, 8, 3)):
    rng = np.random.default_rng(1)
    return [rng.random(shape, dtype=np.float32) for _ in range(count)]


def test_in_a_loop_the_windows_carry_the_state_and_name_their_first_frame():
    frames = images(8)
    view, executor = View(frames, 5), Doubling()
    for n in range(8):
        got = in_loop(view, executor, n)
        np.testing.assert_array_equal(got, frames[n].repeat(2, 0).repeat(2, 1))
    assert LOOK_AHEAD == 1
    assert executor.windows == [(0, 6, 5, False), (5, 3, 3, True)]


def test_another_executor_key_or_frame_shape_starts_cold():
    frames = images(9)
    view, executor = View(frames, 5), Doubling()
    for n in range(3):
        in_loop(view, executor, n)
    other = Doubling("b")
    in_loop(view, other, 3)
    assert other.windows == [(3, 6, 5, False)]
    view.frames = [*frames[:6], *images(3, (4, 8, 3))]
    in_loop(view, other, 6)
    assert other.windows[-1] == (6, 3, 3, False)


def test_an_unread_item_runs_its_window_and_passes_its_input():
    frames = images(6)
    view, executor = View(frames, 5), Doubling()
    assert in_loop(view, executor, 0, unread=True) is frames[0]
    assert executor.windows == [(0, 6, 5, False)]
    np.testing.assert_array_equal(
        in_loop(view, executor, 1), frames[1].repeat(2, 0).repeat(2, 1)
    )
    assert len(executor.windows) == 1


def test_grayscale_runs_as_color_and_comes_back_gray():
    gray = images(1, (6, 8))[0]
    context = cast(NodeContext, SimpleNamespace(iteration=None))
    got = temporal_upscale(context, gray, Doubling(), TileReport())
    assert got.shape == (12, 16)
    np.testing.assert_allclose(got, gray.repeat(2, 0).repeat(2, 1), atol=1e-6)


def test_alpha_is_resized_and_only_color_runs():
    rgba = images(1, (6, 8, 4))[0]
    context = cast(NodeContext, SimpleNamespace(iteration=None))
    got = temporal_upscale(context, rgba, Doubling(), TileReport())
    assert got.shape == (12, 16, 4)
    np.testing.assert_array_equal(
        got[:, :, :3], rgba[:, :, :3].repeat(2, 0).repeat(2, 1)
    )
    alpha = resize(rgba[:, :, 3], (16, 12), ResizeFilter.LANCZOS)
    np.testing.assert_array_equal(got[:, :, 3], np.clip(alpha[:, :, 0], 0, 1))


def test_padding_pads_every_frame_and_crops_the_scaled_padding():
    frames = images(6)
    padded = (6 + 2 * PAD_SIZE, 8 + 2 * PAD_SIZE, 3)
    view, executor = View(frames, 5), Doubling(shape=padded)
    for n in range(6):
        got = in_loop(view, executor, n, PaddingType.REFLECT_MIRROR)
        # Doubling commutes with the reflected border: the crop gives the frame doubled.
        np.testing.assert_array_equal(got, frames[n].repeat(2, 0).repeat(2, 1))
    assert executor.windows == [(0, 6, 5, False), (5, 1, 1, True)]


def test_outside_a_loop_the_image_is_a_window_of_one_and_the_report_says_whole():
    image = images(1)[0]
    context = cast(NodeContext, SimpleNamespace(iteration=None))
    executor, report = Doubling(), TileReport()
    temporal_upscale(context, image, executor, report)
    assert executor.windows == [(0, 1, 1, False)]
    assert report.text == "Whole image"


# Tiles and the fit to the device's memory.


class OutOfMemory(Exception):
    pass


def out_of_memory(error):
    return isinstance(error, OutOfMemory)


RELEASES: list[int] = []


def release():
    RELEASES.append(len(RELEASES))


class Refusing(Doubling):
    """Doubling that runs out of memory for a window of more than `frames` frames or a
    frame of more than `pixels` pixels; it records every attempt as (first, frames,
    frame shape, carried), refused ones included."""

    def __init__(self, frames=99, pixels=10**9, key="a", tile=None):
        super().__init__(key)
        self.frames, self.pixels, self.tile = frames, pixels, tile
        self.attempts: list[tuple[int, int, tuple[int, int], bool]] = []

    def with_tile(self, tile):
        """Runs in tiles of `tile` (width, height): the frame shape it records and
        judges is a tile's; its key is its own."""
        tiled = Refusing(self.frames, self.pixels, f"tiles {self.key} {tile}", tile)
        tiled.attempts = self.attempts
        return tiled

    def run(self, frames, keep, past, first):
        shape = frames[0].shape[:2]
        if self.tile is not None:
            shape = (min(self.tile[1], shape[0]), min(self.tile[0], shape[1]))
        self.attempts.append((first, len(frames), shape, past is not None))
        if len(frames) > self.frames or shape[0] * shape[1] > self.pixels:
            raise OutOfMemory("CUDA out of memory")
        return super().run(frames, keep, past, first)


def fit_of(tile_size, shorten, lighter=()):
    return Fit(
        parse_tile_size_input(tile_size, MaxTileSize),
        shorten,
        out_of_memory,
        release,
        lighter,
        min_tile=8,
    )


def half():
    """The tile the fit's tiler splits a 64x48 frame into first."""
    tile = fit_of(ESTIMATE, True).steps(64, 48, 7)[3][1]
    assert tile is not None and tile[0] * tile[1] < 64 * 48
    return tile


def test_auto_shortens_the_window_before_it_tiles():
    fit = fit_of(ESTIMATE, True)
    steps = fit.steps(64, 48, 7)
    assert steps[:3] == [(7, None, 0), (LOOK_AHEAD + 1, None, 0), (0, None, 0)]
    assert steps[3:6] == [(7, half(), 0), (LOOK_AHEAD + 1, half(), 0), (0, half(), 0)]
    # A size or Maximum keeps the window and halves the tiles; No Tiling has one step.
    sized = fit_of(TileSize(32), False).steps(64, 48, 7)
    assert sized[0][1] is not None and max(sized[0][1]) <= 32
    assert all(ahead == 7 and tile is not None for ahead, tile, _ in sized)
    assert fit_of(MAX_TILE_SIZE, False).steps(64, 48, 7)[:2] == [
        (7, None, 0),
        (7, half(), 0),
    ]
    assert fit_of(NO_TILING, False).steps(64, 48, 7) == [(7, None, 0)]


def test_lighter_memory_modes_come_first_with_the_full_window():
    modes = [("one", lambda e: e), ("two", lambda e: e)]
    steps = fit_of(ESTIMATE, True, modes).steps(64, 48, 7)
    assert steps[:5] == [
        (7, None, 0),
        (7, None, 1),
        (7, None, 2),
        (LOOK_AHEAD + 1, None, 2),
        (0, None, 2),
    ]
    assert steps[5:8] == [(7, half(), 0), (7, half(), 1), (7, half(), 2)]
    # No Tiling keeps the whole frame but still takes the lighter modes.
    assert fit_of(NO_TILING, False, modes).steps(64, 48, 7) == [
        (7, None, 0),
        (7, None, 1),
        (7, None, 2),
    ]


class Lighter(Refusing):
    """Refusing that runs in a memory mode, as `with_mode` makes it; it refuses every
    attempt below mode `needs`, and records the mode of each attempt."""

    def __init__(self, needs, mode=0, modes=None):
        super().__init__()
        self.needs, self.mode = needs, mode
        self.modes = [] if modes is None else modes

    def with_mode(self, mode):
        lighter = Lighter(self.needs, mode, self.modes)
        lighter.attempts = self.attempts
        return lighter

    def run(self, frames, keep, past, first):
        self.modes.append((self.mode, len(frames), past is not None))
        if self.mode < self.needs:
            raise OutOfMemory("CUDA out of memory")
        return super().run(frames, keep, past, first)


def test_a_lighter_memory_mode_carries_the_state_on():
    frames = images(12, (6, 8, 3))
    view = View(frames, 7)
    executor = Lighter(needs=0)
    modes = [("one", lambda e: e.with_mode(1)), ("two", lambda e: e.with_mode(2))]
    fit = fit_of(ESTIMATE, True, modes)
    for n in range(7):
        in_loop_fit(view, executor, n, fit)
    executor.needs = 2
    in_loop_fit(view, executor, 7, fit)
    # Modes 0 and 1 refused; mode 2 ran the full window, carrying the state.
    assert executor.modes[-3:] == [(0, 5, True), (1, 5, True), (2, 5, True)]


class Knowing(Doubling):
    """Doubling that says how many frames ahead fit: `fits` for frames of `shape`."""

    def __init__(self, fits):
        super().__init__()
        self.fits = fits
        self.asked: list[tuple[tuple[int, int], int]] = []

    def fitting_ahead(self, shape, ahead):
        self.asked.append((shape, ahead))
        return min(ahead, self.fits)


def test_an_executor_that_knows_its_memory_caps_the_frames_ahead_first():
    frames = images(10, (6, 8, 3))
    view, executor = View(frames, 7), Knowing(fits=2)
    for n in range(3):
        in_loop_fit(view, executor, n, fit_of(ESTIMATE, True))
    # Windows of 3 frames (2 ahead), never a refused attempt: item 0 keeps 0..1, item 2
    # starts the next window.
    assert executor.windows == [(0, 3, 2, False), (2, 3, 2, True)]
    assert executor.asked[0] == ((6, 8), 7)


def test_out_of_memory_moves_down_the_steps_and_the_choice_is_kept(monkeypatch):
    logged = []
    RELEASES.clear()
    monkeypatch.setattr(temporal_window.logger, "info", lambda *a: logged.append(a))
    frames = images(10, (48, 64, 3))
    view = View(frames, 7)
    # Refuses windows of 3 frames or more, and every whole frame.
    executor = Refusing(frames=2, pixels=48 * 64 - 1)
    fit = fit_of(ESTIMATE, True)
    in_loop_fit(view, executor, 0, fit)
    tried = [(n, shape) for _, n, shape, _ in executor.attempts]
    tile = (min(half()[1], 48), min(half()[0], 64))
    # Whole frame with 8 frames, then with LOOK_AHEAD + 2, then 1; then tiles of half
    # the frame with 8 frames (refused), then with LOOK_AHEAD + 2 (refused), then 1.
    assert LOOK_AHEAD + 2 > 2
    assert tried[:5] == [
        (8, (48, 64)),
        (LOOK_AHEAD + 2, (48, 64)),
        (1, (48, 64)),
        (8, tile),
        (LOOK_AHEAD + 2, tile),
    ]
    assert all(n == 1 and shape == tile for n, shape in tried[5:])
    assert len(logged) == 5
    # Each step change released what the failed attempt left, after its handler.
    assert len(RELEASES) == 5
    executor.attempts.clear()
    # The next item that needs a window starts at the step that ran: no new attempts.
    in_loop_fit(view, executor, 1, fit)
    assert {(n, shape) for _, n, shape, _ in executor.attempts} == {(1, tile)}
    assert len(logged) == 5


def test_a_shorter_window_carries_the_state_on():
    frames = images(10, (6, 8, 3))
    view = View(frames, 7)
    executor = Refusing(frames=99)
    fit = fit_of(ESTIMATE, True)
    for n in range(3):
        in_loop_fit(view, executor, n, fit)
    # From now on windows of more than 2 frames run out of memory.
    executor.frames = 2
    in_loop_fit(view, executor, 7, fit)
    assert executor.attempts[-1][1] == 1
    # The window at item 7 follows the first window (items 0..6 kept): it is carried.
    assert executor.attempts[-1][3]


def test_without_a_fit_or_past_the_last_step_the_error_is_raised():
    view = View(images(4, (6, 8, 3)), 3)
    with pytest.raises(OutOfMemory):
        in_loop(view, Refusing(frames=2), 0)
    with pytest.raises(OutOfMemory):
        in_loop_fit(
            View(images(4, (6, 8, 3)), 3),
            Refusing(pixels=1),
            0,
            fit_of(NO_TILING, False),
        )


def in_loop_fit(view, executor, index, fit):
    view.index, view.unread = index, False
    context = cast(NodeContext, SimpleNamespace(iteration=view))
    return temporal_upscale(
        context, view.frames[index], executor, TileReport(), PaddingType.NONE, fit
    )


class Estimating(Lighter):
    """Lighter that estimates its windows: in memory mode m, `fit[m]` frames ahead fit
    (-1: none); it refuses nothing it says fits."""

    def __init__(self, fits, mode=0, modes=None):
        super().__init__(needs=0, mode=mode, modes=modes)
        self.fits = fits

    def with_mode(self, mode):
        estimating = Estimating(self.fits, mode, self.modes)
        estimating.attempts = self.attempts
        return estimating

    def fitting_ahead(self, shape, ahead):
        return min(ahead, self.fits[self.mode])


def test_the_first_step_is_chosen_from_the_estimate_before_any_window(monkeypatch):
    logged = []
    monkeypatch.setattr(temporal_window.logger, "info", lambda *a: logged.append(a))
    modes = [("one", lambda e: e.with_mode(1)), ("two", lambda e: e.with_mode(2))]
    fit = fit_of(ESTIMATE, True, modes)
    # Mode 0 fits 3 frames ahead of 7, mode 1 all 7: mode 1 runs from the first window.
    executor = Estimating(fits=[3, 7, 7])
    in_loop_fit(View(images(10, (6, 8, 3)), 7), executor, 0, fit)
    assert executor.modes == [(1, 8, False)]
    assert len(logged) == 1
    # Nothing fits whole: the last step (the lightest mode, no frames ahead).
    executor = Estimating(fits=[-1, -1, -1])
    in_loop_fit(View(images(10, (6, 8, 3)), 7), executor, 0, fit)
    assert executor.modes == [(2, 1, False)]


def test_auto_without_tiles_stops_at_the_whole_frame_with_its_message():
    fit = Fit(
        parse_tile_size_input(ESTIMATE, MaxTileSize),
        True,
        out_of_memory,
        release,
        tiles=False,
        exhausted="does not fit: try this",
    )
    assert all(tile is None for _, tile, _ in fit.steps(64, 48, 7))
    view = View(images(4, (48, 64, 3)), 3)
    with pytest.raises(RuntimeError, match="does not fit: try this") as raised:
        in_loop_fit(view, Refusing(pixels=1), 0, fit)
    assert isinstance(raised.value.__cause__, OutOfMemory)


def test_past_only_comes_before_tiles_and_tiles_start_cold():
    frames = images(12, (48, 64, 3))
    view = View(frames, 7)
    executor = Refusing()
    fit = fit_of(ESTIMATE, True)
    for n in range(7):
        in_loop_fit(view, executor, n, fit)
    # From now on every whole frame runs out of memory.
    executor.pixels = 48 * 64 - 1
    executor.attempts.clear()
    in_loop_fit(view, executor, 7, fit)
    tile = (min(half()[1], 48), min(half()[0], 64))
    assert [(n, shape, carried) for _, n, shape, carried in executor.attempts] == [
        (5, (48, 64), True),
        (LOOK_AHEAD + 2, (48, 64), True),
        (1, (48, 64), True),
        # Tiles hold one state per tile: the first tiled window starts cold.
        (5, tile, False),
    ]


def test_an_executor_without_tiles_gets_no_tile_steps():
    # Doubling has no with_tile: the whole frame's steps only, then the error.
    executor = WholeOnly()
    with pytest.raises(OutOfMemory):
        in_loop_fit(
            View(images(6, (48, 64, 3)), 3), executor, 0, fit_of(ESTIMATE, True)
        )
    assert [n for _, n, _, _ in executor.windows] == [4, LOOK_AHEAD + 2, 1]


class WholeOnly(Doubling):
    """Doubling, without tiles, that runs out of memory for every window."""

    def run(self, frames, keep, past, first):
        self.windows.append((first, len(frames), keep, past is not None))
        raise OutOfMemory("CUDA out of memory")
