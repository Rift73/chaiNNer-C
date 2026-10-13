"""Temporal VOSR as a per-item step in a Load Video loop (vosr2_temporal_step.py and
Upscale Image (VOSR)): each output equals the removed Temporal Sequence generator's frame,
each frame is restored once, and short look-aheads fall back to past frames only."""

import gc
import logging
import threading
import time
import weakref
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from test_generator_lifecycle import make_node
from test_item_window import Pipeline, set_schema
from test_iteration_view import Reader

from api import Aborted, BaseInput, InputId, NodeContext, NodeId
from api.settings import SettingsParser
from chain.chain import Edge, EdgeTarget
from nodes.impl.pytorch import vosr2_temporal_step
from nodes.impl.pytorch.vosr2 import (
    BundlePaths,
    VosrBackend,
    VosrColorCorrection,
    VosrMemoryMode,
)
from nodes.impl.pytorch.vosr2_temporal import (
    RestoredFrame,
    TemporalMode,
    TemporalObservation,
    motion_pair,
    stabilize,
    validate_image,
)
from nodes.impl.pytorch.vosr2_temporal_state import carry_detail, initialize_detail
from nodes.impl.pytorch.vosr2_temporal_step import (
    SCENE_START,
    TemporalStep,
    frames_ahead,
)
from nodes.impl.pytorch.vosr2_texture import Grain
from nodes.properties.inputs import DirectoryInput, ImageInput
from nodes.utils.seed import Seed
from packages.chaiNNer_pytorch.pytorch.processing import upscale_image_vosr

INPUTS = ("model", 2, 0)
STRENGTH = 0.65
METHODS = [TemporalMode.THREE_FRAMES, TemporalMode.PERSISTENT_DETAIL]


def noop() -> None:
    pass


def video(length: int, cut: int | None = None, motion: int = 1) -> list[np.ndarray]:
    """Frames of a texture moving `motion` pixels per frame (0: still, each frame a
    little brighter, so that frames differ); from `cut` on, another scene."""
    rng = np.random.default_rng(31)
    texture = cv2.GaussianBlur(rng.random((48, 64, 3), dtype=np.float32), (0, 0), 0.8)
    return [
        np.ascontiguousarray(
            np.roll(
                1 - texture if cut is not None and i >= cut else texture,
                i * motion,
                axis=1,
            )
            + np.float32(0 if motion else i * 1e-4)
        )
        for i in range(length)
    ]


class Restorer:
    """VOSR's stand-in: Lanczos 2x plus the frame's own noise. Records the frame of each
    call, its thread, and a reference to each restoration."""

    def __init__(self, frames: list[np.ndarray]):
        self.frames = frames
        rng = np.random.default_rng(5)
        self.noise = [
            rng.normal(0, 0.01, (96, 128, 3)).astype(np.float32) for _ in frames
        ]
        self.calls: list[int] = []
        self.threads: list[str] = []
        self.restored: list[weakref.ref[np.ndarray]] = []

    def index(self, image: np.ndarray) -> int:
        return next(i for i, f in enumerate(self.frames) if np.array_equal(f, image))

    def __call__(
        self, image: np.ndarray, _context: NodeContext | None = None
    ) -> np.ndarray:
        index = self.index(image)
        self.calls.append(index)
        self.threads.append(threading.current_thread().name)
        result = (
            cv2.resize(image, (128, 96), interpolation=cv2.INTER_LANCZOS4)
            + self.noise[index]
        )
        self.restored.append(weakref.ref(result))
        return result

    def alive(self) -> int:
        return sum(reference() is not None for reference in self.restored)


def old_sequence(
    images: list[np.ndarray],
    restore: Callable[[np.ndarray], np.ndarray],
    strength: float,
    method: TemporalMode,
) -> list[np.ndarray]:
    """The removed generators, vosr2_temporal.observations and
    vosr2_temporal_state._persistent_observations, over images instead of paths."""

    def observations():
        previous = None
        current = None
        previous_match = None

        def read(index: int) -> RestoredFrame:
            source = images[index]
            validate_image(source)
            return RestoredFrame.create(source, restore(source))

        if not images:
            return
        current = read(0)
        for index in range(len(images)):
            assert current is not None
            following = read(index + 1) if index + 1 < len(images) else None
            next_match = None
            reverse_match = None
            if following is not None and strength > 0:
                next_match, reverse_match = motion_pair(
                    current.source, following.source, noop
                )
            neighbors = []
            if previous is not None and previous_match is not None:
                neighbors.append((previous, previous_match))
            if following is not None and next_match is not None:
                neighbors.append((following, next_match))
            result = stabilize(current, neighbors, strength, noop)
            yield TemporalObservation(result, current.source, previous_match)
            previous, current = current, following
            previous_match = reverse_match

    frames = observations()
    if method == TemporalMode.THREE_FRAMES:
        return [item.image for item in frames]
    outputs = []
    pending = None
    state = None
    while True:
        first = pending if pending is not None else next(frames, None)
        pending = None
        if first is None:
            return outputs
        bootstrap = [first]
        for _ in range(4):
            following = next(frames, None)
            if following is None:
                break
            if following.previous_match is not None and not np.any(
                following.previous_match.confidence
            ):
                pending = following
                break
            bootstrap.append(following)
        result, state = initialize_detail(bootstrap, strength, noop)
        outputs.append(result)
        for item in bootstrap[1:]:
            result, state = carry_detail(item, state, strength, noop)
            outputs.append(result)
        if pending is not None:
            state = None
            continue
        for item in frames:
            if item.previous_match is not None and not np.any(
                item.previous_match.confidence
            ):
                pending = item
                state = None
                break
            result, state = carry_detail(item, state, strength, noop)
            outputs.append(result)
        else:
            return outputs


def past_only(
    images: list[np.ndarray],
    restore: Callable[[np.ndarray], np.ndarray],
    strength: float,
    method: TemporalMode,
) -> list[np.ndarray]:
    """The fallback: Three Frames with the previous frame alone; Persistent Detail
    starting each scene (the first frame, a cut) from its first frame alone."""
    outputs = []
    previous = None
    state = None
    for image in images:
        current = RestoredFrame.create(image, restore(image))
        match = None
        neighbors = []
        if previous is not None:
            match = motion_pair(previous.source, current.source, noop)[1]
            neighbors.append((previous, match))
        observation = TemporalObservation(
            stabilize(current, neighbors, strength, noop), current.source, match
        )
        if method == TemporalMode.THREE_FRAMES:
            outputs.append(observation.image)
        elif state is None or (match is not None and not np.any(match.confidence)):
            result, state = initialize_detail([observation], strength, noop)
            outputs.append(result)
        else:
            result, state = carry_detail(observation, state, strength, noop)
            outputs.append(result)
        previous = current
    return outputs


class Context(NodeContext):
    """The node's context. Stopped (aborted) once `stop` is set; its own checks raise
    InterruptedError once `fail` is set."""

    def __init__(self):
        self.stop = threading.Event()
        self.fail = threading.Event()

    @property
    def aborted(self) -> bool:
        return self.stop.is_set()

    @property
    def paused(self) -> bool:
        return False

    def suspend(self) -> None:
        if self.fail.is_set():
            raise InterruptedError("failed")
        super().suspend()

    @property
    def settings(self) -> SettingsParser:
        return SettingsParser({})

    @property
    def node_id(self) -> NodeId:
        return NodeId("vosr")

    @property
    def storage_dir(self) -> Path:
        return Path("unused")

    def set_progress(self, progress: float) -> None:
        pass

    def add_cleanup(self, fn, after="chain") -> None:
        pass


def drive(
    frames: list[np.ndarray],
    method: TemporalMode,
    restore: Callable[[np.ndarray, NodeContext], np.ndarray],
    *,
    lookahead: int = SCENE_START,
    overlap: bool = False,
    missing: frozenset[int] = frozenset(),
    after: Callable[[], object] = noop,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """A loop over frames: item i's call with its image and a peek of later items; an
    item in `missing` fails before the node (peek gives None for it). `after` runs after
    each call. Returns the outputs and every (index, offset) peeked."""
    context = Context()
    step = TemporalStep(method, STRENGTH, lookahead, overlap, INPUTS)
    outputs, peeks = [], []
    for index, image in enumerate(frames):
        if index in missing:
            continue

        def peek(offset: int, index: int = index) -> np.ndarray | None:
            peeks.append((index, offset))
            later = index + offset
            return None if later >= len(frames) or later in missing else frames[later]

        outputs.append(step.frame(context, index, image, peek, restore, INPUTS))
        after()
    return outputs, peeks


def assert_equal(actual: list[np.ndarray], expected: list[np.ndarray]) -> None:
    assert len(actual) == len(expected)
    for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
        assert a.dtype == b.dtype == np.float32, index
        np.testing.assert_array_equal(a, b, err_msg=f"frame {index}")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("motion", [0, 1])
@pytest.mark.parametrize(
    ("length", "cut"),
    [(1, None), (2, None), (3, None), (7, None), (12, 6), (9, 2), (8, 7)],
)
def test_each_output_equals_the_old_generator(method, motion, length, cut):
    # Cuts at 6 (after a full scene start), 2 (inside one) and 7 (the last frame). Still
    # frames carry detail; moving ones (one pixel per frame) mostly keep fresh detail.
    frames = video(length, cut, motion)
    expected = old_sequence(frames, Restorer(frames), STRENGTH, method)
    restorer = Restorer(frames)
    actual, peeks = drive(frames, method, restorer)
    assert_equal(actual, expected)
    assert restorer.calls == list(range(length))  # each frame once, in order
    assert max((offset for _, offset in peeks), default=1) <= frames_ahead(method)


def test_the_cut_frames_are_matched_as_cuts():
    # The cases above exercise scene starts only if the fixture's cut is one.
    frames = video(2, 1)
    _, reverse = motion_pair(frames[0], frames[1], noop)
    assert not reverse.confidence.any()
    first, second = video(2)
    _, reverse = motion_pair(first, second, noop)
    assert reverse.confidence.mean() > 0.5


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("short", [0, 1])
def test_a_shorter_look_ahead_uses_past_frames_only(method, short):
    lookahead = 0 if short == 0 else frames_ahead(method) - 1
    frames = video(10, 6, motion=0)
    expected = past_only(frames, Restorer(frames), STRENGTH, method)
    restorer = Restorer(frames)
    actual, peeks = drive(frames, method, restorer, lookahead=lookahead)
    assert_equal(actual, expected)
    assert restorer.calls == list(range(10)) and peeks == []
    # The fallback differs from the look-ahead output wherever motion matches.
    assert not np.array_equal(actual[3], drive(frames, method, Restorer(frames))[0][3])


def test_persistent_detail_restarts_at_a_cut():
    # After a cut, outputs equal a video that starts at the cut; earlier frames, their
    # detail and their history no longer contribute.
    frames = video(14, 5)
    method = TemporalMode.PERSISTENT_DETAIL
    whole, _ = drive(frames, method, Restorer(frames))
    tail = frames[5:]
    alone, _ = drive(tail, method, Restorer(frames))
    assert_equal(whole[5:], alone)
    # Without the cut, history crosses frame 5.
    smooth = video(14, motion=0)
    first, _ = drive(smooth, method, Restorer(smooth))
    rest = smooth[5:]
    restarted, _ = drive(rest, method, Restorer(smooth))
    assert not np.array_equal(first[7], restarted[2])


@pytest.mark.parametrize("method", METHODS)
def test_an_item_that_never_reaches_the_node_ends_the_sequence_there(method):
    # Item 4 fails before the node (peek gives None) and the loop goes on: frame 3 is
    # the end of one sequence and frame 5 the start of the next.
    frames = video(10)
    restorer = Restorer(frames)
    actual, _ = drive(frames, method, restorer, missing=frozenset({4}))
    head, tail = frames[:4], frames[5:]
    expected = old_sequence(head, Restorer(frames), STRENGTH, method) + old_sequence(
        tail, Restorer(frames), STRENGTH, method
    )
    assert_equal(actual, expected)
    assert restorer.calls == [0, 1, 2, 3, 5, 6, 7, 8, 9]


@pytest.mark.parametrize("method", METHODS)
def test_restorations_are_released_once_no_output_needs_them(method):
    frames = video(40, 20)
    restorer = Restorer(frames)
    held = []
    drive(frames, method, restorer, after=lambda: held.append(restorer.alive()))
    # Between items only the current frame and the next one stay restored.
    assert max(held) <= 2
    gc.collect()
    assert restorer.alive() == 0


def test_only_the_previous_current_and_next_frames_affect_a_three_frames_output():
    frames = video(6)
    altered = [frame.copy() for frame in frames]
    altered[0] = np.clip(altered[0] + 0.01, 0, 1)
    method = TemporalMode.THREE_FRAMES
    a, _ = drive(frames, method, Restorer(frames))
    b, _ = drive(altered, method, Restorer(altered))
    assert not np.array_equal(a[1], b[1])
    for i in (2, 3, 4, 5):
        np.testing.assert_array_equal(a[i], b[i])


def test_frames_of_another_size_fail_explicitly():
    frames = video(3)
    frames[2] = np.ascontiguousarray(frames[2][:, :-1])
    with pytest.raises(ValueError, match="identical dimensions"):
        drive(frames, TemporalMode.THREE_FRAMES, Restorer(frames))


def test_the_settings_must_not_change_within_a_video():
    frames = video(3)
    step = TemporalStep(TemporalMode.THREE_FRAMES, STRENGTH, 1, False, INPUTS)
    context = Context()
    step.frame(context, 0, frames[0], lambda _: frames[1], Restorer(frames), INPUTS)
    with pytest.raises(ValueError, match="same model and settings"):
        step.frame(context, 1, frames[1], lambda _: frames[2], Restorer(frames), ())


@pytest.mark.parametrize("strength", [0, -0.1, 1.1, float("nan")])
def test_the_step_rejects_a_strength_outside_its_range(strength):
    with pytest.raises(ValueError, match="strength"):
        TemporalStep(TemporalMode.THREE_FRAMES, strength, 1, False, INPUTS)


def restoring_threads():
    return [t for t in threading.enumerate() if t.name.startswith("vosr-restore")]


@pytest.mark.parametrize("method", METHODS)
def test_overlap_restores_the_next_frame_in_the_background_and_settles(method):
    # With overlap (the TensorRT backends) the outputs are the same; later frames are
    # restored on a worker while the current output is computed, and no restoration
    # runs, and no worker lives, when an item's output is returned.
    frames = video(9, 6)
    expected, _ = drive(frames, method, Restorer(frames), lookahead=7)
    restorer = Restorer(frames)
    active = threading.Event()

    def restore(image, context):
        active.set()
        try:
            return restorer(image, context)
        finally:
            active.clear()

    def after():
        assert not active.is_set() and restoring_threads() == []

    actual, peeks = drive(
        frames, method, restore, lookahead=7, overlap=True, after=after
    )
    assert_equal(actual, expected)
    assert restorer.calls == list(range(9))
    assert restorer.threads[0] == threading.current_thread().name
    assert all(name.startswith("vosr-restore") for name in restorer.threads[1:])
    assert max(offset for _, offset in peeks) <= frames_ahead(method) + 1


@pytest.mark.parametrize("overlap", [False, True])
def test_motion_pairs_run_on_the_cpu_threads_and_settle_with_each_item(
    monkeypatch, overlap
):
    # Each pair starts once both frames' inputs are taken: during the next frame's
    # restoration (overlap or not), so all of them run on the item's CPU threads, which
    # are joined, like the restoring worker, before the item's output is returned.
    frames = video(8, 5)
    method = TemporalMode.THREE_FRAMES
    expected = old_sequence(frames, Restorer(frames), STRENGTH, method)
    threads = []
    pair = vosr2_temporal_step.motion_pair

    def recorded(*args):
        threads.append(threading.current_thread().name)
        return pair(*args)

    def after():
        assert not [
            t for t in threading.enumerate() if t.name.startswith("vosr-temporal")
        ]

    monkeypatch.setattr(vosr2_temporal_step, "motion_pair", recorded)
    actual, _ = drive(
        frames, method, Restorer(frames), lookahead=7, overlap=overlap, after=after
    )
    assert_equal(actual, expected)
    assert len(threads) == len(frames) - 1
    assert all(name.startswith("vosr-temporal") for name in threads)


def test_a_failing_item_stops_and_joins_its_background_restoration():
    # Frame 2 is restored in the background during item 0; the item fails meanwhile.
    # The worker sees the item's stop (not the run's), and is joined before the error
    # leaves the node.
    frames = video(5)
    restorer = Restorer(frames)
    context = Context()
    stopped = []

    def restore(image, worker_context):
        if restorer.index(image) == 2:
            context.fail.set()
            try:
                while True:
                    worker_context.suspend()
                    time.sleep(0.005)
            except Aborted:
                stopped.append(context.aborted)
                raise
        return restorer(image, worker_context)

    step = TemporalStep(TemporalMode.THREE_FRAMES, STRENGTH, 3, True, INPUTS)
    with pytest.raises(InterruptedError):
        step.frame(
            context, 0, frames[0], lambda offset: frames[offset], restore, INPUTS
        )
    assert stopped == [False] and restoring_threads() == []


def test_a_background_failure_is_raised_by_the_item_that_needs_that_frame():
    frames = video(5)
    restorer = Restorer(frames)
    context = Context()

    def restore(image, worker_context):
        if restorer.index(image) == 2:
            raise ValueError("frame 2 failed")
        return restorer(image, worker_context)

    step = TemporalStep(TemporalMode.THREE_FRAMES, STRENGTH, 3, True, INPUTS)
    output = step.frame(
        context, 0, frames[0], lambda offset: frames[offset], restore, INPUTS
    )
    assert output.shape == (96, 128, 3)
    with pytest.raises(ValueError, match="frame 2 failed"):
        step.frame(
            context, 1, frames[1], lambda offset: frames[1 + offset], restore, INPUTS
        )


# Upscale Image (VOSR) in a real executor loop, over the item window.


def vosr_loop(monkeypatch, tmp_path, frames, window, method, restore, *, source):
    """g0 (as `source`, reading no file per item, as Load Video) -> Upscale Image (VOSR)
    -> save, whose folder comes from a node that runs after VOSR. VOSR's restoration is
    `restore` on the CPU. Returns the pipeline and the list save fills with the frames."""
    node = upscale_image_vosr
    monkeypatch.setattr(
        node,
        "get_settings",
        lambda _: SimpleNamespace(
            device=torch.device("cpu"), budget_limit=0, force_cache_wipe=False
        ),
    )
    monkeypatch.setattr(
        node, "_upscale", lambda context, image, *_settings: restore(image, context)
    )
    monkeypatch.setattr(node, "ensure_model", lambda _: BundlePaths(tmp_path))
    pipeline = Pipeline(
        monkeypatch,
        tmp_path,
        len(frames),
        window=window,
        nodes=(),
        reader=Reader(iter(frames)),
    )
    set_schema(pipeline.chain.nodes[NodeId("g0")], source)
    edge = pipeline.chain.edge_to(NodeId("save"), InputId(0))
    assert edge is not None
    pipeline.chain.remove_edge(edge)

    def upscale(context, image):
        return node.upscale_image_vosr_node(
            context,
            image,
            2,
            Seed(0),
            VosrColorCorrection.WAVELET,
            VosrMemoryMode.DEFAULT,
            VosrBackend.PYTORCH,
            method,
            65,
        )

    vosr = make_node("vosr", "regularNode", upscale, 1, 1)
    object.__setattr__(vosr.data, "inputs", [ImageInput(channels=3).with_id(0)])
    object.__setattr__(vosr.data, "side_effects", False)
    object.__setattr__(vosr.data, "node_context", True)
    set_schema(vosr, "chainner:pytorch:upscale_image_vosr")
    pipeline.chain.add_node(vosr)
    pipeline.chain.add_edge(Edge(edge.source, EdgeTarget(NodeId("vosr"), InputId(0))))
    pipeline.connect("vosr", "save", 0)
    saved = []
    save = pipeline.chain.nodes[NodeId("save")]
    inputs = [BaseInput("any", "Input0").with_id(0), DirectoryInput(must_exist=False)]
    object.__setattr__(save.data, "inputs", [inputs[0], inputs[1].with_id(1)])
    object.__setattr__(save.data, "run", lambda value, _folder: saved.append(value))
    folder = make_node("folder", "regularNode", lambda: tmp_path / "out", 0, 1)
    object.__setattr__(folder.data, "side_effects", False)
    pipeline.chain.add_node(folder)
    pipeline.connect("folder", "save", 1)
    return pipeline, saved


def fallback_logs(caplog):
    return [r.getMessage() for r in caplog.records if "VOSR temporal" in r.getMessage()]


@pytest.mark.parametrize("method", METHODS)
def test_in_a_load_video_loop_vosr_reads_frames_ahead(
    monkeypatch, tmp_path, caplog, method
):
    # Load Video reads no file per item, so save's folder, computed later, does not stop
    # the item window: K = 6 serves five frames ahead, and every output equals the old
    # generator's. At K = 1 the node uses past frames only and says so once.
    frames = video(11, 7)
    runs = {}
    for window in ("6", "1"):
        restorer = Restorer(frames)
        pipeline, saved = vosr_loop(
            monkeypatch,
            tmp_path,
            frames,
            window,
            method,
            restorer,
            source="chainner:image:load_video",
        )
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="sanic.root"):
            assert pipeline.run(workers=4) is None
        assert restorer.calls == list(range(11))
        runs[window] = (pipeline.decisions, saved, fallback_logs(caplog))
    decisions, saved, logs = runs["6"]
    assert decisions == ["item window K=6 (override)"] and logs == []
    assert_equal(saved, old_sequence(frames, Restorer(frames), STRENGTH, method))
    decisions, saved, logs = runs["1"]
    assert decisions == ["item window K=1 (forced)"]
    label = upscale_image_vosr.TEMPORAL_METHODS[method]
    needs = f"{label} needs {frames_ahead(method)} frames ahead"
    assert logs == [
        f"VOSR temporal: {needs}, the item window serves 0; using past frames only"
    ]
    assert_equal(saved, past_only(frames, Restorer(frames), STRENGTH, method))


def test_outside_a_load_video_loop_each_image_is_upscaled_alone(monkeypatch, tmp_path):
    # A Load Images loop: no temporal step, no look-ahead; each output is the image's
    # own restoration.
    frames = video(5)
    restorer = Restorer(frames)
    pipeline, saved = vosr_loop(
        monkeypatch,
        tmp_path,
        frames,
        "3",
        TemporalMode.PERSISTENT_DETAIL,
        restorer,
        source="chainner:image:load_images",
    )
    grains = []

    def restore(context, image, *settings):
        grains.append(settings[-1])
        return restorer(image, context)

    monkeypatch.setattr(upscale_image_vosr, "_upscale", restore)
    assert pipeline.run(workers=4) is None
    assert restorer.calls == list(range(5))
    assert_equal(saved, [Restorer(frames)(frame) for frame in frames])
    # Each image gets the seed's grain alone.
    assert grains == [Grain(Seed(0).to_u32())] * 5


def test_each_video_frame_gets_its_grain_with_the_previous_input(monkeypatch, tmp_path):
    # The grain goes on the stabilized output, with the previous frame's input, so that
    # it changes only where the picture does.
    frames = video(5)
    calls = []

    def record(output, source, scale, grain, check, device):
        calls.append((source, grain))
        return output()

    monkeypatch.setattr(upscale_image_vosr, "restore_with_texture", record)
    pipeline, saved = vosr_loop(
        monkeypatch,
        tmp_path,
        frames,
        "6",
        TemporalMode.THREE_FRAMES,
        Restorer(frames),
        source="chainner:image:load_video",
    )
    assert pipeline.run(workers=4) is None
    assert len(saved) == len(calls) == 5
    for index, (source, grain) in enumerate(calls):
        np.testing.assert_array_equal(source, frames[index])
        assert grain.seed == Seed(0).to_u32()
        if index == 0:
            assert grain.previous is None
        else:
            assert grain.frame == index
            np.testing.assert_array_equal(grain.previous, frames[index - 1])


def test_a_failing_restoration_ends_the_loop_and_drops_the_step(monkeypatch, tmp_path):
    frames = video(8)
    restorer = Restorer(frames)
    contexts, views = [], []

    def restore(image, context):
        # The node's own context (PyTorch restores on the node's thread).
        contexts.append(context)
        views.append(context.iteration)
        if restorer.index(image) == 4:
            raise ValueError("frame 4 failed")
        return restorer(image, context)

    pipeline, saved = vosr_loop(
        monkeypatch,
        tmp_path,
        frames,
        "6",
        TemporalMode.THREE_FRAMES,
        restore,
        source="chainner:image:load_video",
    )
    error = pipeline.run(workers=4)
    assert error is not None and "frame 4 failed" in str(error)
    # Item 3 restores frame 4 ahead, so items 0 to 2 were saved.
    assert len(saved) == 3
    # The loop's end dropped the step and detached the view.
    assert isinstance(views[0].state, type(None)) and len(set(views)) == 1
    assert {context.iteration for context in contexts} == {None}
