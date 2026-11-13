"""Upscale Image with a temporal model (RDUAL): the node outside a loop, its input rules,
and the node in a real executor loop over the item window (CPU; a stand-in for spandrel's
run_window for the window plan, then the real model, random weights)."""

import inspect
import time
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from engine_differential import execute
from spandrel import MAIN_REGISTRY, ImageModelDescriptor
from spandrel.architectures.Compact import Compact
from spandrel.architectures.RDUAL import RDUAL, RDUALArch, run_clip, run_window
from test_engine_scenarios import Builder
from test_generator_lifecycle import make_node
from test_item_window import Pipeline, set_schema
from test_iteration_view import Reader

import api
import native_executor
import process
from api import BaseInput, Generator, InputId, NodeContext, NodeId
from api.group import Group
from chain.chain import Edge, EdgeTarget
from nodes.impl.item_window import ITEM_WINDOW_VARIABLE
from nodes.impl.pytorch.rdual_window import (
    LIGHTER_MODES,
    RDUALWindows,
    is_temporal,
    lighter_modes,
    window_peak,
)
from nodes.impl.resize import ResizeFilter, resize
from nodes.impl.upscale.auto_split_tiles import NO_TILING
from nodes.impl.upscale.basic_upscale import PAD_SIZE, PaddingType
from nodes.impl.upscale.temporal_window import (
    IMAGE_INPUT,
    LOOK_AHEAD,
    WARM_UP,
)
from nodes.properties.inputs import DirectoryInput, ImageInput
from packages.chaiNNer_pytorch.pytorch.processing import upscale_image as node

# A stand-in window: (its frames, how many it keeps, the frame that links its state).
Window = tuple[list[int], int, int | None]


def upscale_node(context, img, model):
    """Upscale Image's image output with its defaults: no Custom Scale, no tiling, no
    padding, no Separate Alpha, no engine."""
    return node.upscale_image_node(
        context, img, model, False, 4, NO_TILING, 256, PaddingType.NONE, False, False
    )[0]


def node_inputs():
    """upscale_node's inputs, in its order: the image (the peeked one), then the model."""
    return [
        ImageInput(channels=3).with_id(IMAGE_INPUT),
        BaseInput("any", "Model").with_id(0),
    ]


# The node: a real RDUAL with random weights, on the CPU.

SIZE = (16, 20)


@pytest.fixture(scope="module")
def descriptor():
    torch.manual_seed(0)
    model = RDUAL(scale=2, use_checkpoint=False)
    with torch.no_grad():
        for parameter in model.temporal.parameters():
            parameter.normal_(0, 0.05)
    model.get_buffer("flow_loaded").fill_(True)
    return RDUALArch().load(model.state_dict())


def registered():
    return next(
        data
        for data in node.processing_group.nodes
        if data.schema_id == "chainner:pytorch:upscale_image"
    )


@pytest.fixture(scope="module")
def image_model():
    """A small 2x image model (Compact), not temporal."""
    torch.manual_seed(1)
    model = Compact(num_feat=4, num_conv=1, upscale=2)
    descriptor = MAIN_REGISTRY.load(model.state_dict())
    assert isinstance(descriptor, ImageModelDescriptor)
    return descriptor


def test_the_node_declares_its_neighbours_for_a_temporal_model(descriptor, image_model):
    data = registered()
    assert data.temporal_context == (WARM_UP, LOOK_AHEAD) == (5, 1)
    assert IMAGE_INPUT == 1
    assert data.temporal_if is not None
    gate, applies = data.temporal_if
    assert gate == 0
    assert applies(descriptor) and is_temporal(descriptor)
    assert not applies(image_model) and not is_temporal(image_model)
    # A value it cannot judge keeps the context.
    assert applies(None) and applies("not a model")


def frames_of(count, seed):
    rng = np.random.default_rng(seed)
    return [rng.random((*SIZE, 3), dtype=np.float32) for _ in range(count)]


def planes(frames):
    """BGR HWC images as the node converts them: RGB T,3,H,W."""
    return torch.stack([torch.from_numpy(f).permute(2, 0, 1).flip(0) for f in frames])


def direct(descriptor, frames):
    """The outputs run_clip gives the whole list as one clip, as BGR images."""
    with torch.inference_mode():
        out = run_clip(descriptor.model, planes(frames)[None])[0]
    return [RDUALWindows.to_image(f) for f in out]


def test_outside_a_loop_the_image_is_a_clip_of_one_frame(monkeypatch, descriptor):
    monkeypatch.setattr(
        node,
        "get_settings",
        lambda _: SimpleNamespace(device=torch.device("cpu"), budget_limit=0),
    )
    image = frames_of(1, 3)[0]
    context = cast(NodeContext, SimpleNamespace(iteration=None, node_id=NodeId("up")))
    got = upscale_node(context, image, descriptor)
    assert got.shape == (SIZE[0] * 2, SIZE[1] * 2, 3)
    np.testing.assert_array_equal(got, direct(descriptor, [image])[0])


def temporal_loop(monkeypatch, tmp_path, frames, window_size, descriptor):
    monkeypatch.setattr(
        node,
        "get_settings",
        lambda _: SimpleNamespace(device=torch.device("cpu"), budget_limit=0),
    )
    pipeline = Pipeline(
        monkeypatch,
        tmp_path,
        len(frames),
        window=window_size,
        nodes=(),
        reader=Reader(iter(frames)),
    )
    set_schema(pipeline.chain.nodes[NodeId("g0")], "chainner:image:load_video")
    edge = pipeline.chain.edge_to(NodeId("save"), InputId(0))
    assert edge is not None
    pipeline.chain.remove_edge(edge)
    model = make_node("model", "regularNode", lambda: descriptor, 0, 1)
    object.__setattr__(model.data, "side_effects", False)
    pipeline.chain.add_node(model)
    upscale = make_node("upscale", "regularNode", upscale_node, 2, 1)
    object.__setattr__(
        upscale.data,
        "inputs",
        node_inputs(),
    )
    object.__setattr__(upscale.data, "side_effects", False)
    object.__setattr__(upscale.data, "node_context", True)
    object.__setattr__(upscale.data, "temporal_context", registered().temporal_context)
    object.__setattr__(upscale.data, "temporal_if", registered().temporal_if)
    set_schema(upscale, "chainner:pytorch:upscale_image")
    pipeline.chain.add_node(upscale)
    pipeline.connect("model", "upscale", 0)
    pipeline.chain.add_edge(
        Edge(edge.source, EdgeTarget(NodeId("upscale"), InputId(IMAGE_INPUT)))
    )
    pipeline.connect("upscale", "save", 0)
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


def window_stand_in(seen: list[Window], mixed=False):
    """A stand-in for run_window over frames whose first pixel is their index / 100: it
    records the window's frames, how many it keeps and the frame that links its carried
    state (None cold), and gives its kept frames at twice the size, plus the window's
    mean when `mixed`, so a window of other frames changes them."""

    def name(frame: torch.Tensor) -> int:
        return round(float(frame[0, 0, 0]) * 100)

    def stand_in(_model, clip, keep, past):
        link = None if past is None else name(past[1][0])
        seen.append(([name(f) for f in clip[0]], keep, link))
        up = clip.repeat_interleave(2, -1).repeat_interleave(2, -2)
        if mixed:
            up = up + up.mean(1, keepdim=True)
        return up[:, :keep], clip[:, keep - 1]

    return stand_in


@pytest.mark.parametrize(
    ("window_size", "windows"),
    [
        # Load Video reads no file per item: K = 6 serves five frames ahead.
        (
            "6",
            [
                ([0, 1, 2, 3, 4, 5], 5, None),
                ([5, 6, 7, 8, 9], 5, 4),
            ],
        ),
        # No window: every frame with the state of the frames before it only.
        ("1", [([n], 1, n - 1 if n else None) for n in range(10)]),
    ],
)
def test_in_a_loop_the_node_reads_the_frames_ahead(
    monkeypatch, tmp_path, descriptor, window_size, windows
):
    # Frame n is filled with n / 100, so a stand-in window function can name its frames.
    frames = [np.full((*SIZE, 3), n / 100, np.float32) for n in range(10)]
    seen: list[Window] = []
    monkeypatch.setattr(
        "spandrel.architectures.RDUAL.run_window", window_stand_in(seen)
    )
    pipeline, saved = temporal_loop(
        monkeypatch, tmp_path, frames, window_size, descriptor
    )
    assert pipeline.run(workers=4) is None
    assert seen == windows
    for n, got in enumerate(saved):
        np.testing.assert_array_equal(
            got, np.full((SIZE[0] * 2, SIZE[1] * 2, 3), n / 100, np.float32)
        )


def add_upscale(b: Builder, name: str, descriptor, source: str) -> None:
    """An Upscale Image `name` of `source`'s image, its model from a node outside the
    loop, registered as the node is: its context and its gate on the model."""
    data = registered()
    b.node(f"{name} model", lambda: descriptor, inputs=0)
    upscale = b.node(
        name,
        upscale_node,
        inputs=2,
        context=True,
        temporal=data.temporal_context,
        gate=data.temporal_if,
    )
    object.__setattr__(upscale.data, "inputs", node_inputs())
    set_schema(upscale, "chainner:pytorch:upscale_image")
    b.edge(f"{name} model", 0, name, 0)
    b.edge(source, 0, name, IMAGE_INPUT)


def frames_node(generator) -> Builder:
    b = Builder()
    set_schema(
        b.node("frames", generator, kind="generator", inputs=0),
        "chainner:image:load_video",
    )
    return b


def upscale_chain(descriptor, generator) -> Builder:
    """frames (a Load Video) -> upscale <- model, as /run gets it."""
    b = frames_node(generator)
    add_upscale(b, "upscale", descriptor, "frames")
    return b


@pytest.mark.parametrize(
    ("models", "asked"),
    [
        # An image model: the loop reads no context item for it.
        (["image"], []),
        (["rdual"], [(WARM_UP, LOOK_AHEAD)]),
        # An image model's Upscale Image before an RDUAL one adds nothing to the sum.
        (["image", "rdual"], [(WARM_UP, LOOK_AHEAD)]),
    ],
)
@pytest.mark.parametrize("kind", ["native", "python"])
def test_only_a_temporal_model_asks_for_context_items(
    monkeypatch, tmp_path, descriptor, image_model, kind, models, asked
):
    monkeypatch.setattr(
        node,
        "get_settings",
        lambda _: SimpleNamespace(
            device=torch.device("cpu"),
            use_fp16=False,
            use_cpu_bf16=False,
            budget_limit=0,
            force_cache_wipe=False,
            gpu_index=0,
        ),
    )
    monkeypatch.setattr(
        api.registry, "get_package", lambda _: SimpleNamespace(id="test")
    )
    monkeypatch.setenv(ITEM_WINDOW_VARIABLE, "6")
    monkeypatch.setattr("spandrel.architectures.RDUAL.run_window", window_stand_in([]))
    frames = [np.full((*SIZE, 3), n / 100, np.float32) for n in range(10)]
    seen: list[tuple[int, int]] = []
    base = 5

    def context(before: int, after: int):
        seen.append((before, after))
        b, a = min(before, base), min(after, len(frames) - base - 1)
        return iter(frames[base - b : base + 1 + a]), b, a

    def generator() -> Generator:
        one = Generator(lambda: iter(frames[base : base + 1]), 1, source_paths=())
        return one.with_range(base, base, context)

    b = frames_node(generator)
    source = "frames"
    for n, kind_of_model in enumerate(models):
        model = descriptor if kind_of_model == "rdual" else image_model
        add_upscale(b, f"upscale {n}", model, source)
        source = f"upscale {n}"
    saved: list[np.ndarray] = []
    b.node("save", saved.append, outputs=0, side_effects=True)
    b.edge(source, 0, "save", 0)
    result = execute(kind, b.chain, storage=tmp_path / "storage")
    assert result.error is None, result.error
    assert seen == asked
    scale = 2 ** len(models)
    assert [s.shape for s in saved] == [(SIZE[0] * scale, SIZE[1] * scale, 3)]


@pytest.mark.parametrize(
    ("base", "window_size", "windows"),
    [
        # K = 8 (32 CPUs): the 5 warm-up frames 10..14, the frame and the context frame
        # after it, 16, run as one window that reaches the last context frame.
        (15, "8", [([10, 11, 12, 13, 14, 15, 16], 7, None)]),
        # K = 6: the first window carries its state to the frame's.
        (15, "6", [([10, 11, 12, 13, 14, 15], 5, None), ([15, 16], 2, 14)]),
        # Fewer than 5 frames before it: the warm-up starts at the clip's first.
        (3, "6", [([0, 1, 2, 3, 4], 5, None)]),
        (1, "6", [([0, 1, 2], 3, None)]),
        # The clip's last frame: no context frame after it.
        (19, "8", [([14, 15, 16, 17, 18, 19], 6, None)]),
        # No window: one window per item. The context items after the frame only serve
        # look-aheads, never the node, so none runs a window nobody reads.
        (15, "1", [([n], 1, n - 1 if n > 10 else None) for n in range(10, 16)]),
    ],
)
@pytest.mark.parametrize("kind", ["native", "python"])
def test_run_this_frame_gives_the_frame_its_neighbours(
    monkeypatch, tmp_path, descriptor, kind, base, window_size, windows
):
    # Load Video's Run this frame (a Range whose Start is its End), as /run runs it on each
    # executor: the loop reads WARM_UP context frames before it and LOOK_AHEAD after.
    monkeypatch.setattr(
        node,
        "get_settings",
        lambda _: SimpleNamespace(device=torch.device("cpu"), budget_limit=0),
    )
    monkeypatch.setattr(
        api.registry, "get_package", lambda _: SimpleNamespace(id="test")
    )
    monkeypatch.setenv(ITEM_WINDOW_VARIABLE, window_size)
    frames = [np.full((*SIZE, 3), n / 100, np.float32) for n in range(20)]
    seen: list[Window] = []
    monkeypatch.setattr(
        "spandrel.architectures.RDUAL.run_window", window_stand_in(seen)
    )
    asked: list[tuple[int, int]] = []

    def context(before: int, after: int):
        asked.append((before, after))
        b, a = min(before, base), min(after, len(frames) - base - 1)
        return iter(frames[base - b : base + 1 + a]), b, a

    def generator() -> Generator:
        one = Generator(lambda: iter(frames[base : base + 1]), 1, source_paths=())
        return one.with_range(base, base, context)

    saved: list[np.ndarray] = []
    b = upscale_chain(descriptor, generator)
    b.node("save", saved.append, outputs=0, side_effects=True)
    b.edge("upscale", 0, "save", 0)
    result = execute(kind, b.chain, storage=tmp_path / "storage")
    assert result.error is None, result.error
    assert asked == [(WARM_UP, LOOK_AHEAD)]
    assert seen == windows
    assert len(saved) == 1
    np.testing.assert_array_equal(
        saved[0], np.full((SIZE[0] * 2, SIZE[1] * 2, 3), base / 100, np.float32)
    )


def ranged(frames, base, count):
    """Load Video's Range of `count` frames from `base`, with its context frames."""

    def context(before: int, after: int):
        b, a = min(before, base), min(after, len(frames) - base - count)
        return iter(frames[base - b : base + count + a]), b, a

    def generator() -> Generator:
        items = Generator(
            lambda: iter(frames[base : base + count]), count, source_paths=()
        )
        return items.with_range(base, None, context)

    return generator


def read_by_all(monkeypatch):
    """Every output read, as before nodes were told of unread ones."""
    for view in (process._IterationView, native_executor._EngineIterationView):
        monkeypatch.setattr(view, "unread", property(lambda _view: False))


@pytest.mark.parametrize("reader", [False, True])
@pytest.mark.parametrize("kind", ["native", "python"])
def test_a_range_saves_the_same_frames_whether_its_warm_up_outputs_are_read_or_not(
    monkeypatch, tmp_path, descriptor, kind, reader
):
    # A stand-in whose outputs depend on the whole window, so a window of other frames
    # changes them; frame n's first pixel names it. With `reader`, a temporal node after
    # the upscale reads its outputs on the context frames too, so they stay real.
    monkeypatch.setattr(
        node,
        "get_settings",
        lambda _: SimpleNamespace(device=torch.device("cpu"), budget_limit=0),
    )
    monkeypatch.setattr(
        api.registry, "get_package", lambda _: SimpleNamespace(id="test")
    )
    monkeypatch.setenv(ITEM_WINDOW_VARIABLE, "6")
    rng = np.random.default_rng(6)
    frames = [
        n / 100 + rng.random((*SIZE, 3), dtype=np.float32) / 1000 for n in range(24)
    ]
    for frame in frames:
        frame[0, 0] = round(float(frame[0, 0, 2]) * 100) / 100

    def run() -> tuple[list[np.ndarray], list[tuple[int, bool, np.ndarray]], list]:
        seen: list[Window] = []
        monkeypatch.setattr(
            "spandrel.architectures.RDUAL.run_window", window_stand_in(seen, mixed=True)
        )
        saved: list[np.ndarray] = []
        read: list[tuple[int, bool, np.ndarray]] = []
        b = upscale_chain(descriptor, ranged(frames, 16, 4))
        b.node("save", saved.append, outputs=0, side_effects=True)
        if reader:

            def temporal(context, image):
                read.append((context.iteration.index, context.iteration.context, image))
                return image

            b.node("reader", temporal, context=True, temporal=(1, 1))
            b.edge("upscale", 0, "reader", 0)
            b.edge("reader", 0, "save", 0)
        else:
            b.edge("upscale", 0, "save", 0)
        result = execute(kind, b.chain, storage=tmp_path / "storage")
        assert result.error is None, result.error
        return saved, read, seen

    saved, read, seen = run()
    read_by_all(monkeypatch)
    before, read_before, seen_before = run()
    assert len(saved) == 4
    for got, want in zip(saved, before, strict=True):
        assert got.tobytes() == want.tobytes()
    if reader:
        # Context frames 10..15 are upscaled as before.
        assert [(i, c) for i, c, _ in read] == [(i, i < 16) for i in range(10, 20)]
        assert [(i, c) for i, c, _ in read_before] == [(i, c) for i, c, _ in read]
        for (_, _, got), (_, _, want) in zip(read, read_before, strict=True):
            assert got.shape == (SIZE[0] * 2, SIZE[1] * 2, 3)
            assert got.tobytes() == want.tobytes()
        assert seen == [
            ([10, 11, 12, 13, 14, 15], 5, None),
            ([15, 16, 17, 18, 19, 20], 5, 14),
        ]
    else:
        # Warm-up frames 11..15: their window keeps them, the next the range's frames.
        assert seen == [
            ([11, 12, 13, 14, 15, 16], 5, None),
            ([16, 17, 18, 19, 20], 5, 15),
        ]
    assert seen_before == seen


@pytest.mark.parametrize("kind", ["native", "python"])
def test_a_context_item_runs_each_temporal_node_once(monkeypatch, tmp_path, kind):
    # b reads a: on a context item a ran a second time once b had taken its output.
    monkeypatch.setattr(
        api.registry, "get_package", lambda _: SimpleNamespace(id="test")
    )
    calls: list[tuple[str, int]] = []

    def temporal(name):
        def run(context, value):
            calls.append((name, context.iteration.index))
            return value

        return run

    b = Builder()
    b.node("frames", ranged(list(range(10)), 4, 2), kind="generator", inputs=0)
    for name in ("a", "b"):
        b.node(name, temporal(name), context=True, temporal=(1, 1))
    b.node("save", lambda _: None, outputs=0, side_effects=True)
    b.edge("frames", 0, "a", 0)
    b.edge("a", 0, "b", 0)
    b.edge("b", 0, "save", 0)
    result = execute(kind, b.chain, storage=tmp_path / "storage")
    assert result.error is None, result.error
    assert sorted(calls) == [(n, i) for n in ("a", "b") for i in range(2, 6)]


@pytest.mark.parametrize(
    ("edges", "unread"),
    [
        # Nothing reads a on a context item.
        ([("frames", "a", 0), ("a", "save", 0)], {"a"}),
        # b, which runs on context items, reads a; only the save reads b.
        ([("frames", "a", 0), ("a", "b", 0), ("b", "save", 0)], {"b"}),
        # The same through a pure node, with a also saved.
        (
            [
                ("frames", "a", 0),
                ("a", "pure", 0),
                ("pure", "b", 0),
                ("b", "save", 0),
                ("a", "save", 1),
            ],
            {"b"},
        ),
        # Side by side.
        (
            [
                ("frames", "a", 0),
                ("frames", "b", 0),
                ("a", "save", 0),
                ("b", "save", 1),
            ],
            {"a", "b"},
        ),
    ],
)
@pytest.mark.parametrize("kind", ["native", "python"])
def test_each_engine_tells_a_node_its_output_is_unread(
    monkeypatch, tmp_path, kind, edges, unread
):
    monkeypatch.setattr(
        api.registry, "get_package", lambda _: SimpleNamespace(id="test")
    )
    monkeypatch.setenv(ITEM_WINDOW_VARIABLE, "4")
    flags: list[tuple[str, int, bool, bool]] = []

    def temporal(name):
        def run(context, value):
            view = context.iteration
            flags.append((name, view.index, view.context, view.unread))
            return value

        return run

    b = Builder()
    b.node("frames", ranged(list(range(10)), 4, 2), kind="generator", inputs=0)
    names = {source for source, _, _ in edges} | {target for _, target, _ in edges}
    for name in sorted(names & {"a", "b"}):
        b.node(name, temporal(name), context=True, temporal=(1, 1))
    if "pure" in names:
        b.node("pure", lambda x: x + 1)
    b.node("save", lambda *_: None, inputs=2, outputs=0, side_effects=True)
    if ("a", "save", 1) not in edges and ("b", "save", 1) not in edges:
        b.edge("frames", 0, "save", 1)
    for source, target, index in edges:
        b.edge(source, 0, target, index)
    result = execute(kind, b.chain, storage=tmp_path / "storage")
    assert result.error is None, result.error
    assert {name for name, _, c, _ in flags if c} == names & {"a", "b"}
    assert all(u == (c and name in unread) for name, _, c, u in flags)


@pytest.fixture
def few_threads():
    # On a busy PC the loop's windows took over four times as long as the same windows
    # run directly; few intra-op threads wait less for cores other processes hold.
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(threads)


def test_in_a_loop_the_real_model_gives_its_windows_outputs(
    monkeypatch, tmp_path, descriptor, few_threads
):
    frames = frames_of(7, 5)
    clip = planes(frames)[None]
    # K = 6: item 0's window is frames 0..5 and keeps 0..4; item 5's is 5..6, the end,
    # with the state of frame 4.
    started = time.perf_counter()
    with torch.inference_mode():
        first, state = run_window(descriptor.model, clip[:, :6], 5)
        second, _ = run_window(descriptor.model, clip[:, 5:], 2, (state, clip[:, 4]))
    windows = time.perf_counter() - started
    want = [RDUALWindows.to_image(f) for f in torch.cat((first, second), 1)[0]]
    pipeline, saved = temporal_loop(monkeypatch, tmp_path, frames, "6", descriptor)
    # The loop runs the same two windows: its hang guard scales with their time under
    # the machine's load.
    assert pipeline.run(workers=4, timeout=10 + round(4 * windows)) is None
    assert len(saved) == 7
    for got, expected in zip(saved, want, strict=True):
        np.testing.assert_array_equal(got, expected)


# Input rules for a temporal model, outside a loop, with a stand-in for run_window.


def doubling(_model, clip, keep, _past):
    up = clip.repeat_interleave(2, -1).repeat_interleave(2, -2)
    return up[:, :keep], clip[:, keep - 1]


def run_node(
    descriptor,
    img,
    *,
    use_custom_scale=False,
    custom_scale=4,
    padding=PaddingType.NONE,
    separate_alpha=False,
):
    context = cast(NodeContext, SimpleNamespace(iteration=None, node_id=NodeId("up")))
    image, text = node.upscale_image_node(
        context,
        img,
        descriptor,
        use_custom_scale,
        custom_scale,
        NO_TILING,
        256,
        padding,
        separate_alpha,
        False,
    )
    assert text == "Whole image"
    return image


@pytest.fixture
def stand_in_window(monkeypatch):
    monkeypatch.setattr(
        node,
        "get_settings",
        lambda _: SimpleNamespace(device=torch.device("cpu"), budget_limit=0),
    )
    monkeypatch.setattr("spandrel.architectures.RDUAL.run_window", doubling)


def test_custom_scale_and_separate_alpha_do_not_apply(descriptor, stand_in_window):
    rgba = np.random.default_rng(2).random((*SIZE, 4), dtype=np.float32)
    plain = run_node(descriptor, rgba)
    assert plain.shape == (SIZE[0] * 2, SIZE[1] * 2, 4)
    ignored = run_node(
        descriptor, rgba, use_custom_scale=True, custom_scale=4, separate_alpha=True
    )
    np.testing.assert_array_equal(ignored, plain)
    np.testing.assert_array_equal(
        plain[:, :, :3], rgba[:, :, :3].repeat(2, 0).repeat(2, 1)
    )
    alpha = resize(rgba[:, :, 3], (SIZE[1] * 2, SIZE[0] * 2), ResizeFilter.LANCZOS)
    np.testing.assert_array_equal(plain[:, :, 3], np.clip(alpha[:, :, 0], 0, 1))


def test_grayscale_and_padding_through_the_node(descriptor, stand_in_window):
    gray = np.random.default_rng(3).random(SIZE, dtype=np.float32)
    got = run_node(descriptor, gray, padding=PaddingType.REPLICATE)
    assert got.shape == (SIZE[0] * 2, SIZE[1] * 2)
    np.testing.assert_allclose(got, gray.repeat(2, 0).repeat(2, 1), atol=1e-6)
    assert PAD_SIZE == 16


def group_of(input_id, items):
    """The innermost group of the node's layout that holds the input."""
    for item in items:
        if isinstance(item, Group):
            inner = group_of(input_id, item.items)
            if inner is not None:
                return inner
            if input_id in item.items:
                return item
    return None


@pytest.mark.parametrize("input_id", [4, 3])
def test_custom_scale_and_separate_alpha_are_hidden_for_a_temporal_model(input_id):
    # 4: Custom Scale, 3: Separate Alpha.
    group = group_of(InputId(input_id), registered().group_layout)
    assert group is not None
    not_temporal = node.NOT_TEMPORAL.to_json()
    assert not_temporal["kind"] == "not"
    assert 'arch: "RDUAL"' in str(not_temporal["condition"])

    def holds(condition):
        if condition == not_temporal:
            return True
        return any(holds(item) for item in condition.get("items", ()))

    assert holds(group.info.options["condition"])


def test_any_other_model_is_not_temporal():
    assert not is_temporal(SimpleNamespace(architecture=SimpleNamespace(id="DUAL")))
    assert not is_temporal(cast(ImageModelDescriptor, None))


def test_the_lighter_memory_modes_are_those_run_window_takes(descriptor):
    taken = inspect.signature(run_window).parameters
    modes = lighter_modes()
    assert [name for name, _ in modes] == [
        name for name, options in LIGHTER_MODES if all(o in taken for o in options)
    ]
    plain = RDUALWindows(descriptor, torch.device("cpu"))
    frames = [plain.convert(f) for f in frames_of(4, 9)]
    want, _ = plain.run(frames, 2, None, 0)
    for _, lighter in modes:
        windows = lighter(plain)
        # A mode changes no state: a window carries it on.
        assert windows.key == plain.key
        got, _ = windows.run(frames, 2, None, 0)
        for g, w in zip(got, want, strict=True):
            torch.testing.assert_close(g, w, atol=1e-4, rtol=0)


@pytest.mark.parametrize(
    ("size", "ahead", "flow_volume", "hold_encodes", "measured_mib"),
    [
        # Measured window peaks (allocated after minus before, 2x RDUAL, steady state).
        ((720, 1280), 7, True, True, 3116),
        ((837, 1488), 7, True, True, 4221),
        ((1080, 1920), 7, True, True, 9015),
        ((837, 1488), 0, False, True, 2618),
        ((1080, 1920), 7, False, False, 4903),
    ],
)
def test_the_window_estimate_covers_the_measured_peaks(
    size, ahead, flow_volume, hold_encodes, measured_mib
):
    estimate = window_peak(
        size[0], size[1], ahead, flow_volume=flow_volume, hold_encodes=hold_encodes
    )
    # Within 2 % below (the margin covers it) and 15 % above the measurement.
    assert 0.98 * measured_mib <= estimate / 2**20 <= 1.15 * measured_mib


def test_tiles_are_run_window_s_and_their_state_restarts_cold(monkeypatch, descriptor):
    calls = []

    def stand_in(_model, clip, keep, past, **options):
        calls.append(options)
        return clip[:, :keep], clip[:, keep - 1]

    monkeypatch.setattr("spandrel.architectures.RDUAL.run_window", stand_in)
    plain = RDUALWindows(descriptor, torch.device("cpu"))
    tiled = plain.with_tile((64, 48))
    # Tile Size gives (width, height); run_window takes (height, width).
    assert tiled.tile == (48, 64)
    assert tiled.key != plain.key
    # A lighter mode keeps the tiles, and the tiles keep a mode.
    assert tiled.with_options(flow_volume=False).tile == (48, 64)
    assert plain.with_options(flow_volume=False).with_tile((64, 48)).options == {
        "flow_volume": False
    }
    frame = plain.convert(frames_of(1, 10)[0])
    tiled.run([frame], 1, None, 0)
    plain.run([frame], 1, None, 0)
    assert calls == [{"tile": (48, 64), "overlap": 64}, {}]


def test_a_tiled_window_s_estimate_keeps_the_whole_frame_s_flow():
    whole = window_peak(1080, 1920, 7, flow_volume=True, hold_encodes=True)
    tiled = window_peak(
        1080, 1920, 7, flow_volume=True, hold_encodes=True, tile=(540, 960)
    )
    # With the flow's volume, the whole frame's flow is the peak either way.
    assert tiled == whole
    lighter = window_peak(1080, 1920, 7, flow_volume=False, hold_encodes=True)
    lighter_tiled = window_peak(
        1080, 1920, 7, flow_volume=False, hold_encodes=True, tile=(540, 960)
    )
    assert lighter_tiled < 0.5 * lighter
