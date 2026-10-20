"""Temporal Fix (vs_temporalfix's AI model): upstream's gen_shifts mirroring at the clip
edges, the past-only fallback, the per-run frame state, strength handling, the pinned
weight download, and the node in a real executor loop (CPU, random weights)."""

import hashlib
import http.server
import logging
import threading
from collections import OrderedDict
from functools import partial
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from test_generator_lifecycle import make_node
from test_item_window import Pipeline, set_schema
from test_iteration_view import Reader

from api import BaseInput, InputId, NodeContext, NodeId
from chain.chain import Edge, EdgeTarget
from nodes.impl.pytorch import temporal_fix
from nodes.impl.pytorch.temporal_fix import RADIUS, FrameWindow, window
from nodes.properties.inputs import DirectoryInput, ImageInput
from packages.chaiNNer_pytorch.pytorch.processing import temporal_fix as node


def gen_shifts(frames: int) -> list[list[int]]:
    """utils.gen_shifts(clip, radius=3) on frame indices, literally: window[n][k]."""
    clip = list(range(frames))
    padded = clip[1 : RADIUS + 1][::-1] + clip + clip[-2 : -RADIUS - 2 : -1]
    shifts = [
        padded[RADIUS + k : RADIUS + k + frames] for k in range(-RADIUS, RADIUS + 1)
    ]
    return [[shift[n] for shift in shifts] for n in range(frames)]


@pytest.mark.parametrize("frames", [4, 5, 9, 30])
def test_the_window_mirrors_like_gen_shifts(frames):
    assert [window(n, 0, frames - 1) for n in range(frames)] == gen_shifts(frames)
    assert window(0, 0, 29) == [3, 2, 1, 0, 1, 2, 3]
    assert window(29, 0, 29) == [26, 27, 28, 29, 28, 27, 26]


def test_shorter_clips_keep_reflecting():
    assert window(0, 0, 0) == [0] * 7
    assert window(1, 0, 1) == [0, 1, 0, 1, 0, 1, 0]


def run_window(frames, ahead, skip=()):
    peeks = []
    state = FrameWindow[int](ahead, (1,))

    def peek(index, offset):
        peeks.append(index + offset)
        return index + offset if index + offset < frames else None

    windows = {}
    for n in range(frames):
        if n in skip:
            continue
        windows[n] = state.gather(n, lambda n=n: n, partial(peek, n))
        assert set(state.frames) <= set(range(n - RADIUS + 1, n + RADIUS + 1))
    return windows, peeks


@pytest.mark.parametrize("frames", [4, 8, 30])
def test_reading_ahead_gives_upstream_windows_and_reads_each_frame_once(frames):
    windows, peeks = run_window(frames, ahead=True)
    assert list(windows.values()) == gen_shifts(frames)
    assert sorted(set(peeks)) == list(range(1, frames + 1))
    assert len([p for p in peeks if p < frames]) == frames - 1


def test_without_frames_ahead_each_frame_is_the_last_one():
    windows, peeks = run_window(8, ahead=False)
    assert peeks == []
    assert windows[0] == [0] * 7
    assert windows[1] == [0, 1, 0, 1, 0, 1, 0]
    assert windows[2] == [1, 0, 1, 2, 1, 0, 1]
    for n in range(3, 8):
        assert windows[n] == [n - 3, n - 2, n - 1, n, n - 1, n - 2, n - 3]


def test_a_missing_item_starts_the_clip_afresh():
    windows, _ = run_window(10, ahead=False, skip={4})
    assert windows[5] == [5] * 7
    assert windows[6] == window(6, 5, 6)
    assert windows[3] == [0, 1, 2, 3, 2, 1, 0]


def write_checkpoints(folder):
    torch.manual_seed(0)
    pins = {}
    for strength in range(4):
        name = f"s{strength}.pth"
        state = OrderedDict(
            weight=torch.randn(3, 2), count=torch.tensor(strength, dtype=torch.int64)
        )
        torch.save(state, folder / name)
        pins[strength] = (
            name,
            hashlib.sha256((folder / name).read_bytes()).hexdigest(),
        )
    return pins


def test_strengths_between_checkpoints_lerp_their_floating_tensors(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(temporal_fix, "CHECKPOINTS", write_checkpoints(tmp_path))
    monkeypatch.setattr(temporal_fix, "URL", "http://127.0.0.1:9/unreachable/")
    s1 = torch.load(tmp_path / "s1.pth", weights_only=True)
    s2 = torch.load(tmp_path / "s2.pth", weights_only=True)
    for strength, expected in ((1.0, s1), (2, s2)):
        state = temporal_fix.load_state(tmp_path, strength)
        assert torch.equal(state["weight"], expected["weight"])
    state = temporal_fix.load_state(tmp_path, 1.254)
    assert torch.equal(
        state["weight"], torch.lerp(s1["weight"], s2["weight"], 1.25 - 1)
    )
    assert torch.equal(state["count"], s1["count"])


class Server:
    def __init__(self, folder):
        self.requests = []
        outer = self

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(folder), **kwargs)

            def do_GET(self):
                outer.requests.append(self.path)
                super().do_GET()

            def log_message(self, format, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()


def test_the_download_is_checked_against_its_pin(monkeypatch, tmp_path):
    served = tmp_path / "served"
    served.mkdir()
    pins = write_checkpoints(served)
    storage = tmp_path / "storage"
    server = Server(served)
    try:
        monkeypatch.setattr(temporal_fix, "URL", server.url)
        monkeypatch.setattr(temporal_fix, "CHECKPOINTS", pins)
        path = temporal_fix.checkpoint_path(storage, 2)
        assert path.read_bytes() == (served / "s2.pth").read_bytes()
        assert server.requests == ["/s2.pth"]
        # A verified file is not downloaded again.
        assert temporal_fix.checkpoint_path(storage, 2) == path
        assert server.requests == ["/s2.pth"]
        # Served bytes that do not match the pin are refused and not saved.
        monkeypatch.setattr(temporal_fix, "CHECKPOINTS", {3: ("s3.pth", "0" * 64)})
        with pytest.raises(RuntimeError, match="does not match the SHA-256"):
            temporal_fix.checkpoint_path(storage, 3)
        assert sorted(p.name for p in storage.iterdir()) == ["s2.pth"]
        # A corrupted local copy is replaced by a verified download.
        path.write_bytes(b"corrupt")
        monkeypatch.setattr(temporal_fix, "CHECKPOINTS", pins)
        assert (
            temporal_fix.checkpoint_path(storage, 2).read_bytes()
            == (served / "s2.pth").read_bytes()
        )
        monkeypatch.setattr(temporal_fix, "URL", server.url + "missing/")
        with pytest.raises(RuntimeError, match="Could not download"):
            temporal_fix.checkpoint_path(storage, 1)
    finally:
        server.close()


def test_strength_0_and_outside_a_loop_return_the_image_unchanged(monkeypatch):
    image = np.random.default_rng(0).random((6, 8, 3), dtype=np.float32)
    monkeypatch.setattr(node, "get_settings", pytest.fail)
    for iteration, strength in ((None, 2.0), (SimpleNamespace(), 0.0)):
        context = cast(NodeContext, SimpleNamespace(iteration=iteration))
        assert node.temporal_fix_node(context, image, strength) is image


# The node in a real executor loop over the item window (CPU, random weights).

SIZE = (32, 48)


def registered():
    """The node's NodeData as the package registered it."""
    return next(
        data
        for data in node.processing_group.nodes
        if data.schema_id == "chainner:pytorch:temporal_fix"
    )


def test_the_node_declares_its_neighbours():
    # The three frames before and after each one: a Load Video Range reads them too.
    assert registered().temporal_context == (RADIUS, RADIUS) == (3, 3)


def random_model(hw):
    torch.manual_seed(1)
    model = temporal_fix.temporalfix_arch(
        fixed_hw=hw, conf_thresh=0.6, min_support=1, gate_slope=12.0, count_slope=4.0
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.1)
    return model.state_dict()


def fix_loop(monkeypatch, tmp_path, frames, window_size, source):
    monkeypatch.setattr(
        node,
        "get_settings",
        lambda _: SimpleNamespace(
            device=torch.device("cpu"), budget_limit=0, force_cache_wipe=False
        ),
    )
    loads = []

    def load_state(_folder, strength):
        loads.append(strength)
        return random_model(SIZE)

    monkeypatch.setattr(node, "load_state", load_state)
    pipeline = Pipeline(
        monkeypatch,
        tmp_path,
        len(frames),
        window=window_size,
        nodes=(),
        reader=Reader(iter(frames)),
    )
    set_schema(pipeline.chain.nodes[NodeId("g0")], source)
    edge = pipeline.chain.edge_to(NodeId("save"), InputId(0))
    assert edge is not None
    pipeline.chain.remove_edge(edge)
    fix = make_node(
        "fix",
        "regularNode",
        lambda context, image: node.temporal_fix_node(context, image, 2.0),
        1,
        1,
    )
    object.__setattr__(fix.data, "inputs", [ImageInput(channels=3).with_id(0)])
    object.__setattr__(fix.data, "side_effects", False)
    object.__setattr__(fix.data, "node_context", True)
    object.__setattr__(fix.data, "temporal_context", registered().temporal_context)
    set_schema(fix, "chainner:pytorch:temporal_fix")
    pipeline.chain.add_node(fix)
    pipeline.chain.add_edge(Edge(edge.source, EdgeTarget(NodeId("fix"), InputId(0))))
    pipeline.connect("fix", "save", 0)
    saved = []
    save = pipeline.chain.nodes[NodeId("save")]
    inputs = [BaseInput("any", "Input0").with_id(0), DirectoryInput(must_exist=False)]
    object.__setattr__(save.data, "inputs", [inputs[0], inputs[1].with_id(1)])
    object.__setattr__(save.data, "run", lambda value, _folder: saved.append(value))
    folder = make_node("folder", "regularNode", lambda: tmp_path / "out", 0, 1)
    object.__setattr__(folder.data, "side_effects", False)
    pipeline.chain.add_node(folder)
    pipeline.connect("folder", "save", 1)
    return pipeline, saved, loads


def expected(frames, windows):
    model = temporal_fix.temporalfix_arch(
        fixed_hw=SIZE, conf_thresh=0.6, min_support=1, gate_slope=12.0, count_slope=4.0
    )
    model.load_state_dict(random_model(SIZE))
    model.eval()
    planes = [torch.from_numpy(f).clamp(0, 1).permute(2, 0, 1).flip(0) for f in frames]
    out = []
    with torch.inference_mode():
        for indices in windows:
            result = model(torch.stack([planes[i] for i in indices])[None])[0]
            out.append(result.flip(0).permute(1, 2, 0).numpy())
    return out


def test_in_a_loop_the_node_uses_upstream_windows(monkeypatch, tmp_path, caplog):
    # Load Video reads no file per item: K = 6 serves five frames ahead. At K = 1 the
    # node mirrors the past frames and says so once.
    rng = np.random.default_rng(2)
    frames = [rng.random((*SIZE, 3), dtype=np.float32) * 1.1 - 0.05 for _ in range(6)]
    for window_size, windows, logs in (
        ("6", gen_shifts(6), 0),
        ("1", [window(n, 0, n) for n in range(6)], 1),
    ):
        pipeline, saved, loads = fix_loop(
            monkeypatch, tmp_path, frames, window_size, "chainner:image:load_video"
        )
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="sanic.root"):
            assert pipeline.run(workers=4) is None
        assert loads == [2.0]
        messages = [
            r.getMessage() for r in caplog.records if "Temporal Fix" in r.getMessage()
        ]
        assert len(messages) == logs
        for got, want in zip(saved, expected(frames, windows), strict=True):
            np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(
    ("base", "window_size"), [(4, "6"), (1, "6"), (8, "6"), (4, "1")]
)
def test_a_range_gives_the_outputs_of_a_full_run(
    monkeypatch, tmp_path, base, window_size
):
    # Load Video's Range base..base + 3 of 12 frames: the context frames give the node
    # the windows a full run gives it at those frames, so the outputs are the full
    # run's, bit for bit, near the clip's start (1) and end (8) too. At K = 1 both use
    # the past frames alone.
    rng = np.random.default_rng(3)
    frames = [rng.random((*SIZE, 3), dtype=np.float32) for _ in range(12)]
    runs = {}
    for ranged in (False, True):
        pipeline, saved, _loads = fix_loop(
            monkeypatch, tmp_path, frames, window_size, "chainner:image:load_video"
        )
        asked = pipeline.read_range(frames, base, 4) if ranged else []
        assert pipeline.run(workers=4) is None
        runs[ranged] = (saved, asked)
    (full, _), (saved, asked) = runs[False], runs[True]
    assert asked == [(RADIUS, RADIUS)] and len(full) == 12
    for got, want in zip(saved, full[base : base + 4], strict=True):
        np.testing.assert_array_equal(got, want)
