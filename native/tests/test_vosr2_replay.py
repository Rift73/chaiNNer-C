"""VOSR's PyTorch path: prepared predictions, graph replays and models kept on the device."""

from __future__ import annotations

from compression import zstd
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from nodes.impl.pytorch import vosr2_context, vosr2_plan_file, vosr2_warm_up
from nodes.impl.pytorch.vosr2 import VosrBundle, by_rows, prepare_image
from nodes.impl.pytorch.vosr2_context import Residents, window_work
from nodes.impl.pytorch.vosr2_replay import Replays
from nodes.impl.pytorch.vosr2_scene import scene_routing
from vendor.vosr2.lightningdit import LightningDiT, LightningDiTBlock
from vendor.vosr2.pos_embed import RopeTable, rotate_half


def small_dit() -> LightningDiT:
    torch.manual_seed(5)
    return LightningDiT(
        input_size=8,
        patch_size=2,
        in_channels=4,
        out_channels=2,
        hidden_size=32,
        depth=2,
        num_heads=4,
        use_rope=True,
        use_rmsnorm=True,
        use_qknorm=True,
        use_swiglu=True,
        z_dims=16,
    ).eval()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rope_table_has_the_bits_of_rotate_half(dtype):
    torch.manual_seed(1)
    t = torch.randn(1, 4, 6, 8).to(dtype)
    cos, sin = torch.rand(6, 8).to(dtype), torch.rand(6, 8).to(dtype)
    assert torch.equal(RopeTable(cos, sin)(t), t * cos + rotate_half(t) * sin)


@torch.inference_mode()
@pytest.mark.parametrize("width", [8, 10])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prepared_prediction_has_the_bits_of_the_flexible_one(width, dtype):
    dit = small_dit().to(dtype)
    x = torch.randn(1, 4, 8, width).to(dtype)
    z = torch.randn(1, 6, 16).to(dtype)
    expected = dit.forward_flexible(
        x, torch.ones(1).to(dtype), z=[z], token_chunk=2048, query_chunk=1024
    )
    c, c0 = dit.time_condition(torch.ones(1).to(dtype))
    actual = dit.forward_prepared(
        x,
        dit.condition_keys(z),
        dit.rope((4, width // 2), x.device, dtype),
        [cast(LightningDiTBlock, block).modulation(c0) for block in dit.blocks],
        dit.final_layer.modulation(c),
    )
    assert torch.equal(actual, expected)


class FakeGraph:
    instances: list[FakeGraph] = []
    fail_at_end = False

    def __init__(self):
        self.replays = 0
        FakeGraph.instances.append(self)

    def capture_begin(self, pool, capture_error_mode):
        assert capture_error_mode == "thread_local"

    def capture_end(self):
        if FakeGraph.fail_at_end:
            raise RuntimeError("the capture was invalidated")

    def replay(self):
        self.replays += 1


@pytest.fixture
def fake_cuda(monkeypatch):
    FakeGraph.instances = []
    FakeGraph.fail_at_end = False
    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: object())
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", object)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.version, "hip", None)


def test_a_shape_runs_plainly_then_is_captured_and_replayed(fake_cuda, tmp_path):
    calls = []

    def function(value):
        calls.append(value)
        return value + 1

    replays = Replays(function, torch.device("cpu"), tmp_path / "mark")
    first = torch.ones(2)
    assert torch.equal(replays(first), first + 1)
    assert len(calls) == 1 and not FakeGraph.instances
    # The second call captures; a replay reads the graph's own input tensor.
    output = replays(first * 2)
    assert len(calls) == 2 and len(FakeGraph.instances) == 1
    graph = FakeGraph.instances[0]
    assert graph.replays == 1
    assert torch.equal(replays.captured[(first.shape,)].inputs[0], first * 2)
    assert replays(first * 3) is output
    assert len(calls) == 2 and graph.replays == 2
    assert torch.equal(replays.captured[(first.shape,)].inputs[0], first * 3)
    # Another shape starts over; the first keeps its graph.
    replays(torch.ones(3))
    assert len(calls) == 3 and len(FakeGraph.instances) == 1


def test_a_rejected_capture_marks_the_gpu_and_asks_for_a_restart(fake_cuda, tmp_path):
    mark = tmp_path / "mark"
    replays = Replays(lambda value: value, torch.device("cpu"), mark)
    replays(torch.ones(2))
    FakeGraph.fail_at_end = True
    with pytest.raises(RuntimeError, match="Restart chaiNNer-C"):
        replays(torch.ones(2))
    assert mark.exists()
    # The next launch does not try.
    later = Replays(lambda value: value + 1, torch.device("cpu"), mark)
    for _ in range(3):
        assert torch.equal(later(torch.ones(2)), torch.full((2,), 2.0))
    assert len(FakeGraph.instances) == 1


def test_a_capture_out_of_memory_leaves_the_shape_plain_and_no_mark(
    fake_cuda, tmp_path
):
    def function(value):
        calls.append(value)
        if len(calls) == 2:
            raise torch.OutOfMemoryError("no room")
        return value + 1

    calls: list[torch.Tensor] = []
    replays = Replays(function, torch.device("cpu"), tmp_path / "mark")
    for _ in range(4):
        assert torch.equal(replays(torch.ones(2)), torch.full((2,), 2.0))
    # Plain, failed capture, then plain for good; another shape is still captured.
    assert len(calls) == 5 and not replays.captured
    replays(torch.ones(3))
    replays(torch.ones(3))
    assert list(replays.captured) == [(torch.Size([3]),)]
    assert not (tmp_path / "mark").exists()


def test_rocm_runs_plainly(fake_cuda, monkeypatch, tmp_path):
    monkeypatch.setattr(torch.version, "hip", "7.0")
    calls = []
    replays = Replays(
        lambda value: calls.append(value) or value, torch.device("cpu"), tmp_path / "m"
    )
    for _ in range(3):
        replays(torch.ones(2))
    assert len(calls) == 3 and not FakeGraph.instances


def residents_with(monkeypatch, free: list[float], alone: bool = False):
    class Model(nn.Module):
        def __init__(self, size: int):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(size))
            self.moved = False

        def to(self, *args, **kwargs):
            self.moved = True
            return self

    offloaded = []
    bundle = SimpleNamespace(
        vae=Model(1), dino=Model(2), dit=Model(1000), offload_model=offloaded.append
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(vosr2_context, "available_memory", lambda device: free[0])
    residents = Residents(
        cast(VosrBundle, bundle), torch.device("cpu"), Path(), alone=alone
    )
    return residents, bundle, offloaded


def test_models_with_room_stay(monkeypatch):
    residents, bundle, offloaded = residents_with(monkeypatch, [1e9])
    residents.place(bundle.dit, 1000)
    assert bundle.dit.moved and not offloaded and not residents.tight


def test_other_models_on_the_device_leave_when_a_model_lacks_room(monkeypatch):
    residents, bundle, offloaded = residents_with(monkeypatch, [100.0])
    # Everything on the CPU: nothing can leave, and the allocator decides.
    residents.place(bundle.dit, 1000)
    assert bundle.dit.moved and not offloaded and not residents.tight
    bundle.vae.weight = nn.Parameter(torch.zeros(1, device="meta"))
    residents.place(bundle.dit, 1000)
    assert offloaded == [bundle.vae] and residents.tight


def test_in_low_memory_a_model_takes_the_device_alone(monkeypatch):
    residents, bundle, offloaded = residents_with(monkeypatch, [1e9], alone=True)
    bundle.vae.weight = nn.Parameter(torch.zeros(1, device="meta"))
    residents.place(bundle.dino, 1000)
    assert offloaded == [bundle.vae] and bundle.dino.moved


def test_window_work_is_what_the_restoration_would_do_inline():
    torch.manual_seed(3)
    image = torch.rand(1, 3, 544, 720) * 2 - 1
    work = window_work(image, 9)
    layout, anchor, strengths = scene_routing(image)
    assert (work.layout, work.anchor, work.strengths) == (layout, anchor, strengths)
    noise = torch.randn(
        (1, 16, 68, 90), generator=torch.Generator(device="cpu").manual_seed(9)
    )
    assert torch.equal(work.noise, noise)
    assert len(work.conditioning) == len(layout.windows)
    y, x = layout.windows[-1]
    expected = torch.nn.functional.interpolate(
        image[:, :, y * 8 : y * 8 + 512, x * 8 : x * 8 + 512] * 0.5 + 0.5,
        size=(448, 448),
        mode="bicubic",
        align_corners=False,
    ).clamp_(0, 1)
    assert torch.equal(work.conditioning[-1], expected)


def test_blocks_that_fit_stay_and_the_others_take_their_cpu_masters(monkeypatch):
    dit = small_dit()
    size = sum(
        tensor.numel() * tensor.element_size() for tensor in dit.blocks[0].parameters()
    )
    released = []
    bundle = SimpleNamespace(dit=dit, release_block=released.append)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    # Room for one kept block beside two streamed ones and the scratch
    monkeypatch.setattr(
        vosr2_context, "available_memory", lambda device: 1000 + size * 3.5
    )
    residents = Residents(cast(VosrBundle, bundle), torch.device("cpu"), Path())
    release = residents.keep_blocks(1000)
    release(dit.blocks[0])
    assert not released
    release(dit.blocks[1])
    assert released == [dit.blocks[1]]


def test_a_released_block_takes_its_master_without_a_copy():
    dit = small_dit()
    blocks = {
        block: {
            name.removeprefix(f"blocks.{index}."): tensor.clone()
            for name, tensor in dit.state_dict().items()
            if name.startswith(f"blocks.{index}.")
        }
        for index, block in enumerate(dit.blocks)
    }
    bundle = SimpleNamespace(_block_states=blocks)
    VosrBundle.release_block(cast(VosrBundle, bundle), dit.blocks[1])
    for name, tensor in dit.blocks[1].state_dict().items():
        assert tensor.data_ptr() == blocks[dit.blocks[1]][name].data_ptr()
    for name, tensor in dit.blocks[0].state_dict().items():
        assert tensor.data_ptr() != blocks[dit.blocks[0]][name].data_ptr()


@pytest.mark.parametrize("rows", [1, 2, 7, 8, 9, 100])
def test_rows_on_threads_join_in_order(rows):
    whole = np.arange(rows * 3).reshape(rows, 3)
    assert np.array_equal(by_rows(rows, lambda top, bottom: whole[top:bottom]), whole)


@pytest.mark.parametrize("scale", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("size", [(1, 1), (2, 5), (7, 3), (33, 57), (250, 64)])
@pytest.mark.parametrize("kind", ["noise", "smooth"])
def test_the_prepared_image_has_the_bytes_of_one_whole_resize(scale, size, kind):
    height, width = size
    rng = np.random.default_rng(height * width)
    if kind == "noise":
        image = rng.random((height, width, 3), dtype=np.float32)
    else:
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        image = np.stack((y / height, x / width, (x + y) / (height + width)), axis=2)
    source = Image.fromarray(np.rint(image[:, :, ::-1] * 255).astype(np.uint8))
    whole = np.asarray(
        source.resize((width * scale, height * scale), Image.Resampling.BICUBIC)
    )
    _, rgb = prepare_image(image, scale)
    assert np.array_equal(rgb, whole.astype(np.float32) / 255)


def test_plans_in_frames_and_in_one_frame_read_back_whole(monkeypatch, tmp_path):
    monkeypatch.setattr(vosr2_plan_file, "FRAME", 1000)
    plan = np.random.default_rng(1).integers(0, 9, 25_500, np.uint8).tobytes()
    source = tmp_path / "stage.plan"
    source.write_bytes(plan)
    target = tmp_path / "stage.plan.zst"
    vosr2_plan_file.compress(source, target)
    data = target.read_bytes()
    assert zstd.get_frame_size(data) < len(data)
    with vosr2_plan_file.open_plan(target) as reader:
        assert reader.read() == plan
    target.write_bytes(zstd.compress(plan, level=vosr2_plan_file.LEVEL))
    with vosr2_plan_file.open_plan(target) as reader:
        assert reader.read() == plan
    with vosr2_plan_file.open_plan(source) as reader:
        assert reader.read() == plan


def test_the_warm_up_runs_once_per_gpu_on_a_thread(monkeypatch):
    jobs = []
    monkeypatch.setattr(
        vosr2_warm_up.background_work, "start", lambda name, job: jobs.append(job)
    )
    monkeypatch.setattr(vosr2_warm_up, "_readied", set())
    vosr2_warm_up.ready_small_steps(torch.device("cpu"), True)
    assert not jobs
    for _ in range(2):
        vosr2_warm_up.ready_small_steps(torch.device("cuda:0"), True)
    assert len(jobs) == 1
    vosr2_warm_up.ready_small_steps(torch.device("cuda:1"), False)
    assert len(jobs) == 2


@pytest.mark.parametrize("session", [False, True])
def test_the_warm_up_runs_the_steps_it_readies(monkeypatch, session):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    vosr2_warm_up._ready(torch.device("cpu"), session)
