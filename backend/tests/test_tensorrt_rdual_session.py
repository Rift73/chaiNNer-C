"""The RDUAL engine set's session (nodes/impl/tensorrt/rdual_session.py) on the CPU, with
stand-in engines that compute simple functions of their bindings: window bookkeeping,
the reuse of the frames two windows share, outputs that later windows leave alone, and
the memory arithmetic. No GPU, TensorRT or CUDA graphs."""

from __future__ import annotations

import contextlib
import math
from collections import Counter
from collections.abc import Hashable
from typing import Any, cast

import numpy as np
import pytest
import torch
from torch.nn import functional

from api import NodeId
from nodes.impl.tensorrt import rdual_session
from nodes.impl.tensorrt.rdual_session import RDUALSession, WindowMemory

HEIGHT, WIDTH, SCALE = 6, 10, 2  # padded to 8x12, trunk 4x6
BF16, F32 = torch.bfloat16, torch.float32
TRUNK = ((1, 4, 6, 2), BF16)
KEY = ((1, 2, 3, 2), BF16)
GUIDANCE = ((1, 5, 4, 6), F32)
VALID = ((1,), torch.bool)
SPECS: dict[str, dict[str, tuple[tuple[int, ...], torch.dtype]]] = {
    "flow": {"frames": ((2, 3, HEIGHT, WIDTH), F32), "guidance": ((2, 2, 4, 6), F32)},
    "prefix": {
        "input": ((1, 3, 8, 12), F32),
        "stem": TRUNK,
        "fine": TRUNK,
        "center": TRUNK,
        "key": KEY,
    },
    "revcore": {
        "state_future": TRUNK,
        "guidance_future": GUIDANCE,
        "valid_future": VALID,
        "fine": TRUNK,
        "center": TRUNK,
        "key": KEY,
        "state_out": TRUNK,
    },
    "fwdcore": {
        "input": ((1, 3, 8, 12), F32),
        "state_past": TRUNK,
        "guidance_past": GUIDANCE,
        "valid_past": VALID,
        "state_future": TRUNK,
        "guidance_future": GUIDANCE,
        "valid_future": VALID,
        "stem": TRUNK,
        "fine": TRUNK,
        "center": TRUNK,
        "key": KEY,
        "output": ((1, 3, 16, 24), F32),
        "state_out": TRUNK,
    },
}


def prefix(t: dict[str, torch.Tensor]) -> None:
    m = t["input"].mean()
    t["stem"].fill_(m)
    t["fine"].copy_(t["input"][0, :2, ::2, ::2].permute(1, 2, 0)[None] * 2)
    t["center"].fill_(m * 3)
    t["key"].fill_(m)


def flow(t: dict[str, torch.Tensor]) -> None:
    d = t["frames"][0].mean() - t["frames"][1].mean()
    t["guidance"][0].fill_(d * 0.1)
    t["guidance"][1].fill_(-d * 0.1)


def revcore(t: dict[str, torch.Tensor]) -> None:
    on = t["valid_future"].float()
    future = 0.5 * t["state_future"].float() + t["guidance_future"].mean()
    t["state_out"].copy_(t["fine"].float() + on * future)


def fwdcore(t: dict[str, torch.Tensor]) -> None:
    past = 0.5 * t["state_past"].float() + t["guidance_past"].mean()
    future = 0.25 * t["state_future"].float() + t["guidance_future"].mean()
    state = (
        t["center"].float()
        + t["valid_past"].float() * past
        + t["valid_future"].float() * future
    )
    t["state_out"].copy_(state)
    t["output"].copy_(
        functional.interpolate(t["input"], scale_factor=SCALE) + state.mean()
    )


OPS = {"flow": flow, "prefix": prefix, "revcore": revcore, "fwdcore": fwdcore}


class FakePart:
    """A part whose engine is OPS[name] on its bindings, counting its calls."""

    def __init__(self, name: str, calls: Counter[str]) -> None:
        self.name = name
        self.calls = calls
        self.workspace = 64
        self.graphs: dict[Hashable, Any] = {}
        self.fail = False

    def spec(self, name: str) -> tuple[tuple[int, ...], torch.dtype]:
        return SPECS[self.name][name]

    def tensor(self, name: str, device: torch.device) -> torch.Tensor:
        shape, dtype = self.spec(name)
        return torch.empty(shape, dtype=dtype, device=device)

    def use_memory(self, memory: torch.Tensor) -> None:
        assert memory.numel() >= self.workspace

    def call(self, key: Hashable, tensors: dict[str, torch.Tensor], _stream: Any):
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        assert set(tensors) == set(SPECS[self.name])
        for name, tensor in tensors.items():
            assert (tuple(tensor.shape), tensor.dtype) == self.spec(name)
        self.calls[self.name] += 1
        OPS[self.name](tensors)

    def close(self) -> None:
        self.graphs.clear()


class FakeStream:
    def wait_stream(self, _other: object) -> None:
        pass

    def synchronize(self) -> None:
        pass


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda _: contextlib.nullcontext())
    return Counter()


def session(calls: Counter[str]) -> RDUALSession:
    return RDUALSession(
        cast(Any, {name: FakePart(name, calls) for name in OPS}),
        None,
        4000,
        FakeStream(),
        torch.device("cpu"),
        (HEIGHT, WIDTH, SCALE),
        "set",
    )


@pytest.fixture(scope="module")
def clip() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return [torch.rand(3, HEIGHT, WIDTH, generator=generator) for _ in range(12)]


def zeros(spec: tuple[tuple[int, ...], torch.dtype]) -> torch.Tensor:
    return torch.zeros(spec[0], dtype=spec[1])


def reference(
    frames: list[torch.Tensor],
    keep: int,
    past: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """A window as RDUAL.infer_window runs it, with OPS for the engines, on fresh
    buffers."""
    pack = rdual_session.model_pack_guidance()

    def run(name: str, **given: torch.Tensor) -> dict[str, torch.Tensor]:
        tensors = {
            k: given[k] if k in given else zeros(SPECS[name][k]) for k in SPECS[name]
        }
        OPS[name](tensors)
        return tensors

    window = frames if past is None else [past[1], *frames]
    pairs = []
    for a, b in zip(window, window[1:], strict=False):
        raw = run("flow", frames=torch.stack([a, b]))["guidance"]
        pairs.append(pack(torch.stack([a, b])[None], raw[None, None])[0, 0])
    if past is not None:
        link, pairs = pairs[0], pairs[1:]
    n = len(frames)
    pre = [
        run("prefix", input=rdual_session.pad_frame(f[None], 8, 12).contiguous())
        for f in frames
    ]
    future: dict[int, torch.Tensor] = {}
    for t in range(n - 1, 0, -1):
        on = t + 1 < n
        future[t] = run(
            "revcore",
            state_future=future[t + 1] if on else zeros(TRUNK),
            guidance_future=pairs[t][1:2] if on else zeros(GUIDANCE),
            valid_future=torch.tensor([on]),
            fine=pre[t]["fine"],
            center=pre[t]["center"],
            key=pre[t]["key"],
        )["state_out"]
    outputs, state = [], None if past is None else past[0]
    for t in range(keep):
        guidance_past = None
        if t:
            guidance_past = pairs[t - 1][0:1]
        elif past is not None:
            guidance_past = link[0:1]
        on = t + 1 < n
        out = run(
            "fwdcore",
            input=pre[t]["input"],
            state_past=zeros(TRUNK) if state is None else state,
            guidance_past=zeros(GUIDANCE) if guidance_past is None else guidance_past,
            valid_past=torch.tensor([state is not None]),
            state_future=future[t + 1] if on else zeros(TRUNK),
            guidance_future=pairs[t][1:2] if on else zeros(GUIDANCE),
            valid_future=torch.tensor([on]),
            stem=pre[t]["stem"],
            fine=pre[t]["fine"],
            center=pre[t]["center"],
            key=pre[t]["key"],
        )
        outputs.append(out["output"][0, :, : HEIGHT * SCALE, : WIDTH * SCALE])
        state = out["state_out"]
    assert state is not None
    return outputs, state


def as_image(output: Any) -> np.ndarray:
    """A session output's image, or a reference output (RGB CHW) as the session gives
    it."""
    if isinstance(output, rdual_session.Output):
        return RDUALSession.to_image(output)
    return output.clamp(0, 1).flip(0).permute(1, 2, 0).numpy()


def same(a: list[Any], b: list[Any]) -> bool:
    return len(a) == len(b) and all(
        np.array_equal(as_image(x), as_image(y)) for x, y in zip(a, b, strict=True)
    )


def chain(run: Any, clip: list[torch.Tensor], ahead: int, look_ahead: int) -> list[Any]:
    """The windows the planner runs over the clip, each carrying the last's state."""
    outputs: list[Any] = []
    past = None
    index = 0
    while index < len(clip):
        frames = clip[index : index + ahead + 1]
        ended = index + len(frames) == len(clip)
        keep = len(frames) if ended else max(1, len(frames) - look_ahead)
        kept, state = run(frames, keep, past, index)
        assert len(kept) == keep
        outputs += kept
        past = (state, frames[keep - 1])
        index += keep
    return outputs


def test_chained_windows_equal_the_reference_and_share_their_frames_work(
    calls: Counter[str], clip: list[torch.Tensor]
):
    s = session(calls)
    mine = chain(s.run, clip, 4, 2)
    theirs = chain(lambda f, k, p, _: reference(f, k, p), clip, 4, 2)

    assert len(mine) == len(clip)
    assert all(as_image(a).shape == (HEIGHT * SCALE, WIDTH * SCALE, 3) for a in mine)
    assert same(mine, theirs)
    # Each frame's prefix and each pair's flow once; the reverse sweep per window.
    assert calls["prefix"] == len(clip)
    assert calls["flow"] == len(clip) - 1
    # windows at 0, 3, 6, 9 (the last keeps all three): 4 + 4 + 4 + 2 reverse steps
    assert calls["revcore"] == 14


def test_a_cold_window_recomputes_what_it_reads(
    calls: Counter[str], clip: list[torch.Tensor]
):
    s = session(calls)
    _, state = s.run(clip[0:5], 3, None, 0)
    s.run(clip[3:8], 3, None, 3)  # cold: nothing is reused
    assert calls["prefix"] == 10
    # A window that does not start where the last one's kept frames end
    s.run(clip[7:12], 3, (state, clip[6]), 7)
    assert calls["prefix"] == 15


def test_outputs_and_states_are_the_callers(
    calls: Counter[str], clip: list[torch.Tensor]
):
    s = session(calls)
    outputs, state = s.run(clip[0:5], 3, None, 0)
    held = [as_image(o).copy() for o in outputs]
    held_state = state.clone()
    s.run(clip[3:8], 3, (state, clip[2]), 3)
    s.run(clip[6:11], 3, (state, clip[5]), 6)

    assert all(
        np.array_equal(as_image(a), b) for a, b in zip(outputs, held, strict=True)
    )
    assert torch.equal(state, held_state)
    assert len({o.image.data_ptr() for o in outputs}) == 3


def test_a_longer_window_grows_the_ring_and_starts_over(
    calls: Counter[str], clip: list[torch.Tensor]
):
    s = session(calls)
    first, state = s.run(clip[0:3], 2, None, 0)
    assert len(s.slots) == 4
    longer, _ = s.run(clip[2:10], 6, (state, clip[1]), 2)
    assert len(s.slots) == 9

    ref_first, ref_state = reference(clip[0:3], 2, None)
    ref_longer, _ = reference(clip[2:10], 6, (ref_state, clip[1]))
    assert same(first, ref_first)
    assert same(longer, ref_longer)


def test_a_failed_window_leaves_nothing_to_reuse(
    calls: Counter[str], clip: list[torch.Tensor]
):
    s = session(calls)
    _, state = s.run(clip[0:5], 3, None, 0)
    reverse = s.parts["revcore"]
    assert isinstance(reverse, FakePart)
    reverse.fail = True
    with pytest.raises(RuntimeError, match="revcore failed"):
        s.run(clip[3:8], 3, (state, clip[2]), 3)
    reverse.fail = False
    before = calls["prefix"]

    outputs, _ = s.run(clip[3:8], 3, (state, clip[2]), 3)

    assert calls["prefix"] == before + 5
    expected, _ = reference(clip[3:8], 3, (state, clip[2]))
    assert same(outputs, expected)


def test_windows_and_frames_are_checked(calls: Counter[str], clip: list[torch.Tensor]):
    s = session(calls)
    with pytest.raises(ValueError, match="cannot keep 4"):
        s.run(clip[0:3], 4, None, 0)
    with pytest.raises(ValueError, match="takes 10x6 frames, not 12x6"):
        s.convert(np.zeros((HEIGHT, 12, 3), np.float32))
    bgr = np.random.default_rng(0).random((HEIGHT, WIDTH, 3), dtype=np.float32)
    frame = s.convert(bgr)
    assert torch.equal(frame, torch.from_numpy(bgr).permute(2, 0, 1).flip(0))
    # On the CPU an output's image is its own, with nothing to wait for.
    image = torch.from_numpy(bgr)
    assert np.array_equal(s.to_image(rdual_session.Output(image, None)), bgr)


def test_the_frames_ahead_fit_the_free_memory(
    calls: Counter[str], clip: list[torch.Tensor], monkeypatch: pytest.MonkeyPatch
):
    s = session(calls)
    per_frame = s.memory.per_frame
    free = [0]
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: (free[0], 1 << 40))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_: 0)
    allocated = [0]
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_: allocated[0])
    fraction = [1.0]
    monkeypatch.setattr(
        torch.cuda, "get_per_process_memory_fraction", lambda *_: fraction[0]
    )

    share, margin = (
        rdual_session.ALLOCATOR_SHARE,
        rdual_session.OUTSIDE_ALLOCATOR_MARGIN,
    )
    # Room for exactly six frames' slots beyond the fixed buffers and the margin: five
    # frames and the previous one, so four ahead.
    carried = s.memory.carried
    free[0] = math.ceil((6 * per_frame + carried + margin) / share)
    assert s.fitting_ahead((HEIGHT, WIDTH), 7) == 4
    assert s.fitting_ahead((HEIGHT, WIDTH), 2) == 2
    # Not even a window of one frame fits once the margin is kept.
    free[0] = math.ceil(6 * per_frame / share)
    assert s.fitting_ahead((HEIGHT, WIDTH), 7) == -1
    # A ring the session holds already counts as room.
    s.run(clip[0:5], 3, None, 0)
    free[0] = math.ceil(margin / share)
    assert s.fitting_ahead((HEIGHT, WIDTH), 7) == 4
    # PyTorch's per-process cap, which the free memory does not show: the ring's six
    # slots allocated, and room under the cap for two more
    free[0] = 1 << 40
    allocated[0] = 6 * per_frame + carried
    cap = allocated[0] + math.ceil((2 * per_frame + margin) / share)
    fraction[0] = cap / (1 << 40)
    assert s.fitting_ahead((HEIGHT, WIDTH), 7) == 6
    # Another frame size is the set's to refuse.
    assert s.fitting_ahead((HEIGHT, WIDTH + 2), 7) == 7


def test_the_memory_follows_the_bindings_and_the_window(calls: Counter[str]):
    s = session(calls)
    trunk = 4 * 6 * 2 * 2  # BF16
    slot = (
        3 * 8 * 12 * 4  # padded input
        + 3 * trunk  # stem, fine, center
        + 2 * 3 * 2 * 2  # key
        + 2 * trunk  # reverse and forward states
        + 2 * 2 * 4 * 6 * 4  # flow
        + 2 * 5 * 4 * 6 * 4  # packed guidance
        + 3 * HEIGHT * WIDTH * 4  # the converted frame
    )
    assert s.memory.per_frame == slot
    output = 3 * HEIGHT * SCALE * WIDTH * SCALE * 4
    shared = 64 + 2 * 3 * HEIGHT * WIDTH * 4 + 3 * 16 * 24 * 4 + trunk + 5 * 4 * 6 * 4
    assert s.memory.fixed == 4000 + shared + 2 * output

    assert s.memory.carried == 2 * trunk
    memory = WindowMemory(fixed=1000, per_frame=100, carried=50)
    assert memory.window_bytes(7) == 1000 + 50 + 8 * 100
    assert memory.longest_window(memory.window_bytes(7)) == 7
    assert memory.longest_window(memory.window_bytes(7) - 1) == 6
    assert memory.longest_window(1050) == 0


def test_each_node_has_its_own_session(monkeypatch: pytest.MonkeyPatch):
    opened: list[object] = []

    def open_session(_engine: Any, _gpu: int) -> Any:
        opened.append(object())
        return opened[-1]

    monkeypatch.setattr(rdual_session, "open_session", open_session)
    monkeypatch.setattr(
        rdual_session, "_sessions", rdual_session.SharedCache(lambda _value: None)
    )
    engine = cast(Any, object())
    with rdual_session.use_rdual_session(engine, 0, NodeId("a")) as a:
        pass
    with rdual_session.use_rdual_session(engine, 0, NodeId("b")) as b:
        pass
    with rdual_session.use_rdual_session(engine, 0, NodeId("a")) as again:
        pass

    assert a is not b and again is a and opened == [a, b]
