"""Activation restore retries use CPU tensors and stand-ins for CUDA operations."""

from __future__ import annotations

import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from nodes.impl.pytorch import offload


def test_failed_restore_evicts_a_non_input_and_retries(monkeypatch: pytest.MonkeyPatch):
    stream = SimpleNamespace(wait_stream=lambda _stream: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda _device: stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(
        torch.cuda, "Event", lambda: SimpleNamespace(record=lambda _stream: None)
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 0)
    empty_like = torch.empty_like
    monkeypatch.setattr(
        torch,
        "empty_like",
        lambda tensor, *, device, pin_memory: empty_like(tensor, device=device),
    )
    # The real eviction path copies between ordinary CPU tensors here.
    monkeypatch.setattr(offload, "_pinned", False)
    steps: list[tuple[str, int]] = []
    monkeypatch.setattr(
        torch.Tensor,
        "record_stream",
        lambda tensor, _stream: steps.append(("evict", id(tensor))),
    )
    mode = offload.OffloadMode(torch.device("cpu"), cap=offload.WORKSPACE + 1)
    moved_input = torch.tensor([1.0])
    resident_input = torch.tensor([2.0])
    victim = torch.tensor([3.0])
    mode.moved[id(moved_input)] = weakref.ref(moved_input)
    for tensor in (resident_input, victim):
        mode.resident[id(tensor)] = weakref.ref(tensor)
        mode.sharers[tensor.untyped_storage().data_ptr()] = 1

    bring_back = mode._bring_back

    def restore(tensor: torch.Tensor) -> None:
        steps.append(("restore", id(tensor)))
        if len(steps) == 1:
            raise torch.OutOfMemoryError("CUDA stand-in restore failed")
        bring_back(tensor)

    monkeypatch.setattr(mode, "_bring_back", restore)

    result = mode.__torch_dispatch__(
        torch.ops.aten.add.Tensor,
        (torch.Tensor,),
        (moved_input,),
        {"other": resident_input},
    )

    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, torch.tensor([3.0]))
    assert steps == [
        ("restore", id(moved_input)),
        ("evict", id(victim)),
        ("restore", id(moved_input)),
    ]
    assert set(mode.moved) == {id(victim)}
    assert id(resident_input) in mode.resident
