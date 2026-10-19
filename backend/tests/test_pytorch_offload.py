"""The VRAM offload moves activations to RAM and back without changing a bit of the output, and
leaves the process's PyTorch memory cap as it was (CUDA only)."""

from __future__ import annotations

import pytest
import torch

from nodes.impl.pytorch import offload

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEVICE = torch.device("cuda", 0)


class Skip(torch.nn.Module):
    """Its first activation stays idle while the next ones run, as a long skip does."""

    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Conv2d(3, 64, 3, padding=1)
        self.body = torch.nn.Sequential(
            *[torch.nn.Conv2d(64, 64, 3, padding=1) for _ in range(4)]
        )
        self.last = torch.nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = torch.relu(self.first(x))
        body = self.body(first)
        view = body[:, :32]  # shares the body's memory: neither may move
        return self.last(body + first) + view.mean()


def model_and_input() -> tuple[torch.nn.Module, torch.Tensor]:
    torch.manual_seed(0)
    model = Skip().to(DEVICE).eval()
    x = torch.rand(1, 3, 512, 512, device=DEVICE)  # 64 MiB per activation
    return model, x


@torch.inference_mode()
def test_moved_activations_give_the_same_output():
    model, x = model_and_input()
    expected = model(x)
    # No room at all: every activation that may move goes to RAM before each op.
    mode = offload.OffloadMode(DEVICE, cap=0)
    with mode:
        out = model(x)
    torch.cuda.synchronize()
    assert mode.out_bytes > 0 and mode.back_bytes > 0
    assert torch.equal(out, expected)
    assert all(p.device == DEVICE for p in model.parameters())


@torch.inference_mode()
def test_the_cap_is_restored(monkeypatch: pytest.MonkeyPatch):
    model, x = model_and_input()
    monkeypatch.setattr(offload, "MIN_PIXELS", 0)
    before = torch.cuda.get_per_process_memory_fraction(DEVICE)
    expected = model(x)
    with offload.offloaded(model, x):
        out = model(x)
    assert torch.cuda.get_per_process_memory_fraction(DEVICE) == before
    assert torch.equal(out, expected)
