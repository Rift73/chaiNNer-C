"""Rotary tables for the exported, fixed one-step VOSR DiT stages."""

from __future__ import annotations

import torch


def rotary_tables(
    width: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the checkpoint's square and separately interpolated rectangular RoPE."""
    if width not in (64, 72):
        raise ValueError("VOSR TensorRT supports 64 or 72 latent columns.")
    frequency = 1 / (10000 ** (torch.arange(0, 32, 2, device=device).float() / 32))
    y = torch.arange(32, device=device).float()
    x = torch.arange(width // 2, device=device).float() / (width // 2) * 32
    fy = (y[:, None] * frequency).repeat_interleave(2, -1)
    fx = (x[:, None] * frequency).repeat_interleave(2, -1)
    angles = torch.cat(
        (
            fy[:, None, :].expand(32, width // 2, 32),
            fx[None, :, :].expand(32, width // 2, 32),
        ),
        dim=-1,
    ).reshape(-1, 64)
    return angles.cos().to(dtype), angles.sin().to(dtype)
