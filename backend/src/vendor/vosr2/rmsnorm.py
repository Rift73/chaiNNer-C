"""RMS normalization for VOSR, implemented from its defining equation.

Statistics are evaluated in float32; the scale parameter keeps checkpoint names.
No Llama model implementation or model-parallel runtime is included.
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = x.float()
        inverse_rms = torch.rsqrt(torch.mean(torch.square(values), dim=-1, keepdim=True) + self.eps)
        return (values * inverse_rms).to(x.dtype) * self.weight
