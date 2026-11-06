"""The LDA upsampler at every input size in an ONNX export.

LDA_AQU takes its output size through `int(H * self.scale_factor)` and reshapes through
einops, and a trace stores both as constants: the size of the export's example. Every
other size then gives an output of the example's size, or an error. The export computes
the same values with sizes the trace records as the graph's own. The loaded model is
not changed: the rewrite works on a copy, so Save Model keeps it as it is."""

from __future__ import annotations

import copy
from typing import TypeVar

import torch
from spandrel import ModelDescriptor
from spandrel.architectures.FDAT.__arch import fdat
from spandrel.architectures.GaterV3.__arch import gaterv3
from spandrel.architectures.GFISRv2.__arch import gfisrv2
from torch import nn
from torch.nn import functional
from torch.onnx.operators import shape_as_tensor

_Descriptor = TypeVar("_Descriptor", bound=ModelDescriptor)

# Spandrel's copies of LDA_AQU, which differ only in comments and an unused method
_LDAS = (fdat.LDA_AQU, gaterv3.LDA_AQU, gfisrv2.LDA_AQU)


class _TracedSizeLda(gfisrv2.LDA_AQU):
    def __init__(self, source: nn.Module) -> None:
        nn.Module.__init__(self)
        self.__dict__.update(source.__dict__)

    def _grid(self, offset: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = offset.shape
        k_u = self.k_u
        rows, columns = torch.meshgrid(
            torch.arange(height), torch.arange(width), indexing="ij"
        )
        position = torch.stack((rows, columns), -1).to(offset)
        grid = offset.view(batch, k_u, k_u, 2, height, width).permute(0, 1, 4, 2, 5, 3)
        grid = grid + position.view(1, 1, height, 1, width, 2)
        grid = grid.contiguous().view(batch, k_u * height, k_u * width, 2)
        # The height and width less one, which a trace records as the graph's own
        last = shape_as_tensor(offset)[[2, 3]].to(offset) - 1
        return (2 * grid / last - 1).flip(-1)

    def _sampled(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Three by three samples per output pixel: (b g) n c h w."""
        k_u = self.k_u
        out = functional.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        batch, channels, _, _ = out.shape
        height, width = grid.shape[1] // k_u, grid.shape[2] // k_u
        return (
            out.view(batch, channels, k_u, height, k_u, width)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch, k_u * k_u, channels, height, width)
        )

    def _per_head(self, x: torch.Tensor, batch: int) -> torch.Tensor:
        """(b g) n c h w, as b heads (h w) n c."""
        _, n, c, height, width = x.shape
        return (
            x.view(batch, self.n_groups, n, c, height, width)
            .permute(0, 4, 5, 2, 1, 3)
            .reshape(batch, height * width, n, self.num_head, -1)
            .permute(0, 3, 1, 2, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        factor = int(self.scale_factor)
        out_height, out_width = height * factor, width * factor
        groups = self.n_groups
        v = x
        x = self.layer_norm(x)
        q = self.proj_q(x)
        k = self.proj_k(x)
        q = functional.interpolate(
            q, (out_height, out_width), mode="bilinear", align_corners=True
        )
        q_offset = q.view(batch * groups, -1, out_height, out_width)
        base_offset = self.get_buffer("base_offset")
        offset = self.conv_offset(q_offset).tanh().mul(self.offset_range_factor)
        offset = offset + base_offset.to(x.dtype)
        k = k.view(batch * groups, self.hidden_dim // groups, height, width)
        v = v.view(batch * groups, channels // groups, height, width)
        grid = self._grid(offset)
        k = self._per_head(self._sampled(k, grid), batch)
        v = self._per_head(self._sampled(v, grid), batch)
        q = (
            q.view(batch, self.num_head, -1, out_height, out_width)
            .permute(0, 1, 3, 4, 2)
            .reshape(batch, self.num_head, out_height * out_width, 1, -1)
        )
        if self.rpb:
            k = k + self.relative_position_bias_table
        attention = (q * self.scale @ k.transpose(-1, -2)).softmax(dim=-1)
        return (
            (attention @ v)
            .view(batch, self.num_head, out_height, out_width, -1)
            .permute(0, 1, 4, 2, 3)
            .reshape(batch, -1, out_height, out_width)
        )


def for_export(model: _Descriptor) -> _Descriptor:
    """A copy of a model with an LDA upsampler with its sizes following the input's; any
    other model as it is."""
    if not any(isinstance(m, _LDAS) for m in model.model.modules()):
        return model
    model = copy.deepcopy(model)
    net = model.model
    for name, module in list(net.named_modules()):
        if isinstance(module, _LDAS):
            net.set_submodule(name, _TracedSizeLda(module))
    return model
