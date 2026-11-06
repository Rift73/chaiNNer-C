"""Models that keep following the input's size in an ONNX export.

A trace stores as a constant any size read through `x.data`, passed through `int` or
numpy, or given to `torch.Tensor`: the size of the export's example. Each class here
computes the same values with sizes the trace records as the graph's own: without it
DITN and HVICIDNet fail at every other size, and FBCNN at sizes that are not multiples
of 8. The loaded model is not changed: the rewrite works on a copy, so Save Model keeps
it as it is. Uformer reads its square side the same way in eight classes, which fail
above its padded example of 128 px: it gets the same side from functions that keep the
trace (uformer_sides).

SeemoRe's blocks pick their experts from the image: a router's largest weights, whose
positions a loop reads. A trace keeps the example's pick for every image, and SeemoRe's
tiny x2 ends 36.8 to 47.8 dB from PyTorch on natural crops. The export runs every expert
and zeroes those the router leaves out, as the model does in training.

RGT's recursive attention reduces its keys twice, and a third time at a side of 1024 px or
more, a count it takes from `math.log`, which a trace keeps: a trained 2x RGT ended 50.8 dB
from PyTorch at 1024x64 with no error. The export reduces three times at every size and picks
by the input's size. A fourth time, at a side of 4096 px, it does not do."""

from __future__ import annotations

import copy
import math
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from math import pi
from threading import Lock
from types import SimpleNamespace
from typing import TypeVar

import torch
from spandrel import ModelDescriptor
from spandrel.architectures.DITN.__arch.DITN_Real import ISA, UFONE
from spandrel.architectures.FBCNN.__arch.FBCNN import FBCNN
from spandrel.architectures.HVICIDNet.__arch.HVI_transform import RGB_HVI
from spandrel.architectures.RGT.__arch.rgt import RG_SA
from spandrel.architectures.SeemoRe.__arch.seemore_arch import MoELayer
from spandrel.architectures.Uformer import Uformer
from torch import nn
from torch.nn import functional
from typing_extensions import override

_Descriptor = TypeVar("_Descriptor", bound=ModelDescriptor)


class _TracedSizeIsa(ISA):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.view(b, c, -1).transpose(-1, -2)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = functional.normalize(q.transpose(-1, -2), dim=-1)
        k = functional.normalize(k.transpose(-1, -2), dim=-1)
        attention = (q @ k.transpose(-2, -1)) * self.temperature
        out = attention.softmax(dim=-1) @ v.transpose(-1, -2)
        return self.project_out(out.view(b, c, h, w))


class _TracedSizeUfone(UFONE):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        size = self.patch_size
        rows, columns = h // size, w // size
        patches = (
            x.view(b, c, rows, size, columns, size)
            .permute(0, 2, 4, 1, 3, 5)
            .contiguous()
            .view(-1, c, size, size)
        )
        patches = self.ITLs(patches).view(b, rows, columns, c, size, size)
        features = patches.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, c, h, w)
        return self.SALs(features)


class _TracedSizeHvi(RGB_HVI):
    @override
    def HVIT(self, img: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        red, green, blue = img[:, 0], img[:, 1], img[:, 2]
        value = img.max(1)[0]
        low = img.min(1)[0]
        span = value - low + eps
        # Each case over the one before, as the masked assignments it replaces
        hue = 4.0 + (red - green) / span
        hue = torch.where(green == value, 2.0 + (blue - red) / span, hue)
        hue = torch.where(red == value, (0.0 + (green - blue) / span) % 6, hue)
        hue = torch.where(low == value, torch.zeros_like(hue), hue) / 6.0
        saturation = (value - low) / (value + eps)
        saturation = torch.where(value == 0, torch.zeros_like(saturation), saturation)
        hue, saturation, value = hue[:, None], saturation[:, None], value[:, None]
        k = self.density_k
        self.this_k = k.item()
        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)
        x = color_sensitive * saturation * (2.0 * pi * hue).cos()
        y = color_sensitive * saturation * (2.0 * pi * hue).sin()
        return torch.cat([x, y, value], dim=1)


class _TracedSizeFbcnn(FBCNN):
    def forward(
        self, x: torch.Tensor, qf_input: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # FBCNN's own padding, which finds nothing left to pad
        h, w = x.shape[-2:]
        padded = functional.pad(
            x, (0, (8 - w % 8) % 8, 0, (8 - h % 8) % 8), "replicate"
        )
        out, qf = super().forward(padded, qf_input)
        return out[..., :h, :w], qf


class _TracedPickMoeLayer(MoELayer):
    def forward(self, inputs: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        weights = functional.softmax(self.gate(inputs), dim=1, dtype=torch.float)
        weights = weights.to(inputs.dtype)
        # By position, not by value: FP16 weights often tie, and topk picks just one of them
        chosen = weights.topk(self.num_expert).indices[:, :, None]
        positions = torch.arange(len(self.experts), device=weights.device)
        picked = weights * (chosen == positions).to(weights.dtype).sum(1)
        out = inputs
        for i, expert in enumerate(self.experts):
            out = out + expert(inputs, k) * picked[:, i : i + 1, None, None]
        return out


class _TracedReductionsRgSa(RG_SA):
    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        two = x.permute(0, 2, 1).reshape(b, c, h, w).contiguous()
        two = self.reduction1(self.reduction1(two))
        h2, w2 = two.shape[-2:]
        # The third's kernel of 4 has to fit whether it is picked or not
        padding = (0, (w2 < 4) * (4 - w2), 0, (h2 < 4) * (4 - h2))
        three = self.reduction1(functional.pad(two, padding))
        h3, w3 = three.shape[-2:]
        third = (h >= 1024) | (w >= 1024)
        rows, columns = h2 + third * (h3 - h2), w2 + third * (w3 - w2)
        start = third * h2 * w2
        reduced = torch.cat((two.flatten(2), three.flatten(2)), dim=2)
        reduced = reduced[:, :, start : start + rows * columns]
        reduced = self.conv(self.dwconv(reduced.view(b, c, rows, columns)))
        reduced = self.norm_act(reduced.reshape(b, self.cr, -1).permute(0, 2, 1))
        heads, q_width = self.num_heads, int(self.cr / self.num_heads)
        v_width = int(c / heads)
        q = self.q(x).reshape(b, n, heads, q_width).permute(0, 2, 1, 3)
        k = self.k(reduced).reshape(b, -1, heads, q_width).permute(0, 2, 1, 3)
        v = self.v(reduced).reshape(b, -1, heads, v_width).permute(0, 2, 1, 3)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        attn = self.attn_drop(attn)
        image = v.transpose(1, 2).reshape(b, -1, c).transpose(1, 2).contiguous()
        cpe = self.cpe(image.view(b, c, rows, columns)).view(b, c, -1)
        v = v + cpe.view(b, heads, v_width, -1).transpose(-1, -2)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


_REWRITES: dict[type[nn.Module], type[nn.Module]] = {
    FBCNN: _TracedSizeFbcnn,
    ISA: _TracedSizeIsa,
    MoELayer: _TracedPickMoeLayer,
    RG_SA: _TracedReductionsRgSa,
    RGB_HVI: _TracedSizeHvi,
    UFONE: _TracedSizeUfone,
}


_uformer_sides_lock = Lock()


def _traced_int(value: float | torch.Tensor) -> int | torch.Tensor:
    return value.to(torch.int64) if isinstance(value, torch.Tensor) else int(value)


def _traced_sqrt(value: float | torch.Tensor) -> float | torch.Tensor:
    if isinstance(value, torch.Tensor):
        return torch.sqrt(value.to(torch.float64))
    return math.sqrt(value)


@contextmanager
def uformer_sides(model: ModelDescriptor) -> Iterator[None]:
    """While it lasts, a Uformer's `int` and `math.sqrt` keep a trace's sizes; other
    values, as another thread's run of a Uformer gives them, are what they were. Another
    export of a Uformer waits for it to end."""
    if not isinstance(model.model, Uformer):
        yield
        return
    names = vars(sys.modules[Uformer.__module__])
    with _uformer_sides_lock:
        names["int"] = _traced_int
        names["math"] = SimpleNamespace(sqrt=_traced_sqrt)
        try:
            yield
        finally:
            del names["int"]
            names["math"] = math


def for_export(model: _Descriptor) -> _Descriptor:
    """A copy of a model with such a class, with sizes following the input's; any other
    model as it is."""
    if not any(type(m) in _REWRITES for m in model.model.modules()):
        return model
    model = copy.deepcopy(model)
    for module in model.model.modules():
        rewrite = _REWRITES.get(type(module))
        if rewrite is not None:
            module.__class__ = rewrite
    return model
