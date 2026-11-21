"""TFDAT's and TFDATv2's TensorRT ONNX for one frame size, exported in a child process
(dual.run_child) from the model's weights: one part, "clip", that maps a clip of
`clip_size` frames stacked on the channels, `x` (1, clip_size * 3, H, W), to the
upscaled centre frame `output` (1, 3, H * scale, W * scale). The model has no state
between clips, so the part is the whole model; every shape is static.

The part computes in FP16 (the model's weights in half precision) with `x` and `output`
in FP32, and its warp in FP32 as the model's own (spandrel's `warp_frame`). Its channel
attention is exported as `SplitChannelAttention`, the same maths in another order: in
TensorRT's FP16 the attention's q @ k^T, a sum over every token of the frame (K = 86400
at 720x480), lost its precision (the trained 2x TFDATv2 on a DVD clip at 720x480: 0.66
of 255 levels mean and 22 largest from PyTorch FP32, against 0.073 and 2.35 for PyTorch
FP16), and in FP32 that product took 2.4 ms where FP16 takes 0.26 (each of 12).

CHECKED: the trained 2x TFDATv2 (clip 9) at 720x480 on 9 frames of an MPEG-2 DVD clip
with motion: 0.153 of 255 levels mean and 4.56 largest from PyTorch FP32 (PyTorch FP16:
0.073 and 2.35), 19.3 ms a frame against PyTorch FP16's 68.8 (RTX 5090).

    python -m nodes.impl.tensorrt.tfdat_export ARCHITECTURE WEIGHTS OUTPUT HEIGHT WIDTH
        [DEVICE]

ARCHITECTURE is spandrel's id ("TFDAT" or "TFDATv2"); DEVICE (a torch device, default
cpu) is where the part is traced; the graph does not depend on it. WEIGHTS is a
safetensors file of the model's state dict whose metadata holds its hyperparameters
(rdual.HYPERPARAMETERS_KEY, JSON); the part goes to OUTPUT/clip (model.onnx and
weights.bin), the manifest to OUTPUT/export.json."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import onnx
import torch
from onnx import helper
from spandrel.architectures.FDAT.__arch.fdat import FastChannelAttention
from spandrel.architectures.TFDAT import TFDAT, TFDATv2
from torch import nn
from torch.nn import functional

from .rdual import HYPERPARAMETERS_KEY
from .rdual_export import save

PART = "clip"
OPSET = 20
CLASSES: dict[str, type[TFDAT | TFDATv2]] = {"TFDAT": TFDAT, "TFDATv2": TFDATv2}
# The partial products of the channel attention's q @ k^T: the tokens of a frame padded
# to the model's alignment are a multiple of 64 at every scale (8 x 8 features).
SPLITS = 64


class SplitChannelAttention(nn.Module):
    """FastChannelAttention's maths for TensorRT's FP16: q and k normalized over the
    tokens in FP32 (back in FP16), q @ k^T as SPLITS FP16 products of a share of the
    tokens each, summed in FP32, and the temperature and softmax in FP32."""

    def __init__(self, attention: FastChannelAttention) -> None:
        super().__init__()
        self.nh = attention.nh
        self.stable_norm = attention.stable_norm
        self.temp = attention.temp
        self.qkv, self.proj = attention.qkv, attention.proj

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # The frame's size is FastChannelAttention's argument too, which reads the tokens.
        del height, width
        batch, tokens, channels = x.shape
        d = channels // self.nh
        qkv = self.qkv(x).view(batch, tokens, 3, self.nh, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.stable_norm:
            inv_n = tokens**-0.5
            q = functional.normalize(
                q.transpose(-2, -1).float() * inv_n, dim=-1, eps=1e-4
            )
            k = functional.normalize(
                k.transpose(-2, -1).float() * inv_n, dim=-1, eps=1e-4
            )
        else:
            q = functional.normalize(q.transpose(-2, -1).float(), dim=-1)
            k = functional.normalize(k.transpose(-2, -1).float(), dim=-1)
        shape = (batch, self.nh, d, SPLITS, tokens // SPLITS)
        q = q.to(x.dtype).reshape(shape).transpose(2, 3)
        k = k.to(x.dtype).reshape(shape).transpose(2, 3)
        dots = (q @ k.transpose(-2, -1)).float().sum(2)
        attn = functional.softmax(dots * self.temp.float(), dim=-1).to(x.dtype)
        return self.proj(
            (attn @ v.transpose(-2, -1))
            .permute(0, 3, 1, 2)
            .reshape(batch, tokens, channels)
        )


class Clip(nn.Module):
    """The stacked clip in FP32 -> the centre frame's output in FP32, through the model
    in half precision."""

    def __init__(self, model: TFDAT | TFDATv2) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        clip = self.model.clip_size
        _, channels, height, width = x.shape
        frames = x.half().view(1, clip, channels // clip, height, width)
        return self.model(frames).float()


def load(architecture: str, weights: Path) -> TFDAT | TFDATv2:
    """The model of a WEIGHTS file, on the CPU in FP32."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(weights), "pt") as file:
        metadata = file.metadata()
    model = CLASSES[architecture](**json.loads(metadata[HYPERPARAMETERS_KEY]))
    model.load_state_dict(load_file(str(weights)), strict=True)
    return model.float().eval().requires_grad_(False)


def for_export(model: TFDAT | TFDATv2) -> TFDAT | TFDATv2:
    """`model` with every channel attention swapped for SplitChannelAttention (in
    place: the export's model is its own)."""
    for module in list(model.modules()):
        for name, child in module.named_children():
            if isinstance(child, FastChannelAttention):
                setattr(module, name, SplitChannelAttention(child))
    return model


def part(
    model: TFDAT | TFDATv2, height: int, width: int, device: str
) -> onnx.ModelProto:
    """The part of the FP32 `model` for frames of height x width, traced in FP16 on
    `device`."""
    half = for_export(model).to(device).half()
    generator = torch.Generator().manual_seed(0)
    x = torch.rand((1, half.clip_size * 3, height, width), generator=generator)
    with tempfile.TemporaryDirectory(prefix="chaiNNer-tfdat-part-") as temporary:
        path = Path(temporary) / "part.onnx"
        with torch.no_grad():
            torch.onnx.export(
                Clip(half),
                (x.to(device),),
                str(path),
                opset_version=OPSET,
                input_names=["x"],
                output_names=["output"],
                dynamo=False,
                do_constant_folding=True,
                external_data=False,
            )
        graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    return graph


def main(
    architecture: str,
    weights: str,
    output: str,
    height: str,
    width: str,
    device: str = "cpu",
) -> None:
    import spandrel

    model = load(architecture, Path(weights))
    graph = part(model, int(height), int(width), device)
    root = Path(output)
    save(graph, root / PART)
    manifest: dict[str, Any] = {
        "height": int(height),
        "width": int(width),
        "scale": model.upscale,
        "clip_size": model.clip_size,
        # The bundle's architecture (rdual_bundle.architecture)
        "architecture": architecture,
        # The library that traced the part, for the log of a failed build
        "spandrel": spandrel.__version__,
        "parts": {
            PART: {
                "nodes": len(graph.graph.node),
                "inputs": [
                    (i.name, helper.printable_type(i.type)) for i in graph.graph.input
                ],
                "outputs": [
                    (o.name, helper.printable_type(o.type)) for o in graph.graph.output
                ],
            }
        },
    }
    (root / "export.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
