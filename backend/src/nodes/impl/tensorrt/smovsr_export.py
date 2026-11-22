"""SMoVSR's TensorRT ONNX parts for one input size, exported in a child process
(dual.run_child) from a folded model's weights, one frame per call, from spandrel's
per-frame steps (the ones its run_window chains, so both routes compute one graph):

- encfwd: x + state_past, valid_past -> feat, fwd_out, state (the frame's encoding and
  the forward sweep's step: SMoVSR.encode_forward)
- bwd:    feat, fwd_out + state_future, valid_future -> bwd_out, state (the backward
  sweep's step: SMoVSR.backward_step)
- head:   x, fwd_out, bwd_out -> output (the upsampler plus the bilinear upscale of the
  frame, clamped to 0..1: SMoVSR.head)

The first step of a sweep is the same graph: a zero state with its validity flag (BOOL
[1]) false selects the model's cold step. The parts compute in FP16 (the model's weights
in half precision) with the image `x` and `output` in FP32 and the head's Resize (the
bilinear upscale of the frame) in FP32 (rsr_export.fp32_resize). Every shape is static.
The binding names are rsr_export's (IMAGE_BINDINGS, VALIDITY_BINDINGS), so RSR's worker
(rsr_engine_worker.py) builds the parts with every 32-channel handoff channel-last.

    python -m nodes.impl.tensorrt.smovsr_export WEIGHTS OUTPUT HEIGHT WIDTH [DEVICE]

DEVICE (a torch device, default cpu) is where the parts are traced; the graphs do not
depend on it. WEIGHTS is a safetensors file of the folded model's state dict whose
metadata holds the model's hyperparameters (rdual.HYPERPARAMETERS_KEY, JSON); each part
goes to OUTPUT/<part> (model.onnx and weights.bin), the manifest to
OUTPUT/export.json."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import onnx
import torch
from onnx import helper
from spandrel.architectures.SMoVSR import (
    SMoVSR,
    backward_step,
    encode_forward,
    head,
)
from torch import nn

from .rdual import HYPERPARAMETERS_KEY
from .rdual_export import save
from .rsr_export import OPSET, fp32_resize

PARTS = ("encfwd", "bwd", "head")
# The Resizes each part runs in FP32: the head's bilinear upscale of the frame only.
RESIZES = {"encfwd": 0, "bwd": 0, "head": 1}


def dtype(model: SMoVSR) -> torch.dtype:
    """The model's weights' type, which the parts compute in (FP16 in a set; an FP32
    model traces the same graphs in FP32)."""
    return model.backward_cat.weight.dtype


class EncodeForward(nn.Module):
    def __init__(self, model: SMoVSR) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, x: torch.Tensor, state_past: torch.Tensor, valid_past: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return encode_forward(
            self.model, x.to(dtype(self.model)), state_past, valid_past
        )


class Backward(nn.Module):
    def __init__(self, model: SMoVSR) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        feat: torch.Tensor,
        fwd_out: torch.Tensor,
        state_future: torch.Tensor,
        valid_future: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return backward_step(self.model, feat, fwd_out, state_future, valid_future)


class Head(nn.Module):
    def __init__(self, model: SMoVSR) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, x: torch.Tensor, fwd_out: torch.Tensor, bwd_out: torch.Tensor
    ) -> torch.Tensor:
        return head(self.model, x.to(dtype(self.model)), fwd_out, bwd_out).float()


def load(weights: Path) -> SMoVSR:
    """The folded model of a WEIGHTS file, on the CPU in FP32: built in its training form,
    loaded, then folded by eval() (which merges every rep convolution from the training
    form it was given, as spandrel's loader does)."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(weights), "pt") as file:
        metadata = file.metadata()
    hyperparameters = json.loads(metadata[HYPERPARAMETERS_KEY])
    model = SMoVSR(**hyperparameters).float()
    model.load_state_dict(load_file(str(weights)), strict=True)
    model.eval()
    model.requires_grad_(False)
    return model


def trace(
    module: nn.Module, inputs: dict[str, torch.Tensor], outputs: tuple[str, ...]
) -> onnx.ModelProto:
    """`module` traced on `inputs` (name -> example, in argument order) at their static
    shapes, with its outputs named `outputs`."""
    with tempfile.TemporaryDirectory(prefix="chaiNNer-smovsr-part-") as temporary:
        path = Path(temporary) / "part.onnx"
        torch.onnx.export(
            module,
            tuple(inputs.values()),
            str(path),
            opset_version=OPSET,
            input_names=list(inputs),
            output_names=list(outputs),
            dynamo=False,
            do_constant_folding=True,
            external_data=False,
        )
        return onnx.load(str(path))


def part(
    module: nn.Module,
    inputs: dict[str, torch.Tensor],
    outputs: tuple[str, ...],
    resizes: int,
) -> onnx.ModelProto:
    """`module` traced (`trace`) with its `resizes` Resizes in FP32."""
    model = trace(module, inputs, outputs)
    # Another count means the exporter's graph is not the one the pass expects (the
    # head's upsampling under another operator would stay in FP16).
    found = fp32_resize(model)
    if found != resizes:
        raise RuntimeError(
            f"SMoVSR's export found {found} Resizes to run in FP32, not {resizes}"
        )
    onnx.checker.check_model(model)
    return model


def parts(model: SMoVSR, height: int, width: int, device: str) -> dict[str, Any]:
    """The three parts of the folded FP32 `model` for frames of height x width, traced
    in FP16 on `device`."""
    half = model.to(device).half()
    generator = torch.Generator().manual_seed(0)

    def example(*shape: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        return torch.rand(shape, generator=generator).to(device=device, dtype=dtype)

    x = example(1, 3, height, width, dtype=torch.float32)
    # The model's width: every handoff and state has its channels
    state = example(1, half.backward_cat.out_channels, height, width)
    valid = torch.ones(1, dtype=torch.bool, device=device)
    with torch.no_grad():
        return {
            "encfwd": part(
                EncodeForward(half),
                {"x": x, "state_past": state, "valid_past": valid},
                ("feat", "fwd_out", "state"),
                RESIZES["encfwd"],
            ),
            "bwd": part(
                Backward(half),
                {
                    "feat": state,
                    "fwd_out": state,
                    "state_future": state,
                    "valid_future": valid,
                },
                ("bwd_out", "state"),
                RESIZES["bwd"],
            ),
            "head": part(
                Head(half),
                {"x": x, "fwd_out": state, "bwd_out": state},
                ("output",),
                RESIZES["head"],
            ),
        }


def main(
    weights: str, output: str, height: str, width: str, device: str = "cpu"
) -> None:
    import spandrel

    model = load(Path(weights))
    exported = parts(model, int(height), int(width), device)
    root = Path(output)
    manifest: dict[str, Any] = {
        "height": int(height),
        "width": int(width),
        "scale": model.scale,
        # The bundle's architecture (rdual_bundle.architecture)
        "architecture": "SMoVSR",
        # The library that traced the parts, for the log of a failed build
        "spandrel": spandrel.__version__,
        "parts": {},
    }
    for name in PARTS:
        graph = exported[name]
        save(graph, root / name)
        manifest["parts"][name] = {
            "nodes": len(graph.graph.node),
            "inputs": [
                (i.name, helper.printable_type(i.type)) for i in graph.graph.input
            ],
            "outputs": [
                (o.name, helper.printable_type(o.type)) for o in graph.graph.output
            ],
        }
    (root / "export.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
