"""Temporal Fix split into the work of one frame and the work of one window, so that a
sliding window computes each frame's part once instead of seven times.

A window's frames are each encoded (Head: four feature channels, upsampled to the frame's
size) and low-passed and resized to the aligner's three stage sizes, frames and features
alike (_resize_like); nothing else in the network depends on one frame only, and the
full-size features reach nothing but those resizes. The confidence head's half-size centre,
centre features and support features are the same resizes at the largest stage size.

`frame_part` maps one frame to its six pyramid levels: the frame at the three stage sizes
(3 channels each), then its features at the three (4 channels each), smallest first.
`window_part` runs the network's own forward on the window's frames with those levels
cached: inside `cached_pyramids` the encoder gives a placeholder that nothing reads, the
aligner's two pyramid builders hand out the cached levels, and the three half-size resizes
of the forward return theirs. Each level enters as a (7, C, h, w) stack in window order.
The vendored network is used as it is; its module and instances are patched only while the
context lasts. The parts are pytorch/temporal_fix_parts.py's (the PyTorch route runs them
too); this module exports them to ONNX."""

from __future__ import annotations

import json
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import onnx
import torch
from onnx import helper
from torch import nn

from vendor.temporalfix import temporalfix_arch as arch

from ..pytorch.temporal_fix import build_model, even_size
from ..pytorch.temporal_fix_parts import (
    FRAMES,
    LEVELS,
    SUPPORTS,
    FramePart,
    WindowPart,
    cached_pyramids,
    frame_levels,
    stage_sizes,
)
from .rdual_export import save
from .temporal_fix_export import OPSET, Clip, mixed_precision

# The set's engines: once per new frame, once per window
PARTS = ("frame", "window")


FRAME_INPUTS = ["frame"]
FRAME_OUTPUTS = [f"frame_{i}" for i in range(LEVELS)] + [
    f"features_{i}" for i in range(LEVELS)
]


def window_inputs() -> list[str]:
    """The window part's inputs: per window position t the frame "x<t>" (1, 3, H, W),
    then per position its six levels "<level>@<t>" (FRAME_OUTPUTS' shapes)."""
    frames = [f"x{t}" for t in range(FRAMES)]
    return frames + [f"{name}@{t}" for t in range(FRAMES) for name in FRAME_OUTPUTS]


class WindowClip(nn.Module):
    """temporal_fix_export.Clip with its frames' levels cached: the window's 7 frames
    and their levels, each its own input (window_inputs), -> the node's image. Every
    input is one frame's, so a session binds each to that frame's memory, which a
    sliding window keeps from step to step; the graph's concatenations are the only
    copies."""

    def __init__(self, model: arch.temporalfix_arch, height: int, width: int) -> None:
        super().__init__()
        self.clip = Clip(model, height, width)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        frames, levels = inputs[:FRAMES], inputs[FRAMES:]
        count = len(FRAME_OUTPUTS)
        stacks = [torch.cat(levels[i::count]) for i in range(count)]
        with cached_pyramids(self.clip.model, stacks):
            return self.clip(torch.cat(frames, dim=1))


def _export(
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
) -> onnx.ModelProto:
    with tempfile.TemporaryDirectory(prefix="chaiNNer-tfix-part-") as temporary:
        path = Path(temporary) / "part.onnx"
        with torch.no_grad():
            torch.onnx.export(
                module,
                inputs,
                str(path),
                opset_version=OPSET,
                input_names=input_names,
                output_names=output_names,
                dynamo=False,
                do_constant_folding=True,
                external_data=False,
            )
        graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    return graph


def parts(
    state: OrderedDict[str, torch.Tensor], height: int, width: int, device: str
) -> tuple[onnx.ModelProto, onnx.ModelProto]:
    """The frame part ("frame" (1, 3, H, W) FP16 -> FRAME_OUTPUTS) and the window part
    (window_inputs, FP16 -> the node's image, as temporal_fix_export.part gives it) of
    a strength's weights for frames of height x width, traced in FP16 on `device`, in
    temporal_fix_export.part's precision (mixed_precision)."""
    hw = even_size(height, width)
    model = build_model(state, hw, torch.device(device)).half()
    generator = torch.Generator().manual_seed(0)
    window = torch.rand((1, FRAMES, 3, *hw), generator=generator).half().to(device)
    with mixed_precision(model):
        frame = _export(FramePart(model), (window[:, 0],), FRAME_INPUTS, FRAME_OUTPUTS)
        with torch.no_grad():
            levels = frame_levels(model, window)
        frames = [window[:, t] for t in range(FRAMES)]
        per_frame = [level[t : t + 1] for t in range(FRAMES) for level in levels]
        clip = _export(
            WindowClip(model, height, width),
            (*frames, *per_frame),
            window_inputs(),
            ["output"],
        )
    return frame, clip


def main(
    weights: str, output: str, height: str, width: str, device: str = "cpu"
) -> None:
    """The two parts of a strength's weights in OUTPUT/frame and OUTPUT/window, the
    manifest in OUTPUT/export.json (temporal_fix_export.main's, with both parts)."""
    from safetensors.torch import load_file

    state = OrderedDict(load_file(weights))
    graphs = dict(zip(PARTS, parts(state, int(height), int(width), device)))
    root = Path(output)
    manifest: dict[str, Any] = {
        "height": int(height),
        "width": int(width),
        "scale": 1,
        "clip_size": FRAMES,
        "architecture": "TemporalFix",
        "parts": {},
    }
    for name, graph in graphs.items():
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


__all__ = [
    "FRAMES",
    "FRAME_INPUTS",
    "FRAME_OUTPUTS",
    "PARTS",
    "SUPPORTS",
    "FramePart",
    "WindowClip",
    "WindowPart",
    "cached_pyramids",
    "frame_levels",
    "parts",
    "stage_sizes",
    "window_inputs",
]


if __name__ == "__main__":
    main(*sys.argv[1:])
