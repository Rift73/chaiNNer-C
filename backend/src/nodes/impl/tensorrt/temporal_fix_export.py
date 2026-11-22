"""Temporal Fix's TensorRT ONNX for one frame size, exported in a child process
(dual.run_child) from the weights of one strength: one part, "clip", that maps the 7
frames of a window stacked on the channels, `x` (1, 21, H, W) in FP16 (RGB, 0..1, the
size made even as the node makes it), to the stabilized centre frame `output`
(HEIGHT, WIDTH, 3) in FP32, BGR, its values as they are: the image the node hands on,
so the engine's output goes to the host in one copy. Every shape is static.

The part computes in a targeted mixed precision (mixed_precision), from a map of each
part's precision against FP32 on real footage: the convolutions (encoder, the three flow
refinement stages, the confidence head) and the image pyramids in FP16; in FP32 the
flow field (each stage's update, its sum, its resizing and scaling), the positions and
the sampling of all five warps, and the confidence-weighted temporal average. Any warp's
positions in FP16 put up to 10 times the pixels off by more than a level. The base grids
are the PyTorch route's stored FP32 ones (stored_grids).

CHECKED (RTX 5090, strength 2, the two-cut engine set against PyTorch FP32, 30 windows of
7 frames per clip, levels of 255: largest, mean, share of the pixels off by more than 1
and by more than 4 levels): a 1080p BluRay episode through the node, 6.85, 0.0189,
0.0014 %, 0.000026 % (PyTorch FP16: 54.3, 0.0489, 0.088 %, 0.0061 %); a 360p clip at its
size in three builds, 9.37 to 16.6, 0.0147 to 0.0151, 0.016 to 0.020 %, 0.00063 to
0.00080 % (PyTorch FP16: 9.54, 0.0297, 0.028 %, 0.00057 %). The first two stages'
convolutions in FP32 instead bring the small clip's largest to 4.6 to 15.0 and its share
off by more than 4 levels to 0.00006 to 0.00049 %, at 0.28 ms at 1920x1080.

    python -m nodes.impl.tensorrt.temporal_fix_export WEIGHTS OUTPUT HEIGHT WIDTH
        [DEVICE]

WEIGHTS is a safetensors file of the strength's state dict (temporal_fix.load_state);
HEIGHT and WIDTH are the frames' own size; DEVICE (a torch device, default cpu) is where
the part is traced; the graph does not depend on it. The part goes to OUTPUT/clip
(model.onnx and weights.bin), the manifest to OUTPUT/export.json."""

from __future__ import annotations

import json
import sys
import tempfile
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import onnx
import torch
from onnx import helper
from torch import nn
from torch.nn import functional

from vendor.temporalfix import temporalfix_arch as arch

from ..pytorch.temporal_fix import RADIUS, build_model, even_size
from .rdual_export import save

PART = "clip"
OPSET = 17
FRAMES = 2 * RADIUS + 1


class Clip(nn.Module):
    """The stacked window in FP16 -> the centre frame as the node's BGR HWC FP32 image,
    cropped to the frames' own size."""

    def __init__(self, model: arch.temporalfix_arch, height: int, width: int) -> None:
        super().__init__()
        self.model = model
        self.height, self.width = height, width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, channels, h, w = x.shape
        frames = x.view(1, FRAMES, channels // FRAMES, h, w)
        out = self.model(frames)[0, :, : self.height, : self.width]
        return out.flip(0).permute(1, 2, 0).float()


def stored_grids(aligner: arch.PairwiseCenterAligner) -> None:
    """The aligner's base grids looked up by the sizes' integers: the trace gives sizes
    as tensors, which miss the stored FP32 grids, and the network would build each grid
    again in FP16 (with Range, which TensorRT does not take, and up to half a pixel off
    at 1920 px). The stored ones, cast to FP16 where the PyTorch route casts them, are
    its exact grids."""
    lookup = aligner.get_static_base_grid

    def by_integers(
        hw: tuple[int, int], *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor | None:
        return lookup((int(hw[0]), int(hw[1])), device=device, dtype=dtype)

    aligner.get_static_base_grid = by_integers


_warp = arch.warp
_average = arch.conservative_temporal_average


def fp32_warp(
    ten_input: torch.Tensor,
    ten_flow: torch.Tensor,
    base_grid: torch.Tensor | None = None,
    force_fp32: bool = False,
) -> torch.Tensor:
    """The network's warp with its positions and sampling in FP32, wherever it is. The
    positions are the stored grid's row and column, broadcast, plus the scaled flow:
    the values of the network's grid, without a grid-sized constant in the graph. The
    network warps a tensor of ones (one channel) for the share of each position that
    falls inside the frame; that share, for a tensor of ones only, is computed from the
    positions (valid_share)."""
    del force_fp32
    if base_grid is None:
        return _warp(ten_input, ten_flow, base_grid=None, force_fp32=True)
    flow = ten_flow.float()
    _, _, h, w = flow.shape
    grid = base_grid.float()
    x = grid[:, 0:1, 0:1, :] + flow[:, 0:1] * (2.0 / float(w))
    y = grid[:, 1:2, :, 0:1] + flow[:, 1:2] * (2.0 / float(h))
    if ten_input.shape[1] == 1 and bool(torch.all(ten_input == 1)):
        # The aligner's tensor of ones (its values are known when the graph is traced)
        return valid_share(x, y).to(ten_input.dtype)
    out = functional.grid_sample(
        input=ten_input.float(),
        grid=torch.cat((x, y), dim=1).permute(0, 2, 3, 1),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return out.to(ten_input.dtype)


def valid_share(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Bilinear sampling of ones with zero padding at the normalized positions x, y
    (align_corners=False): per axis, the weight of the corners inside the frame,
    min(i + 1, size - i) clamped to 0..1 at the pixel position i."""
    # The sizes as integers: constants in the graph, which the plugins' matcher reads
    h, w = int(y.shape[-2]), int(x.shape[-1])

    def share(position: torch.Tensor, size: int) -> torch.Tensor:
        pixel = ((position + 1) * size - 1) / 2
        return torch.minimum(pixel + 1, size - pixel).clamp(0, 1)

    return share(x, w) * share(y, h)


def fp32_average(
    *,
    center: torch.Tensor,
    aligned_supports: torch.Tensor,
    support_conf: torch.Tensor,
    conf_thresh: float,
    min_support: int,
    gate_slope: float,
    count_slope: float,
) -> torch.Tensor:
    """The confidence-weighted temporal average in FP32."""
    return _average(
        center=center.float(),
        aligned_supports=aligned_supports.float(),
        support_conf=support_conf.float(),
        conf_thresh=conf_thresh,
        min_support=min_support,
        gate_slope=gate_slope,
        count_slope=count_slope,
    )


def fp32_flow(block: arch.MCBlock) -> None:
    """A refinement stage whose convolutions take FP16 and whose flow update leaves in
    FP32, so that the flow field is summed, resized and scaled in FP32."""
    forward = block.forward
    dtype = block.out_conv.weight.dtype

    def run(
        x: torch.Tensor, flow: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta, conf, latent = forward(
            x.to(dtype), None if flow is None else flow.to(dtype)
        )
        return delta.float(), conf, latent

    block.forward = run


def fp16_inputs(head: arch.ConfidenceHead) -> None:
    """The confidence head on FP16 inputs (the flow arrives in FP32)."""
    forward = head.forward
    dtype = next(head.parameters()).dtype

    def run(
        center: torch.Tensor,
        warped: torch.Tensor,
        f_center: torch.Tensor,
        f_warped: torch.Tensor,
        flow: torch.Tensor,
        delta_t: torch.Tensor,
        raw_conf_logit: torch.Tensor,
    ) -> torch.Tensor:
        return forward(
            center=center.to(dtype),
            warped=warped.to(dtype),
            f_center=f_center.to(dtype),
            f_warped=f_warped.to(dtype),
            flow=flow.to(dtype),
            delta_t=delta_t.to(dtype),
            raw_conf_logit=raw_conf_logit.to(dtype),
        )

    head.forward = run


def stages(aligner: arch.PairwiseCenterAligner) -> list[arch.MCBlock]:
    """The aligner's refinement stages, coarsest first."""
    blocks = []
    for block in aligner.blocks:
        assert isinstance(block, arch.MCBlock)
        blocks.append(block)
    return blocks


@contextmanager
def mixed_precision(model: arch.temporalfix_arch) -> Iterator[None]:
    """`model` (in FP16) in the export's precision while the context lasts: its own
    modules changed for good, the vendored module's warp and average swapped back
    after."""
    stored_grids(model.aligner)
    for block in stages(model.aligner):
        fp32_flow(block)
    fp16_inputs(model.conf_head)
    arch.warp = fp32_warp
    arch.conservative_temporal_average = fp32_average
    try:
        yield
    finally:
        arch.warp = _warp
        arch.conservative_temporal_average = _average


def part(
    state: OrderedDict[str, torch.Tensor], height: int, width: int, device: str
) -> onnx.ModelProto:
    """The part of a strength's weights for frames of height x width, traced in FP16 on
    `device`."""
    hw = even_size(height, width)
    model = build_model(state, hw, torch.device(device)).half()
    generator = torch.Generator().manual_seed(0)
    x = torch.rand((1, FRAMES * 3, *hw), generator=generator).half()
    with mixed_precision(model), tempfile.TemporaryDirectory(
        prefix="chaiNNer-tfix-part-"
    ) as temporary:
        path = Path(temporary) / "part.onnx"
        with torch.no_grad():
            torch.onnx.export(
                Clip(model, height, width),
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
    weights: str, output: str, height: str, width: str, device: str = "cpu"
) -> None:
    from safetensors.torch import load_file

    state = OrderedDict(load_file(weights))
    graph = part(state, int(height), int(width), device)
    root = Path(output)
    save(graph, root / PART)
    manifest: dict[str, Any] = {
        "height": int(height),
        "width": int(width),
        "scale": 1,
        "clip_size": FRAMES,
        "architecture": "TemporalFix",
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
