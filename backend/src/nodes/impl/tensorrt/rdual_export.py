"""RDUAL's TensorRT ONNX parts for one input size, exported on the CPU in a child process
(dual.run_child with cpu_only) from a folded model's weights:

- prefix:  input -> stem, fine, center (after group 0), key (coarse retrieval keys)
- revcore: fine, center, key + the future donor -> state_out (the reverse sweep's step)
- fwdcore: input, stem, fine, center, key + the past and future donors -> output,
  state_out (the forward sweep's step with the output stage)
- flow:    frames [2, 3, H, W] (earlier, later) -> guidance [2, 2, Ht, Wt], the flow of
  one adjacent pair in both directions in trunk pixels, as RDUAL.guidance gives it

A donor is state_<d> (BF16 NHWC trunk), guidance_<d> (FP32 [1, 5, Ht, Wt]: x/y flow
then the three alignment cues of RDUAL.pack_guidance) and valid_<d> (BOOL [1]).
The trunk parts come from the vendored DUAL graph builder (vendor/dual_tensorrt) with
the state fusion added, so they run in DUAL's precision (BF16 with FP32 statistics);
the fusion's cue network keeps the PyTorch module's FP32. The flow part is the model's
own NeuFlow v2 in the precision flow_precision chooses (MIXED_FLOW: the encoder, cross
attention, merges, refinement and upsampling convolutions in FP16; the global matching,
flow field and its accumulation, correlation sampling and convex upsampling in FP32).
It holds no all-pairs correlation volume: each lookup is a window and a sample plugin
over the two feature maps, and the global matching an online-softmax plugin
(rdual_aot.py), so its working memory stays small at video sizes.

    python -m nodes.impl.tensorrt.rdual_export WEIGHTS OUTPUT HEIGHT WIDTH [DEVICE [FLOW]]

DEVICE (a torch device, default cpu) is where the flow part is traced; the graphs do not
depend on it. FLOW names the flow's precision recipe (FLOW_RECIPES); empty or absent,
flow_precision chooses it by the input size. WEIGHTS is a safetensors file of the folded model's state dict whose metadata holds the
model's hyperparameters (HYPERPARAMETERS_KEY, JSON); each part goes to OUTPUT/<part>
(model.onnx and weights.bin), the manifest to OUTPUT/export.json."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import onnx
import torch
from onnx import TensorProto, helper
from spandrel.architectures.RDUAL.__arch import rdual_arch
from torch import nn
from torch.nn import functional

from .rdual import HYPERPARAMETERS_KEY
from .rdual_recipe import (
    MIXED_FLOW,
    FlowPrecision,
    Size,
    flow_precision,
)

SCRIPTS = (
    Path(__file__).resolve().parents[3] / "vendor/dual_tensorrt/scripts/dual_tensorrt"
)
PARTS = ("prefix", "revcore", "fwdcore", "flow")
# The donors' slots in the temporal attention's slot bias
SLOTS = {"past": 0, "future": 1}
# The vendored graph builder's only context windows
CONTEXT_WINDOWS = [32, 64]
OPSET = 20


def _vendored(name: str) -> ModuleType:
    """A module of the vendored exporter, loaded from its file as dual_aot loads its
    kernels (a standalone module)."""
    spec = importlib.util.spec_from_file_location(
        f"dual_tensorrt_{name}", SCRIPTS / f"{name}.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_graph = _vendored("graph")
_spec = _vendored("spec")


def neighborhoods(model: Any) -> tuple[bool, bool]:
    """The model's local_na and retrieval_na: local neighborhood attention on its local
    layers above 16 positions per axis, and retrieval neighborhood attention on coarse
    grids above 48, for the 2x and the 4x transfer alike. An RDUAL without the flags
    (spandrel's before them) takes the training source's defaults, both on."""
    return getattr(model, "local_na", True), getattr(model, "retrieval_na", True)


class TrunkGraph(_graph.Graph):
    """The vendored DUAL graph with RDUAL's state fusion before group 1 and the cuts
    between the parts recorded, with the model's neighborhoods (neighborhoods) and the
    routers' row sums, as DUAL's export builds it."""

    def __init__(self, model: Any, spec: Any, donors: list[str]) -> None:
        local_na, retrieval_na = neighborhoods(model)
        super().__init__(
            rdual_arch,
            spec,
            local_na=local_na,
            retrieval_na=retrieval_na,
            row_sums=True,
        )
        self.model, self.donors = model, donors
        self.cut: dict[str, str | None] = {
            "stem": None,
            "fine": None,
            "center": None,
            "key": None,
        }
        self.state_out: str | None = None

    def layer(self, x: str, mod: Any, size: Any) -> str:
        if self.tag == "group0/layer0" and self.cut["stem"] is None:
            self.cut["stem"] = x
        out = super().layer(x, mod, size)
        if self.tag == "group0/layer1":
            self.cut["fine"] = out
        return out

    def pw(self, x: str, mod: Any) -> str:
        out = super().pw(x, mod)
        if mod is self.model.xr.key:
            self.cut["key"] = out
        return out

    def fusion(self, module: Any, center: str) -> str:
        """StateFusion.forward over this graph's donors: the query's last lane holds
        sqrt(32), where each key holds its cue score."""
        self.tag = "temporal"
        h, w = self.height, self.width
        n = h * w
        query = self.pw(self.norm(center, module.norm), module.query)
        q = self.trans(self.reshape(query, [1, n, 4, 32]), [0, 2, 1, 3])
        mask = torch.ones(32)
        mask[31] = 0
        lane = torch.zeros(32)
        lane[31] = math.sqrt(32)
        q = self.op(
            "Add",
            [
                self.op("Mul", [q, self.const(mask, TensorProto.BFLOAT16)]),
                self.const(lane, TensorProto.BFLOAT16),
            ],
        )
        weight = torch.cat((module.key.weight, module.value.weight), 0)[:, :, 0, 0]
        cue0, cue2 = module.cue_score[0], module.cue_score[2]
        keys, values, valids = [], [], []
        for donor in self.donors:
            guidance = f"guidance_{donor}"
            coords = self.slice(guidance, [0], [2], [1])
            cues = self.slice(guidance, [2], [5], [1])
            normalized = self.norm(f"state_{donor}", module.state_norm)
            kv = self.op(
                "MatMul", [normalized, self.const(weight.T, TensorProto.BFLOAT16)]
            )
            kv = self.plugin("RDUALFlowKV", [kv, coords], height=h, width=w)
            k = self.reshape(
                self.op("Gather", [kv, self.const(0, TensorProto.INT64)], axis=0),
                [1, 1, 4, n, 32],
            )
            v = self.reshape(
                self.op("Gather", [kv, self.const(1, TensorProto.INT64)], axis=0),
                [1, 1, 4, n, 32],
            )
            # The cue network in FP32 (the PyTorch module's weights), joined to the
            # key's last lane after the warp
            score = self.op(
                "Conv",
                [
                    cues,
                    self.const(cue0.weight, TensorProto.FLOAT),
                    self.const(cue0.bias, TensorProto.FLOAT),
                ],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
            )
            score = self.op("Gelu", [score])
            score = self.op(
                "Conv",
                [
                    score,
                    self.const(cue2.weight, TensorProto.FLOAT),
                    self.const(cue2.bias, TensorProto.FLOAT),
                ],
                kernel_shape=[1, 1],
            )
            score = self.cast(
                self.reshape(score, [1, 1, 4, n, 1]), TensorProto.BFLOAT16
            )
            k_main = self.slice(k, [0], [31], [4])
            k_lane = self.op("Add", [self.slice(k, [31], [32], [4]), score])
            keys.append(self.op("Concat", [k_main, k_lane], axis=4))
            values.append(v)
            valids.append(self.reshape(f"valid_{donor}", [1, 1]))

        def joined(items: list[str], axis: int) -> str:
            return items[0] if len(items) == 1 else self.op("Concat", items, axis=axis)

        rows = [SLOTS[donor] for donor in self.donors]
        inputs = [
            q,
            joined(keys, 1),
            joined(values, 1),
            self.const(module.relative_table(), TensorProto.FLOAT),
            self.const(module.null_key, TensorProto.BFLOAT16),
            self.const(module.null_bias, TensorProto.FLOAT),
            self.const(module.slot_bias[rows], TensorProto.FLOAT),
            joined(valids, 1),
        ]
        out = self.plugin(
            "RDUALTemporal",
            inputs,
            outputs=2,
            height=h,
            width=w,
            slots=len(rows),
            radius=module.radius,
        )
        out = self.reshape(
            self.trans(self.cast(out, TensorProto.BFLOAT16), [0, 2, 1, 3]),
            [1, h, w, 128],
        )
        return self.op("Add", [center, self.pw(out, module.output)])

    def group(self, x: str, mod: Any, index: int) -> str:
        if index == 1:
            self.cut["center"] = x
            if self.donors:
                x = self.fusion(self.model.temporal, x)
        x = super().group(x, mod, index)
        if index == len(self.model.groups) - 1:
            self.state_out = x
        return x

    def build(self, model: Any) -> onnx.ModelProto:
        graph = super().build(model)
        trunk = [1, self.height, self.width, self.c]
        for donor in self.donors:
            graph.graph.input.extend(
                [
                    helper.make_tensor_value_info(
                        f"state_{donor}", TensorProto.BFLOAT16, trunk
                    ),
                    helper.make_tensor_value_info(
                        f"guidance_{donor}",
                        TensorProto.FLOAT,
                        [1, 5, self.height, self.width],
                    ),
                    helper.make_tensor_value_info(
                        f"valid_{donor}", TensorProto.BOOL, [1]
                    ),
                ]
            )
        missing = [name for name, tensor in self.cut.items() if tensor is None]
        if missing or self.state_out is None:
            raise RuntimeError(f"RDUAL export found no {missing or 'state'} tensor")
        return graph


def part(
    full: onnx.ModelProto,
    outputs: dict[str, str],
    cuts: dict[str, str],
    infos: dict[str, tuple[int, list[int]]],
) -> onnx.ModelProto:
    """The part of `full` that computes `outputs` (graph output name -> tensor), with
    each tensor of `cuts` (graph input name -> tensor) read from a graph input instead."""
    rename = {tensor: name for name, tensor in cuts.items()}
    nodes: list[onnx.NodeProto] = []
    for original in full.graph.node:
        if any(output in rename for output in original.output):
            continue
        node = onnx.NodeProto.FromString(original.SerializeToString())
        for index, name in enumerate(node.input):
            if name in rename:
                node.input[index] = rename[name]
        nodes.append(node)
    nodes += [
        helper.make_node("Identity", [tensor], [name], name=f"{name}_out")
        for name, tensor in outputs.items()
        if name != tensor
    ]
    producer = {output: node for node in nodes for output in node.output}
    needed: set[str] = set()
    kept: set[int] = set()
    stack = list(outputs)
    while stack:
        tensor = stack.pop()
        if tensor in needed:
            continue
        needed.add(tensor)
        node = producer.get(tensor)
        if node is not None:
            kept.add(id(node))
            stack.extend(name for name in node.input if name)
    graph = helper.make_graph(
        [node for node in nodes if id(node) in kept],
        "RDUAL",
        [
            *(i for i in full.graph.input if i.name in needed),
            *(
                helper.make_tensor_value_info(name, *infos[name])
                for name in cuts
                if name in needed
            ),
        ],
        [helper.make_tensor_value_info(name, *infos[name]) for name in outputs],
        [i for i in full.graph.initializer if i.name in needed],
    )
    return helper.make_model(
        graph, opset_imports=list(full.opset_import), ir_version=full.ir_version
    )


def save(model: onnx.ModelProto, folder: Path) -> None:
    """The ONNX and one external weights file written in one pass (Windows Defender
    rescans a file on every close, as export.save_onnx notes)."""
    folder.mkdir(parents=True, exist_ok=True)
    blob = bytearray()
    for tensor in model.graph.initializer:
        if not tensor.HasField("raw_data") or len(tensor.raw_data) <= 1024:
            continue
        offset = len(blob)
        blob += tensor.raw_data
        length = len(blob) - offset
        blob += b"\0" * ((-len(blob)) % 64)
        tensor.ClearField("raw_data")
        tensor.data_location = TensorProto.EXTERNAL
        del tensor.external_data[:]
        for key, value in (
            ("location", "weights.bin"),
            ("offset", str(offset)),
            ("length", str(length)),
        ):
            entry = tensor.external_data.add()
            entry.key, entry.value = key, value
    (folder / "weights.bin").write_bytes(bytes(blob))
    onnx.save_model(model, str(folder / "model.onnx"))


def trunk_parts(model: Any, size: Size) -> dict[str, onnx.ModelProto]:
    """The prefix, revcore and fwdcore parts. The vendored graph builder reads DUAL's
    attributes, which RDUAL keeps as constants or names otherwise: the model (the
    child's own copy) gets them first."""
    if list(model.hyperparameters["c_windows"]) != CONTEXT_WINDOWS:
        raise ValueError(
            f"RDUAL's TensorRT export supports the context windows {CONTEXT_WINDOWS}"
            f" only, not {model.hyperparameters['c_windows']}"
        )
    model.unshuffle = True
    local_na = neighborhoods(model)[0]
    for module in model.modules():
        if isinstance(module, rdual_arch.SpatialAttention):
            # Attributes the vendored builder reads, which RDUAL's layer lacks:
            # pre_gate_core's rule (context layers use neighborhood attention once an
            # axis exceeds the window) is DUAL's context_na, and the training source
            # sets local_na on the local slots 0, 1, 3 and 4 of each group, which are
            # every layer that is not a context one (2 and 5).
            attention: Any = module
            attention.context_na = attention.context
            attention.local_na = local_na and not attention.context
            if not hasattr(attention, "factored_bias"):
                attention.factored_bias = None
    channels = model.stem.out_channels
    spec = _spec.Specialization(
        "dual_light",
        model.scale,
        size.padded_height,
        size.padded_width,
        channels,
        True,
        tuple(len(group) for group in rdual_arch.GROUP_INDICES),
        tuple(rdual_arch.GROUP_INDICES),
    )
    with torch.no_grad():
        reverse = TrunkGraph(model, spec, ["future"])
        reverse_full = reverse.build(model)
        forward = TrunkGraph(model, spec, ["past", "future"])
        forward_full = forward.build(model)
    trunk = [1, size.trunk_height, size.trunk_width, channels]
    infos: dict[str, tuple[int, list[int]]] = {
        "output": (
            TensorProto.FLOAT,
            [1, 3, size.padded_height * model.scale, size.padded_width * model.scale],
        ),
        "state_out": (TensorProto.BFLOAT16, trunk),
        "stem": (TensorProto.BFLOAT16, trunk),
        "fine": (TensorProto.BFLOAT16, trunk),
        "center": (TensorProto.BFLOAT16, trunk),
        "key": (
            TensorProto.BFLOAT16,
            [
                1,
                size.trunk_height // 2,
                size.trunk_width // 2,
                model.xr.key.weight.shape[0],
            ],
        ),
    }

    def cuts(graph: TrunkGraph) -> dict[str, str]:
        return {name: tensor for name, tensor in graph.cut.items() if tensor}

    assert reverse.state_out is not None and forward.state_out is not None
    return {
        "prefix": part(forward_full, cuts(forward), {}, infos),
        "revcore": part(
            reverse_full, {"state_out": reverse.state_out}, cuts(reverse), infos
        ),
        "fwdcore": part(
            forward_full,
            {"output": "output", "state_out": forward.state_out},
            cuts(forward),
            infos,
        ),
    }


class _CorrWindow(torch.autograd.Function):
    """Export only: the correlation window (rdual_aot.window_kernel) of channel-last
    features, as the RDUALCorrWindow node."""

    @staticmethod
    def forward(
        ctx: Any,
        f0: torch.Tensor,
        f1: torch.Tensor,
        flow: torch.Tensor,
        batch: int,
        height: int,
        width: int,
        channels: int,
        half: bool,
    ) -> torch.Tensor:
        return flow.new_zeros(batch * height * width, 128)

    @staticmethod
    def symbolic(
        g: Any,
        f0: Any,
        f1: Any,
        flow: Any,
        batch: int,
        height: int,
        width: int,
        channels: int,
        half: bool,
    ) -> Any:
        return g.op(
            "trt::RDUALCorrWindow",
            f0,
            f1,
            flow,
            batch_i=batch,
            height_i=height,
            width_i=width,
            channels_i=channels,
            half_i=int(half),
        ).setType(flow.type().with_sizes([batch * height * width, 128]))


class _CorrSample(torch.autograd.Function):
    """Export only: the lookup reading the window (rdual_aot.sample_kernel), as the
    RDUALCorrSample node."""

    @staticmethod
    def forward(
        ctx: Any,
        d: torch.Tensor,
        flow: torch.Tensor,
        batch: int,
        height: int,
        width: int,
        channels: int,
        half: bool,
    ) -> torch.Tensor:
        return flow.new_zeros(batch, 81, height, width)

    @staticmethod
    def symbolic(
        g: Any,
        d: Any,
        flow: Any,
        batch: int,
        height: int,
        width: int,
        channels: int,
        half: bool,
    ) -> Any:
        return g.op(
            "trt::RDUALCorrSample",
            d,
            flow,
            batch_i=batch,
            height_i=height,
            width_i=width,
            channels_i=channels,
            half_i=int(half),
        ).setType(flow.type().with_sizes([batch, 81, height, width]))


class _Match(torch.autograd.Function):
    """Export only: the global matching as one online-softmax plugin
    (rdual_aot.match_kernel), as the RDUALMatch node."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        g: torch.Tensor,
        batch: int,
        n: int,
        channels: int,
    ) -> torch.Tensor:
        return functional.scaled_dot_product_attention(q, k, g)

    @staticmethod
    def symbolic(
        g_: Any, q: Any, k: Any, g: Any, batch: int, n: int, channels: int
    ) -> Any:
        return g_.op(
            "trt::RDUALMatch", q, k, g, batch_i=batch, n_i=n, channels_i=channels
        ).setType(q.type().with_sizes([batch, n, 2]))


def _applied(function: type[torch.autograd.Function], *arguments: Any) -> torch.Tensor:
    result = function.apply(*arguments)
    assert isinstance(result, torch.Tensor)
    return result


class FlowPair(nn.Module):
    """RDUAL.guidance of one adjacent pair, both directions, as the flow part's graph:
    NeuFlowV2.forward in `precision` with the correlation lookups and the global
    matching as plugin nodes, so that no all-pairs volume or N x N matrix exists. For
    export only (the plugins are stand-ins outside it)."""

    def __init__(self, flow: Any, size: Size, precision: FlowPrecision) -> None:
        super().__init__()
        # The model's NeuFlowV2, read by its own attributes
        self.m: Any = flow
        self.size = size
        self.half_features = precision.half
        for name in precision.half_modules:
            getattr(self.m, name).half()

    @staticmethod
    def features(f0: torch.Tensor, f1: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The two feature maps channel-last, (B, H*W, C), as the window reads them."""
        b, c, h, w = f0.shape
        return (
            f0.reshape(b, c, h * w).transpose(1, 2),
            f1.reshape(b, c, h * w).transpose(1, 2),
        )

    def sample(
        self, features: tuple[torch.Tensor, ...], flow: torch.Tensor
    ) -> torch.Tensor:
        batch, _, height, width = flow.shape
        f0, f1 = features
        geometry = (
            int(batch),
            int(height),
            int(width),
            int(f0.shape[-1]),
            self.half_features,
        )
        window = _applied(_CorrWindow, f0, f1, flow, *geometry)
        return _applied(_CorrSample, window, flow, *geometry)

    def neuflow(self, img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
        a, m = rdual_arch, self.m
        half = torch.float16 if self.half_features else torch.float32
        b, _, height, width = img0.shape
        grids = m.prepare_grids(b, height, width, img0.device)
        img0 = (img0 / 255.0).to(half)
        img1 = (img1 / 255.0).to(half)
        s16, s8 = m.backbone(torch.cat([img0, img1], 0), grids["pos_s16"].to(half))
        s16 = m.cross_attn_s16(s16)
        context_s16, features_s16 = torch.split(
            s16, [a.NF_CONTEXT_DIM_S16, a.NF_FEATURE_DIM_S16], dim=1
        )
        context_s16 = torch.relu(context_s16.chunk(2, 0)[0])
        context_s8, features_s8 = torch.split(
            s8, [a.NF_CONTEXT_DIM_S8, a.NF_FEATURE_DIM_S8], dim=1
        )
        context_s8 = torch.relu(context_s8.chunk(2, 0)[0])
        f0, f1 = features_s16.chunk(2, 0)
        q = f0.float().flatten(-2).permute(0, 2, 1)
        k = f1.float().flatten(-2).permute(0, 2, 1)
        n, channels = int(q.shape[1]), int(q.shape[2])
        correspondence = _applied(
            _Match, q, k, grids["flat_grid_s16"], int(b), n, channels
        )
        h16, w16 = f0.shape[-2:]
        flow = correspondence.view(b, h16, w16, 2).permute(0, 3, 1, 2)
        flow = flow - grids["grid_s16"]
        corr = self.features(f0, f1)
        iteration = torch.zeros(
            b, a.NF_ITER_CONTEXT_DIM_S16, h16, w16, device=img0.device, dtype=half
        )
        for _ in range(a.NF_ITERS_S16):
            corrs = self.sample(corr, flow)
            iteration, delta = m.refine_s16(
                corrs.to(half),
                context_s16,
                iteration,
                flow.to(half),
                grids["radius_s16"].to(half),
            )
            flow = flow + delta.float()
        flow = functional.interpolate(flow, scale_factor=2, mode="nearest") * 2
        features_s16 = functional.interpolate(
            features_s16, scale_factor=2, mode="nearest"
        )
        features_s8 = m.merge_s8(torch.cat([features_s8, features_s16], 1))
        f0, f1 = features_s8.chunk(2, 0)
        corr = self.features(f0, f1)
        context_s16 = functional.interpolate(
            context_s16, scale_factor=2, mode="nearest"
        )
        context_s8 = m.context_merge_s8(torch.cat([context_s8, context_s16], 1))
        iteration = torch.zeros(
            b,
            a.NF_ITER_CONTEXT_DIM_S8,
            *f0.shape[-2:],
            device=img0.device,
            dtype=half,
        )
        for _ in range(a.NF_ITERS_S8):
            corrs = self.sample(corr, flow)
            iteration, delta = m.refine_s8(
                corrs.to(half),
                context_s8,
                iteration,
                flow.to(half),
                grids["radius_s8"].to(half),
            )
            flow = flow + delta.float()
        feature = m.conv_s8(img0)
        up = m.upsample_s8
        k = up.upsample_factor
        mask = up.conv3(
            up.relu(up.conv2(up.relu(up.conv1(torch.cat((flow.to(half), feature), 1)))))
        )
        bb, _, h, w = flow.shape
        mask = torch.softmax(mask.float().view(bb, 1, 9, k, k, h, w), dim=2)
        up_flow = functional.unfold(flow, (3, 3), padding=1).view(bb, 2, 9, 1, 1, h, w)
        up_flow = torch.sum(mask * up_flow, dim=2).permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(bb, 2, k * h, k * w) * 8

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        s = self.size
        x = frames.float() * 255.0
        earlier, later = x[0:1], x[1:2]
        # direction 0 maps the later frame onto the earlier one, 1 the reverse
        source = torch.cat((later, earlier), 0)
        target = torch.cat((earlier, later), 0)
        padding = (0, s.flow_width - s.width, 0, s.flow_height - s.height)
        source = functional.pad(source, padding, mode="replicate")
        target = functional.pad(target, padding, mode="replicate")
        flow = self.neuflow(source, target)[:, :, : s.height, : s.width]
        flow = functional.pad(
            flow,
            (0, s.padded_width - s.width, 0, s.padded_height - s.height),
            mode="replicate",
        )
        flow = functional.interpolate(
            flow,
            size=(s.trunk_height, s.trunk_width),
            mode="bilinear",
            align_corners=False,
        )
        return flow * 0.5


def flow_part(
    model: Any, size: Size, device: str = "cpu", recipe: FlowPrecision = MIXED_FLOW
) -> onnx.ModelProto:
    """The flow part, traced on `device`: the trace runs the network once, and its FP16
    layers take minutes on a CPU at video sizes against about a second on a GPU."""
    pair = FlowPair(copy.deepcopy(model.flow), size, recipe)
    pair = pair.eval().to(device)
    frames = torch.rand(2, 3, size.height, size.width, device=device)
    with tempfile.TemporaryDirectory(prefix="chaiNNer-rdual-flow-") as temporary:
        path = Path(temporary) / "flow.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                pair,
                (frames,),
                str(path),
                opset_version=OPSET,
                dynamo=False,
                input_names=["frames"],
                output_names=["guidance"],
                do_constant_folding=True,
                external_data=False,
            )
        model = onnx.load(str(path))
    # The exporter leaves the output's dimensions symbolic; they are the trunk's.
    output = model.graph.output[0]
    output.CopyFrom(
        helper.make_tensor_value_info(
            output.name, TensorProto.FLOAT, [2, 2, size.trunk_height, size.trunk_width]
        )
    )
    return model


def load(weights: Path) -> Any:
    """The folded model of a WEIGHTS file, on the CPU in FP32."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(weights), "pt") as file:
        metadata = file.metadata()
    hyperparameters = json.loads(metadata[HYPERPARAMETERS_KEY])
    model = rdual_arch.RDUAL(**hyperparameters).float().eval()
    model.prepare_for_export()
    model.load_state_dict(load_file(str(weights)), strict=True)
    model.requires_grad_(False)
    return model


def main(
    weights: str,
    output: str,
    height: str,
    width: str,
    device: str = "cpu",
    flow: str = "",
) -> None:
    model = load(Path(weights))
    size = Size(int(height), int(width))
    recipe = flow_precision(size, flow)
    parts = trunk_parts(model, size)
    parts["flow"] = flow_part(model, size, device, recipe)
    root = Path(output)
    local_na, retrieval_na = neighborhoods(model)
    manifest: dict[str, Any] = {
        "height": size.height,
        "width": size.width,
        "scale": model.scale,
        "plugin_key": f"c{model.stem.out_channels}"
        f"_h{size.trunk_height}_w{size.trunk_width}",
        "flow_precision": recipe.name,
        # The attention operators the parts lower, as DUAL's manifest records them
        # (dual.NA_KEYS); a set exported before "local" and "retrieval" keeps local
        # windows and 96 px retrieval blocks (rdual.build warns).
        "context": _spec.CONTEXT,
        **({"local": _spec.LOCAL} if local_na else {}),
        **({"retrieval": _spec.RETRIEVAL} if retrieval_na else {}),
        "parts": {},
    }
    for name in PARTS:
        graph = parts[name]
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
