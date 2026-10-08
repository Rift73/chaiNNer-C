"""Test-only FP32 interpreter for the lowering, not a runtime/export backend."""

import torch
from torch.nn import functional as F
from onnx import TensorProto as T

from .graph import Graph


class FP32GraphOracle(Graph):
    """Run the same graph construction with FP32 tensors to catch layout/scale bugs.

    Plugin RCA semantics use the canonical reference. This tests lowering wiring,
    NOT CUDA plugin kernels, BF16 rounding, TensorRT parsing or engine performance.
    """

    def __init__(self, arch, spec, input_tensor):
        super().__init__(arch, spec)
        self.input = input_tensor
        self.output = None
        self.core = arch.RegionalChannelAttention(
            spec.channels, spec.channels // 32, core_checkpoint=False
        ).eval()

    def const(self, value, dtype=None):
        return torch.as_tensor(
            value, dtype=torch.int64 if dtype == T.INT64 else torch.float32
        )

    def op(self, kind, inputs, **attrs):
        x = [self.input if isinstance(v, str) and v == "input" else v for v in inputs]
        if kind == "Add":
            result = x[0] + x[1]
        elif kind == "Sub":
            result = x[0] - x[1]
        elif kind == "Mul":
            result = x[0] * x[1]
        elif kind == "Div":
            result = x[0] / x[1]
        elif kind == "Erf":
            result = x[0].erf()
        elif kind == "Sigmoid":
            result = x[0].sigmoid()
        elif kind == "MatMul":
            result = x[0] @ x[1]
        elif kind == "Softmax":
            result = x[0].softmax(attrs["axis"])
        elif kind == "Transpose":
            result = x[0].permute(attrs["perm"])
        elif kind == "Reshape":
            result = x[0].reshape(tuple(x[1].tolist()))
        elif kind == "Concat":
            result = torch.cat(x, dim=attrs["axis"])
        elif kind == "Expand":
            result = x[0].expand(tuple(x[1].tolist()))
        elif kind == "Gather":
            axis = attrs["axis"]
            result = (
                x[0].select(axis, int(x[1]))
                if x[1].ndim == 0
                else x[0].index_select(axis, x[1])
            )
        elif kind == "Slice":
            slices = [slice(None)] * x[0].ndim
            for start, end, axis in zip(x[1], x[2], x[3], strict=True):
                slices[int(axis)] = slice(int(start), int(end))
            result = x[0][tuple(slices)]
        elif kind == "Cast":
            result = x[0].to(torch.int64 if attrs["to"] == T.INT64 else torch.float32)
        elif kind == "Conv":
            pads = attrs.get("pads", [0, 0, 0, 0])
            if pads[:2] != pads[2:]:
                raise AssertionError("Unexpected asymmetric Conv padding")
            result = F.conv2d(
                x[0],
                x[1],
                x[2] if len(x) > 2 else None,
                stride=attrs.get("strides", [1, 1]),
                padding=pads[:2],
                groups=attrs.get("group", 1),
            )
        elif kind == "Pad":
            n = len(x[1]) // 2
            pads = []
            for axis in range(n - 1, -1, -1):
                pads += [int(x[1][axis]), int(x[1][axis + n])]
            mode = "replicate" if attrs.get("mode") == "edge" else "constant"
            if mode == "replicate":
                while len(pads) > 4 and pads[-2:] == [0, 0]:
                    pads = pads[:-2]
            result = F.pad(x[0], pads, mode=mode)
        elif kind == "ReduceMean":
            result = x[0].mean(tuple(x[1].tolist()), keepdim=bool(attrs["keepdims"]))
        elif kind == "LayerNormalization":
            result = F.layer_norm(x[0], (x[0].shape[-1],), x[1], x[2], attrs["epsilon"])
        elif kind == "DepthToSpace":
            result = F.pixel_shuffle(x[0], attrs["blocksize"])
        elif kind == "LeakyRelu":
            result = F.leaky_relu(x[0], attrs["alpha"])
        elif kind == "Resize":
            result = F.interpolate(
                x[0], scale_factor=tuple(x[2][2:].tolist()), mode="nearest"
            )
        else:
            raise AssertionError(kind)
        self.output = result
        return result

    def plugin(self, kind, inputs, outputs=1, **attrs):
        if kind == "DualAttentionBarrier_TRT":
            return inputs[0]
        if kind.startswith("DualNorm_"):
            return F.layer_norm(
                inputs[0], (self.c,), inputs[1], inputs[2], attrs["epsilon"]
            )
        if kind.startswith("DualCore_"):
            x, weight, tau = inputs
            depthwise = F.conv2d(
                x.permute(0, 3, 1, 2),
                weight.permute(2, 0, 1)[:, None],
                padding=1,
                groups=3 * self.c,
            )
            self.core.theta.data.copy_(tau.log())
            return self.core.mix(*depthwise.chunk(3, 1)).permute(0, 2, 3, 1)
        if kind.startswith("DualProject_"):
            return inputs[0] @ inputs[1] + inputs[2]
        raise AssertionError(kind)

    def build(self, model):
        # Stop just before ONNX serialization (tensor-valued inputs cannot be
        # NodeProto names). All graph arithmetic has completed at Identity.
        from unittest.mock import patch

        with (
            patch("scripts.dual_tensorrt.graph.H.make_node", return_value=None),
            patch("scripts.dual_tensorrt.graph.H.make_graph", return_value=None),
            patch("scripts.dual_tensorrt.graph.H.make_model", return_value=None),
        ):
            super().build(model)
        return self.output
