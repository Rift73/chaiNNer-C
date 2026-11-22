"""Temporal Fix's TensorRT plugins as ahead-of-time Python plugins, in DUAL's namespace and
with its compile and launch helpers (dual_aot.py): Triton kernels compiled to PTX for the
building GPU and embedded in the engine.

- round_trip_n<N>_c<C>_h<H>_w<W>_k<K>_b<B>: a half-size map (FP16, or FP32 for the flow)
  upsampled 2x, low-passed with the 5x5 binomial filter and downsampled back, as one
  stencil on the half-size grid (tfix_round_trip_kernel.py), where TensorRT runs an
  upsample, a channel padding copy, a full-size depthwise convolution and a downsample.
- block_input_s<S>_h<H>_w<W>_q<Q>: the last flow refinement block's 26-channel input in one
  pass (tfix_block_input_kernel.py): the supports warped at half size along the flow
  brought up from quarter size, the centre's levels, the time offsets and the previous
  block's logit and latent brought up (that block's output in FP<Q>: 16, or 32 where the
  first two stages run in FP32).
- conf_input_s<S>_h<H>_w<W>_g<G>: the confidence head's 25-channel input per support in
  one pass (tfix_conf_input_kernel.py), where TensorRT's fused kernel branches per part.
  These two (DIRECT) are not engine plugins: the window is cut at them into pieces
  (tfix_cut.py), each kernel launched between two engines, writing the next one's input
  as padded rows (HWC8), which a plugin's output cannot be without a copy behind it.
- tail_warp_s<S>_h<H>_w<W>_p<P> and tail_blend_...: the full-size support warp from the
  half-size flow, then the confidence, the gate and the temporal average in one pass
  (tfix_tail_kernel.py).

`lower` finds every chain by its operations, not by node names: each matcher parses the
chain's exact structure, reads every constant from the operation where it acts and hands
the values to the kernel (P in a plugin's name, the values' FP32 bits: the kernels compute
with the graph's constants, not with built-in ones). What a matcher cannot verify is no
match, and those layers run as exported. It puts a stand-in in place of each chain's
result; `register` registers the engine plugins `lower` used;
dual_aot.attach puts them in after parsing. A graph without such chains lowers to itself.
Needs Triton (its import fails without it); temporal_fix.plugins() then gives None and
the parts are built as exported."""

from __future__ import annotations

import struct
from functools import cache
from typing import Any

import numpy as np
import onnx
import tensorrt as trt
import tensorrt.plugin as trtp
from onnx import helper, numpy_helper, shape_inference

from . import (
    dual_aot,
    tfix_block_input_kernel,
    tfix_conf_input_kernel,
    tfix_round_trip_kernel,
    tfix_tail_kernel,
)
from .dual_aot import Launch, resolved

PREFIX = "tfix"
# The plugin's I/O: the map, its four tap tables, the channel scales, the result
TYPES = "FP16, INT32, FP32, INT32, FP32, FP32, FP16"
# The map types a round trip takes, by ONNX element type
BITS: dict[int, int] = {
    int(onnx.TensorProto.FLOAT16): 16,
    int(onnx.TensorProto.FLOAT): 32,
}
# The ops whose kernels the window's pieces launch directly, between engines (tfix_cut.py)
DIRECT = ("block_input", "conf_input")
# The confidence input's parts' channels
PARTS = (3, 3, 3, 4, 4, 4, 2, 1, 1)
# The network's low-pass filter, the one the confidence kernel's stencil is made of
BINOMIAL = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def round_trip_name(n: int, c: int, h: int, w: int, k: int, b: int) -> str:
    """The plugin of a round trip over N x C maps of H x W with K taps per axis, the
    maps in FP<B> (16, or 32 for the flow)."""
    return f"round_trip_n{n}_c{c}_h{h}_w{w}_k{k}_b{b}"


@cache
def register_round_trip(n: int, c: int, h: int, w: int, k: int, b: int) -> None:
    symbol = f"{dual_aot.NAMESPACE}::{round_trip_name(n, c, h, w, k, b)}"
    types = TYPES.replace("FP16", f"FP{b}")
    sm = dual_aot.device_sm()

    @trtp.register(symbol)
    @resolved
    def round_trip_shape(
        x: trtp.TensorDesc,
        ih: trtp.TensorDesc,
        ch: trtp.TensorDesc,
        iw: trtp.TensorDesc,
        cw: trtp.TensorDesc,
        scale: trtp.TensorDesc,
    ) -> trtp.TensorDesc:
        return x.like()

    @trtp.autotune(symbol)
    @resolved
    def round_trip_layouts(
        x: trtp.TensorDesc,
        ih: trtp.TensorDesc,
        ch: trtp.TensorDesc,
        iw: trtp.TensorDesc,
        cw: trtp.TensorDesc,
        scale: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
    ) -> list[trtp.AutoTuneCombination]:
        tables = "LINEAR, " * 5
        linear = trtp.AutoTuneCombination(types, f"LINEAR, {tables}LINEAR")
        if b == 32:
            # The FP32 flow's channel-last pick ran several times slower in the engine
            return [linear]
        return [trtp.AutoTuneCombination(types, f"HWC, {tables}HWC"), linear]

    @trtp.aot_impl(symbol)
    @resolved
    def round_trip_launch(
        x: trtp.TensorDesc,
        ih: trtp.TensorDesc,
        ch: trtp.TensorDesc,
        iw: trtp.TensorDesc,
        cw: trtp.TensorDesc,
        scale: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        kernel = dual_aot.compile_kernel(
            tfix_round_trip_kernel.round_trip,
            {
                "x_ptr": f"*fp{b}",
                "ih_ptr": "*i32",
                "ch_ptr": "*fp32",
                "iw_ptr": "*i32",
                "cw_ptr": "*fp32",
                "scale_ptr": "*fp32",
                "y_ptr": f"*fp{b}",
            },
            {
                "height": h,
                "width": w,
                "channels": c,
                "k": k,
                "channels_last": x.format == trt.TensorFormat.HWC,
                **tfix_round_trip_kernel.LAUNCH,
            },
            sm,
            tfix_round_trip_kernel.OPTIONS,
        )
        return dual_aot.kernel_launch(kernel, tfix_round_trip_kernel.grid(n * c, h, w))


def conf_input_name(s: int, h: int, w: int, g: int) -> str:
    """The plugin of the confidence input of S supports of H x W, its sampling grid in
    FP<G> (16 or 32)."""
    return f"conf_input_s{s}_h{h}_w{w}_g{g}"


def conf_input_kernel(h: int, w: int, g: int, sm: int) -> Any:
    """The confidence input's kernel for H x W (grid in FP<G>) in its padded-rows form,
    compiled for the GPU architecture `sm`: the window's pieces launch it directly
    between their engines (tfix_cut.py), one program row per support."""
    names = ("centre", "warped", "fcentre", "features", "grid", "flow", "delta")
    return dual_aot.compile_kernel(
        tfix_conf_input_kernel.conf_input,
        {
            **{f"{n}_ptr": "*fp16" for n in names},
            "grid_ptr": f"*fp{g}",
            "out_ptr": "*fp16",
        },
        {
            "height": h,
            "width": w,
            "padded_rows": True,
            **tfix_conf_input_kernel.LAUNCH,
        },
        sm,
        tfix_conf_input_kernel.OPTIONS,
    )


class _Graph:
    def __init__(self, model: onnx.ModelProto) -> None:
        inferred = shape_inference.infer_shapes(model, data_prop=True)
        g = inferred.graph
        self.dims = {
            v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
            for v in [*g.value_info, *g.input, *g.output]
        }
        self.producer = {o: n for n in model.graph.node for o in n.output}
        self.users: dict[str, list[onnx.NodeProto]] = {}
        for node in model.graph.node:
            for name in node.input:
                self.users.setdefault(name, []).append(node)
        self.initializers = {t.name: t for t in model.graph.initializer}
        self.types = {
            v.name: v.type.tensor_type.elem_type
            for v in [*g.value_info, *g.input, *g.output]
        }

    def uncast(self, name: str) -> str:
        """`name` before the Casts that make it."""
        node = self.producer.get(name)
        while node is not None and node.op_type == "Cast":
            name = node.input[0]
            node = self.producer.get(name)
        return name

    def constant(self, name: str) -> np.ndarray | None:
        if name in self.initializers:
            return numpy_helper.to_array(self.initializers[name])
        node = self.producer.get(name)
        if node is None or node.op_type != "Constant":
            return None
        return numpy_helper.to_array(node.attribute[0].t)

    def span(self, name: str) -> tuple[str, int, int] | None:
        """(the tensor, first channel, end channel) of a Slice making `name` on axis 1 with
        step 1, its end clamped to the tensor's channels."""
        node = self.producer.get(name)
        if node is None or node.op_type != "Slice" or len(node.input) < 4:
            return None
        starts, ends, axes = (self.constant(i) for i in node.input[1:4])
        steps = self.constant(node.input[4]) if len(node.input) > 4 else np.ones(1)
        if starts is None or ends is None or axes is None or steps is None:
            return None
        if [int(a) for a in axes.ravel()] != [1] or [int(t) for t in steps.ravel()] != [
            1
        ]:
            return None
        channels = self.dims.get(node.input[0], [0, 0])[1]
        return (
            node.input[0],
            int(starts.ravel()[0]),
            min(int(ends.ravel()[0]), channels),
        )

    def bilinear(self, node: onnx.NodeProto | None, factor: float) -> str | None:
        """The input of a half-pixel bilinear Resize scaling H and W by `factor`."""
        if node is None or node.op_type != "Resize":
            return None
        attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if attributes.get("mode") != b"linear" or attributes.get(
            "coordinate_transformation_mode", b"half_pixel"
        ) not in (b"half_pixel", b"pytorch_half_pixel"):
            return None
        source = self.dims.get(node.input[0])
        result = self.dims.get(node.output[0])
        if not source or not result or len(source) != 4 or 0 in source + result:
            return None
        # Exact 2x sizes only: the kernels index the full size as twice the half size
        small, large = (result, source) if factor < 1 else (source, result)
        if result[:2] != source[:2] or large[2:] != [2 * small[2], 2 * small[3]]:
            return None
        if factor not in (0.5, 2.0):
            return None
        return node.input[0]

    def scaled_upsample(
        self, name: str, channels: int
    ) -> tuple[str, list[float]] | None:
        """(half-resolution source, per-channel scales) of a 2x upsample, directly or
        through a Concat of per-channel Slice * constant."""
        node = self.producer.get(self.uncast(name))
        source = self.bilinear(node, 2.0)
        if source is not None:
            return source, [1.0] * channels
        if node is None or node.op_type != "Concat":
            return None
        if helper.get_attribute_value(node.attribute[0]) != 1:
            return None
        scales: list[float] = []
        upsampled = None
        for part in node.input:
            mul = self.producer.get(part)
            if mul is None or mul.op_type != "Mul":
                return None
            k = self.constant(mul.input[1])
            piece = self.span(mul.input[0])
            if k is None or k.size != 1 or piece is None:
                return None
            if piece[1:] != (len(scales), len(scales) + 1):
                return None
            if upsampled not in (None, piece[0]):
                return None
            upsampled = piece[0]
            scales.append(float(k.reshape(-1)[0]))
        if upsampled is None or len(scales) != channels:
            return None
        source = self.bilinear(self.producer.get(upsampled), 2.0)
        return None if source is None else (source, scales)

    def separable(self, conv: onnx.NodeProto, channels: int) -> np.ndarray | None:
        """The 1-D kernel k of a depthwise Conv whose every filter is outer(k, k),
        stride 1, padding half its size, no bias."""
        if len(conv.input) != 2:
            return None
        attributes = {a.name: helper.get_attribute_value(a) for a in conv.attribute}
        weight = self.constant(conv.input[1])
        if weight is None or attributes.get("group") != channels:
            return None
        size = weight.shape[-1]
        if weight.shape != (channels, 1, size, size) or size % 2 == 0:
            return None
        if any(s != 1 for s in attributes.get("strides", [1, 1])) or any(
            d != 1 for d in attributes.get("dilations", [1, 1])
        ):
            return None
        if list(attributes.get("pads", [0] * 4)) != [size // 2] * 4:
            return None
        w = weight.astype(np.float64)
        k = w[0, 0].sum(1)
        k = k / np.sqrt(k.sum()) if k.sum() > 0 else k
        # Each filter outer(k, k) to its own FP16 rounding (the kernel runs k)
        if not np.allclose(w, np.outer(k, k)[None, None], rtol=1e-3, atol=1e-6):
            return None
        return k


def _conf_input(graph: _Graph, node: onnx.NodeProto) -> list[str] | None:
    """The plugin inputs of the confidence head's input Concat: (centre, warped, |warped -
    centre|, centre features, GridSample(features, grid), |sampled - centre features|,
    flow, ReduceL2(flow), time offset) on the channels."""
    if node.op_type != "Concat" or len(node.input) != len(PARTS):
        return None
    if helper.get_attribute_value(node.attribute[0]) != 1:
        return None
    # The time offset is an Expand whose size the shape inference leaves open
    dims = [graph.dims.get(i, []) for i in node.input[:-1]]
    first = dims[0]
    if any(len(d) != 4 or d[0::2] != first[0::2] for d in dims):
        return None
    if [d[1] for d in dims] != list(PARTS[:-1]):
        return None
    parts = [graph.producer.get(i) for i in node.input]

    def difference(index: int, a: str, b: str) -> bool:
        absolute = parts[index]
        if absolute is None or absolute.op_type != "Abs":
            return False
        sub = graph.producer.get(absolute.input[0])
        return sub is not None and sub.op_type == "Sub" and list(sub.input) == [a, b]

    sample = _sampling(graph, graph.uncast(node.input[4]))
    norm = parts[7]
    if sample is None or norm is None:
        return None
    axes = graph.constant(norm.input[1]) if len(norm.input) > 1 else None
    norm_axes = [list(a.ints) for a in norm.attribute if a.name == "axes"]
    keep = [
        helper.get_attribute_value(a) for a in norm.attribute if a.name == "keepdims"
    ]
    if (
        norm.op_type != "ReduceL2"
        or norm.input[0] != node.input[6]
        or keep not in ([], [1])
    ):
        return None
    if (axes is None and norm_axes != [[1]]) or (
        axes is not None and list(axes) != [1]
    ):
        return None
    if not difference(2, node.input[1], node.input[0]):
        return None
    # The half-size warped frames: the full-size ones low-passed and halved
    filtered = graph.bilinear(graph.producer.get(graph.uncast(node.input[1])), 0.5)
    conv = graph.producer.get(filtered) if filtered is not None else None
    if conv is None or conv.op_type != "Conv":
        return None
    # The kernel's stencil is the binomial: the weights must be its FP16 values exactly
    weight = graph.constant(conv.input[1])
    binomial = np.outer(BINOMIAL, BINOMIAL).astype(np.float16)
    if graph.separable(conv, PARTS[1]) is None or weight is None:
        return None
    if not np.array_equal(
        weight.astype(np.float16), np.broadcast_to(binomial, weight.shape)
    ):
        return None
    if not difference(5, node.input[4], node.input[3]):
        return None
    # The kernel reads the FP16 features (a sampling in FP32 reads their FP32 cast),
    # the grid in its own type
    features = graph.uncast(sample.input[0])
    f16, f32 = onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT
    inputs = [node.input[0], conv.input[0], node.input[3], features]
    inputs += [sample.input[1], node.input[6], node.input[8]]
    if any(graph.types.get(i) != f16 for i in inputs[:4] + inputs[5:]):
        return None
    if graph.types.get(sample.input[1]) not in (f16, f32):
        return None
    return inputs


def _key(values: tuple[float, ...]) -> str:
    """Constants read from the graph as a plugin name's part (their FP32 bits, hex)."""
    return "".join(struct.pack(">f", v).hex() for v in values)


def _unkey(key: str) -> tuple[float, ...]:
    return tuple(
        struct.unpack(">f", bytes.fromhex(key[i : i + 8]))[0]
        for i in range(0, len(key), 8)
    )


# The constants each kernel takes from the graph, in the plugin name's order
WARP_CONSTANTS = ("flow_x", "flow_y")
BLEND_CONSTANTS = (
    *WARP_CONSTANTS,
    "threshold",
    "gate_slope",
    "min_support",
    "offset",
    "count_slope",
    "one",
    "eps",
)


def tail_name(kind: str, s: int, h: int, w: int, constants: tuple[float, ...]) -> str:
    """The plugin of the full-size tail's `kind` pass (warp, blend) for S supports of
    frames of H x W, with the graph's constants (WARP_CONSTANTS, BLEND_CONSTANTS)."""
    return f"tail_{kind}_s{s}_h{h}_w{w}_p{_key(constants)}"


@cache
def register_tail(
    kind: str, s: int, h: int, w: int, constants: tuple[float, ...]
) -> None:
    symbol = f"{dual_aot.NAMESPACE}::{tail_name(kind, s, h, w, constants)}"
    sm = dual_aot.device_sm()
    options = tfix_tail_kernel.OPTIONS

    if kind == "warp":

        @trtp.register(symbol)
        @resolved
        def warp_shape(
            f0: trtp.TensorDesc,
            f1: trtp.TensorDesc,
            f2: trtp.TensorDesc,
            f3: trtp.TensorDesc,
            f4: trtp.TensorDesc,
            f5: trtp.TensorDesc,
            flow: trtp.TensorDesc,
        ) -> trtp.TensorDesc:
            return trtp.from_shape_expr((s, 3, h, w), trt.DataType.HALF)

        @trtp.aot_impl(symbol)
        @resolved
        def warp_launch(
            f0: trtp.TensorDesc,
            f1: trtp.TensorDesc,
            f2: trtp.TensorDesc,
            f3: trtp.TensorDesc,
            f4: trtp.TensorDesc,
            f5: trtp.TensorDesc,
            flow: trtp.TensorDesc,
            outputs: tuple[trtp.TensorDesc],
            tactic: int,
        ) -> Launch:
            frames = {f"f{i}_ptr": "*fp16" for i in range(6)}
            kernel = dual_aot.compile_kernel(
                tfix_tail_kernel.warp,
                {**frames, "flow_ptr": "*fp32", "out_ptr": "*fp16"},
                {
                    "height": h,
                    "width": w,
                    **dict(zip(WARP_CONSTANTS, constants, strict=True)),
                    **tfix_tail_kernel.LAUNCH,
                },
                sm,
                options,
            )
            return dual_aot.kernel_launch(kernel, tfix_tail_kernel.grid(s, h, w))

        return

    @trtp.register(symbol)
    @resolved
    def blend_shape(
        centre: trtp.TensorDesc,
        warped: trtp.TensorDesc,
        logits: trtp.TensorDesc,
        flow: trtp.TensorDesc,
    ) -> trtp.TensorDesc:
        return trtp.from_shape_expr((1, 3, h, w), trt.DataType.FLOAT)

    @trtp.aot_impl(symbol)
    @resolved
    def blend_launch(
        centre: trtp.TensorDesc,
        warped: trtp.TensorDesc,
        logits: trtp.TensorDesc,
        flow: trtp.TensorDesc,
        outputs: tuple[trtp.TensorDesc],
        tactic: int,
    ) -> Launch:
        kernel = dual_aot.compile_kernel(
            tfix_tail_kernel.blend,
            {
                "centre_ptr": "*fp16",
                "warped_ptr": "*fp16",
                "logits_ptr": "*fp16",
                "flow_ptr": "*fp32",
                "out_ptr": "*fp32",
            },
            {
                "height": h,
                "width": w,
                "supports": s,
                **dict(zip(BLEND_CONSTANTS, constants, strict=True)),
                **tfix_tail_kernel.LAUNCH,
            },
            sm,
            options,
        )
        return dual_aot.kernel_launch(kernel, tfix_tail_kernel.grid(1, h, w))


def _leaves(
    graph: _Graph, name: str, ops: set[str], stop: frozenset[str] = frozenset()
) -> tuple[set[str], list[np.ndarray]]:
    """The tensors `name` is computed from through `ops` (stopping at `stop`), and the
    constants on the way."""
    tensors: set[str] = set()
    constants: list[np.ndarray] = []
    stack, seen = [name], set()
    while stack:
        current = stack.pop()
        if current in seen or not current:
            continue
        seen.add(current)
        if current in stop:
            tensors.add(current)
            continue
        value = graph.constant(current)
        if value is not None:
            constants.append(value)
            continue
        node = graph.producer.get(current)
        if node is not None and node.op_type == "Shape":
            # A static shape (every shape of these graphs is), not data
            continue
        if node is None or node.op_type not in ops:
            tensors.add(current)
            continue
        stack.extend(node.input)
    return tensors, constants


def _scalar(graph: _Graph, name: str) -> float | None:
    value = graph.constant(name)
    if value is None or value.size != 1:
        return None
    return float(value.ravel()[0])


def _operands(graph: _Graph, name: str, op: str) -> list[str] | None:
    """The inputs of the `op` node that makes `name`."""
    node = graph.producer.get(name)
    if node is None or node.op_type != op:
        return None
    return list(node.input)


def _pairs(graph: _Graph, name: str, op: str) -> list[tuple[str, str]]:
    """Both orders of the inputs of the commutative binary `op` that makes `name`."""
    inputs = _operands(graph, name, op)
    if inputs is None or len(inputs) != 2:
        return []
    return [(inputs[0], inputs[1]), (inputs[1], inputs[0])]


def _with_constant(
    graph: _Graph, name: str, op: str, constant_first: bool | None = None
) -> tuple[str, float] | None:
    """(the tensor operand, the scalar constant) of the binary `op` making `name` from a
    tensor and a constant; `constant_first` pins the constant's side (None: either, for
    the commutative operations)."""
    inputs = _operands(graph, name, op)
    if inputs is None or len(inputs) != 2:
        return None
    for first, (constant, other) in ((True, inputs), (False, inputs[::-1])):
        if constant_first is not None and constant_first != first:
            continue
        value = _scalar(graph, constant)
        if value is not None and graph.constant(other) is None:
            return other, value
    return None


def _sigmoid(graph: _Graph, name: str) -> str | None:
    inputs = _operands(graph, name, "Sigmoid")
    return None if inputs is None else inputs[0]


def _reduce_sum(graph: _Graph, name: str) -> str | None:
    """The input of a ReduceSum over axis 1 that drops the axis."""
    node = graph.producer.get(name)
    if node is None or node.op_type != "ReduceSum" or len(node.input) != 2:
        return None
    axes = graph.constant(node.input[1])
    keep = [
        helper.get_attribute_value(a) for a in node.attribute if a.name == "keepdims"
    ]
    if axes is None or [int(a) for a in axes.ravel()] != [1] or keep != [0]:
        return None
    return node.input[0]


def _reshaped(graph: _Graph, name: str, dims: list[int]) -> str | None:
    """The tensor of dims[1:] that a Reshape to `dims` (dims[0] == 1) makes `name` from."""
    inputs = _operands(graph, name, "Reshape")
    if inputs is None or graph.dims.get(name) != dims or dims[0] != 1:
        return None
    return inputs[0] if graph.dims.get(inputs[0]) == dims[1:] else None


def _sampling(graph: _Graph, name: str) -> onnx.NodeProto | None:
    """The GridSample making `name`, if bilinear with zero padding, align_corners=False."""
    node = graph.producer.get(name)
    if node is None or node.op_type != "GridSample":
        return None
    attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    if (
        attributes.get("mode", b"bilinear") != b"bilinear"
        or attributes.get("padding_mode", b"zeros") != b"zeros"
        or attributes.get("align_corners", 0) != 0
    ):
        return None
    return node


def _base(size: int) -> np.ndarray:
    """The stored base grid's values along an axis of `size` pixels, as the model makes
    them (temporalfix_arch._make_base_grid: FP32 arithmetic, the scale 2 / size rounded
    to FP32) and holds them (FP16), as the kernels compute them too. In FP64 one value of
    1920 or 3840 rounds to the next FP16."""
    one, half = np.float32(1.0), np.float32(0.5)
    positions = (np.arange(size, dtype=np.float32) + half) * np.float32(
        2.0 / size
    ) - one
    return positions.astype(np.float16)


def _position(graph: _Graph, name: str, channel: int, size: int) -> str | None:
    """The scaled flow behind one axis of fp32_warp's positions, which `name` makes as
    the stored grid's row (channel 0) or column (1) plus flow channel `channel` times
    2 / size."""
    shape = [1, 1, 1, size] if channel == 0 else [1, 1, size, 1]
    for base, scaled in _pairs(graph, name, "Add"):
        values = graph.constant(base)
        if values is None or list(values.shape) != shape:
            continue
        if not np.array_equal(values.ravel().astype(np.float32), _base(size)):
            return None
        product = _with_constant(graph, scaled, "Mul")
        if product is None or np.float32(product[1]) != np.float32(2.0 / size):
            return None
        piece = graph.span(product[0])
        if piece is None or piece[1:] != (channel, channel + 1):
            return None
        return graph.uncast(piece[0])
    return None


def _flow(
    graph: _Graph, scaled: str, h: int, w: int
) -> tuple[str, tuple[float, float]] | None:
    """(the half-size FP32 flow, its per-channel scales) that `scaled` brings up 2x and
    scales, as _resize_flow does, at a full size of H x W."""
    up = graph.scaled_upsample(scaled, 2)
    if up is None:
        return None
    half, scales = up
    if graph.dims.get(half, [])[1:] != [2, h // 2, w // 2]:
        return None
    if graph.types.get(half) != onnx.TensorProto.FLOAT:
        return None
    return half, (scales[0], scales[1])


def _positions(
    graph: _Graph, grid: str, h: int, w: int
) -> tuple[str, str, tuple[float, float]] | None:
    """(the scaled full-size flow, the half-size FP32 flow, its scales) behind a
    sampling grid made as fp32_warp makes it: Transpose(Concat(x, y)) with x and y the
    stored grid's row and column plus the scaled flow's channels times 2 / size."""
    node = graph.producer.get(grid)
    if node is None or node.op_type != "Transpose":
        return None
    perm = [helper.get_attribute_value(a) for a in node.attribute if a.name == "perm"]
    joined = graph.producer.get(node.input[0])
    if perm != [[0, 2, 3, 1]] or joined is None or joined.op_type != "Concat":
        return None
    if helper.get_attribute_value(joined.attribute[0]) != 1 or len(joined.input) != 2:
        return None
    x = _position(graph, joined.input[0], 0, w)
    y = _position(graph, joined.input[1], 1, h)
    if x is None or x != y:
        return None
    flow = _flow(graph, x, h, w)
    return None if flow is None else (x, *flow)


def _share(graph: _Graph, name: str, channel: int, size: int) -> str | None:
    """The scaled flow behind valid_share's factor along one axis, which `name` makes as
    clamp(min(pixel + 1, size - pixel), 0, 1), pixel = ((position + 1) * size - 1) / 2."""
    inputs = _operands(graph, name, "Clip")
    if inputs is None or len(inputs) != 3:
        return None
    if _scalar(graph, inputs[1]) != 0.0 or _scalar(graph, inputs[2]) != 1.0:
        return None
    pixel = None
    for plus, minus in _pairs(graph, inputs[0], "Min"):
        a = _with_constant(graph, plus, "Add")
        b = _with_constant(graph, minus, "Sub", constant_first=True)
        if a is not None and b is not None and a[0] == b[0]:
            if a[1] == 1.0 and b[1] == float(size):
                pixel = a[0]
    if pixel is None:
        return None
    halved = _with_constant(graph, pixel, "Div", constant_first=False)
    if halved is None or halved[1] != 2.0:
        return None
    less = _with_constant(graph, halved[0], "Sub", constant_first=False)
    if less is None or less[1] != 1.0:
        return None
    times = _with_constant(graph, less[0], "Mul")
    if times is None or times[1] != float(size):
        return None
    plus = _with_constant(graph, times[0], "Add")
    if plus is None or plus[1] != 1.0:
        return None
    return _position(graph, plus[0], channel, size)


def _validity(graph: _Graph, name: str, h: int, w: int) -> str | None:
    """The scaled flow behind the share of the positions inside the frame (the warp of
    a tensor of ones, in closed form) that `name` casts: share_x * share_y."""
    for x, y in _pairs(graph, graph.uncast(name), "Mul"):
        fx = _share(graph, x, 0, w)
        if fx is not None and fx == _share(graph, y, 1, h):
            return fx
    return None


def _frame_inputs(graph: _Graph, name: str) -> list[str] | None:
    """The window's frame inputs (1, 3, H, W) that `name` gathers from their
    concatenation on the channels (viewed as (1, frames, 3, H, W)), in its order, or
    None."""
    dims = graph.dims.get(name, [])
    node = graph.producer.get(name)
    if node is not None and node.op_type == "Reshape":
        node = graph.producer.get(node.input[0])
        if node is None or graph.dims.get(node.output[0]) != [1, *dims]:
            return None
    if node is None or node.op_type != "Gather" or len(dims) != 4:
        return None
    attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    indices = graph.constant(node.input[1])
    if attributes.get("axis", 0) != 1 or indices is None:
        return None
    stacked = graph.producer.get(node.input[0])
    if stacked is None or stacked.op_type != "Reshape":
        return None
    joined = graph.producer.get(stacked.input[0])
    if joined is None or joined.op_type != "Concat":
        return None
    frames = list(joined.input)
    view = [1, len(frames), 3, *dims[2:]]
    if helper.get_attribute_value(joined.attribute[0]) != 1:
        return None
    if graph.dims.get(stacked.output[0]) != view:
        return None
    if any(
        graph.dims.get(f) != [1, 3, *dims[2:]] or f in graph.producer for f in frames
    ):
        return None
    return [frames[int(i)] for i in np.ravel(indices)]


def _tail_warp(
    graph: _Graph, node: onnx.NodeProto
) -> tuple[list[str], tuple[float, float]] | None:
    """The plugin inputs (the six support frames, the window's inputs, and the
    half-size FP32 flow) and the flow's scales of the full-size support warp ending in
    `node`, the Cast of the FP32 sampling to FP16."""
    f16 = onnx.TensorProto.FLOAT16
    if node.op_type != "Cast" or graph.types.get(node.output[0]) != f16:
        return None
    sample = _sampling(graph, node.input[0])
    if sample is None:
        return None
    frames = graph.uncast(sample.input[0])
    dims = graph.dims.get(frames, [])
    if len(dims) != 4 or dims[1] != 3 or graph.types.get(frames) != f16:
        return None
    positions = _positions(graph, sample.input[1], dims[2], dims[3])
    supports = _frame_inputs(graph, frames)
    if positions is None or supports is None or len(supports) != dims[0]:
        return None
    return [*supports, positions[1]], positions[2]


def _tail_blend(
    graph: _Graph,
    node: onnx.NodeProto,
    warped: dict[str, tuple[str, tuple[float, float]]],
) -> tuple[list[str], tuple[float, ...]] | None:
    """The plugin inputs (FP16 centre, FP16 warped supports, FP16 half-size confidence
    logits, half-size FP32 flow) and BLEND_CONSTANTS of the temporal average whose
    result `node` makes, as conservative_temporal_average makes it:

        centre + ready * ((centre + sum(weight * support)) / clip(sum(weight) + one, eps)
            - centre)
        ready = sigmoid((sum(gate) - min_support + offset) * count_slope)
        weight = conf * gate, gate = sigmoid((conf - threshold) * gate_slope)

    with conf the sigmoid of the 2x upsampled logits times the validity share of the
    warp's positions. `warped` maps the tail warps' outputs to their flows and scales."""
    f16, f32 = onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT
    dims = graph.dims.get(node.output[0], [])
    if node.op_type != "Add" or len(dims) != 4 or dims[:2] != [1, 3]:
        return None
    if graph.types.get(node.output[0]) != f32:
        return None
    h, w = dims[2], dims[3]
    for centre32, moved in _pairs(graph, node.output[0], "Add"):
        centre = graph.uncast(centre32)
        if centre == centre32 or graph.dims.get(centre) != [1, 3, h, w]:
            continue
        if graph.types.get(centre) != f16:
            continue
        for ready, difference in _pairs(graph, moved, "Mul"):
            found = _average(graph, centre32, ready, difference, h, w, warped)
            if found is not None:
                inputs, constants = found
                return [
                    (_frame_inputs(graph, centre) or [centre])[0],
                    *inputs,
                ], constants
    return None


def _average(
    graph: _Graph,
    centre32: str,
    ready: str,
    difference: str,
    h: int,
    w: int,
    warped: dict[str, tuple[str, tuple[float, float]]],
) -> tuple[list[str], tuple[float, ...]] | None:
    """_tail_blend's parse below `ready * difference`: (support, logits, flow), and the
    constants."""
    f16 = onnx.TensorProto.FLOAT16
    squashed = _sigmoid(graph, ready)
    blend = _operands(graph, difference, "Sub")
    if squashed is None or blend is None or blend[1] != centre32:
        return None
    steep = _with_constant(graph, squashed, "Mul")
    if steep is None:
        return None
    shifted = _with_constant(graph, steep[0], "Add")
    if shifted is None:
        return None
    less = _with_constant(graph, shifted[0], "Sub", constant_first=False)
    if less is None:
        return None
    gates = _reduce_sum(graph, less[0])
    division = _operands(graph, blend[0], "Div")
    if gates is None:
        return None
    if division is None or len(division) != 2:
        return None
    clip = _operands(graph, division[1], "Clip")
    if clip is None or len(clip) not in (2, 3) or (len(clip) == 3 and clip[2]):
        return None
    eps = _scalar(graph, clip[1])
    total = _with_constant(graph, clip[0], "Add")
    if eps is None or total is None:
        return None
    weights = _reduce_sum(graph, total[0])
    if weights is None:
        return None
    supports32 = None
    for first, weighted in _pairs(graph, division[0], "Add"):
        summed = _reduce_sum(graph, weighted)
        if first != centre32 or summed is None:
            continue
        for factor, support in _pairs(graph, summed, "Mul"):
            if factor == weights:
                supports32 = support
    conf32 = None
    for conf, gate in _pairs(graph, weights, "Mul"):
        if gate == gates:
            conf32 = conf
    if supports32 is None or conf32 is None:
        return None
    sloped = _sigmoid(graph, gates)
    slope = None if sloped is None else _with_constant(graph, sloped, "Mul")
    if slope is None:
        return None
    threshold = _with_constant(graph, slope[0], "Sub", constant_first=False)
    if threshold is None or threshold[0] != conf32:
        return None
    # The supports and the confidence, (S, C, H, W) in FP16, viewed as (1, S, C, H, W)
    s = graph.dims.get(conf32, [0, 0])[1]
    support = _reshaped(graph, graph.uncast(supports32), [1, s, 3, h, w])
    conf = _reshaped(graph, graph.uncast(conf32), [1, s, 1, h, w])
    if support is None or support not in warped or conf is None:
        return None
    if supports32 == graph.uncast(supports32) or conf32 == graph.uncast(conf32):
        return None
    for logit, valid in _pairs(graph, conf, "Mul"):
        up = _sigmoid(graph, logit)
        logits = graph.bilinear(graph.producer.get(up), 2.0) if up else None
        scaled = _validity(graph, valid, h, w)
        if logits is None or scaled is None or graph.types.get(logits) != f16:
            continue
        if graph.dims.get(logits) != [s, 1, h // 2, w // 2]:
            continue
        flow = _flow(graph, scaled, h, w)
        if flow is None or flow != warped[support]:
            continue
        constants = (
            *flow[1],
            threshold[1],
            slope[1],
            less[1],
            shifted[1],
            steep[1],
            total[1],
            eps,
        )
        return [support, logits, flow[0]], constants
    return None


def block_input_name(
    s: int, h: int, w: int, q: int, constants: tuple[float, float]
) -> str:
    """The plugin of the last refinement block's input for S supports of H x W, the
    previous block's output in FP<Q> (16 or 32), its flow scaled by the graph's
    constants (WARP_CONSTANTS)."""
    return f"block_input_s{s}_h{h}_w{w}_q{q}_p{_key(constants)}"


def block_input_kernel(
    h: int, w: int, q: int, constants: tuple[float, ...], sm: int
) -> Any:
    """The last refinement block's input kernel for H x W, the previous block's output
    in FP<Q>, its flow scaled by the graph's constants (WARP_CONSTANTS), in its
    padded-rows form, compiled for the GPU architecture `sm` (as conf_input_kernel)."""
    return dual_aot.compile_kernel(
        tfix_block_input_kernel.block_input,
        {
            "centre_ptr": "*fp16",
            "fcentre_ptr": "*fp16",
            "supports_ptr": "*fp16",
            "flow_ptr": "*fp32",
            "previous_ptr": f"*fp{q}",
            "delta_ptr": "*fp16",
            "out_ptr": "*fp16",
        },
        {
            "height": h,
            "width": w,
            **dict(zip(WARP_CONSTANTS, constants, strict=True)),
            "padded_rows": True,
            **tfix_block_input_kernel.LAUNCH,
        },
        sm,
        tfix_block_input_kernel.OPTIONS,
    )


def direct_kernel(op: str, sm: int) -> tuple[Any, tuple[int, int, int]]:
    """The compiled kernel of a directly launched op (DIRECT: block_input_...,
    conf_input_...) for the GPU architecture `sm`, its sizes and constants read from
    its name as `register` reads a plugin's, and its launch grid."""
    name, _, key = op.partition("_p")
    sizes = [int(part[1:]) for part in name.split("_")[2:]]
    s, h, w = sizes[:3]
    if name.startswith("conf_input"):
        kernel = conf_input_kernel(h, w, sizes[3], sm)
        return kernel, tfix_conf_input_kernel.grid(s, h, w)
    kernel = block_input_kernel(h, w, sizes[3], _unkey(key), sm)
    return kernel, tfix_block_input_kernel.grid(s, h, w)


_BROADCAST_OPS = {"Reshape", "Expand", "Unsqueeze"}


def _broadcast_source(graph: _Graph, name: str, shape: list[int]) -> str | None:
    """The FP16 tensor of `shape` that `name` repeats over the supports."""
    tensors, _ = _leaves(graph, name, _BROADCAST_OPS)
    found = [t for t in tensors if graph.dims.get(t) == shape]
    if len(found) != 1 or graph.types.get(found[0]) != onnx.TensorProto.FLOAT16:
        return None
    return found[0]


def _previous(
    graph: _Graph, name: str, first: int, end: int, h: int, w: int
) -> tuple[str, int] | None:
    """(the previous block's output (S, 11, H/2, W/2), its bits) whose channels
    first..end `name` upsamples 2x in FP16: an FP16 output's slice as it is, an FP32
    output's (the first two stages in FP32) after an FP16 cast."""
    small = graph.bilinear(graph.producer.get(name), 2.0)
    if small is None or graph.types.get(small) != onnx.TensorProto.FLOAT16:
        return None
    cast = graph.producer.get(small)
    bits, sliced = 16, small
    if cast is not None and _to_half(cast):
        bits, sliced = 32, cast.input[0]
    piece = graph.span(sliced)
    if piece is None or piece[1:] != (first, end):
        return None
    source = piece[0]
    if graph.dims.get(source, [])[1:] != [11, h // 2, w // 2]:
        return None
    types = {16: onnx.TensorProto.FLOAT16, 32: onnx.TensorProto.FLOAT}
    return (source, bits) if graph.types.get(source) == types[bits] else None


def _to_half(node: onnx.NodeProto) -> bool:
    """Whether `node` is a Cast to FP16."""
    to = [helper.get_attribute_value(a) for a in node.attribute if a.name == "to"]
    return node.op_type == "Cast" and to == [onnx.TensorProto.FLOAT16]


def _block_input(
    graph: _Graph, node: onnx.NodeProto
) -> tuple[list[str], tuple[float, float], int] | None:
    """The plugin inputs (FP16 centre frame and features, FP16 support frames and
    features (S, 7), FP32 quarter-size flow, the previous block's output, FP16 time
    offsets), the flow's scales and the previous output's bits of the last refinement
    block's input Concat:
    (centre, warped support, centre features, warped features, time offset, logit,
    latent) and the flow, the warped parts FP16 casts of one sampling along the flow."""
    f16 = onnx.TensorProto.FLOAT16
    dims = graph.dims.get(node.output[0], [])
    if node.op_type != "Concat" or len(node.input) != 2 or len(dims) != 4:
        return None
    s, channels, h, w = dims
    if channels != 26 or graph.types.get(node.output[0]) != f16:
        return None
    body = graph.producer.get(graph.uncast(node.input[0]))
    if body is None or body.op_type != "Concat" or len(body.input) != 7:
        return None
    scaled = graph.uncast(node.input[1])
    flow = _flow(graph, scaled, h, w)
    if flow is None or scaled == node.input[1]:
        return None
    parts = list(body.input)
    sample = None
    for index, span in ((1, (0, 3)), (3, (3, 7))):
        piece = graph.span(parts[index])
        cast = graph.producer.get(piece[0]) if piece is not None else None
        if piece is None or piece[1:] != span or cast is None or not _to_half(cast):
            return None
        grid_sample = _sampling(graph, cast.input[0])
        if grid_sample is None or sample not in (None, grid_sample):
            return None
        sample = grid_sample
    assert sample is not None
    supports = graph.uncast(sample.input[0])
    if graph.dims.get(supports) != [s, 7, h, w] or graph.types.get(supports) != f16:
        return None
    positions = _positions(graph, sample.input[1], h, w)
    if positions is None or positions[0] != scaled:
        return None
    centre = _broadcast_source(graph, parts[0], [1, 3, h, w])
    fcentre = _broadcast_source(graph, parts[2], [1, 4, h, w])
    delta = _broadcast_source(graph, parts[4], [s, 1, 1, 1])
    logit = _previous(graph, parts[5], 2, 3, h, w)
    latent = _previous(graph, parts[6], 3, 11, h, w)
    if centre is None or fcentre is None or delta is None or logit is None:
        return None
    if latent != logit:
        return None
    return [centre, fcentre, supports, flow[0], logit[0], delta], flow[1], logit[1]


def lower(model: onnx.ModelProto) -> dual_aot.Lowered:
    """The model with each round trip's downsample replaced by a stand-in of its output
    and the round-trip plugins to attach in their place, each input with a named
    reader. The plugins are registered by `register` with the lowered model's plugins."""
    graph = _Graph(model)
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []
    plugins: list[dual_aot.Plugin] = []
    warped: dict[str, tuple[str, tuple[float, float]]] = {}
    # The last refinement block's input only (the finest stage's 26-channel Concat):
    # with every stage in FP16 the second one's input has the same form, and the set's
    # cut is made at the last one; a last one that does not match leaves none
    concats = [
        node.output[0]
        for node in model.graph.node
        if node.op_type == "Concat"
        and graph.dims.get(node.output[0], [0, 0])[1:2] == [26]
    ]
    finest = max(concats, key=lambda name: graph.dims[name][2], default=None)
    for node in model.graph.node:
        block = _block_input(graph, node) if node.output[0] == finest else None
        if block is not None:
            s_, _, h_, w_ = graph.dims[node.output[0]]
            nodes.append(node)
            name = block_input_name(s_, h_, w_, block[2], block[1])
            plugins.append(dual_aot.Plugin(name, block[0], list(node.output)))
            continue
        warp = _tail_warp(graph, node)
        if warp is not None:
            s_, _, h_, w_ = graph.dims[node.output[0]]
            nodes.append(node)
            warped[node.output[0]] = (warp[0][-1], warp[1])
            name = tail_name("warp", s_, h_, w_, warp[1])
            plugins.append(dual_aot.Plugin(name, warp[0], list(node.output)))
            continue
        blend = _tail_blend(graph, node, warped)
        if blend is not None:
            _, _, h_, w_ = graph.dims[node.output[0]]
            nodes.append(node)
            s_ = graph.dims[blend[0][1]][0]
            name = tail_name("blend", s_, h_, w_, blend[1])
            plugins.append(dual_aot.Plugin(name, blend[0], list(node.output)))
            continue
        conf_inputs = _conf_input(graph, node)
        if conf_inputs is not None:
            s_, _, h_, w_ = graph.dims[node.input[0]]
            nodes.append(node)
            g = 32 if graph.types[conf_inputs[4]] == onnx.TensorProto.FLOAT else 16
            name = conf_input_name(s_, h_, w_, g)
            plugins.append(dual_aot.Plugin(name, conf_inputs, list(node.output)))
            continue
        found = None
        source = graph.bilinear(node, 0.5)
        conv = graph.producer.get(source) if source is not None else None
        if conv is not None and conv.op_type == "Conv":
            n, c, h, w = graph.dims.get(node.output[0], [0, 0, 0, 0])
            k = graph.separable(conv, c)
            up = graph.scaled_upsample(conv.input[0], c) if k is not None else None
            if (
                k is not None
                and up is not None
                and graph.dims.get(up[0]) == [n, c, h, w]
                and graph.types.get(up[0]) in BITS
            ):
                found = (n, c, h, w, k, up)
        if found is None:
            nodes.append(node)
            continue
        n, c, h, w, k, (half, scales) = found
        ih, ch = tfix_round_trip_kernel.taps(h, k)
        iw, cw = tfix_round_trip_kernel.taps(w, k)
        taps = max(ih.shape[1], iw.shape[1])
        tables = []
        for label, short_index, short_weight in (("h", ih, ch), ("w", iw, cw)):
            pad = taps - short_index.shape[1]
            last = np.repeat(short_index[:, -1:], pad, 1)
            index = np.concatenate([short_index, last], 1)
            weight = np.pad(short_weight, ((0, 0), (0, pad)))
            for kind, values in (("index", index), ("weight", weight)):
                name = f"{PREFIX}/{node.output[0]}/{kind}_{label}"
                initializers.append(numpy_helper.from_array(values, name))
                tables.append(name)
        scale = f"{PREFIX}/{node.output[0]}/scale"
        initializers.append(
            numpy_helper.from_array(np.array(scales, np.float32), scale)
        )
        nodes.append(helper.make_node("Identity", [half], list(node.output)))
        inputs = [half, tables[0], tables[1], tables[2], tables[3], scale]
        name = round_trip_name(n, c, h, w, taps, BITS[graph.types[half]])
        plugins.append(dual_aot.Plugin(name, inputs, list(node.output)))
    clean = onnx.ModelProto()
    clean.CopyFrom(model)
    del clean.graph.node[:]
    clean.graph.node.extend(nodes)
    clean.graph.initializer.extend(initializers)
    readers = {}
    for index, name in enumerate(sorted({i for p in plugins for i in p.inputs})):
        reader = f"{PREFIX}/reader_{index}"
        clean.graph.node.append(
            helper.make_node("Identity", [name], [reader], name=reader)
        )
        readers[reader] = name
    return dual_aot.Lowered(clean, plugins, readers)


def register(lowered: dual_aot.Lowered) -> None:
    """Register the lowered model's round-trip plugins (once per geometry); needs the
    building GPU, whose architecture the kernels are compiled for."""
    for plugin in lowered.plugins:
        op, _, key = plugin.op.partition("_p")
        sizes = [int(part[1:]) for part in op.split("_")[2:]]
        if op.startswith(DIRECT):
            # launched between the window's pieces, not inside an engine (tfix_cut.py)
            continue
        if op.startswith("tail_"):
            register_tail(op.split("_")[1], *sizes, _unkey(key))
        else:
            register_round_trip(*sizes)


__all__ = [
    "DIRECT",
    "block_input_kernel",
    "block_input_name",
    "conf_input_kernel",
    "conf_input_name",
    "direct_kernel",
    "lower",
    "register",
    "round_trip_name",
    "tail_name",
]
