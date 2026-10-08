"""Factory/shape-specialized ONNX graph for canonical DUAL.

Adapted from the historical explicit NHWC exporter, without legacy architecture
imports. Default geometry only: dot attention, RFB16, G16/C32/C64, K4, F64.
The ONNX is TensorRT-specific, not a generic ONNX Runtime model.
"""

import math
import numpy as np
import torch
from onnx import TensorProto as T, helper as H, numpy_helper as N


def phase_weight(w: np.ndarray, scale: int, grandchildren: bool = False) -> np.ndarray:
    """Move spatial convolution before a shuffle, for either 3x3 or 5x5.

    XG's incoming slots are head/phase/value, unlike standard CRD PixelShuffle.
    Both radii map to the same 3x3 low-resolution support for scales2 and4.
    """
    co, ci, kh, kw = w.shape
    radius = kh // 2
    assert kh == kw and kh in (3, 5)
    result = np.zeros((co * scale**2, ci * scale**2, 3, 3), dtype=w.dtype)
    for out in range(co):
        for a in range(scale):
            for b in range(scale):
                for channel in range(ci):
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            yy, xx = a + dy, b + dx
                            phase = (yy % scale) * scale + xx % scale
                            src = (
                                ((channel // 2) * scale**2 + phase) * 2 + channel % 2
                                if grandchildren
                                else channel * scale**2 + phase
                            )
                            result[
                                out * scale**2 + a * scale + b,
                                src,
                                yy // scale + 1,
                                xx // scale + 1,
                            ] = w[out, channel, dy + radius, dx + radius]
    return result


class Graph:
    def __init__(self, arch, spec):
        self.arch, self.spec = arch, spec
        channels = spec.channels
        self.height, self.width = spec.trunk
        self.dynamic_sites = 0
        self.c, self.nodes, self.constants = channels, [], []
        self.tag, self.counter = "", 0
        self.mha_count = 0

    def name(self, label):
        self.counter += 1
        return f"/{self.tag}/{label}_{self.counter}"

    def const(self, value, dtype=None):
        name = self.name("const")
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        if dtype == T.BFLOAT16:
            v = torch.as_tensor(value).to(torch.bfloat16).contiguous()
            item = H.make_tensor(
                name,
                T.BFLOAT16,
                list(v.shape),
                v.view(torch.uint16).numpy().tobytes(),
                raw=True,
            )
        else:
            if isinstance(value, torch.Tensor):
                value = value.numpy()
            v = np.asarray(value)
            if dtype == T.FLOAT:
                v = v.astype(np.float32)
            if dtype == T.INT64:
                v = v.astype(np.int64)
            item = N.from_array(v, name)
        self.constants.append(item)
        return name

    def op(self, kind, inputs, **attrs):
        name = self.name(kind)
        self.nodes.append(H.make_node(kind, inputs, [name], name=name, **attrs))
        return name

    def plugin(self, kind, inputs, outputs=1, **attrs):
        name = self.name(kind)
        names = [name] + [name + f"_unused_{i}" for i in range(1, outputs)]
        self.nodes.append(
            H.make_node(
                kind,
                inputs,
                names,
                name=name,
                domain="trt",
                plugin_version="1",
                plugin_namespace="",
                **attrs,
            )
        )
        return name

    def reshape(self, x, shape):
        return self.op("Reshape", [x, self.const(shape, T.INT64)])

    def trans(self, x, perm):
        return self.op("Transpose", [x], perm=perm)

    def cast(self, x, dtype):
        return self.op("Cast", [x], to=dtype)

    def slice(self, x, starts, ends, axes):
        return self.op(
            "Slice",
            [
                x,
                self.const(starts, T.INT64),
                self.const(ends, T.INT64),
                self.const(axes, T.INT64),
            ],
        )

    def pw(self, x, mod):
        w = mod.weight.detach()
        if w.ndim == 4:
            w = w[:, :, 0, 0]
        x = self.op("MatMul", [x, self.const(w.T, T.BFLOAT16)])
        return (
            x
            if mod.bias is None
            else self.op("Add", [x, self.const(mod.bias, T.BFLOAT16)])
        )

    def norm(self, x, mod):
        name = f"DualNorm_{self.spec.plugin_key}_TRT"
        return self.plugin(
            name,
            [
                x,
                self.const(mod.norm.weight, T.FLOAT),
                self.const(mod.norm.bias, T.FLOAT),
            ],
            outputs=2,
            residual=0,
            epsilon=mod.norm.eps,
        )

    def conv(self, x, mod, nhwc=False, weight=None, bias=None):
        if isinstance(mod, self.arch.PremixedDeployedDynamicConv3x3):
            if weight is not None or bias is not None:
                raise ValueError(
                    "Dynamic convolution cannot receive a frozen replacement"
                )
            return self.dynamic_conv(x, mod, nhwc)
        if nhwc:
            x = self.trans(x, [0, 3, 1, 2])
        w = mod.weight if weight is None else weight
        b = mod.bias if weight is None else bias
        inputs = [x, self.const(w, T.BFLOAT16)]
        if b is not None:
            inputs.append(self.const(b, T.BFLOAT16))
        x = self.op(
            "Conv", inputs, pads=[1, 1, 1, 1], strides=[1, 1], kernel_shape=[3, 3]
        )
        return self.trans(x, [0, 2, 3, 1]) if nhwc else x

    def gelu(self, x, dtype=T.BFLOAT16):
        z = self.op("Div", [x, self.const(np.sqrt(2), dtype)])
        z = self.op("Add", [self.op("Erf", [z]), self.const(1.0, dtype)])
        return self.op("Mul", [self.op("Mul", [x, self.const(0.5, dtype)]), z])

    def dw(self, x, mod, size, epilogue=False):
        # Same native FP32 shifted-sum implementation as GRAFTV1Deploy.
        h, w = size
        v = self.cast(x, T.FLOAT)
        padded = self.op("Pad", [v, self.const([0, 1, 1, 0, 0, 1, 1, 0], T.INT64)])
        y = None
        weight = mod.weight.detach()[:, 0].permute(1, 2, 0)
        for iy in range(3):
            for ix in range(3):
                term = self.slice(padded, [iy, ix], [iy + h, ix + w], [1, 2])
                term = self.op("Mul", [term, self.const(weight[iy, ix], T.FLOAT)])
                y = term if y is None else self.op("Add", [y, term])
        y = self.op("Add", [y, self.const(mod.bias, T.FLOAT)])
        if epilogue:
            y = self.op("Add", [v, self.gelu(y, T.FLOAT)])
        return self.cast(y, T.BFLOAT16)

    def mha(self, q, k, v, bias=None, barrier=False):
        if barrier:
            q = self.plugin("DualAttentionBarrier_TRT", [q])
        score = self.op("MatMul", [q, self.trans(k, [0, 1, 3, 2])])
        if bias is not None:
            score = self.op("Add", [score, self.const(bias, T.BFLOAT16)])
        self.mha_count += 1
        return self.op("MatMul", [self.op("Softmax", [score], axis=-1), v])

    def attention(self, x, mod, size):
        h, w = size
        c, heads = mod.channels, mod.heads
        rows, cols = (
            self.arch.anchored_axis(h, mod.window, mod.offset),
            self.arch.anchored_axis(w, mod.window, mod.offset),
        )
        nr, wh = rows.indices.shape
        nc, ww = cols.indices.shape
        regular = mod.offset == 0 and h % wh == 0 and w % ww == 0
        image = x
        if not regular:
            image = self.op(
                "Gather", [image, self.const(rows.indices.flatten(), T.INT64)], axis=1
            )
            image = self.op(
                "Gather", [image, self.const(cols.indices.flatten(), T.INT64)], axis=2
            )
        tokens = self.reshape(
            self.trans(self.reshape(image, [1, nr, wh, nc, ww, c]), [0, 1, 3, 2, 4, 5]),
            [nr * nc, wh * ww, c],
        )
        qkv = self.trans(
            self.reshape(self.pw(tokens, mod.qkv), [nr * nc, wh * ww, 3, heads, 32]),
            [2, 0, 3, 1, 4],
        )
        q, k, v = [
            self.op("Gather", [qkv, self.const(i, T.INT64)], axis=0) for i in range(3)
        ]
        q = self.op("Mul", [q, self.const(32**-0.5, T.BFLOAT16)])
        bias = None
        if mod.factored_bias is not None:
            bq, bk = mod.factored_bias(wh, ww, torch.float32)
            for label, factor in (("q", bq), ("k", bk)):
                ext = self.op(
                    "Expand",
                    [
                        self.const(factor[None], T.BFLOAT16),
                        self.const([nr * nc, heads, wh * ww, 16], T.INT64),
                    ],
                )
                if label == "q":
                    q = self.op("Concat", [q, ext], axis=-1)
                else:
                    k = self.op("Concat", [k, ext], axis=-1)
            v = self.op(
                "Pad",
                [
                    v,
                    self.const([0, 0, 0, 0, 0, 0, 0, 16], T.INT64),
                    self.const(0.0, T.BFLOAT16),
                ],
            )
        else:
            bias = mod.position_bias(wh, ww, torch.float32)
        out = self.mha(q, k, v, bias)
        if mod.factored_bias is not None:
            out = self.slice(out, [0], [32], [3])
        out = self.reshape(self.trans(out, [0, 2, 1, 3]), [1, nr, nc, wh, ww, c])
        out = self.reshape(
            self.trans(out, [0, 1, 3, 2, 4, 5]), [1, nr * wh, nc * ww, c]
        )
        if not regular:
            out = self.op("Gather", [out, self.const(rows.restore, T.INT64)], axis=1)
            out = self.op("Gather", [out, self.const(cols.restore, T.INT64)], axis=2)
        gate = self.op(
            "Sigmoid",
            [self.pw(self.dw(x, mod.gate_depthwise, size), mod.gate_pointwise)],
        )
        return self.pw(self.op("Mul", [out, gate]), mod.project)

    def layer(self, x, mod, size):
        x = self.op(
            "Add", [x, self.attention(self.norm(x, mod.norm1), mod.attention, size)]
        )
        if hasattr(mod, "dual"):
            dual = mod.dual
            qkv = self.pw(self.norm(x, dual.norm), dual.qkv)
            core = f"DualCore_{self.spec.plugin_key}_TRT"
            project = f"DualProject_{self.spec.plugin_key}_TRT"
            out = self.plugin(
                core,
                [
                    qkv,
                    self.const(
                        dual.depthwise.weight[:, 0].permute(1, 2, 0), T.BFLOAT16
                    ),
                    self.const(dual.theta.clamp(max=math.log(100)).exp(), T.FLOAT),
                ],
            )
            x = self.plugin(
                project,
                [out, self.const(dual.project.weight[..., 0, 0].T, T.BFLOAT16), x],
            )
        hidden = self.gelu(self.pw(self.norm(x, mod.norm2), mod.ffn.expand))
        return self.op(
            "Add",
            [
                x,
                self.pw(
                    self.dw(hidden, mod.ffn.depthwise, size, True), mod.ffn.project
                ),
            ],
        )

    def group(self, x, mod, index):
        residual = x
        for i, layer in enumerate(mod.layers):
            self.tag = f"group{index}/layer{i}"
            x = self.layer(x, layer, (self.height, self.width))
        self.tag = f"group{index}/conv"
        return self.op("Add", [residual, self.conv(x, mod.conv, True)])

    def pack_slots(self, x, scale, h, w):
        heads, d = self.c // 32, 32 // scale**2
        x = self.reshape(x, [1, h // scale, scale, w // scale, scale, heads, d])
        return self.reshape(
            self.trans(x, [0, 1, 3, 5, 2, 4, 6]), [1, h // scale, w // scale, self.c]
        )

    def unpack_slots(self, x, scale, h, w):
        x = self.reshape(x, [1, h, w, self.c // 32, scale, scale, 32 // scale**2])
        return self.reshape(
            self.trans(x, [0, 1, 4, 2, 5, 3, 6]),
            [1, h * scale, w * scale, self.c // scale**2],
        )

    def pack_block(self, x, block):
        y0, y1, x0, x1, wh, ww = block
        nr, nc = (y1 - y0) // wh, (x1 - x0) // ww
        x = self.slice(x, [y0, x0], [y1, x1], [1, 2])
        x = self.reshape(x, [1, nr, wh, nc, ww, self.c])
        return self.reshape(
            self.trans(x, [0, 1, 3, 2, 4, 5]), [nr * nc, wh * ww, self.c]
        )

    def retrieval_descriptors(self, q, k):
        layout = self.arch.block_layout(self.height, self.width, 96)
        classes = []
        for _, items in self.arch.grouped_layout(layout):
            qs = [self.pack_block(q, block) for _, block in items]
            ks = [
                self.pack_block(k, tuple(t // 2 for t in block)) for _, block in items
            ]
            wh, ww = items[0][1][-2:]
            n = wh * ww
            count = sum((b[1] - b[0]) // wh * ((b[3] - b[2]) // ww) for _, b in items)
            qq, kk = [
                self.trans(
                    self.reshape(
                        self.op("Concat", z, axis=0), [count, length, self.c // 32, 32]
                    ),
                    [0, 2, 1, 3],
                )
                for z, length in ((qs, n), (ks, n // 4))
            ]
            classes.append((items, qq, kk, n, count))
        return layout, classes

    def retrieve(self, packed, v, log_tau):
        layout, classes = packed
        outputs = [None] * len(layout)
        for items, q, k, n, count in classes:
            vs = [
                self.pack_block(v, tuple(t // 2 for t in block)) for _, block in items
            ]
            vv = self.trans(
                self.reshape(
                    self.op("Concat", vs, axis=0), [count, n // 4, self.c // 32, 32]
                ),
                [0, 2, 1, 3],
            )
            q = self.op(
                "Mul",
                [
                    q,
                    self.const(
                        log_tau.exp().clamp(0.1, 10).reshape(1, -1, 1, 1), T.BFLOAT16
                    ),
                ],
            )
            q = self.op(
                "Mul",
                [
                    q,
                    self.const(
                        math.log(n // 4) / (math.log(1024) * math.sqrt(32)), T.BFLOAT16
                    ),
                ],
            )
            out = self.reshape(
                self.trans(self.mha(q, k, vv, barrier=True), [0, 2, 1, 3]),
                [count, n, self.c],
            )
            start = 0
            for index, block in items:
                y0, y1, x0, x1, wh, ww = block
                nr, nc = (y1 - y0) // wh, (x1 - x0) // ww
                z = self.slice(out, [start], [start + nr * nc], [0])
                z = self.reshape(z, [1, nr, nc, wh, ww, self.c])
                outputs[index] = self.reshape(
                    self.trans(z, [0, 1, 3, 2, 4, 5]), [1, y1 - y0, x1 - x0, self.c]
                )
                start += nr * nc
        rows = []
        for y in sorted({b[0] for b in layout}):
            row = [outputs[i] for i, b in enumerate(layout) if b[0] == y]
            rows.append(self.op("Concat", row, axis=2))
        return self.op("Concat", rows, axis=1)

    def downsample(self, shifted, model):
        taps = model.xr.down_taps.to(torch.bfloat16).float()
        x = self.cast(self.cast(shifted, T.BFLOAT16), T.FLOAT)
        for pads, weight, stride in (
            (
                [0, 0, 0, 3, 0, 0, 0, 3],
                taps.reshape(1, 1, 1, 8).expand(3, -1, -1, -1),
                [1, 2],
            ),
            (
                [0, 0, 3, 0, 0, 0, 3, 0],
                taps.reshape(1, 1, 8, 1).expand(3, -1, -1, -1),
                [2, 1],
            ),
        ):
            x = self.op("Pad", [x, self.const(pads, T.INT64)], mode="edge")
            x = self.op(
                "Conv", [x, self.const(weight, T.FLOAT)], group=3, strides=stride
            )
            x = self.cast(self.cast(x, T.BFLOAT16), T.FLOAT)
        return self.cast(x, T.BFLOAT16)

    def dynamic_conv(self, x, mod, nhwc):
        self.dynamic_sites += 1
        if nhwc:
            x = self.trans(x, [0, 3, 1, 2])
        k = mod.experts.shape[0]
        pooled = self.op(
            "ReduceMean",
            [self.cast(x, T.FLOAT), self.const([2, 3], T.INT64)],
            keepdims=0,
        )
        normalized = self.op(
            "LayerNormalization",
            [
                pooled,
                self.const(np.ones(self.c), T.FLOAT),
                self.const(np.zeros(self.c), T.FLOAT),
            ],
            axis=-1,
            epsilon=1e-5,
        )

        def linear(z, layer):
            return self.op(
                "Add",
                [
                    self.op("MatMul", [z, self.const(layer.weight.T, T.FLOAT)]),
                    self.const(layer.bias, T.FLOAT),
                ],
            )

        logits = linear(
            self.gelu(linear(normalized, mod.router.fc1), T.FLOAT), mod.router.fc2
        )
        coefficients = self.op(
            "Sub", [self.op("Softmax", [logits], axis=-1), self.const(1.0 / k, T.FLOAT)]
        )
        residual = self.op(
            "MatMul", [coefficients, self.const(mod.experts.reshape(k, -1), T.FLOAT)]
        )
        weight = self.op(
            "Add",
            [
                self.const(mod.base.weight, T.FLOAT),
                self.reshape(residual, [self.c, self.c, 3, 3]),
            ],
        )
        output = self.op(
            "Conv",
            [x, self.cast(weight, T.BFLOAT16), self.const(mod.base.bias, T.BFLOAT16)],
            pads=[1, 1, 1, 1],
            kernel_shape=[3, 3],
        )
        return self.trans(output, [0, 2, 3, 1]) if nhwc else output

    def shuffle(self, x, scale):
        return self.op("DepthToSpace", [x], blocksize=scale, mode="CRD")

    def unshuffle_rgb(self, x):
        # ONNX SpaceToDepth's order differs from PyTorch PixelUnshuffle.
        h, w = self.spec.height, self.spec.width
        return self.unshuffle_nchw(x, 3, h, w, 2)

    def unshuffle_nchw(self, x, channels, h, w, scale):
        return self.reshape(
            self.trans(
                self.reshape(x, [1, channels, h // scale, scale, w // scale, scale]),
                [0, 1, 3, 5, 2, 4],
            ),
            [1, channels * scale * scale, h // scale, w // scale],
        )

    def tail_sequence(self, x, layers):
        layers = list(layers)
        index = 0
        while index < len(layers):
            module = layers[index]
            if isinstance(module, torch.nn.Conv2d):
                x = self.conv(x, module)
            elif isinstance(module, torch.nn.LeakyReLU):
                x = self.op("LeakyRelu", [x], alpha=module.negative_slope)
            elif isinstance(module, torch.nn.PixelShuffle):
                # Move a following RGB/readout 3x3 before the final shuffle. Only
                # the readout: a moved 3x3 does scale**2 times the dense work.
                if index + 2 == len(layers) and isinstance(
                    layers[index + 1], torch.nn.Conv2d
                ):
                    next_conv = layers[index + 1]
                    weight = phase_weight(
                        next_conv.weight.detach().numpy(), module.upscale_factor
                    )
                    bias = (
                        None
                        if next_conv.bias is None
                        else np.repeat(
                            next_conv.bias.detach().numpy(), module.upscale_factor**2
                        )
                    )
                    x = self.conv(x, next_conv, weight=weight, bias=bias)
                    x = self.shuffle(x, module.upscale_factor)
                    index += 1
                else:
                    x = self.shuffle(x, module.upscale_factor)
            else:
                raise TypeError(f"Unexpected canonical tail module: {type(module)}")
            index += 1
        return x

    def build(self, model):
        h0, w0 = self.height, self.width
        self.tag = "stem"
        mean = self.const(model.mean, T.FLOAT)
        shifted = self.op("Sub", ["input", mean])
        stem_input = self.unshuffle_rgb(shifted) if model.unshuffle else shifted
        stem = self.trans(
            self.conv(self.cast(stem_input, T.BFLOAT16), model.stem), [0, 2, 3, 1]
        )
        h, fine = stem, None
        for index, layer in enumerate(model.groups[0].layers):
            self.tag = f"group0/layer{index}"
            h = self.layer(h, layer, (h0, w0))
            if index == 1:
                fine = h
        self.tag = "group0/conv"
        h = self.op("Add", [stem, self.conv(h, model.groups[0].conv, True)])
        h = self.group(h, model.groups[1], 1)
        self.tag = "coarse"
        coarse_input = self.downsample(shifted, model)
        if model.unshuffle:
            coarse_input = self.unshuffle_nchw(
                coarse_input, 3, self.spec.height // 2, self.spec.width // 2, 2
            )
        coarse = self.trans(self.conv(coarse_input, model.stem), [0, 2, 3, 1])
        for index, layer in enumerate(model.groups[0].layers[:2]):
            self.tag = f"coarse/layer{index}"
            coarse = self.layer(coarse, layer, (h0 // 2, w0 // 2))
        self.tag = "xr"
        u = self.norm(h, model.xr.norm_u)
        q = self.pw(
            self.op("Concat", [self.norm(fine, model.xr.norm_f), u], axis=-1),
            model.xr.query,
        )
        k = self.pw(self.norm(coarse, model.xr.norm_f), model.xr.key)
        packed = self.retrieval_descriptors(q, k)
        o = self.retrieve(
            packed,
            self.pack_slots(self.pw(u, model.xr.value), 2, h0, w0),
            model.xr.log_tau,
        )
        lr = self.pw(
            self.op("Mul", [o, self.op("Sigmoid", [self.pw(u, model.xr.gate)])]),
            model.xr.project,
        )
        h = self.op("Add", [h, lr])
        for index, group in enumerate(model.groups[2:], 2):
            h = self.group(h, group, index)
        self.tag = "tail"
        features = self.op(
            "Add",
            [stem, self.conv(self.norm(h, model.final_norm), model.trunk_end, True)],
        )
        features = self.trans(features, [0, 3, 1, 2])
        if model.internal_scale == 1:
            output = self.tail_sequence(features, model.tail)
        else:
            hr = self.trans(
                self.pw(self.unpack_slots(o, 2, h0, w0), model.xr.hr), [0, 3, 1, 2]
            )
            t2 = self.op("Add", [self.tail_sequence(features, model.tail[:4]), hr])
            output = self.tail_sequence(t2, model.tail[4:])
            if model.internal_scale >= 4:
                self.tag = "xg"
                g = self.pw(self.trans(t2, [0, 2, 3, 1]), model.xg.value)
                second = self.retrieve(
                    packed, self.pack_slots(g, 4, h0 * 2, w0 * 2), model.xg.log_tau
                )
                weight = phase_weight(model.xg.out.weight.detach().numpy(), 4, True)
                extra = self.conv(
                    self.trans(second, [0, 3, 1, 2]), model.xg.out, weight=weight
                )
                extra = self.shuffle(extra, 4)
                if model.internal_scale == 8:
                    extra = self.shuffle(extra, 2)
                output = self.op("Add", [output, extra])
        self.tag = "output"
        output = self.cast(output, T.FLOAT)
        skip = (
            shifted
            if model.scale == 1
            else self.op(
                "Resize",
                [
                    shifted,
                    self.const([], T.FLOAT),
                    self.const([1, 1, model.scale, model.scale], T.FLOAT),
                ],
                mode="nearest",
                coordinate_transformation_mode="asymmetric",
                nearest_mode="floor",
            )
        )
        output = self.op("Add", [self.op("Add", [output, skip]), mean])
        self.nodes.append(H.make_node("Identity", [output], ["output"], name="output"))
        if self.dynamic_sites != len(model.groups):
            raise ValueError("Missing dynamic group convolution(s)")
        return H.make_model(
            H.make_graph(
                self.nodes,
                "DUAL_SPECIALIZED",
                [
                    H.make_tensor_value_info(
                        "input", T.FLOAT, [1, 3, self.spec.height, self.spec.width]
                    )
                ],
                [
                    H.make_tensor_value_info(
                        "output",
                        T.FLOAT,
                        [
                            1,
                            3,
                            self.spec.height * model.scale,
                            self.spec.width * model.scale,
                        ],
                    )
                ],
                self.constants,
            ),
            opset_imports=[H.make_opsetid("", 20), H.make_opsetid("trt", 1)],
            ir_version=9,
        )
