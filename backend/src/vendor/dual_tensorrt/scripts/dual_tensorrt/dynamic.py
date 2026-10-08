"""chaiNNer-C addition (not traiNNer's): DUAL's TensorRT ONNX for dynamic-shape engines,
one graph for every input size. CPU-only; run in an isolated worker like export.py:

    python -m scripts.dual_tensorrt.dynamic --factory F --scale S --checkpoint CKPT
        [--folded] --output DIR

`DynamicGraph` overrides only graph.py's size-dependent parts, computing them from
Shape at run time:
- anchored windows: count `L <= w ? 1 : ceil((L + o) / w)`, extent `min(L, w)`, starts
  clamped to [0, L - extent], owner `min((p + o) // w, count - 1)`; a cropped window's
  position bias (RPB table or RFB factors) is gathered from the nominal window's at
  y * w + x (both depend on window coordinates only);
- retrieval: per axis, 96-px blocks and a leftover strip, so four window classes; an
  absent strip is zero windows of a nonzero extent (Myelin rejects zero extents);
- depthwise shifted sums slice with negative ends; pixel (un)shuffles read their sizes.
The rest (convolutions, plugin nodes, tail) is graph.py unchanged. Plugin nodes carry a
width-only key (DualCore_c128_TRT), which chaiNNer-C lowers to its dynamic plugins.
"""

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from onnx import TensorProto as T

from .export import make_model, read_state
from .graph import Graph
from .spec import FACTORIES, SCHEMA, SOURCE_SHA256, sha256

WINDOW = 96  # retrieval block
END = np.iinfo(np.int64).max


class DynamicGraph(Graph):
    def __init__(self, arch, channels, alignment):
        """alignment: every trunk size the engine will see is a multiple of it, so
        unshifted windows that divide it are regular (no gathers)."""
        key = f"c{channels}"
        super().__init__(
            arch,
            SimpleNamespace(
                channels=channels, trunk=(0, 0), plugin_key=key, height=0, width=0
            ),
        )
        self.alignment = alignment
        # The tail's current scale against the network input, while in the tail.
        self.in_tail, self.tail_scale = False, Fraction(1)

    # Run-time integer helpers: [1]-shaped int64 for shape vectors, 0-d for Range.
    def i(self, value):
        return self.const(np.array([value], np.int64), T.INT64)

    def s(self, value):
        return self.const(np.array(value, np.int64), T.INT64)

    def dim(self, x, axis):
        return self.op("Shape", [x], start=axis, end=axis + 1)

    def squeeze(self, x):
        return self.op("Squeeze", [x, self.const(np.array([0], np.int64), T.INT64)])

    def unsqueeze(self, x, axis):
        return self.op(
            "Unsqueeze", [x, self.const(np.array([axis], np.int64), T.INT64)]
        )

    def shape(self, x, parts):
        return self.op("Reshape", [x, self.op("Concat", parts, axis=0)], allowzero=1)

    def arange(self, n):
        return self.op("Range", [self.s(0), self.squeeze(n), self.s(1)])

    def add(self, a, b):
        return self.op("Add", [a, b])

    def sub(self, a, b):
        return self.op("Sub", [a, b])

    def mul(self, a, b):
        return self.op("Mul", [a, b])

    def div(self, a, b):
        return self.op("Div", [a, b])

    # Size-dependent parts of graph.Graph.
    def dw(self, x, mod, size, epilogue=False):
        v = self.cast(x, T.FLOAT)
        padded = self.op("Pad", [v, self.const([0, 1, 1, 0, 0, 1, 1, 0], T.INT64)])
        y = None
        weight = mod.weight.detach()[:, 0].permute(1, 2, 0)
        for iy in range(3):
            for ix in range(3):
                ends = [iy - 2 if iy < 2 else END, ix - 2 if ix < 2 else END]
                term = self.slice(padded, [iy, ix], ends, [1, 2])
                term = self.op("Mul", [term, self.const(weight[iy, ix], T.FLOAT)])
                y = term if y is None else self.op("Add", [y, term])
        y = self.op("Add", [y, self.const(mod.bias, T.FLOAT)])
        if epilogue:
            y = self.op("Add", [v, self.gelu(y, T.FLOAT)])
        return self.cast(y, T.BFLOAT16)

    def axis_windows(self, length, window, offset):
        w = self.i(window)
        extent = self.op("Min", [length, w])
        ceil = self.div(self.add(length, self.i(offset + window - 1)), w)
        count = self.op("Where", [self.op("LessOrEqual", [length, w]), self.i(1), ceil])
        starts = self.sub(self.mul(self.arange(count), self.s(window)), self.s(offset))
        starts = self.op("Min", [starts, self.squeeze(self.sub(length, extent))])
        starts = self.op("Max", [starts, self.s(0)])
        gather = self.add(
            self.unsqueeze(starts, 1), self.unsqueeze(self.arange(extent), 0)
        )
        gather = self.op(
            "Reshape", [gather, self.const(np.array([-1], np.int64), T.INT64)]
        )
        positions = self.arange(length)
        owner = self.div(self.add(positions, self.s(offset)), self.s(window))
        owner = self.op("Min", [owner, self.squeeze(self.sub(count, self.i(1)))])
        local = self.sub(positions, self.op("Gather", [starts, owner], axis=0))
        restore = self.add(self.mul(owner, self.squeeze(extent)), local)
        return gather, restore, count, extent

    def attention(self, x, mod, size):
        c, heads, window = mod.channels, mod.heads, mod.window
        regular = mod.offset == 0 and self.alignment % window == 0
        if regular:  # graph.py's regular windows: reshapes only
            image, eh, ew = x, self.i(window), self.i(window)
            nr, nc = self.div(self.dim(x, 1), eh), self.div(self.dim(x, 2), ew)
        else:
            ih, rh, nr, eh = self.axis_windows(self.dim(x, 1), window, mod.offset)
            iw, rw, nc, ew = self.axis_windows(self.dim(x, 2), window, mod.offset)
            image = self.op("Gather", [self.op("Gather", [x, ih], axis=1), iw], axis=2)
        windows, tokens = self.mul(nr, nc), self.mul(eh, ew)
        t = self.trans(
            self.shape(image, [self.i(1), nr, eh, nc, ew, self.i(c)]),
            [0, 1, 3, 2, 4, 5],
        )
        t = self.shape(t, [windows, tokens, self.i(c)])
        qkv = self.trans(
            self.shape(
                self.pw(t, mod.qkv),
                [windows, tokens, self.i(3), self.i(heads), self.i(32)],
            ),
            [2, 0, 3, 1, 4],
        )
        q, k, v = [
            self.op("Gather", [qkv, self.const(i, T.INT64)], axis=0) for i in range(3)
        ]
        q = self.op("Mul", [q, self.const(32**-0.5, T.BFLOAT16)])
        # A (possibly cropped) window's positions in the nominal window x window grid.
        positions = self.add(
            self.unsqueeze(self.mul(self.arange(eh), self.s(window)), 1),
            self.unsqueeze(self.arange(ew), 0),
        )
        positions = self.op(
            "Reshape", [positions, self.const(np.array([-1], np.int64), T.INT64)]
        )
        bias = None
        if mod.factored_bias is not None:
            bq, bk = mod.factored_bias(window, window, torch.float32)
            shape = self.op(
                "Concat", [windows, self.i(heads), tokens, self.i(16)], axis=0
            )
            factors = []
            for factor in (bq, bk):
                f = self.op(
                    "Gather", [self.const(factor, T.BFLOAT16), positions], axis=1
                )
                factors.append(self.op("Expand", [self.unsqueeze(f, 0), shape]))
            q = self.op("Concat", [q, factors[0]], axis=-1)
            k = self.op("Concat", [k, factors[1]], axis=-1)
            v = self.op(
                "Pad",
                [
                    v,
                    self.const([0, 0, 0, 0, 0, 0, 0, 16], T.INT64),
                    self.const(0.0, T.BFLOAT16),
                ],
            )
        else:
            table = self.const(
                mod.position_bias(window, window, torch.float32), T.BFLOAT16
            )
            bias = self.op(
                "Gather",
                [self.op("Gather", [table, positions], axis=2), positions],
                axis=3,
            )
        # graph.mha, with the bias a gathered tensor instead of a constant.
        score = self.op("MatMul", [q, self.trans(k, [0, 1, 3, 2])])
        if bias is not None:
            score = self.op("Add", [score, bias])
        self.mha_count += 1
        out = self.op("MatMul", [self.op("Softmax", [score], axis=-1), v])
        if mod.factored_bias is not None:
            out = self.slice(out, [0], [32], [3])
        out = self.shape(
            self.trans(out, [0, 2, 1, 3]), [self.i(1), nr, nc, eh, ew, self.i(c)]
        )
        out = self.shape(
            self.trans(out, [0, 1, 3, 2, 4, 5]),
            [self.i(1), self.mul(nr, eh), self.mul(nc, ew), self.i(c)],
        )
        if not regular:
            out = self.op("Gather", [self.op("Gather", [out, rh], axis=1), rw], axis=2)
        gate = self.op(
            "Sigmoid",
            [self.pw(self.dw(x, mod.gate_depthwise, size), mod.gate_pointwise)],
        )
        return self.pw(self.op("Mul", [out, gate]), mod.project)

    def pack_slots(self, x, scale, h, w):
        heads, d = self.c // 32, 32 // scale**2
        rows, cols = (
            self.div(self.dim(x, 1), self.i(scale)),
            self.div(self.dim(x, 2), self.i(scale)),
        )
        x = self.shape(
            x,
            [
                self.i(1),
                rows,
                self.i(scale),
                cols,
                self.i(scale),
                self.i(heads),
                self.i(d),
            ],
        )
        return self.shape(
            self.trans(x, [0, 1, 3, 5, 2, 4, 6]),
            [self.i(1), rows, cols, self.i(self.c)],
        )

    def unpack_slots(self, x, scale, h, w):
        rows, cols = self.dim(x, 1), self.dim(x, 2)
        x = self.shape(
            x,
            [
                self.i(1),
                rows,
                cols,
                self.i(self.c // 32),
                self.i(scale),
                self.i(scale),
                self.i(32 // scale**2),
            ],
        )
        return self.shape(
            self.trans(x, [0, 1, 4, 2, 5, 3, 6]),
            [
                self.i(1),
                self.mul(rows, self.i(scale)),
                self.mul(cols, self.i(scale)),
                self.i(self.c // scale**2),
            ],
        )

    def unshuffle_nchw(self, x, channels, h, w, scale):
        rows, cols = (
            self.div(self.dim(x, 2), self.i(scale)),
            self.div(self.dim(x, 3), self.i(scale)),
        )
        x = self.shape(
            x, [self.i(1), self.i(channels), rows, self.i(scale), cols, self.i(scale)]
        )
        return self.shape(
            self.trans(x, [0, 1, 3, 5, 2, 4]),
            [self.i(1), self.i(channels * scale * scale), rows, cols],
        )

    def spans(self, length):
        """(start, stop, window count, window extent) of the full blocks and the leftover."""
        count = self.div(length, self.i(WINDOW))
        full = self.mul(count, self.i(WINDOW))
        rest = self.sub(length, full)
        return [
            (self.i(0), full, count, self.i(WINDOW)),
            (
                full,
                length,
                self.op("Min", [rest, self.i(1)]),
                self.op("Max", [rest, self.i(2)]),
            ),
        ]

    def pack_region(self, x, y0, y1, x0, x1, nr, eh, nc, ew):
        axes = self.const(np.array([1, 2], np.int64), T.INT64)
        region = self.op(
            "Slice",
            [
                x,
                self.op("Concat", [y0, x0], axis=0),
                self.op("Concat", [y1, x1], axis=0),
                axes,
            ],
        )
        t = self.trans(
            self.shape(region, [self.i(1), nr, eh, nc, ew, self.i(self.c)]),
            [0, 1, 3, 2, 4, 5],
        )
        t = self.shape(
            t, [self.mul(nr, nc), self.mul(eh, ew), self.i(self.c // 32), self.i(32)]
        )
        return self.trans(t, [0, 2, 1, 3])

    def halves(self, *values):
        return [self.div(v, self.i(2)) for v in values]

    def retrieval_descriptors(self, q, k):
        classes = []
        for y0, y1, nr, eh in self.spans(self.dim(q, 1)):
            for x0, x1, nc, ew in self.spans(self.dim(q, 2)):
                geometry = (y0, y1, x0, x1, nr, eh, nc, ew)
                hy0, hy1, hx0, hx1, heh, hew = self.halves(y0, y1, x0, x1, eh, ew)
                qq = self.pack_region(q, *geometry)
                kk = self.pack_region(k, hy0, hy1, hx0, hx1, nr, heh, nc, hew)
                classes.append((geometry, qq, kk))
        return classes

    def retrieve(self, packed, v, log_tau):
        tau = self.const(log_tau.exp().clamp(0.1, 10).reshape(1, -1, 1, 1), T.BFLOAT16)
        outputs = []
        for (y0, y1, x0, x1, nr, eh, nc, ew), q, k in packed:
            hy0, hy1, hx0, hx1, heh, hew = self.halves(y0, y1, x0, x1, eh, ew)
            vv = self.pack_region(v, hy0, hy1, hx0, hx1, nr, heh, nc, hew)
            keys = self.op("Max", [self.mul(heh, hew), self.i(1)])
            scale = self.op(
                "Div",
                [
                    self.op("Log", [self.cast(keys, T.FLOAT)]),
                    self.const(np.array([math.log(1024) * math.sqrt(32)]), T.FLOAT),
                ],
            )
            q = self.op("Mul", [self.op("Mul", [q, tau]), self.cast(scale, T.BFLOAT16)])
            out = self.trans(self.mha(q, k, vv, barrier=True), [0, 2, 1, 3])
            out = self.shape(out, [self.i(1), nr, nc, eh, ew, self.i(self.c)])
            out = self.trans(out, [0, 1, 3, 2, 4, 5])
            outputs.append(
                self.shape(
                    out, [self.i(1), self.mul(nr, eh), self.mul(nc, ew), self.i(self.c)]
                )
            )
        rows = [self.op("Concat", outputs[i : i + 2], axis=2) for i in (0, 2)]
        return self.op("Concat", rows, axis=1)

    def exact(self, x, channels, scale):
        """x reshaped to the size it has, [1, channels, scale*H, scale*W] of the network
        input: an identity, but computed from the input's shape TensorRT bounds the tail
        exactly. Through the graph's shape chain its fusion compiler finds no kernel for
        the large tail convolutions above ~512x1024 input (found by Codex, 2026-10-09)."""
        hw = self.op("Shape", ["input"], start=2, end=4)
        hw = self.div(self.mul(hw, self.i(scale.numerator)), self.i(scale.denominator))
        return self.shape(x, [self.i(1), self.i(channels), hw])

    def tail_sequence(self, x, layers):
        self.in_tail = True
        try:
            return super().tail_sequence(x, layers)
        finally:
            self.in_tail = False

    def conv(self, x, mod, nhwc=False, weight=None, bias=None):
        if self.in_tail:
            channels = (mod.weight if weight is None else weight).shape[1]
            x = self.exact(x, channels, self.tail_scale)
        return super().conv(x, mod, nhwc, weight, bias)

    def shuffle(self, x, scale):
        if self.in_tail:
            self.tail_scale *= scale
        return super().shuffle(x, scale)

    def build(self, model):
        self.tail_scale = Fraction(1, 2) if model.unshuffle else Fraction(1)
        onnx_model = super().build(model)
        for value, prefix in (
            (onnx_model.graph.input[0], "in"),
            (onnx_model.graph.output[0], "out"),
        ):
            dims = value.type.tensor_type.shape.dim
            for index, name in ((2, "height"), (3, "width")):
                dims[index].Clear()
                dims[index].dim_param = f"{prefix}_{name}"
        return onnx_model


def export_dynamic(
    factory, scale, alignment, checkpoint, output, *, state_key=None, folded=False
):
    """export.py's export_checkpoint for a dynamic-shape graph (same checks); every
    input size the engine will see must be a multiple of `alignment`."""
    import onnx

    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite export directory: {output}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = sha256(checkpoint)
    torch.set_num_threads(1)
    arch, model = make_model(factory, scale)
    if folded:
        model.prepare_for_export()
    model.load_state_dict(read_state(checkpoint, state_key), strict=True)
    model.prepare_for_export()
    if sha256(checkpoint) != checkpoint_sha:
        raise RuntimeError("Checkpoint changed during export")
    if alignment % (8 if model.unshuffle else 4):
        raise ValueError("The input alignment must keep the trunk a multiple of 4")
    graph = DynamicGraph(
        arch, model.embed_dim, alignment // (2 if model.unshuffle else 1)
    )
    with torch.inference_mode():
        onnx_model = graph.build(model)
    onnx.checker.check_model(onnx_model)
    output.mkdir(parents=True, exist_ok=False)
    onnx.save_model(
        onnx_model,
        str(output / "model.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=1024,
    )
    key = f"c{model.embed_dim}"
    manifest = {
        "schema": SCHEMA,
        "status": "exported_unvalidated",
        "source_sha256": SOURCE_SHA256,
        "checkpoint_sha256": checkpoint_sha,
        "state_key": state_key,
        "checkpoint_folded": folded,
        "specialization": {
            "factory": factory,
            "scale": scale,
            "channels": model.embed_dim,
            "unshuffle": model.unshuffle,
            "depths": list(model.depths),
            "indices": [list(v) for v in model.selected_original_indices],
            "batch": 1,
            "dynamic": True,
            "alignment": alignment,
            "internal_scale": model.internal_scale,
            "plugin_key": key,
        },
        "precision": "FP32_IO_BF16_FP32_BODY",
        "dynamic_sites": graph.dynamic_sites,
        "plugin_names": [
            f"Dual{part}_{key}_TRT" for part in ("Norm", "Core", "Project")
        ]
        + ["DualAttentionBarrier_TRT"],
        "files": {
            p.name: sha256(p)
            for p in (output / "model.onnx", output / "weights.bin")
            if p.exists()
        },
        "validation": "ONNX structural checker only; plugins and run time NOT validated",
    }
    (output / "export.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", choices=FACTORIES, required=True)
    parser.add_argument("--scale", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--alignment", type=int, default=64)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-key")
    parser.add_argument("--folded", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(export_dynamic(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
