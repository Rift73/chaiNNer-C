"""DUAL's custom ops (dual_migx_ops, native/src/dual_hip) without MIGraphX: each op's
compute_shape and compute called as the runner calls them (custom_op.h), with the
attributes of the nodes dual_hip.lower makes.

The library is the built one beside dual.py, or the one CHAINNER_C_DUAL_MIGX_OPS names:
for the check on NVIDIA, a build of the same source through HIP's NVIDIA path
(hipother's headers over the ROCm SDK's), e.g. on Linux
    clang++ -std=c++20 -O2 -fPIC -shared -D__HIP_PLATFORM_NVIDIA__ -I<hipother>/hipnv/include
        -I<rocm>/include -I<cuda>/include -Inative/migraphx_runner
        native/src/dual_hip/dual_migx_ops.cpp -lcudart -lcuda -o dual_migx_ops_nv.so
Shapes need no GPU. On an NVIDIA GPU the ops launch the kernels' PTX and must equal
DUAL's TensorRT plugins bit for bit; on an AMD GPU (amd_gpu) their HSACO, and must
equal Triton's own launches of the same kernels bit for bit."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper
from test_tensorrt_dual_na_kernel import (
    assert_matches_the_oracle,
    factor_tables,
    trunk,
)

triton = pytest.importorskip("triton")
dual_hip = pytest.importorskip("nodes.impl.migraphx.dual_hip")

LIBRARY = Path(
    os.environ.get("CHAINNER_C_DUAL_MIGX_OPS")
    or Path(dual_hip.__file__).with_name("dual_migx_ops.so")
)
ABI = 1
MAX_RANK = 8
TYPES = {torch.float32: 0, torch.bfloat16: 2, torch.uint8: 5}
DTYPES = {code: dtype for dtype, code in TYPES.items()}
INT, FLOAT, STRING = 0, 1, 2


class Shape(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int32),
        ("rank", ctypes.c_int32),
        ("dims", ctypes.c_int64 * MAX_RANK),
    ]


class Tensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("shape", Shape)]


class Attribute(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("kind", ctypes.c_int32),
        ("count", ctypes.c_int64),
        ("ints", ctypes.POINTER(ctypes.c_int64)),
        ("floats", ctypes.POINTER(ctypes.c_float)),
        ("bytes", ctypes.c_char_p),
    ]


ComputeShape = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(Shape),
    ctypes.c_int32,
    ctypes.POINTER(Attribute),
    ctypes.c_int32,
    ctypes.POINTER(Shape),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_char_p,
    ctypes.c_size_t,
)
Compute = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.POINTER(Tensor),
    ctypes.c_int32,
    ctypes.POINTER(Attribute),
    ctypes.c_int32,
    ctypes.POINTER(Tensor),
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_char_p,
    ctypes.c_size_t,
)


class Op(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("flags", ctypes.c_uint32),
        ("compute_shape", ComputeShape),
        ("compute", Compute),
    ]


class OpError(RuntimeError):
    pass


def _shape(dtype: torch.dtype, dims: tuple[int, ...]) -> Shape:
    shape = Shape(TYPES[dtype], len(dims))
    for i, dim in enumerate(dims):
        shape.dims[i] = dim
    return shape


def _attributes(values: dict[str, object]) -> tuple[ctypes.Array, list[object]]:
    """The attributes in their order, and what must stay alive while they are used."""
    items = (Attribute * max(len(values), 1))()
    keep: list[object] = []
    for item, (name, value) in zip(items, values.items(), strict=False):
        item.name = name.encode()
        keep.append(item.name)
        if isinstance(value, int):
            array = (ctypes.c_int64 * 1)(value)
            item.kind, item.count, item.ints = INT, 1, array
        elif isinstance(value, float):
            array = (ctypes.c_float * 1)(value)
            item.kind, item.count, item.floats = FLOAT, 1, array
        else:
            data = value if isinstance(value, bytes) else str(value).encode()
            array = ctypes.create_string_buffer(data, len(data) + 1)
            item.kind, item.count = STRING, len(data)
            item.bytes = ctypes.cast(array, ctypes.c_char_p)
        keep.append(array)
    return items, keep


class Ops:
    """The library's op table, called as the MIGraphX runner calls it."""

    def __init__(self, path: Path):
        self.library = ctypes.CDLL(str(path))
        table = ctypes.POINTER(Op)()
        count = ctypes.c_int32()
        assert (
            self.library.chainner_migx_ops(
                ABI, ctypes.byref(table), ctypes.byref(count)
            )
            == 0
        )
        assert (
            self.library.chainner_migx_ops(
                ABI + 1, ctypes.byref(table), ctypes.byref(count)
            )
            != 0
        )
        self.ops = {table[i].name.decode(): table[i] for i in range(count.value)}

    def shape(
        self,
        name: str,
        inputs: list[tuple[torch.dtype, tuple[int, ...]]],
        attributes: dict,
    ) -> tuple[torch.dtype, tuple[int, ...], int]:
        op = self.ops[name]
        shapes = (Shape * len(inputs))(*(_shape(d, s) for d, s in inputs))
        items, keep = _attributes(attributes)
        output, workspace = Shape(), ctypes.c_size_t(0)
        error = ctypes.create_string_buffer(512)
        status = op.compute_shape(
            shapes,
            len(inputs),
            items,
            len(attributes),
            ctypes.byref(output),
            ctypes.byref(workspace),
            error,
            len(error),
        )
        del keep
        if status != 0:
            raise OpError(error.value.decode())
        dims = tuple(output.dims[i] for i in range(output.rank))
        return DTYPES[output.type], dims, workspace.value

    def run(
        self, name: str, inputs: list[torch.Tensor], attributes: dict
    ) -> torch.Tensor:
        """The op's output, computed on the current stream (NaN-filled first, so an
        element it does not write shows)."""
        dtype, dims, workspace_bytes = self.shape(
            name, [(t.dtype, tuple(t.shape)) for t in inputs], attributes
        )
        device = inputs[0].device
        output = torch.full(dims, torch.nan, dtype=dtype, device=device)
        workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
        tensors = (Tensor * len(inputs))(
            *(Tensor(t.data_ptr(), _shape(t.dtype, tuple(t.shape))) for t in inputs)
        )
        out = Tensor(output.data_ptr(), _shape(dtype, dims))
        items, keep = _attributes(attributes)
        error = ctypes.create_string_buffer(512)
        status = self.ops[name].compute(
            torch.cuda.current_stream().cuda_stream,
            tensors,
            len(inputs),
            items,
            len(attributes),
            ctypes.byref(out),
            workspace.data_ptr() if workspace_bytes else None,
            workspace_bytes,
            error,
            len(error),
        )
        del keep
        if status != 0:
            raise OpError(error.value.decode())
        return output


@pytest.fixture(scope="module")
def ops() -> Ops:
    if not LIBRARY.is_file():
        pytest.skip(f"no DUAL op library at {LIBRARY}")
    # Its HIP runtime (the ROCm wheels' libamdhip64.so.7, or CUDA's) is loaded with torch.
    try:
        return Ops(LIBRARY)
    except OSError as error:
        # A runtime whose torch is neither (the CUDA one beside an AMD build) has none.
        if "libamdhip64" not in str(error):
            raise
        pytest.skip(f"no HIP runtime: {error}")


def test_the_library_exports_the_seven_ops(ops: Ops):
    assert sorted(ops.ops) == sorted(
        f"chainner_dual_{op}"
        for op in ("norm", "cell", "region", "apply", "project", "na", "barrier")
    )
    # Every op keeps its types out of fp16/fp8 quantization.
    assert {op.flags for op in ops.ops.values()} == {2}


BF16, F32 = torch.bfloat16, torch.float32


def test_shapes_follow_the_trunk(ops: Ops):
    h, w, c = 37, 70, 160
    heads, cells = c // 32, 3 * 5
    regions = 2 * 3 + 2 * 3 + 2 * 3 + 2 * 3  # 32x32 grids shifted by 0 / 16 px
    qkv, weight = (BF16, (1, h, w, 3 * c)), (BF16, (3, 3, 3 * c))
    trunk = (BF16, (1, h, w, c))
    tables = (BF16, (heads, 79 * 79, 16))
    assert ops.shape("chainner_dual_norm", [trunk, (F32, (c,)), (F32, (c,))], {}) == (
        BF16,
        trunk[1],
        h * w * 4,
    )
    stats = ops.shape("chainner_dual_cell", [qkv, weight], {})
    assert stats == (F32, (cells * heads * 1088,), 0)
    matrices = ops.shape("chainner_dual_region", [stats[:2], (F32, (heads,)), qkv], {})
    assert matrices == (BF16, (regions * heads * 1024,), 0)
    assert (
        ops.shape("chainner_dual_apply", [qkv, weight, matrices[:2]], {})[:2] == trunk
    )
    assert (
        ops.shape("chainner_dual_project", [trunk, (BF16, (c, c)), trunk], {})[:2]
        == trunk
    )
    assert (
        ops.shape("chainner_dual_na", [qkv, tables, tables], {"window": 64})[:2]
        == trunk
    )
    assert ops.shape("chainner_dual_barrier", [trunk], {})[:2] == trunk


@pytest.mark.parametrize(
    ("name", "inputs", "attributes", "message"),
    [
        ("chainner_dual_cell", [(F32, (1, 8, 8, 384)), (BF16, (3, 3, 384))], {}, "QKV"),
        (
            "chainner_dual_cell",
            [(BF16, (1, 8, 8, 384)), (BF16, (3, 3, 96))],
            {},
            "depthwise",
        ),
        (
            "chainner_dual_region",
            [(F32, (1,)), (F32, (4,)), (BF16, (1, 8, 8, 384))],
            {},
            "statistics",
        ),
        (
            "chainner_dual_na",
            [
                (BF16, (1, 8, 8, 384)),
                (BF16, (4, 47 * 47, 16)),
                (BF16, (4, 47 * 47, 16)),
            ],
            {"window": 64},
            "window 64",
        ),
        (
            "chainner_dual_norm",
            [(BF16, (1, 8, 8, 128)), (F32, (96,)), (F32, (128,))],
            {},
            "gamma",
        ),
        (
            "chainner_dual_project",
            [(BF16, (1, 8, 8, 128)), (BF16, (128, 128))],
            {},
            "3 inputs",
        ),
        (
            "chainner_dual_cell",
            [(BF16, (1, 30000, 30000, 384)), (BF16, (3, 3, 384))],
            {},
            "32-bit",
        ),
    ],
)
def test_shapes_it_cannot_compute_are_refused(
    ops: Ops,
    name: str,
    inputs: list[tuple[torch.dtype, tuple[int, ...]]],
    attributes: dict,
    message: str,
):
    with pytest.raises(OpError, match=message):
        ops.shape(name, inputs, attributes)


# The GPU checks: a trunk of partial 16x16 tiles whose sizes and pixel count are not
# multiples of 16 (so Triton's own launches specialize them as the compiles do not).
H, W = 37, 70
OUTPUTS = ("n", "core", "project", "na32", "na64", "copy")


def _inputs(channels: int) -> dict[str, torch.Tensor]:
    heads = channels // 32
    g = torch.Generator(device="cuda").manual_seed(channels)

    def randn(
        *shape: int, scale: float = 1.0, dtype: torch.dtype = BF16
    ) -> torch.Tensor:
        return (torch.randn(shape, generator=g, device="cuda") * scale).to(dtype)

    values = {
        "x": randn(1, H, W, channels),
        "gamma": randn(channels, dtype=F32),
        "beta": randn(channels, dtype=F32),
        "qkv": randn(1, H, W, 3 * channels),
        "wdw": randn(3, 3, 3 * channels, scale=0.3),
        "tau": torch.rand(heads, generator=g, device="cuda") * 10 + 1,
        "wp": randn(channels, channels, scale=channels**-0.5),
        "residual": randn(1, H, W, channels),
    }
    for window in (32, 64):
        u = window + 15
        values[f"bq{window}"] = randn(heads, u * u, 16, scale=0.5)
        values[f"bk{window}"] = randn(heads, u * u, 16, scale=0.5)
    return values


def _plugin_model(channels: int, values: dict[str, torch.Tensor]) -> onnx.ModelProto:
    """Every plugin node of a dynamic-shape DUAL ONNX (key c<C>), on graph inputs."""
    key = f"c{channels}"
    attrs = {"domain": "trt", "plugin_version": "1", "plugin_namespace": ""}
    nodes = [
        helper.make_node(
            f"DualNorm_{key}_TRT",
            ["x", "gamma", "beta"],
            ["n", "n_unused_1"],
            name="norm",
            residual=0,
            epsilon=1e-4,
            **attrs,
        ),
        helper.make_node(
            f"DualCore_{key}_TRT", ["qkv", "wdw", "tau"], ["core"], name="core", **attrs
        ),
        helper.make_node(
            f"DualProject_{key}_TRT",
            ["core", "wp", "residual"],
            ["project"],
            name="project",
            **attrs,
        ),
        *(
            helper.make_node(
                f"DualNA_{key}_TRT",
                ["qkv", f"bq{m}", f"bk{m}"],
                [f"na{m}"],
                name=f"na{m}",
                window=m,
                **attrs,
            )
            for m in (32, 64)
        ),
        helper.make_node(
            "DualAttentionBarrier_TRT", ["x"], ["copy"], name="barrier", **attrs
        ),
    ]
    types = {BF16: TensorProto.BFLOAT16, F32: TensorProto.FLOAT}
    graph = helper.make_graph(
        nodes,
        "plugins",
        [
            helper.make_tensor_value_info(name, types[v.dtype], list(v.shape))
            for name, v in values.items()
        ],
        [
            helper.make_tensor_value_info(
                name, TensorProto.BFLOAT16, [1, H, W, channels]
            )
            for name in OUTPUTS
        ],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("trt", 1)],
    )


def _gpu_target() -> object:
    if torch.version.hip:
        return dual_hip.Target.amd(torch.cuda.get_device_properties(0).gcnArchName)
    major, minor = torch.cuda.get_device_capability()
    return dual_hip.Target("cuda", major * 10 + minor)


def _run_ops(
    ops: Ops, channels: int, values: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """The lowered plugin graph's chainner nodes, run through the library in order."""
    lowered = dual_hip.lower(
        _plugin_model(channels, values), f"c{channels}", _gpu_target()
    )
    values = dict(values)
    for node in lowered.graph.node:
        attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        values[node.output[0]] = ops.run(
            node.op_type, [values[name] for name in node.input], attributes
        )
    torch.cuda.synchronize()
    return values


def _assert_bit_equal(ours: dict, reference: dict) -> None:
    for name in OUTPUTS:
        a, b = ours[name], reference[name]
        assert not a.isnan().any(), name
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), (
            f"{name}: largest difference {(a.float() - b.float()).abs().max()}"
        )


def _tensorrt_reference(
    channels: int, values: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """The plugins' outputs from a TensorRT engine of the plugin graph, its plugins
    those of a dynamic-shape DUAL engine (tensorrt/dual_aot.py)."""
    trt = pytest.importorskip("tensorrt")
    dual_aot = pytest.importorskip("nodes.impl.tensorrt.dual_aot")
    dual_aot.register_dynamic(channels)
    lowered = dual_aot.lower(_plugin_model(channels, values), f"c{channels}")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, logger)
    assert parser.parse(lowered.model.SerializeToString())
    dual_aot.attach(network, lowered)
    serialized = builder.build_serialized_network(
        network, builder.create_builder_config()
    )
    engine = trt.Runtime(logger).deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()
    outputs = {name: torch.empty_like(values["x"]) for name in OUTPUTS}
    for name, tensor in (*values.items(), *outputs.items()):
        context.set_tensor_address(name, tensor.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return outputs


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.cuda), reason="needs an NVIDIA GPU"
)
@pytest.mark.parametrize("channels", [128, 160])
def test_the_ops_equal_the_tensorrt_plugins_on_nvidia(ops: Ops, channels: int):
    values = _inputs(channels)
    reference = _tensorrt_reference(channels, values)
    _assert_bit_equal(_run_ops(ops, channels, values), reference)


def _triton_reference(
    channels: int, values: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """The same kernels launched by Triton itself, with tensorrt/dual_aot.py's dynamic
    plugins' grids and arguments."""
    kernels, na = dual_hip.dual_dynamic_kernels, dual_hip.dual_na_kernel
    cdiv, options = triton.cdiv, dual_hip.OPTIONS
    heads, pixels = channels // 32, H * W
    cells = cdiv(H, 16) * cdiv(W, 16)
    shifts = ((0, 0), (0, 16), (16, 0), (16, 16))
    counts = [cdiv(H + sy, 32) * cdiv(W + sx, 32) for sy, sx in shifts]
    block_c = 1 << (channels - 1).bit_length()
    v = values
    out = {name: torch.empty_like(v["x"]) for name in OUTPUTS}
    sigma = torch.empty(pixels, device="cuda")
    kernels.norm[(cdiv(pixels, dual_hip.NORM_PIXELS),)](
        v["x"],
        v["gamma"],
        v["beta"],
        pixels,
        out["n"],
        sigma,
        channels=channels,
        block_c=block_c,
        block_p=dual_hip.NORM_PIXELS,
        # The ONNX attribute's FP32 value, as the lowering passes it
        eps=float(np.float32(1e-4)),
        **options,
    )
    stats = torch.empty(cells * heads * 1088, device="cuda")
    kernels.cell_stats[(cells, heads)](
        v["qkv"], v["wdw"], H, W, stats, channels=channels, block=256, **options
    )
    matrices = torch.empty(sum(counts) * heads * 1024, device="cuda", dtype=BF16)
    kernels.region_stats[(sum(counts), heads)](
        stats,
        v["tau"],
        v["qkv"],
        H,
        W,
        *counts[:3],
        matrices,
        channels=channels,
        **options,
    )
    kernels.apply_dw[(cells, heads, 256 // dual_hip.APPLY_BLOCK)](
        v["qkv"],
        v["wdw"],
        matrices,
        H,
        W,
        *counts[:3],
        out["core"],
        channels=channels,
        block=dual_hip.APPLY_BLOCK,
        **options,
    )
    kernels.project_add[(cdiv(pixels, dual_hip.PROJECT_ROWS),)](
        out["core"],
        v["wp"],
        v["residual"],
        pixels,
        out["project"],
        channels=channels,
        block_c=block_c,
        block_m=dual_hip.PROJECT_ROWS,
        **options,
    )
    for m in (32, 64):
        na.na_attention[na.grid(H, W, channels, cdiv)](
            v["qkv"],
            v[f"bq{m}"],
            v[f"bk{m}"],
            H,
            W,
            out[f"na{m}"],
            channels=channels,
            window=m,
            **na.LAUNCH,
            **na.OPTIONS,
        )
    out["copy"].copy_(v["x"])
    torch.cuda.synchronize()
    return out


@pytest.mark.amd_gpu
@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.version.hip), reason="needs an AMD GPU"
)
@pytest.mark.parametrize("channels", [128, 160])
def test_the_ops_equal_tritons_own_launches_on_amd(ops: Ops, channels: int):
    values = _inputs(channels)
    reference = _triton_reference(channels, values)
    _assert_bit_equal(_run_ops(ops, channels, values), reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_kernels_load_once_and_their_ids_and_buffers_are_checked(ops: Ops):
    values = _inputs(128)
    attributes = dict(dual_hip.kernel("cell", 128, _gpu_target()))
    inputs = [values["qkv"], values["wdw"]]
    first = ops.run("chainner_dual_cell", inputs, attributes)
    again = ops.run("chainner_dual_cell", inputs, attributes)
    torch.cuda.synchronize()
    assert torch.equal(first, again)
    other = dual_hip.kernel("region", 128, _gpu_target())["code"]
    with pytest.raises(OpError, match="names different code objects"):
        ops.run("chainner_dual_cell", inputs, {**attributes, "code": other})
    # A QKV one BF16 element past a 16-byte boundary
    count = values["qkv"].numel()
    shifted = torch.empty(count + 8, dtype=BF16, device="cuda")[1 : 1 + count]
    with pytest.raises(OpError, match="16-byte aligned buffers; qkv"):
        ops.run(
            "chainner_dual_cell",
            [shifted.view(values["qkv"].shape), values["wdw"]],
            attributes,
        )


@pytest.mark.amd_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    ("window", "height", "width", "heads"),
    [(32, 2, 100, 4), (32, 31, 33, 5), (64, 16, 16, 4), (64, 101, 37, 4)],
)
def test_neighborhood_attention_matches_the_fp64_oracle(
    ops: Ops, window: int, height: int, width: int, heads: int
):
    # The op on this GPU (on AMD, Triton's AMD code) against the operator's definition,
    # as test_tensorrt_dual_na_kernel.py checks the kernel on NVIDIA.
    generator = torch.Generator(device="cuda").manual_seed(height * 1000 + width)
    bq, bk = factor_tables(heads, window, generator)
    qkv = trunk(height, width, heads, "random", generator)
    attributes = {
        "window": window,
        "parts": 256 // dual_hip.dual_na_kernel.LAUNCH["block_m"],
        **dual_hip.kernel("na", heads * 32, _gpu_target(), window),
    }
    out = ops.run("chainner_dual_na", [qkv, bq, bk], attributes)
    torch.cuda.synchronize()
    assert_matches_the_oracle(out, qkv, bq, bk, window)
