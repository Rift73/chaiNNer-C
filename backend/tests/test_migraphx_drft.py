"""Build Program's legacy DRFT route (nodes/impl/migraphx/drft.py): the DRFT ONNX of
Convert To ONNX, its plugin nodes rewritten into the DRFT MIGraphX ops, compiled in a
child process that loads the ops' library. The rewritten graph is checked on the CPU with
ONNX's reference evaluator, each op computed by NumPy as its definition says; the
compile and the GPU are stand-ins, except in the amd_gpu tests."""

from __future__ import annotations

import copy
import ctypes
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun
from spandrel import ImageModelDescriptor, ModelLoader
from spandrel.architectures.DRFT.__arch import legacy_drft_arch as legacy
from test_tensorrt_drft_export import legacy_drft

from nodes.impl.migraphx import drft, program, runner
from nodes.impl.tensorrt import drft as tensorrt_drft
from nodes.impl.tensorrt import drft_export

MIGRAPHX_OPS = {
    "chainner_drft_window_neural_bias",
    "chainner_drft_ocab_relative_bias",
    "chainner_drft_ocab_window_pack",
    "chainner_drft_final_affine_crop",
}


@pytest.fixture(scope="module")
def plugin_model() -> ImageModelDescriptor:
    return legacy_drft(32, 0.5)


@pytest.fixture(scope="module")
def plugin_onnx(plugin_model: ImageModelDescriptor) -> bytes:
    """Convert To ONNX's DRFT ONNX (FP16 body, BF16 attention)."""
    return drft_export.export_onnx(plugin_model)


@pytest.fixture(scope="module")
def small_model() -> ImageModelDescriptor:
    """A smaller random legacy DRFT x2 with the plugins' windows (one group of three
    blocks, two heads): the reference evaluator keeps every intermediate tensor (about
    20 GB for plugin_model at 64x96)."""
    torch.manual_seed(0)
    network = legacy.DRFT(
        embed_dim=32,
        depths=(3,),
        num_heads=(2,),
        window_size=32,
        overlap_ratio=0.5,
        upscale=2,
        num_feat=16,
        rank=8,
        drop_path_rate=0.0,
    )
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.normal_(0.0, 0.2)
    descriptor = ModelLoader().load_from_state_dict(network.state_dict())
    assert isinstance(descriptor, ImageModelDescriptor)
    return descriptor


@pytest.fixture(scope="module")
def fp32_onnx(small_model: ImageModelDescriptor) -> bytes:
    """Its export with the body and tail in FP32, which the reference evaluator
    computes exactly enough to compare with PyTorch."""
    with pytest.MonkeyPatch.context() as patch:
        deployment = drft_export.deployment
        patch.setattr(
            drft_export,
            "deployment",
            lambda network, _body, _tail, device: deployment(network, device=device),
        )
        return drft_export.export_onnx(small_model)


def _rewrite(onnx_bytes: bytes) -> onnx.ModelProto:
    manifest = drft.export_manifest(onnx_bytes)
    assert manifest is not None
    return onnx.load_from_string(drft.migraphx_onnx(onnx_bytes, manifest))


def _attributes(node: onnx.NodeProto) -> dict[str, Any]:
    return {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def test_both_routes_read_the_one_drft_onnx(plugin_onnx: bytes):
    assert drft.export_manifest is tensorrt_drft.export_manifest
    assert drft.check_profile is tensorrt_drft.check_profile
    assert drft.export_manifest(plugin_onnx) == {
        "format": 1,
        "plugins": "legacy-v1",
        "scale": 2,
        "window_size": 32,
        "min_size": 64,
        "alignment": 1,
    }


def test_the_rewrite_holds_the_migraphx_ops_and_no_plugin(plugin_onnx: bytes):
    proto = _rewrite(plugin_onnx)
    counts = Counter(node.op_type for node in proto.graph.node)
    assert not [op for op in counts if op.endswith("_TRT")]
    groups, blocks = 2, 6
    assert {op: counts[op] for op in MIGRAPHX_OPS} == {
        "chainner_drft_window_neural_bias": groups * blocks,
        "chainner_drft_ocab_relative_bias": groups,
        "chainner_drft_ocab_window_pack": 3 * groups,
        "chainner_drft_final_affine_crop": 1,
    }
    original = Counter(
        node.op_type for node in onnx.load_from_string(plugin_onnx).graph.node
    )
    assert (
        counts["Identity"] == original["Identity"] + original["DrftIdentityBarrier_TRT"]
    )
    for node in proto.graph.node:
        assert node.domain == ("chainner" if node.op_type in MIGRAPHX_OPS else "")
    assert {(entry.domain, entry.version) for entry in proto.opset_import} == {
        ("", drft_export.OPSET),
        ("chainner", 1),
    }
    packs = [
        _attributes(node)
        for node in proto.graph.node
        if node.op_type == "chainner_drft_ocab_window_pack"
    ]
    assert packs[:3] == [
        {"heads": 6, "query_window_size": 32, "window_size": window, "component": c}
        for c, window in enumerate((32, 48, 48))
    ]
    (final,) = [
        _attributes(node)
        for node in proto.graph.node
        if node.op_type == "chainner_drft_final_affine_crop"
    ]
    assert final.pop("scale") == 2
    assert set(final) == {"mean_r", "mean_g", "mean_b"}
    bias = [
        _attributes(node)
        for node in proto.graph.node
        if node.op_type == "chainner_drft_ocab_relative_bias"
    ]
    assert bias == [{"q_window_size": 32, "kv_window_size": 48}] * groups


def test_the_rewrite_names_the_input_for_the_alignment(plugin_onnx: bytes):
    proto = _rewrite(plugin_onnx)
    (graph_input,) = proto.graph.input
    assert graph_input.name == "input_aligned_1"
    produced = {value for node in proto.graph.node for value in node.output}
    known = produced | {graph_input.name} | {i.name for i in proto.graph.initializer}
    for node in proto.graph.node:
        assert set(node.input) - {""} <= known
    assert "input" not in {value for node in proto.graph.node for value in node.input}


def window_neural_bias(bias_q: np.ndarray, bias_k: np.ndarray) -> np.ndarray:
    """ACT attention's rank-factored bias (hnr,hmr->hnm)."""
    return np.einsum("hnr,hmr->hnm", bias_q, bias_k)


def relative_bias(table: np.ndarray, q_window_size: int, kv_window_size: int):
    """OCAB's dense relative bias, indexed as spandrel indexes its table."""
    q, k = q_window_size, kv_window_size
    query_y, query_x = np.divmod(np.arange(q * q), q)
    key_y, key_x = np.divmod(np.arange(k * k), k)
    index = (key_y[None, :] - query_y[:, None] + q - 1) * (q + k - 1) + (
        key_x[None, :] - query_x[:, None] + q - 1
    )
    return table[index].transpose(2, 0, 1)[None].copy()


def window_pack(
    qkv: np.ndarray,
    heads: int,
    query_window_size: int,
    window_size: int,
    component: int,
) -> np.ndarray:
    """One of q, k, v: the window around each query window (zero outside the image),
    its tokens split into heads, token major."""
    batch, height, width, _, _channels = qkv.shape
    q, window = query_window_size, window_size
    padding = (window - q) // 2
    plane = np.pad(
        qkv[:, :, :, component],
        ((0, 0), (padding, padding), (padding, padding), (0, 0)),
    )
    windows = np.stack(
        [
            plane[b, y : y + window, x : x + window]
            for b in range(batch)
            for y in range(0, height, q)
            for x in range(0, width, q)
        ]
    )
    windows = windows.reshape(len(windows), window * window, heads, -1)
    return windows.transpose(1, 0, 2, 3).copy()


def final_affine_crop(
    x: np.ndarray,
    image: np.ndarray,
    scale: int,
    mean_r: float,
    mean_g: float,
    mean_b: float,
) -> np.ndarray:
    """The output tail: the mean added, cropped to the input's size times the scale."""
    height, width = image.shape[2] * scale, image.shape[3] * scale
    mean = np.array([mean_r, mean_g, mean_b], np.float32).reshape(1, 3, 1, 1)
    return (x[:, :, :height, :width] + mean).astype(np.float32)


class WindowNeuralBias(OpRun):
    op_domain = "chainner"

    def _run(self, bias_q: np.ndarray, bias_k: np.ndarray):
        return (window_neural_bias(bias_q, bias_k),)


class RelativeBias(OpRun):
    op_domain = "chainner"

    def _run(self, table: np.ndarray, q_window_size: int, kv_window_size: int):
        return (relative_bias(table, q_window_size, kv_window_size),)


class WindowPack(OpRun):
    op_domain = "chainner"

    def _run(
        self,
        qkv: np.ndarray,
        heads: int,
        query_window_size: int,
        window_size: int,
        component: int,
    ):
        return (window_pack(qkv, heads, query_window_size, window_size, component),)


class FinalAffineCrop(OpRun):
    op_domain = "chainner"

    def _run(
        self,
        x: np.ndarray,
        image: np.ndarray,
        scale: int,
        mean_r: float,
        mean_g: float,
        mean_b: float,
    ):
        return (final_affine_crop(x, image, scale, mean_r, mean_g, mean_b),)


def _named(name: str, op: type[OpRun]) -> type[OpRun]:
    """`op` under the name of its op type, by which the reference evaluator finds a
    custom op."""
    named = type(name, (op,), {})
    assert issubclass(named, OpRun)
    return named


REFERENCE_OPS = [
    _named(name, op)
    for name, op in (
        ("chainner_drft_window_neural_bias", WindowNeuralBias),
        ("chainner_drft_ocab_relative_bias", RelativeBias),
        ("chainner_drft_ocab_window_pack", WindowPack),
        ("chainner_drft_final_affine_crop", FinalAffineCrop),
    )
]


@pytest.mark.parametrize(("height", "width"), [(64, 96), (72, 80)])
def test_the_rewritten_graph_computes_the_model_in_fp32(
    small_model: ImageModelDescriptor, fp32_onnx: bytes, height: int, width: int
):
    """The rewrite's ops, computed as their definitions say, give spandrel's own
    forward: the ops get the right inputs, attributes and component order."""
    proto = _rewrite(fp32_onnx)
    evaluator = ReferenceEvaluator(proto, new_ops=REFERENCE_OPS)
    image = torch.rand(1, 3, height, width)
    outputs = evaluator.run(None, {"input_aligned_1": image.numpy()})
    assert isinstance(outputs, list)
    actual = np.asarray(outputs[0])
    network = copy.deepcopy(small_model.model).float().eval()
    with torch.inference_mode():
        network.fold_reparam_conv()
        expected = network(image).numpy()
    assert actual.shape == expected.shape == (1, 3, 2 * height, 2 * width)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_another_plugin_set_is_refused(plugin_onnx: bytes):
    manifest = tensorrt_drft.export_manifest(plugin_onnx)
    assert manifest is not None
    manifest = {**manifest, "plugins": "legacy-v2"}
    with pytest.raises(ValueError, match="plugin set 'legacy-v2'"):
        drft.migraphx_onnx(plugin_onnx, manifest)


def _library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    library = tmp_path / drft.OP_LIBRARY.name
    library.write_bytes(b"library")
    monkeypatch.setattr(drft, "OP_LIBRARY", library)
    return library


def test_a_program_compiles_in_a_child_with_the_op_library(
    plugin_onnx: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    library = _library(tmp_path, monkeypatch)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def compile_in_child(*args: object, **kwargs: object) -> bytes:
        calls.append((args, kwargs))
        return b"program"

    monkeypatch.setattr(program, "compile_in_child", compile_in_child)
    manifest = drft.export_manifest(plugin_onnx)
    assert manifest is not None
    built = drft.build_program(
        plugin_onnx, manifest, 1, (256, 320), exhaustive_tune=True
    )
    assert built == b"program"
    ((args, kwargs),) = calls
    assert args[0] == drft.migraphx_onnx(plugin_onnx, manifest)
    assert args[1:] == (1, (256, 320))
    assert kwargs == {
        "fp16": False,
        "fp8": False,
        "exhaustive_tune": True,
        "op_libraries": (library,),
        "label": "DRFT program build",
        "context": None,
    }


def test_a_build_without_the_op_library_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(drft, "OP_LIBRARY", tmp_path / drft.OP_LIBRARY.name)
    monkeypatch.setattr(program, "compile_in_child", pytest.fail)
    with pytest.raises(RuntimeError, match="this build was made without it"):
        drft.build_program(b"onnx", {"plugins": "legacy-v1"}, 0, (64, 64))


@pytest.mark.parametrize("present", [True, False])
def test_processes_register_the_op_library_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, present: bool
):
    """Programs do not carry the library: a loading process registers it once (a build
    without it registers nothing, and only DRFT programs then fail)."""
    library = tmp_path / drft.OP_LIBRARY.name
    if present:
        library.write_bytes(b"library")
    monkeypatch.setattr(drft, "OP_LIBRARY", library)
    loaded: list[str] = []
    module = SimpleNamespace(load_op_library=loaded.append)
    monkeypatch.setattr(runner, "load_migraphx", lambda: module)
    drft.load_ops.cache_clear()
    try:
        drft.load_ops()
        drft.load_ops()
    finally:
        drft.load_ops.cache_clear()
    assert loaded == ([str(library)] if present else [])


def test_the_build_identity_follows_the_op_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    library = tmp_path / drft.OP_LIBRARY.name
    monkeypatch.setattr(drft, "BUILD_SOURCES", (library, Path(drft.__file__)))
    identities = []
    for content in (None, b"one", b"two"):
        if content is not None:
            library.write_bytes(content)
        drft.build_identity.cache_clear()
        identities.append(drft.build_identity())
    drft.build_identity.cache_clear()
    assert len(set(identities)) == 3


# On an AMD GPU (the kit): each op's kernel against its definition, and DRFT programs
# against PyTorch. Without one they are skipped; with one, anything missing fails them.

POINTER, INT = ctypes.c_void_p, ctypes.c_int32
LAUNCHERS = {
    "drftHipLaunchWindowNeuralBias": [POINTER] * 3 + [INT] * 5,
    "drftHipLaunchOCABRelativeBias": [POINTER] * 2 + [INT] * 4,
    "drftHipLaunchOCABWindowPack": [POINTER] * 2 + [INT] * 8,
    "drftHipLaunchFinalAffineCrop": [POINTER] * 2 + [INT] * 7 + [ctypes.c_float] * 3,
    "drftHipLaunchOCABAttention": [POINTER] * 5 + [INT] * 4,
}
# drft_hip::ElementType
ELEMENT_TYPES = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}


@pytest.fixture(scope="module")
def hip_kernels() -> ctypes.CDLL:
    """The ops' library, whose test launchers take device pointers and a stream of the
    AMD GPU PyTorch runs on (HIP's index 0)."""
    if torch.version.hip is None or not torch.cuda.is_available():
        pytest.skip("no AMD GPU")
    library = ctypes.CDLL(str(drft.op_library()))
    for name, arguments in LAUNCHERS.items():
        launcher = getattr(library, name)
        launcher.argtypes = [*arguments, POINTER]
        launcher.restype = INT
    return library


def _launch(launcher: Any, *arguments: object) -> None:
    status = launcher(*arguments, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert status == 0


def _bits(tensor: torch.Tensor) -> np.ndarray:
    """A tensor's values as their bits (NumPy has no BF16)."""
    width = {2: torch.int16, 4: torch.int32}[tensor.element_size()]
    return tensor.contiguous().view(width).cpu().numpy()


def _random(*shape: int, seed: int = 0, scale: float = 1.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator) * scale


@pytest.mark.amd_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("query_tokens", "key_tokens", "rank"), [(1024, 1024, 8), (100, 130, 20)]
)
def test_the_window_neural_bias_kernel(
    hip_kernels: ctypes.CDLL,
    dtype: torch.dtype,
    query_tokens: int,
    key_tokens: int,
    rank: int,
):
    """Sums in FP32, then rounds once: within one unit of the exact sum's rounding."""
    heads = 6
    bias_q = _random(heads, query_tokens, rank).to(dtype)
    bias_k = _random(heads, key_tokens, rank, seed=1).to(dtype)
    device_q, device_k = bias_q.cuda(), bias_k.cuda()
    output = torch.empty(heads, query_tokens, key_tokens, dtype=dtype, device="cuda")
    _launch(
        hip_kernels.drftHipLaunchWindowNeuralBias,
        device_q.data_ptr(),
        device_k.data_ptr(),
        output.data_ptr(),
        ELEMENT_TYPES[dtype],
        heads,
        query_tokens,
        key_tokens,
        rank,
    )
    exact = window_neural_bias(bias_q.double().numpy(), bias_k.double().numpy())
    expected = torch.from_numpy(exact).to(dtype).float()
    epsilon = torch.finfo(dtype).eps
    torch.testing.assert_close(output.float().cpu(), expected, rtol=epsilon, atol=1e-5)


@pytest.mark.amd_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("key_window", [40, 48])
def test_the_relative_bias_kernel(
    hip_kernels: ctypes.CDLL, dtype: torch.dtype, key_window: int
):
    heads, span = 6, 32 + key_window - 1
    table = _random(span * span, heads).to(dtype)
    device_table = table.cuda()
    output = torch.empty(1, heads, 32 * 32, key_window**2, dtype=dtype, device="cuda")
    _launch(
        hip_kernels.drftHipLaunchOCABRelativeBias,
        device_table.data_ptr(),
        output.data_ptr(),
        ELEMENT_TYPES[dtype],
        heads,
        32,
        key_window,
    )
    assert np.array_equal(_bits(output), relative_bias(_bits(table), 32, key_window))


@pytest.mark.amd_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("component", "window"), [(0, 32), (1, 48), (2, 48), (1, 40), (2, 40)]
)
def test_the_window_pack_kernel(
    hip_kernels: ctypes.CDLL, dtype: torch.dtype, component: int, window: int
):
    batch, height, width, channels, heads = 2, 64, 96, 48, 6
    qkv = _random(batch, height, width, 3, channels).to(dtype)
    device_qkv = qkv.cuda()
    windows = batch * (height // 32) * (width // 32)
    output = torch.empty(
        window * window, windows, heads, channels // heads, dtype=dtype, device="cuda"
    )
    _launch(
        hip_kernels.drftHipLaunchOCABWindowPack,
        device_qkv.data_ptr(),
        output.data_ptr(),
        batch,
        height,
        width,
        channels,
        heads,
        32,
        window,
        component,
    )
    expected = window_pack(_bits(qkv), heads, 32, window, component)
    assert np.array_equal(_bits(output), expected)


@pytest.mark.amd_gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_the_final_affine_crop_kernel(hip_kernels: ctypes.CDLL, dtype: torch.dtype):
    x = _random(1, 3, 192, 256).to(dtype)
    device_x = x.cuda()
    image = np.zeros((1, 3, 72, 100), np.float32)
    mean = (0.4488, 0.4371, 0.4040)
    output = torch.empty(1, 3, 144, 200, device="cuda")
    _launch(
        hip_kernels.drftHipLaunchFinalAffineCrop,
        device_x.data_ptr(),
        output.data_ptr(),
        ELEMENT_TYPES[dtype],
        1,
        3,
        192,
        256,
        144,
        200,
        *mean,
    )
    expected = final_affine_crop(x.float().numpy(), image, 2, *mean)
    assert np.array_equal(output.cpu().numpy(), expected)


def ocab_attention(
    query: np.ndarray,
    key_t: np.ndarray,
    value: np.ndarray,
    table: np.ndarray,
    key_window: int,
) -> np.ndarray:
    """OCAB attention (q32) with its relative bias, exactly in the operands' type."""
    logits = query @ key_t + relative_bias(table, 32, key_window)
    weights = np.exp(logits - logits.max(-1, keepdims=True))
    return (weights / weights.sum(-1, keepdims=True)) @ value


@pytest.mark.amd_gpu
@pytest.mark.parametrize("key_window", [40, 48])
def test_the_ocab_attention_kernel(hip_kernels: ctypes.CDLL, key_window: int):
    """BF16 operands, FP32 sums and BF16 probabilities: within BF16's resolution of
    the exact attention."""
    batch, heads, key_tokens = 2, 6, key_window**2
    span = 32 + key_window - 1
    operands = [
        _random(batch, heads, 1024, 32, scale=0.5).bfloat16(),
        _random(batch, heads, 32, key_tokens, seed=1, scale=0.5).bfloat16(),
        _random(batch, heads, key_tokens, 32, seed=2).bfloat16(),
        _random(span * span, heads, seed=3).bfloat16(),
    ]
    device = [operand.cuda() for operand in operands]
    output = torch.empty_like(device[0])
    _launch(
        hip_kernels.drftHipLaunchOCABAttention,
        *(operand.data_ptr() for operand in device),
        output.data_ptr(),
        batch,
        heads,
        32,
        key_window,
    )
    query, key_t, value, table = (operand.double().numpy() for operand in operands)
    exact = ocab_attention(query, key_t, value, table, key_window)
    expected = torch.from_numpy(exact).float()
    torch.testing.assert_close(output.float().cpu(), expected, rtol=1e-2, atol=1e-2)


def _psnr(actual: np.ndarray, expected: np.ndarray) -> float:
    """On [0, 1], as images are saved."""
    error = np.clip(actual, 0, 1) - np.clip(expected, 0, 1)
    return float(10 * np.log10(1 / np.mean(np.square(error, dtype=np.float64))))


@pytest.mark.amd_gpu
def test_a_drft_program_matches_pytorch_fp32(
    hip_kernels: ctypes.CDLL, plugin_model: ImageModelDescriptor, plugin_onnx: bytes
):
    """Within the precision of the ONNX's types: PyTorch's own deployment of this
    random model in them (FP16 body, BF16 attention) gave 45.3-45.7 dB against FP32 on
    an RTX 5090."""
    from nodes.impl.migraphx.model import MIGraphXProgram
    from nodes.impl.migraphx.program_info import read_program_info
    from nodes.impl.migraphx.session import RunnerSession

    manifest = drft.export_manifest(plugin_onnx)
    assert manifest is not None
    size = (64, 96)
    saved = drft.build_program(plugin_onnx, manifest, 0, size)
    info = read_program_info(saved, 0)
    assert (info.precision, info.scale, info.height, info.width) == ("mixed", 2, *size)
    image = np.random.default_rng(7).random((*size, 3), dtype=np.float32)
    session = RunnerSession(MIGraphXProgram(saved, info), 0)
    try:
        actual = session.upscale(image)
    finally:
        session.close()
    network = copy.deepcopy(plugin_model.model).float().eval()
    rgb = torch.from_numpy(image[:, :, ::-1].copy()).permute(2, 0, 1)[None]
    with torch.inference_mode():
        network.fold_reparam_conv()
        expected = network(rgb)[0].permute(1, 2, 0).numpy()[:, :, ::-1]
    assert _psnr(actual, expected) >= 40


@pytest.mark.amd_gpu
def test_build_program_compiles_a_drft_onnx_once(
    hip_kernels: ctypes.CDLL,
    plugin_onnx: bytes,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Build Program's DRFT route keeps its program in memory and in the Program
    Cache: the same node again, or a process with an empty memory, compiles nothing."""
    from collections import OrderedDict

    from migraphx_stand_ins import context as node_context

    from nodes.impl.migraphx import build_cache
    from nodes.impl.onnx.load import load_onnx_model
    from packages.chaiNNer_migraphx.migraphx.utility import build_program

    compiles: list[tuple[int, int]] = []
    compile_in_child = program.compile_in_child

    def counted(*args: Any, **kwargs: Any) -> bytes:
        compiles.append(args[2])
        return compile_in_child(*args, **kwargs)

    monkeypatch.setattr(program, "compile_in_child", counted)
    model = load_onnx_model(plugin_onnx)
    context = node_context({"gpu_index": "0", "program_cache": str(tmp_path)}, "build")

    def build() -> Any:
        return build_program.build_program_node(
            context, model, build_program.ProgramPrecision.FP32, 64, 96, False
        )

    first = build()
    assert build() is first
    monkeypatch.setattr(build_cache, "_memory", OrderedDict())
    assert build().bytes == first.bytes
    assert compiles == [(64, 96)]
    assert first.info.precision == "mixed"
    assert len(list(tmp_path.iterdir())) == 1


# The ops' library's table (migraphx_runner/custom_op.h), read without a GPU where the
# library is built (Linux with the ROCm SDK) and the HIP runtime loads: its names, flags and
# output shapes, as the MIGraphX runner asks for them while it compiles a program.

HALF, BF16, FLOAT = 1, 2, 0  # chainner_migx_type
INT_ATTRIBUTE, FLOAT_ATTRIBUTE = 0, 1  # chainner_migx_attribute_kind
FIXED_TYPES = 2


class _Shape(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int32),
        ("rank", ctypes.c_int32),
        ("dims", ctypes.c_int64 * 8),
    ]


class _Attribute(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("kind", ctypes.c_int32),
        ("count", ctypes.c_int64),
        ("ints", ctypes.POINTER(ctypes.c_int64)),
        ("floats", ctypes.POINTER(ctypes.c_float)),
        ("bytes", ctypes.c_char_p),
    ]


_COMPUTE_SHAPE = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(_Shape),
    ctypes.c_int32,
    ctypes.POINTER(_Attribute),
    ctypes.c_int32,
    ctypes.POINTER(_Shape),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_char_p,
    ctypes.c_size_t,
)


class _Op(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("flags", ctypes.c_uint32),
        ("compute_shape", _COMPUTE_SHAPE),
        ("compute", ctypes.c_void_p),
    ]


@pytest.fixture(scope="module")
def op_table() -> dict[str, _Op]:
    if not drft.OP_LIBRARY.is_file():
        pytest.skip(f"{drft.OP_LIBRARY.name} is not built")
    if sys.platform == "win32":
        # Its HIP DLL is the ROCm pip packages', as the MIGraphX package finds it.
        try:
            folders = runner.rocm_dll_directories()
        except ImportError as error:
            pytest.skip(str(error))
        for folder in folders:
            os.add_dll_directory(str(folder))
    try:
        library = ctypes.CDLL(str(drft.OP_LIBRARY))
    except OSError as error:
        # A runtime without ROCm's wheels (the CUDA one) has no HIP for it to link.
        if "libamdhip64" not in str(error):
            raise
        pytest.skip(f"no HIP runtime: {error}")
    function = library.chainner_migx_ops
    function.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.POINTER(_Op)),
        ctypes.POINTER(ctypes.c_int32),
    ]
    table, count = ctypes.POINTER(_Op)(), ctypes.c_int32()
    assert function(2, ctypes.byref(table), ctypes.byref(count)) != 0
    assert function(1, ctypes.byref(table), ctypes.byref(count)) == 0
    ops = [table[index] for index in range(count.value)]
    return {op.name.decode(): op for op in ops}


def _output_shape(
    op: _Op, inputs: list[tuple[int, tuple[int, ...]]], **attributes: float
) -> tuple[int, tuple[int, ...]] | str:
    """The op's output (type, dims) for inputs of these (type, dims), or its error."""
    shapes = (_Shape * len(inputs))()
    for shape, (element, dims) in zip(shapes, inputs, strict=True):
        shape.type, shape.rank = element, len(dims)
        shape.dims[: len(dims)] = dims
    values = (_Attribute * max(1, len(attributes)))()
    for value, (name, number) in zip(values, attributes.items(), strict=False):
        value.name, value.count = name.encode(), 1
        if isinstance(number, int):
            ints = (ctypes.c_int64 * 1)(number)
            value.kind, value.ints = INT_ATTRIBUTE, ints
        else:
            floats = (ctypes.c_float * 1)(number)
            value.kind, value.floats = FLOAT_ATTRIBUTE, floats
    output, workspace = _Shape(), ctypes.c_size_t(0)
    error = ctypes.create_string_buffer(512)
    status = op.compute_shape(
        shapes,
        len(inputs),
        values,
        len(attributes),
        ctypes.byref(output),
        ctypes.byref(workspace),
        error,
        len(error),
    )
    if status:
        return error.value.decode()
    assert workspace.value == 0
    return output.type, tuple(output.dims[: output.rank])


def test_the_op_library_registers_its_ops_unquantized(op_table: dict[str, _Op]):
    assert {name: op.flags for name, op in op_table.items()} == {
        "chainner_drft_window_neural_bias": FIXED_TYPES,
        "chainner_drft_ocab_relative_bias": FIXED_TYPES,
        "chainner_drft_ocab_window_pack": FIXED_TYPES,
        "chainner_drft_final_affine_crop": FIXED_TYPES,
        "chainner_drft_ocab_attention": FIXED_TYPES,
    }


def test_the_ops_output_shapes(op_table: dict[str, _Op]):
    bias = op_table["chainner_drft_window_neural_bias"]
    assert _output_shape(bias, [(HALF, (6, 1024, 8)), (HALF, (6, 900, 8))]) == (
        HALF,
        (6, 1024, 900),
    )
    assert "same heads, rank and type" in str(
        _output_shape(bias, [(HALF, (6, 1024, 8)), (BF16, (6, 1024, 8))])
    )
    relative = op_table["chainner_drft_ocab_relative_bias"]
    windows = {"q_window_size": 32, "kv_window_size": 48}
    assert _output_shape(relative, [(BF16, (79 * 79, 6))], **windows) == (
        BF16,
        (1, 6, 1024, 2304),
    )
    assert "kv_window_size 40 or 48" in str(
        _output_shape(
            relative, [(BF16, (75 * 75, 6))], q_window_size=32, kv_window_size=44
        )
    )
    pack = op_table["chainner_drft_ocab_window_pack"]
    key = {"heads": 6, "query_window_size": 32, "window_size": 48, "component": 1}
    assert _output_shape(pack, [(HALF, (1, 64, 96, 3, 48))], **key) == (
        HALF,
        (2304, 6, 6, 8),
    )
    assert "a multiple of 8 per head" in str(
        _output_shape(pack, [(HALF, (1, 64, 96, 3, 36))], **key)
    )
    assert "component 0, 1 or 2" in str(
        _output_shape(pack, [(HALF, (1, 64, 96, 3, 48))], **{**key, "component": 3})
    )
    crop = op_table["chainner_drft_final_affine_crop"]
    tail = {"scale": 2, "mean_r": 0.4488, "mean_g": 0.4371, "mean_b": 0.404}
    inputs = [(HALF, (1, 3, 192, 256)), (FLOAT, (1, 3, 72, 100))]
    assert _output_shape(crop, inputs, **tail) == (FLOAT, (1, 3, 144, 200))
    assert "mean_r, mean_g, mean_b" in str(
        _output_shape(crop, inputs, scale=2, mean_r=0.4, mean_g=0.4)
    )
    attention = op_table["chainner_drft_ocab_attention"]
    operands = [
        (BF16, (2, 6, 1024, 32)),
        (BF16, (2, 6, 32, 1600)),
        (BF16, (2, 6, 1600, 32)),
        (BF16, (71 * 71, 6)),
    ]
    windows = {"q_window_size": 32, "kv_window_size": 40}
    assert _output_shape(attention, operands, **windows) == (BF16, (2, 6, 1024, 32))
    assert "takes BF16 q" in str(
        _output_shape(attention, [(HALF, (2, 6, 1024, 32)), *operands[1:]], **windows)
    )


def test_a_drft_onnx_compiles_on_the_reference_target(plugin_onnx: bytes):
    """Without a GPU, on MIGraphX's CPU reference target, where the runner is built and
    MIGraphX installed: the runner parses the rewritten graph, puts the ops in their
    nodes' places with the shapes they give, and saves and loads the program (running
    it needs the GPU)."""
    if not runner.runner_built() or not drft.OP_LIBRARY.is_file():
        pytest.skip("the MIGraphX runner or the DRFT op library is not built")
    try:
        module = runner.load_migraphx()
    except ImportError as error:
        pytest.skip(str(error))
    drft.load_ops()
    manifest = drft.export_manifest(plugin_onnx)
    assert manifest is not None
    compiled = module.compile_onnx(
        drft.migraphx_onnx(plugin_onnx, manifest),
        input_dims={"input_aligned_1": [1, 3, 72, 80]},
        target="ref",
    )
    assert compiled.compute_type == "mixed"
    assert sorted(compiled.custom_ops) == sorted(MIGRAPHX_OPS)
    assert list(compiled.output_shapes) == [(1, 3, 144, 160)]
    loaded = module.load_program(compiled.to_bytes(), 0)
    assert list(loaded.output_shapes) == [(1, 3, 144, 160)]
