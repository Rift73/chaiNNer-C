from __future__ import annotations

import copy
import gc
import math
import os
import sys
import threading
import weakref
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from ncnn import ncnn
from spandrel import (
    MAIN_REGISTRY,
    ImageModelDescriptor,
    ModelDescriptor,
    ModelLoader,
    SizeRequirements,
)
from spandrel.architectures.__arch_helpers.dysample import DySample
from spandrel.architectures.__arch_helpers.fp32_last_group import (
    FP32Conv2d,
    FP32LayerNorm,
)
from spandrel.architectures.Compact import Compact
from spandrel.architectures.DAT import DAT
from spandrel.architectures.DAT.__arch.DAT import (
    Adaptive_Channel_Attention,
    Adaptive_Spatial_Attention,
)
from spandrel.architectures.DITN import DITN
from spandrel.architectures.DRCT import DRCT
from spandrel.architectures.DRFT.__arch import drft_arch
from spandrel.architectures.DRFT.__arch import legacy_drft_arch as legacy
from spandrel.architectures.DUAL.__arch.dual_arch import dual_xs
from spandrel.architectures.ESRGAN import ESRGAN
from spandrel.architectures.FBCNN import FBCNN
from spandrel.architectures.FDAT import FDAT
from spandrel.architectures.FDAT.__arch.fdat import DySample as FdatDySample
from spandrel.architectures.FDAT.__arch.fdat import FastChannelAttention
from spandrel.architectures.FFTformer import FFTformer
from spandrel.architectures.FIGSR import FIGSR
from spandrel.architectures.GaterV3 import GateRV3
from spandrel.architectures.GaterV3.__arch.gaterv3 import DySample as GaterDySample
from spandrel.architectures.GFISRv2 import GFISRV2
from spandrel.architectures.GFISRv2.__arch.gfisrv2 import DySample as GfisrDySample
from spandrel.architectures.HAT import HAT
from spandrel.architectures.HAT.__arch.HAT import OCAB, WindowAttention
from spandrel.architectures.HVICIDNet import HVICIDNet
from spandrel.architectures.MixDehazeNet import MixDehazeNet
from spandrel.architectures.MoESR import MoESR
from spandrel.architectures.OmniSR import OmniSR
from spandrel.architectures.PLKSR import PLKSR, RealPLKSR
from spandrel.architectures.PLKSR.__arch.PLKSR import PLKConv2d, RectSparsePLKConv2d
from spandrel.architectures.PLKSR.__arch.RealPLKSR import PLKConv2d as RealPLKConv2d
from spandrel.architectures.RealCUGAN import UpCunet2x
from spandrel.architectures.RetinexFormer import RetinexFormer
from spandrel.architectures.RetinexFormer.__arch.retinexformer_arch import IG_MSA
from spandrel.architectures.RGT import RGT
from spandrel.architectures.SAFMNBCIE import SAFMNBCIE
from spandrel.architectures.SeemoRe import SeemoRe
from spandrel.architectures.SeemoRe.__arch.seemore_arch import MoELayer
from spandrel.architectures.SMoSR import SMoSR
from spandrel.architectures.SMoSR.__arch.smosr_arch import DySample as SmosrDySample
from spandrel.architectures.SPAN import SPAN
from spandrel.architectures.SwiftSRGAN import SwiftSRGAN
from spandrel.architectures.Swin2SR import Swin2SR
from spandrel.architectures.SwinIR import SwinIR
from spandrel.architectures.Uformer import Uformer

from api import Aborted, NodeContext, NodeId, SettingsParser
from nodes.impl.ncnn.model import NcnnModelWrapper
from nodes.impl.onnx import model as onnx_model_module
from nodes.impl.onnx.auto_split import onnx_auto_split
from nodes.impl.onnx.load import load_onnx_model, model_facts, with_recorded_arch
from nodes.impl.onnx.model import (
    ARCH_METADATA_KEY,
    FP16_ARCHS,
    PRECISION_METADATA_KEY,
    OnnxGeneric,
    converts_in_fp16,
)
from nodes.impl.onnx.onnx_to_ncnn import Onnx2NcnnConverter
from nodes.impl.onnx.session import get_onnx_session
from nodes.impl.onnx.utils import safely_optimize_onnx_model
from nodes.impl.pytorch import (
    attention_export,
    dysample_export,
    fourier_export,
    fp16_export,
    input_export,
    lda_export,
    plksr_export,
    swin_export,
    traced_size_export,
)
from nodes.impl.pytorch.convert_to_onnx_impl import (
    _example_side,
    convert_to_onnx_impl,
    export_copy,
    export_onnx,
    is_onnx_supported,
)
from nodes.impl.upscale.tiler import NoTiling
from nodes.properties.outputs import OnnxModelOutput
from packages.chaiNNer_onnx.onnx.utility.interpolate_models import (
    interpolate_models_node,
)
from packages.chaiNNer_pytorch.pytorch.io import load_model as load_model_module
from packages.chaiNNer_pytorch.pytorch.io.load_model import load_model
from packages.chaiNNer_pytorch.pytorch.utility import convert_to_onnx
from packages.chaiNNer_pytorch.pytorch.utility.convert_to_ncnn import (
    convert_to_ncnn_node,
)
from packages.chaiNNer_pytorch.pytorch.utility.convert_to_onnx import (
    DualShape,
    OnnxDataType,
    Opset,
    convert_to_onnx_node,
)
from packages.chaiNNer_pytorch.settings import PyTorchSettings

ARCHS: dict[str, Callable[[], torch.nn.Module]] = {
    "Compact": lambda: Compact(num_feat=16, num_conv=2, upscale=2),
    "SwiftSRGAN": lambda: SwiftSRGAN(num_channels=8, num_blocks=1, upscale_factor=2),
    "SPAN": lambda: SPAN(num_in_ch=3, num_out_ch=3, feature_channels=8, upscale=2),
    "ESRGAN": lambda: ESRGAN(num_filters=8, num_blocks=1, scale=4),
    "MixDehazeNet": lambda: MixDehazeNet(
        embed_dims=[12, 24, 48, 24, 12], depths=[1, 1, 1, 1, 1]
    ),
    # Heads of 6 channels, which the export pads to 8
    "HAT": lambda: HAT(
        embed_dim=12,
        depths=(2,),
        num_heads=(2,),
        window_size=8,
        compress_ratio=2,
        squeeze_factor=4,
        upsampler="pixelshuffle",
        upscale=2,
    ),
    # Three blocks: the third shifts its windows; heads of 6 channels, which the export
    # pads to 8 (24 channels: DAT's spatial interaction keeps a sixteenth of them)
    "DAT": lambda: DAT(
        embed_dim=24, depth=[3], num_heads=[4], upsampler="pixelshuffle", upscale=2
    ),
    # GroupNorm, so spandrel refuses FP16 for it in PyTorch
    "PLKSR": lambda: RealPLKSR(dim=16, n_blocks=2, upscaling_factor=2),
    "OmniSR": lambda: OmniSR(num_feat=16, window_size=8, up_scale=2),
    "SwinIR": lambda: SwinIR(
        embed_dim=12,
        depths=[2],
        num_heads=[2],
        window_size=8,
        mlp_ratio=2,
        upsampler="pixelshuffle",
        upscale=2,
    ),
    "Swin2SR": lambda: Swin2SR(
        embed_dim=12,
        depths=[2],
        num_heads=[2],
        window_size=8,
        mlp_ratio=2,
        upsampler="pixelshuffle",
        upscale=2,
    ),
    # 36 channels: its position bias keeps a sixteenth of a half of them
    "RGT": lambda: RGT(embed_dim=36, depth=[3], num_heads=[2], upscale=2),
    "DRCT": lambda: DRCT(
        embed_dim=12,
        depths=[5],
        num_heads=[2],
        window_size=8,
        gc=4,
        upsampler="pixelshuffle",
        upscale=2,
    ),
    "GFISRv2": lambda: GFISRV2(dim=16, n_blocks=2, scale=2),
    "FIGSR": lambda: FIGSR(
        dim=16, n_blocks=2, gc=2, square_kernel_size=3, band_kernel_size=3, scale=2
    ),
    "RetinexFormer": lambda: RetinexFormer(n_feat=16, stage=1),
}
# An architecture with no export rewrite of its own
PLAIN = "SwiftSRGAN"
# The architectures with an LDA upsampler, by scale: each has a copy of its own, or
# (FIGSR) GFISRv2's
LDA: dict[str, Callable[[int], torch.nn.Module]] = {
    "GFISRv2": lambda scale: GFISRV2(
        dim=16, mid_dim=16, n_blocks=1, upsampler="lda", scale=scale
    ),
    "FIGSR": lambda scale: FIGSR(
        dim=16,
        mid_dim=16,
        n_blocks=1,
        gc=2,
        square_kernel_size=3,
        band_kernel_size=3,
        upsampler="lda",
        scale=scale,
    ),
    "GaterV3": lambda scale: GateRV3(
        dim=16,
        enc_blocks=(1, 1),
        dec_blocks=(1, 1),
        num_latent=1,
        sisr_blocks=1,
        upsample="lda",
        upsample_mid_dim=16,
        upsample_version=3,
        scale=scale,
    ),
    "FDAT": lambda scale: FDAT(
        embed_dim=24,
        num_groups=1,
        depth_per_group=1,
        num_heads=2,
        mid_dim=16,
        upsampler_type="lda",
        scale=scale,
    ),
}
# Models whose forward reads a size in a way a trace stores as a constant, by what
# each reads, and sizes beyond the point it matters
TRACED_SIZE: dict[str, tuple[Callable[[], torch.nn.Module], list[tuple[int, int]]]] = {
    # x.data.shape
    "DITN": (
        lambda: DITN(dim=12, ITL_blocks=1, SAL_blocks=1, UFONE_blocks=1, upscale=2),
        [(37, 51), (48, 80), (80, 48), (130, 72)],
    ),
    # torch.Tensor(batch, height, width) for its hue
    "HVICIDNet": (
        lambda: HVICIDNet(channels=[12, 12, 24, 48], heads=[1, 2, 4, 8]),
        [(37, 51), (48, 80), (80, 48), (130, 72)],
    ),
    # int(np.ceil(h / 8) * 8 - h)
    "FBCNN": (
        FBCNN,
        [(37, 51), (41, 48), (48, 83), (130, 72)],
    ),
    # int(math.log(H // 16, 4)), which matters at a side of 1024
    "RGT": (
        lambda: RGT(embed_dim=36, depth=[3], num_heads=[2], upscale=2),
        [(37, 51), (1020, 64), (1024, 64), (64, 1024), (1088, 72)],
    ),
    # int(math.sqrt(tokens)), which matters above its padded square of 128
    "Uformer": (
        lambda: Uformer(embed_dim=8, depths=[1, 1, 1, 1, 1, 1, 1, 1, 1], win_size=8),
        [(96, 40), (130, 72), (144, 200)],
    ),
}
# The export traces a 32x32 image; 37x51 checks that height and width stay dynamic.
SIZES = [(32, 32), (37, 51)]
# MixDehazeNet needs at least 40x40: its graph must pad smaller sizes (also beyond
# reflect padding's limit) and must not crop larger ones.
MIN_40_SIZES = [(16, 30), (37, 51), (40, 40), (48, 56)]


def tiny_model(arch: str) -> ImageModelDescriptor:
    """A randomly initialised model, prepared as PyTorch's Load Model node does."""
    torch.manual_seed(0)
    model = ModelLoader().load_from_state_dict(ARCHS[arch]().state_dict())
    assert isinstance(model, ImageModelDescriptor)
    for _, v in model.model.named_parameters():
        v.requires_grad = False
    return model.eval()


def torch_output(model: ImageModelDescriptor, image: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(image)[None]).numpy()[0]


def images(sizes: list[tuple[int, int]]):
    rng = np.random.default_rng(0)
    return [rng.random((3, h, w), dtype=np.float32) for h, w in sizes]


@pytest.mark.parametrize("net", [SAFMNBCIE, FFTformer])
def test_what_the_exporter_cannot_write_is_refused_by_name(net: type[torch.nn.Module]):
    model = cast(ModelDescriptor, SimpleNamespace(model=net.__new__(net)))
    assert not is_onnx_supported(model)
    assert is_onnx_supported(tiny_model("SwinIR"))


@pytest.mark.parametrize(
    ("arch", "sizes"),
    [
        ("Compact", SIZES),
        ("SPAN", SIZES),
        ("ESRGAN", SIZES),
        ("MixDehazeNet", MIN_40_SIZES),
        ("HAT", SIZES),
        ("DAT", SIZES),
        ("PLKSR", SIZES),
        ("OmniSR", SIZES),
        ("SwinIR", SIZES),
        ("Swin2SR", SIZES),
        ("RGT", SIZES),
        ("DRCT", SIZES),
    ],
)
def test_convert_to_onnx_matches_pytorch(arch: str, sizes: list[tuple[int, int]]):
    model = tiny_model(arch)
    session = ort.InferenceSession(
        convert_to_onnx_impl(model, torch.device("cpu")),
        providers=["CPUExecutionProvider"],
    )
    for image in images(sizes):
        (actual,) = session.run(None, {"input": image[None]})
        np.testing.assert_allclose(
            np.asarray(actual)[0], torch_output(model, image), rtol=0, atol=1e-5
        )


@pytest.mark.parametrize("arch", ["GFISRv2", "FIGSR"])
@pytest.mark.parametrize("opset", [17, 18, 19, 20])
def test_a_fourier_export_loads_and_matches_pytorch(arch: str, opset: int):
    model = tiny_model(arch)
    session = ort.InferenceSession(
        convert_to_onnx_impl(model, torch.device("cpu"), opset_version=opset),
        providers=["CPUExecutionProvider"],
    )
    # Sizes that are and are not powers of two, and an odd one the model pads to even
    for image in images([(32, 32), (37, 51), (48, 64)]):
        expected = torch_output(model, image)
        assert ((expected > 0) & (expected < 1)).mean() > 0.5
        (actual,) = session.run(None, {"input": image[None]})
        np.testing.assert_allclose(np.asarray(actual)[0], expected, rtol=0, atol=1e-4)


def dft_axes(onnx_bytes: bytes) -> set[int]:
    """Each DFT's axis, counted from the end, in an export whose DFTs have 5 dimensions."""
    graph = onnx.load_from_string(onnx_bytes).graph
    constants = {
        node.output[0]: int(onnx.numpy_helper.to_array(node.attribute[0].t))
        for node in graph.node
        if node.op_type == "Constant" and not node.attribute[0].t.dims
    }
    axes: set[int] = set()
    for node in graph.node:
        if node.op_type == "DFT":
            attributes = {a.name: a.i for a in node.attribute}
            axis = attributes.get("axis", constants.get(node.input[-1]))
            assert axis is not None
            axes.add(axis - 5)
    return axes


@pytest.mark.parametrize("arch", ["GFISRv2", "FIGSR"])
@pytest.mark.parametrize("opset", [17, 20])
def test_a_fourier_export_keeps_its_dfts_on_the_axis_tensorrt_takes(
    arch: str, opset: int
):
    # TensorRT 11's parser: "FFTPlugin supports DFT only on axis -2"
    data = convert_to_onnx_impl(
        tiny_model(arch), torch.device("cpu"), opset_version=opset
    )
    assert dft_axes(data) == {-2}


@pytest.mark.parametrize("arch", ["GFISRv2", "FIGSR"])
def test_a_fourier_export_in_fp16_matches_pytorch(arch: str):
    model = tiny_model(arch)
    session = ort.InferenceSession(
        convert_to_onnx_impl(model, torch.device("cpu"), True, opset_version=20),
        providers=["CPUExecutionProvider"],
    )
    for image in images([(32, 32), (37, 51)]):
        (actual,) = session.run(None, {"input": image[None].astype(np.float16)})
        np.testing.assert_allclose(
            np.asarray(actual)[0].astype(np.float32),
            torch_output(model, image),
            rtol=0,
            atol=2e-2,
        )


def output_types(onnx_bytes: bytes) -> tuple[onnx.GraphProto, dict[str, int]]:
    """An export's graph and the element type of each of its values."""
    model = onnx.shape_inference.infer_shapes(onnx.load_from_string(onnx_bytes))
    types = {
        node.output[0]: node.attribute[0].t.data_type
        for node in model.graph.node
        if node.op_type == "Constant"
    }
    for value in (*model.graph.value_info, *model.graph.input, *model.graph.output):
        types[value.name] = value.type.tensor_type.elem_type
    return model.graph, types


@pytest.mark.parametrize("arch", ["GFISRv2", "FIGSR"])
def test_an_fp16_fourier_export_is_fp16_but_for_its_transforms_and_norms(arch: str):
    graph, types = output_types(
        convert_to_onnx_impl(
            tiny_model(arch), torch.device("cpu"), True, opset_version=20
        )
    )
    assert types["input"] == types["output"] == onnx.TensorProto.FLOAT16
    for node in graph.node:
        if node.op_type == "Conv":
            kinds = {types[name] for name in node.input}
            assert kinds == {onnx.TensorProto.FLOAT16}, node.name
        elif node.op_type in ("DFT", "ReduceL2"):
            assert types[node.input[0]] == onnx.TensorProto.FLOAT, node.name


@pytest.mark.parametrize("arch", ["GFISRv2", "FIGSR"])
@pytest.mark.parametrize("size", [1e-5, 1e-3, 1.0, 1e3, 1e5])
def test_an_fp16_fourier_norm_holds_what_fp16_cannot_square(arch: str, size: float):
    model = tiny_model(arch)
    export = fourier_export.for_export(copy.deepcopy(model).half(), 20)
    unit = next(
        m for m in model.model.modules() if isinstance(m, fourier_export.FOURIER_UNITS)
    )
    name = next(n for n, m in model.model.named_modules() if m is unit)
    norm = export.model.get_submodule(name).rn
    x = torch.randn(1, unit.rn.scale.numel(), 4, 4) * size
    with torch.no_grad():
        np.testing.assert_allclose(
            norm(x).float().numpy(), unit.rn(x).numpy(), rtol=0, atol=2e-3
        )


@pytest.mark.parametrize("arch", ["GFISRv2", "FIGSR"])
def test_a_fourier_export_below_opset_17_is_refused(arch: str):
    with pytest.raises(ValueError, match="needs Opset 17"):
        convert_to_onnx_impl(tiny_model(arch), torch.device("cpu"), opset_version=16)


@pytest.mark.parametrize("arch", ["GFISRv2", "FIGSR"])
def test_convert_to_onnx_verifies_a_fourier_export(arch: str):
    model = tiny_model(arch)
    assert convert(model, verify=True)[2] == "opset20"


@pytest.mark.parametrize("arch", ["GFISRv2", "FIGSR"])
def test_the_fourier_rewrite_works_on_a_copy(arch: str):
    model = tiny_model(arch)
    export = fourier_export.for_export(model, 20)
    assert export.model is not model.model
    names = [type(m).__name__ for m in model.model.modules()]
    assert names.count("CustomRfft2Wrap") == names.count("CustomIrfft2Wrap") == 2
    image = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(export(image), model(image))


def lda_model(arch: str, scale: int) -> ImageModelDescriptor:
    torch.manual_seed(0)
    model = ModelLoader().load_from_state_dict(LDA[arch](scale).state_dict())
    assert isinstance(model, ImageModelDescriptor)
    assert any("proj_q" in key for key in model.model.state_dict())
    model.model.requires_grad_(False)
    return model.eval()


@pytest.mark.parametrize("arch", LDA)
@pytest.mark.parametrize("scale", [2, 3, 4])
def test_an_lda_export_follows_the_input_size(arch: str, scale: int):
    model = lda_model(arch, scale)
    session = ort.InferenceSession(
        convert_to_onnx_impl(model, torch.device("cpu"), opset_version=20),
        providers=["CPUExecutionProvider"],
    )
    # Not the 32x32 example, wide and tall
    for image in images([(37, 51), (48, 80), (80, 48), (16, 64)]):
        expected = torch_output(model, image)
        (actual,) = session.run(None, {"input": image[None]})
        assert np.asarray(actual)[0].shape == expected.shape
        np.testing.assert_allclose(np.asarray(actual)[0], expected, rtol=0, atol=1e-4)


@pytest.mark.parametrize("arch", TRACED_SIZE)
def test_an_export_follows_the_input_size(arch: str):
    make, sizes = TRACED_SIZE[arch]
    torch.manual_seed(0)
    model = ModelLoader().load_from_state_dict(make().state_dict())
    assert isinstance(model, ImageModelDescriptor)
    model.model.requires_grad_(False)
    model.eval()
    session = ort.InferenceSession(
        convert_to_onnx_impl(model, torch.device("cpu"), opset_version=20),
        providers=["CPUExecutionProvider"],
    )
    for image in images(sizes):
        expected = torch_output(model, image)
        (actual,) = session.run(None, {"input": image[None]})
        assert np.asarray(actual)[0].shape == expected.shape
        np.testing.assert_allclose(np.asarray(actual)[0], expected, rtol=0, atol=1e-4)


def test_a_seemore_export_runs_the_expert_each_image_picks():
    torch.manual_seed(0)
    make = SeemoRe(scale=2, num_experts=3, num_layers=1, embedding_dim=16)
    model = ModelLoader().load_from_state_dict(make.state_dict())
    assert isinstance(model, ImageModelDescriptor)
    model.model.requires_grad_(False)
    model.eval()
    layer = next(m for m in model.model.modules() if type(m) is MoELayer)
    # A router that picks the first expert for one image's features, the second for the
    # other's
    bright = np.full((3, 37, 51), 1.0, np.float32)
    dark = np.zeros((3, 48, 32), np.float32)
    features: list[torch.Tensor] = []
    hook = layer.gate.register_forward_pre_hook(
        lambda _, inputs: features.append(inputs[0].mean((2, 3))[0])
    )
    torch_output(model, bright)
    torch_output(model, dark)
    hook.remove()
    router = layer.gate.body[2].weight
    router.zero_()
    router[0] = features[0] / features[0].norm()
    router[1] = features[1] / features[1].norm()
    picks: list[int] = []
    hook = layer.gate.register_forward_hook(
        lambda _, __, out: picks.append(int(out.argmax()))
    )
    expected = [torch_output(model, bright), torch_output(model, dark)]
    hook.remove()
    assert picks == [0, 1]

    classes = [type(m) for m in model.model.modules()]
    session = ort.InferenceSession(
        convert_to_onnx_impl(model, torch.device("cpu"), opset_version=20),
        providers=["CPUExecutionProvider"],
    )
    assert [type(m) for m in model.model.modules()] == classes
    for image, want in zip([bright, dark], expected):
        (actual,) = session.run(None, {"input": image[None]})
        np.testing.assert_allclose(np.asarray(actual)[0], want, rtol=0, atol=1e-5)


def fp16_export_session(model: ImageModelDescriptor) -> ort.InferenceSession:
    """An FP16 export of `model`, whatever spandrel says of its FP16 in PyTorch."""
    source = copy.deepcopy(model)
    source.supports_half = True
    half = convert_to_onnx_impl(source, torch.device("cpu"), True, opset_version=20)
    return ort.InferenceSession(half, providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_an_fp16_retinexformer_holds_on_a_flat_image(value: float):
    model = tiny_model("RetinexFormer")
    image = np.full((3, 32, 32), value, np.float32)
    session = fp16_export_session(model)
    (actual,) = session.run(None, {"input": image[None].astype(np.float16)})
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(
        np.asarray(actual, np.float32)[0], torch_output(model, image), atol=0.02
    )


def model_with_a_product_over_every_pixel(arch: str) -> ImageModelDescriptor:
    return dysample_model(arch) if arch == "FDAT" else tiny_model(arch)


@pytest.mark.parametrize(
    ("arch", "attention"),
    [
        ("RetinexFormer", IG_MSA),
        ("FDAT", FastChannelAttention),
        ("DAT", Adaptive_Channel_Attention),
    ],
)
def test_an_fp16_export_takes_a_product_over_every_pixel_in_fp32(
    arch: str, attention: type[torch.nn.Module]
):
    model = model_with_a_product_over_every_pixel(arch)
    model.supports_half = True
    half = convert_to_onnx_impl(model, torch.device("cpu"), True, opset_version=20)
    graph = onnx.shape_inference.infer_shapes(onnx.load_from_string(half)).graph
    kinds = {v.name: v.type.tensor_type.elem_type for v in graph.value_info}
    products = [kinds[n.output[0]] for n in graph.node if n.op_type == "MatMul"]
    attentions = sum(isinstance(m, attention) for m in model.model.modules())
    assert products.count(onnx.TensorProto.FLOAT) == attentions
    assert onnx.TensorProto.FLOAT16 in products


@pytest.mark.parametrize("arch", ["RetinexFormer", "FDAT", "DAT"])
def test_the_fp16_rewrite_computes_the_same_lengths_on_a_copy(arch: str):
    model = model_with_a_product_over_every_pixel(arch)
    assert fp16_export.for_export(model) is model
    model.supports_half = True
    model.half()
    classes = [type(m) for m in model.model.modules()]
    export = fp16_export.for_export(model)
    assert export.model is not model.model
    assert [type(m) for m in model.model.modules()] == classes
    assert [type(m) for m in export.model.modules()] != classes
    model.float()
    export.float()
    image = torch.rand(1, 3, 32, 48)
    with torch.no_grad():
        torch.testing.assert_close(export(image), model(image), atol=1e-5, rtol=0)


class _FixedRouter(torch.nn.Module):
    """A router whose logits are fixed: it reads the features' mean only for their type,
    their batch and a place in the graph."""

    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.register_buffer("logits", torch.tensor([logits]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.get_buffer("logits")
        return x.mean((1, 2, 3), keepdim=True).flatten(1) * 0 + logits.to(x.dtype)


def seemore_with_router(logits: list[float]) -> ImageModelDescriptor:
    torch.manual_seed(0)
    make = SeemoRe(scale=2, num_experts=3, num_layers=1, embedding_dim=16)
    model = ModelLoader().load_from_state_dict(make.state_dict())
    assert isinstance(model, ImageModelDescriptor)
    model.model.requires_grad_(False)
    model.eval()
    layer = next(m for m in model.model.modules() if type(m) is MoELayer)
    layer.gate = _FixedRouter(logits)
    return model


TIED_LOGITS = [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]


@pytest.mark.parametrize("logits", TIED_LOGITS)
def test_a_seemore_export_runs_the_one_expert_topk_picks_of_a_tie(logits: list[float]):
    model = seemore_with_router(logits)
    image = images([(37, 51)])[0]
    expected = torch_output(model, image)
    with torch.no_grad():
        output = traced_size_export.for_export(model)(torch.from_numpy(image)[None])
    assert torch.equal(output[0], torch.from_numpy(expected))
    for opset in (14, 20):
        session = ort.InferenceSession(
            convert_to_onnx_impl(model, torch.device("cpu"), opset_version=opset),
            providers=["CPUExecutionProvider"],
        )
        (actual,) = session.run(None, {"input": image[None]})
        np.testing.assert_allclose(np.asarray(actual)[0], expected, rtol=0, atol=1e-5)


# The last: apart in FP32 and tied in FP16, as topk reads them after their rounding
FP16_TIED_LOGITS = [*TIED_LOGITS, [0.25, 0.25 + 2.0**-12, 0.0]]


@pytest.mark.parametrize("logits", FP16_TIED_LOGITS)
def test_an_fp16_seemore_copy_runs_the_one_expert_topk_picks_of_a_tie(
    logits: list[float],
):
    model = seemore_with_router(logits)
    model.supports_half = True
    model.half()
    weights = torch.tensor([logits]).softmax(dim=1).half()
    assert int((weights == weights.max()).sum()) > 1
    image = torch.from_numpy(images([(37, 51)])[0])[None].half()
    with torch.no_grad():
        expected = model(image)
        actual = traced_size_export.for_export(model)(image)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("opset", [14, 20])
def test_an_fp16_seemore_export_runs_one_expert_of_a_tie(opset: int):
    model = seemore_with_router([1.0, 1.0, 0.0])
    half = copy.deepcopy(model)
    half.supports_half = True
    session = ort.InferenceSession(
        convert_to_onnx_impl(half, torch.device("cpu"), True, opset_version=opset),
        providers=["CPUExecutionProvider"],
    )
    image = images([(37, 51)])[0]
    (actual,) = session.run(None, {"input": image[None].astype(np.float16)})
    np.testing.assert_allclose(
        np.asarray(actual, np.float32)[0], torch_output(model, image), atol=0.02
    )


def test_uformers_sides_are_the_same_and_restored():
    torch.manual_seed(0)
    make = TRACED_SIZE["Uformer"][0]
    model = ModelLoader().load_from_state_dict(make().state_dict())
    assert isinstance(model, ImageModelDescriptor)
    model.model.requires_grad_(False)
    model.eval()
    module = sys.modules[type(model.model).__module__]
    image = torch.rand(1, 3, 130, 72)
    with torch.no_grad():
        expected = model(image.clone())
        with traced_size_export.uformer_sides(model):
            assert "int" in vars(module)
            assert torch.equal(model(image.clone()), expected)
    assert "int" not in vars(module)
    assert module.math is math


@pytest.mark.parametrize("arch", ["GFISRv2", "FIGSR"])
def test_convert_to_ncnn_refuses_a_fourier_model(arch: str):
    with pytest.raises(AssertionError, match="not supported for NCNN"):
        convert_to_ncnn_node(
            cast(NodeContext, Context(CPU_CONTEXT)), tiny_model(arch), False
        )


def test_uformer_exports_at_once_wait_for_each_others_sides():
    torch.manual_seed(0)
    make = TRACED_SIZE["Uformer"][0]
    model = ModelLoader().load_from_state_dict(make().state_dict())
    assert isinstance(model, ImageModelDescriptor)
    module = sys.modules[type(model.model).__module__]
    first_inside, first_may_leave = threading.Event(), threading.Event()
    second_inside, second_may_leave = threading.Event(), threading.Event()
    errors: list[BaseException] = []

    def run(inside: threading.Event, may_leave: threading.Event) -> None:
        try:
            with traced_size_export.uformer_sides(model):
                inside.set()
                assert may_leave.wait(60)
        except BaseException as e:
            errors.append(e)

    first = threading.Thread(target=run, args=(first_inside, first_may_leave))
    second = threading.Thread(target=run, args=(second_inside, second_may_leave))
    first.start()
    assert first_inside.wait(60)
    second.start()
    # While the first export holds them, the second does not enter
    assert not second_inside.wait(0.5)
    first_may_leave.set()
    first.join(60)
    assert second_inside.wait(60)
    # The first export's leaving took nothing from under the second
    assert "int" in vars(module)
    assert module.math is not math
    second_may_leave.set()
    second.join(60)
    assert errors == []
    assert "int" not in vars(module)
    assert module.math is math


@pytest.mark.parametrize(
    ("arch", "half"),
    [
        ("DITN", False),
        ("DITN", True),
        ("HVICIDNet", False),
        ("FBCNN", False),
        ("FBCNN", True),
        ("RGT", False),
    ],
)
def test_the_size_rewrite_computes_the_same_values_on_a_copy(arch: str, half: bool):
    torch.manual_seed(0)
    model = ModelLoader().load_from_state_dict(TRACED_SIZE[arch][0]().state_dict())
    assert isinstance(model, ImageModelDescriptor)
    model.model.requires_grad_(False)
    model.eval()
    if half:
        model.half()
    classes = [type(m) for m in model.model.modules()]
    export = traced_size_export.for_export(model)
    assert export.model is not model.model
    assert [type(m) for m in model.model.modules()] == classes
    assert [type(m) for m in export.model.modules()] != classes
    # With values at each channel's extreme, flat areas and black for HVICIDNet's cases
    image = torch.rand(1, 3, 48, 80, dtype=model.dtype)
    image[:, :, :8] = 0
    image[:, :, 8:16] = 0.5
    image[:, 0, 16:24] = 1
    image[:, 1, 24:32] = 1
    image[:, 2, 32:40] = 1
    with torch.no_grad():
        assert torch.equal(export(image.clone()), model(image.clone()))


@pytest.mark.parametrize("arch", LDA)
@pytest.mark.parametrize("half", [False, True])
def test_the_lda_rewrite_computes_the_same_values_on_a_copy(arch: str, half: bool):
    model = lda_model(arch, 2)
    if half:
        model.half()
    export = lda_export.for_export(model)
    assert export.model is not model.model
    names = [type(m).__name__ for m in model.model.modules()]
    assert names.count("LDA_AQU") == 1
    image = torch.rand(1, 3, 48, 80, dtype=model.dtype)
    with torch.no_grad():
        assert torch.equal(export(image), model(image))


def test_the_fourier_rewrite_leaves_other_models_as_they_are():
    model = tiny_model("Compact")
    assert fourier_export.for_export(model, 14) is model


@pytest.mark.parametrize("arch", ["HAT", "DAT"])
def test_the_attention_rewrite_works_on_a_copy(arch: str):
    model = tiny_model(arch)
    weights = {k: v.clone() for k, v in model.model.state_dict().items()}
    export = attention_export.for_export(model)
    assert export.model is not model.model
    state = model.model.state_dict()
    assert state.keys() == weights.keys()
    assert all(torch.equal(state[k], v) for k, v in weights.items())
    image = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(export(image), model(image), rtol=0, atol=1e-6)


DYSAMPLE_ARCHS: dict[str, Callable[[], torch.nn.Module]] = {
    "RealPLKSR": lambda: RealPLKSR(
        dim=16, n_blocks=2, upscaling_factor=2, kernel_size=7, dysample=True
    ),
    "FDAT": lambda: FDAT(
        scale=2,
        embed_dim=24,
        num_groups=1,
        depth_per_group=2,
        mid_dim=16,
        upsampler_type="dysample",
    ),
    "GaterV3": lambda: GateRV3(
        dim=16,
        enc_blocks=(1, 1, 1, 1),
        dec_blocks=(1, 1, 1, 1),
        num_latent=1,
        upsample="dysample",
        upsample_mid_dim=16,
        sisr_blocks=1,
        upsample_version=2,
    ),
    "SMoSR": lambda: SMoSR(dim=16, n_mb=1, upsampler="dysample", upsampler_mid_dim=16),
    "MoESR": lambda: MoESR(
        dim=16, scale=2, n_blocks=1, n_block=1, upsampler="dysample", upsample_dim=16
    ),
}


def dysample_model(arch: str) -> ImageModelDescriptor:
    torch.manual_seed(0)
    model = ModelLoader().load_from_state_dict(DYSAMPLE_ARCHS[arch]().state_dict())
    assert isinstance(model, ImageModelDescriptor)
    return model.eval()


@pytest.mark.parametrize("arch", DYSAMPLE_ARCHS)
def test_dysample_exports_its_grid_for_every_size(arch: str):
    # Its grid's size must not stay the traced example's. GridSample needs opset 16.
    model = dysample_model(arch)
    session = ort.InferenceSession(
        convert_to_onnx_impl(model, torch.device("cpu"), opset_version=20),
        providers=["CPUExecutionProvider"],
    )
    for image in images([(32, 32), (37, 51), (64, 48)]):
        (actual,) = session.run(None, {"input": image[None]})
        np.testing.assert_allclose(
            np.asarray(actual)[0], torch_output(model, image), rtol=0, atol=1e-4
        )
    # Nor does Load Model's check see a frozen grid in what the node exports.
    assert model_facts(convert(model)[0]).frozen_grid is None


@pytest.mark.parametrize(
    "dysample", [DySample, FdatDySample, GaterDySample, GfisrDySample, SmosrDySample]
)
def test_every_copy_of_dysample_traces_its_grid_for_every_size(
    dysample: Callable[..., torch.nn.Module],
):
    torch.manual_seed(0)
    source = dysample(16, 3, 2, 4).eval().requires_grad_(False)
    holder = cast(ModelDescriptor, SimpleNamespace(model=torch.nn.Sequential(source)))
    export = dysample_export.for_export(holder)
    traced = cast(
        torch.nn.Module, torch.jit.trace(export.model, torch.rand(1, 16, 32, 32))
    )
    image = torch.rand(1, 16, 37, 51)
    torch.testing.assert_close(traced(image), source(image), rtol=0, atol=1e-5)


def test_verify_passes_a_dysample_export():
    # Verify runs a size other than the trace's example; RealPLKSR converts in FP16.
    assert convert(dysample_model("RealPLKSR"), verify=True)[1] == "fp16"


def frozen_like_spandrel_before_c14(monkeypatch: pytest.MonkeyPatch) -> None:
    """Spandrel before 0.4.2+c14 read DySample's size as `torch.tensor([W, H])`, which a
    trace stores as a constant: what the rewrite is for on a runtime not yet updated."""
    monkeypatch.setattr(
        torch, "_shape_as_tensor", lambda tensor: torch.tensor(tensor.shape)
    )


def test_verify_fails_a_size_frozen_from_the_trace(monkeypatch: pytest.MonkeyPatch):
    # A requirement that pads Verify's smaller size to the trace's example must not hide
    # it: without the rewrite, DySample's grid keeps that size.
    model = dysample_model("RealPLKSR")
    model.size_requirements = SizeRequirements(minimum=32)
    device = torch.device("cpu")
    convert_to_onnx.export_verified(model, device, False, 20, True)
    monkeypatch.setattr(dysample_export, "for_export", lambda model: model)
    frozen_like_spandrel_before_c14(monkeypatch)
    with pytest.raises(AssertionError):
        convert_to_onnx.export_verified(model, device, False, 20, True)


def test_the_dysample_rewrite_works_on_a_copy():
    model = dysample_model("RealPLKSR")
    weights = {k: v.clone() for k, v in model.model.state_dict().items()}
    export = dysample_export.for_export(model)
    assert export.model is not model.model
    state = model.model.state_dict()
    assert state.keys() == weights.keys()
    assert all(torch.equal(state[k], v) for k, v in weights.items())
    image = torch.rand(1, 3, 37, 51)
    with torch.no_grad():
        assert torch.equal(export(image), model(image))


def test_a_model_without_dysample_exports_as_it_is():
    model = tiny_model("Compact")
    assert dysample_export.for_export(model) is model


TWO_GROUPS: dict[str, Callable[[], torch.nn.Module]] = {
    "HAT": lambda: HAT(
        embed_dim=12,
        depths=(2, 2),
        num_heads=(2, 2),
        window_size=8,
        compress_ratio=2,
        squeeze_factor=4,
        upsampler="pixelshuffle",
        upscale=2,
    ),
    "DAT": lambda: DAT(
        embed_dim=24,
        depth=[3, 3],
        num_heads=[4, 4],
        upsampler="pixelshuffle",
        upscale=2,
    ),
}
SMOOTH: dict[str, np.ndarray] = {
    "black": np.zeros((1, 3, 32, 32), np.float32),
    "white": np.ones((1, 3, 32, 32), np.float32),
    "gradient": np.linspace(0, 1, 32, dtype=np.float32)[None, None, None, :]
    .repeat(32, axis=2)
    .repeat(3, axis=1),
}


def outgrowing_fp16(arch: str) -> ImageModelDescriptor:
    """A two-group model whose second-to-last group writes past FP16's 65504, as trained
    HAT and DAT models do on smooth tiles and random weights do not."""
    torch.manual_seed(0)
    model = ModelLoader().load_from_state_dict(TWO_GROUPS[arch]().state_dict())
    assert isinstance(model, ImageModelDescriptor)
    conv = model.model.layers[0].conv
    with torch.no_grad():
        conv.weight *= 1e5
        peak = conv(torch.ones(1, conv.in_channels, 8, 8)).abs().max()
    assert peak > 65504
    return model.eval()


@pytest.mark.parametrize("arch", list(TWO_GROUPS))
@pytest.mark.parametrize("image", list(SMOOTH))
def test_an_fp16_export_holds_a_group_past_fp16_on_a_smooth_image(
    arch: str, image: str
):
    model = outgrowing_fp16(arch)
    half = convert_to_onnx_impl(model, torch.device("cpu"), True, opset_version=20)
    full = convert_to_onnx_impl(model, torch.device("cpu"), False, opset_version=20)
    x = SMOOTH[image]
    (actual,) = ort.InferenceSession(half, providers=["CPUExecutionProvider"]).run(
        None, {"input": x.astype(np.float16)}
    )
    (expected,) = ort.InferenceSession(full, providers=["CPUExecutionProvider"]).run(
        None, {"input": x}
    )
    assert np.isfinite(expected).all()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual.astype(np.float32), expected, atol=0.05)


def test_only_an_fp16_export_runs_the_last_groups_in_fp32():
    model = outgrowing_fp16("DAT")
    plain = attention_export.for_export(model).model
    assert not isinstance(plain.layers[0].conv, FP32Conv2d)
    rewritten = attention_export.for_export(model.half()).model
    assert isinstance(rewritten.layers[0].conv, FP32Conv2d)
    assert isinstance(rewritten.layers[1].blocks[0].norm1, FP32LayerNorm)
    # The groups before keep FP16's norms, and the loaded model is as it was.
    assert not isinstance(rewritten.layers[0].blocks[0].norm1, FP32LayerNorm)
    assert not isinstance(model.model.layers[0].conv, FP32Conv2d)
    assert not isinstance(model.model.layers[1].blocks[0].norm1, FP32LayerNorm)


# Each shifts its windows in a block of its only group, in a mask from calculate_mask
SWIN = ["SwinIR", "Swin2SR", "RGT", "DRCT"]
SWIN_SIZES = [(32, 32), (40, 56), (64, 48)]


def swin_in_fp16(model: ImageModelDescriptor) -> ImageModelDescriptor:
    """The copy export_copy makes for an FP16 export, past its refusal of these
    architectures, which do not claim FP16 for PyTorch's sake."""
    allowed = copy.copy(model)
    allowed.supports_half = True
    return export_copy(allowed, torch.device("cpu"), use_half=True)


def swin_closing_convolution(arch: str, net: torch.nn.Module) -> torch.nn.Conv2d:
    group = net.layers[-1]
    if arch == "DRCT":
        return group.adjust5
    return group.patch_embed.proj if arch == "Swin2SR" else group.conv


def stored_at(arch: str, img_size: int, window_size: int = 8) -> ImageModelDescriptor:
    """A tiny model that stored its masks at `img_size`, as a checkpoint trained there."""
    swin = {"embed_dim": 12, "num_heads": [2], "window_size": window_size}
    swin |= {"upsampler": "pixelshuffle", "upscale": 2, "img_size": img_size}
    net = {
        "SwinIR": lambda: SwinIR(depths=[2], **swin),
        "Swin2SR": lambda: Swin2SR(depths=[2], **swin),
        "DRCT": lambda: DRCT(depths=[5], gc=4, **swin),
        "DAT": lambda: DAT(
            embed_dim=24,
            depth=[3],
            num_heads=[4],
            upsampler="pixelshuffle",
            upscale=2,
            img_size=img_size,
        ),
    }[arch]()
    torch.manual_seed(0)
    model = ModelLoader().load_from_state_dict(net.state_dict())
    assert isinstance(model, ImageModelDescriptor)
    model.model.requires_grad_(False)
    return model.eval()


@pytest.mark.parametrize("arch", ["SwinIR", "Swin2SR", "DRCT", "DAT"])
@pytest.mark.parametrize("use_half", [False, True])
def test_a_mask_stored_at_the_examples_size_does_not_stay_in_the_export(
    arch: str, use_half: bool
):
    model = stored_at(arch, 32)
    copied = (
        swin_in_fp16(model) if use_half else export_copy(model, torch.device("cpu"))
    )
    session = ort.InferenceSession(
        export_onnx(copied, opset_version=20), providers=["CPUExecutionProvider"]
    )
    # 40x56 has other windows than the stored mask's; 64x64 as many in another place
    for h, w in [(32, 32), (40, 56), (48, 32), (64, 64), (64, 48), (96, 32)]:
        x = np.linspace(0, 1, h * w, dtype=np.float32).reshape(h, w)
        x = np.repeat(x[None], 3, axis=0)
        feed = x.astype(np.float16) if use_half else x
        (actual,) = session.run(None, {"input": feed[None]})
        np.testing.assert_allclose(
            actual[0].astype(np.float32),
            torch_output(model, x),
            atol=0.02 if use_half else 1e-5,
        )


@pytest.mark.parametrize(
    ("arch", "img_size", "window_size", "side"),
    [
        ("SwinIR", 32, 8, 33),
        ("SwinIR", 32, 16, 33),
        # Its example pads to the window: 35 of 32
        ("SwinIR", 35, 7, 36),
        ("Swin2SR", 32, 8, 33),
        ("DRCT", 32, 8, 33),
        ("DAT", 32, 8, 33),
        # Every other stored size traces what it always did, for the same bytes
        ("SwinIR", 16, 8, 32),
        ("SwinIR", 48, 8, 32),
        ("SwinIR", 64, 8, 32),
        ("SwinIR", 128, 8, 32),
        ("SwinIR", 56, 7, 32),
        ("Swin2SR", 64, 8, 32),
        ("DRCT", 64, 8, 32),
        ("DAT", 64, 8, 32),
    ],
)
def test_an_example_leaves_a_stored_size(
    arch: str, img_size: int, window_size: int, side: int
):
    assert _example_side(stored_at(arch, img_size, window_size)) == side


def test_an_example_stays_as_it_was_for_every_other_model():
    assert _example_side(tiny_model("ESRGAN")) == 32
    assert _example_side(tiny_model("HAT")) == 32
    assert _example_side(tiny_model("RGT")) == 32
    upcunet = ModelLoader().load_from_state_dict(UpCunet2x().state_dict())
    assert _example_side(upcunet) == 33


def float64_operators(data: bytes) -> list[str]:
    graph = onnx.shape_inference.infer_shapes(onnx.load_from_string(data)).graph
    types = {v.name: v.type.tensor_type.elem_type for v in graph.value_info}
    return [
        node.op_type
        for node in graph.node
        if any(types.get(name) == onnx.TensorProto.DOUBLE for name in node.output)
    ]


@pytest.mark.parametrize("arch", [*SWIN, "HAT", "DAT", "OmniSR", "RetinexFormer"])
@pytest.mark.parametrize("opset", [14, 16])
def test_an_fp16_export_below_opset_17_holds_no_float64(arch: str, opset: int):
    model = tiny_model(arch)
    data = export_onnx(swin_in_fp16(model), opset_version=opset)
    assert float64_operators(data) == []
    session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    # Random weights amplify FP16's rounding, so this checks the computation, not precision
    x = np.linspace(0, 1, 40 * 56, dtype=np.float32).reshape(40, 56)
    x = np.repeat(x[None], 3, axis=0)
    # OmniSR takes its input in FP32
    fp16 = session.get_inputs()[0].type == "tensor(float16)"
    feed = x.astype(np.float16) if fp16 else x
    (actual,) = session.run(None, {"input": feed[None]})
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(
        actual[0].astype(np.float32), torch_output(model, x), atol=0.03
    )


def test_layer_norms_stay_as_they_are_but_for_fp16_below_opset_17():
    fp32 = tiny_model("SwinIR")
    assert fp16_export.for_opset(fp32, 14) is fp32
    fp16 = swin_in_fp16(fp32)
    assert fp16_export.for_opset(fp16, 17) is fp16
    rewritten = fp16_export.for_opset(fp16, 16)
    norm1 = rewritten.model.layers[0].residual_group.blocks[0].norm1
    assert isinstance(norm1, FP32LayerNorm)
    norm1 = fp16.model.layers[0].residual_group.blocks[0].norm1
    assert type(norm1) is torch.nn.LayerNorm


@pytest.mark.parametrize("arch", SWIN)
def test_a_swin_export_in_fp32_matches_pytorch_and_is_not_rewritten(arch: str):
    model = tiny_model(arch)
    assert swin_export.for_export(model) is model
    session = ort.InferenceSession(
        convert_to_onnx_impl(model, torch.device("cpu"), opset_version=20),
        providers=["CPUExecutionProvider"],
    )
    for image in images(SWIN_SIZES):
        (actual,) = session.run(None, {"input": image[None]})
        np.testing.assert_allclose(
            np.asarray(actual)[0], torch_output(model, image), rtol=0, atol=1e-5
        )


@pytest.mark.parametrize("arch", SWIN)
@pytest.mark.parametrize("opset", [14, 20])
def test_a_swin_traces_in_fp16_and_follows_the_input_size(arch: str, opset: int):
    model = tiny_model(arch)
    data = export_onnx(swin_in_fp16(model), opset_version=opset)
    session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].type == "tensor(float16)"
    # Random weights amplify FP16's rounding, so this checks the windows, not precision
    for h, w in SWIN_SIZES:
        x = np.linspace(0, 1, h * w, dtype=np.float32).reshape(h, w)
        x = np.repeat(x[None], 3, axis=0)
        (actual,) = session.run(None, {"input": x[None].astype(np.float16)})
        expected = torch_output(model, x)
        assert actual.shape[1:] == expected.shape
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual[0].astype(np.float32), expected, atol=0.02)


@pytest.mark.parametrize("arch", SWIN)
def test_the_swin_rewrite_computes_the_same_values_in_fp32(arch: str):
    model = tiny_model(arch)
    rewritten = swin_export.for_export(swin_in_fp16(model))
    assert rewritten.model.layers[-1] is not model.model.layers[-1]
    rewritten.model.float()
    rewritten.model.load_state_dict(model.model.state_dict())
    for h, w in SWIN_SIZES:
        image = torch.rand(1, 3, h, w)
        with torch.no_grad():
            assert torch.equal(rewritten(image), model(image))


@pytest.mark.parametrize("arch", SWIN)
def test_the_swin_rewrite_leaves_the_loaded_model_as_it_was(arch: str):
    model = tiny_model(arch)
    types = {name: type(m) for name, m in model.model.named_modules()}
    weights = {k: v.clone() for k, v in model.model.state_dict().items()}
    export_onnx(swin_in_fp16(model), opset_version=20)
    assert {name: type(m) for name, m in model.model.named_modules()} == types
    state = model.model.state_dict()
    assert state.keys() == weights.keys()
    assert all(
        v.dtype == weights[k].dtype and torch.equal(v, weights[k])
        for k, v in state.items()
    )
    assert not model.supports_half


@pytest.mark.parametrize("arch", SWIN)
def test_the_end_of_a_swins_last_group_is_fp32_in_the_graph(arch: str):
    data = export_onnx(swin_in_fp16(tiny_model(arch)), opset_version=20)
    graph = onnx.shape_inference.infer_shapes(onnx.load_from_string(data)).graph
    types = {v.name: v.type.tensor_type.elem_type for v in graph.value_info}
    conv = [types[n.output[0]] for n in graph.node if n.op_type == "Conv"]
    norm = [types[n.output[0]] for n in graph.node if n.op_type == "LayerNormalization"]
    # Swin2SR's group ends in its closing convolution and its patch embedding's
    closing = 2 if arch == "Swin2SR" else 1
    assert conv.count(onnx.TensorProto.FLOAT) == closing
    assert conv.count(onnx.TensorProto.FLOAT16) == len(conv) - closing
    assert onnx.TensorProto.FLOAT in norm
    # The norm in FP32 hands the rest of the model FP16 again
    (output,) = graph.output
    assert output.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    assert onnx.TensorProto.FLOAT in [
        types.get(n.output[0]) for n in graph.node if n.op_type == "Add"
    ]


def test_a_swins_three_closing_convolutions_all_run_in_fp32():
    torch.manual_seed(0)
    net = SwinIR(
        embed_dim=12,
        depths=[2, 2],
        num_heads=[2, 2],
        window_size=8,
        mlp_ratio=2,
        upsampler="pixelshuffle",
        upscale=2,
        resi_connection="3conv",
    )
    model = ModelLoader().load_from_state_dict(net.state_dict()).eval()
    assert isinstance(model, ImageModelDescriptor)
    rewritten = swin_export.for_export(swin_in_fp16(model)).model
    last = [m for m in rewritten.layers[-1].conv if isinstance(m, torch.nn.Conv2d)]
    assert len(last) == 3
    assert all(isinstance(m, FP32Conv2d) for m in last)
    # Only the last group's: the groups before stay in range
    assert not any(isinstance(m, FP32Conv2d) for m in rewritten.layers[0].modules())


@pytest.mark.parametrize("arch", SWIN)
@pytest.mark.parametrize("image", list(SMOOTH))
def test_a_swin_holds_a_last_group_past_fp16_on_a_smooth_image(arch: str, image: str):
    model = tiny_model(arch)
    conv = swin_closing_convolution(arch, model.model)
    with torch.no_grad():
        conv.weight *= 1e5
    x = SMOOTH[image]
    data = export_onnx(swin_in_fp16(model), opset_version=20)
    (actual,) = ort.InferenceSession(data, providers=["CPUExecutionProvider"]).run(
        None, {"input": x.astype(np.float16)}
    )
    expected = torch_output(model, x[0])
    assert np.isfinite(expected).all()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual[0].astype(np.float32), expected, atol=0.05)


def test_hat_heads_are_padded_to_a_multiple_of_8():
    export = attention_export.for_export(tiny_model("HAT"))
    attention = [
        m for m in export.model.modules() if isinstance(m, WindowAttention | OCAB)
    ]
    assert len(attention) == 3  # two blocks' window attention and the OCAB
    for module in attention:
        assert module.qkv.out_features == 3 * module.num_heads * 8
        assert module.proj.in_features == module.num_heads * 8


def test_dat_spatial_heads_are_padded_to_a_multiple_of_8():
    export = attention_export.for_export(tiny_model("DAT"))
    attention = [
        m for m in export.model.modules() if isinstance(m, Adaptive_Spatial_Attention)
    ]
    assert len(attention) == 2  # the first and third blocks; the second is channel
    for module in attention:
        assert module.qkv.out_features == 3 * module.num_heads * 8
        # DAT drops the padded channels itself before its output projection
        assert module.proj.in_features == 24


def test_convert_model_with_trainable_weights():
    # PyTorch's Interpolate Models node returns a freshly loaded model, whose weights
    # still require grad (Load Model turns that off).
    torch.manual_seed(0)
    model = ModelLoader().load_from_state_dict(ARCHS["Compact"]().state_dict())
    assert isinstance(model, ImageModelDescriptor)
    assert any(p.requires_grad for p in model.model.parameters())
    session = ort.InferenceSession(
        convert_to_onnx_impl(model, torch.device("cpu")),
        providers=["CPUExecutionProvider"],
    )
    (image,) = images([(37, 51)])
    (actual,) = session.run(None, {"input": image[None]})
    np.testing.assert_allclose(
        np.asarray(actual)[0], torch_output(model, image), rtol=0, atol=1e-5
    )


# MixDehazeNet's graph pads with shape operations, which NCNN does not support.
@pytest.mark.parametrize("arch", ["Compact", "ESRGAN"])
def test_convert_to_ncnn_matches_pytorch(arch: str, tmp_path: Path):
    # Runs native code, which GitHub's checks do not build.
    pytest.importorskip("nodes.impl._chainner_graph")
    # PyTorch's Convert To NCNN: an fp32 ONNX intermediate, then the ONNX converter.
    model = tiny_model(arch)
    onnx_bytes = convert_to_onnx_impl(model, torch.device("cpu"), False, "data")
    converted = Onnx2NcnnConverter(
        safely_optimize_onnx_model(onnx.load_model_from_string(onnx_bytes))
    ).convert(False, False)
    wrapper = NcnnModelWrapper(converted)
    assert wrapper.scale == model.scale

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.use_fp16_packed = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    converted.write_bin(tmp_path / "model.bin")
    assert net.load_param_mem(converted.write_param()) == 0
    assert net.load_model(str(tmp_path / "model.bin")) == 0
    for image in images(SIZES):
        ex = net.create_extractor()
        assert ex.input(converted.layers[0].outputs[0], ncnn.Mat(image)) == 0
        ret, actual = ex.extract(converted.layers[-1].outputs[0])
        assert ret == 0
        np.testing.assert_allclose(
            np.array(actual), torch_output(model, image), rtol=0, atol=1e-4
        )


# What an export could change on the model it converts: dtype, device, mode, grad
# flags and weights.
Snapshot = tuple[torch.dtype, torch.device, bool, list[bool], dict[str, torch.Tensor]]


def snapshot(model: ImageModelDescriptor) -> Snapshot:
    return (
        model.dtype,
        model.device,
        model.model.training,
        [p.requires_grad for p in model.model.parameters()],
        {k: v.clone() for k, v in model.model.state_dict().items()},
    )


def assert_unchanged(model: ImageModelDescriptor, before: Snapshot):
    after = snapshot(model)
    assert after[:4] == before[:4]
    assert after[4].keys() == before[4].keys()
    assert all(torch.equal(after[4][k], v) for k, v in before[4].items())


@pytest.mark.parametrize("arch", ["Compact", "HAT"])
@pytest.mark.parametrize("loaded_half", [False, True])
@pytest.mark.parametrize("use_half", [False, True])
def test_converting_leaves_the_model_unchanged(
    arch: str, loaded_half: bool, use_half: bool
):
    # Load Model halves a model with PyTorch's FP16 on; Upscale Image and Save Model
    # share it with Convert To ONNX.
    model = tiny_model(arch)
    if loaded_half:
        model.half()
    before = snapshot(model)
    convert_to_onnx_impl(model, torch.device("cpu"), use_half)
    assert_unchanged(model, before)


def test_converting_leaves_trainable_weights_trainable():
    torch.manual_seed(0)
    model = ModelLoader().load_from_state_dict(ARCHS["Compact"]().state_dict())
    assert isinstance(model, ImageModelDescriptor)
    before = snapshot(model)
    convert_to_onnx_impl(model, torch.device("cpu"))
    assert_unchanged(model, before)


class Context:
    """The NodeContext members Convert To ONNX uses: PyTorch on the CPU."""

    def __init__(self, settings: dict[str, object], node_id: str = "node") -> None:
        self.settings = SettingsParser(settings)
        self.node_id = NodeId(node_id)
        self.warnings: list[str] = []
        # How many more times Stop is not pressed when the node looks
        self.looks_until_stop: float = math.inf

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def check_aborted(self) -> None:
        self.looks_until_stop -= 1
        if self.looks_until_stop < 0:
            raise Aborted()


CPU_CONTEXT: dict[str, object] = {"use_cpu": True}


def convert(
    model: ImageModelDescriptor,
    data_type: OnnxDataType = OnnxDataType.FP32,
    opset: Opset = Opset.OPSET_20,
    verify: bool = False,
    dual_shape: DualShape = DualShape.DYNAMIC,
    settings: dict[str, object] = CPU_CONTEXT,
    node_id: str = "node",
    context: Context | None = None,
) -> tuple[OnnxGeneric, str, str]:
    return convert_to_onnx_node(
        cast(NodeContext, context or Context(settings, node_id)),
        model,
        data_type,
        opset,
        verify,
        dual_shape,
        512,
        512,
    )


def skip_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """The FP16 check's verdict stays out of a test on something else: random weights can
    saturate on any image, as trained ones do not."""
    monkeypatch.setattr(convert_to_onnx, "verify_models", lambda *args: None)


def count_exports(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Counts Convert To ONNX's exports (one entry each) and skips the FP16 check."""
    exports: list[int] = []
    export_onnx = convert_to_onnx.export_onnx

    def counted(model: ModelDescriptor, opset_version: int) -> bytes:
        exports.append(opset_version)
        return export_onnx(model, opset_version=opset_version)

    monkeypatch.setattr(convert_to_onnx, "export_onnx", counted)
    skip_the_check(monkeypatch)
    return exports


def straight_to_fp32(monkeypatch: pytest.MonkeyPatch, arch: str = PLAIN) -> str:
    """Takes `arch` off the list of FP16 architectures; its name there."""
    name = tiny_model(arch).architecture.name
    monkeypatch.setattr(
        onnx_model_module,
        "FP16_ARCHS",
        {arch: sizes for arch, sizes in FP16_ARCHS.items() if arch != name},
    )
    return name


def test_a_rerun_returns_the_kept_export(monkeypatch: pytest.MonkeyPatch):
    exports = count_exports(monkeypatch)
    model = tiny_model(PLAIN)
    first = convert(model)
    for _ in range(2):
        again = convert(model)
        assert again == first and again[0] is first[0]
    assert len(exports) == 1


def test_each_export_input_has_its_own_export(monkeypatch: pytest.MonkeyPatch):
    exports = count_exports(monkeypatch)
    model = tiny_model(PLAIN)
    other = tiny_model(PLAIN)  # the same weights in another model
    runs = [
        lambda: convert(model),
        lambda: convert(model, opset=Opset.OPSET_17),
        lambda: convert(other),
    ]
    results = [run() for run in runs]
    assert len(exports) == len(runs)
    assert len({id(result[0]) for result in results}) == len(runs)
    # Each model keeps its latest export
    assert runs[-2]()[0] is results[-2][0]
    assert runs[-1]()[0] is results[-1][0]
    assert len(exports) == len(runs)


def test_a_changed_export_input_drops_the_superseded_export(
    monkeypatch: pytest.MonkeyPatch,
):
    # A kept export holds its ONNX sessions (GPU memory) while the model lives.
    count_exports(monkeypatch)
    model = tiny_model(PLAIN)
    first = weakref.ref(convert(model)[0])
    second = weakref.ref(convert(model, opset=Opset.OPSET_17)[0])
    gc.collect()
    assert first() is None
    model.half()
    convert(model)
    gc.collect()
    assert second() is None


def test_two_nodes_converting_one_model_keep_their_exports(
    monkeypatch: pytest.MonkeyPatch,
):
    exports = count_exports(monkeypatch)
    model = tiny_model(PLAIN)
    first = convert(model, node_id="first")
    second = convert(model, opset=Opset.OPSET_17, node_id="second")
    for _ in range(2):
        assert convert(model, node_id="first")[0] is first[0]
        assert convert(model, opset=Opset.OPSET_17, node_id="second")[0] is second[0]
    assert exports == [20, 17]


def test_two_nodes_with_the_same_options_share_one_export(
    monkeypatch: pytest.MonkeyPatch,
):
    # One ONNX model, so one session (and one TensorRT engine) downstream; a node added
    # in place of a deleted one reuses its export.
    exports = count_exports(monkeypatch)
    model = tiny_model(PLAIN)
    first = convert(model, node_id="first")
    assert convert(model, node_id="second")[0] is first[0]
    assert convert(model, node_id="third")[0] is first[0]
    assert exports == [20]


def test_a_model_changed_in_place_is_exported_again(monkeypatch: pytest.MonkeyPatch):
    # A model's precision can change in place (half()), which its export depends on.
    exports = count_exports(monkeypatch)
    model = tiny_model(PLAIN)
    first = convert(model)[0]
    model.half()
    assert convert(model)[0] is not first
    assert len(exports) == 2


def test_verify_checks_an_fp32_export_kept_without_it(monkeypatch: pytest.MonkeyPatch):
    exports = count_exports(monkeypatch)
    straight_to_fp32(monkeypatch)
    verified: list[torch.dtype] = []
    monkeypatch.setattr(
        convert_to_onnx,
        "verify_models",
        lambda model, onnx_model, fp16, device, stop: verified.append(model.dtype),
    )
    model = tiny_model(PLAIN)
    model.half()
    convert(model)
    convert(model, verify=True)  # the kept export, verified
    convert(model, verify=True)
    convert(model)
    assert len(exports) == 1
    # Verified against the loaded model as it is: verify_models takes its own FP32
    # reference from it.
    assert verified == [torch.float16]


def test_an_fp16_export_is_checked_once_with_verify_on_or_off(
    monkeypatch: pytest.MonkeyPatch,
):
    exports = count_exports(monkeypatch)
    checked: list[bool] = []
    monkeypatch.setattr(
        convert_to_onnx,
        "verify_models",
        lambda model, onnx_model, fp16, device, stop: checked.append(fp16),
    )
    model = tiny_model(PLAIN)
    first = convert(model)
    assert convert(model, verify=True)[0] is first[0]
    assert convert(model)[0] is first[0]
    assert (first[1], exports, checked) == ("fp16", [20], [True])


class DualModel:
    """The members of a DUAL model Convert To ONNX uses."""

    def __init__(self) -> None:
        self.architecture = SimpleNamespace(id="DUAL")
        self.dtype = torch.float32
        self.model = torch.nn.Conv2d(3, 3, 1)
        self.tags: list[str] = []
        self.scale = 2


def test_a_dual_rerun_returns_the_kept_export(monkeypatch: pytest.MonkeyPatch):
    exports: list[tuple[int, int] | None] = []
    onnx_bytes = convert_to_onnx_impl(tiny_model("Compact"), torch.device("cpu"))

    def export(
        state: object,
        tags: object,
        scale: object,
        size: tuple[int, int] | None,
        context: object,
    ) -> bytes:
        exports.append(size)
        return onnx_bytes

    monkeypatch.setattr(convert_to_onnx.dual, "export_onnx", export)
    model = cast(ImageModelDescriptor, DualModel())
    dynamic = convert(model)
    assert convert(model)[0] is dynamic[0]
    fixed = convert(model, dual_shape=DualShape.FIXED)
    assert convert(model, dual_shape=DualShape.FIXED)[0] is fixed[0]
    assert exports == [None, (512, 512)]


class GpuSettings(PyTorchSettings):
    """PyTorch's settings as a GPU has them. The CPU stands in for the GPU here (CUDA
    is hidden), so FP16 skips the CPU's fallback to FP32."""

    def __post_init__(self) -> None:
        pass


# Load Model's settings with PyTorch's FP16 on and off.
FP16 = GpuSettings(use_cpu=False, use_fp16=True, gpu_index=0, budget_limit=0)
FP32 = PyTorchSettings(use_cpu=False, use_fp16=False, gpu_index=0, budget_limit=0)


def model_file(directory: Path, make: Callable[[], torch.nn.Module]) -> Path:
    torch.manual_seed(0)
    path = directory / "model.pth"
    torch.save(make().state_dict(), path)
    return path


def warnings_logged(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    logged: list[str] = []
    monkeypatch.setattr(
        load_model_module.logger,
        "warning",
        lambda message, *args: logged.append(message % args),
    )
    return logged


def test_an_export_of_a_model_loaded_in_fp16_carries_its_fp32_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    count_exports(monkeypatch)
    straight_to_fp32(monkeypatch)
    logged = warnings_logged(monkeypatch)

    path = model_file(tmp_path, ARCHS[PLAIN])
    half = load_model(path, FP16)
    assert isinstance(half, ImageModelDescriptor) and half.dtype == torch.float16
    full = load_model(path, FP32)
    assert isinstance(full, ImageModelDescriptor) and full.dtype == torch.float32
    rounded = convert_to_onnx_impl(half, torch.device("cpu"), opset_version=20)
    before = snapshot(half)
    exported = convert(half)[0].bytes
    assert exported == convert(full)[0].bytes
    assert exported != rounded
    assert_unchanged(half, before)
    assert logged == []


def test_a_model_converted_back_to_fp32_in_place_still_exports_its_file_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Upscale Image converts the shared model to PyTorch's precision in place.
    count_exports(monkeypatch)
    straight_to_fp32(monkeypatch)
    path = model_file(tmp_path, ARCHS[PLAIN])
    model = load_model(path, FP16)
    assert isinstance(model, ImageModelDescriptor)
    model.float()
    full = load_model(path, FP32)
    assert isinstance(full, ImageModelDescriptor)
    assert convert(model)[0].bytes == convert(full)[0].bytes


def test_an_fp16_export_of_a_model_loaded_in_fp16_is_checked_against_its_file_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # The check's reference and the fallback's weights are the file's, not the rounded ones.
    checked: list[ImageModelDescriptor] = []
    monkeypatch.setattr(
        convert_to_onnx,
        "verify_models",
        lambda model, onnx_model, fp16, device, stop: checked.append(model),
    )
    path = model_file(tmp_path, ARCHS[PLAIN])
    half = load_model(path, FP16)
    full = load_model(path, FP32)
    assert isinstance(half, ImageModelDescriptor)
    assert isinstance(full, ImageModelDescriptor)
    exported = convert(half)
    assert exported[1] == "fp16"
    assert exported[0].bytes == convert(full)[0].bytes
    source = checked[0].model.state_dict()
    for name, weight in full.model.state_dict().items():
        assert source[name].dtype == weight.dtype
        assert torch.equal(source[name], weight)


def rewrite(path: Path) -> None:
    """Other weights in `path`, a second later (a rewrite within the file system's
    time resolution keeps the identity; the weights' comparison catches that)."""
    before = path.stat()
    torch.manual_seed(1)
    torch.save(ARCHS[PLAIN]().state_dict(), path)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 10**9))


class UnreadableLoader:
    """spandrel's ModelLoader for a file it may not read (another program locked it)."""

    def __init__(self, device: torch.device) -> None:
        pass

    def load_from_file(self, path: Path) -> ModelDescriptor:
        raise PermissionError("Permission denied")


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("none", "does not come from Load Model"),
        ("rewritten", "changed after Load Model loaded it"),
        ("deleted", "changed after Load Model loaded it"),
        ("same identity", "no longer holds the weights Load Model loaded"),
        ("unreadable", "could not be read again (Permission denied)"),
    ],
)
def test_without_its_file_weights_the_fp16_weights_are_exported_and_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str, reason: str
):
    count_exports(monkeypatch)
    name = straight_to_fp32(monkeypatch)
    logged = warnings_logged(monkeypatch)
    if change == "none":
        model = tiny_model(PLAIN)  # as Interpolate Models returns one
        model.half()
    else:
        if change == "same identity":
            monkeypatch.setattr(load_model_module, "file_identity", lambda path: 0)
        path = model_file(tmp_path, ARCHS[PLAIN])
        loaded = load_model(path, FP16)
        assert isinstance(loaded, ImageModelDescriptor)
        model = loaded
        if change == "deleted":
            path.unlink()
        elif change == "unreadable":
            monkeypatch.setattr(load_model_module, "ModelLoader", UnreadableLoader)
        else:
            rewrite(path)
    rounded = convert_to_onnx_impl(model, torch.device("cpu"), opset_version=20)
    assert convert(model)[0].bytes == with_recorded_arch(rounded, name, "fp32")
    assert len(logged) == 1 and reason in logged[0]
    assert logged[0].startswith("Convert To ONNX: ")
    assert "FP16-rounded weights" in logged[0]


def test_a_dual_model_loaded_in_fp16_exports_its_fp32_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    states: list[dict[str, torch.Tensor]] = []
    onnx_bytes = convert_to_onnx_impl(tiny_model("Compact"), torch.device("cpu"))

    def export(
        state: dict[str, torch.Tensor],
        tags: object,
        scale: object,
        size: object,
        context: object,
    ) -> bytes:
        states.append(state)
        return onnx_bytes

    monkeypatch.setattr(convert_to_onnx.dual, "export_onnx", export)
    path = model_file(tmp_path, lambda: dual_xs(scale=1))
    half = load_model(path, FP16)
    full = load_model(path, FP32)
    assert isinstance(half, ImageModelDescriptor) and half.dtype == torch.float16
    assert isinstance(full, ImageModelDescriptor)
    convert(half)
    (state,) = states
    expected = full.model.state_dict()
    assert state.keys() == expected.keys()
    assert all(
        state[k].dtype == v.dtype and torch.equal(state[k], v)
        for k, v in expected.items()
    )


def tiny_legacy_drft() -> torch.nn.Module:
    """A small random legacy DRFT x2 with LayerNorm (DRFT's TensorRT route)."""
    return legacy.DRFT(
        embed_dim=48,
        depths=(6, 6),
        num_heads=(6, 6),
        window_size=16,
        overlap_ratio=0.5,
        upscale=2,
        num_feat=16,
        rank=8,
        drop_path_rate=0.0,
    )


def test_a_legacy_drft_loaded_in_fp16_exports_its_fp32_weights_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    states: list[dict[str, torch.Tensor]] = []
    devices: list[torch.device | None] = []
    onnx_bytes = convert_to_onnx_impl(tiny_model("Compact"), torch.device("cpu"))

    def export(model: ImageModelDescriptor, device: torch.device | None) -> bytes:
        states.append(model.model.state_dict())
        devices.append(device)
        return onnx_bytes

    monkeypatch.setattr(convert_to_onnx.drft_export, "export_onnx", export)
    path = model_file(tmp_path, tiny_legacy_drft)
    half = load_model(path, FP16)
    full = load_model(path, FP32)
    assert isinstance(half, ImageModelDescriptor) and half.dtype == torch.float16
    assert isinstance(full, ImageModelDescriptor)
    first = convert(half)
    assert first[1:] == ("mixed", "opset20")
    assert convert(half)[0] is first[0]
    (state,) = states
    expected = full.model.state_dict()
    assert state.keys() == expected.keys()
    assert all(
        state[k].dtype == v.dtype and torch.equal(state[k], v)
        for k, v in expected.items()
    )
    # traced where Load Model put the model (FP16 is slow on the CPU)
    assert devices == [half.device]


def to_ncnn(
    model: ImageModelDescriptor, is_fp16: int, directory: Path
) -> tuple[str, bytes]:
    """PyTorch's Convert To NCNN of `model`: its param and bin."""
    converted, _ = convert_to_ncnn_node(
        cast(NodeContext, Context(CPU_CONTEXT)), model, is_fp16
    )
    path = directory / "model.bin"
    converted.model.write_bin(path)
    return converted.model.write_param(), path.read_bytes()


@pytest.mark.parametrize("is_fp16", [0, 1], ids=["fp32", "fp16"])
def test_an_ncnn_conversion_of_a_model_loaded_in_fp16_uses_its_fp32_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, is_fp16: int
):
    # Runs native code, which GitHub's checks do not build.
    pytest.importorskip("nodes.impl._chainner_graph")
    logged = warnings_logged(monkeypatch)
    path = model_file(tmp_path, ARCHS[PLAIN])
    half = load_model(path, FP16)
    full = load_model(path, FP32)
    assert isinstance(half, ImageModelDescriptor) and half.dtype == torch.float16
    assert isinstance(full, ImageModelDescriptor)
    converted = to_ncnn(half, is_fp16, tmp_path)
    assert converted == to_ncnn(full, is_fp16, tmp_path)
    assert logged == []
    # The FP16 weights alone (a model Load Model did not load) convert as before. An
    # FP16 conversion differs as well: the converter rounds FP32 weights to FP16 its
    # own way, not as PyTorch rounded them at load.
    rounded = to_ncnn(copy.deepcopy(half), is_fp16, tmp_path)
    assert rounded != converted
    assert len(logged) == 1 and logged[0].startswith("Convert To NCNN: ")
    assert "does not come from Load Model" in logged[0]


# NCNN's layers have fixed shapes; these models take shapes from the image's size as they
# run (DRFT's windows, NAFNet's padding to a multiple). Each failed with the
# converter's bare IndexError or KeyError, or an export error.
NCNN_REFUSED: dict[str, Callable[[], torch.nn.Module]] = {
    "DRFT": lambda: drft_arch.DRFT(
        embed_dim=48,
        depths=(2,),
        num_heads=(6,),
        window_size=8,
        upscale=2,
        num_feat=16,
        rank=8,
        drop_path_rate=0.0,
    ),
    "legacy DRFT": tiny_legacy_drft,
    "legacy DRFT with i-LN": lambda: legacy.DRFT(
        embed_dim=48,
        depths=(2,),
        num_heads=(6,),
        window_size=8,
        upscale=2,
        num_feat=16,
        rank=8,
        drop_path_rate=0.0,
        use_iln=True,
    ),
}


@pytest.mark.parametrize("arch", NCNN_REFUSED)
def test_ncnn_refuses_models_that_take_shapes_from_the_image(arch: str, tmp_path: Path):
    torch.manual_seed(0)
    model = ModelLoader().load_from_state_dict(NCNN_REFUSED[arch]().state_dict())
    assert isinstance(model, ImageModelDescriptor)
    name = model.architecture.name
    with pytest.raises(AssertionError, match=f"^{name} is not supported for NCNN"):
        to_ncnn(model, 0, tmp_path)


# Those of the FP16 architectures with a tiny model here
FP16_MODELS = [
    "ESRGAN",
    "HAT",
    "DAT",
    "Compact",
    "SPAN",
    "PLKSR",
    "OmniSR",
    PLAIN,
    "MixDehazeNet",
    "FIGSR",
    "RGT",
]
# An architecture off the list, as one spandrel adds later is
FP32_MODEL = "GFISRv2"
# Those whose image enters an FP16 export in FP32, up to the first convolution
FP32_INPUT_MODELS = ["Compact", "SPAN", "PLKSR", "OmniSR"]
BRIGHT_RAMP = (
    np.linspace(0.98, 1, 32, dtype=np.float32)[None, None, None, :]
    .repeat(32, axis=2)
    .repeat(3, axis=1)
)


def test_the_fp16_architectures_are_named_as_a_models_type_carries_them():
    names = {support.architecture.name for support in MAIN_REGISTRY.architectures()}
    assert set(FP16_ARCHS) <= names
    assert {tiny_model(arch).architecture.name for arch in FP16_MODELS} <= set(
        FP16_ARCHS
    )
    assert tiny_model(FP32_MODEL).architecture.name not in FP16_ARCHS


@pytest.mark.parametrize(
    ("arch", "tags", "fp16"),
    [
        ("ESRGAN", [], True),
        ("ESRGAN", ["64nf", "large"], True),
        ("SwinIR", ["small", "60dim"], True),
        ("SwinIR", ["medium"], True),
        ("SwinIR", ["large"], True),
        ("SwinIR", [], False),
        ("FDAT", ["120dim", "medium"], True),
        ("FDAT", ["180dim", "xl"], True),
        ("FDAT", ["96dim", "tiny"], False),
        ("Swin2SR", ["medium"], True),
        ("Swin2SR", ["large"], False),
        ("RGT", [], True),
        ("DRCT", ["large"], True),
        ("DRCT", [], False),
        ("KBNet", [], False),
        ("One spandrel adds later", ["small"], False),
    ],
)
def test_fp16_goes_by_the_architecture_and_the_sizes_measured(
    arch: str, tags: list[str], fp16: bool
):
    assert converts_in_fp16(arch, tags) is fp16


@pytest.mark.parametrize(("tag", "precision"), [("small", "fp16"), ("large", "fp32")])
def test_a_size_not_measured_converts_in_fp32_with_no_word(
    tag: str, precision: str, monkeypatch: pytest.MonkeyPatch
):
    skip_the_check(monkeypatch)
    model = tiny_model(PLAIN)
    monkeypatch.setattr(
        onnx_model_module, "FP16_ARCHS", {model.architecture.name: ("small",)}
    )
    model.tags = [tag]
    context = Context(CPU_CONTEXT)
    onnx_model, fp_mode, _ = convert(model, context=context)
    assert fp_mode == precision
    assert context.warnings == []
    # The file says its precision: the tags are not read again.
    monkeypatch.setattr(onnx_model_module, "FP16_ARCHS", {})
    loaded = load_onnx_model(onnx_model.bytes)
    assert (loaded.info.arch, loaded.info.dtype) == (
        model.architecture.name,
        precision,
    )


@pytest.mark.parametrize("hidden", [OnnxDataType.FP32, OnnxDataType.FP16])
def test_an_architecture_off_the_list_converts_in_fp32_with_no_word(
    hidden: OnnxDataType, monkeypatch: pytest.MonkeyPatch
):
    exports = count_exports(monkeypatch)
    model = tiny_model(FP32_MODEL)
    # Also with PyTorch's FP16 setting on: FP16 is not on offer.
    settings: dict[str, object] = {"use_cpu": True, "use_fp16": True}
    context = Context(settings)
    onnx_model, fp_mode, _ = convert(model, hidden, context=context)
    assert (fp_mode, onnx_model.info.dtype) == ("fp32", "fp32")
    assert exports == [20]
    assert context.warnings == []
    assert recorded(onnx_model) == {
        ARCH_METADATA_KEY: model.architecture.name,
        PRECISION_METADATA_KEY: "fp32",
    }
    loaded = load_onnx_model(onnx_model.bytes)
    assert (loaded.info.arch, loaded.info.dtype) == (model.architecture.name, "fp32")


@pytest.mark.parametrize("arch", FP16_MODELS)
@pytest.mark.parametrize("hidden", [OnnxDataType.FP32, OnnxDataType.FP16])
def test_an_architecture_ignores_a_hidden_data_type(
    arch: str, hidden: OnnxDataType, monkeypatch: pytest.MonkeyPatch
):
    # On the CPU PyTorch's FP16 setting is off, which does not refuse these.
    exports = count_exports(monkeypatch)
    onnx_model, fp_mode, opset = convert(tiny_model(arch), hidden)
    assert (fp_mode, opset) == ("fp16", "opset20")
    assert onnx_model.info.dtype == "fp16"
    assert onnx_model.info.arch == tiny_model(arch).architecture.name
    assert exports == [20]


def test_realplksr_converts_in_fp16_and_its_model_stays_as_it_was():
    model = tiny_model("PLKSR")
    assert not model.supports_half
    onnx_model = convert(model, verify=True)[0]
    assert onnx_model.info.dtype == "fp16"
    assert not model.supports_half
    assert model.dtype == torch.float32


def test_the_original_plksr_converts_in_fp16():
    torch.manual_seed(0)
    state = PLKSR(dim=16, n_blocks=2, upscaling_factor=2).state_dict()
    model = ModelLoader().load_from_state_dict(state)
    assert isinstance(model, ImageModelDescriptor)
    model.model.requires_grad_(False)
    assert convert(model.eval())[0].info.arch == "PLKSR"


@pytest.mark.parametrize(
    ("build", "module"),
    [
        (lambda: RealPLKSR(dim=16, n_blocks=2, upscaling_factor=2), RealPLKConv2d),
        (lambda: PLKSR(dim=16, n_blocks=2, upscaling_factor=2), PLKConv2d),
        (
            lambda: PLKSR(
                dim=16, n_blocks=2, upscaling_factor=2, lk_type="RectSparsePLK"
            ),
            RectSparsePLKConv2d,
        ),
    ],
    ids=["RealPLKSR", "PLKSR", "RectSparsePLK"],
)
def test_plksr_exports_without_a_scatter_and_with_the_same_values(
    build: Callable[[], torch.nn.Module], module: type[torch.nn.Module]
):
    torch.manual_seed(0)
    model = ModelLoader().load_from_state_dict(build().state_dict())
    assert isinstance(model, ImageModelDescriptor)
    model.model.requires_grad_(False)
    model.eval()
    assert any(isinstance(m, module) for m in model.model.modules())
    data = convert_to_onnx_impl(model, torch.device("cpu"))
    ops = {node.op_type for node in onnx.load_from_string(data).graph.node}
    assert "ScatterND" not in ops and "Range" not in ops
    session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    for image in images(SIZES):
        (actual,) = session.run(None, {"input": image[None]})
        np.testing.assert_allclose(
            np.asarray(actual)[0], torch_output(model, image), rtol=0, atol=1e-5
        )


def test_realplksr_samples_in_fp32_in_an_fp16_export():
    torch.manual_seed(0)
    state = RealPLKSR(
        dim=16, n_blocks=2, upscaling_factor=2, dysample=True
    ).state_dict()
    model = ModelLoader().load_from_state_dict(state)
    assert isinstance(model, ImageModelDescriptor)
    model.model.requires_grad_(False)
    model.eval()
    # The sampler's weights are the loaded FP32 ones, not FP16 roundings of them.
    export = convert_to_onnx.export_copy(model, torch.device("cpu"), use_half=True)
    loaded = model.model.to_img.state_dict()
    for name, weight in export.model.to_img.source.state_dict().items():
        assert weight.dtype == torch.float32
        assert torch.equal(weight, loaded[name])
    onnx_model = convert(model)[0]
    assert onnx_model.info.dtype == "fp16"
    graph = onnx.shape_inference.infer_shapes(
        onnx.load_from_string(onnx_model.bytes)
    ).graph
    types = {v.name: v.type.tensor_type.elem_type for v in graph.value_info}
    samples = [n for n in graph.node if n.op_type == "GridSample"]
    assert samples
    for node in samples:
        assert [types[name] for name in node.input] == [onnx.TensorProto.FLOAT] * 2
    session = ort.InferenceSession(onnx_model.bytes, providers=["CPUExecutionProvider"])
    # Two sizes: the grid must follow the input, not the export's example.
    for image in images([(32, 32), (48, 56)]):
        actual = np.asarray(session.run(None, {"input": image[None]})[0])
        assert actual.dtype == np.float16
        np.testing.assert_allclose(
            actual[0].astype(np.float32), torch_output(model, image), rtol=0, atol=5e-3
        )


def test_the_plksr_rewrite_works_on_a_copy():
    model = tiny_model("PLKSR")
    rewritten = plksr_export.for_export(model)
    assert isinstance(model.model.feats[1].lk, RealPLKConv2d)
    assert not isinstance(rewritten.model.feats[1].lk, RealPLKConv2d)
    other = tiny_model("Compact")
    assert plksr_export.for_export(other) is other


def test_an_architecture_shares_one_export_across_hidden_data_types(
    monkeypatch: pytest.MonkeyPatch,
):
    exports = count_exports(monkeypatch)
    model = tiny_model(PLAIN)
    first = convert(model, OnnxDataType.FP32)
    assert convert(model, OnnxDataType.FP16)[0] is first[0]
    assert exports == [20]


@pytest.mark.parametrize("arch", ["ESRGAN", PLAIN])
def test_an_architecture_exports_fp16_with_the_fp16_setting_off(arch: str):
    # With the check, and with a model spandrel marks unfit for FP16 or not.
    settings: dict[str, object] = {"use_cpu": True, "use_fp16": False}
    model = tiny_model(arch)
    model.supports_half = False
    onnx_model, fp_mode, _ = convert(model, settings=settings)
    assert (fp_mode, onnx_model.info.dtype) == ("fp16", "fp16")
    assert not model.supports_half and model.dtype == torch.float32


# Random weights make SPAN saturate on any image; trained ones verify as these do.
@pytest.mark.parametrize("arch", ["ESRGAN", "Compact", PLAIN])
def test_an_architecture_verifies_on_the_cpu(arch: str):
    assert convert(tiny_model(arch), verify=True)[1] == "fp16"


def overflowing_fp16() -> torch.nn.Module:
    """A Compact with weights FP16 cannot hold, which FP32 runs well: its first convolution
    scales up by what its last scales down. Its FP16 ONNX gives infinite values."""
    torch.manual_seed(0)
    net = ARCHS["Compact"]()
    with torch.no_grad():
        convs = [m for m in net.modules() if isinstance(m, torch.nn.Conv2d)]
        convs[0].weight *= 1e6
        convs[-1].weight /= 1e6
    return net


def overflowing_model() -> ImageModelDescriptor:
    model = ModelLoader().load_from_state_dict(overflowing_fp16().state_dict())
    assert isinstance(model, ImageModelDescriptor)
    model.model.requires_grad_(False)
    return model.eval()


def recorded(onnx_model: OnnxGeneric) -> dict[str, str]:
    keys = (ARCH_METADATA_KEY, PRECISION_METADATA_KEY)
    entries = onnx.load_from_string(onnx_model.bytes).metadata_props
    return {e.key: e.value for e in entries if e.key in keys}


def test_a_model_that_fails_the_check_converts_in_fp32_with_verify_off():
    model = overflowing_model()
    before = snapshot(model)
    name = model.architecture.name
    context = Context(CPU_CONTEXT)
    onnx_model, fp_mode, _ = convert(model, verify=False, context=context)
    assert fp_mode == "fp32"
    assert onnx_model.info.dtype == "fp32"
    assert onnx_model.info.arch == name
    assert recorded(onnx_model) == {
        ARCH_METADATA_KEY: name,
        PRECISION_METADATA_KEY: "fp32",
    }
    assert len(context.warnings) == 1
    assert f"This {name} model converted in FP32" in context.warnings[0]
    assert "NaN or infinite" in context.warnings[0]
    # What the user receives is right: the FP32 model's own output.
    session = ort.InferenceSession(onnx_model.bytes, providers=["CPUExecutionProvider"])
    image = images([(16, 16)])[0]
    output = session.run(None, {"input": image[None]})[0][0]
    np.testing.assert_allclose(output, torch_output(model, image), rtol=0, atol=1e-3)
    assert_unchanged(model, before)
    # A saved file stays seamless.
    saved = load_onnx_model(onnx_model.bytes)
    assert (saved.info.arch, saved.info.dtype) == (name, "fp32")


def overflowing_inside() -> ImageModelDescriptor:
    """A Compact whose weights all fit FP16 and whose output FP32 and FP16 agree on, but whose
    middle layers write past FP16's 65504: two convolutions scale up by what the next two
    scale down."""
    torch.manual_seed(0)
    net = ARCHS["Compact"]()
    with torch.no_grad():
        convs = [m for m in net.modules() if isinstance(m, torch.nn.Conv2d)]
        assert len(convs) == 4
        convs[0].weight *= 1e3
        convs[1].weight *= 1e3
        convs[2].weight /= 1e3
        convs[3].weight /= 1e3
    model = ModelLoader().load_from_state_dict(net.state_dict())
    assert isinstance(model, ImageModelDescriptor)
    model.model.requires_grad_(False)
    return model.eval()


def test_the_check_sees_an_overflow_inside_a_chain_of_arithmetic_nodes():
    # ONNX Runtime's CPU would run the chain in FP32, and a GPU's FP16 gives infinities.
    model = overflowing_inside()
    assert all(w.abs().max() < 65504 for w in model.model.state_dict().values())
    with torch.no_grad():
        body = torch.nn.Sequential(*list(model.model.body)[:3])
        peak = body(torch.rand(1, 3, 16, 16))
    assert peak.abs().max() > 65504
    context = Context(CPU_CONTEXT)
    assert convert(model, context=context)[1] == "fp32"
    assert "NaN or infinite" in context.warnings[0]


class OverItsDeviation(torch.nn.Module):
    """Divides an image by its deviation in each plane, which is zero on a flat image."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.std(dim=(2, 3), keepdim=True)


def test_a_model_that_fails_in_fp32_too_says_nothing_about_fp16():
    # Flat images can make a model divide by a zero, whatever the precision.
    model = tiny_model(PLAIN)
    model.model.initial = torch.nn.Sequential(OverItsDeviation(), model.model.initial)
    with torch.no_grad():
        assert model(torch.ones(1, 3, 16, 16)).isnan().all()
    assert convert(model)[1] == "fp16"


class HalvesItsInput(torch.nn.Module):
    """Writes to its input, as RCAN does."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x *= 0.5
        return x


def test_the_check_gives_a_model_that_writes_to_its_input_a_copy():
    # Not an image that shares its memory (which PyTorch refuses to write to), and the ONNX
    # gets the image as it was.
    model = tiny_model(PLAIN)
    model.model.initial = torch.nn.Sequential(HalvesItsInput(), model.model.initial)
    context = Context(CPU_CONTEXT)
    assert convert(model, context=context)[1] == "fp16"
    assert context.warnings == []


class FrozenChoice(torch.nn.Module):
    """Halves an image of high contrast: a choice a trace freezes at its example, noise."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5 if x.std() > 0.35 else x


def frozen_choice(monkeypatch: pytest.MonkeyPatch) -> ImageModelDescriptor:
    straight_to_fp32(monkeypatch)
    model = tiny_model(PLAIN)
    model.model.initial = torch.nn.Sequential(FrozenChoice(), model.model.initial)
    assert convert_to_onnx.detail_inputs(model)[0].std() > 0.35
    return model


def test_verify_refuses_an_fp32_export_that_only_a_drawn_image_shows_wrong(
    monkeypatch: pytest.MonkeyPatch,
):
    with pytest.raises(AssertionError, match="generated 64 px"):
        convert(frozen_choice(monkeypatch), verify=True)


def test_verify_off_warns_of_the_same_export(monkeypatch: pytest.MonkeyPatch):
    context = Context(CPU_CONTEXT)
    assert convert(frozen_choice(monkeypatch), context=context)[1] == "fp32"
    assert "does not match it in PyTorch" in context.warnings[0]


class RefusesSmallImages(torch.nn.Module):
    """Takes 24 px or more, though its model's requirement does not say so."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] < 24:
            raise ValueError("this model takes 24 px or more")
        return x


def test_a_comparison_that_could_not_run_does_not_say_the_export_is_wrong(
    monkeypatch: pytest.MonkeyPatch,
):
    straight_to_fp32(monkeypatch)
    context = Context(CPU_CONTEXT)
    model = tiny_model(PLAIN)
    model.model.initial = torch.nn.Sequential(RefusesSmallImages(), model.model.initial)
    assert convert(model, context=context)[1] == "fp32"
    ((warning,),) = [context.warnings]
    assert "could not be compared" in warning
    assert "this model takes 24 px or more" in warning
    assert "does not match" not in warning


@pytest.mark.parametrize("looks", [0, 1, 3])
def test_stop_during_the_fp16_check_stops_with_no_fp32_export(
    looks: int, monkeypatch: pytest.MonkeyPatch
):
    exports: list[torch.dtype] = []
    export_onnx = convert_to_onnx.export_onnx

    def counted(model: ModelDescriptor, opset_version: int) -> bytes:
        exports.append(model.dtype)
        return export_onnx(model, opset_version=opset_version)

    monkeypatch.setattr(convert_to_onnx, "export_onnx", counted)
    context = Context(CPU_CONTEXT)
    context.looks_until_stop = looks
    with pytest.raises(Aborted):
        convert(tiny_model(PLAIN), context=context)
    assert exports == [torch.float16]
    assert context.warnings == []


def test_stop_after_a_refused_fp16_export_stops_before_the_fp32_export(
    monkeypatch: pytest.MonkeyPatch,
):
    exports = count_exports(monkeypatch)
    context = Context(CPU_CONTEXT)

    def refuse(*args: object) -> None:
        context.looks_until_stop = 0
        raise AssertionError("not this one")

    monkeypatch.setattr(convert_to_onnx, "verify_models", refuse)
    with pytest.raises(Aborted):
        convert(tiny_model(PLAIN), context=context)
    assert exports == [20]


def test_stop_during_the_fp32_comparison_is_stop_and_no_warning(
    monkeypatch: pytest.MonkeyPatch,
):
    straight_to_fp32(monkeypatch)
    context = Context(CPU_CONTEXT)
    context.looks_until_stop = 1
    with pytest.raises(Aborted):
        convert(tiny_model(PLAIN), context=context)
    assert context.warnings == []


def fp16_chain(k: float, inverse: float) -> bytes:
    node = onnx.helper.make_node
    graph = onnx.helper.make_graph(
        [
            node("Mul", ["x", "k"], ["a"]),
            node("Mul", ["a", "k"], ["b"]),
            node("Mul", ["b", "inverse"], ["c"]),
            node("Mul", ["c", "inverse"], ["y"]),
        ],
        "chain",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT16, [1])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT16, [1])],
        [
            onnx.numpy_helper.from_array(np.array([k], np.float16), "k"),
            onnx.numpy_helper.from_array(np.array([inverse], np.float16), "inverse"),
        ],
    )
    return onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 20)], ir_version=10
    ).SerializeToString()


def run_chain(data: bytes) -> np.ndarray:
    session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    return session.run(None, {"x": np.array([1.0], np.float16)})[0]


def test_holding_keeps_a_chain_of_arithmetic_in_fp16_on_the_cpu():
    # The chain in FP32, which ONNX Runtime's CPU runs it in; in FP16 its middle overflows.
    assert np.isfinite(run_chain(fp16_chain(1000.0, 0.001))).all()
    assert np.isinf(
        run_chain(convert_to_onnx.with_fp16_held(fp16_chain(1000.0, 0.001)))
    ).all()
    # In range, it gives the same.
    data = fp16_chain(100.0, 0.01)
    np.testing.assert_allclose(
        run_chain(convert_to_onnx.with_fp16_held(data)), run_chain(data), atol=0.002
    )


def test_the_detail_images_are_the_same_in_every_run():
    model = tiny_model("HAT")
    first, second = (
        convert_to_onnx.detail_inputs(model),
        convert_to_onnx.detail_inputs(model),
    )
    for one, other in zip(first, second, strict=True):
        assert one.dtype == torch.float32 and torch.equal(one, other)
        assert one.shape == (1, model.input_channels, 64, 64)
        assert one.min() >= 0 and one.max() <= 1


@pytest.mark.parametrize("channels", [1, 3, 4, 5])
def test_the_detail_images_have_the_models_planes(channels: int):
    model = cast(
        ImageModelDescriptor,
        SimpleNamespace(
            input_channels=channels,
            device=torch.device("cpu"),
            size_requirements=SizeRequirements(multiple_of=16),
        ),
    )
    for image in convert_to_onnx.detail_inputs(model):
        assert image.shape == (1, channels, 64, 64)


def test_the_drawn_image_has_flat_areas_at_the_ends_of_the_range_and_hard_edges():
    drawn = convert_to_onnx.detail_inputs(tiny_model("Compact"))[0]
    assert drawn.eq(0).float().mean() > 0.2
    assert drawn.gt(0.85).float().mean() > 0.2
    assert torch.diff(drawn, dim=-1).abs().max() > 0.8
    # The smooth images have no such steps.
    smooth = convert_to_onnx.verify_inputs(tiny_model("Compact"), True)[0]
    assert torch.diff(smooth, dim=-1).abs().max() < 0.5


@pytest.mark.parametrize(
    ("gap", "holds"),
    [
        # Past the smooth images' limits, which do not apply to these
        (0.01, True),
        (0.03, False),
    ],
)
def test_the_check_takes_its_own_limit_on_the_images_with_detail(
    gap: float, holds: bool, monkeypatch: pytest.MonkeyPatch
):
    session = ort.InferenceSession

    class OffOnDetail:
        def __init__(self, data: bytes, providers: list[str]) -> None:
            self.real = session(data, providers=providers)
            self.get_inputs = self.real.get_inputs
            self.get_outputs = self.real.get_outputs

        def run(
            self, names: list[str], feed: dict[str, np.ndarray]
        ) -> list[np.ndarray]:
            (output,) = self.real.run(names, feed)
            (image,) = feed.values()
            if image.shape[-1] != 64:
                return [output]
            signs = np.where(np.indices(output.shape).sum(axis=0) % 2, 1, -1)
            return [(output.astype(np.float64) + gap * signs).astype(output.dtype)]

    monkeypatch.setattr(ort, "InferenceSession", OffOnDetail)
    model = tiny_model("Compact")
    device = torch.device("cpu")
    data = convert_to_onnx.export_onnx(
        convert_to_onnx.export_copy(model, device, use_half=True), opset_version=20
    )
    if holds:
        convert_to_onnx.verify_models(model, data, True, device)
    else:
        with pytest.raises(AssertionError, match="on a generated 64 px image"):
            convert_to_onnx.verify_models(model, data, True, device)


def test_the_smooth_images_are_the_same_in_every_run():
    model = tiny_model("Compact")
    first = convert_to_onnx.verify_inputs(model, True)
    second = convert_to_onnx.verify_inputs(model, True)
    for one, other in zip(first, second, strict=True):
        assert torch.equal(one, other)


def wrong_export(model: ModelDescriptor, opset_version: int) -> bytes:
    """An FP32 export whose convolutions are not its model's: their weights halved."""
    data = convert_to_onnx_impl(model, torch.device("cpu"), opset_version=opset_version)
    proto = onnx.load_from_string(data)
    tensors = [*proto.graph.initializer]
    for node in proto.graph.node:
        if node.op_type == "Constant":
            tensors += [attribute.t for attribute in node.attribute]
    for tensor in tensors:
        if tensor.data_type == onnx.TensorProto.FLOAT and len(tensor.dims) == 4:
            tensor.CopyFrom(
                onnx.numpy_helper.from_array(
                    onnx.numpy_helper.to_array(tensor) * 0.5, tensor.name
                )
            )
    return proto.SerializeToString()


def test_an_fp32_export_that_does_not_match_warns_with_verify_off(
    monkeypatch: pytest.MonkeyPatch,
):
    # A wrong file must not look fine, as SeemoRe's FP32 export would.
    straight_to_fp32(monkeypatch)
    monkeypatch.setattr(convert_to_onnx, "export_onnx", wrong_export)
    context = Context(CPU_CONTEXT)
    onnx_model, fp_mode, _ = convert(tiny_model(PLAIN), context=context)
    assert (fp_mode, onnx_model.info.dtype) == ("fp32", "fp32")
    assert len(context.warnings) == 1
    assert "does not match it in PyTorch" in context.warnings[0]


def test_an_fp32_export_that_does_not_match_is_refused_with_verify_on(
    monkeypatch: pytest.MonkeyPatch,
):
    straight_to_fp32(monkeypatch)
    monkeypatch.setattr(convert_to_onnx, "export_onnx", wrong_export)
    with pytest.raises(AssertionError):
        convert(tiny_model(PLAIN), verify=True)


def test_an_fp32_export_that_matches_does_not_warn(monkeypatch: pytest.MonkeyPatch):
    straight_to_fp32(monkeypatch)
    context = Context(CPU_CONTEXT)
    assert convert(tiny_model(PLAIN), context=context)[1] == "fp32"
    assert context.warnings == []


@pytest.mark.parametrize("opset", [Opset.OPSET_14, Opset.OPSET_17, Opset.OPSET_23])
def test_the_check_runs_at_every_opset(opset: Opset):
    assert convert(tiny_model("Compact"), opset=opset)[1] == "fp16"


def test_a_kept_fp32_fallback_warns_on_each_run_and_node():
    model = overflowing_model()
    first_run = Context(CPU_CONTEXT)
    first = convert(model, context=first_run)
    assert len(first_run.warnings) == 1
    for context in (Context(CPU_CONTEXT), Context(CPU_CONTEXT, "another node")):
        assert convert(model, context=context)[0] is first[0]
        assert context.warnings == first_run.warnings


def test_a_kept_export_that_does_not_match_warns_on_each_run_and_node(
    monkeypatch: pytest.MonkeyPatch,
):
    straight_to_fp32(monkeypatch)
    monkeypatch.setattr(convert_to_onnx, "export_onnx", wrong_export)
    model = tiny_model(PLAIN)
    first_run = Context(CPU_CONTEXT)
    first = convert(model, context=first_run)
    assert "does not match it in PyTorch" in first_run.warnings[0]
    for context in (Context(CPU_CONTEXT), Context(CPU_CONTEXT, "another node")):
        assert convert(model, context=context)[0] is first[0]
        assert context.warnings == first_run.warnings


def test_a_kept_export_with_nothing_to_say_stays_silent():
    model = tiny_model(PLAIN)
    convert(model)
    context = Context(CPU_CONTEXT)
    convert(model, context=context)
    assert context.warnings == []


def test_a_fallback_verified_on_request_is_kept_as_verified(
    monkeypatch: pytest.MonkeyPatch,
):
    # Verify keeps its meaning for an FP32 export, whichever way it got there.
    verify = convert_to_onnx.verify_models
    runs: list[bool] = []

    def counted(
        model: ImageModelDescriptor,
        onnx_model: bytes,
        fp16: bool,
        device: torch.device,
        stop: Callable[[], None],
    ) -> None:
        runs.append(fp16)
        verify(model, onnx_model, fp16, device, stop)

    monkeypatch.setattr(convert_to_onnx, "verify_models", counted)
    model = overflowing_model()
    first = convert(model, verify=True)
    assert first[1] == "fp32"
    assert runs == [True, False]
    assert convert(model, verify=True)[0] is first[0]
    assert convert(model)[0] is first[0]
    assert runs == [True, False]


def test_an_fp16_export_that_raises_converts_in_fp32(monkeypatch: pytest.MonkeyPatch):
    export_onnx = convert_to_onnx.export_onnx

    def no_fp16(model: ModelDescriptor, opset_version: int) -> bytes:
        if model.dtype == torch.float16:
            raise RuntimeError("a layer without an FP16 form")
        return export_onnx(model, opset_version=opset_version)

    monkeypatch.setattr(convert_to_onnx, "export_onnx", no_fp16)
    context = Context(CPU_CONTEXT)
    onnx_model, fp_mode, _ = convert(tiny_model(PLAIN), context=context)
    assert (fp_mode, onnx_model.info.dtype) == ("fp32", "fp32")
    assert len(context.warnings) == 1
    assert "RuntimeError: a layer without an FP16 form" in context.warnings[0]


def test_a_conversion_that_fails_in_both_precisions_raises_and_does_not_warn(
    monkeypatch: pytest.MonkeyPatch,
):
    # GFISRv2 needs opset 17: the FP32 export's own error tells the user, with no word of
    # an FP32 model that does not exist.
    name = tiny_model("GFISRv2").architecture.name
    monkeypatch.setattr(onnx_model_module, "FP16_ARCHS", {name: ()})
    context = Context(CPU_CONTEXT)
    with pytest.raises(ValueError, match="needs Opset 17"):
        convert(tiny_model("GFISRv2"), opset=Opset.OPSET_16, context=context)
    assert context.warnings == []


def test_an_architecture_off_the_list_skips_the_fp16_attempt(
    monkeypatch: pytest.MonkeyPatch,
):
    exports = count_exports(monkeypatch)
    dtypes: list[torch.dtype] = []
    copy_for_export = convert_to_onnx.export_copy

    def seen(
        model: ImageModelDescriptor, device: torch.device, use_half: bool = False
    ) -> ImageModelDescriptor:
        dtypes.append(torch.float16 if use_half else torch.float32)
        return copy_for_export(model, device, use_half=use_half)

    monkeypatch.setattr(convert_to_onnx, "export_copy", seen)
    name = straight_to_fp32(monkeypatch)
    context = Context(CPU_CONTEXT)
    onnx_model, fp_mode, _ = convert(tiny_model(PLAIN), context=context)
    assert (fp_mode, onnx_model.info.dtype) == ("fp32", "fp32")
    assert (exports, dtypes) == ([20], [torch.float32])
    assert context.warnings == []
    assert recorded(onnx_model) == {
        ARCH_METADATA_KEY: name,
        PRECISION_METADATA_KEY: "fp32",
    }


def test_a_model_loaded_in_fp16_falls_back_to_its_file_weights(tmp_path: Path):
    path = model_file(tmp_path, overflowing_fp16)
    half = load_model(path, FP16)
    full = load_model(path, FP32)
    assert isinstance(half, ImageModelDescriptor)
    assert isinstance(full, ImageModelDescriptor)
    assert half.dtype == torch.float16
    before = snapshot(half)
    exported = convert(half)
    assert exported[1] == "fp32"
    assert exported[0].bytes == convert(full)[0].bytes
    assert_unchanged(half, before)


@pytest.mark.parametrize(("half", "largest_step"), [(True, 1 / 3), (False, 1.0)])
def test_verify_runs_an_fp16_export_on_a_smooth_image(half: bool, largest_step: float):
    # Noise at full amplitude makes trained SPANs amplify FP16's rounding far past any
    # tolerance; FP32 keeps its noise.
    inputs = convert_to_onnx.verify_inputs(tiny_model("Compact"), half)
    sizes = [16, 40, 16, 16, 16] if half else [16, 40]
    assert [image.shape[-1] for image in inputs] == sizes
    for image in inputs:
        steps = torch.diff(image, dim=-1).abs().max()
        assert steps <= largest_step if half else steps > largest_step / 2
    if half:
        # Where trained HAT, DAT and Swin2SR models write past FP16's range
        assert inputs[2].eq(0).all() and inputs[3].eq(1).all()
        assert inputs[4].min() == 0 and inputs[4].max() == 1


def test_omnisr_verifies_against_fp32_where_fp16_pytorch_gives_nan():
    model = tiny_model("OmniSR")
    # A channel of exactly zero: FP16 PyTorch divides its length of zero by an epsilon
    # that FP16 rounds to zero, and the export does not.
    for name, weight in model.model.named_parameters():
        if name.endswith("qkv_dwconv.weight"):
            weight[0] = 0
    half = copy.deepcopy(model).half()
    with torch.no_grad():
        assert half(torch.rand(1, 3, 16, 16).half()).isnan().any()
    assert convert(model, verify=True)[1] == "fp16"


def change_onnx_output(
    monkeypatch: pytest.MonkeyPatch, change: Callable[[np.ndarray], np.ndarray]
) -> None:
    """`change` applied to what Verify's ONNX Runtime session returns, in float64."""
    session = ort.InferenceSession

    class Changed:
        def __init__(self, data: bytes, providers: list[str]) -> None:
            self.real = session(data, providers=providers)
            self.get_inputs = self.real.get_inputs
            self.get_outputs = self.real.get_outputs

        def run(
            self, names: list[str], feed: dict[str, np.ndarray]
        ) -> list[np.ndarray]:
            (output,) = self.real.run(names, feed)
            return [change(output.astype(np.float64)).astype(output.dtype)]

    monkeypatch.setattr(ort, "InferenceSession", Changed)


def one_sample_off(gap: float) -> Callable[[np.ndarray], np.ndarray]:
    def change(output: np.ndarray) -> np.ndarray:
        output[0, 0, 0, 0] += gap
        return output

    return change


def every_sample_off(gap: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda output: output + gap


@pytest.mark.parametrize(
    ("change", "holds"),
    [
        # A correct export of a trained DITN: it amplifies rounding in a few samples, as
        # far as 0.038, with a root mean square of 0.0003
        (one_sample_off(0.038), True),
        (every_sample_off(0.0003), True),
        (one_sample_off(0.2), False),
        (every_sample_off(0.002), False),
        (one_sample_off(float("nan")), False),
    ],
)
def test_verify_of_fp32_passes_rounding_and_nothing_more(
    change: Callable[[np.ndarray], np.ndarray],
    holds: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    change_onnx_output(monkeypatch, change)
    model = tiny_model("Compact")
    data = convert_to_onnx.export_onnx(
        convert_to_onnx.export_copy(model, torch.device("cpu")), opset_version=20
    )
    device = torch.device("cpu")
    if holds:
        convert_to_onnx.verify_models(model, data, False, device)
    else:
        with pytest.raises(AssertionError):
            convert_to_onnx.verify_models(model, data, False, device)


def test_verify_of_fp32_catches_a_size_frozen_in_the_export(
    monkeypatch: pytest.MonkeyPatch,
):
    torch.manual_seed(0)
    net = RealPLKSR(
        dim=16, n_blocks=2, upscaling_factor=2, kernel_size=7, dysample=True
    )
    model = ModelLoader().load_from_state_dict(net.state_dict())
    assert isinstance(model, ImageModelDescriptor)
    model.model.requires_grad_(False)
    model.eval()
    device = torch.device("cpu")
    copied = convert_to_onnx.export_copy(model, device)
    convert_to_onnx.verify_models(
        model, convert_to_onnx.export_onnx(copied, opset_version=20), False, device
    )
    # Without the rewrite that keeps DySample's grid following the input
    monkeypatch.setattr(dysample_export, "for_export", lambda model: model)
    frozen_like_spandrel_before_c14(monkeypatch)
    copied = convert_to_onnx.export_copy(model, device)
    with pytest.raises(AssertionError):
        convert_to_onnx.verify_models(
            model, convert_to_onnx.export_onnx(copied, opset_version=20), False, device
        )


@pytest.mark.parametrize(
    "precision,gap,holds",
    [
        ("fp16", 0.0, True),
        # One 8-bit step over the image: FP16's rounding, within the limits
        ("fp16", 0.004, True),
        # Past the largest difference allowed, though 0.03 is a small root mean square
        ("fp16", 0.03, False),
        ("fp16", 0.2, False),
        ("fp32", 0.0, True),
        ("fp32", 0.015, False),
    ],
)
def test_verify_takes_the_rounding_of_the_export_s_precision(
    precision: str, gap: float, holds: bool, monkeypatch: pytest.MonkeyPatch
):
    def everywhere(output: np.ndarray) -> np.ndarray:
        signs = np.where(np.indices(output.shape).sum(axis=0) % 2, 1, -1)
        return output + gap * signs

    change_onnx_output(monkeypatch, everywhere)
    half = precision == "fp16"
    model = tiny_model("ESRGAN")
    data = convert_to_onnx.export_onnx(
        convert_to_onnx.export_copy(model, torch.device("cpu"), use_half=half),
        opset_version=20,
    )
    device = torch.device("cpu")
    if holds:
        convert_to_onnx.verify_models(model, data, half, device)
    else:
        with pytest.raises(AssertionError):
            convert_to_onnx.verify_models(model, data, half, device)


@pytest.mark.parametrize(
    ("arch", "scale", "dtype", "input_type"),
    [
        ("ESRGAN", 4, np.float16, "tensor(float16)"),
        ("Compact", 2, np.float32, "tensor(float)"),
        ("SPAN", 2, np.float32, "tensor(float)"),
        (PLAIN, 2, np.float16, "tensor(float16)"),
    ],
)
def test_an_fp16_export_runs_in_fp16_on_the_cpu(
    arch: str,
    scale: int,
    dtype: type[np.floating],
    input_type: str,
    monkeypatch: pytest.MonkeyPatch,
):
    skip_the_check(monkeypatch)
    onnx_model = convert(tiny_model(arch))[0]
    session = ort.InferenceSession(onnx_model.bytes, providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].type == input_type
    assert session.get_outputs()[0].type == "tensor(float16)"
    image = np.random.default_rng(0).random((1, 3, 16, 16)).astype(dtype)
    output = session.run(None, {session.get_inputs()[0].name: image})[0]
    assert output.shape == (1, 3, 16 * scale, 16 * scale) and np.isfinite(output).all()


@pytest.mark.parametrize(
    ("arch", "first"),
    [
        ("Compact", "body.0"),
        ("PLKSR", "feats.0"),
        ("OmniSR", "input"),
    ],
)
def test_an_fp16_export_sees_what_fp16_would_round_away(arch: str, first: str):
    # A first convolution that reads differences with a high gain, as trained models do on
    # smooth images, and two images that are one image once rounded to FP16.
    model = tiny_model(arch)
    conv = model.model.get_submodule(first)
    with torch.no_grad():
        conv.weight -= conv.weight.mean((1, 2, 3), keepdim=True)
        conv.weight *= 1000
    half = convert_to_onnx_impl(model, torch.device("cpu"), True, opset_version=20)
    session = ort.InferenceSession(half, providers=["CPUExecutionProvider"])
    a = np.full((1, 3, 32, 32), 0.75, np.float32)
    b = a.copy()
    b[..., ::2, ::2] += 2.0**-13
    assert np.array_equal(a.astype(np.float16), b.astype(np.float16))
    (from_a,) = session.run(None, {"input": a})
    (from_b,) = session.run(None, {"input": b})
    assert not np.array_equal(from_a, from_b)


@pytest.mark.parametrize("arch", FP32_INPUT_MODELS)
@pytest.mark.parametrize("image", [*SMOOTH, "bright ramp"])
def test_an_fp16_export_holds_on_a_smooth_image(arch: str, image: str):
    model = tiny_model(arch)
    half = convert_to_onnx_impl(model, torch.device("cpu"), True, opset_version=20)
    full = convert_to_onnx_impl(model, torch.device("cpu"), False, opset_version=20)
    x = BRIGHT_RAMP if image == "bright ramp" else SMOOTH[image]
    session = ort.InferenceSession(half, providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].type == "tensor(float)"
    assert session.get_outputs()[0].type == "tensor(float16)"
    (actual,) = session.run(None, {"input": x})
    (expected,) = ort.InferenceSession(full, providers=["CPUExecutionProvider"]).run(
        None, {"input": x}
    )
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual.astype(np.float32), expected, atol=0.02)


@pytest.mark.parametrize("arch", FP32_INPUT_MODELS)
def test_upscale_image_runs_a_saved_fp16_export(arch: str):
    model = tiny_model(arch)
    loaded = load_onnx_model(convert(model)[0].bytes)
    assert loaded.info.dtype == "fp16"
    session = get_onnx_session(loaded, 0, "CPUExecutionProvider", False)
    image = np.random.default_rng(0).random((32, 32, 3), dtype=np.float32)
    output = onnx_auto_split(image, session, False, NoTiling())
    assert output.dtype == np.float32
    reference = torch_output(model, image[:, :, ::-1].transpose(2, 0, 1).copy())
    np.testing.assert_allclose(
        output, reference[::-1].transpose(1, 2, 0), rtol=0, atol=0.02
    )


def old_and_new_compact(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ImageModelDescriptor, OnnxGeneric, OnnxGeneric]:
    # Interpolate Models checks that a white image stays bright: a small residual.
    model = tiny_model("Compact")
    last = model.model.body[-1]
    with torch.no_grad():
        last.weight *= 0.001
        last.bias.zero_()

    def saved() -> OnnxGeneric:
        data = convert_to_onnx_impl(model, torch.device("cpu"), True, opset_version=20)
        return load_onnx_model(
            with_recorded_arch(data, model.architecture.name, "fp16")
        )

    new = saved()
    with monkeypatch.context() as patched:
        patched.setattr(input_export, "fp32_first_convolution", lambda _: None)
        old = saved()
    assert ort_input_type(new) == "tensor(float)"
    assert ort_input_type(old) == "tensor(float16)"
    return model, old, new


def ort_input_type(onnx_model: OnnxGeneric) -> str:
    session = ort.InferenceSession(onnx_model.bytes, providers=["CPUExecutionProvider"])
    return session.get_inputs()[0].type


def interpolation_context(storage: Path) -> NodeContext:
    return cast(
        NodeContext,
        SimpleNamespace(
            settings=SettingsParser(
                {"execution_provider": "CPUExecutionProvider", "gpu_index": "0"}
            ),
            add_cleanup=lambda *_, **__: None,
            check_aborted=lambda: None,
            set_progress=lambda _: None,
            storage_dir=storage,
        ),
    )


def test_interpolate_models_mixes_two_fp16_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model, _, new = old_and_new_compact(monkeypatch)
    mixed = interpolate_models_node(interpolation_context(tmp_path), new, new, 50)[0]
    session = ort.InferenceSession(mixed.bytes, providers=["CPUExecutionProvider"])
    image = np.full((1, 3, 16, 16), 0.5, np.float32)
    (output,) = session.run(None, {"input": image})
    np.testing.assert_allclose(
        output.astype(np.float32)[0], torch_output(model, image[0]), atol=0.02
    )


@pytest.mark.parametrize("old_first", [True, False])
def test_interpolate_models_says_why_it_refuses_an_older_fp16_export(
    old_first: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _, old, new = old_and_new_compact(monkeypatch)
    a, b = (old, new) if old_first else (new, old)
    with pytest.raises(ValueError, match="Convert both again"):
        interpolate_models_node(interpolation_context(tmp_path), a, b, 50)


@pytest.mark.parametrize("arch", FP16_MODELS)
def test_a_saved_onnx_keeps_its_record(arch: str, monkeypatch: pytest.MonkeyPatch):
    skip_the_check(monkeypatch)
    model = tiny_model(arch)
    converted = convert(model)[0]
    # The name a PyTorch model's type carries, which the node's conditions compare
    name = model.architecture.name
    assert recorded(converted) == {
        ARCH_METADATA_KEY: name,
        PRECISION_METADATA_KEY: "fp16",
    }
    loaded = load_onnx_model(converted.bytes)
    assert (loaded.info.arch, loaded.info.dtype) == (name, "fp16")
    fields = OnnxModelOutput().get_broadcast_type(loaded)["fields"]
    assert fields["arch"] == {"type": "string-literal", "value": name}


def test_an_export_keeps_its_size_requirement_beside_its_record():
    model = tiny_model("HAT")
    converted = convert(model)[0]
    assert model_facts(converted).size_req is not None
    assert recorded(converted)[ARCH_METADATA_KEY] == model.architecture.name


def test_the_node_hides_data_type_alone_and_keeps_its_input_ids():
    node = next(
        n
        for n in convert_to_onnx.utility_group.nodes
        if n.schema_id == "chainner:pytorch:convert_to_onnx"
    )
    labels = {i.id: i.label for i in node.inputs}
    assert (labels[1], labels[2], labels[3]) == ("Data Type", "Opset", "Verify")
    # A saved chain's Calibration Image and Mixed data types migrate away (migrations.ts).
    assert 7 not in labels
    assert [item.value for item in OnnxDataType] == [0, 1]
    outer = next(
        g
        for g in node.group_layout
        if not isinstance(g, int) and 2 in g.items and 3 in g.items
    )
    condition = str(outer.info.options["condition"])
    assert '"DUAL"' in condition and "ESRGAN" not in condition
    # Data Type sits in a group of its own inside it, hidden for any model, known or not.
    inner = [g for g in outer.items if not isinstance(g, int)]
    assert [g.items for g in inner] == [[1]]
    assert inner[0].info.options["condition"] == {"kind": "or", "items": []}
