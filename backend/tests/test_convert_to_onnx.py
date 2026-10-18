from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from ncnn import ncnn
from spandrel import ImageModelDescriptor, ModelLoader
from spandrel.architectures.Compact import Compact
from spandrel.architectures.DAT import DAT
from spandrel.architectures.DAT.__arch.DAT import Adaptive_Spatial_Attention
from spandrel.architectures.ESRGAN import ESRGAN
from spandrel.architectures.HAT import HAT
from spandrel.architectures.HAT.__arch.HAT import OCAB, WindowAttention
from spandrel.architectures.MixDehazeNet import MixDehazeNet

from nodes.impl.ncnn.model import NcnnModelWrapper
from nodes.impl.onnx.onnx_to_ncnn import Onnx2NcnnConverter
from nodes.impl.onnx.utils import safely_optimize_onnx_model
from nodes.impl.pytorch import attention_export
from nodes.impl.pytorch.convert_to_onnx_impl import convert_to_onnx_impl

ARCHS: dict[str, Callable[[], torch.nn.Module]] = {
    "Compact": lambda: Compact(num_feat=16, num_conv=2, upscale=2),
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


@pytest.mark.parametrize(
    ("arch", "sizes"),
    [
        ("Compact", SIZES),
        ("ESRGAN", SIZES),
        ("MixDehazeNet", MIN_40_SIZES),
        ("HAT", SIZES),
        ("DAT", SIZES),
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
