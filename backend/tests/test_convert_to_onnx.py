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
from spandrel.architectures.ESRGAN import ESRGAN

from nodes.impl.ncnn.model import NcnnModelWrapper
from nodes.impl.onnx.onnx_to_ncnn import Onnx2NcnnConverter
from nodes.impl.onnx.utils import safely_optimize_onnx_model
from nodes.impl.pytorch.convert_to_onnx_impl import convert_to_onnx_impl

ARCHS: dict[str, Callable[[], torch.nn.Module]] = {
    "Compact": lambda: Compact(num_feat=16, num_conv=2, upscale=2),
    "ESRGAN": lambda: ESRGAN(num_filters=8, num_blocks=1, scale=4),
}
# The export traces a 32x32 image; 37x51 checks that height and width stay dynamic.
SIZES = [(32, 32), (37, 51)]


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


def images():
    rng = np.random.default_rng(0)
    return [rng.random((3, h, w), dtype=np.float32) for h, w in SIZES]


@pytest.mark.parametrize("arch", ARCHS)
def test_convert_to_onnx_matches_pytorch(arch: str):
    model = tiny_model(arch)
    session = ort.InferenceSession(
        convert_to_onnx_impl(model, torch.device("cpu")),
        providers=["CPUExecutionProvider"],
    )
    for image in images():
        (actual,) = session.run(None, {"input": image[None]})
        np.testing.assert_allclose(
            np.asarray(actual)[0], torch_output(model, image), rtol=0, atol=1e-5
        )


@pytest.mark.parametrize("arch", ARCHS)
def test_convert_to_ncnn_matches_pytorch(arch: str, tmp_path: Path):
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
    for image in images():
        ex = net.create_extractor()
        assert ex.input(converted.layers[0].outputs[0], ncnn.Mat(image)) == 0
        ret, actual = ex.extract(converted.layers[-1].outputs[0])
        assert ret == 0
        np.testing.assert_allclose(
            np.array(actual), torch_output(model, image), rtol=0, atol=1e-4
        )
