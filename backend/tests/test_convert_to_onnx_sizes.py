"""ONNX Upscale Image of a Convert To ONNX export pads and crops as PyTorch does: the
export records its model's size requirement, and RealCUGAN's graph keeps its own
padding of odd sizes."""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from spandrel import ImageModelDescriptor, ModelLoader
from spandrel.architectures.Compact import Compact
from spandrel.architectures.RealCUGAN import UpCunet2x, UpCunet3x, UpCunet4x
from spandrel.architectures.SST import SST

# Load Model and the node's padding run in the native backend, which GitHub's checks
# do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from nodes.impl.onnx.autocast import autocast
from nodes.impl.onnx.load import load_onnx_model, model_facts
from nodes.impl.onnx.model import SIZE_REQ_METADATA_KEY, OnnxGeneric, SizeReq
from nodes.impl.onnx.session import create_inference_session
from nodes.impl.pytorch.convert_to_onnx_impl import convert_to_onnx_impl
from nodes.impl.upscale.auto_split_tiles import NO_TILING
from packages.chaiNNer_onnx.onnx.processing.upscale_image import upscale_with_session

ARCHS: dict[str, Callable[[], torch.nn.Module]] = {
    "Compact": lambda: Compact(num_feat=16, num_conv=2, upscale=2),
    # A minimum of 24 (its window) alone, which the graph does not pad to
    "SST": lambda: SST(
        dim=12,
        window_sizes=[24],
        num_heads=[2],
        n_blocks=1,
        attn_type="SDPA",
        upscaling_factor=2,
    ),
    "RealCUGAN 2x": lambda: UpCunet2x(in_channels=3, out_channels=3),
    "RealCUGAN 3x": lambda: UpCunet3x(in_channels=3, out_channels=3),
    "RealCUGAN 4x": lambda: UpCunet4x(in_channels=3, out_channels=3),
}


def tiny_model(arch: str) -> ImageModelDescriptor:
    """A randomly initialised model, prepared as PyTorch's Load Model node does."""
    torch.manual_seed(0)
    model = ModelLoader().load_from_state_dict(ARCHS[arch]().state_dict())
    assert isinstance(model, ImageModelDescriptor)
    model.model.requires_grad_(False)
    return model.eval()


def image(h: int, w: int) -> np.ndarray:
    return np.random.default_rng(0).random((h, w, 3), dtype=np.float32)


def pytorch(model: ImageModelDescriptor, img: np.ndarray) -> np.ndarray:
    """PyTorch's upscale of a BGR image, which pads it to the model's requirement."""
    rgb = torch.from_numpy(img[:, :, ::-1].transpose(2, 0, 1).copy())[None]
    with torch.no_grad():
        output = model(rgb).numpy()[0]
    return output.transpose(1, 2, 0)[:, :, ::-1]


def test_an_export_records_its_models_size_requirement():
    for arch, recorded in [
        ("SST", SizeReq(24)),
        ("RealCUGAN 2x", SizeReq(32)),
        ("Compact", None),
    ]:
        onnx_bytes = convert_to_onnx_impl(tiny_model(arch), torch.device("cpu"))
        model = load_onnx_model(onnx_bytes)
        assert model_facts(model).size_req == recorded, arch


@pytest.mark.parametrize("arch", ["SST", "RealCUGAN 2x", "RealCUGAN 3x"])
def test_upscale_pads_an_export_below_its_minimum_as_pytorch_does(arch: str):
    # Before, the size probe padded to 16, where these run but give other pixels
    model = tiny_model(arch)
    exported = load_onnx_model(convert_to_onnx_impl(model, torch.device("cpu")))
    assert isinstance(exported, OnnxGeneric)
    session = create_inference_session(exported, 0, "CPUExecutionProvider")
    for h, w in [(10, 10), (20, 23)]:
        img = image(h, w)
        np.testing.assert_allclose(
            np.clip(
                upscale_with_session(img, exported, session, NO_TILING, False), 0, 1
            ),
            pytorch(model, img),
            rtol=0,
            atol=1e-5,
        )


@pytest.mark.parametrize("arch", ["RealCUGAN 2x", "RealCUGAN 3x", "RealCUGAN 4x"])
def test_a_realcugan_export_runs_odd_sizes_as_pytorch_does(arch: str):
    # RealCUGAN pads odd sizes (3x: sizes not divisible by 4) inside the model; its
    # crop after is a Python branch, which a trace at an even size left out.
    model = tiny_model(arch)
    session = ort.InferenceSession(
        convert_to_onnx_impl(model, torch.device("cpu")),
        providers=["CPUExecutionProvider"],
    )
    for h, w in [(33, 35), (37, 51), (41, 40), (32, 32)]:
        img = image(h, w)
        rgb = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))[None]
        (output,) = session.run(None, {"input": rgb})
        actual = np.asarray(output)[0].transpose(1, 2, 0)[:, :, ::-1]
        np.testing.assert_allclose(
            np.clip(actual, 0, 1), pytorch(model, img), rtol=0, atol=1e-4
        )


def test_mixed_precision_keeps_the_recorded_size_requirement(
    monkeypatch: pytest.MonkeyPatch,
):
    # ModelOpt's converter writes a model of its own, without the source's metadata.
    def convert_to_mixed_precision(onnx_path: str, **_: Any) -> onnx.ModelProto:
        converted = onnx.load(onnx_path)
        del converted.metadata_props[:]
        return converted

    modules = ["modelopt", "modelopt.onnx", "modelopt.onnx.autocast"]
    for name in modules:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    vars(sys.modules[modules[-1]])["convert_to_mixed_precision"] = (
        convert_to_mixed_precision
    )
    onnx_bytes = convert_to_onnx_impl(tiny_model("SST"), torch.device("cpu"))
    calibration = np.zeros((1, 3, 32, 32), np.float32)
    mixed = onnx.load_from_string(autocast(onnx_bytes, "fp16", calibration))
    recorded = [e for e in mixed.metadata_props if e.key == SIZE_REQ_METADATA_KEY]
    assert len(recorded) == 1
    assert model_facts(load_onnx_model(mixed.SerializeToString())).size_req == SizeReq(
        24
    )
