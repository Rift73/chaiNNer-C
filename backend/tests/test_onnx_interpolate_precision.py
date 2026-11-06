"""ONNX Interpolate Models refuses two models of different precisions with a clear error:
the mix puts the interpolated weights into B's graph, which ONNX Runtime refused with
a bare type error ("Type parameter (T) of Optype (Conv) bound to different types").
At 0 % or 100 % the node still returns A or B as it is. A mix keeps the integers of the
graph, and is checked with the planes the model takes."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from spandrel import ModelLoader
from spandrel.architectures.DAT import DAT
from spandrel.architectures.HAT import HAT

# The model loader is native, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from api import NodeContext, Progress, SettingsParser
from nodes.impl.onnx.load import load_onnx_model
from nodes.impl.onnx.model import OnnxModel
from nodes.impl.pytorch.convert_to_onnx_impl import convert_to_onnx_impl
from packages.chaiNNer_onnx.onnx.utility import interpolate_models
from packages.chaiNNer_onnx.onnx.utility.interpolate_models import (
    interpolate_models_node,
)


def model(dtype: type, planes: int = 3) -> OnnxModel:
    """A 1x1 Conv that copies its planes, in this precision."""
    kind = TensorProto.FLOAT16 if dtype is np.float16 else TensorProto.FLOAT
    conv = helper.make_node("Conv", ["x", "w"], ["y"])
    conv.attribute.append(helper.make_attribute("kernel_shape", [1, 1]))
    graph = helper.make_graph(
        [conv],
        "precision",
        [helper.make_tensor_value_info("x", kind, [1, planes, "h", "w"])],
        [helper.make_tensor_value_info("y", kind, [1, planes, "h", "w"])],
        [
            numpy_helper.from_array(
                np.eye(planes, dtype=dtype).reshape(planes, planes, 1, 1), "w"
            )
        ],
    )
    return load_onnx_model(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
        ).SerializeToString()
    )


# The node context is used only by the result's test upscale, after the refusal.
CONTEXT = cast(NodeContext, None)


def test_different_precisions_are_refused():
    fp32, fp16 = model(np.float32), model(np.float16)
    assert (fp32.info.dtype, fp16.info.dtype) == ("fp32", "fp16")
    with pytest.raises(ValueError, match=r"different precisions \(fp32 and fp16\)"):
        interpolate_models_node(CONTEXT, fp32, fp16, 50)


def test_the_ends_still_return_a_model_as_it_is():
    fp32, fp16 = model(np.float32), model(np.float16)
    assert interpolate_models_node(CONTEXT, fp32, fp16, 0)[0] is fp32
    assert interpolate_models_node(CONTEXT, fp32, fp16, 100)[0] is fp16


NETS = {
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
    "DAT": lambda: DAT(
        embed_dim=24, depth=[3], num_heads=[4], upsampler="pixelshuffle", upscale=2
    ),
}


def exported(arch: str, seed: int, half: bool) -> OnnxModel:
    torch.manual_seed(seed)
    net = ModelLoader().load_from_state_dict(NETS[arch]().state_dict()).eval()
    return load_onnx_model(
        convert_to_onnx_impl(net, torch.device("cpu"), half, opset_version=20)
    )


@pytest.mark.parametrize("arch", list(NETS))
@pytest.mark.parametrize("half", [False, True], ids=["fp32", "fp16"])
def test_a_mix_keeps_the_slice_bounds_of_the_graph(
    arch: str, half: bool, monkeypatch: pytest.MonkeyPatch
):
    # Random weights fail the node's check that a white image stays white.
    monkeypatch.setattr(interpolate_models, "check_will_upscale", lambda *_: True)
    a, b = exported(arch, 0, half), exported(arch, 1, half)
    mixed = interpolate_models_node(CONTEXT, a, b, 50)[0]
    session = ort.InferenceSession(mixed.bytes, providers=["CPUExecutionProvider"])
    image = np.random.default_rng(0).random((1, 3, 37, 51), dtype=np.float32)
    (output,) = session.run(
        None, {"input": image.astype(np.float16 if half else np.float32)}
    )
    assert output.shape == (1, 3, 74, 102)
    assert np.isfinite(output).all()


class CpuContext:
    """The NodeContext members ONNX Upscale Image uses, on the CPU provider."""

    def __init__(self):
        self.settings = SettingsParser({"execution_provider": "CPUExecutionProvider"})
        # The size probe's cache folder (size_probe.py); gone with this context.
        self._storage = tempfile.TemporaryDirectory()
        self.storage_dir = Path(self._storage.name)
        progress = Progress.noop_progress()
        self.check_aborted = progress.check_aborted
        self.set_progress = progress.set_progress

    def add_cleanup(self, fn: Callable[[], None], after: str = "chain"):
        pass


@pytest.mark.parametrize("planes", [1, 2, 3, 4, 5, 15])
@pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["fp32", "fp16"])
def test_the_mix_is_checked_with_the_planes_the_model_takes(planes: int, dtype: type):
    a, b = model(dtype, planes), model(dtype, planes)
    context = cast(NodeContext, CpuContext())
    mixed = interpolate_models_node(context, a, b, 50)[0]
    assert mixed.info.input_channels == planes
