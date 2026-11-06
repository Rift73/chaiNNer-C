"""A TensorRT engine gives the same bits for the same tile, run after run: what the
transparency path's white pass reuse (white_pass.py) rests on. Real GPU only."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest
import torch

pytest.importorskip("tensorrt")
pytest.importorskip("cuda.bindings.runtime")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from spandrel import ImageModelDescriptor, ModelLoader
from spandrel.architectures.ESRGAN.__arch.RRDB import RRDBNet

from nodes.impl.pytorch.convert_to_onnx_impl import (
    export_copy,
    export_onnx,
)
from nodes.impl.tensorrt.engine_builder import (
    BuildConfig,
    build_engine_from_onnx,
)
from nodes.impl.tensorrt.inference import TensorRTSession

SIZE = 64


def small_esrgan() -> ImageModelDescriptor:
    torch.manual_seed(0)
    state = RRDBNet(
        in_nc=3, out_nc=3, num_filters=16, num_blocks=1, scale=2
    ).state_dict()
    model = ModelLoader().load_from_state_dict(state)
    assert isinstance(model, ImageModelDescriptor)
    model.model.requires_grad_(False)
    model.eval()
    return model


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_the_same_tile_gives_the_same_bits(precision: Literal["fp16", "fp32"]):
    model = small_esrgan()
    onnx = export_onnx(
        export_copy(model, torch.device("cuda"), use_half=precision == "fp16"),
        opset_version=17,
    )
    config = BuildConfig(
        precision=precision,
        workspace_size_gb=0,
        min_shape=(SIZE, SIZE),
        opt_shape=(SIZE, SIZE),
        max_shape=(SIZE, SIZE),
        use_dynamic_shapes=False,
        allow_tf32=False,
        optimization_level=0,
    )
    engine = build_engine_from_onnx(onnx, config, gpu_index=0)
    session = TensorRTSession(engine, 0)
    try:
        tile = np.random.default_rng(1).random((SIZE, SIZE, 3), dtype=np.float32)
        outputs = []
        for _ in range(3):
            session.submit(tile)
            outputs.append(session.collect(SIZE * 2, SIZE * 2))
    finally:
        session.close()
    first = outputs[0].view(np.uint32)
    for output in outputs[1:]:
        assert output.shape == outputs[0].shape
        assert np.array_equal(output.view(np.uint32), first)
