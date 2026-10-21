"""Stop reaches the tiled ONNX/NCNN nodes without another inference or error wrap."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from test_tiling_control import OLD

from api import Aborted, NodeContext, Progress
from nodes.impl.ncnn import auto_split as ncnn_split
from nodes.impl.onnx import auto_split as onnx_split
from nodes.impl.onnx.model import SizeReq
from nodes.impl.upscale.auto_split_tiles import TileSize
from packages.chaiNNer_ncnn.ncnn.processing import upscale_image as ncnn_node
from packages.chaiNNer_onnx.onnx.processing import upscale_image as onnx_node


class StopProgress(Progress):
    def __init__(self):
        self.stopped = False
        self.cleanups = []

    @property
    def aborted(self) -> bool:
        return self.stopped

    @property
    def paused(self) -> bool:
        return False

    def set_progress(self, progress: float) -> None:
        assert 0 <= progress <= 1
        self.check_aborted()

    def add_cleanup(self, fn, after="chain"):
        self.cleanups.append((fn, after))


@pytest.mark.parametrize("tiling", ["pinned", "native"])
@pytest.mark.parametrize("backend", ["onnx-max", "onnx-exact", "ncnn"])
def test_stop_after_first_tile_propagates(monkeypatch, tiling, backend):
    progress = StopProgress()
    calls = []
    image = np.ones((65, 81, 3), np.float32)
    model = SimpleNamespace(
        in_nc=3,
        out_nc=3,
        scale=1,
        size_multiple=1,
        model=SimpleNamespace(layers=[SimpleNamespace(outputs=["data"])]),
    )

    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name="data", type="tensor(float)")]

        def get_outputs(self):
            return [SimpleNamespace(name="out")]

        def run(self, _names, inputs):
            calls.append("run")
            progress.stopped = True
            return [inputs["data"]]

        def create_extractor(self):
            return Extractor()

    class Extractor:
        def __init__(self):
            self.value = None

        def input(self, _name, value):
            self.value = value
            return 0

        def extract(self, _name):
            calls.append("extract")
            progress.stopped = True
            return 0, self.value

    if backend.startswith("onnx"):
        node = onnx_node
        split = onnx_split
        monkeypatch.setattr(
            node,
            "get_settings",
            lambda _context: SimpleNamespace(
                gpu_index=0,
                execution_provider="CPUExecutionProvider",
                tensorrt_fp16_mode=False,
                tensorrt_cache_path="",
            ),
        )
        monkeypatch.setattr(node, "get_onnx_session", lambda *args: Session())
        size = 32 if backend == "onnx-exact" else None
        monkeypatch.setattr(node, "get_input_shape", lambda _: ("BCHW", 3, size, size))
        monkeypatch.setattr(node, "get_output_shape", lambda _: ("BCHW", 3, size, size))
        monkeypatch.setattr(node, "get_size_req", lambda *_args: SizeReq())
    else:
        node = ncnn_node
        split = ncnn_split
        assert not split.use_gpu and not node.use_gpu
        monkeypatch.setattr(node, "get_settings", lambda _context: SimpleNamespace())
        monkeypatch.setattr(node, "get_ncnn_net", lambda *args, **kwargs: Session())
    if tiling == "pinned":
        monkeypatch.setattr(split, "auto_split", OLD[0].auto_split)

    with pytest.raises(Aborted) as raised:
        node.upscale_image_node(
            cast(NodeContext, progress),
            image,
            cast(Any, model),
            TileSize(32),
            32,
            False,
        )

    assert calls == (["extract"] if backend == "ncnn" else ["run"])
    assert raised.value.__cause__ is None
    assert len(progress.cleanups) == 1
