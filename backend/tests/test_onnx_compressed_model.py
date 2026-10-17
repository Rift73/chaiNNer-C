"""ONNX Save Model's Compress: a model compressed with LZMA or Zstandard, named
"_compressed", loads back through Load Model as the same model; a plain one is written
and read as before."""

from __future__ import annotations

from pathlib import Path

import pytest
from onnx import TensorProto, helper

from nodes.impl.compressed_file import XZ_HEADER, ZSTD_HEADER, Compression
from nodes.impl.onnx.load import load_onnx_model
from nodes.impl.onnx.model import OnnxModel
from packages.chaiNNer_onnx.onnx.io.load_model import load_model_node
from packages.chaiNNer_onnx.onnx.io.save_model import save_model_node

HEADERS = {Compression.LZMA: XZ_HEADER, Compression.ZSTD: ZSTD_HEADER}


def tiny_upscaler() -> OnnxModel:
    """A 2x model: a 3x3 convolution to 12 channels, then DepthToSpace."""
    weights = helper.make_tensor(
        "w", TensorProto.FLOAT, [12, 3, 3, 3], [0.01 * (i % 7) for i in range(324)]
    )
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]),
            helper.make_node("DepthToSpace", ["c"], ["y"], blocksize=2),
        ],
        "upscaler",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, "H", "W"])],
        [weights],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return load_onnx_model(model.SerializeToString())


@pytest.mark.parametrize("compression", list(Compression))
def test_a_compressed_model_loads_back_as_the_same_model(
    tmp_path: Path, compression: Compression
):
    model = tiny_upscaler()
    save_model_node(model, tmp_path, "model", True, compression)
    path = tmp_path / "model_compressed.onnx"
    assert [p.name for p in tmp_path.iterdir()] == [path.name]
    assert path.read_bytes().startswith(HEADERS[compression])

    loaded, directory, name = load_model_node(path)
    assert loaded.bytes == model.bytes
    assert loaded.info == model.info and loaded.info.scale_width == 2
    assert (directory, name) == (tmp_path, "model_compressed")


def test_a_plain_model_is_written_and_read_as_before(tmp_path: Path):
    model = tiny_upscaler()
    save_model_node(model, tmp_path, "model", False, Compression.ZSTD)
    path = tmp_path / "model.onnx"
    assert path.read_bytes() == model.bytes
    assert load_model_node(path)[0].bytes == model.bytes
