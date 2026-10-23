"""Load Model of NCNN and ONNX on damaged files: a ValueError that names the files and
keeps the reader's own error as its cause, where the reader's bare error reached the
user before (e.g. `KeyError: b'\\xd5\\xa5\\x00%'` for weights with an unknown tag).
A missing file still raises its OS error."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from google.protobuf.message import DecodeError

# The NCNN reader is native, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from packages.chaiNNer_ncnn.ncnn.io.load_model import (
    load_model_node as load_ncnn_model_node,
)
from packages.chaiNNer_onnx.onnx.io.load_model import (
    load_model_node as load_onnx_model_node,
)

# A 1x1 convolution that copies its 3 channels.
PARAM = "7767517\n2 2\nInput data 0 1 data\nConvolution conv 1 1 data out 0=3 1=1 6=9\n"
BIN = b"\x00\x00\x00\x00" + np.eye(3, dtype=np.float32).tobytes()


def ncnn_files(directory: Path, param: str, weights: bytes) -> tuple[Path, Path]:
    (directory / "tiny.param").write_text(param, encoding="utf-8")
    (directory / "tiny.bin").write_bytes(weights)
    return directory / "tiny.param", directory / "tiny.bin"


def test_the_tiny_ncnn_model_loads(tmp_path: Path):
    model, _, name = load_ncnn_model_node(*ncnn_files(tmp_path, PARAM, BIN))
    assert (name, model.scale, model.in_nc, model.out_nc) == ("tiny", 1, 3, 3)


@pytest.mark.parametrize(
    ("param", "weights", "cause"),
    [
        # weights whose 4-byte tag NCNN does not know
        (PARAM, b"\xd5\xa5\x00\x25" + BIN[4:], KeyError),
        # a parameter that is not a number
        (PARAM.replace("0=3", "0=nan"), BIN, ValueError),
        # a .bin shorter than the weights the .param declares
        (PARAM, BIN[:20], ValueError),
        # weights of a kernel 0 wide: no channel count follows from them
        (PARAM.replace("1=1", "1=0"), BIN, ZeroDivisionError),
    ],
)
def test_a_damaged_ncnn_model_names_its_files(
    tmp_path: Path, param: str, weights: bytes, cause: type[Exception]
):
    with pytest.raises(
        ValueError,
        match=r"^The NCNN model tiny\.param with tiny\.bin cannot be read \(",
    ) as raised:
        load_ncnn_model_node(*ncnn_files(tmp_path, param, weights))
    assert type(raised.value.__cause__) is cause


def test_a_missing_ncnn_file_raises_its_os_error(tmp_path: Path):
    param, weights = ncnn_files(tmp_path, PARAM, BIN)
    weights.unlink()
    with pytest.raises(FileNotFoundError):
        load_ncnn_model_node(param, weights)


def test_an_empty_onnx_file_is_refused(tmp_path: Path):
    # ONNX parses it as a model without a graph, which Load Model accepted
    path = tmp_path / "model.onnx"
    path.write_bytes(b"")
    with pytest.raises(
        ValueError,
        match=r"^The ONNX model model\.onnx cannot be read \(ValueError: the file is empty\)",
    ):
        load_onnx_model_node(path)


def test_a_damaged_onnx_model_names_its_file(tmp_path: Path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"\x0a\xff\xff\xff\xff\x0fnot a model")
    with pytest.raises(
        ValueError, match=r"^The ONNX model model\.onnx cannot be read \(DecodeError: "
    ) as raised:
        load_onnx_model_node(path)
    assert type(raised.value.__cause__) is DecodeError
