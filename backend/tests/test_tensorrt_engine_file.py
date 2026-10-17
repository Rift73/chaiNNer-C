"""Save Engine's Compress: an engine file compressed with LZMA or Zstandard and named
"_compressed", which Load Engine's reader turns back into the same bytes; plain files
stay as they are."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from nodes.impl.tensorrt.engine_file import (
    XZ_HEADER,
    ZSTD_HEADER,
    EngineCompression,
    read_engine_file,
)
from nodes.impl.tensorrt.model import TensorRTEngine, TensorRTEngineInfo

# A serialized engine stand-in, compressible as real ones are (zero padding).
ENGINE = b"ftrt" + bytes(1 << 16) + os.urandom(4096)
INFO = TensorRTEngineInfo(
    precision="mixed",
    input_channels=3,
    output_channels=3,
    scale=1,
    gpu_architecture="sm_120",
    tensorrt_version="11.2",
    has_dynamic_shapes=False,
    min_shape=None,
    opt_shape=None,
    max_shape=None,
)
SaveEngine = Callable[[TensorRTEngine, Path, str, bool, EngineCompression], None]
HEADERS = {EngineCompression.LZMA: XZ_HEADER, EngineCompression.ZSTD: ZSTD_HEADER}


@pytest.fixture
def save_engine_node() -> SaveEngine:
    io = importlib.import_module("packages.chaiNNer_tensorrt.tensorrt.io.save_engine")
    if not hasattr(io, "save_engine_node"):
        pytest.skip("TensorRT's nodes are not registered here")
    return io.save_engine_node


def saved(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


@pytest.mark.parametrize("compression", list(EngineCompression))
def test_a_compressed_engine_reads_back_to_the_same_bytes(
    tmp_path: Path, save_engine_node: SaveEngine, compression: EngineCompression
):
    engine = TensorRTEngine(ENGINE, INFO)
    save_engine_node(engine, tmp_path, "model", True, compression)

    assert saved(tmp_path) == ["model_compressed.engine"]
    data = (tmp_path / "model_compressed.engine").read_bytes()
    assert data.startswith(HEADERS[compression])
    assert len(data) < len(ENGINE) // 4
    assert read_engine_file(tmp_path / "model_compressed.engine") == ENGINE

    # A compressed engine saved again keeps one suffix.
    save_engine_node(engine, tmp_path, "model_compressed", True, compression)
    assert saved(tmp_path) == ["model_compressed.engine"]


def test_an_uncompressed_engine_is_written_and_read_as_is(
    tmp_path: Path, save_engine_node: SaveEngine
):
    engine = TensorRTEngine(ENGINE, INFO)
    save_engine_node(engine, tmp_path, "model", False, EngineCompression.ZSTD)

    assert saved(tmp_path) == ["model.engine"]
    assert (tmp_path / "model.engine").read_bytes() == ENGINE
    assert read_engine_file(tmp_path / "model.engine") == ENGINE
