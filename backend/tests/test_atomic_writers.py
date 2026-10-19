"""The model, engine and script writers write a temporary file in the target folder and
rename it onto the final name (instant-exec A4; native/include/atomic_file.hpp): the bytes
each wrote before, an equal file left untouched, no partial file after a failure, foreign
files untouched, and errors naming the final path as before. CPU only: the ONNX model and
the TensorRT engine are their bytes, the PyTorch model a CPU module."""

from __future__ import annotations

import importlib
import lzma
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# The writers load the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from compression import zstd

from nodes.impl import compressed_file
from nodes.impl.compressed_file import Compression, write_file
from nodes.impl.ncnn import model as ncnn_model

OLD = 1_000_000_000  # An mtime (2001) no write of this test can produce.
DATA = b"model" + bytes(1 << 12) + os.urandom(512)


def aged(path: Path) -> Path:
    os.utime(path, (OLD, OLD))
    return path


def foreign(folder: Path, name: str) -> dict[Path, bytes]:
    folder.mkdir(parents=True, exist_ok=True)
    files = {
        folder / f"{name}.chainner-c-1-0.tmp": b"another process",
        folder / "unrelated.tmp": b"unrelated",
    }
    for path, data in files.items():
        path.write_bytes(data)
    return files


def contents(folder: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in folder.iterdir()}


@pytest.mark.parametrize("compression", [None, *Compression])
def test_write_file_bytes_equal_skip_and_replace(
    tmp_path: Path, compression: Compression | None
):
    # ONNX Save Model's and TensorRT Save Engine's writer.
    expected = {
        None: DATA,
        Compression.LZMA: lzma.compress(DATA),
        Compression.ZSTD: zstd.compress(DATA, compressed_file.ZSTD_LEVEL),
    }[compression]
    others = foreign(tmp_path, "m.onnx")
    path = tmp_path / "m.onnx"
    write_file(path, DATA, compression=compression)
    assert path.read_bytes() == expected
    write_file(aged(path), DATA, compression=compression)
    assert path.stat().st_mtime == OLD
    write_file(path, DATA + b"2", compression=compression)
    assert path.stat().st_mtime != OLD and path.read_bytes() != expected
    assert set(contents(tmp_path)) == {*others, path}


def test_write_file_errors_as_write_bytes_did(tmp_path: Path):
    path = tmp_path / "missing" / "m.engine"
    with pytest.raises(OSError) as before:
        path.write_bytes(DATA)
    with pytest.raises(OSError) as after:
        write_file(path, DATA, compression=None)
    assert type(after.value) is type(before.value) is FileNotFoundError
    assert str(after.value) == str(before.value)


def test_onnx_and_tensorrt_nodes_skip_an_equal_file(tmp_path: Path):
    from packages.chaiNNer_onnx.onnx.io.save_model import save_model_node

    model: Any = SimpleNamespace(bytes=DATA)  # the node writes model.bytes only
    save_model_node(model, tmp_path, "m", False, Compression.ZSTD)
    path = aged(tmp_path / "m.onnx")
    save_model_node(model, tmp_path, "m", False, Compression.ZSTD)
    assert path.read_bytes() == DATA and path.stat().st_mtime == OLD

    io = importlib.import_module("packages.chaiNNer_tensorrt.tensorrt.io.save_engine")
    if hasattr(io, "save_engine_node"):  # registered where TensorRT is installed
        engine: Any = SimpleNamespace(bytes=DATA)
        io.save_engine_node(engine, tmp_path, "e", False, Compression.LZMA)
        engine_path = aged(tmp_path / "e.engine")
        io.save_engine_node(engine, tmp_path, "e", False, Compression.LZMA)
        assert engine_path.read_bytes() == DATA
        assert engine_path.stat().st_mtime == OLD


def test_ncnn_param_and_bin_are_the_bytes_open_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    text = "7767517\n2 2\nInput in 0 1 in\r\nConvolution c 1 1 in out 0=4\n"
    weights = bytes(range(256)) * 4
    graph = ncnn_model.native_ncnn_graph
    monkeypatch.setattr(graph, "write_param", lambda model: text)
    monkeypatch.setattr(graph, "serialize_weights", lambda model: weights)
    model = ncnn_model.NcnnModel.__new__(ncnn_model.NcnnModel)
    reference = tmp_path / "reference"
    reference.mkdir()
    with open(reference / "m.param", "w", encoding="utf-8") as file:
        file.write(text)
    (reference / "m.bin").write_bytes(weights)
    out = tmp_path / "out"
    others = foreign(out, "m.param")
    model.write_param(out / "m.param")
    model.write_bin(out / "m.bin")
    for name in ("m.param", "m.bin"):
        assert (out / name).read_bytes() == (reference / name).read_bytes()
        aged(out / name)
    model.write_param(out / "m.param")
    model.write_bin(out / "m.bin")
    assert all((out / name).stat().st_mtime == OLD for name in ("m.param", "m.bin"))
    assert set(contents(out)) == {*others, out / "m.param", out / "m.bin"}
    assert model.write_param() == text  # without a file name: the text, no file


def test_pytorch_save_model_bytes_equal_skip_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    from packages.chaiNNer_pytorch.pytorch.io.save_model import (
        WeightFormat,
        save_model_node,
    )

    torch.manual_seed(0)
    module = torch.nn.Conv2d(3, 4, 3)
    model: Any = SimpleNamespace(model=module)
    reference = tmp_path / "reference"
    reference.mkdir()
    # torch.save names the archive inside after the file: the same name, the same bytes.
    torch.save(module.state_dict(), reference / "m.pth")
    save_file(module.state_dict(), reference / "m.safetensors")
    out = tmp_path / "out"
    others = foreign(out, "m.pth")
    for weight_format in WeightFormat:
        path = out / f"m.{weight_format.value}"
        save_model_node(model, out, "m", weight_format)
        assert path.read_bytes() == (reference / path.name).read_bytes()
        aged(path)
        save_model_node(model, out, "m", weight_format)
        assert path.stat().st_mtime == OLD

    staged: list[Path] = []

    def fails(state: object, file: Path):
        staged.append(Path(file))
        Path(file).write_bytes(b"partial")
        raise RuntimeError("the disk is full")

    monkeypatch.setattr(torch, "save", fails)
    with torch.no_grad():
        module.weight.add_(1)
    with pytest.raises(RuntimeError, match="the disk is full"):
        save_model_node(model, out, "m", WeightFormat.PTH)
    # Staged beside the target (one volume: the move is a rename), under the final name.
    assert staged[0].parent.parent == out and staged[0].name == "m.pth"
    assert (out / "m.pth").read_bytes() == (reference / "m.pth").read_bytes()
    assert set(contents(out)) == {*others, out / "m.pth", out / "m.safetensors"}


def test_vapoursynth_script_is_the_bytes_write_text_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    chains = importlib.import_module("packages.chaiNNer_vapoursynth.chains")
    script = "import vapoursynth as vs\ncore = vs.core\r\nclip.set_output(0)\n"
    monkeypatch.setattr(chains, "compile_script", lambda outputs, cache_mb: script)
    monkeypatch.setattr(
        chains, "get_settings", lambda context: SimpleNamespace(cache_mb=0)
    )
    reference = tmp_path / "reference.vpy"
    reference.write_text(script, encoding="utf-8")
    out = tmp_path / "out"
    others = foreign(out, "s.vpy")
    path = out / "s.vpy"
    assert chains.write_script(None, [], path) == path
    assert path.read_bytes() == reference.read_bytes()
    chains.write_script(None, [], aged(path))
    assert path.stat().st_mtime == OLD
    assert set(contents(out)) == {*others, path}


def test_align_model_download_writes_through_a_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    align = importlib.import_module(
        "packages.chaiNNer_pytorch.pytorch.processing.align_image_to_reference"
    )
    response = SimpleNamespace(content=DATA, raise_for_status=lambda: None)
    monkeypatch.setattr(align.requests, "get", lambda url: response)
    folder = tmp_path / "xfeat" / "weights"
    path = align.download_model("https://example.invalid/xfeat.pt", folder, "xfeat.pt")
    assert path == folder / "xfeat.pt"
    assert contents(folder) == {path: DATA}
