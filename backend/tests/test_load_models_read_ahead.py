"""Load Models (PyTorch and ONNX) reads the next model on one background thread while
the current one is used (from_list_read_ahead), and its items are what
Load Model gives for each file, in order, a failed file failing its own item."""

from __future__ import annotations

import threading
import time
from collections.abc import Generator as PythonGenerator
from pathlib import Path
from typing import TypeVar, cast

import pytest
import torch
from test_convert_to_onnx import ARCHS, FP16, FP32
from test_onnx_load_once import gain_model

from api import Generator, NodeContext, SettingsParser
from nodes.impl import numpy_pool
from nodes.impl.read_ahead import from_list_read_ahead
from nodes.utils.utils import list_all_files_sorted
from packages.chaiNNer_onnx.onnx.batch_processing.load_models import (
    load_models_node as load_onnx_models_node,
)
from packages.chaiNNer_onnx.onnx.io.load_model import (
    load_model_node as load_onnx_model_node,
)
from packages.chaiNNer_pytorch.pytorch.io.load_model import load_model
from packages.chaiNNer_pytorch.pytorch.iteration.load_models import load_models_node
from packages.chaiNNer_pytorch.settings import PyTorchSettings

T = TypeVar("T")


def items(generator: Generator[T]) -> list[T | Exception]:
    return list(generator.supplier())


def test_the_next_element_is_read_while_the_current_is_used():
    reads: list[tuple[str, int]] = []
    read_ahead = threading.Event()

    def read(x: str) -> str:
        reads.append((x, threading.get_ident()))
        if x == "b":
            read_ahead.set()
        return x.upper()

    def mapped(x: str, read: str, index: int) -> str:
        if index == 0:
            # "b" is read before item "a" is even returned
            assert read_ahead.wait(timeout=60)
        return f"{index}{x}{read}"

    assert items(from_list_read_ahead(["a", "b", "c"], read, mapped)) == [
        "0aA",
        "1bB",
        "2cC",
    ]
    assert [x for x, _ in reads] == ["a", "b", "c"]
    threads = {thread for _, thread in reads}
    assert len(threads) == 1 and threading.get_ident() not in threads


def test_a_failed_read_or_map_fails_its_own_item():
    def read(x: int) -> int:
        if x == 2:
            raise ValueError("unreadable 2")
        return x

    def mapped(x: int, read: int, index: int) -> int:
        if x == 3:
            raise RuntimeError("unmappable 3")
        return read * 10

    result = items(from_list_read_ahead([1, 2, 3, 4], read, mapped))
    assert result[0] == 10 and result[3] == 40
    assert isinstance(result[1], ValueError) and str(result[1]) == "unreadable 2"
    assert isinstance(result[2], RuntimeError) and str(result[2]) == "unmappable 3"


def test_a_stopped_iteration_reads_at_most_one_more():
    reads: list[int] = []
    release = threading.Event()

    def read(x: int) -> int:
        reads.append(x)
        if x == 1:
            assert release.wait(timeout=60)
        return x

    iterator = cast(
        PythonGenerator[int | Exception, None, None],
        from_list_read_ahead(
            [0, 1, 2, 3], read, lambda x, read, index: read
        ).supplier(),
    )
    assert next(iterator) == 0
    start = time.perf_counter()
    iterator.close()  # does not wait for the read in progress
    assert time.perf_counter() - start < 30
    release.set()
    time.sleep(0.2)
    assert reads == [0, 1]


def test_the_reader_installs_the_numpy_pool_handler(monkeypatch: pytest.MonkeyPatch):
    installed: list[int] = []
    monkeypatch.setattr(
        numpy_pool, "install", lambda: installed.append(threading.get_ident())
    )
    readers = items(
        from_list_read_ahead([0, 1], lambda x: threading.get_ident(), lambda x, r, i: r)
    )
    assert installed and set(readers) == set(installed)


def test_an_element_changed_after_its_read_ahead_is_read_again():
    # The chain writes the next file while the current item is used: the next item
    # is read again, as it would be without the read-ahead.
    versions = {"a": 0, "b": 0}
    b_read = threading.Event()

    def read(x: str) -> tuple[str, int]:
        version = versions[x]
        if x == "b":
            b_read.set()
        return x, version

    def use(x: str, value: tuple[str, int], index: int) -> tuple[str, int]:
        if x == "a":
            assert b_read.wait(60)
            versions["b"] += 1
        return value

    generator = from_list_read_ahead(["a", "b"], read, use, identity=versions.get)
    assert items(generator) == [("a", 0), ("b", 1)]


def test_an_empty_list_reads_nothing():
    assert items(from_list_read_ahead([], str, lambda x, r, i: r)) == []


class Context:
    """The NodeContext members PyTorch's Load Models uses."""

    def __init__(self, settings: PyTorchSettings) -> None:
        self.settings = SettingsParser(
            {"use_cpu": settings.use_cpu, "use_fp16": settings.use_fp16}
        )


def model_folder(directory: Path) -> None:
    for index, name in enumerate(["Compact", "ESRGAN", "HAT"]):
        torch.manual_seed(index)
        path = directory / ("sub" if name == "HAT" else "") / f"{index}_{name}.pth"
        path.parent.mkdir(exist_ok=True)
        torch.save(ARCHS[name]().state_dict(), path)
    (directory / "3_broken.pth").write_bytes(b"not a model")


@pytest.mark.parametrize("settings", [FP32, FP16], ids=["fp32", "fp16"])
def test_pytorch_load_models_gives_what_load_model_gives(
    tmp_path: Path, settings: PyTorchSettings
):
    model_folder(tmp_path)
    paths = list_all_files_sorted(tmp_path, [".pth"])
    generator, directory = load_models_node(
        cast(NodeContext, Context(settings)), tmp_path, False
    )
    assert directory == tmp_path and generator.expected_length == 4
    result = items(generator)
    for index, (path, item) in enumerate(zip(paths, result, strict=True)):
        if path.name == "3_broken.pth":
            with pytest.raises(ValueError) as expected_error:
                load_model(path, settings)
            assert isinstance(item, ValueError)
            assert str(item) == str(expected_error.value)
            continue
        assert isinstance(item, tuple)
        model, subdirectory, name, number = item
        expected = load_model(path, settings)
        assert (subdirectory, name, number) == (
            path.parent.relative_to(tmp_path).as_posix(),
            path.stem,
            index,
        )
        assert model.dtype == expected.dtype and model.device == expected.device
        state, reference = model.model.state_dict(), expected.model.state_dict()
        assert state.keys() == reference.keys()
        assert all(torch.equal(state[k], v) for k, v in reference.items())
        assert not any(p.requires_grad for p in model.model.parameters())
        assert not model.model.training


def test_onnx_load_models_gives_what_load_model_gives(tmp_path: Path):
    paths = []
    for index in range(3):
        path = tmp_path / f"{index}.onnx"
        path.write_bytes(gain_model({"index": str(index)}).SerializeToString())
        paths.append(path)
    generator, _ = load_onnx_models_node(tmp_path, False)
    result = items(generator)
    for index, (path, item) in enumerate(zip(paths, result, strict=True)):
        assert isinstance(item, tuple)
        model, subdirectory, name, number = item
        expected, _, _ = load_onnx_model_node(path)
        assert (subdirectory, name, number) == (".", path.stem, index)
        assert model.bytes == expected.bytes and model.info == expected.info
