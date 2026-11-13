"""RDUAL's engine sets on the app side (nodes/impl/tensorrt/rdual.py): the gate, the
weights the export reads, and the build's checks; no GPU, TensorRT or Triton."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from spandrel.architectures.RDUAL import RDUAL, RDUALArch
from test_tensorrt_rdual_engine_set import INFO, parts

from nodes.impl.pytorch import auto_engine
from nodes.impl.tensorrt import build_cache, rdual, rdual_export
from nodes.impl.tensorrt.rdual_engine_set import pack


@pytest.fixture(scope="module")
def descriptor() -> Any:
    torch.manual_seed(0)
    model = RDUAL(scale=2, use_checkpoint=False)
    model.get_buffer("flow_loaded").fill_(True)
    return RDUALArch().load(model.state_dict())


def test_the_export_reads_the_folded_model_back(descriptor: Any, tmp_path: Path):
    weights = rdual.write_weights(descriptor.model, tmp_path / "w.safetensors")

    read = rdual_export.load(weights)

    # the hyperparameters as JSON gives them: a tuple comes back as a list
    assert json.dumps(read.hyperparameters) == json.dumps(
        descriptor.model.hyperparameters
    )
    mine, theirs = descriptor.model.state_dict(), read.state_dict()
    assert mine.keys() == theirs.keys()
    assert all(torch.equal(mine[k], theirs[k]) for k in mine)


def test_a_model_without_hyperparameters_is_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="hyperparameters"):
        rdual.write_weights(torch.nn.Linear(2, 2), tmp_path / "w.safetensors")


def test_only_checked_scales_and_rdual_models_have_sets(
    descriptor: Any, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(auto_engine, "no_engine_here", lambda _device: None)
    monkeypatch.setattr(auto_engine, "migraphx_here", lambda: False)
    gpu = torch.device("cuda")

    assert rdual.unavailable(descriptor, gpu) is None
    other = SimpleNamespace(architecture=SimpleNamespace(id="DUAL", name="DUAL"))
    assert "not a temporal model" in str(rdual.unavailable(cast(Any, other), gpu))
    four = SimpleNamespace(architecture=descriptor.architecture, scale=4)
    assert "checked" in str(rdual.unavailable(cast(Any, four), gpu))


def test_the_machines_reason_is_passed_on(
    descriptor: Any, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(auto_engine, "no_engine_here", lambda _device: "no TensorRT")
    assert rdual.unavailable(descriptor, torch.device("cuda")) == "no TensorRT"
    monkeypatch.setattr(auto_engine, "no_engine_here", lambda _device: None)
    monkeypatch.setattr(auto_engine, "migraphx_here", lambda: True)
    # MIGraphX's machines: no engine set, and nothing to warn about
    assert rdual.unavailable(descriptor, torch.device("cuda")) == ""


def test_a_build_without_triton_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(rdual, "find_spec", lambda _name: None)

    with pytest.raises(RuntimeError, match="Triton"):
        rdual.build(tmp_path, tmp_path / "engines", 0)


def test_a_set_is_kept_in_a_cache_folder_made_for_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(build_cache, "_memory", OrderedDict())
    monkeypatch.setattr(rdual, "engine_key", lambda *_: "key")
    monkeypatch.setattr(rdual, "build_set", lambda *_: pack(parts(), INFO))
    folder = tmp_path / "not" / "yet"

    engine = rdual.engine_set(cast(Any, None), 384, 384, 0, str(folder))

    assert engine.info == INFO
    assert [p.name for p in folder.iterdir()] == [f"key{build_cache.SUFFIX}"]


def test_a_build_passes_its_timing_cache_to_the_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls: list[list[str]] = []
    monkeypatch.setattr(rdual, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        rdual, "run_child", lambda arguments, *_: calls.append(arguments)
    )
    (tmp_path / "export.json").write_text(
        json.dumps({"local": "na-v1", "retrieval": "na-v1"}), encoding="utf-8"
    )

    rdual.build(tmp_path, tmp_path / "engines", 0)
    rdual.build(tmp_path, tmp_path / "engines", 0, timing_cache="cache.file")

    assert calls[0][-len(rdual.BUILD_SETTINGS) :] == list(rdual.BUILD_SETTINGS)
    assert calls[1][-1] == "cache.file"


@pytest.mark.parametrize(
    ("keys", "named"),
    [
        ((), "local and retrieval"),
        (("local",), "retrieval"),
        (("local", "retrieval"), None),
    ],
)
def test_a_bundle_from_before_local_and_retrieval_na_builds_with_a_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    keys: tuple[str, ...],
    named: str | None,
):
    calls: list[list[str]] = []
    monkeypatch.setattr(rdual, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        rdual, "run_child", lambda arguments, *_: calls.append(arguments)
    )
    manifest = {"context": "na-v1", **dict.fromkeys(keys, "na-v1")}
    (tmp_path / "export.json").write_text(json.dumps(manifest), encoding="utf-8")

    rdual.build(tmp_path, tmp_path / "engines", 0)

    assert len(calls) == 1
    warned = [
        r.getMessage() for r in caplog.records if "earlier windows" in r.getMessage()
    ]
    if named is None:
        assert warned == []
    else:
        (message,) = warned
        assert f"at its {named} sites" in message


def test_the_timing_cache_file_is_one_per_size_and_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from nodes.impl.tensorrt import memory

    monkeypatch.setattr(
        memory, "get_cuda_device_name", lambda _i: "NVIDIA GeForce RTX 5090"
    )
    monkeypatch.setattr(memory, "get_cuda_compute_capability", lambda _i: (12, 0))
    folder = tmp_path / "export"
    (folder / "prefix").mkdir(parents=True)
    (folder / "prefix" / "model.onnx").write_bytes(b"onnx")
    (folder / "export.json").write_text(
        json.dumps({"height": 837, "width": 1488, "parts": {"prefix": {}}}),
        encoding="utf-8",
    )

    name = Path(rdual.timing_cache_file(str(tmp_path / "timing"), folder, 0)).name

    assert name.startswith("rdual_1488x837_timing_")
    assert "RTX-5090_sm120" in name and name.endswith(".cache")
