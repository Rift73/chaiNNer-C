"""ONNX Load Model parses a model once and keeps the file's own bytes when no tensor
can be kept outside it (its bytes lack the key "location"); a session and Build Engine
reuse what that parse found instead of parsing the bytes again."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import onnx
import pytest
from google.protobuf.message import DecodeError
from onnx import TensorProto, helper, numpy_helper

from nodes.impl.compressed_file import Compression
from nodes.impl.native_onnx_runtime import NativeSession
from nodes.impl.onnx import load as load_module
from nodes.impl.onnx.load import load_onnx_model, model_facts
from nodes.impl.onnx.model import MIXED, OnnxGeneric, OnnxInfo
from nodes.impl.onnx.session import create_inference_session
from nodes.impl.tensorrt.dual import METADATA_KEY
from packages.chaiNNer_onnx.onnx.io.load_model import load_model_node
from packages.chaiNNer_onnx.onnx.io.save_model import save_model_node

# Runs native code, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")


def gain_model(metadata: dict[str, str] | None = None) -> onnx.ModelProto:
    """A 1x model: each channel times its own weight."""
    weights = numpy_helper.from_array(
        np.array([0.5, 1.0, 2.0], np.float32).reshape(1, 3, 1, 1), "w"
    )
    graph = helper.make_graph(
        [helper.make_node("Mul", ["x", "w"], ["y"])],
        "gain",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [weights],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
    )
    for key, value in (metadata or {}).items():
        model.metadata_props.add(key=key, value=value)
    return model


def non_canonical(model: onnx.ModelProto) -> bytes:
    """`model` with its graph before its header: protobuf merges the two parts into the
    same model, which serializes in field order instead (other bytes)."""
    graph_only = onnx.ModelProto()
    graph_only.graph.CopyFrom(model.graph)
    header = onnx.ModelProto()
    header.CopyFrom(model)
    header.ClearField("graph")
    data = graph_only.SerializeToString() + header.SerializeToString()
    assert onnx.load_model_from_string(data) == model
    assert data != model.SerializeToString()
    return data


def run(session: object, image: np.ndarray) -> np.ndarray:
    assert isinstance(session, NativeSession)
    return np.asarray(session.run(None, {"x": image})[0])


def test_a_model_keeps_its_file_bytes(tmp_path: Path):
    path = tmp_path / "model.onnx"
    data = non_canonical(gain_model())
    path.write_bytes(data)
    loaded, _, _ = load_model_node(path)
    assert loaded.bytes == data
    assert onnx.load_model_from_string(loaded.bytes) == onnx.load_model(str(path))
    assert loaded.info.scale_width == 1 and loaded.info.dtype == "fp32"

    # Save Model writes those bytes back.
    save_model_node(loaded, tmp_path / "saved", "model", False, Compression.ZSTD)
    assert (tmp_path / "saved" / "model.onnx").read_bytes() == data

    # The native session runs them.
    image = np.random.default_rng(0).random((1, 3, 5, 7), dtype=np.float32)
    session = create_inference_session(loaded, 0, "CPUExecutionProvider")
    expected = image * np.array([0.5, 1.0, 2.0], np.float32)[None, :, None, None]
    np.testing.assert_array_equal(run(session, image), expected)


def test_a_compressed_model_keeps_its_decompressed_bytes(tmp_path: Path):
    data = non_canonical(gain_model())
    model = OnnxGeneric(data, OnnxInfo(17, "fp32"))
    save_model_node(model, tmp_path, "model", True, Compression.ZSTD)
    loaded, _, _ = load_model_node(tmp_path / "model_compressed.onnx")
    assert loaded.bytes == data


def test_a_model_with_external_data_loads_as_before(tmp_path: Path):
    path = tmp_path / "model.onnx"
    onnx.save_model(
        gain_model(),
        str(path),
        save_as_external_data=True,
        location="weights.bin",
        size_threshold=0,
    )
    assert b"location" in path.read_bytes()
    loaded, _, _ = load_model_node(path)
    assert loaded.bytes == onnx.load_model(str(path)).SerializeToString()
    assert b"location" not in loaded.bytes  # its weights are inside now


def test_a_corrupt_file_raises_what_it_raised(tmp_path: Path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"\x0a\xff\xff\xff\xff\x0fnot a model")
    with pytest.raises(DecodeError) as expected:
        onnx.load_model(str(path))
    with pytest.raises(DecodeError) as actual:
        load_model_node(path)
    assert str(actual.value) == str(expected.value)


def count_parses(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    parses: list[int] = []
    for name in ("load_model_from_string", "load_from_string"):
        parse = getattr(onnx, name)

        def counted(
            data: bytes,
            *args: object,
            parse: Callable[..., onnx.ModelProto] = parse,
        ) -> onnx.ModelProto:
            parses.append(len(data))
            return parse(data, *args)

        monkeypatch.setattr(onnx, name, counted)
    return parses


def test_a_session_and_the_dual_manifest_reuse_the_load_parse(
    monkeypatch: pytest.MonkeyPatch,
):
    manifest = {"specialization": {"dynamic": True}}
    plain = load_onnx_model(gain_model().SerializeToString())
    dual = load_onnx_model(
        gain_model({METADATA_KEY: json.dumps(manifest)}).SerializeToString()
    )
    assert dual.info.dtype == MIXED
    parses = count_parses(monkeypatch)
    assert model_facts(plain).native_session
    assert model_facts(plain).dual_manifest is None
    assert model_facts(dual).dual_manifest == manifest
    assert isinstance(
        create_inference_session(plain, 0, "CPUExecutionProvider"), NativeSession
    )
    assert parses == []


def test_a_model_made_otherwise_is_parsed_once(monkeypatch: pytest.MonkeyPatch):
    model = OnnxGeneric(gain_model().SerializeToString(), OnnxInfo(17, "fp32"))
    parses = count_parses(monkeypatch)
    assert model_facts(model) == model_facts(model)
    assert model_facts(model).native_session
    assert len(parses) == 1


def test_undecodable_bytes_reach_onnx_runtime(monkeypatch: pytest.MonkeyPatch):
    model = OnnxGeneric(b"\x0a\xff\xff\xff\xff\x0f", OnnxInfo(17, "fp32"))
    facts = model_facts(model)
    assert facts.native_session and facts.dual_manifest is None
    assert load_module.model_facts(model) is facts
