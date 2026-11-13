"""RDUAL through the nodes: Convert To ONNX's ONNX bundle (nodes/impl/onnx/rdual_bundle.py),
ONNX Save Model and Load Model, Build Engine's engine set, Save Engine, Load Engine and
Get Engine Info, and the refusals of every other user. Stand-in parts and engines, no
GPU; one real export of a small RDUAL model at 128x128 in the export child."""

from __future__ import annotations

import dataclasses
import importlib
import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import onnx
import pytest
import torch
from spandrel.architectures.RDUAL import RDUAL, RDUALArch

from api import NodeContext, NodeId, SettingsParser
from nodes.impl.compressed_file import Compression
from nodes.impl.onnx import rdual_bundle
from nodes.impl.onnx.load import load_onnx_model, model_facts
from nodes.impl.onnx.model import MIXED, OnnxGeneric, OnnxInfo, OnnxModel
from nodes.impl.tensorrt import build_cache, rdual
from nodes.impl.tensorrt import memory as trt_memory
from nodes.impl.tensorrt.model import TemporalInfo, TensorRTEngine, TensorRTEngineInfo
from nodes.impl.tensorrt.rdual_engine_set import (
    FORMAT_VERSION,
    EngineSetError,
    pack,
    pack_container,
    unpack,
)
from nodes.properties.outputs import OnnxModelOutput
from packages.chaiNNer_onnx.onnx.io.load_model import read_model
from packages.chaiNNer_onnx.onnx.io.save_model import save_model_node
from packages.chaiNNer_onnx.onnx.utility.optimize_model import optimize_model_node
from packages.chaiNNer_pytorch.pytorch.utility import convert_to_onnx
from packages.chaiNNer_pytorch.pytorch.utility.convert_to_onnx import (
    DualShape,
    OnnxDataType,
    Opset,
)
from packages.chaiNNer_tensorrt.settings import TensorRTSettings

PARTS = ("prefix", "revcore", "fwdcore", "flow")
MANIFEST = {
    "height": 96,
    "width": 160,
    "scale": 2,
    "plugin_key": "c64_h48_w80",
    "flow_precision": "mixed",
    "parts": {name: {"nodes": 1, "inputs": [], "outputs": []} for name in PARTS},
}
SETTINGS = TensorRTSettings(gpu_index=0, timing_cache_path=None, engine_cache_path=None)


def write_export(folder: Path, manifest: dict[str, Any] = MANIFEST) -> Path:
    """A stand-in of rdual_export.py's output folder."""
    for i, name in enumerate(manifest["parts"]):
        (folder / name).mkdir(parents=True)
        (folder / name / "model.onnx").write_bytes(f"onnx {name}".encode() * 50)
        (folder / name / "weights.bin").write_bytes(bytes([i]) * (4096 + i))
    (folder / "export.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


@pytest.fixture
def bundle(tmp_path: Path) -> bytes:
    return rdual_bundle.pack(write_export(tmp_path / "export"), 20)


def tensorrt_nodes(module: str, node: str) -> Any:
    """A TensorRT node's module, which defines `node` where its group is registered."""
    nodes = importlib.import_module(f"packages.chaiNNer_tensorrt.tensorrt.{module}")
    if not hasattr(nodes, node):
        pytest.skip("TensorRT's nodes are not registered here")
    return nodes


# The ONNX bundle


def test_a_bundle_holds_the_exports_files_and_writes_them_back(
    bundle: bytes, tmp_path: Path
):
    assert rdual_bundle.is_bundle(bundle)
    manifest, opset, files = rdual_bundle.unpack(bundle)
    assert (manifest, opset) == (MANIFEST, 20)
    assert list(files) == [f"{p}/{n}" for p in PARTS for n in rdual_bundle.PART_FILES]

    written = rdual_bundle.write(bundle, tmp_path / "out")

    assert written == MANIFEST
    source, out = tmp_path / "export", tmp_path / "out"
    for path in files:
        assert (out / path).read_bytes() == (source / path).read_bytes()
    assert json.loads((out / "export.json").read_text(encoding="utf-8")) == MANIFEST


def test_a_bundle_loads_as_a_fixed_mixed_rdual_onnx(bundle: bytes):
    model = load_onnx_model(bundle)

    assert isinstance(model, OnnxGeneric) and model.bytes == bundle
    assert model.info == OnnxInfo(
        opset=20,
        dtype=MIXED,
        scale_width=2,
        scale_height=2,
        fixed_input_width=160,
        fixed_input_height=96,
        input_channels=3,
        output_channels=3,
        arch="RDUAL",
    )
    assert not model_facts(model).native_session
    # Also for a model made from the bytes elsewhere
    assert not model_facts(OnnxGeneric(bundle, model.info)).native_session
    output = OnnxModelOutput()
    assert "RDUAL" in json.dumps(output.get_broadcast_type(model))
    assert "mixed (BF16/FP16/FP32)" in output.get_broadcast_data(model)["tags"]


def test_another_bundle_version_asks_for_a_new_conversion(tmp_path: Path):
    files = {f"{p}/{n}": b"x" for p in PARTS for n in rdual_bundle.PART_FILES}
    other = bytearray(
        pack_container(rdual_bundle.MAGIC, {"manifest": MANIFEST, "opset": 20}, files)
    )
    header = other.index(b'"format": 1')
    other[header : header + 11] = f'"format": {FORMAT_VERSION + 1}'.encode()

    with pytest.raises(EngineSetError, match="Convert the model to ONNX again"):
        rdual_bundle.unpack(bytes(other))


@pytest.mark.parametrize("damage", ["truncated", "flipped", "path"])
def test_a_damaged_bundle_is_refused(bundle: bytes, damage: str):
    data = bundle
    if damage == "truncated":
        data = bundle[:-1]
    elif damage == "flipped":
        data = bundle[:-100] + bytes([bundle[-100] ^ 0xFF]) + bundle[-99:]
    else:
        # A file outside the export's layout is never written out.
        manifest, _, files = rdual_bundle.unpack(bundle)
        files = {
            ("../" + k if k == "flow/model.onnx" else k): v for k, v in files.items()
        }
        data = pack_container(
            rdual_bundle.MAGIC, {"manifest": manifest, "opset": 20}, files
        )

    with pytest.raises(EngineSetError, match="RDUAL ONNX bundle"):
        rdual_bundle.unpack(data)


@pytest.mark.parametrize("compression", [None, *Compression])
def test_save_and_load_model_carry_a_bundle_as_one_file(
    bundle: bytes, tmp_path: Path, compression: Compression | None
):
    model = load_onnx_model(bundle)
    folder = tmp_path / "saved"
    save_model_node(
        model, folder, "rdual", compression is not None, compression or Compression.ZSTD
    )

    (path,) = folder.iterdir()
    loaded, _, _ = read_model(path)

    assert loaded.bytes == bundle and loaded.info == model.info


def test_load_model_refuses_a_damaged_bundle_file(bundle: bytes, tmp_path: Path):
    path = tmp_path / "rdual.onnx"
    path.write_bytes(bundle[:-1])

    with pytest.raises(
        ValueError, match=r"cannot be read.*RDUAL ONNX bundle is damaged"
    ):
        read_model(path)


def test_other_onnx_users_refuse_a_bundle(bundle: bytes):
    from nodes.impl.onnx.session import create_inference_session

    model = load_onnx_model(bundle)
    with pytest.raises(ValueError, match="ONNX Runtime cannot use an RDUAL ONNX"):
        create_inference_session(model, 0, "CPUExecutionProvider")
    with pytest.raises(ValueError, match="Optimize Model cannot use an RDUAL ONNX"):
        optimize_model_node(model)


# Convert To ONNX


class Context:
    """The NodeContext members Convert To ONNX uses: PyTorch on the CPU."""

    def __init__(self) -> None:
        self.settings = SettingsParser({"use_cpu": True})
        self.node_id = NodeId("node")
        self.aborted_checks = 0

    def check_aborted(self) -> None:
        self.aborted_checks += 1


@pytest.fixture(scope="module")
def descriptor() -> Any:
    torch.manual_seed(0)
    model = RDUAL(scale=2, use_checkpoint=False)
    model.get_buffer("flow_loaded").fill_(True)
    return RDUALArch().load(model.state_dict())


def convert(model: Any, shape: DualShape, height: int, width: int) -> Any:
    return convert_to_onnx.convert_to_onnx_node(
        cast(NodeContext, Context()),
        model,
        OnnxDataType.FP32,
        Opset.OPSET_20,
        False,
        shape,
        height,
        width,
    )


def test_convert_to_onnx_refuses_a_dynamic_rdual(descriptor: Any):
    with pytest.raises(ValueError, match="RDUAL converts to ONNX for one frame size"):
        convert(descriptor, DualShape.DYNAMIC, 96, 160)


def test_convert_to_onnx_bundles_rduals_export_once(
    descriptor: Any, monkeypatch: pytest.MonkeyPatch
):
    exports: list[tuple[int, int, int | None]] = []

    def export(
        model: Any, height: int, width: int, folder: Path, gpu: Any, context: Any
    ) -> dict:
        exports.append((height, width, gpu))
        manifest = {**MANIFEST, "height": height, "width": width}
        write_export(folder, manifest)
        return manifest

    monkeypatch.setattr(rdual, "export", export)

    # any frame size, not only multiples of 4 as for DUAL
    onnx_model, fp_mode, opset = convert(descriptor, DualShape.FIXED, 837, 1490)
    again = convert(descriptor, DualShape.FIXED, 837, 1490)

    assert exports == [(837, 1490, None)]
    assert again[0] is onnx_model
    assert (fp_mode, opset) == (MIXED, "opset20")
    assert rdual_bundle.is_bundle(onnx_model.bytes)
    info = onnx_model.info
    assert (info.fixed_input_height, info.fixed_input_width, info.arch) == (
        837,
        1490,
        "RDUAL",
    )


def test_a_small_rdual_converts_saves_and_loads_with_its_four_onnx_parts(
    descriptor: Any, tmp_path: Path
):
    onnx_model, _, _ = convert(descriptor, DualShape.FIXED, 128, 128)
    save_model_node(onnx_model, tmp_path, "rdual", True, Compression.ZSTD)
    loaded, _, _ = read_model(tmp_path / "rdual_compressed.onnx")

    manifest = rdual_bundle.write(loaded.bytes, tmp_path / "export")

    assert (manifest["height"], manifest["width"], manifest["scale"]) == (128, 128, 2)
    assert list(manifest["parts"]) == list(PARTS)
    for name in PARTS:
        part = onnx.load(str(tmp_path / "export" / name / "model.onnx"))
        assert len(part.graph.node) == manifest["parts"][name]["nodes"]


# Build Engine


@pytest.fixture
def memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_cache, "_memory", OrderedDict())
    monkeypatch.setattr(build_cache, "_building", {})


Builds = list[tuple[dict[str, bytes], tuple[str, ...]]]


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch, memory: None) -> Builds:
    """rdual.build's calls (the export folder's files and the settings), each writing
    stand-in engines, on a stand-in GPU."""
    calls: Builds = []

    def build(
        folder: Path, engines: Path, gpu: int, context: Any, settings: tuple[str, ...]
    ) -> None:
        files = {
            p.relative_to(folder).as_posix(): p.read_bytes()
            for p in sorted(folder.rglob("*"))
            if p.is_file()
        }
        calls.append((files, settings))
        engines.mkdir()
        for i, name in enumerate(json.loads(files["export.json"])["parts"]):
            (engines / f"{name}.engine").write_bytes(bytes([i + 1]) * (2000 + i))

    monkeypatch.setattr(rdual, "build", build)
    monkeypatch.setattr(rdual, "recipe_identity", lambda: "recipe")
    monkeypatch.setattr(trt_memory, "get_cuda_compute_capability", lambda _: (12, 0))
    build_engine = tensorrt_nodes("utility.build_engine", "build_engine_node")
    monkeypatch.setattr(build_engine, "get_cuda_compute_capability", lambda _: (12, 0))
    monkeypatch.setattr(build_engine, "get_cuda_device_name", lambda _: "RTX 5090")
    monkeypatch.setattr(build_engine, "get_settings", lambda _: SETTINGS)
    return calls


def build_node(model: OnnxModel, workspace: float = 0.0, level: int = 3) -> Any:
    build_engine = tensorrt_nodes("utility.build_engine", "build_engine_node")
    return build_engine.build_engine_node(
        cast(NodeContext, SimpleNamespace()),
        model,
        build_engine.Precision.FP16,
        build_engine.ShapeMode.DYNAMIC,  # ignored: the bundle fixes the size
        64,
        64,
        512,
        512,
        1024,
        1024,
        256,
        256,
        workspace,
        True,
        False,
        level,
        build_engine.TilingOptimization.FAST,
    )


def test_build_engine_packs_a_bundles_engines_into_one_set(
    bundle: bytes, builds: Builds
):
    engine = build_node(load_onnx_model(bundle))

    ((files, settings),) = builds
    assert settings == ("8.0", "3", "1", "0", "fast")
    _, _, bundled = rdual_bundle.unpack(bundle)
    assert {k: v for k, v in files.items() if k != "export.json"} == bundled
    assert json.loads(files["export.json"]) == MANIFEST
    info, parts = unpack(engine.bytes)
    assert engine.info == info
    assert parts == {n: bytes([i + 1]) * (2000 + i) for i, n in enumerate(PARTS)}
    assert info.temporal == TemporalInfo("RDUAL", 96, 160, "c64_h48_w80", PARTS)
    assert (info.precision, info.scale, info.gpu_architecture) == ("mixed", 2, "sm_120")
    assert info.min_shape == info.max_shape == (1, 3, 96, 160)
    assert not info.has_dynamic_shapes


def test_build_engine_keys_a_set_by_its_bundle_and_settings(
    bundle: bytes, builds: Builds, tmp_path: Path
):
    model = load_onnx_model(bundle)
    first = build_node(model)
    assert build_node(load_onnx_model(bundle)) is first  # the same bytes: kept
    build_node(model, workspace=4.0)
    build_node(model, level=5)
    other = rdual_bundle.pack(
        write_export(tmp_path / "other", {**MANIFEST, "height": 100}), 20
    )
    build_node(load_onnx_model(other))

    assert [settings[:2] for _, settings in builds] == [
        ("8.0", "3"),
        ("4.0", "3"),
        ("8.0", "5"),
        ("8.0", "3"),
    ]


# Save Engine, Load Engine, Get Engine Info

INFO = TensorRTEngineInfo(
    precision="mixed",
    input_channels=3,
    output_channels=3,
    scale=2,
    gpu_architecture="sm_120",
    tensorrt_version="11.4.0",
    has_dynamic_shapes=False,
    min_shape=(1, 3, 96, 160),
    opt_shape=(1, 3, 96, 160),
    max_shape=(1, 3, 96, 160),
    temporal=TemporalInfo("RDUAL", 96, 160, "c64_h48_w80", PARTS),
)


def engine_set() -> TensorRTEngine:
    return pack({n: bytes([i]) * 3000 for i, n in enumerate(PARTS)}, INFO)


def load_engine(monkeypatch: pytest.MonkeyPatch, path: Path) -> TensorRTEngine:
    nodes = tensorrt_nodes("io.load_engine", "load_engine_node")

    def deserialize(*_: object) -> None:
        raise AssertionError("an engine set is not deserialized as one engine")

    monkeypatch.setattr(nodes, "deserialize_engine", deserialize)
    monkeypatch.setattr(nodes, "get_settings", lambda _: SETTINGS)
    context = SimpleNamespace(node_id=NodeId(f"load {path.name}"))
    engine, _, _ = nodes.load_engine_node(cast(NodeContext, context), path)
    return engine


@pytest.mark.parametrize("compression", [None, *Compression])
def test_save_and_load_engine_carry_a_set_as_one_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compression: Compression | None
):
    saved = engine_set()
    save = tensorrt_nodes("io.save_engine", "save_engine_node").save_engine_node
    save(
        saved,
        tmp_path,
        "rdual",
        compression is not None,
        compression or Compression.LZMA,
    )
    (path,) = tmp_path.iterdir()

    loaded = load_engine(monkeypatch, path)

    assert loaded.bytes == saved.bytes and loaded.info == INFO


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("truncated", "RDUAL engine set is damaged"),
        ("version", r"format version 2.*Build the engine again"),
    ],
)
def test_load_engine_refuses_a_damaged_or_other_version_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str, message: str
):
    data = engine_set().bytes
    if damage == "truncated":
        data = data[:-1]
    else:
        data = data.replace(b'"format": 1', b'"format": 2', 1)
    path = tmp_path / "rdual.engine"
    path.write_bytes(data)

    with pytest.raises(EngineSetError, match=message):
        load_engine(monkeypatch, path)


def test_get_engine_info_shows_a_sets_temporal_info():
    info_node = tensorrt_nodes(
        "utility.get_engine_info", "get_engine_info_node"
    ).get_engine_info_node

    scale, precision, _, _, shape, temporal = info_node(engine_set())
    image = info_node(
        TensorRTEngine(b"engine", dataclasses.replace(INFO, temporal=None))
    )

    assert (scale, precision, shape) == (2, "MIXED", "Fixed: 96x160")
    assert temporal == (
        "RDUAL engine set: 160x96 frames (width x height), engines prefix, revcore,"
        " fwdcore, flow, plugins c64_h48_w80"
    )
    assert image[5] == "None"


def test_image_paths_refuse_a_set():
    from nodes.impl.tensorrt.engine_info import deserialize_engine

    with pytest.raises(ValueError, match="RDUAL engine set"):
        deserialize_engine(engine_set().bytes, 0)
