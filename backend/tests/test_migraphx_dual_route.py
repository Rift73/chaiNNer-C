"""Build Program's DUAL route (nodes/impl/migraphx/dual.py): the DUAL ONNX of Convert To
ONNX, lowered in a child process to DUAL's custom ops with their kernels compiled for the
GPU's gfx (dual_lower_worker.py), then compiled in a child that loads the ops' library
(program.compile_in_child). The compile and the GPU are stand-ins here (the lowering is
real: Triton compiles for an explicit gfx without a GPU); test_migraphx_dual_gpu.py
runs the whole route on an AMD GPU."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest
import torch
from onnx import helper
from spandrel import ModelLoader
from spandrel.architectures.DUAL.__arch.dual_arch import dual_xs

import gpu
from nodes.impl.migraphx import dual, program, runner

# The lowering compiles kernels with Triton's AMD backend.
triton = pytest.importorskip("triton")
dual_hip = pytest.importorskip("nodes.impl.migraphx.dual_hip")
if "amd" not in triton.backends.backends:
    pytest.skip("this Triton has no AMD backend", allow_module_level=True)


def _export(tmp_path: Path, size: tuple[int, int] | None) -> bytes:
    torch.manual_seed(0)
    path = tmp_path / "xs.pth"
    torch.save(dual_xs(scale=1).eval().state_dict(), path)
    descriptor = ModelLoader("cpu").load_from_file(path)  # folded on load
    return dual.export_onnx(
        descriptor.model.state_dict(), descriptor.tags, descriptor.scale, size
    )


@pytest.fixture(scope="module")
def fixed_onnx(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    return _export(tmp_path_factory.mktemp("fixed"), (32, 64))


@pytest.fixture(scope="module")
def dynamic_onnx(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    return _export(tmp_path_factory.mktemp("dynamic"), None)


@pytest.fixture
def library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    library = tmp_path / dual.OP_LIBRARY.name
    library.write_bytes(b"library")
    monkeypatch.setattr(dual, "OP_LIBRARY", library)
    return library


@pytest.fixture
def compiles(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple, dict]]:
    calls: list[tuple[tuple, dict]] = []

    def compile_in_child(*args: object, **kwargs: object) -> bytes:
        calls.append((args, kwargs))
        return b"program"

    monkeypatch.setattr(program, "compile_in_child", compile_in_child)
    # HIP's device 1, an RDNA3 GPU
    amd = SimpleNamespace(devices=[gpu.AmdDevice(1, "Radeon RX 7900 XTX", "gfx1100")])
    monkeypatch.setattr(gpu, "amd", amd)
    return calls


def test_a_fixed_onnx_compiles_lowered_at_its_own_size(
    fixed_onnx: bytes, library: Path, compiles: list[tuple[tuple, dict]]
):
    manifest = dual.export_manifest(fixed_onnx)
    assert manifest is not None
    built = dual.build_program(fixed_onnx, manifest, 1, None, exhaustive_tune=True)
    assert built == b"program"
    ((args, kwargs),) = compiles
    assert args[1:] == (1, (32, 64))
    assert kwargs == {
        "fp16": False,
        "fp8": False,
        "exhaustive_tune": True,
        "op_libraries": (library,),
        "label": "DUAL program build",
        "context": None,
    }
    lowered = onnx.load_from_string(args[0])
    graph = lowered.graph
    # Named for the alignment, as a TensorRT engine's input
    assert graph.input[0].name == dual.aligned_input_name(dual.ALIGNMENT)
    assert {node.domain for node in graph.node} == {"", "chainner"}
    assert dual.export_manifest(lowered) == manifest
    # The child compiled the kernels for the device's gfx (gfx1100).
    (norm, *_) = [n for n in graph.node if n.op_type == "chainner_dual_norm"]
    attributes = {a.name: helper.get_attribute_value(a) for a in norm.attribute}
    epsilon = next(
        helper.get_attribute_value(a)
        for n in onnx.load_from_string(fixed_onnx).graph.node
        if n.name == norm.name
        for a in n.attribute
        if a.name == "epsilon"
    )
    expected = dual_hip.kernel("norm", 128, dual_hip.Target.amd("gfx1100"), epsilon)
    assert attributes["code_id"].decode() == expected["code_id"]


def test_a_dynamic_onnx_compiles_at_the_given_size(
    dynamic_onnx: bytes, library: Path, compiles: list[tuple[tuple, dict]]
):
    manifest = dual.export_manifest(dynamic_onnx)
    assert manifest is not None
    dual.build_program(dynamic_onnx, manifest, 1, (48, 72))
    ((args, _),) = compiles
    assert args[1:] == (1, (48, 72))
    graph = onnx.load_from_string(args[0]).graph
    assert graph.input[0].name == dual.aligned_input_name(4)
    # Made one of that size (dual_hip.fix_size)
    dims = graph.input[0].type.tensor_type.shape.dim
    assert [d.dim_value for d in dims] == [1, 3, 48, 72]
    assert not [n for n in graph.node if n.domain == "trt"]


def test_only_a_dynamic_onnx_takes_a_size(
    fixed_onnx: bytes, dynamic_onnx: bytes, library: Path, compiles: list
):
    for onnx_bytes, size in ((fixed_onnx, (32, 64)), (dynamic_onnx, None)):
        manifest = dual.export_manifest(onnx_bytes)
        assert manifest is not None
        with pytest.raises(ValueError, match="Only a dynamic DUAL ONNX takes a size"):
            dual.build_program(onnx_bytes, manifest, 1, size)
    assert not compiles


def test_a_gpu_that_is_not_an_amd_one_is_refused(
    fixed_onnx: bytes, library: Path, compiles: list
):
    manifest = dual.export_manifest(fixed_onnx)
    assert manifest is not None
    with pytest.raises(ValueError, match="no AMD GPU 0"):
        dual.build_program(fixed_onnx, manifest, 0, None)


def test_a_gpu_without_a_gfx_target_is_refused(
    fixed_onnx: bytes, library: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(program, "compile_in_child", pytest.fail)
    amd = SimpleNamespace(devices=[gpu.AmdDevice(0, "Radeon", "")])
    monkeypatch.setattr(gpu, "amd", amd)
    manifest = dual.export_manifest(fixed_onnx)
    assert manifest is not None
    with pytest.raises(ValueError, match="no gfx target for Radeon"):
        dual.build_program(fixed_onnx, manifest, 0, None)


def test_a_build_without_the_op_library_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(dual, "OP_LIBRARY", tmp_path / dual.OP_LIBRARY.name)
    monkeypatch.setattr(program, "compile_in_child", pytest.fail)
    manifest = {"specialization": {"dynamic": False}}
    with pytest.raises(RuntimeError, match="this build was made without it"):
        dual.build_program(b"onnx", manifest, 0, None)


@pytest.mark.parametrize("present", [True, False])
def test_processes_register_the_op_library_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, present: bool
):
    """Every process that loads programs registers the library first, once; a build
    without it registers nothing (only DUAL programs then fail to load)."""
    library = tmp_path / dual.OP_LIBRARY.name
    if present:
        library.write_bytes(b"library")
    monkeypatch.setattr(dual, "OP_LIBRARY", library)
    loaded: list[str] = []
    module = SimpleNamespace(load_op_library=loaded.append)
    monkeypatch.setattr(runner, "load_migraphx", lambda: module)
    dual.load_ops.cache_clear()
    try:
        dual.load_ops()
        dual.load_ops()
    finally:
        dual.load_ops.cache_clear()
    assert loaded == ([str(library)] if present else [])


def test_the_build_identity_follows_the_op_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    library = tmp_path / dual.OP_LIBRARY.name
    monkeypatch.setattr(dual, "OP_LIBRARY", library)
    identities = []
    for content in (None, b"one", b"two"):
        if content is not None:
            library.write_bytes(content)
        dual.build_identity.cache_clear()
        identities.append(dual.build_identity())
    dual.build_identity.cache_clear()
    assert len(set(identities)) == 3


def test_the_build_identity_covers_the_kernels_and_the_lowering():
    names = {path.name for path in dual.BUILD_SOURCES}
    assert names == {
        "dual.py",
        "dual_hip.py",
        "dual_lower_worker.py",
        "dual_dynamic_kernels.py",
        "dual_na_kernel.py",
    }
    assert all(path.is_file() for path in dual.BUILD_SOURCES)
