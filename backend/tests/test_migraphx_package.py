"""The MIGraphX package (packages/chaiNNer_migraphx): when it is enabled, what it installs,
and its nodes, on stand-ins for the AMD GPUs, the ROCm dependency lists, the runner's
module and the DUAL and DRFT routes (migraphx_stand_ins.py). No AMD GPU or MIGraphX is
needed."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from migraphx_stand_ins import (
    ABSENT,
    INSTALLED,
    RADEON,
    TRITON_ROCM,
    TRITON_WINDOWS,
    Here,
    Imported,
    Runner,
    amd_device,
    context,
    import_package,
    package_imports,
)
from onnx import TensorProto, helper

from api import Dependency
from nodes.condition import Condition
from nodes.impl.compressed_file import Compression, read_file
from nodes.impl.migraphx import build_cache
from nodes.impl.migraphx.model import MIGraphXProgram, MIGraphXProgramInfo
from nodes.impl.onnx.load import load_onnx_model
from nodes.impl.onnx.model import (
    ARCH_METADATA_KEY,
    PRECISION_METADATA_KEY,
    OnnxGeneric,
    OnnxInfo,
)
from nodes.impl.tensorrt import cache

WINDOWS = Here(linux=False, windows=True)
INFO = MIGraphXProgramInfo(
    precision="fp16",
    input_channels=3,
    output_channels=3,
    scale=2,
    gfx="gfx1100",
    migraphx_version="2.16.1",
    height=256,
    width=256,
)


Load = Callable[..., Imported]


@pytest.fixture
def load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Load]:
    def load(here: Here | None = None) -> Imported:
        return import_package(monkeypatch, tmp_path, here or Here())

    with package_imports():
        yield load


@pytest.mark.parametrize(
    ("here", "reason"),
    [
        (Here(devices=[]), "MIGraphX requires an AMD GPU with ROCm support"),
        (
            Here(
                devices=[
                    amd_device(0, "AMD Radeon Graphics", "gfx1036", compute_units=2)
                ]
            ),
            "MIGraphX requires an AMD GPU with at least 8 compute units",
        ),
        (
            Here(
                devices=[
                    amd_device(0, "AMD Radeon Graphics", "gfx1036", compute_units=2),
                    amd_device(1, "Radeon VII", "gfx906"),
                ]
            ),
            "supports with at least 8 compute units",
        ),
        # macOS
        (Here(linux=False), "MIGraphX requires Linux or Windows"),
        (Here(devices=[], linux=False, windows=True), "requires an AMD GPU with ROCm"),
        # An NVIDIA GPU keeps the NVIDIA stack alone, on Windows too.
        (Here(nvidia=True), "MIGraphX runs only where no NVIDIA GPU is found"),
        (
            Here(nvidia=True, linux=False, windows=True),
            "MIGraphX runs only where no NVIDIA GPU is found",
        ),
        (
            Here(devices=[amd_device(0, "Radeon VII", "gfx906")]),
            "an AMD GPU that ROCm 7.14 supports (this computer's: gfx906)",
        ),
        (Here(runner_built=False), "has no MIGraphX runner (_chainner_migraphx."),
        (
            Here(load_error=ImportError("libmigraphx_c.so.3: cannot open")),
            "MIGraphX did not load: libmigraphx_c.so.3: cannot open",
        ),
    ],
)
def test_the_package_is_disabled_with_its_reason_and_has_no_nodes(
    load: Load, here: Here, reason: str
):
    imported = load(here)
    package = imported.package.package
    assert package.disabled
    assert reason in package.disabled_reason
    assert package.categories == []
    # No MIGraphX settings tab on a computer the package does not run on
    assert package.settings == []
    # The node modules import, registering nothing
    assert not hasattr(imported.nodes["utility.build_program"], "build_program_node")


def test_a_disabled_package_lists_no_dependencies(load: Load):
    # ONNX Runtime's MIGraphX build replaces the onnxruntime module of the ONNX package.
    assert load(Here(devices=[])).package.package.dependencies == []
    enabled = load().package.package
    assert enabled.dependencies == [INSTALLED, INSTALLED, TRITON_ROCM]


def test_on_windows_it_lists_triton_for_windows(load: Load):
    # ROCm's runtime, ONNX Runtime's MIGraphX build and the Triton DUAL's programs need.
    imported = load(WINDOWS)
    package = imported.package.package
    assert not package.disabled
    assert imported.package.migraphx_loaded
    assert package.dependencies == [INSTALLED, INSTALLED, TRITON_WINDOWS]
    assert hasattr(imported.nodes["utility.build_program"], "build_program_node")


def test_without_its_dependencies_it_stays_installable_without_nodes(load: Load):
    imported = load(Here(dependencies=[ABSENT]))
    package = imported.package.package
    assert not package.disabled
    assert not imported.package.migraphx_loaded
    assert not hasattr(imported.nodes["utility.build_program"], "build_program_node")


def test_a_runner_that_does_not_load_is_checked_only_at_the_listed_versions(load: Load):
    # Another version is the dependency manager's to update, which a disabled package
    # could not: the package stays enabled, without nodes.
    other = Dependency("pytest", "pytest", "0.0.1", 0)
    imported = load(Here(dependencies=[other], load_error=OSError("no HIP")))
    assert not imported.package.package.disabled


def test_with_migraphx_loaded_every_node_is_registered(load: Load):
    imported = load()
    assert imported.package.migraphx_loaded
    assert not imported.package.package.disabled
    for name, function in [
        ("io.load_program", "load_program_node"),
        ("io.save_program", "save_program_node"),
        ("processing.upscale_image", "upscale_image_node"),
        ("utility.build_program", "build_program_node"),
        ("utility.get_program_info", "get_program_info_node"),
    ]:
        assert hasattr(imported.nodes[name], function), name


def test_the_gpu_setting_lists_the_gpus_rocm_supports_with_8_compute_units(load: Load):
    imported = load(
        Here(
            devices=[
                # A Ryzen's graphics often come first in HIP's order.
                amd_device(0, "AMD Radeon Graphics", "gfx1036", compute_units=2),
                amd_device(1, "Radeon VII", "gfx906"),
                amd_device(2, RADEON, "gfx1100", compute_units=84),
                amd_device(3, "AMD Instinct MI300X", "gfx942", compute_units=304),
            ]
        )
    )
    setting = imported.package.package.settings[0]
    assert setting.key == "gpu_index"
    assert setting.options == [
        {"label": f"{RADEON} (gfx1100)", "value": "2"},
        {"label": "AMD Instinct MI300X (gfx942)", "value": "3"},
    ]
    # The strongest GPU
    assert setting.default == "3"


def test_the_default_gpu_is_a_listed_one(load: Load):
    # The strongest GPU has no ROCm kernels: the first listed one is the default.
    imported = load(
        Here(
            devices=[
                amd_device(0, RADEON, "gfx1100", compute_units=84),
                amd_device(1, "Radeon VII", "gfx906", compute_units=120),
            ]
        )
    )
    assert imported.package.package.settings[0].default == "0"


def node_data(imported: Imported, schema_id: str) -> Any:
    category = imported.package.package.categories[0]
    return next(
        node
        for group in category.node_groups
        for node in group.nodes
        if node.schema_id == schema_id
    )


@pytest.mark.parametrize(("gfx", "fp8"), [("gfx1100", False), ("gfx942", True)])
def test_fp8_is_offered_only_when_a_gpu_supports_it(load: Load, gfx: str, fp8: bool):
    imported = load(Here(devices=[amd_device(0, "GPU", gfx)]))
    node = node_data(imported, "chainner:migraphx:build_program")
    precision = next(i for i in node.inputs if i.label == "Precision")
    option = next(o for o in precision.options if o["value"] == "fp8")
    assert option["condition"] == Condition.const(fp8).to_json()


class Compiles:
    """compile_in_child's stand-in: records its calls and makes distinct programs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, onnx_bytes: bytes, gpu_index: int, size: Any, **kwargs) -> bytes:
        self.calls.append({"gpu_index": gpu_index, "size": size, **kwargs})
        return f"program {len(self.calls)}".encode()


@pytest.fixture
def build(load: Load, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Build Program with stand-ins for the compile and the program's info."""
    monkeypatch.setattr(build_cache, "_memory", OrderedDict())
    monkeypatch.setattr(build_cache, "_building", {})
    imported = load(
        Here(
            devices=[
                amd_device(0, RADEON, "gfx1100"),
                amd_device(1, "MI300X", "gfx942"),
            ]
        )
    )
    module = imported.nodes["utility.build_program"]
    build = SimpleNamespace(
        node=module.build_program_node,
        precision=module.ProgramPrecision,
        compiles=Compiles(),
        manifests=SimpleNamespace(drft_manifest=None, dual_manifest=None),
        read=[],
        imported=imported,
    )
    monkeypatch.setattr(module, "compile_in_child", build.compiles)
    monkeypatch.setattr(module, "model_facts", lambda _model: build.manifests)
    monkeypatch.setattr(module, "runner_identity", lambda: "runner")

    def read_program_info(data: bytes, gpu_index: int) -> MIGraphXProgramInfo:
        build.read.append((data, gpu_index))
        return INFO

    monkeypatch.setattr(module, "read_program_info", read_program_info)
    return build


def onnx(dtype: str = "fp32", data: bytes = b"onnx model") -> OnnxGeneric:
    return OnnxGeneric(data, OnnxInfo(17, dtype, input_channels=3))


def run(
    build: SimpleNamespace,
    model: OnnxGeneric,
    precision: str = "FP16",
    hip_graph: bool = False,
    **settings,
) -> Any:
    return build.node(
        context({"gpu_index": "0", **settings}),
        model,
        build.precision[precision],
        128,
        192,
        False,
        hip_graph,
    )


def test_build_program_compiles_once_per_build_inputs(build: SimpleNamespace):
    model = onnx()
    program = run(build, model)
    assert build.compiles.calls == [
        {
            "gpu_index": 0,
            "size": (128, 192),
            "fp16": True,
            "fp8": False,
            "exhaustive_tune": False,
            "hip_graph": False,
            "channels": 3,
            "context": build.compiles.calls[0]["context"],
        }
    ]
    assert program.bytes == b"program 1" and program.info is INFO
    # The info comes from the compiled program, loaded on the GPU it was built for.
    assert build.read == [(b"program 1", 0)]
    assert run(build, model) is program
    assert run(build, model, "FP32").bytes == b"program 2"
    assert build.compiles.calls[1]["fp16"] is False


def test_use_hip_graph_is_a_new_input_off_by_default_with_a_stable_id(load: Load):
    imported = load()
    node = node_data(imported, "chainner:migraphx:build_program")
    # A saved chain holds its inputs by id: the older inputs keep theirs, the new one
    # keeps this one, and a chain without it takes the default.
    assert [(i.id, i.label) for i in node.inputs] == [
        (0, "ONNX Model"),
        (1, "Precision"),
        (2, "Height"),
        (3, "Width"),
        (4, "Exhaustive Tuning"),
        (5, "Use HIP Graph"),
    ]
    option = next(i for i in node.inputs if i.label == "Use HIP Graph")
    assert option.enforce(option.default) is False
    assert "Experimental" in option.description
    assert "Get Program Info" in option.description


def test_use_hip_graph_reaches_the_compile_and_changes_the_cache_key(
    build: SimpleNamespace,
):
    model = onnx()
    off = run(build, model)
    on = run(build, model, hip_graph=True)
    assert [call["hip_graph"] for call in build.compiles.calls] == [False, True]
    assert off.bytes != on.bytes
    # Each is kept under its own key: neither build runs again.
    assert run(build, model) is off
    assert run(build, model, hip_graph=True) is on
    assert len(build.compiles.calls) == 2


@pytest.mark.parametrize(
    ("manifests", "route"),
    [
        (
            {
                "drft_manifest": None,
                "dual_manifest": {
                    "specialization": {"dynamic": True, "factory": "s", "scale": 2}
                },
            },
            "dual",
        ),
        (
            {"drft_manifest": {"scale": 2, "window_size": 8}, "dual_manifest": None},
            "drft",
        ),
    ],
)
def test_use_hip_graph_reaches_the_custom_op_routes_under_their_own_keys(
    build: SimpleNamespace, manifests: dict[str, Any], route: str
):
    vars(build.manifests).update(manifests)
    routed = getattr(build.imported, route)
    run(build, onnx("mixed"))
    run(build, onnx("mixed"), hip_graph=True)
    run(build, onnx("mixed"), hip_graph=True)
    assert routed.hip_graph_builds == [False, True]


def test_a_mixed_precision_onnx_keeps_its_own_types(build: SimpleNamespace):
    run(build, onnx("mixed-fp16"), "FP16")
    assert build.compiles.calls[0]["fp16"] is False
    assert build.compiles.calls[0]["fp8"] is False


def test_fp8_needs_a_gpu_that_supports_it(build: SimpleNamespace):
    with pytest.raises(ValueError, match="FP8 needs a GPU that supports it"):
        run(build, onnx(), "FP8")
    run(build, onnx(), "FP8", gpu_index="1")
    assert build.compiles.calls[0]["fp8"] is True
    assert build.compiles.calls[0]["gpu_index"] == 1


def test_a_gpu_index_rocm_does_not_support_is_refused(build: SimpleNamespace):
    with pytest.raises(RuntimeError, match="names GPU 7"):
        run(build, onnx(), gpu_index="7")


def test_the_program_cache_folder_keeps_programs_across_restarts(
    build: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    folder = tmp_path / "cache"
    run(build, onnx(), program_cache=str(folder))
    assert build.compiles.calls and folder.is_dir()
    monkeypatch.setattr(build_cache, "_memory", OrderedDict())  # a new process
    program = run(build, onnx(), program_cache=str(folder))
    assert len(build.compiles.calls) == 1
    assert program.bytes == b"program 1" and program.info == INFO


def test_a_fixed_dual_onnx_builds_its_own_size(build: SimpleNamespace):
    dual = build.imported.dual
    build.manifests.dual_manifest = {
        "specialization": {"dynamic": False, "factory": "dual_s", "scale": 2}
    }
    program = run(build, onnx("mixed"))
    assert dual.profiles == []
    assert dual.builds == [(0, None, False)]
    assert program.bytes == b"route program" and program.info is INFO
    assert build.compiles.calls == []


def test_a_dynamic_dual_onnx_takes_the_nodes_size(build: SimpleNamespace):
    dual = build.imported.dual
    build.manifests.dual_manifest = {
        "specialization": {"dynamic": True, "factory": "dual_s", "scale": 2}
    }
    run(build, onnx("mixed"))
    assert dual.profiles == [((128, 192),) * 3]
    assert dual.builds == [(0, (128, 192), False)]


def test_a_drft_onnx_takes_the_nodes_size(build: SimpleNamespace):
    drft = build.imported.drft
    build.manifests.drft_manifest = {"scale": 2, "window_size": 16}
    run(build, onnx("mixed"))
    assert drft.profiles == [((128, 192),) * 3]
    assert drft.builds == [(0, (128, 192), False)]


@pytest.mark.parametrize(
    ("here", "triton", "other"),
    [
        (Here(), TRITON_ROCM, TRITON_WINDOWS),
        (WINDOWS, TRITON_WINDOWS, TRITON_ROCM),
    ],
    ids=["linux", "windows"],
)
def test_each_system_installs_its_triton(
    load: Load, here: Here, triton: Dependency, other: Dependency
):
    dependencies = load(here).package.package.dependencies
    assert triton in dependencies
    assert other not in dependencies


def test_on_windows_dual_and_drft_build(load: Load, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(build_cache, "_memory", OrderedDict())
    monkeypatch.setattr(build_cache, "_building", {})
    imported = load(WINDOWS)
    module = imported.nodes["utility.build_program"]
    manifests = SimpleNamespace(
        drft_manifest=None,
        dual_manifest={
            "specialization": {"dynamic": True, "factory": "dual_s", "scale": 2}
        },
    )
    monkeypatch.setattr(module, "model_facts", lambda _model: manifests)
    monkeypatch.setattr(module, "runner_identity", lambda: "runner")
    monkeypatch.setattr(module, "read_program_info", lambda _data, _gpu: INFO)

    def build() -> Any:
        return module.build_program_node(
            context({"gpu_index": "0"}),
            onnx("mixed"),
            module.ProgramPrecision.FP16,
            128,
            192,
            False,
            False,
        )

    assert build().bytes == b"route program"
    assert imported.dual.profiles == [((128, 192),) * 3]
    assert imported.dual.builds == [(0, (128, 192), False)]
    manifests.dual_manifest, manifests.drft_manifest = (
        None,
        {"scale": 2, "window_size": 16},
    )
    assert build().bytes == b"route program"
    assert imported.drft.builds == [(0, (128, 192), False)]


def test_each_route_and_its_identity_is_in_the_cache_key(build: SimpleNamespace):
    build.manifests.drft_manifest = {"scale": 2, "window_size": 16}
    run(build, onnx("mixed"))
    build.manifests.drft_manifest = None
    build.manifests.dual_manifest = {
        "specialization": {"dynamic": True, "factory": "dual_s", "scale": 2}
    }
    run(build, onnx("mixed"))
    assert len(build.imported.drft.builds) == len(build.imported.dual.builds) == 1


@pytest.fixture
def loaded(load: Load) -> Imported:
    Runner.made = []
    return load()


def test_save_and_load_program_round_trip(
    loaded: Imported, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    save = loaded.nodes["io.save_program"].save_program_node
    program = MIGraphXProgram(b"\x00chaiNNer program" * 100, INFO)
    save(program, tmp_path, "model", True, Compression.ZSTD)
    path = tmp_path / "model_compressed.mxr"
    assert read_file(path) == program.bytes
    save(program, tmp_path, "plain", False, Compression.LZMA)
    assert (tmp_path / "plain.mxr").read_bytes() == program.bytes

    module = loaded.nodes["io.load_program"]
    reads: list[tuple[bytes, int]] = []

    def read_program_info(data: bytes, gpu_index: int) -> MIGraphXProgramInfo:
        reads.append((data, gpu_index))
        return INFO

    monkeypatch.setattr(module, "read_program_info", read_program_info)
    load_context = context({"gpu_index": "0"}, "load")
    loaded_program, directory, name = module.load_program_node(load_context, path)
    assert loaded_program.bytes == program.bytes and loaded_program.info is INFO
    assert (directory, name) == (tmp_path, "model_compressed")
    # Kept across runs: Upscale Image keeps its session
    assert module.load_program_node(load_context, path)[0] is loaded_program
    assert len(reads) == 1
    cache.release_node(load_context.node_id)
    assert module.load_program_node(load_context, path)[0] is not loaded_program


def test_upscale_image_runs_the_runner_and_keeps_its_session(loaded: Imported):
    upscale = loaded.nodes["processing.upscale_image"].upscale_image_node
    program = MIGraphXProgram(b"program", INFO)
    upscale_context = context({"gpu_index": "0"}, "upscale")
    image = np.random.default_rng(1).random((5, 7, 3), dtype=np.float32)
    result = upscale(upscale_context, image, program, False)
    assert np.array_equal(result, image.repeat(2, 0).repeat(2, 1))
    assert upscale(upscale_context, image, program, False).shape == (10, 14, 3)
    (made,) = Runner.made
    assert (made.program, made.device, made.streams) == (b"program", 0, 1)
    cache.release_node(upscale_context.node_id)
    assert made.closed


def test_a_new_file_in_load_program_frees_the_old_program_s_session(
    loaded: Imported, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = loaded.nodes["io.load_program"]
    monkeypatch.setattr(module, "read_program_info", lambda _data, _gpu_index: INFO)
    upscale = loaded.nodes["processing.upscale_image"].upscale_image_node
    load_context = context({"gpu_index": "0"}, "load")
    upscale_context = context({"gpu_index": "0"}, "upscale")
    image = np.random.default_rng(1).random((5, 7, 3), dtype=np.float32)
    first, second = tmp_path / "first.mxr", tmp_path / "second.mxr"
    first.write_bytes(b"first program")
    second.write_bytes(b"second program")
    program = module.load_program_node(load_context, first)[0]
    upscale(upscale_context, image, program, False)
    (old,) = Runner.made
    module.load_program_node(load_context, second)  # Load Program's preview
    assert old.closed  # no run can pass the first program again
    cache.release_node(load_context.node_id)
    cache.release_node(upscale_context.node_id)


def test_get_program_info(loaded: Imported):
    info = loaded.nodes["utility.get_program_info"].get_program_info_node
    assert info(MIGraphXProgram(b"", INFO)) == (
        2,
        "FP16",
        "gfx1100",
        "2.16.1",
        "256x256",
        0,
    )
    held = MIGraphXProgramInfo(**{**vars(INFO), "hip_graphs": 3})
    assert info(MIGraphXProgram(b"", held))[-1] == 3


@pytest.mark.parametrize("hidden", ["FP32", "FP16", "FP8"])
@pytest.mark.parametrize("recorded", ["fp16", "fp32"])
def test_a_record_compiles_in_its_precision(
    build: SimpleNamespace, hidden: str, recorded: str
):
    # FP8 is not refused either, on a GPU without it: the hidden input does not apply.
    model = OnnxGeneric(
        b"onnx model", OnnxInfo(17, recorded, input_channels=3, arch="SwinIR")
    )
    run(build, model, hidden)
    ((call,),) = [build.compiles.calls]
    assert call["fp16"] is (recorded == "fp16") and call["fp8"] is False


def test_precision_is_shown_until_an_onnx_is_known_to_have_a_record(
    build: SimpleNamespace,
):
    # tests/common/backend-nodes.test.ts evaluates this condition.
    module = build.imported.nodes["utility.build_program"]
    assert module.PRECISION_APPLIES.to_json() == {
        "kind": "type",
        "input": 0,
        "condition": 'OnnxModel { arch: "unrecorded" }',
        "ifNotConnected": True,
    }


def recorded_file(fp16: bool, metadata: dict[str, str]) -> bytes:
    elem_type = TensorProto.FLOAT16 if fp16 else TensorProto.FLOAT
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "plain",
        [helper.make_tensor_value_info("input", elem_type, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("output", elem_type, [1, 3, 4, 4])],
    )
    model = helper.make_model(graph)
    for key, value in metadata.items():
        model.metadata_props.add(key=key, value=value)
    return model.SerializeToString()


@pytest.mark.parametrize(
    ("file_fp16", "metadata", "chosen", "compiled_fp16"),
    [
        # This version's record of FP32, whatever a saved chain holds
        (
            False,
            {ARCH_METADATA_KEY: "SwinIR", PRECISION_METADATA_KEY: "fp32"},
            "FP16",
            False,
        ),
        (
            True,
            {ARCH_METADATA_KEY: "SwinIR", PRECISION_METADATA_KEY: "fp16"},
            "FP32",
            True,
        ),
        # A released version's record, the architecture alone
        (True, {ARCH_METADATA_KEY: "ESRGAN"}, "FP32", True),
        # No record: the dropdown's value stands
        (True, {}, "FP32", False),
        (False, {}, "FP16", True),
    ],
)
def test_a_reloaded_file_s_record_decides_the_precision(
    build: SimpleNamespace,
    file_fp16: bool,
    metadata: dict[str, str],
    chosen: str,
    compiled_fp16: bool,
):
    loaded = load_onnx_model(recorded_file(file_fp16, metadata))
    assert isinstance(loaded, OnnxGeneric)
    run(build, loaded, chosen)
    assert build.compiles.calls[0]["fp16"] is compiled_fp16


def test_an_onnx_without_a_record_keeps_the_chosen_precision(build: SimpleNamespace):
    run(build, onnx(), "FP32")
    assert build.compiles.calls[0]["fp16"] is False
