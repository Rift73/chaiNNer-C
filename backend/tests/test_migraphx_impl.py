"""MIGraphX's Python glue (nodes/impl/migraphx) without an AMD GPU or MIGraphX: loading the
runner's module after MIGraphX's libraries, a program's info, the compile in a child
process and its worker, the program cache and the runner session, on stand-ins
(migraphx_stand_ins.py)."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import os
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from migraphx_stand_ins import (
    RADEON,
    Program,
    Runner,
    binding,
    routes,
    session_module,
)

from nodes.impl.migraphx import build_cache, program, program_info, runner
from nodes.impl.migraphx.build_cache import SUFFIX, cached_program, program_key
from nodes.impl.migraphx.model import MIGraphXProgram, MIGraphXProgramInfo
from nodes.impl.onnx.model import OnnxGeneric, OnnxInfo

INFO = MIGraphXProgramInfo(
    precision="fp16",
    input_channels=3,
    output_channels=3,
    scale=2,
    gfx="gfx1100",
    migraphx_version="2.16.1",
    height=64,
    width=64,
)


@pytest.mark.parametrize(
    ("compute_type", "precision"),
    [
        ("float16", "fp16"),
        ("float32", "fp32"),
        ("bfloat16", "bf16"),
        ("fp8e4m3fnuz", "fp8"),
        ("fp8e4m3fn", "fp8"),
        ("mixed", "mixed"),
        # No convolution or matrix product: the input's type
        ("", "fp32"),
    ],
)
def test_a_programs_precision_is_its_compute_type(compute_type: str, precision: str):
    info = program_info.describe_program(Program(compute_type=compute_type), "2.16.1")
    assert info == MIGraphXProgramInfo(
        precision=precision,
        input_channels=3,
        output_channels=3,
        scale=2,
        gfx="gfx1100",
        migraphx_version="2.16.1",
        height=64,
        width=64,
    )


def test_a_dual_or_drft_program_is_mixed_precision():
    # Their input is named for their alignment, as TensorRT's engines'.
    loaded = Program(name="input_aligned_16", compute_type="bfloat16")
    assert program_info.describe_program(loaded, "2.16.1").precision == "mixed"


@pytest.mark.parametrize(
    ("loaded", "message"),
    [
        (Program(output=(1, 3, 128, 192)), "not a uniform integer upscale"),
        (Program(output=(1, 3, 96, 96)), "not a uniform integer upscale"),
        (Program(shape=(3, 64, 64), output=(3, 128, 128)), "Expected NCHW"),
    ],
)
def test_a_program_that_is_not_an_upscale_is_refused(loaded: Program, message: str):
    with pytest.raises(ValueError, match=message):
        program_info.describe_program(loaded, "2.16.1")


def test_a_program_with_two_inputs_is_refused():
    loaded = Program()
    loaded.input_names = ["input", "noise"]
    with pytest.raises(ValueError, match="one input and one output"):
        program_info.describe_program(loaded, "2.16.1")


def test_reading_a_programs_info_closes_the_program(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    module = binding(tmp_path)
    loads: list[tuple[bytes, int]] = []
    programs = [Program(), Program(output=(1, 3, 100, 100))]

    def load_program(data: bytes, device: int) -> Program:
        loads.append((data, device))
        return programs[len(loads) - 1]

    vars(module)["load_program"] = load_program
    monkeypatch.setattr(program_info, "native", lambda: module)
    assert program_info.read_program_info(b"saved", 1) == INFO
    assert loads == [(b"saved", 1)] and programs[0].closed
    with pytest.raises(ValueError):
        program_info.read_program_info(b"saved", 1)
    assert programs[1].closed


def onnxruntime_at(monkeypatch: pytest.MonkeyPatch, folder: Path | None) -> None:
    def find_spec(name: str) -> Any:
        assert name == "onnxruntime"
        if folder is None:
            return None
        return SimpleNamespace(submodule_search_locations=[str(folder)])

    monkeypatch.setattr(runner, "find_spec", find_spec)


def test_without_onnx_runtimes_migraphx_build_its_libraries_are_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    onnxruntime_at(monkeypatch, None)
    with pytest.raises(ImportError, match="onnxruntime-migraphx"):
        runner.migraphx_libraries()
    # Another ONNX Runtime build has no MIGraphX libraries.
    (tmp_path / "capi").mkdir()
    (tmp_path / "capi" / "libonnxruntime_providers_shared.so").write_bytes(b"")
    onnxruntime_at(monkeypatch, tmp_path)
    with pytest.raises(ImportError, match="has no MIGraphX libraries"):
        runner.migraphx_libraries()


@pytest.mark.parametrize(
    ("platform", "files", "loaded"),
    [
        (
            "linux",
            ("libmigraphx_gpu.so.2016001", "libmigraphx.so.2016001", "x.so"),
            ("libmigraphx.so.2016001", "libmigraphx_gpu.so.2016001"),
        ),
        (
            "win32",
            ("migraphx_gpu.dll", "migraphx.dll", "onnxruntime.dll", "migraphx-x.exe"),
            ("migraphx.dll", "migraphx_gpu.dll"),
        ),
    ],
)
def test_migraphx_loads_before_the_runners_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform: str,
    files: tuple[str, ...],
    loaded: tuple[str, ...],
):
    capi = tmp_path / "capi"
    capi.mkdir()
    for name in files:
        (capi / name).write_bytes(b"")
    folders = {
        "onnxruntime": tmp_path,
        "_rocm_sdk_core": tmp_path / "core",
        "_rocm_sdk_libraries": tmp_path / "libraries",
    }
    monkeypatch.setattr(
        runner,
        "find_spec",
        lambda name: SimpleNamespace(submodule_search_locations=[str(folders[name])]),
    )
    monkeypatch.setattr(runner.sys, "platform", platform)
    monkeypatch.setenv("PATH", "earlier")
    events: list[tuple[str, ...]] = []

    def cdll(path: str, mode: int) -> None:
        events.append(("load", Path(path).name, str(mode == ctypes.RTLD_LOCAL)))

    def import_module(name: str) -> str:
        events.append(("import", name))
        return "module"

    def add_dll_directory(path: str) -> None:
        events.append(("DLL folder", path))

    monkeypatch.setattr(runner.ctypes, "CDLL", cdll)
    monkeypatch.setattr(runner, "import_module", import_module)
    monkeypatch.setattr(
        runner.os, "add_dll_directory", add_dll_directory, raising=False
    )
    assert runner.load_migraphx.__wrapped__() == "module"
    # Windows: HIP and the ROCm libraries become findable first.
    folder_events = [
        ("DLL folder", str(tmp_path / "core" / "bin")),
        ("DLL folder", str(tmp_path / "libraries" / "bin")),
    ]
    assert events == [
        *(folder_events if platform == "win32" else []),
        *(("load", name, "True") for name in loaded),
        ("import", runner.MODULE),
    ]
    # Windows: MIGraphX's folder and ROCm's lead PATH for its child processes.
    path = [str(capi), *(str(tmp_path / f / "bin") for f in ("core", "libraries"))]
    expected = [*(path if platform == "win32" else []), "earlier"]
    assert os.environ["PATH"].split(os.pathsep) == expected


def test_on_windows_migraphx_needs_the_rocm_packages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    (tmp_path / "capi").mkdir()
    (tmp_path / "capi" / "migraphx.dll").write_bytes(b"")
    monkeypatch.setattr(
        runner,
        "find_spec",
        lambda name: (
            None
            if name == "_rocm_sdk_libraries"
            else SimpleNamespace(submodule_search_locations=[str(tmp_path)])
        ),
    )
    monkeypatch.setattr(runner.sys, "platform", "win32")
    # No folder is added before every package is found.
    monkeypatch.setattr(runner.os, "add_dll_directory", pytest.fail, raising=False)
    with pytest.raises(ImportError, match="ROCm's _rocm_sdk_libraries"):
        runner.load_migraphx.__wrapped__()


def test_programs_load_with_the_dual_and_drft_operators(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    dual, drft = routes(monkeypatch)
    session = session_module(monkeypatch)
    module = binding(tmp_path)
    monkeypatch.setattr(session, "load_migraphx", lambda: module)
    assert session.native.__wrapped__() is module
    assert dual.loads == drft.loads == 1


def test_the_runner_identity_names_its_module_and_migraphx(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    module = binding(tmp_path)
    monkeypatch.setattr(runner, "load_migraphx", lambda: module)
    versions = {"onnxruntime-migraphx": "1.30.0+rocm7.14"}
    monkeypatch.setattr(importlib.metadata, "version", versions.__getitem__)
    digest = hashlib.sha256(b"runner module").hexdigest()
    assert runner.runner_identity.__wrapped__() == f"2.16.1 {digest} 1.30.0+rocm7.14"


def test_a_compile_runs_the_worker_and_returns_its_program(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[list[str], str, Path, Any]] = []

    def run_child(arguments: list[str], label: str, cwd: Path, context: Any) -> str:
        calls.append((arguments, label, cwd, context))
        assert Path(arguments[2]).read_bytes() == b"onnx"
        Path(arguments[3]).write_bytes(b"saved program")
        return ""

    monkeypatch.setattr(program, "run_child", run_child)
    saved = program.compile_in_child(
        b"onnx",
        2,
        (96, 128),
        fp16=True,
        fp8=False,
        exhaustive_tune=True,
        op_libraries=[Path("ops.so")],
        label="DRFT program build",
        channels=1,
    )
    assert saved == b"saved program"
    ((arguments, label, cwd, context),) = calls
    assert arguments[:2] == ["-m", "nodes.impl.migraphx.program_worker"]
    assert arguments[4:] == ["2", "1x96x128", "1", "0", "1", "ops.so"]
    assert (label, context) == ("DRFT program build", None)
    assert cwd == Path(program.__file__).resolve().parents[3]


def test_the_worker_compiles_the_image_input_after_the_op_libraries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    from nodes.impl.migraphx import program_worker

    # The weight is a graph input too (older exporters list them), before the image.
    weight = helper.make_tensor("w", TensorProto.FLOAT, [3, 3, 1, 1], [1.0] * 9)
    graph = helper.make_graph(
        [helper.make_node("Conv", ["image", "w"], ["out"])],
        "g",
        [
            helper.make_tensor_value_info("w", TensorProto.FLOAT, [3, 3, 1, 1]),
            helper.make_tensor_value_info(
                "image", TensorProto.FLOAT, [1, 3, None, None]
            ),
        ],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3, None, None])],
        initializer=[weight],
    )
    path = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph), str(path))

    module = binding(tmp_path)
    events: list[Any] = []
    vars(module)["load_op_library"] = lambda library: events.append(("ops", library))

    class Compiled:
        def save(self, target: str) -> None:
            Path(target).write_bytes(b"compiled")

    def compile_onnx(model: str, **options: Any) -> Compiled:
        events.append(("compile", model, options))
        return Compiled()

    vars(module)["compile_onnx"] = compile_onnx
    monkeypatch.setattr(runner, "load_migraphx", lambda: module)
    out = tmp_path / "model.mxr"
    program_worker.main(str(path), str(out), "1", "3x64x96", "0", "1", "0", "a.so")
    ops, (step, model, options) = events
    assert ops == ("ops", "a.so") and (step, model) == ("compile", str(path))
    calibration = options.pop("calibration")
    assert options == {
        "device": 1,
        "fp16": False,
        "fp8": True,
        "input_dims": {"image": (1, 3, 64, 96)},
        "exhaustive_tune": False,
    }
    assert out.read_bytes() == b"compiled"
    # FP8's scales come from three images in [0, 1] of the input's size and type.
    assert len(calibration) == 3
    for inputs in calibration:
        (array,) = inputs.values()
        assert list(inputs) == ["image"]
        assert array.shape == (1, 3, 64, 96) and array.dtype == np.float32
        assert array.flags.c_contiguous
        assert 0 <= array.min() and array.max() <= 1


def test_without_fp8_there_is_no_calibration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    from nodes.impl.migraphx import program_worker

    graph = helper.make_graph(
        [helper.make_node("Identity", ["image"], ["out"])],
        "g",
        [
            helper.make_tensor_value_info(
                "image", TensorProto.FLOAT16, [1, 1, None, None]
            )
        ],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT16, [1, 1, None, None])],
    )
    path = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph), str(path))
    module = binding(tmp_path)
    compiled: list[dict[str, Any]] = []

    class Compiled:
        def save(self, target: str) -> None:
            Path(target).write_bytes(b"")

    def compile_onnx(model: str, **options: Any) -> Compiled:
        compiled.append(options)
        return Compiled()

    vars(module)["compile_onnx"] = compile_onnx
    monkeypatch.setattr(runner, "load_migraphx", lambda: module)
    program_worker.main(str(path), str(tmp_path / "p.mxr"), "0", "1x8x8", "1", "0", "0")
    assert compiled[0]["calibration"] == ()
    assert compiled[0]["input_dims"] == {"image": (1, 1, 8, 8)}
    fp16 = program_worker.calibration("image", np.dtype(np.float16), (1, 1, 8, 8))
    assert all(inputs["image"].dtype == np.float16 for inputs in fp16)


def test_the_runner_session_upscales_bgr_and_gray_images(
    monkeypatch: pytest.MonkeyPatch,
):
    routes(monkeypatch)
    sessions = session_module(monkeypatch)
    monkeypatch.setattr(sessions, "native", lambda: SimpleNamespace(Runner=Runner))
    session = sessions.RunnerSession(MIGraphXProgram(b"program", INFO), 3)
    assert (session.runner.device, session.runner.streams) == (3, 1)
    bgr = np.random.default_rng(0).random((4, 6, 3), dtype=np.float32)
    assert np.array_equal(session.upscale(bgr), bgr.repeat(2, 0).repeat(2, 1))
    gray = bgr[:, :, 0].copy()
    assert np.array_equal(session.upscale(gray), gray.repeat(2, 0).repeat(2, 1))
    session.close()
    assert session.runner.closed


@pytest.fixture
def memory(monkeypatch: pytest.MonkeyPatch) -> OrderedDict[str, MIGraphXProgram]:
    """The programs kept in memory, empty at the start."""
    kept: OrderedDict[str, MIGraphXProgram] = OrderedDict()
    monkeypatch.setattr(build_cache, "_memory", kept)
    monkeypatch.setattr(build_cache, "_building", {})
    return kept


def model(data: bytes = b"onnx model") -> OnnxGeneric:
    return OnnxGeneric(data, OnnxInfo(17, "fp32"))


GPU = (RADEON, "gfx1100", "runner", None)


def test_each_key_input_changes_the_key():
    base = program_key(model(), ("onnx", True), *GPU)
    assert program_key(model(), ("onnx", True), *GPU) == base
    others = [
        program_key(model(b"other"), ("onnx", True), *GPU),
        program_key(model(), ("onnx", False), *GPU),
        program_key(model(), ("onnx", True), "AMD Instinct MI300X", *GPU[1:]),
        program_key(model(), ("onnx", True), RADEON, "gfx1101", *GPU[2:]),
        program_key(model(), ("onnx", True), *GPU[:2], "other runner", None),
        program_key(model(), ("onnx", True), *GPU[:3], "dual identity"),
    ]
    assert len({base, *others}) == 7


class Builds:
    def __init__(self) -> None:
        self.runs = 0

    def __call__(self) -> MIGraphXProgram:
        self.runs += 1
        return MIGraphXProgram(bytes([self.runs]) * 64, INFO)


def test_a_build_of_the_same_inputs_runs_once_and_is_kept_on_disk(
    tmp_path: Path, memory: OrderedDict[str, MIGraphXProgram]
):
    key = program_key(model(), ("onnx",), *GPU)
    build = Builds()
    first = cached_program(key, build, str(tmp_path))
    assert cached_program(key, build, str(tmp_path)) is first
    assert build.runs == 1
    memory.clear()  # a new process
    again = cached_program(key, build, str(tmp_path))
    assert (again.bytes, again.info) == (first.bytes, first.info)
    assert build.runs == 1
    assert [p.name for p in tmp_path.iterdir()] == [f"{key}{SUFFIX}"]


def test_a_damaged_file_is_removed_and_built_again(
    tmp_path: Path, memory: OrderedDict[str, MIGraphXProgram]
):
    key = program_key(model(), ("onnx",), *GPU)
    build = Builds()
    cached_program(key, build, str(tmp_path))
    path = tmp_path / f"{key}{SUFFIX}"
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    memory.clear()
    assert cached_program(key, build, str(tmp_path)).bytes == bytes([2]) * 64
    assert build.runs == 2


def test_at_most_four_programs_are_kept_in_memory(
    memory: OrderedDict[str, MIGraphXProgram],
):
    build = Builds()
    keys = [program_key(model(bytes([i])), ("onnx",), *GPU) for i in range(5)]
    for key in keys:
        cached_program(key, build, None)
    assert list(memory) == keys[1:]
