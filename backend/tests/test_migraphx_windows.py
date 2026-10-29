"""The MIGraphX runner on Windows, where MIGraphX's DLLs come from ONNX Runtime's Windows
MIGraphX build (onnxruntime/capi) and HIP and the ROCm libraries from the ROCm pip
packages: the MIGraphX package's loader imports the runner with those files, the DRFT and
VOSR op libraries register their ops through it, and without an AMD GPU a compile for the
GPU fails with an error and the process goes on. Skipped elsewhere and where the runner or
MIGraphX's libraries are missing; an op library that is not built is skipped."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import psutil
import pytest
from onnx import TensorProto, helper

from gpu import amd
from nodes.impl.migraphx import runner

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows' MIGraphX")

FOLDER = Path(runner.__file__).parent
OP_LIBRARIES = {
    "drft_migx_ops.dll": [
        "chainner_drft_window_neural_bias",
        "chainner_drft_ocab_relative_bias",
        "chainner_drft_ocab_window_pack",
        "chainner_drft_final_affine_crop",
        "chainner_drft_ocab_attention",
    ],
    "vosr2_migx_ops.dll": ["chainner_vosr2_qkv"],
}


@pytest.fixture(scope="module")
def migx() -> ModuleType:
    if not runner.runner_built():
        pytest.skip(f"this build has no {runner.MODULE_FILE}")
    try:
        return runner.load_migraphx()
    except ImportError as error:
        pytest.skip(str(error))


def relu_model() -> bytes:
    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"])],
        "relu",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])],
    )
    return helper.make_model(graph).SerializeToString()


def test_migraphx_and_hip_come_from_the_installed_packages(migx: ModuleType):
    assert migx.MIGRAPHX_VERSION.startswith("2.16.")
    loaded = {
        Path(m.path).name.lower(): Path(m.path) for m in psutil.Process().memory_maps()
    }
    capi = runner.migraphx_libraries()[0].parent
    core, libraries = runner.rocm_dll_directories()
    assert loaded["migraphx.dll"].parent == capi
    assert loaded["migraphx_onnx.dll"].parent == capi
    # The ROCm SDK's HIP, not a copy of the AMD driver's in System32
    assert loaded["amdhip64_7.dll"].parent == core
    assert loaded["_chainner_migraphx.pyd"].parent == FOLDER
    assert core.parent.name == "_rocm_sdk_core"
    assert libraries.parent.name == "_rocm_sdk_libraries"


@pytest.mark.parametrize("name", list(OP_LIBRARIES))
def test_the_op_libraries_register_their_ops(migx: ModuleType, name: str):
    library = FOLDER / name
    if not library.is_file():
        pytest.skip(f"this build has no {name}")
    assert migx.load_op_library(str(library)) == OP_LIBRARIES[name]
    # A second load is the first one.
    assert migx.load_op_library(str(library)) == OP_LIBRARIES[name]


def test_without_an_amd_gpu_a_gpu_compile_fails_and_the_process_goes_on(
    migx: ModuleType,
):
    if amd.devices:
        pytest.skip("this computer has an AMD GPU")
    with pytest.raises(RuntimeError, match=r"(?i)device"):
        migx.compile_onnx(relu_model(), device=0, target="gpu")
    assert migx.compile_onnx(relu_model(), target="ref").target == "ref"
