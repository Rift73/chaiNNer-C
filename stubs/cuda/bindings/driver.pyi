# Minimal type stubs for the parts of cuda-python's driver API that chaiNNer uses
# (nodes/impl/tensorrt/tfix_cut.py). cuda.bindings.driver is a compiled extension without
# type information.

from enum import IntEnum
from typing import Any

class CUresult(IntEnum):
    CUDA_SUCCESS = 0

class CUmodule: ...
class CUfunction: ...

def cuModuleLoadData(image: bytes) -> tuple[CUresult, CUmodule]: ...
def cuModuleGetFunction(
    module: CUmodule, name: bytes
) -> tuple[CUresult, CUfunction]: ...
def cuModuleUnload(module: CUmodule) -> tuple[CUresult]: ...
def cuFuncGetParamInfo(
    function: CUfunction, index: int
) -> tuple[CUresult, int, int]: ...
def cuLaunchKernel(
    function: CUfunction,
    grid_x: int,
    grid_y: int,
    grid_z: int,
    block_x: int,
    block_y: int,
    block_z: int,
    shared: int,
    stream: int,
    params: tuple[tuple[int, ...], tuple[Any, ...]],
    extra: int,
) -> tuple[CUresult]: ...
