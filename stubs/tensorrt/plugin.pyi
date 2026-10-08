# Minimal type stubs for the parts of tensorrt.plugin (TensorRT's Python plugin API)
# that chaiNNer uses (nodes/impl/tensorrt/dual_aot.py).

from typing import Any, Callable, TypeVar

from . import DataType

_F = TypeVar("_F", bound=Callable[..., Any])

class ShapeExpr:
    @property
    def is_constant(self) -> bool: ...
    def constant_value(self) -> int: ...

class ShapeExprs:
    def __getitem__(self, index: int) -> ShapeExpr: ...
    def numel(self) -> ShapeExpr: ...

class TensorDesc:
    shape_expr: ShapeExprs
    dtype: DataType
    def like(self) -> TensorDesc: ...

class KernelLaunchParams:
    grid_x: int
    grid_y: int
    grid_z: int
    block_x: int
    shared_mem: int

class SymInt32:
    def __init__(self, value: int) -> None: ...

class SymExprs: ...

class SymIntExprs(SymExprs):
    def __init__(self, count: int) -> None: ...
    def __setitem__(self, index: int, value: SymInt32) -> None: ...

def register(plugin_id: str, lazy_register: bool = False) -> Callable[[_F], _F]: ...
def aot_impl(plugin_id: str) -> Callable[[_F], _F]: ...
def from_shape_expr(
    shape_expr: tuple[ShapeExpr | int, ...], dtype: DataType
) -> TensorDesc: ...

op: Any
