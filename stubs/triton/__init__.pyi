# Minimal type stubs for the parts of Triton that chaiNNer uses
# (nodes/impl/tensorrt/dual_aot.py, which compiles kernels ahead of time).

from typing import Any, Callable, Generic, ParamSpec, TypeVar

_T = TypeVar("_T")
_P = ParamSpec("_P")
_R = TypeVar("_R")

__version__: str

class KernelInterface(Generic[_T]):
    run: _T
    def __getitem__(self, grid: Any) -> _T: ...

class JITFunction(KernelInterface[_T]):
    fn: _T
    def __call__(
        self: JITFunction[Callable[_P, _R]], *args: _P.args, **kwargs: _P.kwargs
    ) -> _R: ...

def jit(fn: _T) -> JITFunction[_T]: ...
def cdiv(x: int, y: int) -> int: ...
def compile(
    src: Any, target: Any = ..., options: dict[str, Any] | None = ...
) -> Any: ...
