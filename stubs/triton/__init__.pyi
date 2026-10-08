# Minimal type stubs for the parts of Triton that chaiNNer uses
# (nodes/impl/tensorrt/dual_aot.py, which compiles kernels ahead of time).

from typing import Any, Callable, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])

__version__: str

def jit(fn: _F) -> _F: ...
def compile(
    src: Any, target: Any = ..., options: dict[str, Any] | None = ...
) -> Any: ...
