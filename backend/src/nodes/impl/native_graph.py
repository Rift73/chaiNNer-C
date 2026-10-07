"""Load the pinned native graph module without importing optional frameworks."""

from functools import lru_cache
from importlib import import_module
from types import ModuleType

from . import native_profile


@lru_cache(maxsize=1)
def graph() -> ModuleType:
    module = import_module("nodes.impl._chainner_graph")
    if module.abi_version != 1:
        raise RuntimeError("Unsupported chaiNNer graph ABI")
    if native_profile.enabled():
        native_profile.instrument_graph(module)
    return module
