"""Stand-ins for the MIGraphX tests (test_migraphx_*.py), which run without an AMD GPU or
MIGraphX: the runner's module _chainner_migraphx, its programs and runners, the AMD GPUs
(gpu.amd), the ROCm dependency lists (dependencies.rocm) and the DUAL and DRFT routes
(nodes/impl/migraphx/dual.py and drft.py)."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import numpy as np
import pytest

import api
import gpu
import nodes.impl.migraphx as migraphx_impl
import system
from api import Dependency, NodeContext, NodeId, SettingsParser
from gpu import AmdDevice, AmdInfo
from nodes.impl.migraphx import runner

PACKAGE = "packages.chaiNNer_migraphx"
NODE_MODULES = (
    "io.load_program",
    "io.save_program",
    "processing.upscale_image",
    "utility.build_program",
    "utility.get_program_info",
)
# A few of ROCm 7.14.1's gfx targets
ROCM_TARGETS = {"gfx942", "gfx1036", "gfx1100", "gfx1201"}
INSTALLED = Dependency("pytest", "pytest", importlib.metadata.version("pytest"), 0)
ABSENT = Dependency("Absent", "chainner-c-absent-test-dependency", "1.0", 0)
RADEON = "AMD Radeon RX 7900 XTX"


def amd_device(index: int, name: str, gfx: str, compute_units: int = 96) -> AmdDevice:
    """An AMD GPU, by default a discrete one (gfx94x, gfx950 and gfx12 do FP8)."""
    return AmdDevice(index, name, gfx, compute_units=compute_units)


class Program:
    """A loaded _chainner_migraphx.Program: one NCHW input and output."""

    def __init__(
        self,
        name: str = "input",
        shape: tuple[int, ...] = (1, 3, 64, 64),
        output: tuple[int, ...] = (1, 3, 128, 128),
        compute_type: str = "float16",
        input_type: str = "float32",
        gfx: str = "gfx1100",
    ) -> None:
        self.input_names = [name]
        self.input_shapes = [shape]
        self.output_shapes = [output]
        self.input_types = [input_type]
        self.compute_type = compute_type
        self.gfx = gfx
        self.closed = False

    def close(self) -> None:
        self.closed = True


class Runner:
    """A _chainner_migraphx.Runner that upscales each plane by pixel repetition."""

    made: ClassVar[list[Runner]] = []

    def __init__(self, program: bytes, device: int, streams: int) -> None:
        self.program, self.device, self.streams = program, device, streams
        self.input_channels = self.output_channels = 3
        self.scale = 2
        self.closed = False
        Runner.made.append(self)

    def upscale(self, inputs: list[np.ndarray], outputs: list[np.ndarray]) -> None:
        for plane, out in zip(inputs, outputs):
            out[...] = plane.repeat(self.scale, 0).repeat(self.scale, 1)

    def capture_failure(self) -> str:
        return ""

    def close(self) -> None:
        self.closed = True


def binding(tmp_path: Path) -> types.ModuleType:
    """A stand-in _chainner_migraphx with a module file of its own."""
    module = types.ModuleType(runner.MODULE)
    module.__spec__ = ModuleSpec(runner.MODULE, None)
    path = tmp_path / "_chainner_migraphx.so"
    path.write_bytes(b"runner module")
    module.__file__ = str(path)
    vars(module).update(MIGRAPHX_VERSION="2.16.1", Runner=Runner)
    return module


class Route(types.ModuleType):
    """The DUAL or DRFT route's functions Build Program and the runner call, recorded."""

    def __init__(self, name: str) -> None:
        super().__init__(f"nodes.impl.migraphx.{name}")
        self.profiles: list[tuple[Any, Any, Any]] = []
        self.builds: list[tuple[int, Any, bool]] = []
        self.loads = 0

    def check_profile(self, manifest: dict, low: Any, opt: Any, high: Any) -> None:
        self.profiles.append((low, opt, high))

    def build_program(
        self,
        onnx_bytes: bytes,
        manifest: dict,
        gpu_index: int,
        size: Any,
        *,
        exhaustive_tune: bool = False,
        context: Any = None,
    ) -> bytes:
        self.builds.append((gpu_index, size, exhaustive_tune))
        return b"route program"

    def build_identity(self) -> str:
        return f"{self.__name__} identity"

    def load_ops(self) -> None:
        self.loads += 1


def routes(monkeypatch: pytest.MonkeyPatch) -> tuple[Route, Route]:
    """Stand-in DUAL and DRFT routes in place of nodes/impl/migraphx/dual.py and drft.py."""
    dual, drft = Route("dual"), Route("drft")
    for route in (dual, drft):
        name = route.__name__.rsplit(".", 1)[1]
        monkeypatch.setitem(sys.modules, route.__name__, route)
        monkeypatch.setattr(migraphx_impl, name, route, raising=False)
    return dual, drft


def session_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """nodes.impl.migraphx.session on the stand-in routes (call routes first)."""
    module = importlib.import_module("nodes.impl.migraphx.session")
    monkeypatch.setattr(module, "dual", sys.modules["nodes.impl.migraphx.dual"])
    monkeypatch.setattr(module, "drft", sys.modules["nodes.impl.migraphx.drft"])
    return module


@dataclass
class Here:
    """The computer the package is imported on."""

    devices: list[AmdDevice] = field(
        default_factory=lambda: [amd_device(0, RADEON, "gfx1100")]
    )
    linux: bool = True
    windows: bool = False
    nvidia: bool = False
    runner_built: bool = True
    # rocm_runtime's list (onnxruntime_migraphx and triton_rocm are installed)
    dependencies: list[Dependency] = field(default_factory=lambda: [INSTALLED])
    load_error: Exception | None = None


@dataclass
class Imported:
    package: types.ModuleType
    nodes: dict[str, types.ModuleType]
    dual: Route
    drft: Route


def package_modules() -> dict[str, types.ModuleType]:
    return {
        name: module
        for name, module in sys.modules.items()
        if name == PACKAGE or name.startswith(f"{PACKAGE}.")
    }


def forget_package() -> None:
    """Drop the package's modules, so the next import runs them again."""
    for name in package_modules():
        del sys.modules[name]


@contextmanager
def package_imports() -> Iterator[None]:
    """The package's modules as they were before, after imports on stand-ins."""
    before = package_modules()
    try:
        yield
    finally:
        forget_package()
        sys.modules.update(before)


def import_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, here: Here
) -> Imported:
    """packages.chaiNNer_migraphx and its node modules, imported anew on `here` (inside
    package_imports)."""
    monkeypatch.setattr(gpu, "amd", AmdInfo(here.devices))
    monkeypatch.setattr(gpu, "nvidia", SimpleNamespace(is_available=here.nvidia))
    monkeypatch.setattr(system, "is_linux", here.linux)
    monkeypatch.setattr(system, "is_windows", here.windows)

    def architectures() -> tuple[list[str], list[str]]:
        found = {d.gfx for d in here.devices}
        return sorted(found & ROCM_TARGETS), sorted(found - ROCM_TARGETS)

    rocm = types.ModuleType("dependencies.rocm")
    vars(rocm).update(
        gpu_architectures=architectures,
        rocm_runtime=lambda _architectures: list(here.dependencies),
        onnxruntime_migraphx=lambda: INSTALLED,
        triton_rocm=lambda: INSTALLED,
    )
    monkeypatch.setitem(sys.modules, "dependencies.rocm", rocm)

    module = binding(tmp_path)

    def load_migraphx() -> types.ModuleType:
        if here.load_error is not None:
            raise here.load_error
        return module

    monkeypatch.setattr(runner, "runner_built", lambda: here.runner_built)
    monkeypatch.setattr(runner, "load_migraphx", load_migraphx)
    dual, drft = routes(monkeypatch)
    monkeypatch.setattr(session_module(monkeypatch), "native", load_migraphx)

    forget_package()
    monkeypatch.setattr(api.registry, "packages", dict(api.registry.packages))
    package = importlib.import_module(PACKAGE)
    nodes = {
        name: importlib.import_module(f"{PACKAGE}.migraphx.{name}")
        for name in NODE_MODULES
    }
    return Imported(package, nodes, dual, drft)


class _Context:
    """The NodeContext members the MIGraphX nodes use."""

    def __init__(self, settings: dict[str, object], node_id: str) -> None:
        self.settings = SettingsParser(settings)
        self.node_id = NodeId(node_id)

    def check_aborted(self) -> None:
        pass


def context(settings: dict[str, object], node_id: str = "node") -> NodeContext:
    """A NodeContext with the MIGraphX package's settings."""
    return cast(NodeContext, _Context(settings, node_id))
