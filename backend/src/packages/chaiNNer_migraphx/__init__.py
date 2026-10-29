from __future__ import annotations

import importlib.metadata
from collections.abc import Iterable

from sanic.log import logger

from api import Category, Dependency, add_package
from dependencies.rocm import (
    gpu_architectures,
    onnxruntime_migraphx,
    rocm_runtime,
    triton_rocm,
)
from dependencies.store import listed_version
from gpu import amd, nvidia
from nodes.impl.migraphx.runner import MODULE_FILE, load_migraphx, runner_built
from system import is_linux, is_windows

package_description = (
    "MIGraphX is AMD's graph compiler for inference on AMD GPUs. Its nodes compile ONNX"
    " models into programs optimized for your GPU, as the TensorRT nodes build engines for"
    " NVIDIA GPUs."
)

install_hint = (
    "MIGraphX compiles ONNX models for AMD GPUs. Requires Linux or Windows and an AMD GPU"
    " that ROCm 7.14 supports."
)

# The gfx targets of this computer's AMD GPUs that ROCm 7.14.1 has kernels for, and the
# others
supported, unsupported = gpu_architectures()
# The GPUs MIGraphX runs on: worth running models on and with ROCm's kernels
devices = [d for d in amd.devices if d.eligible and d.gfx in supported]


def disabled_reason() -> str | None:
    """Why the package cannot run here, from what is known without loading MIGraphX. A
    computer with an NVIDIA GPU keeps the NVIDIA stack alone, whatever else it has."""
    if not (is_linux or is_windows):
        return "MIGraphX requires Linux or Windows"
    if not amd.is_available:
        return "MIGraphX requires an AMD GPU with ROCm support"
    if not amd.eligible:
        return (
            "MIGraphX requires an AMD GPU with at least 8 compute units (a Radeon card"
            " or a large APU); this computer's are a processor's small graphics"
        )
    if nvidia.is_available:
        return (
            "MIGraphX runs only where no NVIDIA GPU is found: this computer's NVIDIA GPU"
            " runs TensorRT and the CUDA builds"
        )
    if not supported:
        # A GPU whose driver reports no gfx target is in neither list.
        found = f" (this computer's: {', '.join(unsupported)})" if unsupported else ""
        return f"MIGraphX requires an AMD GPU that ROCm 7.14 supports{found}"
    if not devices:
        return (
            "MIGraphX requires an AMD GPU that ROCm 7.14 supports with at least 8 compute"
            " units; this computer's supported ones are smaller"
        )
    if not runner_built():
        return f"This build of chaiNNer-C has no MIGraphX runner ({MODULE_FILE})"
    return None


def installed_as_listed(dependencies: Iterable[Dependency]) -> bool:
    """Whether every dependency is installed at its listed version. Only then is MIGraphX
    loaded to check it: a missing or other version is the dependency manager's to install,
    which a disabled package could not."""
    for dependency in dependencies:
        try:
            installed = importlib.metadata.version(dependency.pypi_name)
        except importlib.metadata.PackageNotFoundError:
            return False
        if listed_version(installed) != listed_version(dependency.version):
            return False
    return True


reason = disabled_reason()

# Registered everywhere (disabled with its reason); dependencies only where it can run, as
# ONNX Runtime's MIGraphX build replaces the onnxruntime module.
package = add_package(
    __file__,
    id="chaiNNer_migraphx",
    name="MIGraphX",
    description=package_description,
    dependencies=(
        []
        if reason is not None
        else [
            *rocm_runtime(supported),
            # MIGraphX's libraries ship inside it (nodes/impl/migraphx/runner.py).
            onnxruntime_migraphx(),
            # Compiles DUAL's kernels for its MIGraphX operators (Build Program, in a
            # child process). Linux only: Triton has no Windows build for AMD GPUs,
            # so Windows has no DUAL programs.
            *([triton_rocm()] if is_linux else []),
        ]
    ),
    icon="BsAmd",
    color="#ED1C24",
)

# The nodes exist once MIGraphX loads, as those of a package whose node modules import its
# framework exist once the dependency manager has installed it.
migraphx_loaded = False
if reason is None and installed_as_listed(package.dependencies):
    try:
        load_migraphx()
        migraphx_loaded = True
    except (ImportError, OSError) as e:
        logger.warning("MIGraphX did not load", exc_info=True)
        reason = f"MIGraphX did not load: {e}"

if reason is not None:
    package.disabled = True
    package.disabled_reason = reason

migraphx_category: Category | None = None
if not package.disabled:
    migraphx_category = package.add_category(
        name="MIGraphX",
        description="Nodes for compiling and running ONNX models with AMD MIGraphX.",
        icon="BsAmd",
        color="#ED1C24",
        install_hint=install_hint,
    )
    logger.debug("Loaded package %s", package.name)
