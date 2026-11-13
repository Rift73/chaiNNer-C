"""DUAL's MIGraphX path, the counterpart of the TensorRT one (tensorrt/dual.py) on AMD
GPUs: the same exporter's ONNX (Convert To ONNX), whose plugin nodes a child process
lowers to `chainner`-domain nodes of DUAL's custom ops, each carrying its Triton kernel
compiled for the GPU (dual_lower_worker.py, dual_hip.py), then compiled into a program
of one size in another child (program.compile_in_child) with the ops' library
(dual_migx_ops, native/src/dual_hip) loaded. As with TensorRT, the chaiNNer worker never
imports Triton: a program carries its kernels, and every process that loads programs
loads the library once first (load_ops)."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import sys
import tempfile
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sanic.log import logger

from ..tensorrt.dual import (
    ALIGNMENT,
    SOURCE,
    VENDOR,
    aligned_input_name,
    check_profile,
    export_manifest,
    export_onnx,
    factory_of,
    input_alignment,
    run_child,
)
from . import program, runner

if TYPE_CHECKING:
    from api import NodeContext

# The ops' library (native/src/dual_hip, built with the ROCm SDK). Its kernels come from
# Triton: triton-rocm on Linux, on Windows triton-windows 3.8.0.post29 or newer (3.5.1
# fails on its temporary files).
OP_LIBRARY = Path(__file__).with_name(
    "dual_migx_ops.dll" if sys.platform == "win32" else "dual_migx_ops.so"
)
# What a program's lowering and kernels come from (build_identity, with OP_LIBRARY)
BUILD_SOURCES = (
    Path(__file__),
    Path(__file__).with_name("dual_hip.py"),
    Path(__file__).with_name("dual_lower_worker.py"),
    SOURCE / "nodes" / "impl" / "tensorrt" / "dual_dynamic_kernels.py",
    SOURCE / "nodes" / "impl" / "tensorrt" / "dual_na_kernel.py",
)
# The vendored exporter, which writes the graph a program is lowered from (build_identity):
# the automatic route keys its programs by the weights, not by the ONNX, so a program of
# an earlier exporter's graph is not read for a later one.
EXPORTER_SOURCES = tuple(sorted(VENDOR.rglob("*.py")))
# The manifest keys of the exporter's local and retrieval sites' operators (vendor/
# dual_tensorrt/scripts/dual_tensorrt/spec.py): TensorRT plugins only, for which the
# library has no op (dual_hip.TENSORRT_ONLY). Manifests without them are earlier exports
# (local windows, 96 px retrieval blocks), which lower as before.
TENSORRT_ONLY_KEYS = ("local", "retrieval")

Size = tuple[int, int]


def gfx_of(gpu_index: int) -> str:
    """The gfx name of the AMD GPU of HIP device index `gpu_index`."""
    from gpu import amd

    for device in amd.devices:
        if device.index == gpu_index:
            if not device.gfx:
                raise ValueError(
                    f"The driver reports no gfx target for {device.name}, which DUAL's"
                    " kernels are compiled for."
                )
            return device.gfx
    raise ValueError(f"There is no AMD GPU {gpu_index} to build a DUAL program for.")


@cache
def load_ops() -> None:
    """Load the ops' library into this process's MIGraphX, once, before any program is
    loaded. A build without the library loads nothing: DUAL programs then fail to load
    with a message naming the missing op, other programs are unaffected."""
    if OP_LIBRARY.is_file():
        runner.load_migraphx().load_op_library(str(OP_LIBRARY))


@cache
def build_identity() -> str:
    """The SHA-256 of what a DUAL program is built from: the kernels, the lowering, the
    exporter of the graph it lowers, the library and the Triton that compiles the
    kernels."""
    digest = hashlib.sha256()
    for source in (*BUILD_SOURCES, *EXPORTER_SOURCES, OP_LIBRARY):
        # A build without the library fails (op_library), so no program is kept under
        # the identity of a missing one.
        if source.is_file():
            digest.update(source.read_bytes())
    for distribution in ("triton-rocm", "triton-windows", "triton"):
        try:
            digest.update(importlib.metadata.version(distribution).encode())
            break
        except importlib.metadata.PackageNotFoundError:
            # The DUAL build reports a missing Triton itself.
            continue
    return digest.hexdigest()


def refusal(keys: list[str]) -> str:
    """Why a DUAL ONNX with these TENSORRT_ONLY_KEYS has no program."""
    return (
        f"This DUAL ONNX's {' and '.join(keys)} neighborhood attention runs only as"
        " TensorRT plugins, which MIGraphX has no operators for, so it has no MIGraphX"
        " program. Run the model with PyTorch's Upscale Image."
    )


@cache
def exported_tensorrt_only_keys() -> list[str]:
    """The TENSORRT_ONLY_KEYS every manifest of the vendored exporter holds (its spec's
    LOCAL and RETRIEVAL, which export.py and dynamic.py write whatever the model), so the
    automatic route knows before exporting that build_program would refuse the ONNX."""
    path = VENDOR / "scripts" / "dual_tensorrt" / "spec.py"
    spec = importlib.util.spec_from_file_location("dual_tensorrt_spec", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [key for key in TENSORRT_ONLY_KEYS if hasattr(module, key.upper())]


def op_library() -> Path:
    """The ops' library a program build loads, or an error that says how to get it."""
    if not OP_LIBRARY.is_file():
        raise RuntimeError(
            f"DUAL's MIGraphX op library {OP_LIBRARY} is missing: this build was made"
            " without it. Use the PyTorch backend, or build chaiNNer-C's native layer"
            " with the ROCm SDK (native/BUILDING.md)."
        )
    return OP_LIBRARY


def build_program(
    onnx_bytes: bytes,
    manifest: dict[str, Any],
    gpu_index: int,
    size: Size | None,
    *,
    exhaustive_tune: bool = False,
    hip_graph: bool = False,
    context: NodeContext | None = None,
) -> bytes:
    """A saved MIGraphX program of a DUAL ONNX for the AMD GPU of HIP device index
    `gpu_index`, in the ONNX's mixed precision (BF16 body, FP32 input and output), with
    DUAL's ops and their kernels compiled for that GPU. A fixed ONNX has its own size
    (`size` None); a dynamic one is built for one (height, width) `size`, which
    check_profile(manifest, size, size, size) accepts, and made one of that size first
    (dual_hip.fix_size). The program's input is named for the ONNX's alignment
    (aligned_input_name), as a TensorRT engine's. An ONNX with local or retrieval
    neighborhood attention (TENSORRT_ONLY_KEYS) has no program."""
    tensorrt_only = [key for key in TENSORRT_ONLY_KEYS if key in manifest]
    if tensorrt_only:
        raise ValueError(refusal(tensorrt_only))
    # Exported before local and retrieval neighborhood attention: it still builds, as
    # the file holds it.
    logger.warning(
        "This DUAL ONNX was exported with the old local windows and retrieval blocks;"
        " its program keeps them, while PyTorch's Upscale Image runs the model with"
        " neighborhood attention there. Converting the model to ONNX again gives an"
        " ONNX that only TensorRT builds."
    )
    spec = manifest["specialization"]
    if spec.get("dynamic", False) != (size is not None):
        raise ValueError("Only a dynamic DUAL ONNX takes a size.")
    library = op_library()
    with tempfile.TemporaryDirectory(prefix="chaiNNer-dual-program-") as temporary:
        model = Path(temporary) / "model.onnx"
        model.write_bytes(onnx_bytes)
        lowered = Path(temporary) / "lowered.onnx"
        run_child(
            [
                "-m",
                "nodes.impl.migraphx.dual_lower_worker",
                str(model),
                str(lowered),
                gfx_of(gpu_index),
                *(str(value) for value in size or ()),
            ],
            "DUAL kernel build",
            SOURCE,
            context,
            cpu_only=True,
        )
        lowered_bytes = lowered.read_bytes()
    return program.compile_in_child(
        lowered_bytes,
        gpu_index,
        size or (spec["height"], spec["width"]),
        fp16=False,
        fp8=False,
        exhaustive_tune=exhaustive_tune,
        hip_graph=hip_graph,
        op_libraries=(library,),
        label="DUAL program build",
        context=context,
    )


__all__ = [
    "ALIGNMENT",
    "aligned_input_name",
    "build_identity",
    "build_program",
    "check_profile",
    "export_manifest",
    "export_onnx",
    "factory_of",
    "input_alignment",
    "load_ops",
    "op_library",
]
