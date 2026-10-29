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
import tempfile
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..tensorrt.dual import (
    ALIGNMENT,
    SOURCE,
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

# The ops' library (native/src/dual_hip, built with the ROCm SDK; Linux only: its kernels
# come from Triton, which has no Windows build for AMD GPUs)
OP_LIBRARY = Path(__file__).with_name("dual_migx_ops.so")
# What a program's lowering and kernels come from (build_identity, with OP_LIBRARY)
BUILD_SOURCES = (
    Path(__file__),
    Path(__file__).with_name("dual_hip.py"),
    Path(__file__).with_name("dual_lower_worker.py"),
    SOURCE / "nodes" / "impl" / "tensorrt" / "dual_dynamic_kernels.py",
    SOURCE / "nodes" / "impl" / "tensorrt" / "dual_na_kernel.py",
)

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
    """The SHA-256 of what a DUAL program's ops are built from: the kernels, the
    lowering, the library and the Triton that compiles the kernels."""
    digest = hashlib.sha256()
    for source in (*BUILD_SOURCES, OP_LIBRARY):
        # A build without the library fails (op_library), so no program is kept under
        # the identity of a missing one.
        if source.is_file():
            digest.update(source.read_bytes())
    for distribution in ("triton-rocm", "triton"):
        try:
            digest.update(importlib.metadata.version(distribution).encode())
            break
        except importlib.metadata.PackageNotFoundError:
            # The DUAL build reports a missing Triton itself.
            continue
    return digest.hexdigest()


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
    context: NodeContext | None = None,
) -> bytes:
    """A saved MIGraphX program of a DUAL ONNX for the AMD GPU of HIP device index
    `gpu_index`, in the ONNX's mixed precision (BF16 body, FP32 input and output), with
    DUAL's ops and their kernels compiled for that GPU. A fixed ONNX has its own size
    (`size` None); a dynamic one is built for one (height, width) `size`, which
    check_profile(manifest, size, size, size) accepts, and made one of that size first
    (dual_hip.fix_size). The program's input is named for the ONNX's alignment
    (aligned_input_name), as a TensorRT engine's."""
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
