"""Temporal Fix's TensorRT engine sets: a set's key, and the set itself through the Engine
Cache (build_cache.cached_engine), as tfdat.py gives TFDAT's. The network has no state
between its windows; its set is two engines (temporal_fix_split.py): "frame", run once
per new frame, and "window", run per window on the frames' cached levels, built by
temporal_fix_worker.py (with tfix_aot's plugins where Triton is installed; the window is
then cut at its directly launched kernels, and the set holds its pieces, their layout
and the kernels' cubins instead: tfix_cut.py); its manifest and container are TFDAT's
(tfdat.set_info). The key holds the strength's weights rather than the
exported graph, so a later run, after a restart too, reads the set without exporting.

This module must not import TensorRT at its top: the PyTorch package loads without it."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import tempfile
from collections import OrderedDict
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import torch
from sanic.log import logger

from .dual import SOURCE, run_child
from .model import TensorRTEngine
from .rdual import BUILD_SETTINGS
from .rdual_engine_set import pack
from .rsr import timing_cache_file
from .tfdat import set_info

if TYPE_CHECKING:
    from api import NodeContext

_HERE = Path(__file__).parent
# What a set's export, build and container run
RECIPE_SOURCES = (
    _HERE / "temporal_fix.py",
    _HERE / "temporal_fix_export.py",
    _HERE / "temporal_fix_split.py",
    # the window part's plugins, where Triton is installed
    _HERE / "tfix_aot.py",
    _HERE / "tfix_round_trip_kernel.py",
    _HERE / "tfix_conf_input_kernel.py",
    _HERE / "tfix_tail_kernel.py",
    _HERE / "tfix_block_input_kernel.py",
    # the window's pieces and their directly launched kernels
    _HERE / "tfix_cut.py",
    _HERE / "dual_aot.py",
    _HERE / "tfdat.py",
    # the build's worker, and the binding rule and timing cache it shares with RSR's
    _HERE / "temporal_fix_worker.py",
    _HERE / "rsr.py",
    _HERE / "rsr_export.py",
    _HERE / "rsr_engine_worker.py",
    # its save(), which writes the part
    _HERE / "rdual_export.py",
    _HERE / "rdual_engine_set.py",
    _HERE / "dual_engine_worker.py",
    _HERE.parent / "pytorch" / "temporal_fix.py",
    # the split's parts, which the export traces
    _HERE.parent / "pytorch" / "temporal_fix_parts.py",
)


@cache
def _sources_identity() -> str:
    """The SHA-256 of these modules, the network's file (its graph) and PyTorch's
    version, read once per process."""
    from vendor.temporalfix import temporalfix_arch

    digest = hashlib.sha256(repr(torch.__version__).encode())
    for source in (*RECIPE_SOURCES, Path(temporalfix_arch.__file__)):
        digest.update(source.read_bytes())
    return digest.hexdigest()


def recipe_identity() -> str:
    """The SHA-256 of the code a set is exported and built with: _sources_identity and
    Triton (or its absence). With the plugins the parts are other engines, so their
    state is read again for every key: a set built without them is keyed as plain."""
    version = triton_version() if plugins() is not None else None
    state = repr(("plugins", version)).encode()
    return hashlib.sha256(_sources_identity().encode() + state).hexdigest()


def triton_version() -> str | None:
    """The installed Triton's version, under whichever distribution provides `triton`
    (triton-windows on Windows), or None."""
    for distribution in ("triton-windows", "triton"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


@cache
def _warn_broken(cause: str) -> None:
    """One warning per process (and cause) that a broken Triton leaves the plugins out."""
    logger.warning(
        "Temporal Fix: Triton does not import (%s); its engines are built without their "
        "plugins.",
        cause,
    )


def plugins() -> ModuleType | None:
    """tfix_aot, the parts' plugins, or None where Triton is not installed or does not
    import (a broken install: one warning with its cause); the parts are then built as
    exported. Any other failure to import the plugins is raised."""
    try:
        importlib.import_module("triton.language")
    except ModuleNotFoundError as error:
        if error.name is None or error.name.partition(".")[0] != "triton":
            raise
        return None
    except Exception as error:
        _warn_broken(f"{type(error).__name__}: {error}")
        return None
    try:
        from . import tfix_aot
    except ModuleNotFoundError as error:
        if error.name is None or error.name.partition(".")[0] != "triton":
            raise
        return None
    return tfix_aot


def engine_key(
    state: OrderedDict[str, torch.Tensor], height: int, width: int, gpu_index: int
) -> str:
    """The key of a strength's set for frames of height x width on a GPU: everything
    the set depends on (as build_cache.engine_key, with the weights for the graph)."""
    import tensorrt as trt

    from .build_cache import FORMAT_VERSION
    from .memory import get_cuda_compute_capability, get_cuda_device_name

    digest = hashlib.sha256()
    for name, value in state.items():
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        digest.update(repr((name, array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    digest.update(
        repr(
            (
                FORMAT_VERSION,
                ("TemporalFix", height, width, BUILD_SETTINGS),
                recipe_identity(),
                get_cuda_device_name(gpu_index),
                tuple(get_cuda_compute_capability(gpu_index)),
                trt.__version__,
            )
        ).encode()
    )
    return digest.hexdigest()


def export(
    state: OrderedDict[str, torch.Tensor],
    height: int,
    width: int,
    folder: Path,
    gpu_index: int | None,
    context: NodeContext | None = None,
) -> dict:
    """temporal_fix_split.py's parts of a strength's weights for frames of height x
    width in `folder`; returns its manifest. The part is traced on the GPU `gpu_index`
    (the one the build uses), else on the CPU (the graph is the same either way)."""
    from safetensors.torch import save_file

    on_gpu = gpu_index is not None and torch.cuda.is_available()
    device = f"cuda:{gpu_index}" if on_gpu else "cpu"
    with tempfile.TemporaryDirectory(prefix="chaiNNer-tfix-weights-") as temporary:
        weights = Path(temporary) / "weights.safetensors"
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in state.items()},
            str(weights),
        )
        run_child(
            [
                "-m",
                "nodes.impl.tensorrt.temporal_fix_split",
                str(weights),
                str(folder),
                str(height),
                str(width),
                device,
            ],
            "Temporal Fix export",
            SOURCE,
            context,
            cpu_only=not on_gpu,
        )
    return json.loads((folder / "export.json").read_text(encoding="utf-8"))


def build_set(
    state: OrderedDict[str, torch.Tensor],
    height: int,
    width: int,
    gpu_index: int,
    context: NodeContext | None = None,
    timing_cache_directory: str | None = None,
) -> TensorRTEngine:
    """Export, build and pack a strength's set for frames of height x width; with the
    Timing Cache folder, its build keeps its timing cache there."""
    with tempfile.TemporaryDirectory(prefix="chaiNNer-tfix-engines-") as temporary:
        root = Path(temporary)
        manifest = export(state, height, width, root / "export", gpu_index, context)
        if context is not None:
            context.check_aborted()
        timing_cache = None
        if timing_cache_directory is not None:
            Path(timing_cache_directory).mkdir(parents=True, exist_ok=True)
            timing_cache = timing_cache_file(
                timing_cache_directory, root / "export", gpu_index
            )
        build(root / "export", root / "engines", gpu_index, context, timing_cache)
        parts = set_parts(root / "engines", list(manifest["parts"]))
        info = set_info({**manifest, "parts": dict.fromkeys(parts)}, gpu_index)
    return pack(parts, info)


def build(
    folder: Path,
    engines: Path,
    gpu_index: int,
    context: NodeContext | None = None,
    timing_cache: str | None = None,
) -> None:
    """The engines of an export in `engines`, by Temporal Fix's own worker
    (temporal_fix_worker.py) with Build Engine's settings, as rsr.build runs RSR's;
    `timing_cache`, a file, is read before and written after the builds."""
    run_child(
        [
            "-m",
            "nodes.impl.tensorrt.temporal_fix_worker",
            str(folder),
            str(engines),
            str(gpu_index),
            *BUILD_SETTINGS,
            *(() if timing_cache is None else (timing_cache,)),
        ],
        "Temporal Fix engine build",
        SOURCE,
        context,
    )


def set_parts(engines: Path, exported: list[str]) -> dict[str, bytes]:
    """A built set's parts in order: each exported part's engine, or for a window cut
    into pieces (tfix_cut.py) the pieces' engines, then its layout and cubins."""
    from .tfix_cut import LAYOUT, cubin_name

    parts: dict[str, bytes] = {}
    layout = engines / LAYOUT
    for name in exported:
        if name != "window" or not layout.is_file():
            parts[name] = (engines / f"{name}.engine").read_bytes()
            continue
        plan = json.loads(layout.read_text(encoding="utf-8"))
        for piece in plan["pieces"]:
            parts[piece] = (engines / f"{piece}.engine").read_bytes()
        parts[LAYOUT] = layout.read_bytes()
        for kernel in plan["kernels"]:
            cubin = cubin_name(kernel["op"])
            parts[cubin] = (engines / cubin).read_bytes()
    return parts


def engine_set(
    state: OrderedDict[str, torch.Tensor],
    height: int,
    width: int,
    gpu_index: int,
    folder: str | None,
    context: NodeContext | None = None,
    timing_cache_directory: str | None = None,
) -> TensorRTEngine:
    """A strength's set for frames of height x width: kept in memory, read from the
    Engine Cache `folder` (None: off), or built and kept in both. A build keeps its
    timing cache in `timing_cache_directory` (the Timing Cache setting; None: off)."""
    from .build_cache import cached_engine

    if folder is not None:
        Path(folder).mkdir(parents=True, exist_ok=True)
    return cached_engine(
        engine_key(state, height, width, gpu_index),
        lambda: build_set(
            state, height, width, gpu_index, context, timing_cache_directory
        ),
        folder,
    )


__all__ = [
    "build",
    "build_set",
    "engine_key",
    "engine_set",
    "export",
    "plugins",
    "recipe_identity",
    "set_parts",
    "triton_version",
]
