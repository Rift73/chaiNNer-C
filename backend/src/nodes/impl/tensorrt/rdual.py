"""RDUAL's TensorRT engine sets: whether a model may have one here, its key, and the
set itself through the Engine Cache (build_cache.cached_engine, one entry per set). A
set is built in two child processes as dual.py builds DUAL's engines: the export
(rdual_export.py) and the build of every part with its plugins (rdual_engine_worker.py,
which needs Triton); the parts are then packed into one container (rdual_engine_set.py).
The set's key holds the weights rather than the exported graphs, so a later run, after a
restart too, reads the set without exporting.

This module must not import TensorRT at its top: the PyTorch package loads without it."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import tempfile
from functools import cache
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

import spandrel
import torch
from safetensors.torch import save_file
from sanic.log import logger

from .dual import SOURCE, run_child
from .model import TemporalInfo, TensorRTEngine, TensorRTEngineInfo
from .rdual_engine_set import pack
from .rdual_recipe import Size, flow_precision

if TYPE_CHECKING:
    from spandrel import ImageModelDescriptor

    from api import NodeContext

# The model's hyperparameters in the weights file the export reads (rdual_export.py)
HYPERPARAMETERS_KEY = "chainner_c.rdual.hyperparameters"
# The scales whose engines were checked against PyTorch
CHECKED_SCALES = (2,)
# Build Engine's guide settings: workspace GiB, optimization level, TF32, sparse
# weights, tiling optimization (dual_engine_worker.builder_config)
BUILD_SETTINGS = ("8", "3", "0", "0", "none")
# The manifest keys of the local and retrieval sites' operators (rdual_export.main); a
# manifest without them is an earlier export, whose context sites already ran
# neighborhood attention
NEIGHBORHOOD_KEYS = ("local", "retrieval")
_HERE = Path(__file__).parent
# What a set's export, build and container run, besides DUAL's plugins
_RECIPE_SOURCES = (
    _HERE / "rdual.py",
    _HERE / "rdual_aot.py",
    _HERE / "rdual_export.py",
    _HERE / "rdual_engine_worker.py",
    _HERE / "rdual_engine_set.py",
    _HERE / "dual_engine_worker.py",
)


def unavailable(model: ImageModelDescriptor, device: torch.device) -> str | None:
    """Why this model has no engine set on this machine, or None if it may ("" for a
    machine without NVIDIA's TensorRT, which is no surprise worth a warning)."""
    from ..pytorch.auto_engine import migraphx_here, no_engine_here

    if model.architecture.id != "RDUAL":
        return f"{model.architecture.name} is not a temporal model."
    if model.scale not in CHECKED_SCALES:
        return f"RDUAL's engines are checked for {CHECKED_SCALES} scale only."
    reason = no_engine_here(device)
    if reason is not None:
        return reason
    if migraphx_here():
        return ""
    return None


@cache
def recipe_identity() -> str:
    """The SHA-256 of the code a set is exported and built with: these modules, DUAL's
    plugins, the model's file (its kernels and export names), Triton, PyTorch and
    spandrel."""
    from spandrel.architectures.RDUAL.__arch import rdual_arch

    from .build_cache import dual_plugin_identity

    digest = hashlib.sha256(
        repr((torch.__version__, spandrel.__version__, dual_plugin_identity())).encode()
    )
    for source in (*_RECIPE_SOURCES, Path(rdual_arch.__file__)):
        digest.update(source.read_bytes())
    try:
        digest.update(importlib.metadata.version("triton").encode())
    except importlib.metadata.PackageNotFoundError:
        # A build without Triton fails (engine_set says why), so none is kept.
        pass
    return digest.hexdigest()


def flow_recipe(height: int, width: int, flow: str | None = None) -> str:
    """The name of the flow precision recipe a set of height x width gets
    (rdual_export.flow_precision; `flow` names one explicitly)."""
    return flow_precision(Size(height, width), flow).name


def engine_key(
    model: ImageModelDescriptor,
    height: int,
    width: int,
    gpu_index: int,
    flow: str | None = None,
) -> str:
    """The key of a model's set for frames of height x width on a GPU: everything the
    set depends on (as build_cache.engine_key, with the weights for the graphs),
    the flow's precision recipe included."""
    import tensorrt as trt

    from ..pytorch.auto_engine import weights_digest
    from .build_cache import FORMAT_VERSION
    from .memory import get_cuda_compute_capability, get_cuda_device_name

    digest = hashlib.sha256(weights_digest(model))
    digest.update(
        repr(
            (
                FORMAT_VERSION,
                (
                    "rdual",
                    model.scale,
                    height,
                    width,
                    BUILD_SETTINGS,
                    flow_recipe(height, width, flow),
                ),
                recipe_identity(),
                get_cuda_device_name(gpu_index),
                tuple(get_cuda_compute_capability(gpu_index)),
                trt.__version__,
            )
        ).encode()
    )
    return digest.hexdigest()


def write_weights(model: torch.nn.Module, path: Path) -> Path:
    """The weights file the export reads: the folded model's state dict, with its
    hyperparameters in the metadata."""
    hyperparameters = getattr(model, "hyperparameters", None)
    if not isinstance(hyperparameters, dict):
        raise ValueError("An RDUAL model records its hyperparameters")
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        str(path),
        metadata={HYPERPARAMETERS_KEY: json.dumps(hyperparameters)},
    )
    return path


def export(
    model: ImageModelDescriptor,
    height: int,
    width: int,
    folder: Path,
    gpu_index: int | None,
    context: NodeContext | None = None,
    flow: str | None = None,
) -> dict:
    """rdual_export.py's parts of the model for frames of height x width in `folder`;
    returns their manifest. The flow part is traced on the GPU `gpu_index` (the one the
    build uses), else on the CPU, which takes minutes at video sizes (the graphs are the
    same either way)."""
    on_gpu = gpu_index is not None and torch.cuda.is_available()
    device = f"cuda:{gpu_index}" if on_gpu else "cpu"
    with tempfile.TemporaryDirectory(prefix="chaiNNer-rdual-weights-") as temporary:
        weights = write_weights(model.model, Path(temporary) / "weights.safetensors")
        run_child(
            [
                "-m",
                "nodes.impl.tensorrt.rdual_export",
                str(weights),
                str(folder),
                str(height),
                str(width),
                device,
                flow or "",
            ],
            "RDUAL export",
            SOURCE,
            context,
            cpu_only=not on_gpu,
        )
    return json.loads((folder / "export.json").read_text(encoding="utf-8"))


def build(
    folder: Path,
    engines: Path,
    gpu_index: int,
    context: NodeContext | None = None,
    settings: tuple[str, ...] = BUILD_SETTINGS,
    timing_cache: str | None = None,
) -> None:
    """Every part's engine of an export in `engines` (<part>.engine), with Build
    Engine's `settings` (BUILD_SETTINGS' order); `timing_cache`, a file, is read before
    and written after the builds (see timing_cache_file)."""
    if find_spec("triton") is None:
        raise RuntimeError(
            "RDUAL's engines need Triton to compile their plugins (the TensorRT"
            " package installs it)."
        )
    manifest = json.loads((folder / "export.json").read_text(encoding="utf-8"))
    missing = [key for key in NEIGHBORHOOD_KEYS if key not in manifest]
    if missing:
        # Exported before local and retrieval neighborhood attention (an ONNX bundle of
        # an earlier version): it still builds, as the bundle holds it.
        logger.warning(
            "This RDUAL ONNX bundle was exported without neighborhood attention at its"
            f" {' and '.join(missing)} sites; its engines keep their earlier windows."
            " Convert the model to ONNX again for neighborhood attention."
        )
    run_child(
        [
            "-m",
            "nodes.impl.tensorrt.rdual_engine_worker",
            str(folder),
            str(engines),
            str(gpu_index),
            *settings,
            *(() if timing_cache is None else (timing_cache,)),
        ],
        "RDUAL engine build",
        SOURCE,
        context,
    )


def timing_cache_file(directory: str, folder: Path, gpu_index: int) -> str:
    """The timing cache an export's build reads and writes in the TensorRT package's
    Timing Cache folder: Build Engine's name (engine_builder.timing_cache_name) of its
    first part, so one file per model and input size, GPU and TensorRT version."""
    import tensorrt as trt

    from .engine_builder import timing_cache_name
    from .memory import get_cuda_compute_capability, get_cuda_device_name

    manifest = json.loads((folder / "export.json").read_text(encoding="utf-8"))
    first = next(iter(manifest["parts"]))
    name = timing_cache_name(
        (folder / first / "model.onnx").read_bytes(),
        get_cuda_device_name(gpu_index),
        get_cuda_compute_capability(gpu_index),
        trt.__version__,
    )
    return str(
        Path(directory) / f"rdual_{manifest['width']}x{manifest['height']}_{name}"
    )


def set_info(manifest: dict, gpu_index: int) -> TensorRTEngineInfo:
    """A built set's info, from its export's manifest."""
    import tensorrt as trt

    from .memory import get_cuda_compute_capability

    major, minor = get_cuda_compute_capability(gpu_index)
    height, width = int(manifest["height"]), int(manifest["width"])
    shape = (1, 3, height, width)
    return TensorRTEngineInfo(
        precision="mixed",
        input_channels=3,
        output_channels=3,
        scale=int(manifest["scale"]),
        gpu_architecture=f"sm_{major}{minor}",
        tensorrt_version=trt.__version__,
        has_dynamic_shapes=False,
        min_shape=shape,
        opt_shape=shape,
        max_shape=shape,
        temporal=TemporalInfo(
            architecture="RDUAL",
            height=height,
            width=width,
            plugin_key=str(manifest["plugin_key"]),
            parts=tuple(manifest["parts"]),
            flow_precision=str(manifest["flow_precision"]),
        ),
    )


def build_set(
    model: ImageModelDescriptor,
    height: int,
    width: int,
    gpu_index: int,
    context: NodeContext | None = None,
    timing_cache_directory: str | None = None,
    flow: str | None = None,
) -> TensorRTEngine:
    """Export, build and pack the model's set for frames of height x width; with the
    Timing Cache folder, its build keeps its timing cache there. `flow` names the flow's
    precision recipe; None lets the size choose it."""
    with tempfile.TemporaryDirectory(prefix="chaiNNer-rdual-engines-") as temporary:
        root = Path(temporary)
        manifest = export(
            model, height, width, root / "export", gpu_index, context, flow
        )
        if context is not None:
            context.check_aborted()
        timing_cache = None
        if timing_cache_directory is not None:
            Path(timing_cache_directory).mkdir(parents=True, exist_ok=True)
            timing_cache = timing_cache_file(
                timing_cache_directory, root / "export", gpu_index
            )
        build(
            root / "export",
            root / "engines",
            gpu_index,
            context,
            timing_cache=timing_cache,
        )
        info = set_info(manifest, gpu_index)
        assert info.temporal is not None
        parts = {
            name: (root / "engines" / f"{name}.engine").read_bytes()
            for name in info.temporal.parts
        }
    return pack(parts, info)


def engine_set(
    model: ImageModelDescriptor,
    height: int,
    width: int,
    gpu_index: int,
    folder: str | None,
    context: NodeContext | None = None,
    timing_cache_directory: str | None = None,
    flow: str | None = None,
) -> TensorRTEngine:
    """The model's set for frames of height x width: kept in memory, read from the
    Engine Cache `folder` (None: off), or built and kept in both. A build keeps its
    timing cache in `timing_cache_directory` (the Timing Cache setting; None: off)."""
    from .build_cache import cached_engine

    if folder is not None:
        Path(folder).mkdir(parents=True, exist_ok=True)
    return cached_engine(
        engine_key(model, height, width, gpu_index, flow),
        lambda: build_set(
            model, height, width, gpu_index, context, timing_cache_directory, flow
        ),
        folder,
    )


__all__ = [
    "CHECKED_SCALES",
    "HYPERPARAMETERS_KEY",
    "build",
    "build_set",
    "engine_key",
    "engine_set",
    "export",
    "flow_recipe",
    "recipe_identity",
    "set_info",
    "timing_cache_file",
    "unavailable",
    "write_weights",
]
