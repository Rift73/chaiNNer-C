"""SMoVSR's TensorRT engine sets: a set's key, and the set itself through the Engine
Cache (build_cache.cached_engine, one entry per set), as rsr.py gives RSR's. A set is
built in two child processes: the export (smovsr_export.py) and the build of its three
parts by RSR's worker (rsr.build: plain ONNX parts, no plugins, so no Triton); the parts
are then packed into one container (rdual_engine_set.py) whose info names the
architecture "SMoVSR". The set's key holds the weights rather than the exported graphs,
so a later run, after a restart too, reads the set without exporting.

This module must not import TensorRT at its top: the PyTorch package loads without it."""

from __future__ import annotations

import hashlib
import json
import tempfile
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import spandrel
import torch

from .dual import SOURCE, run_child
from .model import TemporalInfo, TensorRTEngine, TensorRTEngineInfo
from .rdual import BUILD_SETTINGS, write_weights
from .rdual_engine_set import pack
from .rsr import build as build_parts
from .rsr import timing_cache_file

if TYPE_CHECKING:
    from spandrel import ImageModelDescriptor

    from api import NodeContext

# The scales whose engines were checked against PyTorch FP32: 2 on the trained model
# (smovsr_session.py's module docstring).
CHECKED_SCALES = (2,)
_HERE = Path(__file__).parent
# What a set's export, build and container run
RECIPE_SOURCES = (
    _HERE / "smovsr.py",
    _HERE / "smovsr_export.py",
    # the worker, and the export's FP32 Resize pass and binding names
    _HERE / "rsr.py",
    _HERE / "rsr_engine_worker.py",
    _HERE / "rsr_export.py",
    # its save(), which writes the parts
    _HERE / "rdual_export.py",
    _HERE / "rdual_engine_set.py",
    _HERE / "dual_engine_worker.py",
)


@cache
def recipe_identity() -> str:
    """The SHA-256 of the code a set is exported and built with: these modules, the
    model's files (its graph and its per-frame steps), PyTorch and spandrel."""
    from spandrel.architectures import SMoVSR
    from spandrel.architectures.SMoVSR.__arch import smovsr_arch

    digest = hashlib.sha256(repr((torch.__version__, spandrel.__version__)).encode())
    for source in (
        *RECIPE_SOURCES,
        Path(smovsr_arch.__file__),
        Path(SMoVSR.__file__),
    ):
        digest.update(source.read_bytes())
    return digest.hexdigest()


def engine_key(
    model: ImageModelDescriptor, height: int, width: int, gpu_index: int
) -> str:
    """The key of a model's set for frames of height x width on a GPU: everything the
    set depends on (as build_cache.engine_key, with the weights for the graphs, and the
    hyperparameters, which hold the state bound the weights cannot show: h_clamp is set
    from the file's training settings after loading)."""
    import tensorrt as trt

    from ..pytorch.utils import weights_digest
    from .build_cache import FORMAT_VERSION
    from .memory import get_cuda_compute_capability, get_cuda_device_name

    digest = hashlib.sha256(weights_digest(model))
    digest.update(
        repr(
            (
                FORMAT_VERSION,
                ("smovsr", model.scale, height, width, BUILD_SETTINGS),
                json.dumps(model.model.hyperparameters, sort_keys=True),
                recipe_identity(),
                get_cuda_device_name(gpu_index),
                tuple(get_cuda_compute_capability(gpu_index)),
                trt.__version__,
            )
        ).encode()
    )
    return digest.hexdigest()


def export(
    model: ImageModelDescriptor,
    height: int,
    width: int,
    folder: Path,
    gpu_index: int | None,
    context: NodeContext | None = None,
) -> dict:
    """smovsr_export.py's parts of the model for frames of height x width in `folder`;
    returns their manifest. The parts are traced on the GPU `gpu_index` (the one the
    build uses), else on the CPU (the graphs are the same either way)."""
    on_gpu = gpu_index is not None and torch.cuda.is_available()
    device = f"cuda:{gpu_index}" if on_gpu else "cpu"
    with tempfile.TemporaryDirectory(prefix="chaiNNer-smovsr-weights-") as temporary:
        weights = write_weights(model.model, Path(temporary) / "weights.safetensors")
        run_child(
            [
                "-m",
                "nodes.impl.tensorrt.smovsr_export",
                str(weights),
                str(folder),
                str(height),
                str(width),
                device,
            ],
            "SMoVSR export",
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
    """Every part's engine of an export in `engines` (<part>.engine), by RSR's worker,
    which builds any export of plain ONNX parts (rsr.build)."""
    build_parts(folder, engines, gpu_index, context, settings, timing_cache, "SMoVSR")


def set_info(manifest: dict, gpu_index: int) -> TensorRTEngineInfo:
    """A built set's info, from its export's manifest."""
    import tensorrt as trt

    from .memory import get_cuda_compute_capability

    major, minor = get_cuda_compute_capability(gpu_index)
    height, width = int(manifest["height"]), int(manifest["width"])
    shape = (1, 3, height, width)
    return TensorRTEngineInfo(
        # FP16, with the images in FP32
        precision="fp16",
        input_channels=3,
        output_channels=3,
        scale=int(manifest["scale"]),
        gpu_architecture=f"sm_{major}{minor}",
        tensorrt_version=trt.__version__,
        has_dynamic_shapes=False,
        min_shape=shape,
        opt_shape=shape,
        max_shape=shape,
        # No plugins, so no plugin key
        temporal=TemporalInfo(
            architecture="SMoVSR",
            height=height,
            width=width,
            plugin_key="",
            parts=tuple(manifest["parts"]),
        ),
    )


def build_set(
    model: ImageModelDescriptor,
    height: int,
    width: int,
    gpu_index: int,
    context: NodeContext | None = None,
    timing_cache_directory: str | None = None,
) -> TensorRTEngine:
    """Export, build and pack the model's set for frames of height x width; with the
    Timing Cache folder, its build keeps its timing cache there."""
    with tempfile.TemporaryDirectory(prefix="chaiNNer-smovsr-engines-") as temporary:
        root = Path(temporary)
        manifest = export(model, height, width, root / "export", gpu_index, context)
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
) -> TensorRTEngine:
    """The model's set for frames of height x width: kept in memory, read from the
    Engine Cache `folder` (None: off), or built and kept in both. A build keeps its
    timing cache in `timing_cache_directory` (the Timing Cache setting; None: off)."""
    from .build_cache import cached_engine

    if folder is not None:
        Path(folder).mkdir(parents=True, exist_ok=True)
    return cached_engine(
        engine_key(model, height, width, gpu_index),
        lambda: build_set(
            model, height, width, gpu_index, context, timing_cache_directory
        ),
        folder,
    )


__all__ = [
    "CHECKED_SCALES",
    "build",
    "build_set",
    "engine_key",
    "engine_set",
    "export",
    "recipe_identity",
    "set_info",
]
