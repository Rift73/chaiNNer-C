"""RSR's TensorRT engine sets: a set's key, and the set itself through the Engine Cache
(build_cache.cached_engine, one entry per set), as rdual.py gives RDUAL's. A set is
built in two child processes: the export (rsr_export.py) and the build of its three
parts (rsr_engine_worker.py, no plugins, so no Triton); the parts are then packed into
one container (rdual_engine_set.py) whose info names the architecture "RSR". The set's
key holds the weights rather than the exported graphs, so a later run, after a restart
too, reads the set without exporting.

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

if TYPE_CHECKING:
    from spandrel import ImageModelDescriptor

    from api import NodeContext

# The scales whose engines were checked against PyTorch FP32 (run_window's chain, 13
# frames in windows of 7 keeping 6). 2x on a real model at 960x540: RMSE at most 3.75e-4
# per frame, PyTorch FP16's own about 2.5e-4. 1x and 4x on random-weight models at
# 256x256: at most 1.48e-3 (FP16: about 8.5e-4) and 1.09e-3 (FP16: about 6.6e-4). At
# every scale the engines are 1.5 to 1.75 times PyTorch FP16's distance, the same on
# every frame, edges included.
CHECKED_SCALES = (1, 2, 4)
_HERE = Path(__file__).parent
# What a set's export, build and container run
RECIPE_SOURCES = (
    _HERE / "rsr.py",
    _HERE / "rsr_export.py",
    _HERE / "rsr_engine_worker.py",
    # its save(), which writes the parts
    _HERE / "rdual_export.py",
    _HERE / "rdual_engine_set.py",
    _HERE / "dual_engine_worker.py",
)


@cache
def recipe_identity() -> str:
    """The SHA-256 of the code a set is exported and built with: these modules, the
    model's file (its graph), PyTorch and spandrel."""
    from spandrel.architectures.RSR.__arch import rsr_arch

    digest = hashlib.sha256(repr((torch.__version__, spandrel.__version__)).encode())
    for source in (*RECIPE_SOURCES, Path(rsr_arch.__file__)):
        digest.update(source.read_bytes())
    return digest.hexdigest()


def engine_key(
    model: ImageModelDescriptor, height: int, width: int, gpu_index: int
) -> str:
    """The key of a model's set for frames of height x width on a GPU: everything the
    set depends on (as build_cache.engine_key, with the weights for the graphs)."""
    import tensorrt as trt

    from ..pytorch.utils import weights_digest
    from .build_cache import FORMAT_VERSION
    from .memory import get_cuda_compute_capability, get_cuda_device_name

    digest = hashlib.sha256(weights_digest(model))
    digest.update(
        repr(
            (
                FORMAT_VERSION,
                ("rsr", model.scale, height, width, BUILD_SETTINGS),
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
    """rsr_export.py's parts of the model for frames of height x width in `folder`;
    returns their manifest. The parts are traced on the GPU `gpu_index` (the one the
    build uses), else on the CPU (the graphs are the same either way)."""
    on_gpu = gpu_index is not None and torch.cuda.is_available()
    device = f"cuda:{gpu_index}" if on_gpu else "cpu"
    with tempfile.TemporaryDirectory(prefix="chaiNNer-rsr-weights-") as temporary:
        weights = write_weights(model.model, Path(temporary) / "weights.safetensors")
        run_child(
            [
                "-m",
                "nodes.impl.tensorrt.rsr_export",
                str(weights),
                str(folder),
                str(height),
                str(width),
                device,
            ],
            "RSR export",
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
    architecture: str = "RSR",
) -> None:
    """Every part's engine of an export in `engines` (<part>.engine), with Build
    Engine's `settings` (rdual.BUILD_SETTINGS' order); `timing_cache`, a file, is read
    before and written after the builds (see timing_cache_file). The worker builds any
    export of plain ONNX parts (tfdat.py's too); `architecture` names it in errors."""
    run_child(
        [
            "-m",
            "nodes.impl.tensorrt.rsr_engine_worker",
            str(folder),
            str(engines),
            str(gpu_index),
            *settings,
            *(() if timing_cache is None else (timing_cache,)),
        ],
        f"{architecture} engine build",
        SOURCE,
        context,
    )


def timing_cache_file(directory: str, folder: Path, gpu_index: int) -> str:
    """The timing cache an export's build reads and writes in the TensorRT package's
    Timing Cache folder: Build Engine's name (engine_builder.timing_cache_name) of its
    first part, after its architecture, so one file per model and input size, GPU and
    TensorRT version."""
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
    prefix = manifest["architecture"].lower()
    return str(
        Path(directory) / f"{prefix}_{manifest['width']}x{manifest['height']}_{name}"
    )


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
            architecture="RSR",
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
    with tempfile.TemporaryDirectory(prefix="chaiNNer-rsr-engines-") as temporary:
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
    "timing_cache_file",
]
