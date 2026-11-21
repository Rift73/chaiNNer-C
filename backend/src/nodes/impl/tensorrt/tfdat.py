"""TFDAT's and TFDATv2's TensorRT engine sets: a set's key, and the set itself through the
Engine Cache (build_cache.cached_engine), as rsr.py gives RSR's. Each model has no state
between its clips, so its set is one engine, "clip" (tfdat_export.py), built without
plugins by rsr_engine_worker.py; the set container (rdual_engine_set.py) names the
architecture, which Upscale Image's temporal path runs (tfdat_session.py). The set's key
holds the weights rather than the exported graph, so a later run, after a restart too,
reads the set without exporting.

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

# The scales whose engines were checked against PyTorch FP32: 2 on the trained TFDATv2
# (CHECKED in the module docstring of tfdat_export.py), 1, 2 and 4 on small models of
# both versions with every parameter random (test_tensorrt_tfdat.py).
CHECKED_SCALES = (1, 2, 4)
_HERE = Path(__file__).parent
# What a set's export, build and container run
RECIPE_SOURCES = (
    _HERE / "tfdat.py",
    _HERE / "tfdat_export.py",
    _HERE / "rsr_engine_worker.py",
    # its save(), which writes the part
    _HERE / "rdual_export.py",
    _HERE / "rdual_engine_set.py",
    _HERE / "dual_engine_worker.py",
)


@cache
def recipe_identity() -> str:
    """The SHA-256 of the code a set is exported and built with: these modules, the
    models' files (their graphs), PyTorch and spandrel."""
    from spandrel.architectures.TFDAT.__arch import tfdat, tfdat_v2

    digest = hashlib.sha256(repr((torch.__version__, spandrel.__version__)).encode())
    for source in (*RECIPE_SOURCES, Path(tfdat.__file__), Path(tfdat_v2.__file__)):
        digest.update(source.read_bytes())
    return digest.hexdigest()


def engine_key(
    model: ImageModelDescriptor, height: int, width: int, gpu_index: int
) -> str:
    """The key of a model's set for frames of height x width on a GPU: everything the
    set depends on (as build_cache.engine_key, with the weights for the graph)."""
    import tensorrt as trt

    from ..pytorch.utils import weights_digest
    from .build_cache import FORMAT_VERSION
    from .memory import get_cuda_compute_capability, get_cuda_device_name

    digest = hashlib.sha256(weights_digest(model))
    digest.update(
        repr(
            (
                FORMAT_VERSION,
                (model.architecture.id, model.scale, height, width, BUILD_SETTINGS),
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
    """tfdat_export.py's part of the model for frames of height x width in `folder`;
    returns its manifest. The part is traced on the GPU `gpu_index` (the one the build
    uses), else on the CPU (the graph is the same either way)."""
    on_gpu = gpu_index is not None and torch.cuda.is_available()
    device = f"cuda:{gpu_index}" if on_gpu else "cpu"
    architecture = model.architecture.id
    with tempfile.TemporaryDirectory(prefix="chaiNNer-tfdat-weights-") as temporary:
        weights = write_weights(model.model, Path(temporary) / "weights.safetensors")
        run_child(
            [
                "-m",
                "nodes.impl.tensorrt.tfdat_export",
                architecture,
                str(weights),
                str(folder),
                str(height),
                str(width),
                device,
            ],
            f"{architecture} export",
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
    """The part's engine of an export in `engines` (clip.engine), by RSR's worker, which
    builds any export of plain ONNX parts (rsr.build)."""
    architecture = json.loads((folder / "export.json").read_text(encoding="utf-8"))[
        "architecture"
    ]
    build_parts(
        folder, engines, gpu_index, context, settings, timing_cache, architecture
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
            architecture=str(manifest["architecture"]),
            height=height,
            width=width,
            plugin_key="",
            parts=tuple(manifest["parts"]),
            clip_size=int(manifest["clip_size"]),
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
    with tempfile.TemporaryDirectory(prefix="chaiNNer-tfdat-engines-") as temporary:
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
