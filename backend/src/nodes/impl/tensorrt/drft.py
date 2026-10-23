"""Legacy DRFT's TensorRT path: Convert To ONNX exports the model with TensorRT plugin
nodes (drft_export.py), and Build Engine builds it in a child process with chaiNNer-C's
plugin library (drft_trt_plugins.dll, native/src/drft). Engines do not carry the
library: an engine that carries one registers it at every load, and TensorRT refuses the
second registration in a process, so a second DRFT engine (or the same one again) would
not load. Every process that loads engines registers the library once (load_plugins)."""

from __future__ import annotations

import json
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import dual

if TYPE_CHECKING:
    import onnx

    from api import NodeContext

# The exporter's manifest, carried in the ONNX model's metadata.
METADATA_KEY = "chainner_c.drft_tensorrt.export"
# The plugins' library (native/Build.ps1 with the TensorRT and CUDA roots).
PLUGIN_LIBRARY = Path(__file__).with_name("drft_trt_plugins.dll")


def export_manifest(model: bytes | onnx.ModelProto) -> dict[str, Any] | None:
    """The exporter's manifest of a DRFT TensorRT ONNX (its bytes or the parsed model);
    None for any other model."""
    import onnx

    if isinstance(model, bytes):
        model = onnx.load_from_string(model)
    for entry in model.metadata_props:
        if entry.key == METADATA_KEY:
            return json.loads(entry.value)
    return None


def plugin_library() -> Path:
    """The plugin library an engine build loads and serializes, or an error that says
    how to get it."""
    if not PLUGIN_LIBRARY.is_file():
        raise RuntimeError(
            f"The DRFT TensorRT plugin library {PLUGIN_LIBRARY} is missing: this build"
            " was made without it. Use the PyTorch backend, or build it with"
            " native/Build.ps1 -Define CHAINNER_C_TENSORRT_ROOT=<TensorRT 11 SDK>,"
            "CHAINNER_C_CUDA_ROOT=<CUDA 13 toolkit> (native/README.md, Build)."
        )
    return PLUGIN_LIBRARY


@lru_cache(maxsize=1)
def load_plugins() -> None:
    """Register the plugin library's creators in this process's TensorRT plugin
    registry, once, before any engine is deserialized (engine_info.deserialize_engine,
    runner.load_tensorrt). A build without the library registers nothing; DRFT engines
    then fail to deserialize, other engines are unaffected."""
    import tensorrt

    if PLUGIN_LIBRARY.is_file():
        tensorrt.get_plugin_registry().load_library(str(PLUGIN_LIBRARY))


Size = tuple[int, int]


def check_profile(manifest: dict[str, Any], low: Size, opt: Size, high: Size) -> None:
    """Refuse a DRFT engine's (height, width) profile that the ONNX does not take: sizes
    must be multiples of its alignment, at least its minimum size on each side, and
    ordered. One size three times is a fixed engine (Shape Mode Fixed), and the message
    says so. Upscale Image's tiler pads every tile to the engine's minimum, so no tile
    is smaller than the ONNX's minimum size."""
    alignment, min_size = manifest["alignment"], manifest["min_size"]
    fixed = low == opt == high
    for name, (h, w) in (("minimum", low), ("optimal", opt), ("maximum", high)):
        what = (
            "A fixed DRFT engine's size"
            if fixed
            else f"A dynamic DRFT engine's {name} size"
        )
        if h % alignment or w % alignment:
            raise ValueError(
                f"{what} must be a multiple of {alignment} px (got {h}x{w})."
            )
        if h < min_size or w < min_size:
            raise ValueError(
                f"{what} must be at least {min_size} px on each side (got {h}x{w})."
            )
    if not (low[0] <= opt[0] <= high[0] and low[1] <= opt[1] <= high[1]):
        raise ValueError(
            "A dynamic DRFT engine needs minimum <= optimal <= maximum sizes."
        )


def build_engine(
    onnx_bytes: bytes,
    manifest: dict[str, Any],
    gpu_index: int,
    profile: tuple[Size, Size, Size],
    *,
    workspace_gib: float = 0.0,
    optimization_level: int = 3,
    tf32: bool = False,
    sparse_weights: bool = True,
    tiling_optimization: str = "none",
    context: NodeContext | None = None,
) -> bytes:
    """A TensorRT engine of a DRFT ONNX for a (minimum, optimal, maximum) (height,
    width) profile (see check_profile), referencing the plugin library's plugins,
    built by chaiNNer-C's TensorRT in drft_engine_worker.py (a child process: the
    library's plugins register there, never in this process). A workspace of 0 is
    TensorRT's default; the defaults are the reference build policy (TensorRT's workspace,
    level 3, TF32 off, sparse weights, no tiling optimization); `tiling_optimization`
    is none, fast, moderate or full."""
    library = plugin_library()
    settings = [
        str(workspace_gib),
        str(optimization_level),
        "1" if tf32 else "0",
        "1" if sparse_weights else "0",
        tiling_optimization,
    ]
    sizes = [*(f"{h}x{w}" for h, w in profile), str(manifest["alignment"])]
    with tempfile.TemporaryDirectory(prefix="chainner-drft-engine-") as temporary:
        model = Path(temporary) / "model.onnx"
        model.write_bytes(onnx_bytes)
        engine = Path(temporary) / "model.engine"
        dual.run_child(
            [
                "-m",
                "nodes.impl.tensorrt.drft_engine_worker",
                str(model),
                str(engine),
                str(gpu_index),
                str(library),
                *settings,
                *sizes,
            ],
            "DRFT engine build",
            dual.SOURCE,
            context,
        )
        return engine.read_bytes()


__all__ = [
    "METADATA_KEY",
    "PLUGIN_LIBRARY",
    "build_engine",
    "check_profile",
    "export_manifest",
    "load_plugins",
    "plugin_library",
]
