"""PyTorch Upscale Image's "Use TensorRT Engine": a model that has an automatic recipe
(FP16_ARCHS, DUAL, legacy DRFT) runs through a TensorRT engine, exported and built once
through Convert To ONNX's and Build Engine's own paths.

An engine is kept on disk by the identity of its model's weights, not of its ONNX, so a
later run, also after a restart, reads it without exporting. It stays among the Engine
Cache's files (build_cache): one folder, one limit, one eviction. With the setting on, a
user has chosen to keep engines; with it off, which is its default, the same folder is
used all the same, as the input that was turned on promises a build once, and its docs
state the disk use. With the input off, this feature keeps nothing. A model that cannot
use an engine runs on PyTorch for the rest of the chain, and the next run tries again.

This module must not import TensorRT, ONNX or the modules that need them at its top:
PyTorch's Upscale Image loads without them, and with the setting off it needs none.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from functools import cache, partial
from importlib.util import find_spec
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import numpy as np
import spandrel
import torch
from sanic.log import logger
from spandrel import ImageModelDescriptor

from api import Aborted, NodeContext, NodeId
from gpu import nvidia

from ..image_op import ImageOp
from ..onnx.model import FP16_ARCHS_NAVI, converts_in_fp16
from ..upscale.auto_split_tiles import TileSize
from .utils import safe_cuda_cache_empty

if TYPE_CHECKING:
    # Annotations only: no import at runtime.
    from ..tensorrt.model import TensorRTEngine
    from ..upscale.tile_report import TileReport

# The navi type of a model the setting applies to; a legacy DRFT cannot be told from
# another DRFT by its type.
ENGINE_MODEL_NAVI = f'PyTorchModel {{ arch: "DUAL" | "DRFT" | {FP16_ARCHS_NAVI} }}'

# Every automatic engine is dynamic, up to the node's Tile Size, so an image below the tile
# runs at its own size. Its smallest size is Build Engine's default; these, which are all
# convolutions, also take Build Engine's lowest, measured against PyTorch at the sizes
# that a pad to 64 px put at 42 to 50 dB.
MIN_SIZE = 64
CONVOLUTION_MIN_SIZE = 16
CONVOLUTION_ARCHS = frozenset({"ESRGAN", "RealESRGAN Compact", "SPAN", "PLKSR"})
# The largest size for a Tile Size with no number (Auto, Maximum, No Tiling): an engine
# is built for a size, and a build per image would not end.
DEFAULT_TILE = 512
# The largest size an architecture's engine takes, whatever the Tile Size. RGT's attention
# holds every token against its reduced keys: a Dynamic engine at 512 px nearly filled a
# 32 GB card, and its process could not be ended while it lasted. 256 px holds.
LARGEST_TILE = {"RGT": 256}
OPSET = 20

_HERE = Path(__file__).parent
# The code an engine's export and build run, besides DUAL's and DRFT's plugins. Every
# step of the export, by its name, also one added later.
_RECIPE_SOURCES = (
    *sorted(_HERE.glob("*_export.py")),
    _HERE / "auto_engine.py",
    _HERE / "convert_to_onnx_impl.py",
    _HERE.parent / "onnx" / "load.py",
    _HERE.parent / "onnx" / "model.py",
    _HERE.parent / "tensorrt" / "dense_blocks.py",
    _HERE.parent / "tensorrt" / "drft_export.py",
    _HERE.parent / "tensorrt" / "dual.py",
    _HERE.parent / "tensorrt" / "engine_builder.py",
    _HERE.parent / "tensorrt" / "group_norm.py",
    _HERE.parents[2] / "packages/chaiNNer_pytorch/pytorch/utility/convert_to_onnx.py",
    _HERE.parents[2] / "packages/chaiNNer_tensorrt/tensorrt/utility/build_engine.py",
)

_lock = Lock()
# Per model, the device, GPU and largest size it was asked for, and its engine or the reason it has
# none. The app keeps a model between runs, so one model sees PyTorch's settings change.
_Where = tuple[torch.device, int, int]
_outcomes: WeakKeyDictionary[
    ImageModelDescriptor, tuple[_Where, TensorRTEngine | str]
] = WeakKeyDictionary()
# The nodes that may hold a session on an automatic engine.
_engine_nodes: set[NodeId] = set()
# Per model, the key and folder of the engine it read rather than built, until that engine
# has run an image; and the engines dropped and built again in this process, for once.
_unproven: WeakKeyDictionary[ImageModelDescriptor, tuple[str, str]] = (
    WeakKeyDictionary()
)
_rebuilt: set[str] = set()


def _recipe(model: ImageModelDescriptor) -> str | None:
    if converts_in_fp16(model.architecture.name, model.tags):
        return "fp16"
    if model.architecture.id == "DUAL":
        return "dual"
    if model.architecture.name == "DRFT":
        from ..tensorrt import drft_export

        if drft_export.is_legacy_drft(model):
            return "drft"
    return None


def _no_engine_here(device: torch.device) -> str | None:
    """Why this machine or this run's settings have no engine; None if they may. Empty
    without NVIDIA: the node hides the setting there, so a saved chain's On cannot be turned
    off, and this is no surprise worth a warning."""
    if torch.version.hip is not None or not nvidia.is_available:
        logger.info("Use TensorRT Engine: no NVIDIA GPU here; running on PyTorch.")
        return ""
    if device.type != "cuda":
        return "PyTorch is set to the CPU."
    from packages.chaiNNer_tensorrt import package

    if package.disabled:
        return package.disabled_reason
    if find_spec("tensorrt") is None:
        return "The TensorRT package is not installed (see the Dependency Manager)."
    return None


def _weights_digest(model: ImageModelDescriptor) -> bytes:
    """An ESRGAN's in FP16, as export_copy exports them, so PyTorch's FP16 setting makes no
    second engine of a byte-identical ONNX."""
    digest = hashlib.sha256()
    as_exported = model.architecture.name == "ESRGAN"
    for name, value in model.model.state_dict().items():
        rounded = value.half() if as_exported and value.is_floating_point() else value
        array = np.ascontiguousarray(rounded.detach().cpu().numpy())
        digest.update(repr((name, array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.digest()


@cache
def _recipe_identity() -> str:
    digest = hashlib.sha256(repr((torch.__version__, spandrel.__version__)).encode())
    for source in _RECIPE_SOURCES:
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _engine_tile(model: ImageModelDescriptor, tile_size: TileSize) -> int:
    """The largest size of the model's engine for the node's Tile Size."""
    tile = max(int(tile_size), MIN_SIZE) if tile_size > 0 else DEFAULT_TILE
    return min(tile, LARGEST_TILE.get(model.architecture.name, tile))


def _engine_key(
    model: ImageModelDescriptor, recipe: str, gpu_index: int, tile: int
) -> str:
    """As build_cache.engine_key, with the weights in place of the ONNX, which a later
    run does not export: everything the engine depends on."""
    import tensorrt as trt

    from ..tensorrt import build_cache
    from ..tensorrt.memory import get_cuda_compute_capability, get_cuda_device_name

    plugins = {
        "dual": build_cache.dual_plugin_identity,
        "drft": build_cache.drft_plugin_identity,
    }.get(recipe)
    digest = hashlib.sha256(_weights_digest(model))
    digest.update(
        repr(
            (
                build_cache.FORMAT_VERSION,
                (
                    recipe,
                    model.architecture.name,
                    model.scale,
                    tile,
                    _recipe_identity(),
                ),
                get_cuda_device_name(gpu_index),
                tuple(get_cuda_compute_capability(gpu_index)),
                trt.__version__,
                None if plugins is None else plugins(),
            )
        ).encode()
    )
    return digest.hexdigest()


def _with_aligned_input(onnx_bytes: bytes, multiple_of: int) -> bytes:
    """The ONNX with its input named for the multiple its sizes must be
    (dual.aligned_input_name), which Upscale Image's tiler pads to instead of its default
    64 px. An image pads as far as PyTorch pads it, so its right and bottom edges come out
    as PyTorch's: padded to 64 px, a 4x ESRGAN differs there by up to 0.98."""
    import onnx

    from ..tensorrt.dual import aligned_input_name

    model = onnx.load_model_from_string(onnx_bytes)
    graph = model.graph
    source, name = graph.input[0].name, aligned_input_name(multiple_of)
    graph.input[0].name = name
    for node in graph.node:
        for index, value in enumerate(node.input):
            if value == source:
                node.input[index] = name
    return model.SerializeToString()


def _build(
    context: NodeContext, model: ImageModelDescriptor, tile: int
) -> TensorRTEngine:
    """Convert To ONNX's and Build Engine's own paths. Both nodes read their settings from
    `context`, PyTorch's here: the GPU's key is the same in both packages, and the TensorRT
    package's two cache folders are unset, so the engine is kept only by `cached_engine`
    in `_engine_of`."""
    from packages.chaiNNer_pytorch.pytorch.utility.convert_to_onnx import (
        DualShape,
        OnnxDataType,
        Opset,
        convert_to_onnx_node,
    )
    from packages.chaiNNer_tensorrt.tensorrt.utility.build_engine import (
        Precision,
        ShapeMode,
        TilingOptimization,
        build_engine_node,
    )

    from ..onnx.load import load_onnx_model
    from ..tensorrt import drft_export

    onnx_model = convert_to_onnx_node(
        context,
        model,
        OnnxDataType.FP16,
        Opset(OPSET),
        False,
        DualShape.DYNAMIC,
        tile,
        tile,
    )[0]
    context.check_aborted()
    low = MIN_SIZE
    recipe = _recipe(model)
    if recipe == "drft":
        low = drft_export.min_size(model.model.window_size)
    elif recipe == "fp16":
        req = model.size_requirements
        onnx_model = load_onnx_model(
            _with_aligned_input(onnx_model.bytes, req.multiple_of)
        )
        if model.architecture.name in CONVOLUTION_ARCHS:
            low = CONVOLUTION_MIN_SIZE
        low = -(-max(low, req.minimum) // req.multiple_of) * req.multiple_of
        tile = -(-tile // req.multiple_of) * req.multiple_of
    tile = max(tile, low)
    engine = build_engine_node(
        context,
        onnx_model,
        Precision.FP16,
        ShapeMode.DYNAMIC,
        low,
        low,
        tile,
        tile,
        tile,
        tile,
        tile,
        tile,
        0,
        False,
        False,
        3,
        TilingOptimization.NONE,
    )
    if engine.scale != model.scale:
        raise ValueError(
            f"The engine upscales {engine.scale}x, the model {model.scale}x."
        )
    return engine


def _described(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _forget_reason(model: ImageModelDescriptor) -> None:
    with _lock:
        kept = _outcomes.get(model)
        if kept is not None and isinstance(kept[1], str):
            del _outcomes[model]


def _no_engine(
    context: NodeContext, model: ImageModelDescriptor, where: _Where, reason: str
) -> str:
    """Remember for this chain that the model runs on PyTorch, once saying why (an empty
    `reason` says nothing)."""
    with _lock:
        _outcomes[model] = (where, reason)
    context.add_cleanup(partial(_forget_reason, model), after="chain")
    if reason:
        context.warn(
            f"Upscale Image runs this {model.architecture.name} model on PyTorch, not"
            f" through a TensorRT engine: {reason}"
        )
    return reason


def _cache_folder(context: NodeContext) -> Path:
    """The Engine Cache's folder, its setting on or off: the one the app sets, or where
    the app would set it, in the root that holds the backend's storage folder."""
    from packages.chaiNNer_tensorrt.settings import ENGINE_CACHE_DIRECTORY

    settings = context.settings.of_package("chaiNNer_tensorrt")
    location = settings.get_cache_location("engine_cache")
    if location is not None:
        return Path(location)
    return context.storage_dir.parent / "cache" / ENGINE_CACHE_DIRECTORY


def _engine_of(
    context: NodeContext,
    model: ImageModelDescriptor,
    device: torch.device,
    gpu_index: int,
    tile_size: TileSize,
) -> TensorRTEngine | str:
    """The model's engine for this Tile Size, or why it has none. Stop raises Aborted."""
    tile = _engine_tile(model, tile_size)
    where = (device, gpu_index, tile)
    with _lock:
        kept = _outcomes.get(model)
    if kept is not None and kept[0] == where:
        return kept[1]

    recipe = _recipe(model)
    if recipe is None:
        # The node hides the setting for these, so this is no surprise worth a warning;
        # except a DRFT, whose type does not tell a legacy one.
        logger.info(
            "Use TensorRT Engine: this %s model has no automatic engine; running on"
            " PyTorch.",
            model.architecture.name,
        )
        reason = ""
        if model.architecture.name == "DRFT":
            reason = "only legacy DRFT models (LayerNorm, not i-LN) have one."
        return _no_engine(context, model, where, reason)
    if tile < tile_size:
        context.warn(
            f"Use TensorRT Engine: this {model.architecture.name} model's engine stops at"
            f" {tile} px, so it runs in tiles of that size, not {tile_size} px: above that"
            " its memory can fill the GPU."
        )
    try:
        reason = _no_engine_here(device)
        if reason is not None:
            return _no_engine(context, model, where, reason)
        from ..tensorrt.build_cache import cached_engine

        key = _engine_key(model, recipe, gpu_index, tile)
        folder = _cache_folder(context)
        folder.mkdir(parents=True, exist_ok=True)
        # What PyTorch keeps cached on the GPU is the build's and the session's now.
        safe_cuda_cache_empty()
        built = False

        def build() -> TensorRTEngine:
            nonlocal built
            built = True
            return _build(context, model, tile)

        engine = cached_engine(key, build, str(folder))
    except Aborted:
        raise
    except Exception as e:
        logger.exception("Use TensorRT Engine: no engine for this model")
        return _no_engine(context, model, where, _described(e))
    with _lock:
        _outcomes[model] = (where, engine)
        if not built and key not in _rebuilt:
            _unproven[model] = (key, str(folder))
    return engine


def release_engine(node_id: NodeId) -> None:
    """Free a session this node holds on an automatic engine, if any: its GPU memory is
    PyTorch's again."""
    with _lock:
        held = node_id in _engine_nodes
        _engine_nodes.discard(node_id)
    if held:
        from ..tensorrt.cache import release_node

        release_node(node_id)


def _is_finite(img: np.ndarray) -> bool:
    # NaN propagates through min and max: two passes and no temporary array
    return math.isfinite(img.min()) and math.isfinite(img.max())


def _engine_op(
    context: NodeContext,
    model: ImageModelDescriptor,
    engine: TensorRTEngine,
    report: TileReport | None = None,
) -> ImageOp:
    """TensorRT's own Upscale Image, on `context`'s GPU, raising where PyTorch's would on
    invalid values. `report` takes the tiles it ran."""
    from packages.chaiNNer_tensorrt.tensorrt.processing.upscale_image import (
        engine_upscale,
    )

    def run(img: np.ndarray) -> np.ndarray:
        def checked(raw: np.ndarray) -> None:
            if not _is_finite(raw) and _is_finite(img):
                raise RuntimeError(
                    f"The engine of this {model.architecture.name} model returned invalid"
                    " values (NaN or infinity), which would be saved as black or white"
                    " pixels."
                )

        output, tiles = engine_upscale(
            context, img, engine, False, False, before_clip=checked
        )
        if report is not None:
            report.text = tiles
        return output

    return run


def _drop_refused_file(model: ImageModelDescriptor, error: Exception) -> bool:
    """Whether `error`, which PyTorch did not repeat on the same image, is TensorRT refusing
    the engine read from its file: then the file goes, and the next image builds. Never run
    here, and not for want of memory. Once per engine in a process: one built anew and
    refused too says nothing of the file."""
    from ..tensorrt.build_cache import forget_engine
    from ..tensorrt.memory import CudaOutOfMemory

    with _lock:
        unproven = _unproven.pop(model, None)
        if unproven is None or isinstance(error, CudaOutOfMemory):
            return False
        _rebuilt.add(unproven[0])
        _outcomes.pop(model, None)
    forget_engine(*unproven)
    return True


def upscale_through_engine(
    context: NodeContext,
    model: ImageModelDescriptor,
    device: torch.device,
    gpu_index: int,
    tile_size: TileSize,
    upscale_with: Callable[[ImageOp], np.ndarray],
    pytorch: ImageOp,
    report: TileReport | None = None,
) -> np.ndarray:
    """`upscale_with` the model's engine, or `pytorch` when it has none or fails. The whole
    image starts over: an image with transparency must not mix the two (its two passes
    are compared). `report` takes the engine's tiles; `pytorch` records its own."""
    engine = _engine_of(context, model, device, gpu_index, tile_size)
    if isinstance(engine, str):
        release_engine(context.node_id)
        return upscale_with(pytorch)
    with _lock:
        _engine_nodes.add(context.node_id)
    try:
        output = upscale_with(_engine_op(context, model, engine, report))
    except Aborted:
        raise
    except Exception as e:
        logger.exception("Use TensorRT Engine: the engine failed on this image")
        error = e
    else:
        with _lock:
            _unproven.pop(model, None)
        return output
    release_engine(context.node_id)
    # An error here is the image's own, as bad on either route: the engine stays for the
    # next image, and the error is the one to report.
    output = upscale_with(pytorch)
    if _drop_refused_file(model, error):
        context.warn(
            f"Upscale Image ran this image on PyTorch: TensorRT could not start this"
            f" {model.architecture.name} model's kept engine, so it is built again for"
            f" the next image. {_described(error)}"
        )
    else:
        where = (device, gpu_index, _engine_tile(model, tile_size))
        _no_engine(context, model, where, _described(error))
    return output
