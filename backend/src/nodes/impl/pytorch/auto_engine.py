"""PyTorch Upscale Image's "Use TensorRT Engine" and, where MIGraphX runs, its "Use MIGraphX
Program": a model that has an automatic recipe (FP16_ARCHS, DUAL, legacy DRFT) runs through
a TensorRT engine, exported and built once through Convert To ONNX's and Build Engine's own
paths, or a MIGraphX program through Convert To ONNX's and Build Program's. One flow
(_engine_of, upscale_through_engine) on two routes (_TensorRT, _MIGraphX). What follows of
an engine and the Engine Cache holds of a program and the Program Cache, except its size:
a program takes one, the Tile Size (_program_size).

An engine is kept on disk by the identity of its model's weights, not of its ONNX, so a
later run, also after a restart, reads it without exporting. It stays among the Engine
Cache's files (build_cache): one folder, one limit, one eviction. With the setting on, a
user has chosen to keep engines; with it off, which is its default, the same folder is
used all the same, as the input that was turned on promises a build once, and its docs
state the disk use. With the input off, this feature keeps nothing. A model that cannot
use an engine runs on PyTorch for the rest of the chain, and the next run tries again.

This module must not import TensorRT, MIGraphX, ONNX or the modules that need them at its
top: PyTorch's Upscale Image loads without them, and with the setting off it needs none.
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
from gpu import amd, nvidia

from ..image_op import ImageOp
from ..onnx.model import FP16_ARCHS_NAVI, converts_in_fp16
from ..upscale.auto_split_tiles import TileSize
from .utils import safe_cuda_cache_empty

if TYPE_CHECKING:
    # Annotations only: no import at runtime.
    from ..migraphx.model import MIGraphXProgram
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
# The largest size of a TensorRT engine for a Tile Size with no number, for the
# architectures measured whole: an image up to it runs as one tile, and one that does not
# fit the GPU's memory goes on in smaller tiles (inference.py). DAT and HAT pool and
# normalize over the whole tile, so a tiled image differs from the whole one far from
# its seams: against strict FP32 on the whole image, a 4x DAT2 at 1200x864 read 47.9 dB
# in 512 px tiles (51 codes off at worst) and 68.3 dB whole, a 4x HAT-L 58.2 and 67.3 dB.
# A whole image also takes no overlap: on an RTX 5090, 1200x864 DAT2 1.33 -> 1.03 s and
# HAT-L 2.26 -> 1.55 s, 960x540 0.89 -> 0.47 s and 1.53 -> 0.73 s. Their closing
# convolutions stay FP32, for which TensorRT 11.4 finds no kernel over the whole range of
# a larger engine: a 4x DAT2 builds at 1408 px and fails at 1536. Such an engine keeps the
# optimum of its profile at DEFAULT_TILE (_build), and takes about 2 GiB more at any size
# (the DAT2's 2.8 GiB at 128x128, against 0.8 GiB at 256x256 for the 512 px engine), with
# 11.7 GiB for a whole 1200x864 image: a GPU with less memory than WHOLE_IMAGE_MEMORY (a
# card sold as 16 GB reports a little under 16 GiB) keeps DEFAULT_TILE. A program (MIGraphX)
# has one size, DEFAULT_TILE.
# DUAL's tiles change its output too: a 2x DUAL Light read 52.1 dB at 1200x864 and 48.9 dB
# at 1920x1080 in 512 px tiles (48 codes off at worst), and runs 0.093 -> 0.070 s and
# 0.225 -> 0.144 s whole. At 128 channels it builds up to 1920 px; at 2048 px one of its
# tactics asks more than the build's 8 GiB workspace and the build fails. A wider DUAL,
# not measured, takes a size of as many pixels times channels (_whole_image_tile), which
# keeps its kernels' 32-bit offsets (dual.check_profile). FIGSR's Fourier units transform
# the whole tile: in 512 px tiles a 4x FIGSR read 47.3 dB at 1200x864 and 43.5 dB at
# 960x540 (118 codes off at worst). Up to 1408 px every multiple of 64 runs; an engine up
# to 1920 or 2048 px fails to run its DFTs at widths from 832 px (cuFFT: no workspace).
WHOLE_IMAGE_TILE = {"DAT": 1408, "HAT": 1408, "DUAL": 1920, "FIGSR": 1408}
WHOLE_IMAGE_MEMORY = 15 * 1024**3
DUAL_WHOLE_IMAGE_CHANNELS = 128
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
# What a program's build runs besides those, its runner and its DUAL or DRFT route.
_PROGRAM_SOURCES = (
    _HERE.parent / "migraphx" / "program.py",
    _HERE.parent / "migraphx" / "program_worker.py",
    _HERE.parents[2] / "packages/chaiNNer_migraphx/migraphx/utility/build_program.py",
)

_lock = Lock()
# Per model, the device, GPU and largest size it was asked for, and its engine or the reason it has
# none. The app keeps a model between runs, so one model sees PyTorch's settings change.
_Where = tuple[torch.device, int, int]
_outcomes: WeakKeyDictionary[
    ImageModelDescriptor, tuple[_Where, TensorRTEngine | MIGraphXProgram | str]
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
    # Both routes take one list, whose FP16 was measured on TensorRT.
    if converts_in_fp16(model.architecture.name, model.tags):
        return "fp16"
    if model.architecture.id == "DUAL":
        return "dual"
    if model.architecture.name == "DRFT":
        from ..tensorrt import drft_export

        if drft_export.is_legacy_drft(model):
            return "drft"
    return None


def migraphx_here() -> bool:
    """Whether the MIGraphX package can run on this PC, which makes MIGraphX the route. A PC
    with an NVIDIA GPU or no worthwhile AMD one does not even import the package."""
    if nvidia.is_available or not amd.eligible:
        return False
    from packages.chaiNNer_migraphx import disabled_reason

    return disabled_reason() is None


def _no_engine_here(device: torch.device) -> str | None:
    """Why this machine or this run's settings have no engine or program; None if they may.
    Empty without NVIDIA or MIGraphX: the node hides the setting there, so a saved chain's On
    cannot be turned off, and this is no surprise worth a warning."""
    migraphx = migraphx_here()
    if not migraphx and (torch.version.hip is not None or not nvidia.is_available):
        logger.info("Use TensorRT Engine: no NVIDIA GPU here; running on PyTorch.")
        return ""
    if device.type != "cuda":
        return "PyTorch is set to the CPU."
    if migraphx:
        from packages.chaiNNer_migraphx import migraphx_loaded
        from packages.chaiNNer_migraphx import package as migraphx_package

        if migraphx_package.disabled:
            return migraphx_package.disabled_reason
        if not migraphx_loaded:
            return "The MIGraphX package is not installed (see the Dependency Manager)."
        return None
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
def _recipe_identity(route_sources: tuple[Path, ...] = ()) -> str:
    digest = hashlib.sha256(repr((torch.__version__, spandrel.__version__)).encode())
    for source in (*_RECIPE_SOURCES, *route_sources):
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _whole_image_tile(model: ImageModelDescriptor) -> int:
    """The size of the TensorRT engine for whole images of a model whose architecture
    WHOLE_IMAGE_TILE lists."""
    tile = WHOLE_IMAGE_TILE[model.architecture.name]
    if model.architecture.name == "DUAL":
        pixels = tile * tile * DUAL_WHOLE_IMAGE_CHANNELS // model.model.embed_dim
        tile = min(tile, math.isqrt(pixels) // 64 * 64)
    return tile


def _whole_images(
    model: ImageModelDescriptor, tile_size: TileSize, gpu_index: int
) -> bool:
    """Whether the model's engine for the node's Tile Size takes whole images: a Tile Size
    with no number, an architecture WHOLE_IMAGE_TILE lists, TensorRT, and a CUDA GPU of
    WHOLE_IMAGE_MEMORY or more."""
    return (
        tile_size <= 0
        and model.architecture.name in WHOLE_IMAGE_TILE
        and _route() is _TensorRT
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties(gpu_index).total_memory
        >= WHOLE_IMAGE_MEMORY
    )


def _engine_tile(
    model: ImageModelDescriptor, tile_size: TileSize, gpu_index: int
) -> int:
    """The largest size of the model's engine or program for the node's Tile Size on the
    GPU of `gpu_index`."""
    arch = model.architecture.name
    if tile_size > 0:
        tile = max(int(tile_size), MIN_SIZE)
    elif _whole_images(model, tile_size, gpu_index):
        tile = _whole_image_tile(model)
    else:
        tile = DEFAULT_TILE
    return min(tile, LARGEST_TILE.get(arch, tile))


def _program_size(model: ImageModelDescriptor, recipe: str, tile: int) -> int:
    """The one input size of the model's program for its `tile` (_engine_tile), as a program
    has no range of sizes: the tile aligned up to the model's multiple, and at least its
    smallest. DUAL's ONNX takes multiples of its own alignment, which its size requirement
    does not name, and a legacy DRFT's smallest side is two windows."""
    req = model.size_requirements
    multiple, low = req.multiple_of, req.minimum
    if recipe == "dual":
        from ..tensorrt.dual import ALIGNMENT

        multiple = ALIGNMENT
    elif recipe == "drft":
        from ..tensorrt import drft_export

        low = max(low, drft_export.min_size(model.model.window_size))
    return -(-max(tile, low) // multiple) * multiple


def _program_key(
    context: NodeContext, model: ImageModelDescriptor, recipe: str, tile: int
) -> str:
    """As migraphx.build_cache.program_key, with the weights in place of the ONNX, which a
    later run does not export: everything the program depends on."""
    from packages.chaiNNer_migraphx.settings import get_settings

    from ..migraphx import build_cache, drft, dual
    from ..migraphx.runner import runner_identity

    settings = get_settings(context)
    routes = {"dual": dual.build_identity, "drft": drft.build_identity}.get(recipe)
    digest = hashlib.sha256(_weights_digest(model))
    digest.update(
        repr(
            (
                build_cache.FORMAT_VERSION,
                (
                    recipe,
                    model.architecture.name,
                    model.scale,
                    _program_size(model, recipe, tile),
                    _recipe_identity(_PROGRAM_SOURCES),
                ),
                settings.gpu_name,
                settings.gfx,
                runner_identity(),
                None if routes is None else routes(),
            )
        ).encode()
    )
    return digest.hexdigest()


def _engine_key(
    model: ImageModelDescriptor,
    recipe: str,
    gpu_index: int,
    tile: int,
    whole: bool = False,
) -> str:
    """As build_cache.engine_key, with the weights in place of the ONNX, which a later
    run does not export: everything the engine depends on. An engine for whole images has
    its own optimum, so its own key."""
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
                    (DEFAULT_TILE, tile) if whole else tile,
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
    context: NodeContext, model: ImageModelDescriptor, tile: int, whole: bool = False
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
    recipe = _recipe(model)
    # A DAT or HAT engine for whole images keeps the optimum of its profile at
    # DEFAULT_TILE, where tiles and most images are: its kernels for that size are as fast
    # at its largest, not the other way round (a 4x DAT2, Dynamic up to 1280 px, at
    # 1216x896 1035 ms either way, at 256x256 41.3 ms against 44.7 with the optimum at
    # 1216). DUAL's keeps it at its largest: with the optimum at 512 px, a DUAL Light up
    # to 1920 px has a tactic that asks more than the build's 8 GiB workspace, and fails.
    opt = DEFAULT_TILE if whole and recipe == "fp16" else tile
    low = MIN_SIZE
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
        opt = -(-opt // req.multiple_of) * req.multiple_of
    tile = max(tile, low)
    opt = min(max(opt, low), tile)
    engine = build_engine_node(
        context,
        onnx_model,
        Precision.FP16,
        ShapeMode.DYNAMIC,
        low,
        low,
        opt,
        opt,
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


def _build_program(
    context: NodeContext, model: ImageModelDescriptor, tile: int
) -> MIGraphXProgram:
    """Convert To ONNX's and Build Program's own paths. Both nodes read their settings from
    `context`, PyTorch's here: the GPU's key is the same in both packages, and the MIGraphX
    package's Program Cache is unset, so the program is kept only by `cached_program` in
    `_engine_of`. The recipe's precision is the ONNX's record, or its own for DUAL and DRFT:
    Build Program's Precision does not apply to any of them."""
    from packages.chaiNNer_migraphx.migraphx.utility.build_program import (
        ProgramPrecision,
        build_program_node,
    )
    from packages.chaiNNer_pytorch.pytorch.utility.convert_to_onnx import (
        DualShape,
        OnnxDataType,
        Opset,
        convert_to_onnx_node,
    )

    recipe = _recipe(model)
    assert recipe is not None
    size = _program_size(model, recipe, tile)
    onnx_model = convert_to_onnx_node(
        context,
        model,
        OnnxDataType.FP16,
        Opset(OPSET),
        False,
        DualShape.DYNAMIC,
        size,
        size,
    )[0]
    context.check_aborted()
    program = build_program_node(
        context, onnx_model, ProgramPrecision.FP16, size, size, False, False
    )
    if program.scale != model.scale:
        raise ValueError(
            f"The program upscales {program.scale}x, the model {model.scale}x."
        )
    return program


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
    route = _route()
    with _lock:
        _outcomes[model] = (where, reason)
    context.add_cleanup(partial(_forget_reason, model), after="chain")
    if reason:
        context.warn(
            f"Upscale Image runs this {model.architecture.name} model on PyTorch, not"
            f" through a {route.maker} {route.made}: {reason}"
        )
    return reason


def _cache_folder(
    context: NodeContext, package: str, setting: str, directory: str
) -> Path:
    """A package's cache folder, its setting on or off: the one the app sets, or where the
    app would set it, in the root that holds the backend's storage folder."""
    location = context.settings.of_package(package).get_cache_location(setting)
    if location is not None:
        return Path(location)
    return context.storage_dir.parent / "cache" / directory


class _TensorRT:
    """The route on NVIDIA GPUs: Build Engine's engines, in the Engine Cache's folder."""

    input = "Use TensorRT Engine"
    maker = "TensorRT"
    made = "engine"

    @staticmethod
    def key(
        context: NodeContext,
        model: ImageModelDescriptor,
        recipe: str,
        gpu_index: int,
        tile: int,
        whole: bool,
    ) -> str:
        return _engine_key(model, recipe, gpu_index, tile, whole)

    @staticmethod
    def folder(context: NodeContext) -> Path:
        from packages.chaiNNer_tensorrt.settings import ENGINE_CACHE_DIRECTORY

        return _cache_folder(
            context, "chaiNNer_tensorrt", "engine_cache", ENGINE_CACHE_DIRECTORY
        )

    @staticmethod
    def cached(
        key: str, build: Callable[[], TensorRTEngine], folder: str
    ) -> TensorRTEngine:
        from ..tensorrt.build_cache import cached_engine

        return cached_engine(key, build, folder)

    @staticmethod
    def forget(key: str, folder: str) -> None:
        from ..tensorrt.build_cache import forget_engine

        forget_engine(key, folder)

    @staticmethod
    def build(
        context: NodeContext, model: ImageModelDescriptor, tile: int, whole: bool
    ) -> TensorRTEngine:
        return _build(context, model, tile, whole)

    @staticmethod
    def op(
        context: NodeContext,
        model: ImageModelDescriptor,
        engine: TensorRTEngine,
        report: TileReport | None,
    ) -> ImageOp:
        return _engine_op(context, model, engine, report)

    @staticmethod
    def short_of_memory(error: Exception) -> bool:
        from ..tensorrt.memory import CudaOutOfMemory

        return isinstance(error, CudaOutOfMemory)


class _MIGraphX:
    """The route where MIGraphX runs: Build Program's programs, in the Program Cache's
    folder."""

    input = "Use MIGraphX Program"
    maker = "MIGraphX"
    made = "program"

    @staticmethod
    def key(
        context: NodeContext,
        model: ImageModelDescriptor,
        recipe: str,
        _gpu_index: int,
        tile: int,
        _whole: bool,
    ) -> str:
        return _program_key(context, model, recipe, tile)

    @staticmethod
    def folder(context: NodeContext) -> Path:
        from packages.chaiNNer_migraphx.settings import PROGRAM_CACHE_DIRECTORY

        return _cache_folder(
            context, "chaiNNer_migraphx", "program_cache", PROGRAM_CACHE_DIRECTORY
        )

    @staticmethod
    def cached(
        key: str, build: Callable[[], MIGraphXProgram], folder: str
    ) -> MIGraphXProgram:
        from ..migraphx.build_cache import cached_program

        return cached_program(key, build, folder)

    @staticmethod
    def forget(key: str, folder: str) -> None:
        from ..migraphx.build_cache import forget_program

        forget_program(key, folder)

    @staticmethod
    def build(
        context: NodeContext, model: ImageModelDescriptor, tile: int, _whole: bool
    ) -> MIGraphXProgram:
        return _build_program(context, model, tile)

    @staticmethod
    def op(
        context: NodeContext,
        model: ImageModelDescriptor,
        program: MIGraphXProgram,
        report: TileReport | None,
    ) -> ImageOp:
        return _program_op(context, model, program, report)

    @staticmethod
    def short_of_memory(error: Exception) -> bool:
        # The runner's HIP errors name hipErrorOutOfMemory; MIGraphX's own say it in words.
        return "outofmemory" in str(error).replace(" ", "").lower()


def _route() -> type[_TensorRT] | type[_MIGraphX]:
    return _MIGraphX if migraphx_here() else _TensorRT


def _engine_of(
    context: NodeContext,
    model: ImageModelDescriptor,
    device: torch.device,
    gpu_index: int,
    tile_size: TileSize,
) -> TensorRTEngine | MIGraphXProgram | str:
    """The model's engine or program for this Tile Size, or why it has none. Stop raises
    Aborted."""
    tile = _engine_tile(model, tile_size, gpu_index)
    whole = _whole_images(model, tile_size, gpu_index)
    where = (device, gpu_index, tile, whole)
    with _lock:
        kept = _outcomes.get(model)
    if kept is not None and kept[0] == where:
        return kept[1]

    route = _route()
    recipe = _recipe(model)
    if recipe is None:
        # The node hides the setting for these, so this is no surprise worth a warning;
        # except a DRFT, whose type does not tell a legacy one.
        logger.info(
            "%s: this %s model has no automatic %s; running on PyTorch.",
            route.input,
            model.architecture.name,
            route.made,
        )
        reason = ""
        if model.architecture.name == "DRFT":
            reason = "only legacy DRFT models (LayerNorm, not i-LN) have one."
        return _no_engine(context, model, where, reason)
    if tile < tile_size:
        context.warn(
            f"{route.input}: this {model.architecture.name} model's {route.made} stops at"
            f" {tile} px, so it runs in tiles of that size, not {tile_size} px: above that"
            " its memory can fill the GPU."
        )
    try:
        reason = _no_engine_here(device)
        if reason is not None:
            return _no_engine(context, model, where, reason)
        folder = route.folder(context)
        folder.mkdir(parents=True, exist_ok=True)
        # What PyTorch keeps cached on the GPU is the build's and the session's now.
        safe_cuda_cache_empty()
        built = False

        def engine_of_size(
            size: int, whole_image: bool
        ) -> tuple[str, TensorRTEngine | MIGraphXProgram]:
            def build() -> TensorRTEngine | MIGraphXProgram:
                nonlocal built
                built = True
                return route.build(context, model, size, whole_image)

            key = route.key(context, model, recipe, gpu_index, size, whole_image)
            return key, route.cached(key, build, str(folder))

        try:
            key, engine = engine_of_size(tile, whole)
        except Aborted:
            raise
        except Exception:
            if not whole:
                raise
            # An engine for whole images that does not build (a variant not measured,
            # or the GPU short of memory at the time) leaves this process's images to
            # DEFAULT_TILE's engine; the next process tries again.
            logger.exception(
                "%s: no %s px engine for whole images; using one of %s px",
                route.input,
                tile,
                DEFAULT_TILE,
            )
            key, engine = engine_of_size(DEFAULT_TILE, False)
    except Aborted:
        raise
    except Exception as e:
        logger.exception("%s: no %s for this model", route.input, route.made)
        return _no_engine(context, model, where, _described(e))
    with _lock:
        _outcomes[model] = (where, engine)
        if not built and key not in _rebuilt:
            _unproven[model] = (key, str(folder))
    return engine


def release_engine(node_id: NodeId) -> None:
    """Free a session this node holds on an automatic engine or program, if any: its GPU
    memory is PyTorch's again. release_node goes through every session cache, MIGraphX's
    too."""
    with _lock:
        held = node_id in _engine_nodes
        _engine_nodes.discard(node_id)
    if held:
        from ..tensorrt.cache import release_node

        release_node(node_id)


def _is_finite(img: np.ndarray) -> bool:
    # NaN propagates through min and max: two passes and no temporary array
    return math.isfinite(img.min()) and math.isfinite(img.max())


def _checked(
    model: ImageModelDescriptor, made: str, img: np.ndarray
) -> Callable[[np.ndarray], None]:
    """What raises where PyTorch would on invalid values from the finite `img`."""

    def check(raw: np.ndarray) -> None:
        if not _is_finite(raw) and _is_finite(img):
            raise RuntimeError(
                f"The {made} of this {model.architecture.name} model returned invalid"
                " values (NaN or infinity), which would be saved as black or white"
                " pixels."
            )

    return check


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
        output, tiles = engine_upscale(
            context,
            img,
            engine,
            False,
            False,
            before_clip=_checked(model, "engine", img),
        )
        if report is not None:
            report.text = tiles
        return output

    return run


def _program_op(
    context: NodeContext,
    model: ImageModelDescriptor,
    program: MIGraphXProgram,
    report: TileReport | None = None,
) -> ImageOp:
    """MIGraphX's own Upscale Image, on `context`'s GPU, as `_engine_op`."""
    from packages.chaiNNer_migraphx.migraphx.processing.upscale_image import (
        program_upscale,
    )

    def run(img: np.ndarray) -> np.ndarray:
        output, tiles = program_upscale(
            context, img, program, False, before_clip=_checked(model, "program", img)
        )
        if report is not None:
            report.text = tiles
        return output

    return run


def _drop_refused_file(model: ImageModelDescriptor, error: Exception) -> bool:
    """Whether `error`, which PyTorch did not repeat on the same image, is the route refusing
    the engine or program read from its file: then the file goes, and the next image builds.
    Never run here, and not for want of memory. Once per engine in a process: one built anew
    and refused too says nothing of the file."""
    route = _route()
    short_of_memory = route.short_of_memory(error)
    with _lock:
        unproven = _unproven.pop(model, None)
        if unproven is None or short_of_memory:
            return False
        _rebuilt.add(unproven[0])
        _outcomes.pop(model, None)
    route.forget(*unproven)
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
    """`upscale_with` the model's engine or program, or `pytorch` when it has none or fails.
    The whole image starts over: an image with transparency must not mix the two (its two
    passes are compared). `report` takes the engine's tiles; `pytorch` records its own."""
    route = _route()
    engine = _engine_of(context, model, device, gpu_index, tile_size)
    if isinstance(engine, str):
        release_engine(context.node_id)
        return upscale_with(pytorch)
    with _lock:
        _engine_nodes.add(context.node_id)
    try:
        output = upscale_with(route.op(context, model, engine, report))
    except Aborted:
        raise
    except Exception as e:
        logger.exception("%s: the %s failed on this image", route.input, route.made)
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
            f"Upscale Image ran this image on PyTorch: {route.maker} could not start this"
            f" {model.architecture.name} model's kept {route.made}, so it is built again"
            f" for the next image. {_described(error)}"
        )
    else:
        tile = _engine_tile(model, tile_size, gpu_index)
        where = (device, gpu_index, tile, _whole_images(model, tile_size, gpu_index))
        _no_engine(context, model, where, _described(error))
    return output
