"""AI Upscale inside a VapourSynth script: each frame of an RGBS or GRAYS clip goes
through chaiNNer's own upscalers (PyTorch, ONNX, TensorRT engines) in the script's
process.

A frame becomes the float32 BGR image (HxW for gray) that chaiNNer's Upscale Image
nodes take, and the nodes' own processing upscales it, so every frame comes out as the
node would make it from the same image. chaiNNer's modules (and torch, onnxruntime or
TensorRT) are imported when a script first loads a model of that backend."""

from __future__ import annotations

import importlib
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# VapourSynth by name: the type checker runs on chaiNNer's development runtime, which
# has no VapourSynth, and would take this package's parent folder (also named
# vapoursynth) for it; VapourSynth's own stubs leave the plugin namespaces untyped.
vs: Any = importlib.import_module("vapoursynth")

# The AI Upscale node's backends, with the model files each one runs.
BACKENDS: dict[str, tuple[str, ...]] = {
    "PyTorch": (".pth", ".pt", ".ckpt", ".safetensors"),
    "ONNX": (".onnx",),
    "TensorRT": (".engine", ".trt", ".plan"),
}
PRECISIONS = ("fp32", "fp16")

# chaiNNer's default GPU setting for TensorRT: the first CUDA device.
GPU_INDEX = 0


def gpu_index() -> int:
    """chaiNNer's default GPU setting for PyTorch and ONNX: CUDA's first device; where
    no NVIDIA GPU is detected, HIP's strongest AMD GPU (a Ryzen's graphics often come
    first in HIP's order), which PyTorch on ROCm and MIGraphX take."""
    from gpu import amd, nvidia

    return amd.default_index if amd.is_available and not nvidia.is_available else 0


@dataclass(frozen=True)
class Upscaler:
    """A loaded model. `run(image, tile)` upscales one float32 BGR image (HxW for gray)
    with a chaiNNer tile size value, by the scale in each direction."""

    scale_width: int
    scale_height: int
    input_channels: int
    output_channels: int
    run: Callable[[np.ndarray, int], np.ndarray]


# Models by (path, backend, precision), each loaded once per process; the precision is
# "" for ONNX and TensorRT. vsview's Reload Script imports this module again, so a
# reload loads its models again.
loaded: dict[tuple[str, str, str], Upscaler] = {}
_loading = threading.Lock()
# One model call at a time. VapourSynth runs one ModifyFrame's selector for one frame
# at a time, but the selectors of two AI Upscale clips in one script run on two worker
# threads at once, and the clips may share a model (a TensorRT session serves one
# image at a time).
_model_call = threading.Lock()


def _pytorch(path: str, precision: str) -> Upscaler:
    import torch
    from spandrel import ImageModelDescriptor

    from api import Progress
    from nodes.impl.upscale.auto_split_tiles import TileSize
    from nodes.impl.upscale.basic_upscale import UpscaleInfo, basic_upscale
    from packages.chaiNNer_pytorch.pytorch.io.load_model import load_model
    from packages.chaiNNer_pytorch.pytorch.processing.upscale_image import (
        upscale as node_upscale,
    )
    from packages.chaiNNer_pytorch.settings import PyTorchSettings

    # chaiNNer's CPU mode when CUDA (HIP in a ROCm build of PyTorch) has no GPU
    # (FP16 then falls back to FP32).
    settings = PyTorchSettings(
        use_cpu=not torch.cuda.is_available(),
        use_fp16=precision == "fp16",
        gpu_index=gpu_index(),
        budget_limit=0,
    )
    model = load_model(Path(path), settings)
    if not isinstance(model, ImageModelDescriptor):
        raise ValueError(
            f"{Path(path).name} is an inpainting model; AI Upscale runs upscaling and"
            " restoration models."
        )
    info = UpscaleInfo(
        in_nc=model.input_channels, out_nc=model.output_channels, scale=model.scale
    )
    progress = Progress.noop_progress()

    def run(img: np.ndarray, tile: int) -> np.ndarray:
        # PyTorch's Upscale Image node at the model's own scale, without padding.
        return basic_upscale(
            img,
            lambda i: node_upscale(i, model, TileSize(tile), settings, progress),
            upscale_info=info,
            scale=model.scale,
            separate_alpha=False,
            clip=False,  # pytorch_auto_split clips
        )

    return Upscaler(
        model.scale, model.scale, model.input_channels, model.output_channels, run
    )


def _onnx(path: str) -> Upscaler:
    from gpu import cuda_pci_addresses
    from nodes.impl.onnx.model import OnnxGeneric
    from nodes.impl.onnx.session import (
        get_input_shape,
        get_onnx_session,
        get_output_shape,
    )
    from nodes.impl.upscale.auto_split_tiles import ESTIMATE, TILE_SIZE_256, TileSize
    from packages.chaiNNer_onnx.onnx.io.load_model import load_model_node
    from packages.chaiNNer_onnx.onnx.processing.upscale_image import (
        upscale_with_session,
    )
    from packages.chaiNNer_onnx.settings import default_provider, provider_device

    model = load_model_node(Path(path))[0]
    if not isinstance(model, OnnxGeneric):
        raise ValueError(
            f"{Path(path).name} is a background removal model; AI Upscale runs"
            " upscaling and restoration models."
        )
    scale_width, scale_height = model.info.scale_width, model.info.scale_height
    if scale_width is None or scale_height is None:
        raise ValueError(
            f"chaiNNer cannot tell the scale of {Path(path).name}, so AI Upscale"
            " cannot size its clip."
        )
    # chaiNNer's default execution provider (MIGraphX only where there is an AMD
    # GPU, DirectML on Windows there), on the device it numbers the GPU by; the CPU
    # when CUDA has no GPU. MIGraphX runs the model in its own precision, without the
    # ONNX settings' program cache.
    provider = default_provider
    if provider == "CUDAExecutionProvider" and not cuda_pci_addresses():
        provider = "CPUExecutionProvider"
    session = get_onnx_session(
        model, provider_device(provider, gpu_index()), provider, False
    )

    def run(img: np.ndarray, tile: int) -> np.ndarray:
        # ONNX has no tile estimate: Auto is the ONNX node's default tile size.
        size = TILE_SIZE_256 if tile == ESTIMATE else TileSize(tile)
        return upscale_with_session(img, model, session, size, False)

    return Upscaler(
        scale_width,
        scale_height,
        get_input_shape(session)[1],
        get_output_shape(session)[1],
        run,
    )


def _require_gpu() -> None:
    from gpu import cuda_pci_addresses

    if not cuda_pci_addresses():
        raise RuntimeError(
            "TensorRT engines run on an NVIDIA GPU, and CUDA finds none here (or"
            " CUDA_VISIBLE_DEVICES hides it). Choose the PyTorch or ONNX backend"
            " to run on the CPU."
        )


def _tensorrt(path: str) -> Upscaler:
    _require_gpu()

    from nodes.impl.compressed_file import read_file
    from nodes.impl.tensorrt.engine_info import deserialize_engine, read_engine_info
    from nodes.impl.tensorrt.inference import TensorRTSession
    from nodes.impl.tensorrt.model import TensorRTEngine
    from nodes.impl.tensorrt.tiling import tiled_upscale
    from nodes.impl.upscale.convenient_upscale import convenient_upscale

    # As Load Engine and TensorRT's Upscale Image node do.
    engine_bytes = read_file(path)
    deserialized = deserialize_engine(engine_bytes, GPU_INDEX)
    try:
        engine = TensorRTEngine(
            engine_bytes, read_engine_info(deserialized), deserialized
        )
    except BaseException:
        deserialized.release()
        raise
    scale = engine.scale
    if scale is None:
        raise ValueError("The TensorRT engine's upscale factor is unknown.")
    session = TensorRTSession(engine, GPU_INDEX)

    def run(img: np.ndarray, tile: int) -> np.ndarray:
        # The engine's optimization profile sets the tiles, not `tile`.
        return convenient_upscale(
            img,
            engine.input_channels,
            engine.output_channels,
            lambda i: tiled_upscale(i, scale, session.bounds, session),
            False,
        )

    return Upscaler(scale, scale, engine.input_channels, engine.output_channels, run)


def load(model: str, backend: str, precision: str) -> Upscaler:
    """A model file on one backend; `precision` applies to PyTorch only (ONNX models
    and TensorRT engines carry their own)."""
    if backend == "PyTorch":
        return _pytorch(model, precision)
    if backend == "ONNX":
        return _onnx(model)
    return _tensorrt(model)


def _model_key(model: str, backend: str, precision: str) -> tuple[str, str, str]:
    """Checks a model file for a backend; its key in `loaded`."""
    suffixes = BACKENDS.get(backend)
    if suffixes is None:
        raise ValueError(
            f"Unknown backend {backend!r}: choose one of {', '.join(BACKENDS)}."
        )
    if precision not in PRECISIONS:
        raise ValueError(
            f"Unknown precision {precision!r}: choose one of {', '.join(PRECISIONS)}."
        )
    name = Path(model).name
    suffix = Path(model).suffix.lower()
    if suffix not in suffixes:
        owner = next((b for b, s in BACKENDS.items() if suffix in s), None)
        hint = (
            f"choose the {owner} backend"
            if owner
            else f"the {backend} backend runs {', '.join(suffixes)} files"
        )
        raise ValueError(f"{name} is not a {backend} model file: {hint}.")
    if not os.path.isfile(model):
        raise FileNotFoundError(f"The model file {model} does not exist.")
    # ONNX models and TensorRT engines run in their own precision: nodes that differ
    # only in their (hidden) precision input share one session.
    return (model, backend, precision if backend == "PyTorch" else "")


def _upscaler(model: str, backend: str, precision: str) -> Upscaler:
    key = _model_key(model, backend, precision)
    with _loading:
        upscaler = loaded.get(key)
        if upscaler is None:
            upscaler = load(*key)
            loaded[key] = upscaler
    return upscaler


def _image(frame: Any) -> np.ndarray:
    """A frame as the float32 BGR image (HxW for gray) chaiNNer's nodes take."""
    if frame.format.color_family == vs.GRAY:
        return np.array(frame[0])
    # VapourSynth's RGB planes are R, G, B; one copy interleaves them.
    return np.stack(
        (np.asarray(frame[2]), np.asarray(frame[1]), np.asarray(frame[0])), axis=2
    )


def _write(img: np.ndarray, frame: Any) -> None:
    height, width = img.shape[:2]
    if (width, height) != (frame.width, frame.height):
        raise RuntimeError(
            f"The model made a {width}x{height} image where AI Upscale expected"
            f" {frame.width}x{frame.height}."
        )
    if frame.format.num_planes == 1:
        np.copyto(np.asarray(frame[0]), img.reshape(height, width))
        return
    for plane, channel in enumerate((2, 1, 0)):
        np.copyto(np.asarray(frame[plane]), img[:, :, channel])


def _runner_clip(clip: Any, model: str, plugin: Path, streams: int) -> Any:
    """AI Upscale on the native TensorRT runner's VapourSynth filter (`plugin`, see
    nodes/impl/tensorrt/runner.py) with `streams` execution contexts: frames are
    converted, tiled and blended on the GPU, with no Python per frame, and come out as
    the bridge makes them (clipped to 0-1 before and after the engine, properties and
    _Matrix as below)."""
    _require_gpu()

    from nodes.impl.compressed_file import read_file
    from nodes.impl.tensorrt.runner import load_tensorrt

    load_tensorrt()  # the plugin's delay-loaded TensorRT imports find these DLLs
    core = vs.core
    if not hasattr(core, "chainner_trt"):
        core.std.LoadPlugin(str(plugin))
    upscaled = core.chainner_trt.Upscale(
        clip, engine=read_file(model), streams=streams, device=GPU_INDEX
    )
    core.std.SetVideoCache(upscaled, mode=1)  # as `upscale` caches its clip
    return upscaled


def upscale(
    clip: Any, model: str, backend: str, precision: str, tile: int, streams: int = 1
) -> Any:
    """The VideoNode `clip` (RGBS or GRAYS) upscaled by the model file `model` on
    `backend` ("PyTorch", "ONNX" or "TensorRT"), as a VideoNode. `precision` ("fp32" or
    "fp16") is PyTorch's; `tile` is chaiNNer's tile size value: 0 Auto, -1 no tiling,
    -2 maximum, else the largest tile in pixels. `streams` (1 to 4) is the TensorRT
    runner's: frames upscaled at once, each holding its own buffers on the GPU; the
    Python bridge (CHAINNER_C_TRT_RUNNER=0, or a build without the runner) ignores it.

    The model gets each frame clipped to [0, 1], as chaiNNer's images always are (a
    float clip need not be), so a frame comes out as the Upscale Image node makes it
    from the clipped image. Frame properties carry over; when the model changes the
    colour family (a 1 -> 3 or 3 -> 1 channel model), _Matrix becomes the new family's:
    0 (RGB) on an RGB result, none on a gray one."""
    fmt = clip.format
    if fmt.id not in (vs.RGBS, vs.GRAYS):
        # A clip whose format varies has the format id 0 (named "None").
        name = fmt.name if fmt.id else "a format that changes between frames"
        raise ValueError(
            "AI Upscale takes 32-bit float RGB or gray clips (RGBS or GRAYS), not"
            f" {name}. Convert the clip first."
        )
    if not clip.width or not clip.height:
        raise ValueError(
            "AI Upscale needs a clip whose frames all have one size. Resize the clip"
            " first."
        )
    if tile < -2:
        raise ValueError(
            f"{tile} is not a tile size: use 0 (Auto), -1 (no tiling), -2 (maximum) or"
            " a size in pixels."
        )
    if not 1 <= streams <= 4:  # the AI Upscale node's Streams range
        raise ValueError(f"{streams} is not a stream count: use 1 to 4.")
    if backend == "TensorRT":
        from nodes.impl.tensorrt.runner import runner_plugin

        plugin = runner_plugin()
        if plugin is not None:
            _model_key(model, backend, precision)
            return _runner_clip(clip, model, plugin, streams)
    upscaler = _upscaler(model, backend, precision)
    # As the Upscale Image nodes: the clip's channels, unless the model changes them
    # (a gray clip goes through an RGB model as BGR and comes back as gray).
    channels = 1 if fmt.color_family == vs.GRAY else 3
    if upscaler.input_channels != upscaler.output_channels:
        channels = upscaler.output_channels
    if channels not in (1, 3):
        raise ValueError(
            f"{Path(model).name} makes {channels}-channel images; AI Upscale returns"
            " RGB or gray clips."
        )
    out = clip.std.BlankClip(
        width=clip.width * upscaler.scale_width,
        height=clip.height * upscaler.scale_height,
        format=vs.GRAYS if channels == 1 else vs.RGBS,
        keep=True,
    )
    core = vs.core
    # A YUV _Matrix on an RGB clip, or RGB's on a gray one, fails the next resize.
    family_changed = out.format.color_family != fmt.color_family

    def frame(n: int, f: Any) -> Any:
        img = _image(f)  # a copy of the frame's planes, clipped in place
        np.clip(img, 0.0, 1.0, out=img)
        with _model_call:
            upscaled = upscaler.run(img, tile)
        dst = core.create_video_frame(out.format, out.width, out.height)
        _write(upscaled, dst)
        dst.props.update(f.props)
        if family_changed:
            if channels == 3:
                dst.props["_Matrix"] = 0
            else:
                dst.props.pop("_Matrix", None)
        return dst

    upscaled_clip = out.std.ModifyFrame(clips=clip, selector=frame)
    # Always cache the frames (within core.max_cache_size): in the automatic mode a
    # clip with one consumer keeps none, and every frame property vs-jetpack reads
    # while the script is evaluated would run the model again.
    core.std.SetVideoCache(upscaled_clip, mode=1)
    return upscaled_clip
