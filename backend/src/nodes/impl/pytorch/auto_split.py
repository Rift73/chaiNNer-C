from __future__ import annotations

import gc
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch
from spandrel import ImageModelDescriptor

from api import Aborted, Progress

from ...utils.utils import Region
from ..native_framework_images import planar_to_interleaved_checked
from ..native_tensors import cast_numpy
from ..upscale.auto_split import Split, Tiler, auto_split
from ..upscale.bounded_tiler import BoundedTiler
from ..upscale.tile_report import TileReport
from ..upscale.white_pass import current_white_pass
from .offload import offloaded
from .utils import (
    PINNED_IMAGE_LIMIT,
    host_image,
    is_finite,
    is_out_of_memory,
    on_device,
    safe_cuda_cache_empty,
    to_host,
)


class _Stopped(Exception):
    """Stop, seen between a model's blocks. The forward catches it, so its frames (the
    activations) are freed before the node raises Aborted."""


# The progress of the forward this thread runs (another run may use the same model).
_forward = threading.local()


def _stop_between_blocks(_module: torch.nn.Module, _args: object) -> None:
    progress: Progress | None = getattr(_forward, "progress", None)
    if progress is not None and progress.aborted:
        raise _Stopped


@contextmanager
def _stoppable(module: torch.nn.Module) -> Iterator[None]:
    """Checks for Stop at each block of `module` (each module with children). Without
    it a tile's forward runs to its end after Stop, keeping its memory: for minutes
    when the GPU's memory overflows into shared memory."""
    handles = [
        m.register_forward_pre_hook(_stop_between_blocks)
        for m in module.modules()
        if next(m.children(), None) is not None
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _forward_or_stopped(
    model: ImageModelDescriptor[torch.nn.Module],
    input_tensor: torch.Tensor,
    progress: Progress,
) -> torch.Tensor | None:
    """The model's output, or None when Stop interrupted the forward (whose frames are
    freed by the time this returns)."""
    _forward.progress = progress
    try:
        return model(input_tensor)
    except _Stopped:
        return None
    finally:
        _forward.progress = None


def _into_standard_image_form(t: torch.Tensor) -> torch.Tensor:
    if len(t.shape) == 2:
        # (H, W)
        return t
    elif len(t.shape) == 3:
        # (C, H, W) -> (H, W, C)
        return t.permute(1, 2, 0)
    elif len(t.shape) == 4:
        # (1, C, H, W) -> (H, W, C)
        return t.squeeze(0).permute(1, 2, 0)
    else:
        raise ValueError("Unsupported output tensor shape")


def _into_batched_form(t: torch.Tensor) -> torch.Tensor:
    if len(t.shape) == 2:
        # (H, W) -> (1, 1, H, W)
        return t.unsqueeze(0).unsqueeze(0)
    elif len(t.shape) == 3:
        # (H, W, C) -> (1, C, H, W)
        return t.permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError("Unsupported input tensor shape")


def _rgb_to_bgr(t: torch.Tensor) -> torch.Tensor:
    if len(t.shape) == 3 and t.shape[2] == 3:
        # (H, W, C) RGB -> BGR
        return t.flip(2)
    elif len(t.shape) == 3 and t.shape[2] == 4:
        # (H, W, C) RGBA -> BGRA
        return torch.cat((t[:, :, 2:3], t[:, :, 1:2], t[:, :, 0:1], t[:, :, 3:4]), 2)
    else:
        return t


def _into_tensor(
    img: np.ndarray, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    img = np.ascontiguousarray(img)
    writeable = img.flags.writeable
    try:
        if not writeable and device == torch.device("cpu"):
            img = np.copy(img)
        else:
            # since we are going to copy the image to the GPU, we can skip the copy here
            try:
                img.flags.writeable = True
            except Exception:
                # Some arrays cannot be made writeable, and we need to copy them
                img = np.copy(img)
        input_tensor = torch.from_numpy(img).to(device, dtype)
        return input_tensor
    finally:
        img.flags.writeable = writeable


# Architectures measured at least 5 % slower on the CPU in FP32 with a channels-last
# input (default networks at 128 px): they keep NCHW.
_NCHW_ON_CPU = frozenset({"MixDehazeNet"})
# Architectures measured faster on CUDA in FP16 with a contiguous channels-last input:
# the batched form is channels-last in memory, but its batch stride (3, from the
# unsqueeze of a permuted view) makes cuDNN take it for NCHW: the first convolution
# writes NCHW, and cuDNN converts to NHWC and back around each FP16 convolution after it
# (a 2x Compact at 960x540: 72 of 112 kernels move layouts, 18 of 76 from a contiguous
# input). On an RTX 5090, the route per image through the node's output at 960x540 and
# 1920x1080: a 2x WSR Compact 11.71 -> 10.02 and 51.23 -> 46.30 ms, a 4x WSR Default
# (Halo) 28.40 -> 26.38 and 132.81 -> 128.61 ms, the same peak memory; a 2x Compact
# 9.47 -> 6.89 and 38.71 -> 28.30 ms (peak 258 -> 131 MiB at 960x540), a 2x SPAN 9.72
# -> 7.92 and 47.82 -> 36.77 ms, a 2x MoSR 108.90 -> 87.15 and 448.75 -> 354.36 ms; in
# 512 px tiles a 1920x1080 image 10 % faster for Compact and SPAN. RMSE against FP32 the
# same. FP32 is not measured and keeps NCHW; OmniSR copies its input itself (no change).
_CHANNELS_LAST_ON_CUDA = frozenset({"WSR", "Compact", "SPAN", "MoSR"})


def _channels_last_input(
    model: ImageModelDescriptor[torch.nn.Module],
    device: torch.device,
    dtype: torch.dtype,
    use_bf16: bool,
) -> bool:
    """Whether the forward gets its input in channels-last order: on the CPU in FP32
    and in BF16, and on CUDA in FP16 for _CHANNELS_LAST_ON_CUDA. oneDNN's convolutions
    on NCHW reorder each layer's input and output; in channels-last order they run on
    them directly (a 4x ESRGAN 1.5x faster, in FP32 as in BF16). Other GPU models and
    precisions keep NCHW."""
    architecture = model.architecture.id
    return (
        use_bf16
        or (
            device.type == "cpu"
            and dtype == torch.float32
            and architecture not in _NCHW_ON_CPU
        )
        or (
            device.type == "cuda"
            and dtype == torch.float16
            and architecture in _CHANNELS_LAST_ON_CUDA
        )
    )


def _interleaved(t: torch.Tensor) -> tuple[np.ndarray, bool] | None:
    """A (1, C, H, W) or (C, H, W) float16 or float32 output, rows contiguous, as
    the BGR(A) float32 image the device permute, flip and CPU cast make, interleaved
    from its planes on the host in one pass, and whether it is finite; else None.
    A finite image is the cast's bit for bit; NaN can differ (the cast quiets a
    signalling half), so a caller converts a non-finite output as before."""
    if t.ndim == 4 and t.shape[0] == 1:
        t = t[0]
    if t.ndim != 3 or t.dtype not in (torch.float16, torch.float32) or t.stride(2) != 1:
        return None
    planes = t.detach().cpu().numpy()
    # Rows that overlap (an output expanded along its height has row stride 0)
    row_stride = planes.strides[1]
    if row_stride <= 0 or row_stride < planes.shape[2] * planes.itemsize:
        return None
    return planar_to_interleaved_checked(
        planes, swap_red_blue=planes.shape[0] in (3, 4)
    )


# The devices whose one-tile outputs _device_image makes where they are
_DEVICE_IMAGE_TYPES = frozenset({"cuda"})


def _device_image(t: torch.Tensor) -> np.ndarray | None:
    """A (1, 3, H, W) or (3, H, W) float16 or float32 output on a GPU, finite and of at
    most PINNED_IMAGE_LIMIT bytes as float32, as _interleaved's BGR float32 image made on
    the device (layout, channel order and cast in one pass, no clamp: spandrel's call
    clamped it) and copied into page-locked memory: the host makes no pass over it.
    Bit for bit _interleaved's image. Else None: the caller converts it as before."""
    if t.ndim == 4 and t.shape[0] == 1:
        t = t[0]
    if (
        t.device.type not in _DEVICE_IMAGE_TYPES
        or t.ndim != 3
        or t.shape[0] != 3
        or t.dtype not in (torch.float16, torch.float32)
        or t.numel() * 4 > PINNED_IMAGE_LIMIT
        or not bool(torch.isfinite(t).all())
    ):
        return None
    return host_image(to_host(t, clamp=False))


@torch.inference_mode()
def pytorch_auto_split(
    img: np.ndarray,
    model: ImageModelDescriptor[torch.nn.Module],
    device: torch.device,
    use_fp16: bool,
    tiler: Tiler,
    progress: Progress,
    use_bf16: bool = False,
    report: TileReport | None = None,
    expected_bytes_per_pixel: float | None = None,
) -> np.ndarray:
    """`use_bf16` runs an FP32 model's forward in BF16 on the CPU (autocast: the
    weights stay FP32). `expected_bytes_per_pixel` is the model's whole-image estimate,
    for the activation offload (offloaded)."""
    dtype = torch.float16 if use_fp16 else torch.float32
    model = on_device(model, device, dtype)
    channels_last = _channels_last_input(model, device, dtype, use_bf16)
    # Other devices' float16 casts were not compared with the host's.
    tiles = current_white_pass() if not use_fp16 or device.type == "cuda" else None

    img_h, img_w = img.shape[:2]

    def model_input(img: np.ndarray) -> np.ndarray:
        return cast_numpy(img, np.dtype(np.float16)) if use_fp16 else img

    def upscale(img: np.ndarray, region: Region):
        progress.check_aborted()
        if progress.paused:
            # Only clear CUDA cache before pausing, skip gc.collect() (too slow)
            safe_cuda_cache_empty()
            progress.suspend()

        if tiles is not None:
            reused = tiles.reuse(region.x, region.y, img)
            if reused is not None:
                return reused

        input_tensor = None
        try:
            # convert to tensor
            input_tensor = _into_tensor(img, device, dtype)
            input_tensor = _rgb_to_bgr(input_tensor)
            input_tensor = _into_batched_form(input_tensor)
            if channels_last:
                # A copy: the batched form is channels-last in memory, but its batch
                # stride makes PyTorch take it for NCHW.
                input_tensor = input_tensor.clone(memory_format=torch.channels_last)

            # inference; activations past the GPU's memory go to RAM, not the driver's overflow
            with offloaded(
                model.model, input_tensor, expected_bytes_per_pixel
            ), torch.autocast("cpu", torch.bfloat16, enabled=use_bf16):
                output_tensor = _forward_or_stopped(model, input_tensor, progress)
            if output_tensor is None:
                # The kernels already queued end inside the node, so the memory is
                # free once the node ends, and the release does not wait for them.
                input_tensor = None
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                raise Aborted()

            # convert back to numpy: an image that runs as one tile is made on the GPU,
            # where it is; the tiles of a split image keep the host's interleave
            whole = (region.width, region.height) == (img_w, img_h)
            on_gpu = _device_image(output_tensor) if whole else None
            interleaved = (
                (on_gpu, True) if on_gpu is not None else _interleaved(output_tensor)
            )
            if interleaved is not None and interleaved[1]:
                result, finite = interleaved
            else:
                # NaN or infinity, or an output the interleave does not read
                output_tensor = _into_standard_image_form(output_tensor)
                output_tensor = _rgb_to_bgr(output_tensor)
                result = output_tensor.detach().cpu().float().numpy()
                finite = interleaved is None and is_finite(result)
        except (RuntimeError, MemoryError) as e:
            if not is_out_of_memory(e):
                raise
            input_tensor = None
            gc.collect()
            safe_cuda_cache_empty()
            return Split()

        # Saving turns NaN into black pixels. NaN already in the input passes through;
        # NaN the model made from a finite tile is an error, raised at the first such
        # tile instead of after the whole image.
        if not finite and is_finite(img):
            message = (
                f"The {model.architecture.name} model returned invalid values (NaN or"
                " infinity), which would be saved as black pixels."
            )
            if use_fp16:
                message += (
                    " Models can overflow in FP16: turn off 'Use FP16 Mode' in the"
                    " PyTorch settings to run this model in FP32."
                )
            if use_bf16:
                message += (
                    " Models can lose precision in BF16: turn off 'Use BF16 Mode (CPU)'"
                    " in the PyTorch settings to run this model in FP32."
                )
            raise RuntimeError(message)
        if tiles is not None:
            tiles.keep(region.x, region.y, img, result, model_input)
        return result

    with _stoppable(model.model):
        return auto_split(
            img,
            upscale if report is None else report.recording(img, upscale),
            BoundedTiler(tiler),
            progress=progress,
        )
