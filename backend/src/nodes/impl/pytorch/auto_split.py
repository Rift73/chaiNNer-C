from __future__ import annotations

import gc
import math
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch
from spandrel import ImageModelDescriptor

from api import Aborted, Progress

from ..native_framework_images import planar_to_interleaved_checked
from ..upscale.auto_split import Split, Tiler, auto_split
from ..upscale.bounded_tiler import BoundedTiler
from .offload import offloaded
from .utils import safe_cuda_cache_empty


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


def _is_finite(img: np.ndarray) -> bool:
    # NaN propagates through min and max: two passes and no temporary array
    return math.isfinite(img.min()) and math.isfinite(img.max())


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


@torch.inference_mode()
def pytorch_auto_split(
    img: np.ndarray,
    model: ImageModelDescriptor[torch.nn.Module],
    device: torch.device,
    use_fp16: bool,
    tiler: Tiler,
    progress: Progress,
) -> np.ndarray:
    dtype = torch.float16 if use_fp16 else torch.float32
    if model.dtype != dtype or model.device != device:
        model = model.to(device, dtype)

    def upscale(img: np.ndarray, _: object):
        progress.check_aborted()
        if progress.paused:
            # Only clear CUDA cache before pausing, skip gc.collect() (too slow)
            safe_cuda_cache_empty()
            progress.suspend()

        input_tensor = None
        try:
            # convert to tensor
            input_tensor = _into_tensor(img, device, dtype)
            input_tensor = _rgb_to_bgr(input_tensor)
            input_tensor = _into_batched_form(input_tensor)

            # inference; activations past the GPU's memory go to RAM, not the driver's overflow
            with offloaded(model.model, input_tensor):
                output_tensor = _forward_or_stopped(model, input_tensor, progress)
            if output_tensor is None:
                # The kernels already queued end inside the node, so the memory is
                # free once the node ends, and the release does not wait for them.
                input_tensor = None
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                raise Aborted()

            # convert back to numpy
            interleaved = _interleaved(output_tensor)
            if interleaved is not None and interleaved[1]:
                result, finite = interleaved
            else:
                # NaN or infinity, or an output the interleave does not read
                output_tensor = _into_standard_image_form(output_tensor)
                output_tensor = _rgb_to_bgr(output_tensor)
                result = output_tensor.detach().cpu().detach().float().numpy()
                finite = interleaved is None and _is_finite(result)
        except (RuntimeError, MemoryError) as e:
            # Out of memory: the device's or the host's (torch's RuntimeError, or
            # NumPy's MemoryError for the converted output)
            if (
                isinstance(e, torch.OutOfMemoryError)
                or isinstance(e, MemoryError)
                or "allocate" in str(e).lower()
                or "out of memory" in str(e).lower()
                or "ALLOC_FAILED" in str(e)
            ):
                # Collect garbage (clear VRAM)
                if input_tensor is not None:
                    try:
                        input_tensor.detach().cpu()
                    except Exception:
                        pass
                    del input_tensor
                gc.collect()
                safe_cuda_cache_empty()
                return Split()
            else:
                # Re-raise the exception if not an OOM error
                raise

        # Saving turns NaN into black pixels. NaN already in the input passes through;
        # NaN the model made from a finite tile is an error, raised at the first such
        # tile instead of after the whole image.
        if not finite and _is_finite(img):
            message = (
                f"The {model.architecture.name} model returned invalid values (NaN or"
                " infinity), which would be saved as black pixels."
            )
            if use_fp16:
                message += (
                    " Models can overflow in FP16: turn off 'Use FP16 Mode' in the"
                    " PyTorch settings to run this model in FP32."
                )
            raise RuntimeError(message)
        return result

    with _stoppable(model.model):
        return auto_split(img, upscale, BoundedTiler(tiler))
