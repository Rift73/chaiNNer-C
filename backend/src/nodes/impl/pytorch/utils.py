from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from threading import Lock
from typing import TypeVar, cast
from weakref import WeakKeyDictionary

import numpy as np
import torch
from spandrel import ImageModelDescriptor, ModelDescriptor
from torch import Tensor

from ..image_utils import MAX_VALUES_BY_DTYPE, as_3d

_Descriptor = TypeVar("_Descriptor", bound=ModelDescriptor)

# A model's copy on another device or in another precision while the model lives (one
# per model: a copy on a new device or in a new precision replaces it).
_copies: WeakKeyDictionary[ModelDescriptor, ModelDescriptor] = WeakKeyDictionary()
_copies_lock = Lock()
# Set once RAM was pinned through PyTorch's host allocator (the VRAM offload's moved
# activations, the temporal routes' frames and images): the allocator keeps freed pinned
# blocks locked until release_pinned_memory.
_pinned = False
# The largest image, in bytes, that a single-image route hands on in page-locked memory
# (auto_split._device_image): a node or cache downstream holds it locked while it keeps
# the image, and the host allocator rounds a block up to a power of two, so a larger one
# takes a pageable copy. 256 MiB holds a 3840x2160 output in FP32 with room to spare.
PINNED_IMAGE_LIMIT = 256 * 2**20


def on_device(
    model: _Descriptor, device: torch.device, dtype: torch.dtype
) -> _Descriptor:
    """`model` on `device` in `dtype`: `model` itself when it is, else its copy there.
    `model` is not changed: nodes share it (Save Model saves its weights, Convert To
    ONNX exports them, another upscale may run it in another precision). A copy no
    longer asked for is dropped (a copy in use is held by its caller). A CUDA device
    without an index is the current one, as tensors on it report it: its copy is kept
    like any other."""
    if device == torch.device("cuda"):
        device = torch.device("cuda", torch.cuda.current_device())
    with _copies_lock:
        moved = _copies.pop(model, None)
        if model.device == device and model.dtype == dtype:
            return model
        if moved is None or moved.device != device or moved.dtype != dtype:
            # The superseded copy goes before the new one is made
            del moved
            moved = copy.deepcopy(model).to(device, dtype)
        _copies[model] = moved
    return cast(_Descriptor, moved)


def _swap_red_blue(image: Tensor) -> Tensor:
    """A (C, H, W) image with its first and third channels swapped: all C reversed for a
    multiple of 3 channels, the alpha kept for 4."""
    if image.shape[0] == 4:
        return image[[2, 1, 0, 3], :, :]
    return image.flip(-3)


def np2tensor(
    img: np.ndarray,
    bgr2rgb: bool = True,
    change_range: bool = True,
    add_batch: bool = True,
) -> Tensor:
    """A (H, W[, C]) image as a float32 (1, C, H, W) tensor, or (C, H, W) without
    `add_batch`; `change_range` scales integer images to [0, 1]."""
    if change_range:
        maxval = MAX_VALUES_BY_DTYPE.get(img.dtype.name, 1.0)
        img = img.astype(np.float32) / maxval
    tensor = torch.from_numpy(
        np.ascontiguousarray(np.transpose(as_3d(img), (2, 0, 1)))
    ).float()
    if bgr2rgb and (tensor.shape[0] % 3 == 0 or tensor.shape[0] == 4):
        tensor = _swap_red_blue(tensor)
    if add_batch:
        tensor.unsqueeze_(0)
    return tensor


def tensor2np(
    img: Tensor,
    rgb2bgr: bool = True,
    data_range: float = 255,
    change_range: bool = True,
    imtype: type = np.uint8,
) -> np.ndarray:
    """A (1, C, H, W), (C, H, W) or (H, W) RGB(A) tensor as a (H, W[, C]) BGR(A) image of
    `imtype`; `change_range` scales [0, 1] to [0, `data_range`], clipped and rounded."""
    n_dim = img.dim()
    img = img.float().cpu()

    img_np: np.ndarray
    if n_dim in (4, 3):
        if n_dim == 4:
            img = img.squeeze(dim=0)
        if rgb2bgr and img.shape[0] in (3, 4):
            img = _swap_red_blue(img)
        img_np = np.transpose(img.numpy(), (1, 2, 0))
    elif n_dim == 2:
        img_np = img.numpy()
    else:
        raise TypeError(
            f"Only support 4D, 3D and 2D tensor. But received with dimension: {n_dim:d}"
        )

    if change_range:
        img_np = np.clip(data_range * img_np, 0, data_range).round()
    return img_np.astype(imtype)


def note_pinned() -> None:
    """Says that RAM was pinned through PyTorch's host allocator, for
    release_pinned_memory."""
    global _pinned
    _pinned = True


def release_pinned_memory() -> None:
    """Gives the RAM pinned through PyTorch's host allocator back to Windows (when a run
    ends, so an idle worker holds no locked RAM; a run's later images reuse it). PyTorch
    keeps it until the process ends otherwise, and 2.14 frees it only through this
    internal call (the runtime pins torch). Blocks still in use stay."""
    global _pinned
    if _pinned:
        _pinned = False
        # Only GPU builds of torch have it, and only a GPU pins.
        from torch._C import _host_emptyCache

        _host_emptyCache()


def swapped_channels(
    image: Tensor, dtype: torch.dtype = torch.float32, clamp: bool = True
) -> Tensor:
    """An H,W,3 image (any strides) with its channels reversed (BGR <-> RGB), clamped to
    0..1 unless `clamp` is False, as a contiguous H,W,3 tensor of `dtype` on its device:
    one pass over the image when it needs no clamp or `dtype` is its own, else the clamp
    in its own dtype and then the cast."""
    out = torch.empty(image.shape, dtype=dtype, device=image.device)
    for c in range(3):
        channel = image[..., 2 - c]
        if not clamp:
            out[..., c] = channel
        elif image.dtype == dtype:
            torch.clamp(channel, 0, 1, out=out[..., c])
        else:
            out[..., c] = channel.clamp(0, 1)
    return out


def upload_frame(
    frame: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32
) -> Tensor:
    """A BGR HWC float32 frame -> RGB CHW in 0..1 in `dtype` on `device`, on the current
    stream: a CHW view of an HWC tensor (channels-last memory). On a GPU the frame goes
    there without waiting from a page-locked copy the host allocator owns, which it keeps
    until the copy has ended: never from the frame's own memory, which may be another
    route's page-locked image (pin_memory would take it as it is, and its owner could
    free it while the copy reads it)."""
    host = torch.from_numpy(np.ascontiguousarray(frame))
    if device.type == "cuda":
        note_pinned()
        host = torch.empty_like(host, pin_memory=True).copy_(host)
    tensor = host.to(device, non_blocking=True)
    return swapped_channels(tensor.float(), dtype).permute(2, 0, 1)


@dataclass(frozen=True)
class Output:
    """A model output on its way to the host: the BGR HWC float32 image in 0..1, whose
    copy has ended once `ready` has passed (None: already there)."""

    image: Tensor
    ready: torch.cuda.Event | None


def to_host(output: Tensor, clamp: bool = True) -> Output:
    """An RGB CHW output, on its device's current stream, turned to BGR HWC float32 in
    0..1 (its values as they are unless `clamp`) on its device in one pass and copied to
    the host: on a GPU into page-locked memory, without waiting for the copy. The copy's
    end is recorded on that same stream, which is not the current device's when the
    output is on another GPU."""
    return host_copy(swapped_channels(output.permute(1, 2, 0), clamp=clamp))


def host_copy(image: Tensor) -> Output:
    """A contiguous image, on its device's current stream, copied to the host as it is:
    on a GPU into page-locked memory, without waiting for the copy, whose end is recorded
    on that stream."""
    if image.device.type != "cuda":
        return Output(image, None)
    note_pinned()
    host = torch.empty(image.shape, pin_memory=True)
    host.copy_(image, non_blocking=True)
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(image.device))
    return Output(host, ready)


def host_image(output: Output) -> np.ndarray:
    """The output's BGR HWC float32 image in 0..1, once its copy has ended."""
    if output.ready is not None:
        output.ready.synchronize()
    return output.image.numpy()


def is_finite(img: np.ndarray) -> bool:
    # NaN propagates through min and max: two passes and no temporary array
    return math.isfinite(img.min()) and math.isfinite(img.max())


def is_out_of_memory(error: Exception) -> bool:
    """Whether `error` says the device or the host ran out of memory: torch's
    OutOfMemoryError or a RuntimeError naming it, or NumPy's MemoryError."""
    message = str(error)
    return (
        isinstance(error, (torch.OutOfMemoryError, MemoryError))
        or "allocate" in message.lower()
        or "out of memory" in message.lower()
        or "ALLOC_FAILED" in message
    )


def safe_cuda_cache_empty():
    """Empties the CUDA cache if CUDA is available."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def window_memory(device: torch.device, budget: int) -> int:
    """The bytes a temporal window may add on a CUDA `device` now: what the allocator
    can still hand out, within the process's memory fraction and the PyTorch settings'
    `budget` (0: none)."""
    # The settings' device may name no index: inside it, the calls take its index.
    with torch.cuda.device(device):
        free, total = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        fraction = torch.cuda.get_per_process_memory_fraction()
    available = free + reserved - allocated
    available = min(available, int(fraction * total) - allocated)
    if budget > 0:
        available = min(available, budget - allocated)
    return available


def weights_digest(model: ImageModelDescriptor) -> bytes:
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
