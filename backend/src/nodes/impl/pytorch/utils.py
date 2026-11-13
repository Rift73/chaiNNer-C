from __future__ import annotations

import copy
import math
from threading import Lock
from typing import TypeVar, cast
from weakref import WeakKeyDictionary

import numpy as np
import torch
from spandrel import ModelDescriptor
from torch import Tensor

from ..image_utils import MAX_VALUES_BY_DTYPE, as_3d

_Descriptor = TypeVar("_Descriptor", bound=ModelDescriptor)

# A model's copy on another device or in another precision while the model lives (one
# per model: a copy on a new device or in a new precision replaces it).
_copies: WeakKeyDictionary[ModelDescriptor, ModelDescriptor] = WeakKeyDictionary()
_copies_lock = Lock()


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
