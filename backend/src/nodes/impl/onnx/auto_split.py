from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Callable, cast

import numpy as np

from api import Progress
from nodes.impl.onnx.model import SizeReq

from ..native_framework_images import planar_to_interleaved, reflect_pad, swap_red_blue
from ..native_tensors import cast_numpy
from ..upscale.auto_split import Tiler, auto_split
from .errors import ort_error_message

if TYPE_CHECKING:
    # Annotation only: no new import at runtime.
    from .session import OnnxSession


def _into_batched_form(img: np.ndarray, change_shape: bool) -> np.ndarray:
    shape_size = len(img.shape)
    if shape_size == 3:
        if change_shape:
            return img[np.newaxis, :]
        else:
            return img.transpose((2, 0, 1))[np.newaxis, :]
    elif shape_size == 2:
        if change_shape:
            return img[np.newaxis, :, :, np.newaxis]
        else:
            return img[np.newaxis, np.newaxis, :]
    else:
        raise ValueError("Unsupported input tensor shape")


def _into_standard_image_form(img: np.ndarray, change_shape: bool) -> np.ndarray:
    shape_size = len(img.shape)
    if shape_size == 4:
        if change_shape:
            return img.squeeze(0)
        else:
            return img.squeeze(0).transpose(1, 2, 0)
    elif shape_size == 3:
        if change_shape:
            return img
        else:
            return img.transpose(1, 2, 0)
    elif shape_size == 2:
        return img
    else:
        raise ValueError("Unsupported output tensor shape")


def _flip_r_b_channels(img: np.ndarray) -> np.ndarray:
    return swap_red_blue(img)


def _output_planes(output: np.ndarray, change_shape: bool) -> np.ndarray | None:
    """A (1, C, H, W) or (C, H, W) float16 or float32 output's (C, H, W) planes,
    rows contiguous, as planar_to_interleaved reads them; else None."""
    if change_shape or output.dtype not in (
        np.dtype(np.float16),
        np.dtype(np.float32),
    ):
        return None
    if output.ndim == 4 and output.shape[0] == 1:
        planes = output[0]
    elif output.ndim == 3:
        planes = output
    else:
        return None
    if min(planes.strides) <= 0 or planes.strides[2] != planes.itemsize:
        return None
    return planes


def _pad(
    img: np.ndarray, req: SizeReq
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    w = img.shape[1]
    h = img.shape[0]
    pad_w, pad_h = req.get_padding(w, h)

    def remove_padding(i: np.ndarray) -> np.ndarray:
        new_w = i.shape[1]
        new_h = i.shape[0]
        scale_w = new_w // (w + pad_w)
        scale_h = new_h // (h + pad_h)
        new_pad_w = int(pad_w * scale_w)
        new_pad_h = int(pad_h * scale_h)
        return i[: new_h - new_pad_h, : new_w - new_pad_w]

    if pad_w or pad_h:
        # As spandrel pads a PyTorch model's input: reflection reaches at most
        # size - 1, and the rest replicates the edge.
        reflect_w, reflect_h = min(pad_w, w - 1), min(pad_h, h - 1)
        if reflect_w or reflect_h:
            img = reflect_pad(img, reflect_w, reflect_h)
        if reflect_w < pad_w or reflect_h < pad_h:
            paddings = [(0, pad_h - reflect_h), (0, pad_w - reflect_w)]
            if img.ndim == 3:
                paddings.append((0, 0))
            img = np.pad(img, paddings, "edge")
        return (img, remove_padding)
    else:
        return (img, lambda i: i)


def onnx_auto_split(
    img: np.ndarray,
    session: OnnxSession,
    change_shape: bool,
    tiler: Tiler,
    size_req: SizeReq | None = None,
    collect_after: bool = True,
    progress: Progress | None = None,
) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    is_fp16_model = session.get_inputs()[0].type == "tensor(float16)"

    def upscale(img: np.ndarray, _: object):
        if progress is not None:
            progress.check_aborted()
        try:
            lr_img = cast_numpy(img, np.dtype(np.float16)) if is_fp16_model else img
            lr_img, remove_pad = _pad(lr_img, size_req or SizeReq())
            lr_img = _flip_r_b_channels(lr_img)
            lr_img = _into_batched_form(lr_img, change_shape)
            output = cast(
                np.ndarray, session.run([output_name], {input_name: lr_img})[0]
            )
            planes = _output_planes(output, change_shape)
            if planes is not None:
                # One pass from the cropped planes to the interleaved BGR(A) image
                # the steps below make.
                planes = remove_pad(planes.transpose(1, 2, 0)).transpose(2, 0, 1)
                return planar_to_interleaved(
                    planes, swap_red_blue=planes.shape[0] in (3, 4)
                )
            output = _into_standard_image_form(output, change_shape)
            output = _flip_r_b_channels(output)
            output = remove_pad(output)
            return cast_numpy(output, np.dtype(np.float32))
        except Exception as e:
            message = ort_error_message(e)
            if "ONNXRuntimeError" in message and (
                "allocate memory" in message
                or "out of memory" in message
                or "cudaMalloc" in message
            ):
                raise RuntimeError(
                    "A VRAM out-of-memory error has occurred. Please try using a more extreme tiling mode."
                ) from e
            elif isinstance(e, UnicodeDecodeError):
                raise RuntimeError(message) from e
            else:
                raise

    try:
        return auto_split(img, upscale, tiler, progress=progress)
    finally:
        if collect_after:
            gc.collect()
