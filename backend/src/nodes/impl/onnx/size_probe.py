"""The size requirement that ONNX Upscale Image pads each tile to.

Load Model records the requirement that ONNX shape inference gives (load.py): the first
probe size whose output shape infers. That is 16 for every convolutional model, which runs
at any size, and none for exports whose window partitions or pixel unshuffles became
reshapes of computed shapes, which run only at a multiple. So the requirement is measured:
the model runs once on a CPU session at a few small sizes. Shape inference's requirement
remains the fallback when the model does not run there.
"""

from __future__ import annotations

from threading import Lock
from weakref import WeakKeyDictionary

import numpy as np
import onnxruntime as ort
from sanic.log import logger

from .errors import ort_error_message
from .model import OnnxGeneric, SizeReq
from .session import get_input_shape

# The size whose output gives the scale. The probe for a multiple m runs (64+m)x(64+3m):
# both sides are multiples of m and, for m < 64, of no larger power of two, and they
# differ, since window partitions split height and width separately.
_BASE = 64
_MULTIPLES = (1, 2, 4, 8, 16, 32, 64)

_lock = Lock()
_cache: WeakKeyDictionary[OnnxGeneric, SizeReq | None] = WeakKeyDictionary()


def get_size_req(model: OnnxGeneric) -> SizeReq | None:
    """The size requirement of a model with a dynamic input height and width, or None for
    no padding. The model is probed once; later calls return the cached result."""
    with _lock:
        if model not in _cache:
            _cache[model] = _probe(model)
        return _cache[model]


def _probe(model: OnnxGeneric) -> SizeReq | None:
    info = model.info
    # Shape inference's requirement, which applies only when it also gave the scale.
    fallback = (
        info.size_req
        if info.scale_width is not None and info.scale_height is not None
        else None
    )
    options = ort.SessionOptions()
    # The probe expects the model to reject some sizes; output_size logs each rejection.
    options.log_severity_level = 4
    try:
        session = ort.InferenceSession(
            model.bytes, options, providers=["CPUExecutionProvider"]
        )
    except Exception as e:
        logger.info(
            f"ONNX size probe: no CPU session ({ort_error_message(e)}); keeping {fallback}"
        )
        return fallback
    input_arg = session.get_inputs()[0]
    output_name = session.get_outputs()[0].name
    dtype = np.float16 if input_arg.type == "tensor(float16)" else np.float32
    input_shape, channels, _, _ = get_input_shape(session)
    bhwc = input_shape == "BHWC"

    def output_size(height: int, width: int) -> tuple[int, int] | None:
        shape = (1, height, width, channels) if bhwc else (1, channels, height, width)
        try:
            output = np.asarray(
                session.run(
                    [output_name], {input_arg.name: np.full(shape, 0.5, dtype)}
                )[0]
            )
            out_h, out_w = output.shape[-3:-1] if bhwc else output.shape[-2:]
        except Exception as e:
            logger.debug(
                f"ONNX size probe: {width}x{height} failed: {ort_error_message(e)}"
            )
            return None
        return out_h, out_w

    base = output_size(_BASE, _BASE)
    if base is None or base[0] % _BASE or base[1] % _BASE or 0 in base:
        logger.info(
            f"ONNX size probe: no integer scale at {_BASE}px; keeping {fallback}"
        )
        return fallback
    scale_h, scale_w = base[0] // _BASE, base[1] // _BASE

    def runs(height: int, width: int) -> bool:
        return output_size(height, width) == (scale_h * height, scale_w * width)

    multiple = next((m for m in _MULTIPLES if runs(_BASE + m, _BASE + 3 * m)), None)
    if multiple is None:
        logger.info(f"ONNX size probe: no multiple runs; keeping {fallback}")
        return fallback
    # The minimum stays shape inference's (16 for most models; the 3x3 test image of
    # Interpolate Models and tiny tiles pad up to it), raised to the smallest of 16, 32
    # and 64 that runs when the model fails at it, as a RealCUGAN export does below 20px.
    kept = SizeReq(minimum=fallback.minimum if fallback else 1, multiple_of=multiple)
    minimum = next(
        size
        for size in sorted({kept.minimum, 16, 32, _BASE})
        if size >= kept.minimum and (size >= _BASE or runs(size, size))
    )
    size_req = SizeReq(minimum=minimum, multiple_of=multiple)
    logger.info(f"ONNX size probe: {size_req} (shape inference: {fallback})")
    return size_req
