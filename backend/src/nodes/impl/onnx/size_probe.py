"""The size requirement that ONNX Upscale Image pads each tile to.

Load Model records the requirement that ONNX shape inference gives (load.py): the first
probe size whose output shape infers. That is 16 for every convolutional model, which runs
at any size, and none for exports whose window partitions or pixel unshuffles became
reshapes of computed shapes, which run only at a multiple. So the requirement is measured:
the model runs once on a CPU session at a few small sizes. Shape inference's requirement
remains the fallback when the model does not run there.

The result depends only on the model's bytes and ONNX Runtime, so it is kept per model
and per SHA-256 of the bytes with ORT's version (a model loaded again is not probed
again). One model's probe never waits for another's. The probe runs on the node's own
session when that is a CPU one, else on a CPU session of chaiNNer-C's native adapter
(which, unlike ORT's Python binding, releases the GIL while the model runs), with ORT's
log of the sizes the model rejects kept quiet.
"""

from __future__ import annotations

import hashlib
from threading import Lock
from weakref import WeakKeyDictionary

import numpy as np
import onnxruntime as ort
from sanic.log import logger

from ..native_onnx_runtime import NativeSession, supports_model
from .errors import ort_error_message
from .model import OnnxGeneric, SizeReq
from .session import OnnxSession, get_input_shape

# The size whose output gives the scale. The probe for a multiple m runs (64+m)x(64+3m):
# both sides are multiples of m and, for m < 64, of no larger power of two, and they
# differ, since window partitions split height and width separately.
_BASE = 64
_MULTIPLES = (1, 2, 4, 8, 16, 32, 64)

# _lock guards the dictionaries; a model's own lock serializes its probe.
_lock = Lock()
_model_locks: WeakKeyDictionary[OnnxGeneric, Lock] = WeakKeyDictionary()
_cache: WeakKeyDictionary[OnnxGeneric, SizeReq | None] = WeakKeyDictionary()
_by_content: dict[tuple[bytes, str], SizeReq | None] = {}


def get_size_req(
    model: OnnxGeneric, session: OnnxSession | None = None
) -> SizeReq | None:
    """The size requirement of a model with a dynamic input height and width, or None for
    no padding. The model is probed once; later calls return the cached result.
    session: the node's session for the model, which the probe uses when it runs on
    the CPU."""
    with _lock:
        if model in _cache:
            return _cache[model]
        model_lock = _model_locks.setdefault(model, Lock())
    with model_lock:
        with _lock:
            if model in _cache:
                return _cache[model]
        key = (hashlib.sha256(model.bytes).digest(), ort.__version__)
        with _lock:
            known = key in _by_content
            size_req = _by_content.get(key)
        if not known:
            size_req = _probe(model, session)
        with _lock:
            _by_content[key] = size_req
            _cache[model] = size_req
        return size_req


def _cpu_session(model: OnnxGeneric) -> OnnxSession:
    """A new CPU session for the probe: the native adapter's where it reads the
    model, else ORT's Python binding's."""
    if supports_model(model.bytes):
        return NativeSession(model.bytes)
    options = ort.SessionOptions()
    options.log_severity_level = 4
    return ort.InferenceSession(
        model.bytes, options, providers=["CPUExecutionProvider"]
    )


def _probe(model: OnnxGeneric, node_session: OnnxSession | None) -> SizeReq | None:
    info = model.info
    # Shape inference's requirement, which applies only when it also gave the scale.
    fallback = (
        info.size_req
        if info.scale_width is not None and info.scale_height is not None
        else None
    )
    if node_session is not None and list(node_session.get_providers()) == [
        "CPUExecutionProvider"
    ]:
        return _measure(node_session, fallback)
    try:
        session = _cpu_session(model)
    except Exception as e:
        logger.info(
            f"ONNX size probe: no CPU session ({ort_error_message(e)}); keeping"
            f" {fallback}"
        )
        return fallback
    try:
        return _measure(session, fallback)
    finally:
        if isinstance(session, NativeSession):
            session.close()


def _measure(session: OnnxSession, fallback: SizeReq | None) -> SizeReq | None:
    # The probe expects the model to reject some sizes; output_size logs each rejection.
    quiet = ort.RunOptions()
    quiet.log_severity_level = 4
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
                    [output_name], {input_arg.name: np.full(shape, 0.5, dtype)}, quiet
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
