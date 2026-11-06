"""The size requirement that ONNX Upscale Image pads each tile to.

Load Model records the requirement that ONNX shape inference gives (load.py): the first
probe size whose output shape infers. That is 16 for every convolutional model, which runs
at any size, and none for exports whose window partitions or pixel unshuffles became
reshapes of computed shapes, which run only at a multiple. So the requirement is measured:
the model runs once on a CPU session at a few small sizes. Shape inference's requirement
remains the fallback when the model does not run there. An ONNX from Convert To ONNX also
records its PyTorch model's requirement (load.py), which is read first: the padding is
at least PyTorch's, and more where the exported graph needs it.

The result depends only on the model's bytes and ONNX Runtime, so it is kept per model,
and also per SHA-256 of the bytes with ORT's version (a model loaded again is not probed
again), unless no CPU session could be made for a reason other than the model, or a run
failed for lack of memory: that result is kept per model only. Where neither the probe
nor shape inference gives a requirement, it is a minimum of 16 px (see _measure). One
model's probe never waits for another's. The probe runs on the node's own session when
that is a CPU one, else on a CPU session of chaiNNer-C's native adapter (which, unlike
ORT's Python binding, releases the GIL while the model runs), with ORT's log of the
sizes the model rejects kept quiet.

A requirement kept by content is also kept as a file in `<storage>/onnx-size-probe`,
under a key of all it depends on: the model's bytes, ORT's version, this module's code
and shape inference's requirement. Later starts read it instead of probing. A file also
holds its key and a checksum; a damaged one, or one of another key, is removed and the
model is probed again, so it cannot change a requirement. A model that did not run keeps
nothing there: the next start tries again, as a failure may not last. Up to DISK_ENTRIES
files are kept, the least recently used removed first; a file that cannot be read or
written is logged.

That CPU session is most of a GPU provider's first run, so Load Model starts the probe
in the background (the app runs Load Model as soon as a file is picked), and so does
Upscale Image, for a model from elsewhere; get_size_req then waits for it. On a CPU
provider the node's own session probes at little cost, so nothing starts early.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import tempfile
from functools import cache
from pathlib import Path
from threading import Lock, Thread
from weakref import WeakKeyDictionary, WeakSet

import numpy as np
import onnxruntime as ort
from sanic.log import logger

from ..native_onnx_runtime import NativeSession, ort_errors
from .errors import is_out_of_memory, ort_error_message
from .load import model_facts
from .model import OnnxGeneric, SizeReq, content_digest
from .session import OnnxSession, get_input_shape

# The multiples the probe tries, smallest first, and its smallest input side.
_MULTIPLES = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
_BASE = 64
# ORT's refusals of the model itself (an op it does not know or has no CPU kernel for)
_MODEL_ERRORS = (ort_errors.InvalidGraph, ort_errors.NotImplemented)
# The folder in the storage directory and its files, each one requirement
DISK_FOLDER = "onnx-size-probe"
DISK_ENTRIES = 1024
_SUFFIX = ".chainner-c.size-probe"

# _lock guards the dictionaries; a model's own lock serializes its probe.
_lock = Lock()
_model_locks: WeakKeyDictionary[OnnxGeneric, Lock] = WeakKeyDictionary()
_cache: WeakKeyDictionary[OnnxGeneric, SizeReq] = WeakKeyDictionary()
# The measured requirement, or None where the model does not run (the fallback applies)
_by_content: dict[tuple[bytes, str], SizeReq | None] = {}
# The models whose probe was started in the background
_started: WeakSet[OnnxGeneric] = WeakSet()


def get_size_req(
    model: OnnxGeneric,
    session: OnnxSession | None = None,
    storage_dir: Path | None = None,
) -> SizeReq:
    """The size requirement of a model with a dynamic input height and width. The model
    is probed once; later calls return the cached result.
    session: the node's session for the model, which the probe uses when it runs on
    the CPU.
    storage_dir: where the result is kept for later starts; None keeps it in memory."""
    return _requirement(model, session, storage_dir, background=False)


def probe_in_background(model: OnnxGeneric, storage_dir: Path | None) -> None:
    """Start the probe of `model` on a thread of its own, unless the model has a fixed
    input size (it is not padded) or the probe has already started."""
    info = model.info
    if info.fixed_input_width is not None or info.fixed_input_height is not None:
        return
    with _lock:
        if model in _cache or model in _started:
            return
        _started.add(model)
    Thread(
        target=_background_probe,
        args=(model, storage_dir),
        name="onnx-size-probe",
        daemon=True,
    ).start()


def _background_probe(model: OnnxGeneric, storage_dir: Path | None) -> None:
    try:
        _requirement(model, None, storage_dir, background=True)
    except Exception:
        # Nothing waits for this thread: get_size_req probes then, and raises.
        logger.warning("ONNX size probe: the background probe failed.", exc_info=True)


def _requirement(
    model: OnnxGeneric,
    session: OnnxSession | None,
    storage_dir: Path | None,
    background: bool,
) -> SizeReq:
    """See get_size_req. A result of a background probe that does not hold for the
    model's bytes is not kept for the model either: its node probes again then."""
    with _lock:
        if model in _cache:
            return _cache[model]
        model_lock = _model_locks.setdefault(model, Lock())
    with model_lock:
        with _lock:
            if model in _cache:
                return _cache[model]
        # The PyTorch model's requirement, which Convert To ONNX records
        recorded = model_facts(model).size_req
        info = model.info
        # Shape inference's requirement, which applies only when it also gave the scale.
        fallback = (
            info.size_req
            if info.scale_width is not None and info.scale_height is not None
            else None
        )
        digest = content_digest(model)
        key = (digest, ort.__version__)
        with _lock:
            probed = key in _by_content
            size_req = _by_content.get(key)
        if not probed and storage_dir is not None:
            size_req = _read_kept(storage_dir, digest, fallback)
            if size_req is not None:
                probed = True
                with _lock:
                    _by_content[key] = size_req
        kept = True
        if not probed:
            size_req, kept = _probe(model, session, fallback)
            if kept:
                with _lock:
                    _by_content[key] = size_req
                if storage_dir is not None and size_req is not None:
                    _keep(storage_dir, digest, fallback, size_req)
        if size_req is None:
            size_req = fallback if fallback is not None else SizeReq(16)
        if recorded is not None:
            size_req = _both(recorded, size_req)
        if kept or not background:
            with _lock:
                _cache[model] = size_req
        return size_req


@cache
def _code_identity() -> str:
    """The SHA-256 of this module: a changed probe measures again."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _disk_key(digest: bytes, fallback: SizeReq | None) -> str:
    """The key of a measurement: everything it depends on."""
    others = repr((ort.__version__, _code_identity(), fallback))
    return hashlib.sha256(digest + others.encode()).hexdigest()


def _encode(key: str, size_req: SizeReq) -> bytes:
    values = [size_req.minimum, size_req.multiple_of]
    body = json.dumps({"key": key, "size_req": values})
    checksum = hashlib.sha256(body.encode()).hexdigest()
    return json.dumps({"body": body, "sha256": checksum}).encode()


def _decode(data: bytes, key: str) -> SizeReq:
    """The measurement `_encode` wrote for `key`. A damaged record raises ValueError,
    KeyError, TypeError or AttributeError."""
    record = json.loads(data)
    body = record["body"]
    if hashlib.sha256(body.encode()).hexdigest() != record["sha256"]:
        raise ValueError("the checksum does not match")
    fields = json.loads(body)
    if fields["key"] != key:
        raise ValueError("it is the record of another key")
    minimum, multiple_of = fields["size_req"]
    if type(minimum) is not int or type(multiple_of) is not int:
        raise TypeError("the requirement is not two integers")
    return SizeReq(minimum, multiple_of)


def _read_kept(
    storage_dir: Path, digest: bytes, fallback: SizeReq | None
) -> SizeReq | None:
    """The measurement kept on disk, if any."""
    key = _disk_key(digest, fallback)
    path = storage_dir / DISK_FOLDER / f"{key}{_SUFFIX}"
    try:
        size_req = _decode(path.read_bytes(), key)
    except FileNotFoundError:
        return None
    except OSError:
        logger.warning(f"ONNX size probe: cannot read {path}.", exc_info=True)
        return None
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        logger.warning(f"ONNX size probe: {path} is damaged ({e!r}); probing again.")
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning(f"ONNX size probe: cannot remove {path}.", exc_info=True)
        return None
    try:
        os.utime(path)  # its last use, for the eviction
    except OSError:
        logger.warning(f"ONNX size probe: cannot mark {path} used.", exc_info=True)
    return size_req


def _keep(
    storage_dir: Path, digest: bytes, fallback: SizeReq | None, size_req: SizeReq
) -> None:
    """Write a measurement atomically, then remove the least recently used files
    beyond DISK_ENTRIES. A failure is logged: the measurement itself holds."""
    key = _disk_key(digest, fallback)
    folder = storage_dir / DISK_FOLDER
    try:
        folder.mkdir(parents=True, exist_ok=True)
        output = tempfile.NamedTemporaryFile(
            dir=folder, prefix=f"{key}.", suffix=".tmp", delete=False
        )
        temporary = Path(output.name)
        try:
            with output:
                output.write(_encode(key, size_req))
            os.replace(temporary, folder / f"{key}{_SUFFIX}")
        finally:
            temporary.unlink(missing_ok=True)
        # Not another probe's temporary file, which it renames next; a file that another
        # probe's removal takes meanwhile is gone, as this one would have it.
        files: list[tuple[int, Path]] = []
        for path in folder.glob(f"*{_SUFFIX}"):
            try:
                if path.is_file():
                    files.append((path.stat().st_mtime_ns, path))
            except FileNotFoundError:
                continue
        for _, path in sorted(files)[:-DISK_ENTRIES]:
            path.unlink(missing_ok=True)
    except OSError:
        logger.warning(
            f"ONNX size probe: cannot keep {key} in {folder}.", exc_info=True
        )


def _both(a: SizeReq, b: SizeReq) -> SizeReq:
    """The requirement that meets both `a` and `b`."""
    return SizeReq(max(a.minimum, b.minimum), math.lcm(a.multiple_of, b.multiple_of))


def _cpu_session(model: OnnxGeneric) -> OnnxSession:
    """A new CPU session for the probe: the native adapter's where it reads the
    model, else ORT's Python binding's."""
    if model_facts(model).native_session:
        return NativeSession(model.bytes)
    options = ort.SessionOptions()
    options.log_severity_level = 4
    return ort.InferenceSession(
        model.bytes, options, providers=["CPUExecutionProvider"]
    )


def _probe(
    model: OnnxGeneric, node_session: OnnxSession | None, fallback: SizeReq | None
) -> tuple[SizeReq | None, bool]:
    """The measured requirement, or None where the model does not run (`fallback`
    then applies), and whether that holds for the model's bytes: not when no CPU
    session could be made for a reason other than the model, or a run failed for lack
    of memory."""
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
        return None, isinstance(e, _MODEL_ERRORS)
    try:
        return _measure(session, fallback)
    finally:
        if isinstance(session, NativeSession):
            session.close()


def _sides(multiple: int) -> tuple[int, int]:
    """The probe's input height and width for `multiple`: the two smallest sizes from
    _BASE up that are multiples of it and of no entry of _MULTIPLES it is not a
    multiple of. A model that needs an entry of _MULTIPLES runs there exactly when the
    entry divides `multiple`, so the first multiple that runs is the model's. The sides
    differ, since window partitions split height and width separately."""
    sizes = (
        size
        for size in itertools.count(_BASE)
        if all((size % m == 0) == (multiple % m == 0) for m in _MULTIPLES)
    )
    return next(sizes), next(sizes)


def _measure(
    session: OnnxSession, fallback: SizeReq | None
) -> tuple[SizeReq | None, bool]:
    """The measured requirement, or None, and whether it holds for the model's bytes
    (see _probe)."""
    # The probe expects the model to reject some sizes; output_size logs each rejection.
    quiet = ort.RunOptions()
    quiet.log_severity_level = 4
    input_arg = session.get_inputs()[0]
    output_name = session.get_outputs()[0].name
    dtype = np.float16 if input_arg.type == "tensor(float16)" else np.float32
    input_shape, channels, _, _ = get_input_shape(session)
    bhwc = input_shape == "BHWC"
    # A run that failed for lack of memory, not because the model rejects its size
    short_of_memory = False

    def output_size(height: int, width: int) -> tuple[int, int] | None:
        nonlocal short_of_memory
        shape = (1, height, width, channels) if bhwc else (1, channels, height, width)
        try:
            output = np.asarray(
                session.run(
                    [output_name], {input_arg.name: np.full(shape, 0.5, dtype)}, quiet
                )[0]
            )
            out_h, out_w = output.shape[-3:-1] if bhwc else output.shape[-2:]
        except Exception as e:
            if isinstance(e, MemoryError) or is_out_of_memory(e):
                short_of_memory = True
            logger.debug(
                f"ONNX size probe: {width}x{height} failed: {ort_error_message(e)}"
            )
            return None
        return out_h, out_w

    def scale(height: int, width: int) -> tuple[int, int] | None:
        """The integer scale of the model's output at this size, if it runs."""
        size = output_size(height, width)
        if size is None or 0 in size or size[0] % height or size[1] % width:
            return None
        return size[0] // height, size[1] // width

    # The first multiple whose sides run at an integer scale, which is the model's
    for multiple in _MULTIPLES:
        sides = _sides(multiple)
        factors = scale(*sides)
        if factors is not None:
            scale_h, scale_w = factors
            break
    else:
        logger.info(
            f"ONNX size probe: no multiple runs at an integer scale; keeping {fallback}"
        )
        return None, not short_of_memory

    def runs(height: int, width: int) -> bool:
        return output_size(height, width) == (scale_h * height, scale_w * width)

    # The minimum stays shape inference's (16 for most models; the 3x3 test image of
    # Interpolate Models and tiny tiles pad up to it), or is 16 where it gives none:
    # spandrel pads most window models to 16 before they run, and below it a converted
    # one can end the process (an OmniSR export divides by zero at 8px). It is raised
    # to the smallest of 16, 32 and 64 (as multiples) that runs when the model fails at
    # it, as a RealCUGAN export does below 20px; else to the probe's larger side, which
    # ran.
    kept = SizeReq(minimum=fallback.minimum if fallback else 16, multiple_of=multiple)
    sizes = {SizeReq(size, multiple).minimum for size in (kept.minimum, 16, 32, _BASE)}
    minimum = next(
        (size for size in sorted(sizes) if size >= kept.minimum and runs(size, size)),
        max(sides),
    )
    size_req = SizeReq(minimum=minimum, multiple_of=multiple)
    logger.info(f"ONNX size probe: {size_req} (shape inference: {fallback})")
    return size_req, not short_of_memory
