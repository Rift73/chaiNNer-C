"""CPU-only fpng integration: exact pixels, dispatch, layouts and file errors."""

from __future__ import annotations

import builtins
import os
import struct
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from test_image_io import OLD_UTILS, image, modules

from nodes.impl import native_image_io
from nodes.impl.native_graph import graph
from nodes.utils.utils import split_file_path


def chunks(data):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    offset = 8
    result = []
    while offset < len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        crc = struct.unpack_from(">I", data, offset + 8 + length)[0]
        assert crc == zlib.crc32(kind + payload)
        result.append(kind)
        offset += length + 12
    assert offset == len(data) and result[-1] == b"IEND"
    return result


def decode(path) -> np.ndarray:
    image = cv2.imdecode(
        np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED
    )
    assert image is not None, f"OpenCV could not decode {path}"
    return image


def no_opencv():
    def forbidden(*args):
        raise AssertionError("Compatible PNG must use fpng")

    return {
        "cv2": SimpleNamespace(imencode=forbidden),
        "split_file_path": split_file_path,
    }


def view_layout(value, kind):
    if kind == "reverse":
        return value[::-1, ::-1, ::-1]
    if kind == "fortran":
        return np.asfortranarray(value)
    if kind == "readonly":
        value.setflags(write=False)
    if kind == "broadcast":
        return np.broadcast_to(value[:1, :1], value.shape)
    if kind == "stride":
        return value[::2, ::2]
    return value


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (19, 1), (7, 13), (32, 33)])
@pytest.mark.parametrize(
    "kind", ["plain", "reverse", "fortran", "readonly", "broadcast", "stride"]
)
def test_fpng_exact_pixels_and_unicode(tmp_path, channels, shape, kind):
    value = np.random.default_rng(13).integers(
        0, 256, (*shape, channels), dtype=np.uint8
    )
    value = view_layout(value, kind)
    before = value.tobytes()
    path = tmp_path / "雪 é 😀.PNG"
    graph().image_io_cv_save(no_opencv(), path, value, [])
    assert b"fdEC" in chunks(path.read_bytes())  # fpng encoder's private identifier
    actual = decode(path)
    assert actual.shape == value.shape and actual.dtype == value.dtype
    assert actual.tobytes() == before == value.tobytes()
    first = path.read_bytes()
    native_image_io.cv_save_image(path, value, [])
    assert path.read_bytes() == first


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((3, 5), np.uint8),
        np.zeros((3, 5, 1), np.uint8),
        np.zeros((3, 5, 3), np.uint16),
        np.zeros((3, 5, 4), np.uint16),
        np.zeros((3, 5, 3), np.float32),
        np.zeros((0, 5, 3), np.uint8),
        np.zeros((3, 5, 2), np.uint8),
    ],
)
def test_other_types_keep_opencv_contract(tmp_path, value):
    calls = []

    def encode(extension, pixels, params):
        calls.append((extension, pixels, params))
        return True, b"original encoder result"

    params = []
    graph().image_io_cv_save(
        {"cv2": SimpleNamespace(imencode=encode), "split_file_path": split_file_path},
        tmp_path / "x.png",
        value,
        params,
    )
    assert len(calls) == 1 and calls[0][1] is value and calls[0][2] is params
    assert (tmp_path / "x.png").read_bytes() == b"original encoder result"


@pytest.mark.parametrize(
    "suffix,params",
    [
        (".jpg", []),
        (".webp", []),
        (".png", [cv2.IMWRITE_PNG_COMPRESSION, 9]),
        (".png", [cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_RLE]),
    ],
)
def test_explicit_options_and_other_formats_unchanged(tmp_path, suffix, params):
    value = np.arange(11 * 13 * 3, dtype=np.uint8).reshape(11, 13, 3)
    old, new = tmp_path / ("old" + suffix), tmp_path / ("new" + suffix)
    OLD_UTILS.cv_save_image(old, value, params)
    native_image_io.cv_save_image(new, value, params)
    assert old.read_bytes() == new.read_bytes()


@pytest.mark.parametrize("as_directory", [False, True])
def test_file_errors_preserved(tmp_path, as_directory):
    path = (
        tmp_path / "directory.png" if as_directory else tmp_path / "missing" / "雪.png"
    )
    if as_directory:
        path.mkdir()
    value = np.zeros((2, 3, 4), np.uint8)
    failures = []
    for save in (OLD_UTILS.cv_save_image, native_image_io.cv_save_image):
        with pytest.raises(OSError) as captured:
            save(path, value, [])
        failures.append(
            (type(captured.value), captured.value.args, captured.value.filename)
        )
    assert failures[0] == failures[1]


def test_write_failure_propagates_without_codec_retry(tmp_path, monkeypatch):
    class FailingFile:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def write(self, _):
            raise OSError("owned write failure")

    monkeypatch.setattr(builtins, "open", lambda *args: FailingFile())
    with pytest.raises(OSError, match="owned write failure"):
        graph().image_io_cv_save(
            no_opencv(), tmp_path / "x.png", np.zeros((2, 3, 3), np.uint8), []
        )


def test_concurrent_saves(tmp_path):
    def save(index):
        value = np.random.default_rng(index).integers(
            0, 256, (17, 19, 3 + index % 2), dtype=np.uint8
        )
        path = tmp_path / f"线程-{index}.png"
        graph().image_io_cv_save(no_opencv(), path, value, ())
        assert decode(path).tobytes() == value.tobytes()
        assert b"fdEC" in chunks(path.read_bytes())

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(save, range(32)))


@pytest.mark.parametrize("system", ["Windows", "Darwin", "Linux"])
@pytest.mark.parametrize("missing", [False, True])
def test_viewer_uses_fpng_without_launching_real_viewer(tmp_path, system, missing):
    module = modules(True).view
    directory = tmp_path / "absent" if missing else tmp_path
    module.mkdtemp = lambda **kwargs: str(directory)
    module.time = SimpleNamespace(time=lambda: 123)
    module.platform = SimpleNamespace(system=lambda: system)
    launches = []
    module.os = SimpleNamespace(path=os.path, startfile=launches.append)
    module.subprocess = SimpleNamespace(call=lambda args: launches.append(args[1]))
    module.cv2 = SimpleNamespace(
        imwrite=lambda *args: pytest.fail("fpng compatible viewer PNG")
    )
    value = image(4)
    module.view_image_external_node(value)
    path = directory / "123.png"
    assert launches == ([] if missing else [str(path)])
    if not missing:
        assert b"fdEC" in chunks(path.read_bytes())
        assert (
            decode(path).tobytes()
            == OLD_UTILS.to_uint8(value, normalized=True).tobytes()
        )


def test_oversize_vendor_geometry_dispatched_without_reading_foreign_buffer(tmp_path):
    # A valid broadcast view has only one backing pixel; do not allocate or read
    # the huge logical image. The sentinel encoder proves capability dispatch.
    value = np.broadcast_to(np.zeros((1, 1, 3), np.uint8), (1, 65536, 3))
    calls = []
    env = {
        "split_file_path": split_file_path,
        "cv2": SimpleNamespace(
            imencode=lambda *args: calls.append(args) or (True, b"unsupported geometry")
        ),
    }
    graph().image_io_cv_save(env, tmp_path / "large.png", value, [])
    assert len(calls) == 1


@pytest.mark.parametrize("shape", [(1, 65535), (65535, 1)])
def test_vendor_dimension_boundary_has_exact_ihdr(tmp_path, shape):
    value = np.random.default_rng(51).integers(0, 256, (*shape, 3), dtype=np.uint8)
    path = tmp_path / "boundary.png"
    graph().image_io_cv_save(no_opencv(), path, value, [])
    encoded = path.read_bytes()
    assert struct.unpack_from(">II", encoded, 16) == (shape[1], shape[0])
    assert b"fdEC" in chunks(encoded)
    assert decode(path).tobytes() == value.tobytes()


def test_first_initialization_is_safe_for_concurrent_callers(tmp_path):
    # A fresh child has not initialized fpng. The barrier lets its first calls
    # enter the native encoder concurrently; no elapsed-time assertion is used.
    script = r"""
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import cv2
import numpy as np
from nodes.impl.native_image_io import cv_save_image

root = Path(sys.argv[1])
barrier = Barrier(8)
def save(index):
    value = np.random.default_rng(index).integers(0, 256, (21, 23, 3 + index % 2), dtype=np.uint8)
    barrier.wait()
    path = root / f"initial-{index}.png"
    cv_save_image(path, value, [])
    encoded = path.read_bytes()
    assert b"fdEC" in encoded
    decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.shape == value.shape and decoded.tobytes() == value.tobytes()
with ThreadPoolExecutor(max_workers=8) as workers:
    list(workers.map(save, range(8)))
"""
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script, str(tmp_path)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
