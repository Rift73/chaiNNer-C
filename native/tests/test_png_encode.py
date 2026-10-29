"""chaiNNer-C's default PNG encoder (native/src/png_encode.c, decision 2026-10-21), CPU only.

Every default 8-bit PNG (grey, grey + alpha, BGR, BGRA) is MinSum-filtered and compressed by libdeflate level 6
in row bands, one zlib stream. Checked here: exact pixels through OpenCV, Pillow and chaiNNer-C's own decoder;
the stream (chunks, CRCs, Adler-32, the filtered rows under libpng's MinSum rule, one IDAT per band); the same
bytes at every ISA level; the size on the owner's sample frames; dispatch to OpenCV for everything else; layouts,
file errors and concurrent callers.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image
from test_image_io import OLD_UTILS, image, modules

from nodes.impl import native_image_io
from nodes.impl.native_graph import graph
from nodes.utils.utils import split_file_path
from packages.chaiNNer_standard.image.io import load_image

SIGNATURE = b"\x89PNG\r\n\x1a\n"
COLOUR_TYPES = {1: 0, 2: 4, 3: 2, 4: 6}
# png_encode.c's band rule: at most 32 bands, each at least 16 KiB of filtered rows and 8 rows.
MAX_BANDS, MIN_BAND_BYTES, MIN_BAND_ROWS = 32, 16 * 1024, 8
# Six real sample images (machine-specific: the folder CHAINNER_C_PNG_SAMPLES names; the test skips without it)
# and the size of fpng 1.0.6's default PNG of each, as cv2.imread(IMREAD_COLOR) decodes it: the encoder this one
# replaced (2026-10-21).
SAMPLES_VARIABLE = "CHAINNER_C_PNG_SAMPLES"
FPNG_DEFAULT_BYTES = {
    "00022_upscaled.png": 13478313,
    "arch/ESC/figs/ArbVisual.png": 1810672,
    "arch/ESC/figs/ClassicVisual.png": 6994564,
    "arch/ESC/figs/MainDIV2K.png": 2491951,
    "arch/ESC/figs/RealWorldVisual.png": 13979125,
    "arch/ESC/figs/imgsli1.png": 3565981,
}


class Probe:
    """A module whose `blocked` names raise; every other attribute is the module's."""

    def __init__(self, module: object, blocked: set[str]):
        self._module, self._blocked = module, blocked

    def __getattr__(self, name: str):
        if name in self._blocked:
            raise AssertionError(f"OpenCV called: {name}")
        return getattr(self._module, name)


def chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    """The file's chunks as (type, payload); every CRC checked."""
    assert data[:8] == SIGNATURE
    offset = 8
    result = []
    while offset < len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        crc = struct.unpack_from(">I", data, offset + 8 + length)[0]
        assert crc == zlib.crc32(kind + payload)
        result.append((kind, payload))
        offset += length + 12
    assert offset == len(data) and result[-1] == (b"IEND", b"")
    return result


def no_opencv():
    def forbidden(*args):
        raise AssertionError("A default 8-bit PNG must use chaiNNer-C's encoder")

    return {
        "cv2": SimpleNamespace(imencode=forbidden),
        "split_file_path": split_file_path,
    }


def encode(value: np.ndarray, tmp_path: Path, name: str = "x.png") -> bytes:
    path = tmp_path / name
    graph().image_io_cv_save(no_opencv(), path, value, [])
    return path.read_bytes()


def png_order(value: np.ndarray) -> np.ndarray:
    """The image's samples in the PNG's channel order, h x w x c."""
    pixels = value.reshape(value.shape[0], value.shape[1], -1)
    if pixels.shape[2] >= 3:
        pixels = pixels[..., [2, 1, 0, *range(3, pixels.shape[2])]]
    return np.ascontiguousarray(pixels)


def opencv_expected(value: np.ndarray) -> np.ndarray:
    """What OpenCV's PNG decoder (and chaiNNer-C's, which follows it) returns: grey + alpha as BGRA."""
    pixels = value.reshape(value.shape[0], value.shape[1], -1)
    if pixels.shape[2] == 1:
        return np.ascontiguousarray(pixels[..., 0])
    if pixels.shape[2] == 2:
        return np.ascontiguousarray(pixels[..., [0, 0, 0, 1]])
    return np.ascontiguousarray(pixels)


def opencv_decode(data: bytes) -> np.ndarray:
    pixels = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    assert pixels is not None, "OpenCV could not decode the PNG"
    return pixels


def decode_all(data: bytes, path: Path) -> dict[str, np.ndarray]:
    """The file through OpenCV, Pillow (in its PNG channel order) and chaiNNer-C's own decoder (lane I's
    libpng decoder behind Load Image, with OpenCV's calls made to raise)."""
    opencv = opencv_decode(data)
    with Image.open(io.BytesIO(data)) as decoded:
        pillow = np.asarray(decoded)
    with Image.open(io.BytesIO(data)) as checked:
        checked.verify()  # CRCs and chunk structure
    globals_ = dict(vars(load_image))
    globals_["cv2"] = Probe(cv2, {"imdecode", "imread", "cvtColor"})
    path.write_bytes(data)
    own = graph().image_io_read_cv(globals_, path)
    return {"opencv": opencv, "pillow": pillow, "own": own}


def assert_exact(value: np.ndarray, data: bytes, path: Path) -> None:
    decoded = decode_all(data, path)
    expected = opencv_expected(value)
    for name in ("opencv", "own"):
        assert decoded[name] is not None, name
        assert (
            decoded[name].dtype == np.uint8 and decoded[name].shape == expected.shape
        ), name
        assert np.array_equal(decoded[name], expected), name
    pillow = decoded["pillow"].reshape(value.shape[0], value.shape[1], -1)
    assert np.array_equal(pillow, png_order(value))


def minsum_stream(pixels: np.ndarray) -> bytes:
    """libpng's MinSum filtering of h x w x c samples (PNG order): each row's filter type, then its bytes."""
    height, width, bpp = pixels.shape
    x = pixels.reshape(height, width * bpp).astype(np.int16)
    zero_row = np.zeros((1, x.shape[1]), np.int16)
    zero_left = np.zeros((height, bpp), np.int16)
    up = np.vstack([zero_row, x[:-1]])
    left = np.hstack([zero_left, x[:, :-bpp]])
    up_left = np.hstack([zero_left, up[:, :-bpp]])
    p = left + up - up_left
    pa, pb, pc = np.abs(p - left), np.abs(p - up), np.abs(p - up_left)
    paeth = np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, up, up_left))
    predictions = [np.zeros_like(x), left, up, (left + up) >> 1, paeth]
    filtered = np.stack([((x - q) & 0xFF).astype(np.uint8) for q in predictions])
    costs = np.abs(filtered.view(np.int8).astype(np.int64)).sum(axis=2)
    choice = np.argmin(costs, axis=0)  # the first minimum: ties to the lower type
    rows = np.empty((height, x.shape[1] + 1), np.uint8)
    rows[:, 0] = choice
    rows[:, 1:] = filtered[choice, np.arange(height)]
    return rows.tobytes()


def band_count(height: int, width: int, channels: int) -> int:
    filtered = height * (width * channels + 1)
    return max(1, min(MAX_BANDS, filtered // MIN_BAND_BYTES, height // MIN_BAND_ROWS))


def assert_stream(value: np.ndarray, data: bytes) -> None:
    """The file's chunks: IHDR, one IDAT per band, IEND; a zlib stream (level 6's header) of the MinSum rows whose
    Adler-32 ends it."""
    pixels = png_order(value)
    height, width, channels = pixels.shape
    found = chunks(data)
    assert found[0] == (
        b"IHDR",
        struct.pack(">IIBBBBB", width, height, 8, COLOUR_TYPES[channels], 0, 0, 0),
    )
    assert [kind for kind, _ in found[1:-1]] == [b"IDAT"] * band_count(
        height, width, channels
    )
    stream = b"".join(payload for _, payload in found[1:-1])
    assert stream[:2] == b"\x78\x9c"
    rows = zlib.decompress(stream)
    assert rows == minsum_stream(pixels)
    assert stream[-4:] == struct.pack(">I", zlib.adler32(rows))


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


def test_minsum_reference_picks_each_filter():
    # The reference (minsum_stream) against hand-made rows, ties to the lower type:
    # all filters cost 0 on the zero first row (None); a constant row after another
    # costs 5 under Sub and Paeth (Sub), a repeated row 0 under Up and Paeth (Up);
    # 200s after 5s cost 56 under Sub, 61 under Paeth, more under the rest.
    pixels = np.array([[0] * 4, [5] * 4, [5] * 4, [200] * 4], np.uint8)
    rows = np.frombuffer(minsum_stream(pixels[..., None]), np.uint8).reshape(4, 5)
    assert rows[:, 0].tolist() == [0, 1, 2, 1]


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (19, 1), (7, 13), (32, 33)])
@pytest.mark.parametrize(
    "kind", ["plain", "reverse", "fortran", "readonly", "broadcast", "stride"]
)
def test_exact_pixels_layouts_and_unicode(tmp_path, channels, shape, kind):
    value = np.random.default_rng(13).integers(
        0, 256, (*shape, channels), dtype=np.uint8
    )
    value = view_layout(value, kind)
    before = value.tobytes()
    data = encode(value, tmp_path, "雪 é 😀.PNG")
    assert_exact(value, data, tmp_path / "decode.png")
    assert_stream(value, data)
    assert value.tobytes() == before
    native_image_io.cv_save_image(tmp_path / "again.png", value, [])
    assert (tmp_path / "again.png").read_bytes() == data


def test_grey_as_two_or_three_dimensions(tmp_path):
    value = np.random.default_rng(3).integers(0, 256, (21, 34), dtype=np.uint8)
    flat = encode(value, tmp_path, "flat.png")
    assert encode(value[..., None], tmp_path, "one.png") == flat
    assert_exact(value, flat, tmp_path / "decode.png")


def content(kind: str, shape: tuple[int, int, int], seed: int) -> np.ndarray:
    """Noise, a smooth ramp, or a ramp with 10 % noise (so every filter wins somewhere)."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, shape, dtype=np.uint8)
    h, w, c = shape
    ramp = (
        np.add.outer(np.arange(h) * 3, np.arange(w) * 2)[..., None] + np.arange(c) * 40
    )
    smooth = (ramp % 256).astype(np.uint8)
    if kind == "noise":
        return noise
    if kind == "smooth":
        return smooth
    return np.where(rng.random((h, w, 1)) < 0.1, noise, smooth).astype(np.uint8)


# Heights on both sides of the band rule's steps for 200 x 3 rows (601 bytes): 8 rows and 16 KiB per band, 32
# bands from 873 rows; uneven splits in between.
@pytest.mark.parametrize(
    "height",
    [1, 2, 7, 8, 15, 16, 17, 27, 28, 54, 55, 56, 64, 65, 255, 256, 257, 872, 873, 1001],
)
def test_band_boundary_heights(tmp_path, height):
    value = content("mixed", (height, 200, 3), height)
    data = encode(value, tmp_path)
    assert_exact(value, data, tmp_path / "decode.png")
    assert_stream(value, data)


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("kind", ["noise", "smooth", "mixed"])
@pytest.mark.parametrize("shape", [(64, 64), (331, 517), (1080, 1920)])
def test_stream_and_pixels(tmp_path, channels, kind, shape):
    value = content(kind, (*shape, channels), channels)
    data = encode(value if channels > 1 else value[..., 0], tmp_path)
    assert_exact(value, data, tmp_path / "decode.png")
    assert_stream(value, data)


def test_sixteen_megapixel_image(tmp_path):
    value = content("mixed", (4096, 4096, 3), 16)
    data = encode(value, tmp_path)
    assert_exact(value, data, tmp_path / "decode.png")
    found = chunks(data)
    assert [kind for kind, _ in found] == [b"IHDR"] + [b"IDAT"] * 32 + [b"IEND"]
    rows = zlib.decompress(b"".join(payload for _, payload in found[1:-1]))
    assert len(rows) == 4096 * (4096 * 3 + 1)


# Images for the ISA comparison: every channel count on narrow, odd, wide and band-split shapes, noise, ramps and
# mixed content, and a row pair that walks every (up, up-left) pair under 256 left values (Paeth's whole domain).
ISA_CHILD = r"""
import hashlib, json, sys
import numpy as np
from nodes.impl import native
from nodes.impl.native_graph import graph
from nodes.utils.utils import split_file_path
from pathlib import Path
from types import SimpleNamespace

root = Path(sys.argv[1])
rng = np.random.default_rng(7)
images = []
for shape in [(1, 1), (1, 70), (70, 1), (33, 65), (64, 64), (129, 257), (512, 512), (1080, 1920)]:
    for c in (1, 2, 3, 4):
        noise = rng.integers(0, 256, (*shape, c), dtype=np.uint8)
        smooth = (np.add.outer(np.arange(shape[0]), np.arange(shape[1]) * 2)[..., None] + np.arange(c)).astype(np.uint8)
        images += [noise, smooth, np.where(rng.random(shape)[..., None] < 0.1, noise, smooth).astype(np.uint8)]
pairs = np.arange(65536 * 2 + 1) // 2
up = np.where(np.arange(pairs.size) % 2 == 0, pairs >> 8, pairs & 255).astype(np.uint8)
rows = [up]
for k in range(256):
    rows += [((np.arange(up.size) * 7 + k * 13) & 255).astype(np.uint8), up]
images.append(np.stack(rows)[..., None])
env = {"cv2": SimpleNamespace(imencode=None), "split_file_path": split_file_path}
digests = []
for i, value in enumerate(images):
    path = root / f"{i}.png"
    graph().image_io_cv_save(env, path, value, [])
    digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
    path.unlink()
print(json.dumps({"line": native.isa_line(), "digests": digests}))
"""


def test_identical_bytes_at_every_isa_level(tmp_path):
    results = {}
    for level in ("scalar", "avx2", "avx512"):
        environment = dict(
            os.environ, PYTHONPATH=os.pathsep.join(sys.path), CHAINNER_C_ISA=level
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", ISA_CHILD, str(tmp_path)],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        results[level] = json.loads(completed.stdout.splitlines()[-1])
        assert f"requested {level}" in results[level]["line"]
    assert results["scalar"]["digests"] == results["avx2"]["digests"]
    assert results["scalar"]["digests"] == results["avx512"]["digests"]


def test_sample_frames_are_at_most_80_percent_of_fpng_default(tmp_path):
    folder = os.environ.get(SAMPLES_VARIABLE)
    if not folder:
        pytest.skip(f"{SAMPLES_VARIABLE} names no folder of sample images")
    paths = [Path(folder) / name for name in FPNG_DEFAULT_BYTES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        pytest.skip(f"the sample images are not in {SAMPLES_VARIABLE}: {missing}")
    total = 0
    for index, path in enumerate(paths):
        value = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert value is not None, path
        data = encode(value, tmp_path, f"{index}.png")
        assert np.array_equal(opencv_decode(data), value)
        total += len(data)
    assert total <= 0.80 * sum(FPNG_DEFAULT_BYTES.values())


def save_with_fake_imencode(path, value, params):
    calls = []

    def encode_today(extension, pixels, params):
        calls.append((extension, pixels, params))
        return True, b"original encoder result"

    graph().image_io_cv_save(
        {
            "cv2": SimpleNamespace(imencode=encode_today),
            "split_file_path": split_file_path,
        },
        path,
        value,
        params,
    )
    return calls


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((3, 5, 3), np.float32),
        np.zeros((0, 5, 3), np.uint8),
        np.zeros((3, 5, 5), np.uint8),
        np.zeros((3, 5, 3), np.int8),
        np.zeros((3, 5, 3), bool),
    ],
)
def test_other_types_keep_opencv_contract(tmp_path, value):
    params = []
    calls = save_with_fake_imencode(tmp_path / "x.png", value, params)
    assert len(calls) == 1 and calls[0][1] is value and calls[0][2] is params
    assert (tmp_path / "x.png").read_bytes() == b"original encoder result"


@pytest.mark.parametrize(
    "value",
    [
        np.arange(15, dtype=np.uint16).reshape(3, 5) * 4099,
        np.arange(45, dtype=np.uint16).reshape(3, 5, 3) * 1009,
        np.arange(60, dtype=np.uint16).reshape(3, 5, 4) * 997,
    ],
)
def test_sixteen_bit_encodes_as_opencv(tmp_path, monkeypatch, value):
    # 16-bit PNGs keep OpenCV's PngEncoder: the vendored libpng writes its bytes, and
    # CHAINNER_C_NATIVE_CODECS=0 hands them to cv2.imencode as before.
    calls = save_with_fake_imencode(tmp_path / "native.png", value, [])
    assert not calls
    expected = cv2.imencode(".png", value)[1].tobytes()
    assert (tmp_path / "native.png").read_bytes() == expected
    monkeypatch.setenv("CHAINNER_C_NATIVE_CODECS", "0")
    params = []
    calls = save_with_fake_imencode(tmp_path / "x.png", value, params)
    assert len(calls) == 1 and calls[0][1] is value and calls[0][2] is params


@pytest.mark.parametrize("disabled", [False, True])
def test_eight_bit_takes_the_encoder_with_or_without_native_codecs(
    tmp_path, monkeypatch, disabled
):
    # The encoder replaces fpng, which ran whatever CHAINNER_C_NATIVE_CODECS said.
    if disabled:
        monkeypatch.setenv("CHAINNER_C_NATIVE_CODECS", "0")
    for channels in (1, 2, 3, 4):
        value = content("mixed", (9, 11, channels), channels)
        data = encode(value, tmp_path)
        assert_stream(value, data)


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


def test_write_failure_propagates_without_codec_retry(tmp_path):
    # The write is atomic: its failure (here a directory in the way) raises, encodes
    # nothing again (no cv2 in these globals) and leaves no temporary file behind.
    target = tmp_path / "x.png"
    target.mkdir()
    # What open() raises for a directory: POSIX names the cause
    error = PermissionError if sys.platform == "win32" else IsADirectoryError
    with pytest.raises(error):
        graph().image_io_cv_save(no_opencv(), target, np.zeros((2, 3, 3), np.uint8), [])
    assert target.is_dir() and not any(target.iterdir())
    assert [path.name for path in tmp_path.iterdir()] == ["x.png"]


def test_concurrent_encodes(tmp_path):
    # Python threads encode at once, single- and multi-band images (the bands of
    # every call share the native pool); each file is the one a lone call writes.
    shapes = [(17, 19), (300, 301), (640, 480), (5, 2000)]

    def value_of(index):
        return content("mixed", (*shapes[index % 4], 1 + index % 4), index)

    expected = {
        index: hashlib.sha256(
            encode(value_of(index), tmp_path, "alone.png")
        ).hexdigest()
        for index in range(32)
    }

    def save(index):
        value = value_of(index)
        path = tmp_path / f"线程-{index}.png"
        graph().image_io_cv_save(no_opencv(), path, value, ())
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected[index]
        assert np.array_equal(opencv_decode(data), opencv_expected(value))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(save, range(32)))


@pytest.mark.parametrize("system", ["Windows", "Darwin", "Linux"])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_viewer_uses_the_encoder_without_launching_real_viewer(
    tmp_path, system, missing, channels
):
    module = modules(True).view
    directory = tmp_path / "absent" if missing else tmp_path
    module.mkdtemp = lambda **kwargs: str(directory)
    module.time = SimpleNamespace(time=lambda: 123)
    module.platform = SimpleNamespace(system=lambda: system)
    launches = []
    module.os = SimpleNamespace(path=os.path, startfile=launches.append)
    module.subprocess = SimpleNamespace(call=lambda args: launches.append(args[1]))
    module.cv2 = SimpleNamespace(
        imwrite=lambda *args: pytest.fail("a default 8-bit viewer PNG")
    )
    value = image(channels)
    module.view_image_external_node(value)
    path = directory / "123.png"
    assert launches == ([] if missing else [str(path)])
    if not missing:
        pixels = OLD_UTILS.to_uint8(value, normalized=True)
        data = path.read_bytes()
        assert_exact(pixels, data, tmp_path / "decode.png")
        assert_stream(pixels, data)


@pytest.mark.parametrize("shape", [(1, 2**31, 3), (1, 2**28, 3), (2**31, 1, 1)])
def test_out_of_range_geometry_declines_without_reading(tmp_path, shape):
    # A side beyond PNG's 2^31 - 1, or rows beyond the encoder's 2^29 - 1 samples, go
    # to OpenCV. A broadcast view has one backing pixel: the encoder must decline
    # without allocating or reading the huge logical image.
    value = np.broadcast_to(np.zeros((1, 1, shape[2]), np.uint8), shape)
    calls = save_with_fake_imencode(tmp_path / "large.png", value, [])
    assert len(calls) == 1


@pytest.mark.parametrize("shape", [(1, 65535), (65535, 1), (1, 70001), (70001, 1)])
def test_long_sides_have_exact_ihdr(tmp_path, shape):
    value = np.random.default_rng(51).integers(0, 256, (*shape, 3), dtype=np.uint8)
    data = encode(value, tmp_path)
    assert struct.unpack_from(">II", data, 16) == (shape[1], shape[0])
    assert_exact(value, data, tmp_path / "decode.png")
    assert_stream(value, data)


def test_first_calls_are_safe_for_concurrent_callers(tmp_path):
    # A fresh child has not started the native pool or run the encoder. The barrier
    # lets its first calls enter the encoder together; no elapsed-time assertion.
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
    value = np.random.default_rng(index).integers(0, 256, (210 + index, 230, 3 + index % 2), dtype=np.uint8)
    barrier.wait()
    path = root / f"initial-{index}.png"
    cv_save_image(path, value, [])
    decoded = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
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
