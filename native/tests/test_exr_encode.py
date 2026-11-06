"""Save Image's OpenEXR files (cn_exr_encode in native/src/codecs/exr_codec.c over OpenEXRCore 3.5.2).

Two independent readers check them: Load Image's decoder for the round trip, and a small parser here, written from the
OpenEXR file layout, that reads the header and chunk table and undoes no compression or ZIP with zlib, so the channel
names, order and byte layout are not only consistent with the library that wrote them.
"""

from __future__ import annotations

import hashlib
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from api import Lazy
from nodes.impl.native_image_io import cv_save_image
from packages.chaiNNer_standard.image.io import load_image, save_image

NONE, RLE, ZIPS, ZIP, PIZ = (save_image.ExrCompression[n] for n in "NONE RLE ZIPS ZIP PIZ".split())
COMPRESSIONS = [NONE, RLE, ZIPS, ZIP, PIZ]
# ZIP's chunks are 16 lines, PIZ's 32: sizes on both sides of their edges.
SIZES = [(1, 1), (1, 7), (7, 1), (15, 3), (16, 3), (17, 3), (31, 3), (32, 3), (33, 3), (65, 64)]
# Fixed pattern, uncompressed float32 (no deflate version in it): the sha256 of the files of 1, 3 and 4 channels.
FROZEN_NONE = {1: "6eb70e62a15a4b54fe7bd9ae0f3081cbe96330fe827bdc0c5d5151c058ffc124", 3: "d151bde7997938a950e813e8becfb3f2c519b7974fc765fe5d22516d0a00d332", 4: "6e47b630c642fb3d523b52f40c761f785d57d7a5fdc06d6dc1d04142bae9798e"}


def random_bits(shape: tuple[int, ...], seed: int = 3) -> np.ndarray:
    """Every float32 bit pattern can come up: signs, denormals, infinities and NaN payloads."""
    return np.random.default_rng(seed).integers(0, 2**32, shape, dtype=np.uint32).view(np.float32)


def shaped(height: int, width: int, channels: int) -> tuple[int, ...]:
    return (height, width) if channels == 1 else (height, width, channels)


def pattern(channels: int) -> np.ndarray:
    shape = shaped(21, 37, channels)
    values = ((np.arange(np.prod(shape), dtype=np.int64) * 2861) % 8191) / 8190
    return (values * 3 - 1).reshape(shape).astype(np.float32)


def save(
    directory: Path,
    image,
    compression=ZIP,
    depth=save_image.ExrColorDepth.F32,
    skip: bool = False,
) -> Path:
    save_image.save_image_node(
        image if isinstance(image, Lazy) else Lazy.ready(image),
        directory,
        None,
        "image",
        save_image.ImageFormat.EXR,
        save_image.PngColorDepth.U8,
        False,
        95,
        save_image.JpegSubsampling.FACTOR_422,
        False,
        save_image.TiffColorDepth.U8,
        save_image.TiffCompression.LZW,
        "BC1_UNORM",
        save_image.BC7Compression.DEFAULT,
        save_image.DDSErrorMetric.PERCEPTUAL,
        False,
        0,
        0,
        save_image.DDSMipFilter.DEFAULT,
        False,
        save_image.DDSColourConversion.KEEP,
        False,
        False,
        save_image.AvifSubsampling.FACTOR_420,
        depth,
        compression,
        skip,
        None,
    )
    return directory / "image.exr"


def assert_bits(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == np.float32
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


class Header:
    """The header of a single-part scanline file, from the OpenEXR file layout."""

    def __init__(self, data: bytes):
        self.data = data
        magic, version = struct.unpack_from("<II", data, 0)
        assert magic == 20000630
        assert version == 2  # file version 2, none of the tiled, long name, deep or multipart flags
        self.attributes: dict[str, tuple[str, bytes]] = {}
        at = 8
        while data[at] != 0:
            end = data.index(0, at)
            name = data[at:end].decode()
            at = end + 1
            end = data.index(0, at)
            kind = data[at:end].decode()
            at = end + 1
            size = struct.unpack_from("<i", data, at)[0]
            at += 4
            self.attributes[name] = (kind, data[at : at + size])
            at += size
        self.table = at + 1

    def channels(self) -> list[tuple[str, int, int, int]]:
        kind, value = self.attributes["channels"]
        assert kind == "chlist"
        result, at = [], 0
        while value[at] != 0:
            end = value.index(0, at)
            name = value[at:end].decode()
            pixel_type = struct.unpack_from("<i", value, end + 1)[0]
            xs, ys = struct.unpack_from("<ii", value, end + 1 + 8)
            result.append((name, pixel_type, xs, ys))
            at = end + 1 + 16
        return result

    def window(self, name: str) -> tuple[int, int, int, int]:
        kind, value = self.attributes[name]
        assert kind == "box2i"
        return struct.unpack("<iiii", value)

    def compression(self) -> int:
        kind, value = self.attributes["compression"]
        assert kind == "compression"
        return value[0]


def unzip(payload: bytes, size: int) -> bytes:
    """A ZIP or ZIPS chunk: stored when deflate did not shrink it, else zlib, the predictor, then the two byte halves."""
    if len(payload) == size:
        return payload
    raw = np.frombuffer(zlib.decompress(payload), np.uint8).astype(np.int64)
    assert raw.size == size
    raw[1:] -= 128
    t = np.cumsum(raw) & 0xFF
    out = np.empty(size, np.uint8)
    half = (size + 1) // 2
    out[0::2] = t[:half]
    out[1::2] = t[half:]
    return out.tobytes()


def planes(path: Path, lines_per_chunk: int, sample: str = "<f4") -> dict[str, np.ndarray]:
    """Every channel of an uncompressed, ZIP or ZIPS file through the chunk table."""
    data = path.read_bytes()
    header = Header(data)
    x0, y0, x1, y1 = header.window("dataWindow")
    assert (x0, y0) == (0, 0)
    assert header.window("displayWindow") == (x0, y0, x1, y1)
    width, height = x1 + 1, y1 + 1
    names = [c[0] for c in header.channels()]
    assert names == sorted(names)
    count = (height + lines_per_chunk - 1) // lines_per_chunk
    table = struct.unpack_from(f"<{count}Q", data, header.table)
    assert table[0] == header.table + 8 * count
    samples = np.dtype(sample)
    result = {n: np.empty((height, width), samples) for n in names}
    at = table[0]
    for chunk, offset in enumerate(table):
        assert offset == at  # one chunk after another, in order, nothing between them
        y, size = struct.unpack_from("<ii", data, offset)
        assert y == chunk * lines_per_chunk
        lines = min(lines_per_chunk, height - y)
        expected = lines * width * len(names) * samples.itemsize
        payload = data[offset + 8 : offset + 8 + size]
        body = payload if header.compression() == 0 else unzip(payload, expected)
        assert len(body) == expected
        rows = np.frombuffer(body, samples).reshape(lines, len(names), width)
        for c, name in enumerate(names):
            result[name][y : y + lines] = rows[:, c]
        at = offset + 8 + size
    assert at == len(data)
    return result


@pytest.mark.parametrize("compression", COMPRESSIONS, ids=lambda c: c.name)
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_float32_round_trip_gives_every_bit(tmp_path, compression, channels, size):
    image = random_bits(shaped(*size, channels))
    before = image.tobytes()
    path = save(tmp_path, image, compression)
    assert_bits(load_image._read_cv(path), image)
    assert image.tobytes() == before


@pytest.mark.parametrize("compression", COMPRESSIONS, ids=lambda c: c.name)
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_half_round_trip_is_half_precision(tmp_path, compression, channels):
    # Normal values and above half's range, by sign; every value rounds as NumPy's float16 does.
    rng = np.random.default_rng(4)
    shape = shaped(40, 33, channels)
    image = (rng.normal(size=shape) * np.exp(rng.uniform(-9, 11.5, shape))).astype(np.float32)
    image.flat[:4] = (0.0, -0.0, np.inf, -np.inf)
    path = save(tmp_path, image, compression, save_image.ExrColorDepth.F16)
    with np.errstate(over="ignore"):
        rounded = image.astype(np.float16).astype(np.float32)
    assert_bits(load_image._read_cv(path), rounded)
    assert {c[1] for c in Header(path.read_bytes()).channels()} == {1}


@pytest.mark.parametrize(
    "channels,names",
    [(1, ["Y"]), (3, ["B", "G", "R"]), (4, ["A", "B", "G", "R"])],
)
@pytest.mark.parametrize(
    "compression,lines", [(NONE, 1), (ZIPS, 1), (ZIP, 16)], ids=["none", "zips", "zip"]
)
def test_independent_reader_finds_each_channel_in_its_place(tmp_path, channels, names, compression, lines):
    image = random_bits(shaped(37, 19, channels), seed=5)
    path = save(tmp_path, image, compression)
    header = Header(path.read_bytes())
    assert header.compression() == compression.value
    assert [c[0] for c in header.channels()] == names
    assert {c[1:] for c in header.channels()} == {(2, 1, 1)}  # float32, not subsampled
    assert header.attributes["lineOrder"][1] == b"\x00"  # increasing y
    found = planes(path, lines)
    # In memory the image is grey or BGR(A); in the file the channels carry their names.
    wanted = {"Y": 0} if channels == 1 else {"B": 0, "G": 1, "R": 2, "A": 3}
    for name in names:
        plane = image if channels == 1 else image[:, :, wanted[name]]
        np.testing.assert_array_equal(
            found[name].astype(np.float32).view(np.uint32), plane.view(np.uint32), err_msg=name
        )


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_frozen_uncompressed_file(tmp_path, channels):
    path = save(tmp_path, pattern(channels), NONE)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FROZEN_NONE[channels]


def test_compressions_differ_and_shrink_a_smooth_image(tmp_path):
    ramp = np.tile(np.linspace(0, 1, 256, dtype=np.float32), (256, 1))
    sizes = {}
    for compression in COMPRESSIONS:
        directory = tmp_path / compression.name
        sizes[compression.name] = save(directory, ramp, compression).stat().st_size
    assert sizes["NONE"] > 256 * 256 * 4
    for name in ("ZIPS", "ZIP", "PIZ"):
        assert sizes[name] < sizes["NONE"] * 0.8, name


def unaligned(image: np.ndarray) -> np.ndarray:
    view = np.ndarray(image.shape, np.float32, bytes(1) + image.tobytes(), 1)
    assert not view.flags.aligned
    return view


@pytest.mark.parametrize("kind", ["reversed", "strided", "readonly", "fortran", "unaligned"])
def test_any_layout_of_the_image(tmp_path, kind):
    image = random_bits((24, 17, 3), seed=6)
    view = {
        "reversed": lambda a: a[::-1, ::-1],
        "strided": lambda a: np.repeat(a, 2, axis=1)[:, ::2],
        "readonly": lambda a: a,
        "fortran": lambda a: np.asfortranarray(a),
        "unaligned": unaligned,
    }[kind](image)
    if kind == "readonly":
        view.setflags(write=False)
    assert kind in ("readonly", "unaligned") or not view.flags.c_contiguous
    assert_bits(load_image._read_cv(save(tmp_path, view)), np.array(view))


def test_prepared_bytes_and_commit_match_the_save(tmp_path):
    image = random_bits((40, 40, 4), seed=7)
    path = save(tmp_path / "direct", image)
    inputs = [
        image,
        tmp_path / "prepared",
        None,
        "image",
        save_image.ImageFormat.EXR,
        save_image.PngColorDepth.U8,
        False,
        95,
        save_image.JpegSubsampling.FACTOR_422,
        False,
        save_image.TiffColorDepth.U8,
        save_image.TiffCompression.LZW,
        "BC1_UNORM",
        save_image.BC7Compression.DEFAULT,
        save_image.DDSErrorMetric.PERCEPTUAL,
        False,
        0,
        0,
        save_image.DDSMipFilter.DEFAULT,
        False,
        save_image.DDSColourConversion.KEEP,
        False,
        False,
        save_image.AvifSubsampling.FACTOR_420,
        save_image.ExrColorDepth.F32,
        ZIP,
        False,
        None,
    ]
    assert bytes(save_image.prepare_save_image(inputs)) == path.read_bytes()


def test_node_round_trip_and_skip_existing(tmp_path):
    image = np.random.default_rng(8).random((12, 9, 4), dtype=np.float32)
    path = save(tmp_path, image)
    written = path.read_bytes()
    loaded, directory, name, _ = load_image.load_image_node(path)
    assert_bits(loaded, image)
    assert (Path(directory), name) == (tmp_path, "image")

    def never():
        raise AssertionError("A skipped file must not read its image")

    save(tmp_path, Lazy(never), NONE, skip=True)
    assert path.read_bytes() == written


def test_saves_with_the_native_codecs_turned_off(tmp_path, monkeypatch):
    # The runtime's OpenCV has no OpenEXR, so the vendored encoder takes .exr either way, as the decoder does.
    image = random_bits((9, 9, 3), seed=9)
    expected = save(tmp_path / "on", image).read_bytes()
    monkeypatch.setenv("CHAINNER_C_NATIVE_CODECS", "0")
    assert save(tmp_path / "off", image).read_bytes() == expected


def test_without_parameters_float32_zip(tmp_path):
    path = tmp_path / "image.exr"
    image = random_bits((20, 20, 3), seed=11)
    cv_save_image(path, image, [])
    assert_bits(load_image._read_cv(path), image)
    assert Header(path.read_bytes()).compression() == ZIP.value


def test_what_the_encoder_declines_is_an_error(tmp_path):
    path = tmp_path / "image.exr"
    with pytest.raises(cv2.error):
        cv_save_image(path, np.zeros((4, 4, 3), np.uint8), [])
    # A lossy code: the encoder takes the lossless ones the node offers.
    with pytest.raises(cv2.error):
        cv_save_image(
            path,
            np.zeros((4, 4, 3), np.float32),
            [cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_DWAA],
        )
    assert not path.exists()


ONE_CPU_CHILD = """
import ctypes
import hashlib
import os
import sys

if sys.platform == "win32":
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), 1):
        raise ctypes.WinError(ctypes.get_last_error())
else:
    os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
sys.path.insert(0, sys.argv[1])
from pathlib import Path

import test_exr_encode as t
from nodes.impl.native import lib

path = t.save(Path(sys.argv[2]), t.random_bits((300, 200, 4), seed=10), t.ZIP)
print(lib().cn_parallel_capacity(), hashlib.sha256(path.read_bytes()).hexdigest())
"""


def test_the_bytes_do_not_depend_on_the_thread_count(tmp_path):
    path = save(tmp_path, random_bits((300, 200, 4), seed=10), ZIP)
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    # A file, since nodes.impl.dds.texconv looks beside the main script.
    child = tmp_path / "one_cpu.py"
    child.write_text(ONE_CPU_CHILD, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-B", str(child), os.path.dirname(__file__), str(tmp_path / "child")],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    capacity, digest = completed.stdout.splitlines()[-1].split()
    assert int(capacity) == 1
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
