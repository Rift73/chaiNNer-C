"""Load Image's OpenEXR decoder (native/src/codecs/exr_codec.c over OpenEXRCore 3.5.2).

The fixtures in exr/ were written by exr/gen_exr_fixtures.c with OpenEXRCore's own writer; their pixels follow
formulas recomputed here with NumPy (exact half bits, float32 and uint32 values, or a smooth ramp for the lossy
codecs), so every reference is independent of OpenEXR. The decode mirrors OpenCV's ExrDecoder under
IMREAD_UNCHANGED (the runtime's OpenCV has no OpenEXR): float32, BGR(A), the data window, part 0 and level 0.
"""

from __future__ import annotations

import hashlib
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest
from numpy._core._multiarray_umath import __cpu_features__

from packages.chaiNNer_standard.image.io import load_image

EXR = Path(__file__).with_name("exr")


class Probe:
    """A cv2 stand-in whose decoders fail the test: the EXR must decode natively."""

    def __init__(self, cv2):
        self._cv2 = cv2

    def __getattr__(self, name):
        if name in ("imdecode", "imread"):
            raise AssertionError(f"cv2.{name} was called")
        return getattr(self._cv2, name)


def grid(
    width: int, height: int, xs: int = 1, ys: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Data-window-relative coordinates of a channel's samples (every xs-th column, ys-th row)."""
    x = np.arange(0, width, xs, dtype=np.int64)[None, :]
    y = np.arange(0, height, ys, dtype=np.int64)[:, None]
    return x, y


def counter(c: int, x: np.ndarray, y: np.ndarray, level: int = 0) -> np.ndarray:
    return x * 7919 + y * 104729 + c * 15485863 + level * 31


def exact_half(c: int, width: int, height: int, level: int = 0) -> np.ndarray:
    x, y = grid(width, height)
    n = counter(c, x, y, level)
    bits = (n % 0x7C00).astype(np.uint16)
    bits = np.where((x + y + c) % 3 == 0, bits | 0x8000, bits).astype(np.uint16)
    if c == 0 and level == 0:
        bits[0, :3] = (0x7C00, 0x7E00, 0xFC00)  # +inf, NaN, -inf
    return bits.view(np.float16).astype(np.float32)


def exact_float(c: int, width: int, height: int) -> np.ndarray:
    x, y = grid(width, height)
    return ((counter(c, x, y) % 200001) - 100000).astype(np.float32) / np.float32(64)


def exact_uint(c: int, width: int, height: int) -> np.ndarray:
    x, y = grid(width, height)
    return (counter(c, x, y) % 2**32).astype(np.uint32).astype(np.float32)


def smooth(c: int, width: int, height: int, xs: int = 1, ys: int = 1) -> np.ndarray:
    """The lossy codecs' ramp, at a channel's sample positions (float64, then float32, then half)."""
    x, y = grid(width, height, xs, ys)
    value = 0.1 + 3.0 * (x / width) * (y / height) + 0.5 * c / 3.0
    return value.astype(np.float32).astype(np.float16).astype(np.float32)


def read(path: Path) -> np.ndarray:
    image = load_image._read_cv(path)
    assert image is not None
    return image


def decode(name: str) -> np.ndarray:
    return read(EXR / name)


def assert_bits(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == np.float32
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(
        actual.view(np.uint32), expected.astype(np.float32).view(np.uint32)
    )


def rgba_half(width: int = 23, height: int = 37) -> np.ndarray:
    """BGRA of a file whose sorted channels are A, B, G, R (positions 0-3)."""
    a, b, g, r = (exact_half(c, width, height) for c in range(4))
    return np.dstack([b, g, r, a])


LOSSLESS = ["none", "rle", "zips", "zip", "piz", "pxr24", "htj2k256", "htj2k32", "zstd"]
# Relative error bound of each lossy codec on the ramp (B44's fixed-rate 4x4 blocks lose most where it is steep).
LOSSY = {"b44": 3e-2, "b44a": 3e-2, "dwaa": 1e-2, "dwab": 1e-2, "lj2k": 1e-2}
# SHA-256 of the decoded arrays, pinned after they matched OpenEXR 3.4.4's C++ RgbaInputFile (ImageMagick 7.1.2's
# EXR reader) to within one float32 ulp. B44 is integer arithmetic; DWA's inverse DCT runs OpenEXR's scalar code on
# CPUs with AVX (its AVX kernel is GCC inline assembly for LP64 targets only) and its SSE2 code without, which may
# round differently, so the DWA hashes are checked on AVX CPUs. Lossy HTJ2K is left out: OpenJPH's irreversible
# wavelet runs per-ISA float kernels.
AVX = __cpu_features__["AVX"]
LOSSY_DECODED = {
    "b44": "e77570222a6bfc4979ffa14adddeb036df445e10e6ccbe13ae398ecbef3a56b0",
    "b44a": "e77570222a6bfc4979ffa14adddeb036df445e10e6ccbe13ae398ecbef3a56b0",
    "dwaa": "eed32c4b1d17539741f3770989bc5b5b1a65e8ce0bcba1baf78c073b2d13b3da",
    "dwab": "b3e14a9ba1a0d47b911f8b9d709dc48afa9100247898613515270f00c1f44d61",
}


@pytest.mark.parametrize("codec", LOSSLESS)
def test_lossless_codecs_give_every_half_bit(
    codec: str, monkeypatch: pytest.MonkeyPatch
):
    """Half samples widen exactly to float32 (denormals, negatives, values above 1, +-inf and NaN), BGRA."""
    monkeypatch.setattr(load_image, "cv2", Probe(load_image.cv2))
    assert_bits(decode(f"rgba_half_{codec}.exr"), rgba_half())


@pytest.mark.parametrize("codec", sorted(LOSSY))
def test_lossy_codecs_stay_near_the_source(codec: str):
    """B44, B44A, DWAA, DWAB and lossy HTJ2K: BGRA float32 within the codec's error of the source ramp."""
    image = decode(f"rgba_half_{codec}.exr")
    a, b, g, r = (smooth(c, 23, 37) for c in range(4))
    source = np.dstack([b, g, r, a])
    assert image.dtype == np.float32 and image.shape == (37, 23, 4)
    np.testing.assert_allclose(image, source, rtol=LOSSY[codec], atol=LOSSY[codec])
    if codec in LOSSY_DECODED and (AVX or not codec.startswith("dwa")):
        assert hashlib.sha256(image.tobytes()).hexdigest() == LOSSY_DECODED[codec]


@pytest.mark.parametrize(
    ("name", "make"),
    [
        (
            "rgb_float_zip.exr",
            lambda: np.dstack([exact_float(c, 23, 37) for c in range(3)]),
        ),
        (
            "rgb_float_piz.exr",
            lambda: np.dstack([exact_float(c, 23, 37) for c in range(3)]),
        ),
        (
            "rgb_uint_zip.exr",
            lambda: np.dstack([exact_uint(c, 23, 37) for c in range(3)]),
        ),
        # A half, B float, G uint, R half
        (
            "rgba_mixed_rle.exr",
            lambda: np.dstack(
                [
                    exact_float(1, 23, 37),
                    exact_uint(2, 23, 37),
                    exact_half(3, 23, 37),
                    exact_half(0, 23, 37),
                ]
            ),
        ),
    ],
)
def test_float_and_uint_samples(name: str, make):
    """float32 samples as they are, uint32 samples converted to float32, as OpenEXR's float slices give them."""
    assert_bits(decode(name), make())


def test_pxr24_rounds_float_to_24_bits():
    """PXR24 keeps 24 of a float's 32 bits (16 of its mantissa): lossy for float, lossless for half."""
    expected = np.dstack([exact_float(c, 23, 37) for c in range(3)])
    np.testing.assert_allclose(
        decode("rgb_float_pxr24.exr"), expected, rtol=2.0**-16, atol=0
    )


def test_missing_colour_channel_is_zero():
    """R and G only: B reads as 0, OpenCV's slice fill value."""
    g, r = exact_half(0, 23, 37), exact_half(1, 23, 37)
    assert_bits(decode("rg_half_zip.exr"), np.dstack([np.zeros_like(g), g, r]))


def test_other_channels_are_ignored():
    """Z and a layer's channels next to R, G, B, A are not read."""
    assert_bits(decode("rgba_extra_zip.exr"), rgba_half())


@pytest.mark.parametrize(
    ("name", "make"),
    [
        ("y_half_piz.exr", lambda: exact_half(0, 23, 37)),
        ("z_float_none.exr", lambda: exact_float(0, 23, 37)),
    ],
)
def test_luminance_or_depth_is_grey(name: str, make):
    """Y, or Z without Y, loads as one channel: a 2-D array."""
    assert_bits(decode(name), make())


def test_luminance_with_alpha_is_bgra():
    """Y with A: OpenCV gives two channels, which chaiNNer's images do not have; the grey goes to B, G and R."""
    a, y = exact_half(0, 23, 37), exact_half(1, 23, 37)
    assert_bits(decode("ya_half_zip.exr"), np.dstack([y, y, y, a]))


def luminance_weights(chromaticities: tuple[float, ...]) -> np.ndarray:
    """OpenEXR's computeYw (row Y of RGBtoXYZ for Y = 1, normalised), in float64, rounded to float32."""
    rx, ry, gx, gy, bx, by, wx, wy = (float(np.float32(v)) for v in chromaticities)
    x_w, y_w, z_w = wx / wy, 1.0, (1.0 - wx - wy) / wy  # the white point's XYZ
    d = rx * (by - gy) + bx * (gy - ry) + gx * (ry - by)
    sr = (
        x_w * (by - gy)
        - gx * (y_w * (by - 1) + by * (x_w + z_w))
        + bx * (y_w * (gy - 1) + gy * (x_w + z_w))
    ) / d
    sg = (
        x_w * (ry - by)
        + rx * (y_w * (by - 1) + by * (x_w + z_w))
        - bx * (y_w * (ry - 1) + ry * (x_w + z_w))
    ) / d
    sb = (
        x_w * (gy - ry)
        - rx * (y_w * (gy - 1) + gy * (x_w + z_w))
        + gx * (y_w * (ry - 1) + ry * (x_w + z_w))
    ) / d
    weights = np.array([sr * ry, sg * gy, sb * by])
    return (weights / weights.sum()).astype(np.float32)


REC709 = (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290)
REC2020 = (0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290)


def test_rec709_luminance_weights():
    np.testing.assert_allclose(
        luminance_weights(REC709), [0.2126, 0.7152, 0.0722], atol=5e-5
    )


@pytest.mark.parametrize(
    ("name", "first", "chromaticities"),
    [("yc_half_zip.exr", 0, REC709), ("yca_half_piz_chroma.exr", 1, REC2020)],
)
def test_luminance_chroma_is_bgr(name: str, first: int, chromaticities):
    """Y with 2x2-subsampled RY and BY: the chroma repeated over each 2x2 block (OpenCV's UpSample), then
    R = (RY + 1) Y, B = (BY + 1) Y, G = (Y - R Yw.r - B Yw.b) / Yw.g with OpenEXR's weights for the file's
    chromaticities (Rec. 709 by default); alpha kept."""
    by = np.repeat(np.repeat(smooth(first, 24, 36, 2, 2), 2, 0), 2, 1).astype(
        np.float64
    )
    ry = np.repeat(np.repeat(smooth(first + 1, 24, 36, 2, 2), 2, 0), 2, 1).astype(
        np.float64
    )
    y = smooth(first + 2, 24, 36).astype(np.float64)
    weights = luminance_weights(chromaticities).astype(np.float64)
    r, b = (ry + 1) * y, (by + 1) * y
    g = (y - b * weights[2] - r * weights[0]) / weights[1]
    expected = np.dstack([b, g, r]).astype(np.float32)
    if first:
        expected = np.dstack([expected, smooth(0, 24, 36)])
    assert_bits(decode(name), expected)


def test_numbered_channels_are_copied_in_order():
    """Channels "0", "1", "2" (OpenCV's multispectral case) load as they are, without a BGR swap."""
    assert_bits(
        decode("multi3_float_zip.exr"),
        np.dstack([exact_float(c, 23, 37) for c in range(3)]),
    )


@pytest.mark.parametrize(
    ("name", "make"),
    [
        ("tiled_rgba_zip.exr", rgba_half),
        (
            "tiled_mip_rgb_piz.exr",
            lambda: np.dstack([exact_half(c, 23, 37) for c in range(3)]),
        ),
        ("window_rgba_zip.exr", rgba_half),
        ("window_tiled_rgba_rle.exr", rgba_half),
        (
            "multipart_rgb_zip.exr",
            lambda: np.dstack([exact_half(c, 23, 37) for c in range(3)]),
        ),
    ],
)
def test_tiles_windows_and_parts(name: str, make):
    """Tiled files (partial edge tiles, level 0 of a mipmap), data windows off the origin inside a larger
    display window (the data window is the image) and part 0 of a two-part file."""
    assert_bits(decode(name), make())


def test_ripmap_level_zero():
    image = decode("tiled_rip_rgba_dwab.exr")
    a, b, g, r = (smooth(c, 23, 37) for c in range(4))
    np.testing.assert_allclose(image, np.dstack([b, g, r, a]), rtol=1e-2, atol=1e-2)


def minimal_exr(
    width: int, height: int, channels: dict[str, np.ndarray], window=(0, 0)
) -> bytes:
    """A single-part scanline EXR without compression, written here from the file format's specification."""

    def attribute(name: str, kind: str, value: bytes) -> bytes:
        return (
            name.encode()
            + b"\0"
            + kind.encode()
            + b"\0"
            + struct.pack("<i", len(value))
            + value
        )

    names = sorted(channels)
    chlist = (
        b"".join(n.encode() + b"\0" + struct.pack("<iB3xii", 1, 0, 1, 1) for n in names)
        + b"\0"
    )
    box = struct.pack(
        "<iiii", window[0], window[1], window[0] + width - 1, window[1] + height - 1
    )
    header = (
        b"\x76\x2f\x31\x01"
        + struct.pack("<I", 2)
        + attribute("channels", "chlist", chlist)
        + attribute("compression", "compression", b"\0")
        + attribute("dataWindow", "box2i", box)
        + attribute("displayWindow", "box2i", box)
        + attribute("lineOrder", "lineOrder", b"\0")
        + attribute("pixelAspectRatio", "float", struct.pack("<f", 1))
        + attribute("screenWindowCenter", "v2f", struct.pack("<ff", 0, 0))
        + attribute("screenWindowWidth", "float", struct.pack("<f", 1))
        + b"\0"
    )
    if not all(len(v) for v in channels.values()):  # header and offset table only
        return header + bytes(8 * height)
    lines = []
    for y in range(height):
        data = b"".join(channels[n][y].astype("<f2").tobytes() for n in names)
        lines.append(struct.pack("<ii", window[1] + y, len(data)) + data)
    offsets, at = [], len(header) + 8 * height
    for line in lines:
        offsets.append(at)
        at += len(line)
    return header + struct.pack(f"<{height}Q", *offsets) + b"".join(lines)


def test_file_written_from_the_specification(tmp_path: Path):
    """An EXR written byte by byte from the format specification (not by OpenEXR) loads."""
    rgb = np.arange(5 * 3 * 3, dtype=np.float32).reshape(3, 5, 3) / 8 - 2
    path = tmp_path / "spec.exr"
    path.write_bytes(
        minimal_exr(
            5, 3, {"R": rgb[:, :, 0], "G": rgb[:, :, 1], "B": rgb[:, :, 2]}, (-2, 4)
        )
    )
    assert_bits(read(path), rgb[:, :, ::-1])


def test_decoded_by_signature_whatever_the_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """imdecode picks its decoder by content, so OpenEXR named .tif decodes too."""
    path = tmp_path / "exr_named.tif"
    shutil.copyfile(EXR / "rgba_half_zip.exr", path)
    monkeypatch.setattr(load_image, "cv2", Probe(load_image.cv2))
    assert_bits(read(path), rgba_half())


def test_decoded_with_native_codecs_off(monkeypatch: pytest.MonkeyPatch):
    """CHAINNER_C_NATIVE_CODECS=0 sends other files to OpenCV, which has no OpenEXR: EXR still decodes natively."""
    monkeypatch.setenv("CHAINNER_C_NATIVE_CODECS", "0")
    monkeypatch.setattr(load_image, "cv2", Probe(load_image.cv2))
    assert_bits(decode("rgba_half_piz.exr"), rgba_half())


def test_load_image_node_returns_the_raw_array():
    """The node gives the decoded float32 array, its folder and its name; LargeImageOutput normalises it later."""
    image, directory, name = load_image.load_image_node(EXR / "rgba_half_dwaa.exr")
    assert image.dtype == np.float32 and image.shape == (37, 23, 4) and image.max() > 1
    assert (directory, name) == (EXR, "rgba_half_dwaa")


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(lambda data: data[: len(data) - 200], id="truncated"),
        pytest.param(
            lambda data: data[:400] + bytes(len(data) - 400), id="zeroed-chunks"
        ),
        pytest.param(lambda data: data[:20], id="header-only"),
    ],
)
def test_damaged_files_raise(tmp_path: Path, corrupt):
    """A file OpenEXRCore reports an error for declines, and Load Image raises its error for the file."""
    path = tmp_path / "damaged.exr"
    path.write_bytes(corrupt((EXR / "rgba_half_zip.exr").read_bytes()))
    with pytest.raises(RuntimeError, match="Error reading image"):
        load_image._read_cv(path)


def test_damaged_htj2k_tile_raises(tmp_path: Path):
    """A tile OpenJPH cannot parse (one bit of rgba_half_lj2k.exr's codestream flipped, found by fuzzing): OpenJPH
    reports it by throwing from inside its own catch handler, OpenEXRCore turns that into an error, and Load Image
    raises its error for the file (the sanitizer build cannot run this path: tests/fuzz/CMakeLists.txt)."""
    data = bytearray((EXR / "rgba_half_lj2k.exr").read_bytes())
    data[4272] ^= 0x10
    path = tmp_path / "damaged.exr"
    path.write_bytes(data)
    with pytest.raises(RuntimeError, match="Error reading image"):
        load_image._read_cv(path)


def test_htj2k_nlt_marker_raises():
    """damaged_htj2k_nlt.exr (a fuzzer's damaged HTJ2K chunk) has an NLT marker of one-byte points, which OpenJPH
    0.32.0 read past its table's allocation (third_party/openjph/PROVENANCE.md's patch): the chunk now fails as a
    damaged one, and Load Image raises its error for the file."""
    with pytest.raises(RuntimeError, match="Error reading image"):
        load_image._read_cv(EXR / "damaged_htj2k_nlt.exr")


def test_unknown_channels_raise():
    """No R, G, B, numbered, Y or Z channel: OpenCV reads no image, and neither does Load Image."""
    with pytest.raises(RuntimeError, match="Error reading image"):
        decode("other_channels_zip.exr")


@pytest.mark.parametrize(("width", "taken"), [(40_000, True), (40_001, False)])
def test_size_limit(width: int, taken: bool, monkeypatch: pytest.MonkeyPatch):
    """The owner's limit, 40,000 x 40,000 pixels: the decoder asks for that buffer and declines a column more
    before allocating anything (header-only files; the requests are refused)."""
    from nodes.impl.native_graph import graph

    limit = graph().image_io_max_pixels
    monkeypatch.setenv("OPENCV_IO_MAX_IMAGE_PIXELS", str(limit))
    empty = np.zeros(0, np.float32)
    header = minimal_exr(width, 40_000, {"R": empty, "G": empty, "B": empty})
    request = graph().image_io_decode_request(header, "cv", limit)
    assert request == ((40_000, width, 3) if taken else None)
