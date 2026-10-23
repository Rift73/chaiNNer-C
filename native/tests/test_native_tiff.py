"""TIFF through the vendored libtiff 4.7.1: Load Image's `cv` and `pil-tiff-alpha`
decoders and Save Image's encoder give today's arrays and bytes
(CHAINNER_C_NATIVE_CODECS=0 runs today's OpenCV 5.0 and Pillow 12.3 paths), with cv2
and Pillow out of the native run."""

from __future__ import annotations

import io
import struct
import warnings
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from nodes.impl.native_graph import graph
from nodes.utils.utils import split_file_path
from packages.chaiNNer_standard.image.io import load_image

RNG = np.random.default_rng(42)
H, W = 37, 53
RGBA8 = RNG.integers(0, 256, (H, W, 4), dtype=np.uint8)
RGBA16 = RNG.integers(0, 65536, (H, W, 4), dtype=np.uint16)
FLOAT = RNG.random((H, W, 4), dtype=np.float32)
FLOAT[0, 0] = [np.nan, -1.0, 2.0, np.inf]


def declined(*args, **kwargs):
    raise RuntimeError("today's codec called")


def raw_tiff(
    samples: np.ndarray,
    tags: dict[int, list[int]],
    tile: tuple[int, int] | None = None,
    planar: bool = False,
) -> bytes:
    """An uncompressed little-endian TIFF: one strip (one per plane when `planar`),
    or tiles of `tile` (width, height) samples. Tags given override the ones derived
    from `samples`."""
    h, w = samples.shape[:2]
    channels = samples.shape[2] if samples.ndim == 3 else 1
    if planar:
        blocks = [
            np.ascontiguousarray(plane).tobytes()
            for plane in samples.transpose(2, 0, 1)
        ]
        tags = {284: [2], **tags}
    elif tile is None:
        blocks = [samples.astype(samples.dtype.newbyteorder("<")).tobytes()]
    else:
        tw, th = tile
        padded = np.zeros(
            ((h + th - 1) // th * th, (w + tw - 1) // tw * tw, channels), samples.dtype
        )
        padded[:h, :w] = samples.reshape(h, w, channels)
        blocks = [
            padded[y : y + th, x : x + tw].tobytes()
            for y in range(0, h, th)
            for x in range(0, w, tw)
        ]
    tags = {256: [w], 257: [h], 259: [1], 277: [channels], **tags}
    offsets, counts = (324, 325) if tile else (273, 279)
    if tile:
        tags.update({322: [tile[0]], 323: [tile[1]]})
    else:
        tags.setdefault(278, [h])
    tags[offsets], tags[counts] = [0] * len(blocks), [len(b) for b in blocks]
    data_at = 8 + 2 + 12 * len(tags) + 4
    blob_sizes = [4 * len(v) for v in tags.values() if 4 * len(v) > 4]
    at = data_at + sum(blob_sizes)
    tags[offsets] = [at + sum(len(b) for b in blocks[:i]) for i in range(len(blocks))]
    entries, blobs = b"", b""
    for tag, values in sorted(tags.items()):
        payload = struct.pack(f"<{len(values)}I", *values)
        if len(payload) <= 4:
            entries += struct.pack("<HHI", tag, 4, len(values)) + payload
        else:
            entries += struct.pack("<HHII", tag, 4, len(values), data_at + len(blobs))
            blobs += payload
    return (
        b"II*\0"
        + struct.pack("<IH", 8, len(tags))
        + entries
        + b"\0" * 4
        + blobs
        + b"".join(blocks)
    )


def pillow_tiff(image: Image.Image, **options) -> bytes:
    stream = io.BytesIO()
    image.save(stream, "TIFF", **options)
    return stream.getvalue()


def packed(samples: np.ndarray, bits: int) -> np.ndarray:
    """Rows of `bits`-bit samples packed big-endian, each row padded to a byte."""
    rows = []
    for row in samples.reshape(samples.shape[0], -1):
        bitstring = "".join(format(int(v), f"0{bits}b") for v in row)
        bitstring += "0" * (-len(bitstring) % 8)
        rows.append(
            np.frombuffer(
                int(bitstring, 2).to_bytes(len(bitstring) // 8, "big"), np.uint8
            )
        )
    return np.stack(rows)


def opencv_files() -> dict[str, bytes]:
    files = {}
    for name, data in {
        "u8": RGBA8,
        "u16": RGBA16,
        "f32": FLOAT,
        "i8": RGBA8.view(np.int8),
        "i16": RGBA16.view(np.int16),
        "i32": RGBA16.astype(np.int32) * -3,
        "f64": FLOAT.astype(np.float64),
    }.items():
        for channels in (1, 3, 4):
            image = np.ascontiguousarray(
                data[:, :, 0] if channels == 1 else data[:, :, :channels]
            )
            for compression in (1, 5, 8):
                params = [
                    cv2.IMWRITE_TIFF_COMPRESSION,
                    compression,
                    cv2.IMWRITE_TIFF_ROWSPERSTRIP,
                    5,
                ]
                files[f"cv-{name}-{channels}-{compression}"] = cv2.imencode(
                    ".tif", image, params
                )[1].tobytes()
    return files


def other_files() -> dict[str, bytes]:
    rgb, grey = Image.fromarray(RGBA8[:, :, :3]), Image.fromarray(RGBA8[:, :, 0])
    files = {
        "palette": pillow_tiff(rgb.quantize(29), compression="tiff_lzw"),
        "palette-4bit": raw_tiff(
            packed(RGBA8[:, :, 0] >> 4, 4),
            {256: [W], 258: [4], 262: [3], 320: list(range(0, 65536, 1366))[:48]},
        ),
        "bilevel-g4": pillow_tiff(grey.convert("1"), compression="group4"),
        "cmyk": pillow_tiff(
            Image.frombytes("CMYK", (W, H), RGBA8.tobytes()), compression="packbits"
        ),
        "ycbcr-jpeg": pillow_tiff(rgb, compression="jpeg", quality=80),
        "grey-jpeg": pillow_tiff(grey, compression="jpeg"),
        "min-is-white": raw_tiff(RGBA8[:, :, 1], {258: [8], 262: [0]}),
        "min-is-white-16": raw_tiff(RGBA16[:, :, 1], {258: [16], 262: [0]}),
        "grey-16-from-rgb-samples": raw_tiff(
            RGBA16[:, :, :3], {258: [16] * 3, 262: [1]}
        ),
        "rgba8-associated": raw_tiff(RGBA8, {258: [8] * 4, 262: [2], 338: [1]}),
        "grey8-tiled": raw_tiff(RGBA8[:, :, :1], {258: [8], 262: [1]}, tile=(32, 32)),
        "rgb16-tiled": raw_tiff(
            RGBA16[:, :, :3], {258: [16] * 3, 262: [2]}, tile=(32, 16)
        ),
        "float-tiled": raw_tiff(
            FLOAT[:, :, :3], {258: [32] * 3, 262: [2], 339: [3] * 3}, tile=(16, 16)
        ),
        "rgb12": raw_tiff(
            packed(RGBA16[:, :, :3] >> 4, 12),
            {256: [W], 258: [12] * 3, 262: [2], 277: [3]},
        ),
        "grey10": raw_tiff(
            packed(RGBA16[:, :, 0] >> 6, 10), {256: [W], 258: [10], 262: [1]}
        ),
        "multipage": pillow_tiff(rgb, save_all=True, append_images=[grey]),
    }
    for orientation in range(1, 9):
        files[f"rgb8-orientation-{orientation}"] = raw_tiff(
            RGBA8[:, :, :3], {258: [8] * 3, 262: [2], 274: [orientation]}
        )
        files[f"grey16-orientation-{orientation}"] = raw_tiff(
            RGBA16[:, :, 0], {258: [16], 262: [1], 274: [orientation]}
        )
        files[f"palette-orientation-{orientation}"] = pillow_tiff(
            rgb.quantize(17), tiffinfo={274: orientation}
        )
    return files


@pytest.mark.parametrize("files", [opencv_files, other_files])
def test_load_image_decodes_today_arrays_natively(tmp_path, monkeypatch, files):
    for name, data in files().items():
        path = tmp_path / f"{name}.tif"
        path.write_bytes(data)
        monkeypatch.setenv("CHAINNER_C_NATIVE_CODECS", "0")
        today = load_image.load_image_node(path)[0]
        monkeypatch.delenv("CHAINNER_C_NATIVE_CODECS")
        with monkeypatch.context() as patch:
            patch.setattr(
                load_image,
                "cv2",
                SimpleNamespace(
                    imdecode=declined, imread=declined, IMREAD_UNCHANGED=-1
                ),
            )
            patch.setattr(
                load_image,
                "Image",
                SimpleNamespace(open=declined, MAX_IMAGE_PIXELS=89478485),
            )
            native = load_image.load_image_node(path)[0]
        assert (native.dtype, native.shape) == (today.dtype, today.shape), name
        assert native.tobytes() == today.tobytes(), name


def straight_alpha_files(tmp_path) -> dict[str, bytes]:
    """8-bit RGBA and grey+alpha TIFFs with ExtraSamples (2,): Save Image's own
    (OpenCV plus add_extra_samples) and Pillow's, which load_image's pil-tiff-alpha
    decoder takes."""
    files = {}
    for compression in (1, 5, 8):
        path = tmp_path / f"saved-{compression}.tiff"
        graph().image_io_cv_save(
            {"cv2": cv2, "split_file_path": split_file_path},
            path,
            RGBA8,
            [cv2.IMWRITE_TIFF_COMPRESSION, compression],
        )
        files[path.stem] = path.read_bytes()
    rgba, la = (
        Image.fromarray(RGBA8, "RGBA"),
        Image.fromarray(RGBA8[:, :, ::3].copy(), "LA"),
    )
    for compression in ("raw", "tiff_lzw", "tiff_adobe_deflate", "packbits"):
        files[f"pillow-rgba-{compression}"] = pillow_tiff(rgba, compression=compression)
        files[f"pillow-la-{compression}"] = pillow_tiff(la, compression=compression)
    files["pillow-rgba-bigtiff"] = pillow_tiff(rgba, big_tiff=True)
    files["pillow-rgba-orientation-6"] = pillow_tiff(rgba, tiffinfo={274: 6})
    files["pillow-la-orientation-3"] = pillow_tiff(
        la, compression="tiff_lzw", tiffinfo={274: 3}
    )
    # Without an Orientation tag Pillow turns the image by the XMP packet's (getexif,
    # TiffImageFile.load_end); with one, the tag alone counts.
    files["pillow-rgba-xmp-orientation-6"] = pillow_tiff(rgba, tiffinfo={700: XMP_6})
    files["pillow-la-xmp-orientation-8"] = pillow_tiff(
        la, compression="tiff_lzw", tiffinfo={700: XMP_8}
    )
    files["pillow-rgba-xmp-under-orientation-1"] = pillow_tiff(
        rgba, tiffinfo={274: 1, 700: XMP_6}
    )
    files["pillow-rgba-exif-ifd"] = bytes(with_exif_ifd())
    # Pillow's Exif table gives Orientation no count there: two values load quietly.
    files["pillow-rgba-exif-ifd-two-orientations"] = bytes(with_exif_ifd((274, (1, 1))))
    return files


XMP_6 = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:Description tiff:Orientation="6"/></x:xmpmeta>'
XMP_8 = b"<x:xmpmeta><tiff:Orientation>8</tiff:Orientation></x:xmpmeta>"


def ifd0_value_at(data: bytes | bytearray, tag: int) -> int:
    """Where the value of IFD0's entry `tag` is (a little-endian classic TIFF)."""
    ifd = struct.unpack_from("<I", data, 4)[0]
    count = struct.unpack_from("<H", data, ifd)[0]
    for entry in range(ifd + 2, ifd + 2 + 12 * count, 12):
        if struct.unpack_from("<H", data, entry)[0] == tag:
            return entry + 8
    raise AssertionError(f"no tag {tag}")


def with_exif_ifd(shorts: tuple[int, tuple[int, ...]] | None = None) -> bytearray:
    """A Pillow RGBA TIFF with an Exif IFD after its data: DateTimeOriginal, and a tag of
    one or two SHORT values when given as (tag, values)."""
    data = bytearray(pillow_tiff(Image.fromarray(RGBA8, "RGBA"), tiffinfo={34665: 0}))
    data += bytes(len(data) % 2)
    text = b"2026:10:23 04:00:00\0"
    count = 1 + bool(shorts)
    entries = struct.pack("<HHII", 0x9003, 2, len(text), len(data) + 2 + 12 * count + 4)
    if shorts:
        tag, values = shorts
        inline = struct.pack(f"<{len(values)}H", *values).ljust(4, b"\0")
        entries += struct.pack("<HHI", tag, 3, len(values)) + inline
    struct.pack_into("<I", data, ifd0_value_at(data, 34665), len(data))
    return data + struct.pack("<H", count) + entries + bytes(4) + text


def test_straight_alpha_tiffs_decode_natively(tmp_path, monkeypatch):
    for name, data in straight_alpha_files(tmp_path).items():
        path = tmp_path / f"{name}.tif"
        path.write_bytes(data)
        monkeypatch.setenv("CHAINNER_C_NATIVE_CODECS", "0")
        today = load_image.load_image_node(path)[0]
        monkeypatch.delenv("CHAINNER_C_NATIVE_CODECS")
        assert today.dtype == np.uint8 and today.shape[2] == 4, (
            name
        )  # Pillow's straight alpha
        with monkeypatch.context() as patch:
            patch.setattr(
                load_image,
                "TiffImagePlugin",
                SimpleNamespace(ImageFileDirectory_v2=declined, TiffImageFile=declined),
            )
            patch.setattr(
                load_image,
                "cv2",
                SimpleNamespace(
                    imdecode=declined, imread=declined, IMREAD_UNCHANGED=-1
                ),
            )
            native = load_image.load_image_node(path)[0]
        assert native.shape == today.shape and native.tobytes() == today.tobytes(), name


def corrupt_exif_ifd(kind: str) -> bytes:
    """A Pillow RGBA TIFF whose Exif IFD Pillow's load warns about: its pointer past the
    end, an IFD cut short after its entry count, or two numbers for ExifVersion, which
    Pillow's Exif table gives one value."""
    if kind == "two":
        return bytes(with_exif_ifd((36864, (48, 50))))
    data = with_exif_ifd()
    pointer = ifd0_value_at(data, 34665)
    if kind == "past-the-end":
        struct.pack_into("<I", data, pointer, len(data) - 1)
    else:
        struct.pack_into("<I", data, pointer, len(data))
        data += struct.pack("<H", 3) + bytes(5)
    return bytes(data)


@pytest.mark.parametrize("kind", ["past-the-end", "cut", "two"])
def test_straight_alpha_tiff_with_corrupt_exif_ifd_goes_to_today_path(
    tmp_path, monkeypatch, kind
):
    """Pillow's load reads the Exif IFD (TiffImageFile.load_end) and warns about one it
    cannot read whole or that gives a tag more values than it takes: the native decoder
    declines such a file, so today's path gives the image and the warning."""
    data = corrupt_exif_ifd(kind)
    assert graph().image_io_decode_request(data, "pil-tiff-alpha", H * W) is None
    path = tmp_path / f"exif-{kind}.tif"
    path.write_bytes(data)
    results = []
    for native in (False, True):
        if native:
            monkeypatch.delenv("CHAINNER_C_NATIVE_CODECS")
        else:
            monkeypatch.setenv("CHAINNER_C_NATIVE_CODECS", "0")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            image = load_image.load_image_node(path)[0]
        results.append((image.shape, image.tobytes(), [str(w.message) for w in caught]))
    assert results[0][2], "today's path warns"
    assert results[1] == results[0]


@pytest.mark.parametrize("limit", [H * W - 1, (H * W - 1) // 2])
def test_straight_alpha_tiff_over_pillow_limit_goes_to_today_path(
    tmp_path, monkeypatch, limit
):
    """Above Image.MAX_IMAGE_PIXELS Pillow's load warns (DecompressionBombWarning),
    above twice that it refuses and OpenCV decodes: the native decoder declines both.
    (A one-strip raw file Pillow maps skips the check; declining it changes nothing.)"""
    path = tmp_path / "large.tif"
    path.write_bytes(
        pillow_tiff(Image.fromarray(RGBA8, "RGBA"), compression="tiff_lzw")
    )
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", limit)
    with monkeypatch.context() as patch:
        patch.setattr(
            load_image,
            "TiffImagePlugin",
            SimpleNamespace(ImageFileDirectory_v2=declined, TiffImageFile=declined),
        )
        with pytest.raises(RuntimeError, match="today's codec called"):
            load_image._read_tiff_straight_alpha(path)
    bomb = Image.DecompressionBombWarning if limit == H * W - 1 else None
    monkeypatch.setenv("CHAINNER_C_NATIVE_CODECS", "0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        today = load_image.load_image_node(path)[0]
    monkeypatch.delenv("CHAINNER_C_NATIVE_CODECS")
    with warnings.catch_warnings(record=True) as again:
        warnings.simplefilter("always")
        native = load_image.load_image_node(path)[0]
    kinds = [w.category for w in caught]
    assert (bomb in kinds) == (bomb is not None)
    assert kinds == [w.category for w in again]
    assert native.shape == today.shape and native.tobytes() == today.tobytes()


class Unlimited:
    """load_image's Image with MAX_IMAGE_PIXELS None; Pillow itself keeps its limit."""

    MAX_IMAGE_PIXELS = None

    def __getattr__(self, name: str):
        return getattr(Image, name)


def test_straight_alpha_tiff_failing_allocation_declines(tmp_path, monkeypatch):
    """The array the native decoder asks for cannot be made (NumPy refuses its size):
    the decoder declines, and today's path gives Pillow's own error."""
    side = 2**31 - 1
    short, long = 3, 4
    entries = [
        (256, long, side),
        (257, long, side),
        (258, short, 8),  # BitsPerSample 8 (one value for all four)
        (259, short, 5),  # LZW: Pillow's libtiff load, which checks the size
        (262, short, 2),
        (273, long, 8),
        (277, short, 4),
        (278, long, side),
        (279, long, 1),
        (338, short, 2),
    ]
    # Little-endian: a SHORT value is the low half of the 4-byte value field.
    ifd = struct.pack("<H", len(entries)) + b"".join(
        struct.pack("<HHII", tag, kind, 1, value) for tag, kind, value in entries
    )
    path = tmp_path / "huge.tif"
    path.write_bytes(b"II*\0" + struct.pack("<I", 8) + ifd + bytes(4))
    monkeypatch.setattr(load_image, "Image", Unlimited())
    with pytest.raises(Image.DecompressionBombError):
        load_image._read_tiff_straight_alpha(path)


UNPORTED = {
    # OpenCV reads the first plane's strips as if they held every sample (unset
    # memory): not ported.
    "planar16": raw_tiff(RGBA16[:, :, :3], {258: [16] * 3, 262: [2]}, planar=True),
    # LogLuv (OpenCV's XYZ floats) is not ported.
    "logluv": cv2.imencode(
        ".tif", FLOAT[:, :, :3].clip(0, 1), [cv2.IMWRITE_TIFF_COMPRESSION, 34676]
    )[1].tobytes(),
    # An uncompressed tile of other than a multiple of 1024 bytes fails imdecode;
    # today's imread retry decides.
    "small-tiles": raw_tiff(RGBA8[:, :, :3], {258: [8] * 3, 262: [2]}, tile=(16, 16)),
}


@pytest.mark.parametrize("name", UNPORTED)
def test_unported_tiffs_go_to_today_path(tmp_path, monkeypatch, name):
    data = UNPORTED[name]
    path = tmp_path / "unported.tif"
    path.write_bytes(data)
    monkeypatch.setattr(
        load_image,
        "cv2",
        SimpleNamespace(imdecode=declined, imread=declined, IMREAD_UNCHANGED=-1),
    )
    with pytest.raises(RuntimeError, match="Error reading image"):
        load_image.load_image_node(path)


def save(path, image, params, codec):
    graph().image_io_cv_save(
        {"cv2": codec, "split_file_path": split_file_path}, path, image, params
    )
    return path.read_bytes()


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_save_image_tiff_bytes_natively(tmp_path, monkeypatch, dtype, channels):
    source = {np.uint8: RGBA8, np.uint16: RGBA16, np.float32: FLOAT}[dtype]
    image = np.ascontiguousarray(
        source[:, :, 0] if channels == 1 else source[:, :, :channels]
    )
    native_cv2 = SimpleNamespace(
        imencode=declined, IMWRITE_TIFF_COMPRESSION=cv2.IMWRITE_TIFF_COMPRESSION
    )
    # Save Image's parameters: the compression for U8/U16, none for F32.
    choices = (
        [[]]
        if dtype == np.float32
        else [[cv2.IMWRITE_TIFF_COMPRESSION, c] for c in (1, 5, 8)]
    )
    for params in choices:
        monkeypatch.setenv("CHAINNER_C_NATIVE_CODECS", "0")
        today = save(tmp_path / "today.tiff", image, params, cv2)
        monkeypatch.delenv("CHAINNER_C_NATIVE_CODECS")
        assert save(tmp_path / "native.tiff", image, params, native_cv2) == today, (
            params
        )
