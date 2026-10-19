"""Native BMP and PxM decoding (OpenCV 5.0's BmpDecoder and PxMDecoder ported to C): Load Image's result is today's.

Each file is generated here. Today's path (CHAINNER_C_NATIVE_CODECS=0) is the oracle; the native path must give
the same array, and run on its own: a copy of load_image whose cv2 and Image raise on any decode."""

from __future__ import annotations

import struct
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from test_image_io import modules


def forbidden(*args, **kwargs):
    raise AssertionError("today's decoder ran")


LOAD = modules(True).load
NATIVE_ONLY = modules(True).load
NATIVE_ONLY.cv2 = SimpleNamespace(
    imdecode=forbidden, imread=forbidden, IMREAD_UNCHANGED=cv2.IMREAD_UNCHANGED
)
NATIVE_ONLY.Image = SimpleNamespace(open=forbidden, MAX_IMAGE_PIXELS=89478485)


def bmp(
    width,
    height,
    bpp,
    pixels,
    compression=0,
    palette=b"",
    header=40,
    clrused=0,
    masks=(),
    gap=b"",
):
    if header == 12:
        info = struct.pack("<IHHHH", 12, width, height, 1, bpp)
    else:
        info = struct.pack(
            "<IiiHHIIiiII",
            header,
            width,
            height,
            1,
            bpp,
            compression,
            len(pixels),
            0,
            0,
            clrused,
            0,
        )
        info = (info + struct.pack(f"<{len(masks)}I", *masks)).ljust(header, b"\0")
    offset = 14 + len(info) + len(gap) + len(palette)
    body = info + gap + palette + pixels
    return b"BM" + struct.pack("<IHHI", 14 + len(body), 0, 0, offset) + body


def entries(colours, size=4):
    return b"".join(bytes(c) + b"\0" * (size - 3) for c in colours)


def rows(width, height, bpp, seed):
    pitch = (width * bpp + 31) // 32 * 4
    return (
        np.random.default_rng(seed).integers(0, 256, pitch * height, np.uint8).tobytes()
    )


COLOUR16 = entries([(i * 16, 255 - i * 16, i * 37 % 256) for i in range(16)])
GREY16 = entries([(i * 17,) * 3 for i in range(16)])
COLOUR256 = entries([(i, i * 3 % 256, 255 - i) for i in range(256)])
GREY256 = entries([(i,) * 3 for i in range(256)])
BW = entries([(0, 0, 0), (255, 255, 255)])
TWO = entries([(10, 20, 30), (200, 100, 50)])
MASKS32 = {
    "8888": (0xFF0000, 0xFF00, 0xFF, 0xFF000000),
    "888": (0xFF0000, 0xFF00, 0xFF, 0),
    "1010102": (0x3FF00000, 0xFFC00, 0x3FF, 0xC0000000),
    "nored": (0, 0xFF00, 0xFF, 0xFF000000),
    "odd": (0xFFFFFFFF, 0xFFFF, 0xF0F000, 0x80000001),
}
# RLE streams for a 7x3 image as their operations: encoded runs (count, colour), absolute runs (0, count, indices,
# padded to a word), end of line, end of bitmap, delta (0, 2, dx, dy).
EOL, EOF = (0, 0), (0, 1)
RLE8 = {
    "basic": [
        (3, 5),
        (0, 4, 1, 2, 3, 4),
        EOL,
        (7, 9),
        EOL,
        (0, 3, 10, 11, 12, 0),
        (4, 200),
        EOF,
    ],
    "exact": [(7, 1), (7, 2), (7, 3)],
    "exact_eol": [(7, 1), EOL, (7, 2), EOL, (7, 3), EOL],
    "absolute_end": [(0, 7, 1, 2, 3, 4, 5, 6, 7, 0), EOL, (7, 9), EOF],
    "early_eof": [(2, 8), EOF],
    "delta": [(2, 8), (0, 2, 3, 1), (2, 9), EOL, (1, 1), EOF],
    "delta_wrap": [(5, 8), (0, 2, 4, 0), (2, 9), EOF],
    "delta_out": [(2, 8), (0, 2, 1, 9)],
}
RLE4 = {
    "basic": [
        (3, 0x12),
        (0, 3, 0x12, 0x30),
        (1, 0x45),
        EOL,
        (7, 0x9A),
        EOL,
        (0, 5, 0xAB, 0xCD, 0xE0, 0),
        EOL,
    ],
    "eof_is_eol": [(7, 0x11), EOF, (7, 0x22), EOF, (7, 0x33), EOF],
    "delta": [(2, 0x18), (0, 2, 3, 2), (2, 0x29), EOL, (7, 0x31), EOL, (1, 1), EOL],
    "delta_wrap": [(5, 0x81), (0, 2, 4, 0), (2, 0x92), EOL, (7, 3), EOL],
}


def rle(operations):
    return bytes(byte for operation in operations for byte in operation)


def bmp_cases():
    for channels in [1, 3, 4]:  # grey palette, 24-bit, V5 BITFIELDS with alpha
        value = (
            (np.arange(9 * 13 * channels) * 7 % 256)
            .astype(np.uint8)
            .reshape(9, 13, channels)
        )
        yield f"opencv_{channels}", cv2.imencode(".bmp", value)[1].tobytes()
    palettes = {
        "bw": (1, BW),
        "two": (1, TWO),
        "c16": (4, COLOUR16),
        "g16": (4, GREY16),
        "c256": (8, COLOUR256),
        "g256": (8, GREY256),
    }
    for w, h in [(1, 1), (5, 3), (13, 7)]:
        for tag, (bpp, palette) in palettes.items():
            px = rows(w, h, bpp, w + bpp)
            yield f"{w}x{h}_{tag}", bmp(w, h, bpp, px, palette=palette)
            yield f"{w}x{h}_{tag}_topdown", bmp(w, -h, bpp, px, palette=palette)
            os2 = b"".join(palette[i : i + 3] for i in range(0, len(palette), 4))
            yield f"{w}x{h}_{tag}_os2", bmp(w, h, bpp, px, palette=os2, header=12)
        yield (
            f"{w}x{h}_8_clrused5",
            bmp(w, h, 8, rows(w, h, 8, 3), palette=COLOUR256[:20], clrused=5),
        )
        for bpp in [16, 24, 32]:
            px = rows(w, h, bpp, w * bpp)
            yield f"{w}x{h}_{bpp}", bmp(w, h, bpp, px)
            yield f"{w}x{h}_{bpp}_topdown", bmp(w, -h, bpp, px, header=124)
        yield f"{w}x{h}_24_os2", bmp(w, h, 24, rows(w, h, 24, 5), header=12)
        yield f"{w}x{h}_32_os2", bmp(w, h, 32, rows(w, h, 32, 5), header=12)
        px = rows(w, h, 16, 6)
        yield (
            f"{w}x{h}_565",
            bmp(
                w, h, 16, px, compression=3, gap=struct.pack("<3I", 0xF800, 0x7E0, 0x1F)
            ),
        )
        yield (
            f"{w}x{h}_555",
            bmp(
                w, h, 16, px, compression=3, gap=struct.pack("<3I", 0x7C00, 0x3E0, 0x1F)
            ),
        )
        px = rows(w, h, 32, 7)
        for name, masks in MASKS32.items():
            yield (
                f"{w}x{h}_32_{name}_v5",
                bmp(w, h, 32, px, compression=3, header=124, masks=masks),
            )
        yield (
            f"{w}x{h}_32_bitfields_v3",
            bmp(
                w,
                h,
                32,
                px,
                compression=3,
                gap=struct.pack("<3I", 0xFF, 0xFF00, 0xFF0000),
            ),
        )
    for name, stream in RLE8.items():
        for tag, palette in [("colour", COLOUR256), ("grey", GREY256)]:
            yield (
                f"rle8_{name}_{tag}",
                bmp(7, 3, 8, rle(stream), compression=1, palette=palette),
            )
        yield (
            f"rle8_{name}_topdown",
            bmp(7, -3, 8, rle(stream), compression=1, palette=COLOUR256),
        )
    for name, stream in RLE4.items():
        for tag, palette in [("colour", COLOUR16), ("grey", GREY16)]:
            yield (
                f"rle4_{name}_{tag}",
                bmp(7, 3, 4, rle(stream), compression=2, palette=palette),
            )


def pxm_cases():
    image = np.arange(13 * 9 * 3, dtype=np.uint8).reshape(9, 13, 3) * 7
    for binary in [0, 1]:
        for ext, value in [
            (".pbm", image[..., 0]),
            (".pgm", image[..., 1]),
            (".ppm", image),
            (".pgm", image[..., 2].astype(np.uint16) * 257),
            (".ppm", image.astype(np.uint16) * 129),
        ]:
            yield (
                f"cv{ext}_{value.dtype}_{binary}",
                cv2.imencode(ext, value, [cv2.IMWRITE_PXM_BINARY, binary])[1].tobytes(),
            )
    samples = bytes(range(0, 250, 7)) * 2
    for maxval in [1, 2, 15, 100, 255, 256, 1023, 65535]:
        values = [v * 2654435761 % (maxval + 3) for v in range(24)]  # some above maxval
        text = " ".join(map(str, values)).encode()
        yield f"p2_{maxval}", b"P2\n# grey\n6 4\n%d\n" % maxval + text + b"\n"
        yield f"p3_{maxval}", b"P3 4 2 %d\n" % maxval + text + b"\n"
        size = 2 if maxval > 255 else 1
        yield (
            f"p5_{maxval}",
            b"P5\n6 4\n%d\n" % maxval + samples[: 24 * size] + b"\0" * 8,
        )
        yield (
            f"p6_{maxval}",
            b"P6\r\n4\t2\v%d\f" % maxval + samples[: 24 * size] + b"\0" * 8,
        )
    yield "p1", b"P1\n#c\r5 3\n1 0 1 0 1\n0 1 # x\n0 1 0\n10111" + b" " * 8
    yield "p1_digits", b"P1\n5 3\n1 2 3 0 9\n0 1 0 1 0\n1 1 7 1 1" + b" " * 8
    yield "p4", b"P4\n13 3\n" + bytes([0xAA, 0xF0, 0x0F, 0xFF, 0x55, 0x81]) + b"\0" * 16
    yield "comment_high_bytes", b"P2\n# caf\xe9 \xa0\n4 2\n255\n0 50 255 6 7 99 51 2\n"


DECLINED = {
    "truncated_rows": bmp(5, 3, 24, rows(5, 3, 24, 1)[:-1]),
    "truncated_palette": bmp(5, 3, 8, b"\0" * 64, palette=COLOUR256[:300]),
    "rle8_run_past_row": bmp(
        7, 3, 8, bytes([8, 1, 0, 1]), compression=1, palette=COLOUR256
    ),
    "rle8_truncated": bmp(
        7, 3, 8, bytes([7, 1, 3, 2] + [0] * 2), compression=1, palette=COLOUR256
    )[:-2],
    "rle4_run_after_row_end": bmp(
        7, 3, 4, bytes([7, 0x11, 1, 0x22, 0, 0]), compression=2, palette=COLOUR16
    ),
    "bitfields_444": bmp(
        5,
        3,
        16,
        rows(5, 3, 16, 2),
        compression=3,
        gap=struct.pack("<3I", 0xF00, 0xF0, 0xF),
    ),
    "jpeg_compression": bmp(5, 3, 24, rows(5, 3, 24, 1), compression=4),
    "too_wide": bmp(1 << 20 | 1, 1, 24, b"\0" * 64),
    "p2_no_terminator": b"P2\n4 2\n255\n0 50 255 6\n7 99 51 2",
    "p2_comment_after_number": b"P2\n4#x\n2\n255\n0 50 255 6 7 99 51 2\n",
    "p2_high_byte_separator": b"P2\n4\xa02\n255\n0 50 255 6 7 99 51 2\n" + b" " * 4,
    "p5_maxval_65536": b"P5\n4 2\n65536\n" + b"\0" * 32,
    "p6_truncated": b"P6\n3 2\n255\n" + bytes(range(17)),
    "p7": b"P7\nWIDTH 2\nHEIGHT 2\nDEPTH 1\nMAXVAL 255\nTUPLTYPE GRAYSCALE\nENDHDR\n"
    + bytes(4),
}


def write(path, data):
    # A file below 32 bytes (imdecode's signature length) declines: pad with bytes both decoders leave unread.
    path.write_bytes(data.ljust(32, b"\0"))


def outcome(module, path):
    try:
        array = module.load_image_node(path)[0]
    except Exception as error:
        return type(error), str(error)
    return array.dtype.str, array.shape, array.tobytes()


def today(monkeypatch, path):
    with monkeypatch.context() as patch:
        patch.setenv("CHAINNER_C_NATIVE_CODECS", "0")
        return outcome(LOAD, path)


@pytest.mark.parametrize("suffix", [".bmp", ".dib"])
def test_bmp_decodes_natively_as_today(tmp_path, monkeypatch, suffix):
    monkeypatch.delenv("CHAINNER_C_NATIVE_CODECS", raising=False)
    names = []
    for name, data in bmp_cases():
        path = tmp_path / f"{name}{suffix}"
        write(path, data)
        expected = today(monkeypatch, path)
        assert isinstance(expected[0], str), (name, expected)
        assert outcome(NATIVE_ONLY, path) == expected, name
        names.append(name)
    assert len(names) > 100


@pytest.mark.parametrize("suffix", [None, ".pnm", ".pxm", ".bmp"])
def test_pxm_decodes_natively_as_today(tmp_path, monkeypatch, suffix):
    monkeypatch.delenv("CHAINNER_C_NATIVE_CODECS", raising=False)
    for name, data in pxm_cases():
        ext = suffix or {b"1": ".pbm", b"4": ".pbm", b"2": ".pgm", b"5": ".pgm"}.get(
            data[1:2], ".ppm"
        )
        path = tmp_path / f"{name}{ext}"
        write(path, data)
        expected = today(monkeypatch, path)
        assert isinstance(expected[0], str), (name, expected)
        assert outcome(NATIVE_ONLY, path) == expected, name


@pytest.mark.parametrize("name", sorted(DECLINED))
def test_declined_files_take_todays_path(tmp_path, monkeypatch, name):
    monkeypatch.delenv("CHAINNER_C_NATIVE_CODECS", raising=False)
    data = DECLINED[name]
    path = tmp_path / (f"{name}.bmp" if data[:2] == b"BM" else f"{name}.pgm")
    path.write_bytes(data)
    assert not isinstance(outcome(NATIVE_ONLY, path)[0], str)
    assert outcome(LOAD, path) == today(monkeypatch, path)
