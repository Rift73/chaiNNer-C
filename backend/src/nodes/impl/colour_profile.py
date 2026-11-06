"""An image's embedded ICC colour profile to sRGB, with the relative colorimetric result
of ImageCms (Little CMS). Matrix and curve profiles (Adobe RGB, Display P3, ProPhoto, grey
gamma, ...) convert in float32 here, as ImageCms has no 16-bit or float RGB image to take;
a table-based RGB profile goes through ImageCms's 8-bit transform."""

from __future__ import annotations

import io
import struct
from functools import lru_cache
from typing import NamedTuple

import numpy as np
from sanic.log import logger

from nodes.utils.utils import get_h_w_c

from .image_utils import normalize

# rXYZ, gXYZ, bXYZ (columns) of ImageCms's sRGB profile, inverted.
_XYZ_D50_TO_LINEAR_SRGB = np.linalg.inv(
    np.array(
        [
            [0.4360352, 0.3851166, 0.1430511],
            [0.2224884, 0.7169037, 0.0606079],
            [0.0139160, 0.0970612, 0.7139130],
        ]
    )
)
# An sRGB profile's colorants, which profiles adapt and round in their own ways, stay this
# close to sRGB's, and its curve leaves every 8-bit value where it is.
_SRGB_MATRIX_TOLERANCE = 1e-3
_SRGB_CURVE_TOLERANCE = 0.5 / 255
_TABLE_TAGS = frozenset({b"A2B0", b"A2B1", b"D2B0", b"D2B1"})


def _tag_table(icc: bytes) -> dict[bytes, bytes]:
    count = struct.unpack_from(">I", icc, 128)[0]
    table: dict[bytes, bytes] = {}
    for i in range(count):
        signature, offset, size = struct.unpack_from(">4sII", icc, 132 + 12 * i)
        table[signature] = icc[offset : offset + size]
    return table


def _colorant(tag: bytes) -> np.ndarray:
    return np.array(struct.unpack_from(">3i", tag, 8)) / 65536


def _curve(tag: bytes) -> np.ndarray:
    """The linear values of a tone curve (curv or para) at the 65536 16-bit codes."""
    codes = np.linspace(0, 1, 65536)
    kind = tag[:4]
    if kind == b"curv":
        count = struct.unpack_from(">I", tag, 8)[0]
        if count == 0:
            values = codes
        elif count == 1:
            values = codes ** (struct.unpack_from(">H", tag, 12)[0] / 256)
        else:
            table = np.frombuffer(tag, ">u2", count, 12) / 65535
            values = np.interp(codes, np.linspace(0, 1, count), table)
    elif kind == b"para":
        function = struct.unpack_from(">H", tag, 8)[0]
        names = {0: "g", 1: "gab", 2: "gabc", 3: "gabcd", 4: "gabcdef"}[function]
        read = struct.unpack_from(f">{len(names)}i", tag, 12)
        p = dict(zip(names, (value / 65536 for value in read)))
        g, a, b = p["g"], p.get("a", 1.0), p.get("b", 0.0)
        if function in (1, 2):
            # Below -b/a these are 0 (type 1) or c (type 2), and c is added above it.
            start, slope, low, high = -b / a, 0.0, p.get("c", 0.0), p.get("c", 0.0)
        else:
            start, slope = p.get("d", 0.0), p.get("c", 0.0)
            low, high = p.get("f", 0.0), p.get("e", 0.0)
        powered = np.maximum(a * codes + b, 0) ** g + high
        values = np.where(codes >= start, powered, slope * codes + low)
    else:
        raise ValueError(f"unknown tone curve type {kind!r}")
    return np.clip(values, 0, 1).astype(np.float32)


def _encode_srgb(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0, 1)
    return np.where(
        linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055
    )


class _Shaper(NamedTuple):
    """A matrix and curve profile: one curve for grey, else red's, green's and blue's
    with the matrix from their linear values to linear sRGB (None for grey)."""

    curves: tuple[np.ndarray, ...]
    to_srgb: np.ndarray | None
    is_srgb: bool = False

    def apply(self, planes: list[np.ndarray]) -> list[np.ndarray]:
        linear = [
            curve[np.rint(plane * 65535).astype(np.uint16)]
            for curve, plane in zip(self.curves, planes)
        ]
        if self.to_srgb is None:
            return [_encode_srgb(linear[0])]
        m = self.to_srgb.astype(np.float32)
        r, g, b = linear
        return [
            _encode_srgb(m[row, 0] * r + m[row, 1] * g + m[row, 2] * b)
            for row in range(3)
        ]

    def same_as_srgb(self) -> bool:
        if self.to_srgb is not None:
            distance = np.abs(self.to_srgb - np.eye(3)).max()
            if distance > _SRGB_MATRIX_TOLERANCE:
                return False
        codes = np.linspace(0, 1, 65536, dtype=np.float32)
        return all(
            np.abs(_encode_srgb(curve) - codes).max() < _SRGB_CURVE_TOLERANCE
            for curve in self.curves
        )


@lru_cache(maxsize=8)
def _shaper_of(icc: bytes) -> _Shaper | None:
    """The profile's matrix and curves; None for a table-based profile. A batch's images
    usually share one profile, so its curves are kept."""
    tags = _tag_table(icc)
    grey = icc[16:20] == b"GRAY"
    if tags.keys() & _TABLE_TAGS or (b"kTRC" if grey else b"rTRC") not in tags:
        return None  # ImageCms takes a profile's tables over its curves
    if grey:
        shaper = _Shaper((_curve(tags[b"kTRC"]),), None)
    else:
        colorants = np.stack(
            [
                _colorant(tags[b"rXYZ"]),
                _colorant(tags[b"gXYZ"]),
                _colorant(tags[b"bXYZ"]),
            ],
            axis=1,
        )
        curves = _curve(tags[b"rTRC"]), _curve(tags[b"gTRC"]), _curve(tags[b"bTRC"])
        shaper = _Shaper(curves, _XYZ_D50_TO_LINEAR_SRGB @ colorants)
    return shaper._replace(is_srgb=shaper.same_as_srgb())


def _through_imagecms(rgb: np.ndarray, icc: bytes) -> np.ndarray:
    # Here, so that loading the nodes does not load Little CMS for this rare path.
    from PIL import Image, ImageCms

    try:
        transform = ImageCms.buildTransform(
            ImageCms.ImageCmsProfile(io.BytesIO(icc)),
            ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")),
            "RGB",
            "RGB",
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            # Its optimization resamples the pipeline on a grid that misses the shadows.
            flags=ImageCms.Flags.NOOPTIMIZE,
        )
    except ImageCms.PyCMSError as error:
        raise ValueError(str(error)) from error
    im = Image.fromarray(np.rint(rgb * 255).astype(np.uint8))
    return np.asarray(transform.apply(im), dtype=np.float32) / 255


def _is_grey_with_alpha(image: np.ndarray) -> bool:
    """Load Image gives grey + alpha as BGRA, with the grey in B, G and R."""
    return (
        get_h_w_c(image)[2] == 4
        and np.array_equal(image[:, :, 0], image[:, :, 1])
        and np.array_equal(image[:, :, 0], image[:, :, 2])
    )


def convert_to_srgb(image: np.ndarray, icc: bytes, path: object) -> np.ndarray:
    """The BGR(A) or grey image in sRGB as float32 with alpha as it was; the image itself
    when the profile is sRGB's, or is none it can be converted from (a warning says why)."""
    channels = get_h_w_c(image)[2]
    grey = icc[16:20] == b"GRAY"
    fits = (
        (channels == 1 or _is_grey_with_alpha(image))
        if grey
        else (icc[16:20] == b"RGB " and channels != 1)
    )
    if not fits or icc[36:40] != b"acsp":
        logger.warning(
            f"Not converted to sRGB: a colour profile of space {icc[16:20]!r} does not"
            f" fit an image of {channels} channels: {path}"
        )
        return image
    try:
        shaper = _shaper_of(icc)
        if shaper is not None and shaper.is_srgb:
            return image
        pixels = normalize(image)
        if grey:
            if shaper is None:
                raise ValueError("a grey profile without a tone curve")
            if channels == 1:
                return shaper.apply([pixels.reshape(pixels.shape[:2])])[0]
            value = shaper.apply([pixels[:, :, 0]])[0]
            return np.dstack([value, value, value, pixels[:, :, 3]])
        rgb = pixels[:, :, 2::-1]
        if shaper is not None:
            r, g, b = shaper.apply([rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]])
        else:
            if image.dtype != np.uint8:
                logger.warning(
                    f"A table-based colour profile converts at 8 bits per channel: {path}"
                )
            converted = _through_imagecms(rgb, icc)
            r, g, b = converted[:, :, 0], converted[:, :, 1], converted[:, :, 2]
    except (ValueError, KeyError, ZeroDivisionError, OSError, struct.error):
        logger.warning(
            f"Not converted to sRGB: the colour profile cannot be read: {path}",
            exc_info=True,
        )
        return image
    return np.dstack([b, g, r] if channels == 3 else [b, g, r, pixels[:, :, 3]])
