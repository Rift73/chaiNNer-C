from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import TypedDict

import cv2
import numpy as np

import navi
from api import BaseOutput, BroadcastData, InputId, OutputKind
from events import PreviewUrl

from ...impl.image_utils import normalize, to_uint8
from ...impl.native_buffers import normalized_readonly, register_normalized_copy
from ...impl.native_graph import graph
from ...impl.resize import ResizeFilter, resize
from ...utils.format import format_image_with_channels
from ...utils.utils import get_h_w_c, round_half_up


class LargeImageBroadcast(TypedDict):
    """LargeImageOutput.get_broadcast_data: the previews (each {"width", "height", "url"}, largest first) and the
    image's size."""

    previews: list[dict[str, object]]
    height: int
    width: int
    channels: int


class NumPyOutput(BaseOutput[np.ndarray]):
    """Output a NumPy array"""

    def __init__(
        self,
        output_type: navi.ExpressionJson,
        label: str,
        kind: OutputKind = "generic",
        has_handle: bool = True,
    ):
        super().__init__(
            output_type,
            label,
            kind=kind,
            has_handle=has_handle,
            associated_type=np.ndarray,
        )

    def enforce(self, value: object) -> np.ndarray:
        assert isinstance(value, np.ndarray)
        return value


class ImageOutput(NumPyOutput):
    def __init__(
        self,
        label: str = "Image",
        *,
        image_type: navi.ExpressionJson = "Image",
        kind: OutputKind = "generic",
        has_handle: bool = True,
        channels: int | None = None,
        shape_as: int | InputId | None = None,
        size_as: int | InputId | None = None,
        assume_normalized: bool = False,
    ):
        if channels is not None:
            image_type = navi.intersect_with_error(
                image_type, navi.Image(channels=channels)
            )
        if shape_as is not None:
            image_type = navi.intersect_with_error(image_type, f"Input{shape_as}")
        if size_as is not None:
            image_type = navi.intersect_with_error(
                image_type, navi.Image(size_as=f"Input{size_as}")
            )
        super().__init__(image_type, label, kind=kind, has_handle=has_handle)
        self.channels: int | None = channels
        self.assume_normalized: bool = assume_normalized
        if shape_as is not None:
            self.as_passthrough_of(shape_as)

    def get_broadcast_data(self, value: np.ndarray) -> BroadcastData:
        h, w, c = get_h_w_c(value)
        return {"height": h, "width": w, "channels": c}

    def get_broadcast_type(self, value: np.ndarray):
        h, w, c = get_h_w_c(value)
        return navi.Image(width=w, height=h, channels=c)

    def enforce(self, value: object) -> np.ndarray:
        assert isinstance(value, np.ndarray)
        h, w, c = get_h_w_c(value)
        if h == 0 or w == 0:
            raise ValueError(
                f"The output {self.label} returned an empty image (w={w} h={h}). This is a bug in the implementation of the node. Please report this bug."
            )
        if self.channels is not None and c != self.channels:
            expected = format_image_with_channels([self.channels])
            actual = format_image_with_channels([c])
            raise ValueError(
                f"The output {self.label} was supposed to return {expected} but actually returned {actual}. This is a bug in the implementation of the node. Please report this bug."
            )
        if c == 1 and value.ndim == 3:
            value = value[:, :, 0]
        normalized_copy = False
        if not self.assume_normalized:
            borrowed = normalized_readonly(value)
            if borrowed is None:
                value = normalize(value)
                normalized_copy = True
            else:
                value = borrowed
        assert value.dtype == np.float32, (
            f"The output {self.label} did not return a normalized image. This is a bug in the implementation of the node. Please report this bug.\n\nTo the author of this node: Either use `normalize` or remove `assume_normalized=True` from this output."
        )
        value.setflags(write=False)
        if normalized_copy:
            register_normalized_copy(value)
        return value


# The most pixels a full-size preview has (8192 x 8192; image_preview.hpp's full_size_pixels): a 16K picture as
# base64 inside one event is too large to send and decode.
FULL_SIZE_PREVIEW_PIXELS = 8192 * 8192
# get_broadcast_data's preview sizes, largest first, and how far past a size an image may be and still use it.
PREVIEW_SIZES = (2048, 1024, 512, 256)
PREVIEW_SIZE_GRACE = 1.2
# One node's previews travel in one event, which the window parses as one JavaScript string (at most 536,870,888
# characters in V8). Its previews may take at most NODE_PREVIEW_CHARS characters together (90 % of that), counted
# as their pixels times PREVIEW_CENTICHARS_PER_PIXEL / 100: the base64 of incompressible RGBA as a PNG, the most a
# preview takes (5.336 measured on RGBA noise with the app's encoder on 2026-11-20; stored deflate's 4 bytes a pixel
# and a filter byte a row give 5.334; a JPEG tier takes fewer). image_preview.hpp holds the same constants.
PREVIEW_CENTICHARS_PER_PIXEL = 534
NODE_PREVIEW_CHARS = 483_000_000


def preview_shapes(h: int, w: int) -> list[tuple[int, int]]:
    """The (height, width) of an h x w image's previews other than the full-size one, in get_broadcast_data's
    order: from the largest size that applies, each from the one before, resized when a side exceeds its size by
    more than the grace. _previews encodes exactly these; image_preview.cpp's tier_shapes is the same."""
    image_size = max(h, w)
    start = len(PREVIEW_SIZES) - 1
    for i, size in enumerate(PREVIEW_SIZES):
        if size <= image_size <= size * PREVIEW_SIZE_GRACE:
            start = i
            break
        if image_size > size:
            start = max(0, i - 1)
            break
    shapes = []
    for size in PREVIEW_SIZES[start:]:
        max_size = size * PREVIEW_SIZE_GRACE
        if w > max_size or h > max_size:
            f = max(w / size, h / size)
            h, w = max(1, round_half_up(h / f)), max(1, round_half_up(w / f))
        shapes.append((h, w))
    return shapes


def full_size_cost(h: int, w: int) -> int:
    """The pixels the full-size preview of an h x w image holds (LargeImageOutput.full_size): its own when it is
    larger than the largest preview size allows and has at most FULL_SIZE_PREVIEW_PIXELS, else 0 (it gets none);
    image_preview.hpp's full_size_cost."""
    pixels = h * w
    full = max(h, w) > PREVIEW_SIZES[0] * PREVIEW_SIZE_GRACE
    return pixels if full and pixels <= FULL_SIZE_PREVIEW_PIXELS else 0


def full_size_previews(
    values: Sequence[object], outputs: Sequence[BaseOutput]
) -> dict[int, bool]:
    """Whether each of a node's full-size LargeImageOutputs, by position, gets its full-size preview
    (get_broadcast_data's full_size), so that the node's event stays one string (NODE_PREVIEW_CHARS): every
    LargeImageOutput's ordinary previews count, and the full-size ones are granted in output order while they fit
    (Run::broadcast_now's rule); an image past that gets the other previews alone. Ten large images share about
    34 M pixels of full-size previews, while one image of FULL_SIZE_PREVIEW_PIXELS keeps its own."""
    shown = [
        (position, output, get_h_w_c(value))
        for position, (value, output) in enumerate(zip(values, outputs))
        if isinstance(output, LargeImageOutput) and isinstance(value, np.ndarray)
    ]
    pixels = sum(
        h * w
        for _, _, (height, width, _) in shown
        for h, w in preview_shapes(height, width)
    )
    granted: dict[int, bool] = {}
    for position, output, (h, w, _) in shown:
        if not output.full_size:
            continue
        cost = full_size_cost(h, w)
        granted[position] = (
            pixels + cost
        ) * PREVIEW_CENTICHARS_PER_PIXEL <= NODE_PREVIEW_CHARS * 100
        if granted[position]:
            pixels += cost
    return granted


def preview_encode(
    img: np.ndarray, shape: tuple[int, int], lossless: bool = False
) -> tuple[str, np.ndarray]:
    """
    The image as a preview of `shape` (height, width; a preview_shapes entry, or its own for the full-size one),
    resized when it has another, so the preview loads faster and doesn't lag the UI
    """
    h, w, c = get_h_w_c(img)
    if (h, w) != shape:
        img = resize(img, (shape[1], shape[0]), ResizeFilter.BOX)
    image_format = "png" if c > 3 or lossless else "jpg"
    _, encoded_img = cv2.imencode(f".{image_format}", to_uint8(img, normalized=True))
    base64_img = base64.b64encode(encoded_img).decode("utf8")
    return (f"data:image/{image_format};base64,{base64_img}", img)


class LargeImageOutput(ImageOutput):
    def __init__(
        self,
        label: str = "Image",
        image_type: navi.ExpressionJson = "Image",
        kind: OutputKind = "large-image",
        has_handle: bool = True,
        assume_normalized: bool = False,
        full_size: bool = False,
    ):
        super().__init__(
            label,
            image_type=image_type,
            kind=kind,
            has_handle=has_handle,
            assume_normalized=assume_normalized,
        )
        # An image larger than the largest preview size allows, of at most FULL_SIZE_PREVIEW_PIXELS, also
        # broadcasts a preview at its own size (first, lossless); the smaller ones are then all lossy.
        self.full_size: bool = full_size

    def get_broadcast_data(
        self,
        value: np.ndarray,
        full_size: bool | None = None,
        sizes: Sequence[int] = (),
    ) -> LargeImageBroadcast:
        """full_size: whether the image gets its full-size preview (full_size_previews' answer for the node), by
        default the output's own full_size. sizes: the widths of the views that show it (a seek request's
        previewSizes): only their previews are made (made_previews); empty: every one."""
        img = value
        h, w, c = get_h_w_c(img)
        full = self.full_size if full_size is None else full_size
        # The same previews natively (image_preview.cpp), without the GIL; None when it declines. Without sizes the
        # call is the one a native module older than sizes takes too (a Python update ahead of its native module
        # keeps every other preview).
        native = graph().image_io_previews
        previews = (
            native(img, full_size=full, sizes=list(sizes))
            if sizes
            else native(img, full_size=full)
        )
        if previews is None:
            previews = _previews(img, full, sizes)
        # /sse splices a PreviewUrl into its text, unescaped.
        for preview in previews:
            preview["url"] = PreviewUrl(preview["url"])
        return {"previews": previews, "height": h, "width": w, "channels": c}


def made_previews(h: int, w: int, sizes: Sequence[int]) -> tuple[list[bool], bool]:
    """Which of an h x w image's preview_shapes are made for views `sizes` wide, and whether its full-size preview
    may be (when it has one): for each width the smallest tier at least that wide (the largest when none is), and
    the full-size one for a width past the largest tier's; every one for no sizes. image_preview.cpp's previews
    chooses the same."""
    shapes = preview_shapes(h, w)
    if not sizes:
        return [True] * len(shapes), True
    made = [False] * len(shapes)
    for size in sizes:
        wide = [n for n, (_, width) in enumerate(shapes) if width >= size]
        made[wide[-1] if wide else 0] = True
    return made, any(size > shapes[0][1] for size in sizes)


def _previews(
    img: np.ndarray, full_size: bool = False, sizes: Sequence[int] = ()
) -> list[dict[str, object]]:
    h, w, _ = get_h_w_c(img)
    full = full_size and full_size_cost(h, w) > 0
    made, full_shown = made_previews(h, w, sizes)
    previews = []
    if full and full_shown:
        url, _ = preview_encode(img, (h, w), lossless=True)
        previews.append({"width": w, "height": h, "url": url})
    last_encoded = img
    shapes = preview_shapes(h, w)
    last = max(n for n, chosen in enumerate(made) if chosen)
    for n, shape in enumerate(shapes[: last + 1]):
        if not made[n]:
            # Each tier from the one before, as when it is made.
            if get_h_w_c(last_encoded)[:2] != shape:
                last_encoded = resize(
                    last_encoded, (shape[1], shape[0]), ResizeFilter.BOX
                )
            continue
        url, last_encoded = preview_encode(
            last_encoded, shape, lossless=n == 0 and not full
        )
        previews.append({"width": shape[1], "height": shape[0], "url": url})
    return previews
