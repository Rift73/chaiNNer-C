from __future__ import annotations

from enum import Enum

import numpy as np

from nodes.groups import Condition, if_enum_group, if_group
from nodes.impl.blend import BlendMode
from nodes.impl.color.color import Color
from nodes.impl.native_composite import canvas, dimensions
from nodes.properties.inputs import (
    BlendModeDropdown,
    BoolInput,
    EnumInput,
    ImageInput,
    NumberInput,
    SliderInput,
)
from nodes.properties.outputs import ImageOutput

from .. import compositing_group


class BlendOverlayPosition(Enum):
    TOP_LEFT = "top_left"
    TOP = "top_centered"
    TOP_RIGHT = "top_right"
    LEFT = "centered_left"
    CENTER = "centered"
    RIGHT = "centered_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM = "bottom_centered"
    BOTTOM_RIGHT = "bottom_right"
    PERCENT_OFFSET = "percent_offset"
    PIXEL_OFFSET = "pixel_offset"


BLEND_OVERLAY_POSITION_LABELS = {
    BlendOverlayPosition.TOP_LEFT: "Top Left",
    BlendOverlayPosition.TOP: "Top",
    BlendOverlayPosition.TOP_RIGHT: "Top Right",
    BlendOverlayPosition.LEFT: "Left",
    BlendOverlayPosition.CENTER: "Center",
    BlendOverlayPosition.RIGHT: "Right",
    BlendOverlayPosition.BOTTOM_LEFT: "Bottom Left",
    BlendOverlayPosition.BOTTOM: "Bottom",
    BlendOverlayPosition.BOTTOM_RIGHT: "Bottom Right",
    BlendOverlayPosition.PERCENT_OFFSET: "Offset (%)",
    BlendOverlayPosition.PIXEL_OFFSET: "Offset (pixels)",
}

BLEND_OVERLAY_X0_Y0_FACTORS = {
    BlendOverlayPosition.TOP_LEFT: np.array([0, 0]),
    BlendOverlayPosition.TOP: np.array([0.5, 0]),
    BlendOverlayPosition.TOP_RIGHT: np.array([1, 0]),
    BlendOverlayPosition.LEFT: np.array([0, 0.5]),
    BlendOverlayPosition.CENTER: np.array([0.5, 0.5]),
    BlendOverlayPosition.RIGHT: np.array([1, 0.5]),
    BlendOverlayPosition.BOTTOM_LEFT: np.array([0, 1]),
    BlendOverlayPosition.BOTTOM: np.array([0.5, 1]),
    BlendOverlayPosition.BOTTOM_RIGHT: np.array([1, 1]),
    BlendOverlayPosition.PERCENT_OFFSET: np.array([1, 1]),
    BlendOverlayPosition.PIXEL_OFFSET: np.array([0, 0]),
}


@compositing_group.register(
    schema_id="chainner:image:blend",
    name="Blend Images",
    description="""Blends an overlay image onto a base image using the specified mode.""",
    icon="BsLayersHalf",
    inputs=[
        ImageInput("Base Layer", channels=[1, 3, 4], allow_colors=True),
        ImageInput("Overlay Layer", channels=[1, 3, 4], allow_colors=True),
        BlendModeDropdown(),
        if_group(Condition.type(0, "Image") & Condition.type(1, "Image"))(
            EnumInput(
                BlendOverlayPosition,
                label="Overlay position",
                option_labels=BLEND_OVERLAY_POSITION_LABELS,
                default=BlendOverlayPosition.CENTER,
            ),
            if_enum_group(3, (BlendOverlayPosition.PERCENT_OFFSET))(
                SliderInput("X offset", min=-200, max=200, default=0, unit="%"),
                SliderInput("Y offset", min=-200, max=200, default=0, unit="%"),
            ),
            if_enum_group(3, (BlendOverlayPosition.PIXEL_OFFSET))(
                NumberInput("X offset", min=None, max=None, default=0, unit="px"),
                NumberInput("Y offset", min=None, max=None, default=0, unit="px"),
            ),
            BoolInput("Crop to fit base layer", default=False),
        ),
    ],
    outputs=[
        ImageOutput(
            image_type="""
            let base = Input0;
            let overlay = Input1;
            let position:BlendOverlayPosition = Input3;
            let cropToFit = bothImages and Input8;

            def isImage(x: any) = match x { Image => true, _ => false };
            let bothImages = isImage(base) and isImage(overlay);

            def getWidth(img: any) = match img { Image => img.width, _ => -inf };
            def getHeight(img: any) = match img { Image => img.height, _ => -inf };

            struct Size { width: uint, height: uint}
            def imageToSize(x: any): Size = Size {
                width: getWidth(x) & uint,
                height: getHeight(x) & uint
            };
            let baseSize:Size = imageToSize(Input0);
            let overlaySize:Size = imageToSize(Input1);
            let maxSize:Size = Size {
                width: max(getWidth(Input0), getWidth(Input1)) & uint,
                height: max(getHeight(Input0), getHeight(Input1)) & uint,
            };

            def getExtendedDim(base_dim: uint, ov_dim: uint, offset: int): uint {
                abs(min(offset, 0)) + max(base_dim, (offset + ov_dim))
            }
            def getExtendedCanvasSize(b: Size, o: Size, x_offset: int, y_offset: int): Size {
                Size {
                    width: getExtendedDim(b.width, o.width, x_offset),
                    height: getExtendedDim(b.height, o.height, y_offset),
                }
            }
            def percentToOffset(b: uint, o: uint, percent: int): int {
                round(((b - o) * percent / 100.0)) & int
            }
            let extendedCanvasSize = match position {
                BlendOverlayPosition::PercentOffset => getExtendedCanvasSize(
                    baseSize,
                    overlaySize,
                    percentToOffset(baseSize.width, overlaySize.width, Input4),
                    percentToOffset(baseSize.height, overlaySize.height, Input5),
                ),
                BlendOverlayPosition::PixelOffset => getExtendedCanvasSize(
                    baseSize, overlaySize, Input6, Input7
                ),
                _ => maxSize
            };

            let canvasSize = if cropToFit {
                baseSize
            } else {
                if bothImages {extendedCanvasSize} else {maxSize}
            };

            Image {
                width: canvasSize.width & uint,
                height: canvasSize.height & uint,
                channels: max(base.channels, overlay.channels)
            }
            """,
            assume_normalized=True,
        ).with_never_reason("At least one layer must be an image"),
    ],
)
def blend_images_node(
    base: np.ndarray | Color,
    ov: np.ndarray | Color,
    blend_mode: BlendMode,
    overlay_position: BlendOverlayPosition,
    x_percent: int,
    y_percent: int,
    x_px: int,
    y_px: int,
    crop_to_fit: bool,
) -> np.ndarray:
    (base_height, base_width, _), (overlay_height, overlay_width, _) = dimensions(
        base, ov
    )
    if overlay_position == BlendOverlayPosition.PERCENT_OFFSET:
        x0 = round((base_width - overlay_width) * x_percent / 100)
        y0 = round((base_height - overlay_height) * y_percent / 100)
    elif overlay_position == BlendOverlayPosition.PIXEL_OFFSET:
        x0, y0 = x_px, y_px
    else:
        x0, y0 = np.array(
            [base_width - overlay_width, base_height - overlay_height]
            * BLEND_OVERLAY_X0_Y0_FACTORS[overlay_position]
        ).astype("int")
    return canvas(base, ov, blend_mode.value, x0, y0, crop_to_fit)
