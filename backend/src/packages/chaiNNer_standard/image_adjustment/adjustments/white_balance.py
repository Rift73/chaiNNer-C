from __future__ import annotations

import math
from enum import Enum

import numpy as np

from api import KeyInfo
from nodes.groups import Condition, if_enum_group, if_group
from nodes.impl.color.color import Color
from nodes.impl.linear_light import LightTransfer, encode_inplace, linearize
from nodes.impl.native_analysis import channel_sums, scale_channels
from nodes.properties.inputs import (
    BoolInput,
    ColorInput,
    EnumInput,
    ImageInput,
    LightTransferInput,
    SliderInput,
)
from nodes.properties.outputs import ImageOutput
from nodes.utils.utils import get_h_w_c

from .. import adjustments_group

# The luminance of a linear BGR color (BT.709, the primaries of sRGB).
LUMINANCE = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)

NEUTRAL_TEMPERATURE = 6500
# A channel below a 64th of the reference's luminance is no cast but a color that has next to
# none of it, and its factor would mostly amplify noise.
MAX_GAIN = 64


class WhiteBalanceMode(Enum):
    GRAY_WORLD = 0
    WHITE_PATCH = 1
    NEUTRAL_COLOR = 2
    TEMPERATURE = 3


def planckian_bgr(temperature: float) -> np.ndarray:
    """The linear BGR color of a black body: Kim et al.'s cubic fit of its chromaticity
    (1667 K to 25000 K), then the XYZ to linear sRGB matrix."""
    t = 1000 / temperature
    if temperature < 4000:
        x = -0.2661239 * t**3 - 0.2343589 * t**2 + 0.8776956 * t + 0.179910
    else:
        x = -3.0258469 * t**3 + 2.1070379 * t**2 + 0.2226347 * t + 0.240390
    if temperature < 2222:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif temperature < 4000:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    big_x = x / y
    big_z = (1 - x - y) / y
    return np.array(
        [
            0.0557 * big_x - 0.2040 + 1.0570 * big_z,
            -0.9689 * big_x + 1.8758 + 0.0415 * big_z,
            3.2406 * big_x - 1.5372 - 0.4986 * big_z,
        ]
    )


def mean_color(rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """The mean color, each pixel counted as much as it is visible."""
    if alpha is None:
        return channel_sums(rgb) / (rgb.shape[0] * rgb.shape[1])
    visible = np.sum(alpha, dtype=np.float64)
    if visible == 0:
        return np.ones(3)
    weighted = rgb * alpha[:, :, None]
    return channel_sums(weighted) / visible


def brightest_color(
    rgb: np.ndarray, alpha: np.ndarray | None, percent: float
) -> np.ndarray:
    """The mean color of the given share of the visible pixels with the highest luminance."""
    pixels = rgb.reshape((-1, 3)) if alpha is None else rgb[alpha > 0]
    if len(pixels) == 0:
        return np.ones(3)
    luminance = pixels @ LUMINANCE
    start = len(pixels) - max(1, math.ceil(len(pixels) * percent / 100))
    threshold = np.partition(luminance, start)[start]
    return np.mean(pixels[luminance >= threshold], axis=0, dtype=np.float64)


def gains_for(reference: np.ndarray) -> np.ndarray:
    """The channel factors that make the reference color a gray of its own luminance, so the
    brightness of the image stays. A channel without any light in the reference stays too."""
    gains = np.ones(3)
    lit = reference > 0
    gains[lit] = float(reference @ LUMINANCE.astype(np.float64)) / reference[lit]
    largest = int(np.argmax(gains))
    if gains[largest] > MAX_GAIN:
        name = ("blue", "green", "red")[largest]
        raise ValueError(
            f"The color taken as neutral has almost no {name}: {name} would have to be multiplied by {gains[largest]:.0f}. Use a color closer to gray or another mode."
        )
    return gains.astype(np.float32)


@adjustments_group.register(
    schema_id="chainner:image:white_balance",
    name="White Balance",
    description=[
        "Removes a color cast by scaling the R, G and B channels, so that a color that should be neutral becomes a gray of the same brightness. Alpha stays as it is, and a grayscale image has no color to balance.",
    ],
    icon="MdWbAuto",
    inputs=[
        ImageInput(channels=[1, 3, 4]),
        EnumInput(
            WhiteBalanceMode,
            label="Mode",
            label_style="inline",
            default=WhiteBalanceMode.GRAY_WORLD,
        )
        .with_docs(
            "How the color of the light is found:",
            "- Gray World: the mean color of the image is taken as neutral. This works for scenes with many colors, and gives a wrong result when one color dominates the image.",
            "- White Patch: the mean color of the brightest pixels is taken as neutral. This works for scenes with something white in them.",
            "- Neutral Color: the given color is taken as neutral. Use Pick Color to get it from a part of the image that should be gray or white.",
            "- Temperature: the color of the light is given as a color temperature.",
            "Transparent pixels don't count for Gray World and White Patch.",
            "A color with almost none of R, G or B can't be taken as neutral: the node stops with an error when a channel would have to be multiplied by more than 64.",
        )
        .with_id(1),
        if_enum_group(1, WhiteBalanceMode.WHITE_PATCH)(
            SliderInput(
                "Brightest Pixels",
                min=0.01,
                max=50,
                default=1,
                precision=2,
                step=0.1,
                unit="%",
                scale="log",
            ).with_id(2),
        ),
        if_enum_group(1, WhiteBalanceMode.NEUTRAL_COLOR)(
            ColorInput("Neutral Color", channels=[3, 4]).with_id(3),
        ),
        if_enum_group(1, WhiteBalanceMode.TEMPERATURE)(
            SliderInput(
                "Temperature",
                min=2000,
                max=15000,
                default=NEUTRAL_TEMPERATURE,
                precision=0,
                step=100,
                unit="K",
                gradient=["#5f9fff", "#ffffff", "#ffb050"],
            )
            .with_docs(
                "The color temperature of the light in the image: about 3000 K for light bulbs, 5500 K for sunlight and 7500 K for shade. Like in a camera, a higher value makes the image warmer. 6500 K changes nothing.",
            )
            .with_id(4),
            SliderInput(
                "Tint",
                min=-100,
                max=100,
                default=0,
                precision=1,
                step=1,
                gradient=["#40c040", "#ffffff", "#d040d0"],
            )
            .with_docs(
                "Corrects a green or magenta cast that a color temperature doesn't explain. Higher values take green away, and 100 halves it.",
            )
            .with_id(5),
        ),
        BoolInput("Linear Light", default=True)
        .with_docs(
            "Scale the channels in linear light, as a change of light does: the colour channels go through the Transfer curve to linear light and back after. Turn this off for an image that is linear already.",
        )
        .with_id(6),
        if_group(Condition.bool(6, True))(
            LightTransferInput(LightTransfer.SRGB).with_id(7),
        ),
    ],
    outputs=[
        ImageOutput(shape_as=0),
    ],
    key_info=KeyInfo.enum(1),
)
def white_balance_node(
    img: np.ndarray,
    mode: WhiteBalanceMode,
    brightest_percent: float,
    neutral_color: Color,
    temperature: float,
    tint: float,
    linear_light: bool,
    transfer: LightTransfer,
) -> np.ndarray:
    c = get_h_w_c(img)[2]
    if c == 1:
        return img

    image = linearize(img, transfer) if linear_light else np.array(img, order="C")
    rgb = image[:, :, :3]
    alpha = image[:, :, 3] if c == 4 else None

    if mode is WhiteBalanceMode.GRAY_WORLD:
        reference = mean_color(rgb, alpha)
    elif mode is WhiteBalanceMode.WHITE_PATCH:
        reference = brightest_color(rgb, alpha, brightest_percent)
    elif mode is WhiteBalanceMode.NEUTRAL_COLOR:
        color = np.array(neutral_color.value[:3], dtype=np.float32)[None, None]
        reference = (linearize(color, transfer) if linear_light else color)[0, 0]
        reference = reference.astype(np.float64)
    else:
        neutral = planckian_bgr(NEUTRAL_TEMPERATURE)
        reference = planckian_bgr(temperature) / neutral
        reference[1] *= 2 ** (tint / 100)

    gains = gains_for(reference)
    if np.all(gains == 1):
        return img

    scale_channels(image, gains)
    return encode_inplace(image, transfer) if linear_light else image
