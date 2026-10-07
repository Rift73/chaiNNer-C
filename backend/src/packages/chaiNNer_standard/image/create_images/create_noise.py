from __future__ import annotations

from enum import Enum

import numpy as np

import navi
from nodes.groups import if_enum_group, seed_group
from nodes.impl import native_blue, native_procedural
from nodes.impl.noise_functions.blue import create_blue_noise
from nodes.properties.inputs import (
    BoolInput,
    EnumInput,
    NumberInput,
    SeedInput,
    SliderInput,
)
from nodes.properties.outputs import ImageOutput
from nodes.utils.seed import Seed

from .. import create_images_group


class NoiseMethod(Enum):
    VALUE_NOISE = "Value Noise"
    SMOOTH_VALUE_NOISE = "Smooth Value Noise"
    SIMPLEX = "Simplex"
    BLUE_NOISE = "Blue Noise"


class FractalMethod(Enum):
    NONE = "None"
    PINK_NOISE = "Pink noise"


@create_images_group.register(
    schema_id="chainner:image:create_noise",
    name="Create Noise",
    description="Create an image of specified dimensions filled with one of a variety of noises.",
    icon="MdFormatColorFill",
    inputs=[
        NumberInput("Width", min=1, unit="px", default=256),
        NumberInput("Height", min=1, unit="px", default=256),
        seed_group(SeedInput()),
        EnumInput(
            NoiseMethod,
            default=NoiseMethod.SIMPLEX,
            option_labels={NoiseMethod.SMOOTH_VALUE_NOISE: "Value Noise (smooth)"},
        ).with_id(3),
        if_enum_group(
            3,
            (
                NoiseMethod.SIMPLEX,
                NoiseMethod.VALUE_NOISE,
                NoiseMethod.SMOOTH_VALUE_NOISE,
            ),
        )(
            NumberInput("Scale", min=1, default=50, precision=1).with_id(4),
            SliderInput("Brightness", min=0, default=100, max=100, precision=2).with_id(
                5
            ),
            BoolInput("Tile Horizontal", default=False).with_id(10),
            BoolInput("Tile Vertical", default=False).with_id(11),
            BoolInput("Tile Spherical", default=False).with_id(12),
            EnumInput(FractalMethod, default=FractalMethod.NONE).with_id(6),
            if_enum_group(6, FractalMethod.PINK_NOISE)(
                NumberInput("Layers", min=2, max=20, default=3).with_id(7),
                NumberInput("Scale Ratio", min=1, default=2, precision=2).with_id(8),
                NumberInput("Brightness Ratio", min=1, default=2, precision=2).with_id(
                    9
                ),
                BoolInput("Increment Seed", default=True).with_id(13),
            ),
        ),
        if_enum_group(3, NoiseMethod.BLUE_NOISE)(
            SliderInput(
                "Standard Deviation",
                min=1,
                max=100,
                default=1.5,
                precision=3,
                scale="log-offset",
            ).with_id(14)
        ),
    ],
    outputs=[
        ImageOutput(
            image_type=navi.Image(
                width="Input0",
                height="Input1",
            ),
            channels=1,
        )
    ],
)
def create_noise_node(
    width: int,
    height: int,
    seed_obj: Seed,
    noise_method: NoiseMethod,
    scale: float,
    brightness: float,
    tile_horizontal: bool,
    tile_vertical: bool,
    tile_spherical: bool,
    fractal_method: FractalMethod,
    layers: int,
    scale_ratio: float,
    brightness_ratio: float,
    increment_seed: bool,
    standard_deviation: float,
) -> np.ndarray:
    brightness /= 100
    seed = seed_obj.to_u32()

    if noise_method == NoiseMethod.BLUE_NOISE:
        return native_blue.normalize(
            create_blue_noise(
                (height, width),
                standard_deviation=standard_deviation,
                seed=seed_obj.to_u32(),
            )
        )

    return native_procedural.create(
        width,
        height,
        seed,
        {
            NoiseMethod.VALUE_NOISE: 0,
            NoiseMethod.SMOOTH_VALUE_NOISE: 1,
            NoiseMethod.SIMPLEX: 2,
        }[noise_method],
        scale,
        brightness,
        tile_horizontal,
        tile_vertical,
        tile_spherical,
        layers if fractal_method == FractalMethod.PINK_NOISE else 0,
        scale_ratio if fractal_method == FractalMethod.PINK_NOISE else 1,
        brightness_ratio if fractal_method == FractalMethod.PINK_NOISE else 1,
        increment_seed,
    )
