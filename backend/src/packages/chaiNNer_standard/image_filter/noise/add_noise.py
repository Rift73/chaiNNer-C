from __future__ import annotations

from enum import Enum

import numpy as np

from nodes.groups import seed_group
from nodes.impl.noise import (
    NoiseColor,
    gaussian_noise,
    poisson_noise,
    salt_and_pepper_noise,
    speckle_noise,
    uniform_noise,
)
from nodes.properties.inputs import EnumInput, ImageInput, SeedInput, SliderInput
from nodes.properties.outputs import ImageOutput
from nodes.utils.seed import Seed

from .. import noise_group


class NoiseType(Enum):
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    SALT_AND_PEPPER = "salt_and_pepper"
    SPECKLE = "speckle"
    POISSON = "poisson"


@noise_group.register(
    schema_id="chainner:image:add_noise",
    name="Add Noise",
    description="Add various kinds of noise to an image.",
    icon="CgEditNoise",
    inputs=[
        ImageInput(channels=[1, 3, 4]),
        EnumInput(
            NoiseType, option_labels={NoiseType.SALT_AND_PEPPER: "Salt & Pepper"}
        ),
        EnumInput(
            NoiseColor,
            option_labels={
                NoiseColor.RGB: "Color",
                NoiseColor.GRAY: "Monochrome",
            },
        ),
        SliderInput(
            "Amount",
            min=0.0,
            max=100.0,
            default=50.0,
            precision=1,
            step=0.1,
            slider_step=0.1,
        ),
        seed_group(SeedInput()),
    ],
    outputs=[
        ImageOutput(
            image_type="""
                Image {
                    width: Input0.width,
                    height: Input0.height,
                    channels: max(
                        Input0.channels,
                        match Input2 { NoiseColor::Rgb => 3, NoiseColor::Gray => 1 }
                    )
                }""",
            assume_normalized=True,
        )
    ],
)
def add_noise_node(
    img: np.ndarray,
    noise_type: NoiseType,
    noise_color: NoiseColor,
    amount: float,
    seed: Seed,
) -> np.ndarray:
    # The generators take non-negative seeds only. A negative seed (a Seed input
    # accepts any integer) goes through Seed.to_u32() as Create Noise's does; the
    # others keep their exact noise.
    value = seed.value if seed.value >= 0 else seed.to_u32()
    if noise_type == NoiseType.GAUSSIAN:
        return gaussian_noise(img, amount / 100.0, noise_color, value)
    elif noise_type == NoiseType.UNIFORM:
        return uniform_noise(img, amount / 100.0, noise_color, value)
    elif noise_type == NoiseType.SALT_AND_PEPPER:
        return salt_and_pepper_noise(img, amount / 100.0, noise_color, value)
    elif noise_type == NoiseType.POISSON:
        return poisson_noise(img, amount / 100.0, noise_color, value)
    elif noise_type == NoiseType.SPECKLE:
        return speckle_noise(img, amount / 100.0, noise_color, value)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
