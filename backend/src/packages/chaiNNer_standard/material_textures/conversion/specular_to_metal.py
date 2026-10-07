from __future__ import annotations

import numpy as np

from nodes.impl.native_channels import combine_rgb_alpha
from nodes.impl.native_color_ops import linear, material
from nodes.impl.resize import ResizeFilter, resize
from nodes.properties.inputs import ImageInput, SliderInput
from nodes.properties.outputs import ImageOutput
from nodes.utils.utils import get_h_w_c

from .. import conversion_group


def get_size(img: np.ndarray) -> tuple[int, int]:
    h, w, _ = get_h_w_c(img)
    return w, h


def spec_to_metal(
    diff: np.ndarray,
    spec: np.ndarray,
    gloss: np.ndarray | None,
    metallic_min: float,
    metallic_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert get_h_w_c(diff)[2] == 3, "Expected the diffuse map to be an RGB image"
    assert get_h_w_c(spec)[2] == 3, "Expected the specular map to be an RGB image"

    metallic_diff = (
        0.0001 if metallic_min == metallic_max else metallic_max - metallic_min
    )

    # Metal is approximated using the magnitude of the specular map.

    metal = material(spec, 3, minimum=metallic_min, span=metallic_diff)

    # This uses the conversion method described here:
    # https://marmoset.co/posts/pbr-texture-conversion/

    diff_size = get_size(diff)
    spec_size = get_size(spec)

    if diff_size == spec_size:
        sped_scaled = spec
        metal_scaled = metal
    else:
        # to prevent color bleeding from non-metal parts of the specular map,
        # we apply the metal map as alpha and resize before combining with diffuse
        scaled = resize(combine_rgb_alpha(spec, metal), diff_size, ResizeFilter.LANCZOS)
        sped_scaled: np.ndarray = scaled[:, :, 0:3]
        metal_scaled: np.ndarray = scaled[:, :, 3]
    albedo = material(sped_scaled, 4, b=diff, mask=metal_scaled)

    if gloss is None:
        roughness = np.zeros((1, 1), np.float32) + 0.5
    else:
        roughness = linear(gloss, -1, 1)

    return albedo, metal, roughness


@conversion_group.register(
    schema_id="chainner:image:specular_to_metal",
    name="Specular to Metal",
    description=("Converts a Specular/Gloss material into a Metal/Roughness material."),
    icon="MdChangeCircle",
    inputs=[
        ImageInput("Diffuse", channels=[3, 4]),
        ImageInput("Specular", channels=3),
        ImageInput("Gloss", channels=1).make_optional(),
        SliderInput(
            "Metallic Min",
            min=0,
            max=100,
            default=23,
            precision=1,
            slider_step=1,
        ),
        SliderInput(
            "Metallic Max",
            min=0,
            max=100,
            default=30,
            precision=1,
            slider_step=1,
        ),
    ],
    outputs=[
        ImageOutput("Albedo", shape_as=0),
        ImageOutput("Metal", size_as=1, channels=1),
        ImageOutput(
            "Roughness",
            image_type="""
                    match Input2 {
                        Image as i => i,
                        null => Image { width: 1, height: 1, channels: 1 }
                    }
                """,
            channels=1,
        ),
    ],
)
def specular_to_metal_node(
    diff: np.ndarray,
    spec: np.ndarray,
    gloss: np.ndarray | None,
    metallic_min: float,
    metallic_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metallic_min /= 100
    metallic_max /= 100

    diff_channels = get_h_w_c(diff)[2]

    if diff_channels == 4:
        diff_alpha = diff[:, :, 3]
        diff = diff[:, :, :3]
    else:
        diff_alpha = None

    albedo, metal, roughness = spec_to_metal(
        diff, spec, gloss, metallic_min, metallic_max
    )

    if diff_alpha is not None:
        albedo = combine_rgb_alpha(albedo, diff_alpha)

    return albedo, metal, roughness
