from __future__ import annotations

import numpy as np

from nodes.impl.native_channels import combine_rgb_alpha
from nodes.impl.native_color_ops import linear, material
from nodes.impl.resize import ResizeFilter, resize
from nodes.properties.inputs import ImageInput
from nodes.properties.outputs import ImageOutput
from nodes.utils.utils import get_h_w_c

from .. import conversion_group


def get_size(img: np.ndarray) -> tuple[int, int]:
    h, w, _ = get_h_w_c(img)
    return w, h


def metal_to_spec(
    albedo: np.ndarray,
    metal: np.ndarray,
    roughness: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert get_h_w_c(albedo)[2] == 3, "Expected the albedo map to be an RGB image"

    # This uses the conversion method described here:
    # https://marmoset.co/posts/pbr-texture-conversion/
    metal3_inv = material(metal, 0)

    albedo_size = get_size(albedo)
    metal_size = get_size(metal)

    if metal_size == albedo_size:
        metal3_inv_scaled = metal3_inv
    else:
        metal3_inv_scaled = resize(metal3_inv, albedo_size, ResizeFilter.LANCZOS)
    diff = material(albedo, 1, b=metal3_inv_scaled)

    if metal_size == albedo_size:
        scaled_albedo = albedo
    else:
        scaled_albedo = resize(albedo, metal_size, ResizeFilter.LANCZOS)
    spec = material(scaled_albedo, 2, mask=metal)

    if roughness is None:
        gloss = np.zeros((1, 1), np.float32) + 0.5
    else:
        gloss = linear(roughness, -1, 1)

    return diff, spec, gloss


@conversion_group.register(
    schema_id="chainner:image:metal_to_specular",
    name="Metal to Specular",
    description=("Converts a Metal/Roughness material into a Specular/Gloss material."),
    icon="MdChangeCircle",
    inputs=[
        ImageInput("Albedo", channels=[3, 4]),
        ImageInput("Metal", channels=1),
        ImageInput("Roughness", channels=1).make_optional(),
    ],
    outputs=[
        ImageOutput("Diffuse", shape_as=0),
        ImageOutput("Specular", size_as=1, channels=3),
        ImageOutput(
            "Gloss",
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
def metal_to_specular_node(
    albedo: np.ndarray,
    metal: np.ndarray,
    roughness: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    albedo_channels = get_h_w_c(albedo)[2]
    if albedo_channels == 4:
        albedo_alpha = albedo[:, :, 3]
        albedo = albedo[:, :, :3]
    else:
        albedo_alpha = None

    diff, spec, gloss = metal_to_spec(albedo, metal, roughness)

    if albedo_alpha is not None:
        diff = combine_rgb_alpha(diff, albedo_alpha)

    return diff, spec, gloss
