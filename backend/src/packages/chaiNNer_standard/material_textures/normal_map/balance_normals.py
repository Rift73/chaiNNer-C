from __future__ import annotations

import numpy as np

import navi
from nodes.impl import native_geometry
from nodes.properties.inputs import ImageInput
from nodes.properties.outputs import ImageOutput

from .. import normal_map_group


@normal_map_group.register(
    schema_id="chainner:image:balance_normals",
    name="Balance Normals",
    description=[
        "This ensures that the average of all normals is pointing straight up. The input normal map is normalized before this operation is applied. The output normal map is guaranteed to be normalized.",
    ],
    icon="MdExpand",
    inputs=[
        ImageInput("Normal Map", channels=[3, 4]),
    ],
    outputs=[
        ImageOutput("Normal Map", image_type=navi.Image(size_as="Input0"), channels=3),
    ],
)
def balance_normals_node(n: np.ndarray) -> np.ndarray:
    return native_geometry.normal_map(n, 1)
