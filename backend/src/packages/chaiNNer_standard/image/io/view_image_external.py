from __future__ import annotations

# `x as x` imports: the native mirror reads these names in native/src/image_io.cpp
import os as os
import platform as platform
import subprocess as subprocess
import time as time
from tempfile import mkdtemp as mkdtemp

import cv2 as cv2
import numpy as np
from sanic.log import logger as logger

from nodes.impl.image_utils import to_uint8 as to_uint8
from nodes.impl.native_graph import graph
from nodes.properties.inputs import ImageInput

from .. import io_group


@io_group.register(
    schema_id="chainner:image:preview",
    name="View Image (external)",
    description=[
        "Open the image in your default image viewer.",
        "This works by saving a temporary file that will be deleted after chaiNNer is closed. It is not recommended to be used when performing batch processing.",
    ],
    icon="BsEyeFill",
    inputs=[ImageInput()],
    outputs=[],
    side_effects=True,
    limited_to_8bpc="The temporary file is an 8-bit PNG.",
)
def view_image_external_node(img: np.ndarray) -> None:
    return graph().image_io_view(globals(), img)
