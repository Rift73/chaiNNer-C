from __future__ import annotations

from pathlib import Path

import numpy as np

from api import Generator, IteratorOutputInfo
from nodes.impl.dds.texconv import read_dds_images
from nodes.properties.inputs import FileInput, NumberInput
from nodes.properties.outputs import LargeImageOutput, NumberOutput

from .. import batch_processing_group
from ..io.load_dds import dds_image


@batch_processing_group.register(
    schema_id="chainner:image:load_dds_images",
    name="Load DDS Images",
    description=(
        "Iterate over the images of a DDS texture at one mip level: each layer of a"
        " texture array, each face of a cubemap (+X, -X, +Y, -Y, +Z, -Z, per cube) or"
        " each depth slice of a volume texture. 16-bit and float formats load at their"
        " depth; values outside 0-1 are kept."
    ),
    icon="BsFillImageFill",
    inputs=[
        FileInput(
            label="DDS File", file_kind="image", filetypes=[".dds"], primary_input=True
        ),
        NumberInput("Mip Level", default=0, min=0).with_docs(
            "0 is the full-size image, each next level half its size."
        ),
    ],
    outputs=[
        LargeImageOutput(assume_normalized=True),
        NumberOutput("Index", output_type="uint").with_docs(
            "The layer, face or slice: 0 for the first image, counting up by 1."
        ),
    ],
    iterator_outputs=IteratorOutputInfo(outputs=[0, 1]),
    kind="generator",
    side_effects=True,
)
def load_dds_images_node(
    path: Path, mip_level: int
) -> Generator[tuple[np.ndarray, int]]:
    # One texconv conversion reads them all; the items read no file.
    images = read_dds_images(path, mip_level)
    generator = Generator.from_list(
        images, lambda image, index: (dds_image(image), index)
    )
    generator.source_paths = ()
    return generator
