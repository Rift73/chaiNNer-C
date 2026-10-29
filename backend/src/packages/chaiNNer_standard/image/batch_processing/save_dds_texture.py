from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np

from api import Collector, IteratorInputInfo

# `x as x` imports: the native mirror reads these names in native/src/image_io.cpp
from nodes.impl.dds.format import LEGACY_TO_DXGI as LEGACY_TO_DXGI
from nodes.impl.dds.format import PREFER_DX9 as PREFER_DX9
from nodes.impl.dds.format import DDSFormat
from nodes.impl.dds.format import to_dxgi as to_dxgi
from nodes.impl.dds.texconv import save_as_dds as save_as_dds
from nodes.impl.native_graph import graph
from nodes.properties.inputs import (
    DirectoryInput,
    EnumInput,
    ImageInput,
    RelativePathInput,
)

from .. import batch_processing_group
from ..io.save_image import BC7Compression as BC7Compression
from ..io.save_image import (
    DDSColourConversion,
    DDSMipFilter,
    ImageFormat,
    dds_inputs,
    get_full_path,
)
from ..io.save_image import DDSErrorMetric as DDSErrorMetric


class TextureKind(Enum):
    ARRAY = "array"
    CUBEMAP = "cubemap"
    VOLUME = "volume"


@batch_processing_group.register(
    schema_id="chainner:image:save_dds_texture",
    name="Save DDS Texture",
    description=(
        "Saves a sequence of images of one size as one DDS texture: a texture array, a"
        " cubemap or a volume texture (the images its depth slices), with Save Image's"
        " DDS options."
    ),
    icon="MdSave",
    inputs=[
        ImageInput("Image Sequence"),
        DirectoryInput(must_exist=False),
        RelativePathInput("Subdirectory Path").make_optional(),
        RelativePathInput("Texture Name").with_docs(
            "The name of the file **without** the extension. An existing file is"
            " overwritten."
        ),
        EnumInput(
            TextureKind,
            "Texture Type",
            default=TextureKind.ARRAY,
            option_labels={
                TextureKind.ARRAY: "Texture Array",
                TextureKind.CUBEMAP: "Cubemap",
                TextureKind.VOLUME: "Volume Texture",
            },
        ).with_docs(
            "A cubemap takes exactly 6 images, its faces in the order +X, -X, +Y, -Y,"
            " +Z, -Z."
        ),
        *dds_inputs(),
    ],
    iterator_inputs=IteratorInputInfo(inputs=0),
    outputs=[],
    side_effects=True,
    kind="collector",
)
def save_dds_texture_node(
    _: None,
    base_directory: Path,
    relative_path: str | None,
    name: str,
    kind: TextureKind,
    dds_format: DDSFormat,
    dds_bc7_compression: BC7Compression,
    dds_error_metric: DDSErrorMetric,
    dds_dithering: bool,
    dds_mipmap_levels: int,
    dds_mip_levels: int,
    dds_mip_filter: DDSMipFilter,
    dds_separate_alpha: bool,
    dds_colour_conversion: DDSColourConversion,
    dds_premultiply_alpha: bool,
    dds_gpu: bool,
) -> Collector[np.ndarray, None]:
    images: list[np.ndarray] = []

    def on_complete():
        if kind == TextureKind.CUBEMAP and len(images) != 6:
            raise ValueError(
                f"A cubemap has 6 faces (+X, -X, +Y, -Y, +Z, -Z), not {len(images)}."
            )
        path = get_full_path(
            base_directory,
            relative_path,
            name,
            ImageFormat.DDS,
            "Save DDS Texture",
            "Texture Name",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        options = (
            dds_format,
            dds_bc7_compression,
            dds_error_metric,
            dds_dithering,
            dds_mipmap_levels,
            dds_mip_levels,
            dds_mip_filter,
            dds_separate_alpha,
            dds_colour_conversion,
            dds_premultiply_alpha,
            dds_gpu,
        )
        graph().image_io_save_dds_inputs(globals(), path, images, kind.value, options)

    return Collector(on_iterate=images.append, on_complete=on_complete)
