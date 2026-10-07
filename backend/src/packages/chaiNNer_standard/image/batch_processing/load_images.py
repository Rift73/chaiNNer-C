from __future__ import annotations

# `x as x` imports: the native mirror reads these names in native/src/file_sequence.cpp
import os as os
from pathlib import Path

import numpy as np
from wcmatch import glob as glob

from api import Generator, IteratorOutputInfo
from nodes.groups import Condition, if_group
from nodes.impl.image_formats import (
    get_available_image_formats as get_available_image_formats,
)
from nodes.impl.native_graph import graph
from nodes.properties.inputs import BoolInput, DirectoryInput, NumberInput, TextInput
from nodes.properties.outputs import (
    DirectoryOutput,
    ImageOutput,
    NumberOutput,
    TextOutput,
)

from .. import batch_processing_group
from ..io.load_image import load_image_node as load_image_node


def extension_filter(lst: list[str]) -> str:
    """generates a mcmatch.glob expression to filter files with specific extensions
    ex. {*,**/*}@(*.png|*.jpg|...)"""
    return graph().file_sequence_extension_filter(lst)


def list_glob(directory: Path, globexpr: str, ext_filter: list[str]) -> list[Path]:
    return graph().file_sequence_list_glob(globals(), directory, globexpr, ext_filter)


@batch_processing_group.register(
    schema_id="chainner:image:load_images",
    name="Load Images",
    description=[
        "Iterate over all files in a directory/folder (batch processing) and run the provided nodes on just the image files. Supports the same file types as `chainner:image:load`.",
        "Optionally, you can toggle whether to iterate recursively (subdirectories) or use a glob expression to filter the files.",
    ],
    icon="BsImages",
    inputs=[
        DirectoryInput(),
        BoolInput("Use WCMatch glob expression", default=False),
        if_group(Condition.bool(1, False))(
            BoolInput("Recursive").with_docs("Iterate recursively over subdirectories.")
        ),
        if_group(Condition.bool(1, True))(
            TextInput("WCMatch Glob expression", default="**/*").with_docs(
                "For information on how to use WCMatch glob expressions, see [here](https://facelessuser.github.io/wcmatch/glob/)."
            ),
        ),
        BoolInput("Use limit", default=False).with_id(4),
        if_group(Condition.bool(4, True))(
            NumberInput("Limit", default=10, min=1)
            .with_docs(
                "Limit the number of images to iterate over. This can be useful for testing the iterator without having to iterate over all images."
            )
            .with_id(5)
        ),
        BoolInput("Stop on first error", default=False).with_docs(
            "Instead of collecting errors and throwing them at the end of processing, stop iteration and throw an error as soon as one occurs.",
            hint=True,
        ),
    ],
    outputs=[
        ImageOutput(),
        DirectoryOutput("Directory", output_type="Input0"),
        TextOutput("Subdirectory Path"),
        TextOutput("Name"),
        NumberOutput("Index", output_type="min(uint, max(0, IterOutput0.length - 1))"),
    ],
    iterator_outputs=IteratorOutputInfo(
        outputs=[0, 2, 3, 4],
        length_type="if Input4 { min(uint, Input5) } else { uint }",
    ),
    kind="generator",
    side_effects=True,
)
def load_images_node(
    directory: Path,
    use_glob: bool,
    is_recursive: bool,
    glob_str: str,
    use_limit: bool,
    limit: int,
    fail_fast: bool,
) -> tuple[Generator[tuple[np.ndarray, str, str, int]], Path]:
    return graph().file_sequence_load_images(
        globals(),
        directory,
        use_glob,
        is_recursive,
        glob_str,
        use_limit,
        limit,
        fail_fast,
    )
