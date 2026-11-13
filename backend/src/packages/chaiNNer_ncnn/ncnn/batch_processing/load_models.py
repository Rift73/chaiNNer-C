from __future__ import annotations

import os
from pathlib import Path

from sanic.log import logger

from api import Generator, IteratorOutputInfo
from nodes.impl.ncnn.model import NcnnModelWrapper
from nodes.properties.inputs import BoolInput, DirectoryInput
from nodes.properties.outputs import (
    DirectoryOutput,
    NcnnModelOutput,
    NumberOutput,
    TextOutput,
)
from nodes.utils.utils import list_all_files_sorted

from .. import batch_processing_group
from ..io.load_model import load_model_node


@batch_processing_group.register(
    schema_id="chainner:ncnn:load_models",
    name="Load Models",
    description="Iterate over all files in a directory and run the provided nodes on just the NCNN model files (.param/.bin). Supports the same models as `chainner:ncnn:load_model`.",
    icon="MdLoop",
    inputs=[
        DirectoryInput(),
        BoolInput("Stop on first error", default=False).with_docs(
            "Instead of collecting errors and throwing them at the end of processing, stop iteration and throw an error as soon as one occurs.",
            hint=True,
        ),
    ],
    outputs=[
        NcnnModelOutput(),
        DirectoryOutput("Directory", output_type="Input0"),
        TextOutput("Subdirectory Path"),
        TextOutput("Name"),
        NumberOutput("Index", output_type="uint").with_docs(
            "A counter that starts at 0 and increments by 1 for each model."
        ),
    ],
    iterator_outputs=IteratorOutputInfo(outputs=[0, 2, 3, 4]),
    kind="generator",
)
def load_models_node(
    directory: Path, fail_fast: bool
) -> tuple[Generator[tuple[NcnnModelWrapper, str, str, int]], Path]:
    logger.debug("Iterating over models in directory: %s", directory)

    def load_model(filepath_pairs: tuple[Path, Path], index: int):
        model, dirname, basename = load_model_node(filepath_pairs[0], filepath_pairs[1])
        rel_path = os.path.relpath(dirname, directory)
        return (model, rel_path, basename, index)

    files = list_all_files_sorted(directory, [".param", ".bin"])
    bin_files = {
        path.with_suffix(""): path for path in files if path.suffix.lower() == ".bin"
    }
    model_files: list[tuple[Path, Path]] = []
    for param_file in files:
        if param_file.suffix.lower() != ".param":
            continue
        bin_file = bin_files.pop(param_file.with_suffix(""), None)
        if bin_file is None:
            raise ValueError(
                f"Param file {param_file} has no bin file of the same name. Please check your files."
            )
        model_files.append((param_file, bin_file))
    if bin_files:
        raise ValueError(
            f"Bin file {next(iter(bin_files.values()))} has no param file of the same name. Please check your files."
        )
    return (
        Generator.from_list(model_files, load_model).with_fail_fast(fail_fast),
        directory,
    )
