from __future__ import annotations

from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save
from sanic.log import logger
from spandrel import ModelDescriptor

from nodes.impl.native_graph import graph
from nodes.properties.inputs import (
    DirectoryInput,
    EnumInput,
    ModelInput,
    RelativePathInput,
)

from .. import io_group


class WeightFormat(Enum):
    PTH = "pth"
    ST = "safetensors"


@io_group.register(
    schema_id="chainner:pytorch:save_model",
    name="Save Model",
    description=[
        "Save a PyTorch model to specified directory.",
        'This is not guaranteed to save certain models in a way that can be read by other programs. Notably, it saves without the "load key" some scripts expect.',
    ],
    icon="MdSave",
    inputs=[
        ModelInput(),
        DirectoryInput(must_exist=False),
        RelativePathInput("Model Name"),
        EnumInput(
            WeightFormat,
            "Weight Format",
            default=WeightFormat.PTH,
            option_labels={
                WeightFormat.PTH: "PyTorch (.pth)",
                WeightFormat.ST: "SafeTensors (.safetensors)",
            },
        ),
    ],
    outputs=[],
    side_effects=True,
)
def save_model_node(
    model: ModelDescriptor, directory: Path, name: str, weight_format: WeightFormat
) -> None:
    full_path = (directory / f"{name}.{weight_format.value}").resolve()
    logger.debug(f"Writing model to path: {full_path}")
    full_path.parent.mkdir(parents=True, exist_ok=True)
    # Through a temporary file in the folder (native/include/atomic_file.hpp): no partial
    # file, and a file that already holds these bytes is left untouched.
    if weight_format == WeightFormat.PTH:
        # torch.save names the archive inside after the file, so it writes the final name in
        # a temporary folder beside the target: one volume, so the move is a rename.
        with TemporaryDirectory(prefix=".chainner-c-", dir=full_path.parent) as folder:
            staged = Path(folder) / full_path.name
            torch.save(model.model.state_dict(), staged)
            graph().atomic_move(staged, full_path)
    elif weight_format == WeightFormat.ST:
        graph().atomic_write(full_path, save(model.model.state_dict()))
    else:
        raise ValueError(f"Unknown weight format: {weight_format}")
