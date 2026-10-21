from __future__ import annotations

import os
from pathlib import Path

import onnx
from sanic.log import logger

from nodes.impl.compressed_file import decompressed
from nodes.impl.onnx.load import load_onnx_model
from nodes.impl.onnx.model import OnnxModel
from nodes.properties.inputs import OnnxFileInput
from nodes.properties.outputs import DirectoryOutput, FileNameOutput, OnnxModelOutput
from nodes.utils.utils import split_file_path

from .. import io_group


@io_group.register(
    schema_id="chainner:onnx:load_model",
    name="Load Model",
    description=(
        "Load ONNX model file (.onnx), compressed by Save Model or not. Theoretically"
        " supports any ONNX Super-Resolution model that doesn't expect non-standard"
        " preprocessing. Also supports RemBG background removal models."
    ),
    icon="ONNX",
    inputs=[OnnxFileInput(primary_input=True)],
    outputs=[
        OnnxModelOutput(kind="tagged").suggest(),
        DirectoryOutput("Directory", of_input=0).with_id(2),
        FileNameOutput("Name", of_input=0).with_id(1),
    ],
    see_also=[
        "chainner:onnx:load_models",
    ],
    side_effects=True,
)
def load_model_node(path: Path) -> tuple[OnnxModel, Path, str]:
    assert os.path.exists(path), f"Model file at location {path} does not exist"

    assert os.path.isfile(path), f"Path {path} is not a file"

    logger.debug(f"Reading onnx model from path: {path}")
    # A plain model may keep its weights in files beside it, which load_model reads; a
    # compressed one (Save Model's Compress) is one whole model. A tensor kept outside
    # names its file under the key "location", so bytes without that word hold the
    # whole model, which is parsed once and kept as it is.
    data = decompressed(path)
    if data is None:
        data = path.read_bytes()
        model = data if b"location" not in data else onnx.load_model(str(path))
    else:
        model = data if b"location" not in data else onnx.load_from_string(data)

    dirname, basename, _ = split_file_path(path)
    return load_onnx_model(model), dirname, basename
