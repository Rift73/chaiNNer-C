from __future__ import annotations

import os
from pathlib import Path

import onnx
from sanic.log import logger

from api import NodeContext
from nodes.impl.compressed_file import decompressed
from nodes.impl.onnx.load import load_onnx_model, warn_if_grid_frozen
from nodes.impl.onnx.model import OnnxGeneric, OnnxModel
from nodes.impl.onnx.size_probe import probe_in_background
from nodes.properties.inputs import OnnxFileInput
from nodes.properties.outputs import DirectoryOutput, FileNameOutput, OnnxModelOutput
from nodes.utils.utils import split_file_path

from ...settings import default_provider
from .. import io_group


def load_model_node(path: Path) -> tuple[OnnxModel, Path, str]:
    """What the Load Model node outputs, which other callers read a model with."""
    assert os.path.exists(path), f"Model file at location {path} does not exist"

    assert os.path.isfile(path), f"Path {path} is not a file"

    logger.debug(f"Reading onnx model from path: {path}")
    # A plain model may keep its weights in files beside it, which load_model reads; a
    # compressed one (Save Model's Compress) is one whole model. A tensor kept outside
    # names its file under the key "location", so bytes without that word hold the
    # whole model, which is parsed once and kept as it is.
    try:
        data = decompressed(path)
        if data is None:
            data = path.read_bytes()
            model = data if b"location" not in data else onnx.load_model(str(path))
        else:
            model = data if b"location" not in data else onnx.load_from_string(data)
        if not data:
            # ONNX parses no bytes as a model without a graph, which only fails later.
            raise ValueError("the file is empty")
        onnx_model = load_onnx_model(model)
    except OSError:
        raise
    except Exception as e:
        # A damaged file fails anywhere in the reading, with the reader's own words.
        raise ValueError(
            f"The ONNX model {path.name} cannot be read ({type(e).__name__}: {e})."
            " The file may be damaged or hold a model chaiNNer does not support."
        ) from e

    dirname, basename, _ = split_file_path(path)
    return onnx_model, dirname, basename


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
    node_context=True,
)
def load_model_and_probe_node(
    context: NodeContext, path: Path
) -> tuple[OnnxModel, Path, str]:
    loaded = load_model_node(path)
    model = loaded[0]
    warn_if_grid_frozen(context, model, path.name)
    provider = context.settings.get_str("execution_provider", default_provider)
    if isinstance(model, OnnxGeneric) and provider != "CPUExecutionProvider":
        probe_in_background(model, context.storage_dir)
    return loaded
