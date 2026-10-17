from __future__ import annotations

from pathlib import Path

from sanic.log import logger

from nodes.groups import Condition, if_group
from nodes.impl.compressed_file import (
    COMPRESSION_LABELS,
    Compression,
    compressed_name,
    write_file,
)
from nodes.impl.onnx.model import OnnxModel
from nodes.properties.inputs import (
    BoolInput,
    DirectoryInput,
    EnumInput,
    OnnxModelInput,
    RelativePathInput,
)

from .. import io_group


@io_group.register(
    schema_id="chainner:onnx:save_model",
    name="Save Model",
    description="""Save ONNX model to file (.onnx).""",
    icon="MdSave",
    inputs=[
        OnnxModelInput(),
        DirectoryInput(must_exist=False),
        RelativePathInput("Model Name"),
        BoolInput("Compress", default=True).with_docs(
            'Compresses the model and adds "_compressed" to its name. Upscaling models'
            " become about 40 to 90 % of their size (a HAT-L model 47 %), depending on"
            " how much of them is weights.",
            "Only chaiNNer-C reads compressed models (Load Model, Load Models and"
            " VapourSynth's AI Upscale); turn this off for a model meant for other"
            " programs.",
        ),
        if_group(Condition.bool(3, True))(
            EnumInput(
                Compression,
                label="Compression",
                default=Compression.LZMA,
                option_labels=COMPRESSION_LABELS,
            )
            .with_id(4)
            .with_docs(
                "LZMA (7-Zip's method) makes the smallest file and takes longest to save"
                " and load. Zstandard loads at about 1 GB/s.",
                hint=True,
            ),
        ),
    ],
    outputs=[],
    side_effects=True,
)
def save_model_node(
    model: OnnxModel,
    directory: Path,
    model_name: str,
    compress: bool,
    compression: Compression,
) -> None:
    name = compressed_name(model_name, compress)
    full_path = (directory / f"{name}.onnx").resolve()
    logger.debug(f"Writing file to path: {full_path}")
    full_path.parent.mkdir(parents=True, exist_ok=True)
    write_file(full_path, model.bytes, compression=compression if compress else None)
