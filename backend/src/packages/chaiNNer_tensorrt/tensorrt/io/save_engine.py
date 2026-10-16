from __future__ import annotations

from pathlib import Path

from sanic.log import logger

from nodes.impl.tensorrt.engine_file import COMPRESSED_SUFFIX, write_engine_file
from nodes.impl.tensorrt.model import TensorRTEngine
from nodes.properties.inputs import (
    BoolInput,
    DirectoryInput,
    RelativePathInput,
    TensorRTEngineInput,
)

from .. import io_group

if io_group is not None:

    @io_group.register(
        schema_id="chainner:tensorrt:save_engine",
        name="Save Engine",
        description=(
            "Save a TensorRT engine to a file (.engine). "
            "The saved engine is specific to your GPU architecture and TensorRT version."
        ),
        icon="MdSave",
        inputs=[
            TensorRTEngineInput(),
            DirectoryInput(must_exist=False),
            RelativePathInput("Engine Name"),
            BoolInput("Compress", default=True).with_docs(
                "Compresses the engine with LZMA (7-Zip's method), about 4x smaller for a"
                ' DUAL engine, and adds "_compressed" to its name. It runs at the same speed'
                " once loaded; loading takes up to about half a second longer.",
                "Only chaiNNer-C reads compressed engines (Load Engine and VapourSynth's AI"
                " Upscale); turn this off for an engine meant for other TensorRT tools.",
            ),
        ],
        outputs=[],
        side_effects=True,
    )
    def save_engine_node(
        engine: TensorRTEngine, directory: Path, engine_name: str, compress: bool
    ) -> None:
        if compress and not engine_name.endswith(COMPRESSED_SUFFIX):
            engine_name += COMPRESSED_SUFFIX
        full_path = (directory / f"{engine_name}.engine").resolve()
        logger.debug("Writing TensorRT engine to path: %s", full_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        write_engine_file(full_path, engine.bytes, compressed=compress)
