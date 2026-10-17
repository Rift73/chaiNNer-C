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
from nodes.impl.tensorrt.model import TensorRTEngine
from nodes.properties.inputs import (
    BoolInput,
    DirectoryInput,
    EnumInput,
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
                "Compresses the engine, about 4x smaller for a DUAL engine, and adds"
                ' "_compressed" to its name. It runs at the same speed once loaded.',
                "Only chaiNNer-C reads compressed engines (Load Engine and VapourSynth's AI"
                " Upscale); turn this off for an engine meant for other TensorRT tools.",
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
                    "LZMA (7-Zip's method) makes the smallest file; loading takes up to"
                    " about half a second longer. Zstandard is a little larger (a 1x DUAL"
                    " Light engine: 13.3 MB instead of 11.7 MB, from 47.5 MB) and loads in"
                    " about a twentieth of a second.",
                    hint=True,
                ),
            ),
        ],
        outputs=[],
        side_effects=True,
    )
    def save_engine_node(
        engine: TensorRTEngine,
        directory: Path,
        engine_name: str,
        compress: bool,
        compression: Compression,
    ) -> None:
        name = compressed_name(engine_name, compress)
        full_path = (directory / f"{name}.engine").resolve()
        logger.debug("Writing TensorRT engine to path: %s", full_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        write_file(
            full_path, engine.bytes, compression=compression if compress else None
        )
