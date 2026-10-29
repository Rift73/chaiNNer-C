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
from nodes.impl.migraphx.model import MIGraphXProgram
from nodes.properties.inputs import (
    BoolInput,
    DirectoryInput,
    EnumInput,
    MIGraphXProgramInput,
    RelativePathInput,
)

from .. import io_group

if io_group is not None:

    @io_group.register(
        schema_id="chainner:migraphx:save_program",
        name="Save Program",
        description=(
            "Save a MIGraphX program to a file (.mxr). "
            "The saved program is specific to your GPU target (gfx) and MIGraphX version."
        ),
        icon="MdSave",
        inputs=[
            MIGraphXProgramInput(),
            DirectoryInput(must_exist=False),
            RelativePathInput("Program Name"),
            BoolInput("Compress", default=True).with_docs(
                'Compresses the program and adds "_compressed" to its name. It runs at the'
                " same speed once loaded.",
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
                    "LZMA (7-Zip's method) makes the smallest file and loads a little"
                    " slower; Zstandard is a little larger and loads faster.",
                    hint=True,
                ),
            ),
        ],
        outputs=[],
        side_effects=True,
    )
    def save_program_node(
        program: MIGraphXProgram,
        directory: Path,
        program_name: str,
        compress: bool,
        compression: Compression,
    ) -> None:
        name = compressed_name(program_name, compress)
        full_path = (directory / f"{name}.mxr").resolve()
        logger.debug("Writing MIGraphX program to path: %s", full_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        write_file(
            full_path, program.bytes, compression=compression if compress else None
        )
