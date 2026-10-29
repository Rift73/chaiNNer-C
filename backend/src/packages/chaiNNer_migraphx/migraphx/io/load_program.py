from __future__ import annotations

import os
from pathlib import Path

from sanic.log import logger

from api import NodeContext
from nodes.impl.compressed_file import read_file
from nodes.impl.migraphx.model import MIGraphXProgram
from nodes.impl.migraphx.program_info import read_program_info
from nodes.impl.migraphx.session import programs
from nodes.properties.inputs import MIGraphXFileInput
from nodes.properties.outputs import (
    DirectoryOutput,
    FileNameOutput,
    MIGraphXProgramOutput,
)
from nodes.utils.utils import file_identity, split_file_path

from ...settings import get_settings
from .. import io_group

if io_group is not None:

    @io_group.register(
        schema_id="chainner:migraphx:load_program",
        name="Load Program",
        description=(
            "Load a MIGraphX program file (.mxr) from Save Program, compressed or not."
            " A program is compiled for one GPU target (gfx) and MIGraphX version; one for"
            " another fails to load with a message saying to build it again."
        ),
        icon="BsAmd",
        inputs=[MIGraphXFileInput(primary_input=True)],
        outputs=[
            MIGraphXProgramOutput(kind="tagged").suggest(),
            DirectoryOutput("Directory", of_input=0).with_id(2),
            FileNameOutput("Name", of_input=0).with_id(1),
        ],
        side_effects=True,
        node_context=True,
    )
    def load_program_node(
        context: NodeContext, path: Path
    ) -> tuple[MIGraphXProgram, Path, str]:
        assert os.path.exists(path), f"Program file at location {path} does not exist"
        assert os.path.isfile(path), f"Path {path} is not a file"

        settings = get_settings(context)
        gpu_index = settings.gpu_index

        def load() -> MIGraphXProgram:
            logger.debug("Reading MIGraphX program from path: %s", path)
            data = read_file(path)
            return MIGraphXProgram(data, read_program_info(data, gpu_index))

        # Read once and kept across runs until no node uses it, so that Upscale Image
        # keeps its session; a changed file reloads.
        with programs.use(
            (file_identity(path), gpu_index), context.node_id, load
        ) as program:
            dirname, basename, _ = split_file_path(path)
            return program, dirname, basename
