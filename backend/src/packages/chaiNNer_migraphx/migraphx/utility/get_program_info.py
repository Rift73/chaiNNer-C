from __future__ import annotations

from nodes.impl.migraphx.model import MIGraphXProgram
from nodes.properties.inputs import MIGraphXProgramInput
from nodes.properties.outputs import NumberOutput, TextOutput

from .. import utility_group

if utility_group is not None:

    @utility_group.register(
        schema_id="chainner:migraphx:program_info",
        name="Get Program Info",
        description="Returns information about a MIGraphX program.",
        icon="ImInfo",
        inputs=[MIGraphXProgramInput()],
        outputs=[
            NumberOutput("Scale", output_type="Input0.scale"),
            TextOutput("Precision"),
            TextOutput("GPU Target"),
            TextOutput("MIGraphX Version"),
            TextOutput("Input Size"),
        ],
    )
    def get_program_info_node(
        program: MIGraphXProgram,
    ) -> tuple[int, str, str, str, str]:
        info = program.info
        return (
            info.scale,
            info.precision.upper(),
            info.gfx,
            info.migraphx_version,
            f"{info.width}x{info.height}",
        )
