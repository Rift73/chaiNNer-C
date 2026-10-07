from __future__ import annotations

from enum import Enum

from nodes.impl.native_graph import graph
from nodes.properties.inputs import EnumInput, NumberInput, TextInput
from nodes.properties.outputs import TextOutput

from .. import text_group


class PaddingAlignment(Enum):
    START = "start"
    END = "end"
    CENTER = "center"


@text_group.register(
    schema_id="chainner:utility:text_padding",
    name="Text Padding",
    description="Pads text until it has a certain length.",
    icon="MdTextFields",
    inputs=[
        TextInput("Text"),
        NumberInput("Width", unit="chars"),
        TextInput(
            "Padding Character",
            has_handle=False,
            allow_numbers=False,
            min_length=1,
            max_length=1,
            placeholder="e.g. '0' or ' '",
        ),
        EnumInput(PaddingAlignment, label="Alignment"),
    ],
    outputs=[
        TextOutput(
            "Output Text",
            output_type="""
                match Input3 {
                    PaddingAlignment::Start => padStart(Input0, Input1, Input2),
                    PaddingAlignment::End => padEnd(Input0, Input1, Input2),
                    PaddingAlignment::Center => padCenter(Input0, Input1, Input2),
                }
                """,
        )
    ],
)
def text_padding_node(
    text: str, width: int, padding: str, alignment: PaddingAlignment
) -> str:
    return graph().utility_text_padding(
        text, width, padding, alignment, PaddingAlignment
    )
