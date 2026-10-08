from __future__ import annotations

from nodes.groups import optional_list_group
from nodes.impl.native_graph import graph
from nodes.properties.inputs import TextInput
from nodes.properties.outputs import TextOutput
from nodes.utils.replacement import ReplacementString

from .. import text_group


@text_group.register(
    schema_id="chainner:utility:text_pattern",
    name="Text Pattern",
    description=(
        "Concatenate text using a pattern: {1} to {9} insert the inputs, and {{ writes"
        " a literal {. Python format specs such as {1:x} are not supported."
    ),
    icon="MdTextFields",
    inputs=[
        TextInput("Pattern", has_handle=False, placeholder='E.g. "{1} and {2}"'),
        TextInput("{1}").make_optional(),
        TextInput("{2}").make_optional(),
        optional_list_group(
            *[TextInput(f"{{{number}}}").make_optional() for number in range(3, 10)],
        ),
    ],
    outputs=[
        TextOutput(
            "Output Text",
            output_type="""
                formatPattern(
                    Input0,
                    Input1,
                    Input2,
                    Input3,
                    Input4,
                    Input5,
                    Input6,
                    Input7,
                    Input8,
                    Input9
                )
                """,
        ).with_never_reason(
            "The pattern is either syntactically invalid or contains replacements that do not have a value."
            '\n\nHint: Use "{{" to escape a single "{" inside the pattern.'
        )
    ],
    see_also=["chainner:utility:text_append"],
)
def text_pattern_node(
    pattern: str,
    *args: str | None,
) -> str:
    return graph().utility_text_pattern(pattern, args, ReplacementString)
