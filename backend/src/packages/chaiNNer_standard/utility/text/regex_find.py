from __future__ import annotations

from enum import Enum

from nodes.groups import if_enum_group
from nodes.impl.native_graph import graph
from nodes.impl.rust_regex import RustRegex
from nodes.properties.inputs import EnumInput, TextInput
from nodes.properties.outputs import TextOutput
from nodes.utils.replacement import ReplacementString

from .. import text_group


class OutputMode(Enum):
    FULL_MATCH = 0
    PATTERN = 1


@text_group.register(
    schema_id="chainner:utility:regex_find",
    name="Regex Find",
    description=[
        "Find some text matching a given regex.",
        "This node has 2 modes for output: full match and pattern.",
        (
            "- **Full Match:** return the full match. E.g. for the regex `\\d+` and the text `My two cats caught 32 mice in 14 days`, the output will be `32`."
            "\n- **Pattern:** using the same pattern syntax as in other nodes, return a formatted pattern of the match. E.g. for the regex `(\\w+) is (\\w+)`, the pattern is `{1}={2}`, and the text `My name is Jane.`, the output will be `name=Jane`."
        ),
    ],
    icon="MdTextFields",
    inputs=[
        TextInput("Text"),
        TextInput("Regex", placeholder=r'E.g. "\b\w+\b"'),
        EnumInput(
            OutputMode,
            label="Output",
            default=OutputMode.FULL_MATCH,
            label_style="inline",
        ).with_id(2),
        if_enum_group(2, OutputMode.PATTERN)(
            TextInput("Output Pattern", default="Found {0}")
        ),
    ],
    outputs=[
        TextOutput(
            "Text",
            output_type="""
                let pattern = match Input2 {
                    OutputMode::FullMatch => "{0}",
                    OutputMode::Pattern => Input3,
                };
                regexFind(Input0, Input1, pattern)
            """,
        ).with_never_reason(
            "Either the regex pattern or the replacement pattern is invalid"
        ),
    ],
    see_also=["chainner:utility:text_replace"],
)
def regex_find_node(
    text: str,
    regex_pattern: str,
    output: OutputMode,
    output_pattern: str,
) -> str:
    return graph().utility_regex_find(
        text,
        regex_pattern,
        output,
        output_pattern,
        OutputMode,
        RustRegex,
        ReplacementString,
        True,
    )
