from __future__ import annotations

import numpy as np

from nodes.impl.native_graph import graph
from nodes.properties.inputs import ClipboardInput

from .. import clipboard_group


@clipboard_group.register(
    schema_id="chainner:utility:copy_to_clipboard",
    name="Copy To Clipboard",
    description="Copies the input to the clipboard.",
    icon="BsClipboard",
    inputs=[
        ClipboardInput(),
    ],
    outputs=[],
    side_effects=True,
    limited_to_8bpc="The image will be copied to clipboard with 8 bits/channel.",
)
def copy_to_clipboard_node(value: str | np.ndarray) -> None:
    graph().utility_copy_to_clipboard(value, np.ndarray)
