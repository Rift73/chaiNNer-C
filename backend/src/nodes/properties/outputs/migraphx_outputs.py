from __future__ import annotations

import navi
from api import BaseOutput, OutputKind
from nodes.utils.format import format_channel_numbers

from ...impl.migraphx.model import MIGraphXProgram


class MIGraphXProgramOutput(BaseOutput):
    """Output for a compiled MIGraphX program."""

    def __init__(
        self,
        model_type: navi.ExpressionJson = "MIGraphXProgram",
        label: str = "MIGraphX Program",
        kind: OutputKind = "generic",
    ):
        super().__init__(model_type, label, kind=kind, associated_type=MIGraphXProgram)

    def get_broadcast_data(self, value: MIGraphXProgram):
        i = value.info
        return {
            "tags": [
                format_channel_numbers(i.input_channels, i.output_channels),
                f"{i.scale}x",
                "mixed" if i.precision == "mixed" else i.precision.upper(),
                f"{i.width}x{i.height}",
                i.gfx,
            ]
        }

    def get_broadcast_type(self, value: MIGraphXProgram):
        i = value.info
        return navi.named(
            "MIGraphXProgram",
            {
                "scale": i.scale,
                "inputChannels": i.input_channels,
                "outputChannels": i.output_channels,
                "precision": navi.literal(i.precision),
                "height": i.height,
                "width": i.width,
            },
        )
