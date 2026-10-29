# This class defines an interface.
# It is important that it does not contain types that depend on MIGraphX.
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MIGraphXProgramInfo:
    """Metadata about a compiled MIGraphX program. A program takes one input size."""

    # The type of its convolutions and matrix products ("fp32", "fp16", "bf16", "fp8"),
    # or "mixed": a DUAL or DRFT program, whose ONNX keeps its own types
    precision: str
    input_channels: int
    output_channels: int
    scale: int
    # The GPU target it was compiled for, e.g. "gfx1100"
    gfx: str
    migraphx_version: str
    height: int
    width: int


class MIGraphXProgram:
    """A compiled MIGraphX program: its serialized bytes (MIGraphX's .mxr format) and
    metadata, without MIGraphX imported. Sessions load the program from the bytes."""

    def __init__(self, program_bytes: bytes, info: MIGraphXProgramInfo):
        self.bytes: bytes = program_bytes
        self.info: MIGraphXProgramInfo = info

    @property
    def precision(self) -> str:
        return self.info.precision

    @property
    def scale(self) -> int:
        return self.info.scale

    @property
    def input_channels(self) -> int:
        return self.info.input_channels

    @property
    def output_channels(self) -> int:
        return self.info.output_channels
