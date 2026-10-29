"""Read the metadata of a compiled MIGraphX program."""

from __future__ import annotations

from typing import Any

from ..tensorrt.dual import input_alignment
from .model import MIGraphXProgramInfo
from .session import native

# MIGraphX's element types, as Build Program names precisions (FP8 has several formats)
_PRECISIONS = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}


def precision_of(element_type: str) -> str:
    if element_type.startswith("fp8"):
        return "fp8"
    return _PRECISIONS.get(element_type, element_type)


def describe_program(program: Any, migraphx_version: str) -> MIGraphXProgramInfo:
    """A loaded program's info (`program` is a _chainner_migraphx.Program): one NCHW
    input and output, a uniform integer upscale. Its precision is that of its
    convolutions and matrix products (compute_type; the input's type in a program
    without any), and "mixed" for a DUAL or DRFT program (its input named for its
    alignment, as TensorRT's), which keeps its ONNX's own types."""
    names = list(program.input_names)
    outputs = list(program.output_shapes)
    if len(names) != 1 or len(outputs) != 1:
        raise ValueError(
            f"Expected a program with one input and one output, got inputs {names} and"
            f" {len(outputs)} outputs."
        )
    shape = tuple(program.input_shapes[0])
    output = tuple(outputs[0])
    if len(shape) != 4 or len(output) != 4:
        raise ValueError(f"Expected NCHW tensors, got {shape} -> {output}.")
    _, channels, height, width = shape
    _, out_channels, out_height, out_width = output
    if (
        out_height % height
        or out_width % width
        or out_height // height != out_width // width
    ):
        raise ValueError(
            f"Input {shape} -> output {output} is not a uniform integer upscale."
        )
    if input_alignment(names[0]) is not None:
        precision = "mixed"
    else:
        precision = precision_of(program.compute_type or program.input_types[0])
    return MIGraphXProgramInfo(
        precision=precision,
        input_channels=channels,
        output_channels=out_channels,
        scale=out_height // height,
        gfx=program.gfx,
        migraphx_version=migraphx_version,
        height=height,
        width=width,
    )


def read_program_info(program_bytes: bytes, gpu_index: int) -> MIGraphXProgramInfo:
    """The info of a saved program, loaded on the GPU of HIP index `gpu_index`, which
    checks that it runs there (a program of another gfx target or MIGraphX version fails
    to load with a message saying to build it again)."""
    migraphx = native()
    program = migraphx.load_program(program_bytes, gpu_index)
    try:
        return describe_program(program, migraphx.MIGRAPHX_VERSION)
    finally:
        program.close()
