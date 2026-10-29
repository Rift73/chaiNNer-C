"""Compile a MIGraphX program in a child process (program.compile_in_child runs it from
backend/src): the custom-op libraries named load first, then the ONNX is parsed and
compiled for one size of its image input (its first input that is not a weight) on one
GPU and saved in MIGraphX's .mxr format. FP8's scales are calibrated on three images in
[0, 1], the range of every image Upscale Image gives a program: a smooth gradient, noise
and their mean.

    python -m nodes.impl.migraphx.program_worker ONNX PROGRAM GPU_INDEX CxHxW FP16 FP8
        EXHAUSTIVE [OP_LIBRARY ...]
"""

from __future__ import annotations

import sys

import numpy as np

from ..pytorch.vosr2_process import end_with_parent


def calibration(
    name: str, dtype: np.dtype, shape: tuple[int, int, int, int]
) -> list[dict[str, np.ndarray]]:
    """FP8's calibration inputs for the image input `name` of `shape` (NCHW)."""
    _, _, height, width = shape
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    gradient = np.broadcast_to((y + x) / 2, shape)
    noise = np.random.default_rng(0).random(shape, dtype=np.float32)
    return [
        {name: np.ascontiguousarray(image, dtype=dtype)}
        for image in (gradient, noise, (gradient + noise) / 2)
    ]


def main(
    onnx_path: str,
    program_path: str,
    gpu_index: str,
    dims: str,
    fp16: str,
    fp8: str,
    exhaustive: str,
    *op_libraries: str,
) -> None:
    import onnx
    from onnx.helper import tensor_dtype_to_np_dtype

    from .runner import load_migraphx

    graph = onnx.load(onnx_path, load_external_data=False).graph
    weights = {tensor.name for tensor in graph.initializer}
    image = next(i for i in graph.input if i.name not in weights)
    name = image.name
    dtype = tensor_dtype_to_np_dtype(image.type.tensor_type.elem_type)
    del graph, image
    migraphx = load_migraphx()
    for library in op_libraries:
        migraphx.load_op_library(library)
    channels, height, width = (int(v) for v in dims.split("x"))
    shape = (1, channels, height, width)
    program = migraphx.compile_onnx(
        onnx_path,
        device=int(gpu_index),
        fp16=fp16 == "1",
        fp8=fp8 == "1",
        input_dims={name: shape},
        exhaustive_tune=exhaustive == "1",
        calibration=calibration(name, dtype, shape) if fp8 == "1" else (),
    )
    program.save(program_path)


if __name__ == "__main__":
    end_with_parent()
    main(*sys.argv[1:])
