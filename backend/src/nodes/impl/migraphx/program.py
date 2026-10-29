"""Compiling an ONNX model into a MIGraphX program, in a child process: a compile takes
minutes and its own memory, Stop ends it (tensorrt.dual.run_child), and a crash in
MIGraphX ends only the child. The child (program_worker.py) loads the custom-op libraries
it is given before it compiles, so the DUAL and DRFT routes (dual.py, drft.py) compile
their lowered ONNX here too."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ..tensorrt.dual import SOURCE, run_child

if TYPE_CHECKING:
    from api import NodeContext


def compile_in_child(
    onnx_bytes: bytes,
    gpu_index: int,
    size: tuple[int, int],
    *,
    fp16: bool,
    fp8: bool,
    exhaustive_tune: bool,
    op_libraries: Sequence[Path] = (),
    label: str = "MIGraphX program build",
    channels: int = 3,
    context: NodeContext | None = None,
) -> bytes:
    """The saved program (MIGraphX's .mxr bytes) of an ONNX model compiled on the GPU of
    HIP index `gpu_index` for one input of `channels` x `size` (height, width), in FP16
    or FP8 if asked (else in the ONNX's own types), with MIGraphX's exhaustive tuning if
    asked, after loading `op_libraries`. An error starts with `label`."""
    height, width = size
    with tempfile.TemporaryDirectory(prefix="chaiNNer-migraphx-") as temporary:
        model = Path(temporary) / "model.onnx"
        model.write_bytes(onnx_bytes)
        program = Path(temporary) / "model.mxr"
        run_child(
            [
                "-m",
                "nodes.impl.migraphx.program_worker",
                str(model),
                str(program),
                str(gpu_index),
                f"{channels}x{height}x{width}",
                "1" if fp16 else "0",
                "1" if fp8 else "0",
                "1" if exhaustive_tune else "0",
                *(str(library) for library in op_libraries),
            ],
            label,
            SOURCE,
            context,
        )
        return program.read_bytes()
