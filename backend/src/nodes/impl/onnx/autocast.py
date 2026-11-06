"""The mixed precision an earlier version's Convert To ONNX gave an FP32 ONNX with NVIDIA
ModelOpt's AutoCast: every node in FP16 or BF16 except those that would overflow or lose
too much, which stayed FP32, with the input and output in FP32. Such files still load and
build. TensorRT 11 builds strongly typed engines, so these types are the engine's
precision."""

from __future__ import annotations

import onnx

# The low precision AutoCast used, in the ONNX's metadata (load.py reads it back).
METADATA_KEY = "chainner_c.autocast"


def mixed_precision(model: onnx.ModelProto) -> str | None:
    """The data type of an ONNX from AutoCast ("mixed-fp16" or "mixed-bf16"); None
    for any other."""
    for entry in model.metadata_props:
        if entry.key == METADATA_KEY:
            return f"mixed-{entry.value}"
    return None
