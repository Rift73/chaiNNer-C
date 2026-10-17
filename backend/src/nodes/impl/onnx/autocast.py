"""Mixed precision for an FP32 ONNX with NVIDIA ModelOpt's AutoCast (the NVIDIA ModelOpt
package in the dependency manager). AutoCast runs the model once on a calibration image
and converts every node to FP16 or BF16 except those whose activations or weights would
overflow or lose too much there, which stay FP32; the input and output stay FP32.
TensorRT 11 builds strongly typed engines, so these types are the engine's precision."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Literal

import numpy as np
import onnx

# The low precision AutoCast used, in the ONNX's metadata (load.py reads it back).
METADATA_KEY = "chainner_c.autocast"

LowPrecision = Literal["fp16", "bf16"]


def autocast(onnx_bytes: bytes, low: LowPrecision, calibration: np.ndarray) -> bytes:
    """`onnx_bytes` (FP32) in mixed precision, calibrated on `calibration` (NCHW
    float32, a size the model accepts)."""
    try:
        from modelopt.onnx.autocast import convert_to_mixed_precision
    except ImportError as e:
        raise RuntimeError(
            "Mixed precision needs the NVIDIA ModelOpt package: install it in the"
            " dependency manager."
        ) from e

    name = onnx.load_from_string(onnx_bytes).graph.input[0].name
    with tempfile.TemporaryDirectory(prefix="chainner-autocast-") as temporary:
        root = Path(temporary)
        model_path = root / "model.onnx"
        model_path.write_bytes(onnx_bytes)
        calibration_path = root / "calibration.npz"
        # An .npz of the model's input, by name.
        with zipfile.ZipFile(calibration_path, "w") as archive:
            with archive.open(f"{name}.npy", "w") as file:
                np.save(file, calibration)
        model = convert_to_mixed_precision(
            onnx_path=str(model_path),
            low_precision_type=low,
            keep_io_types=True,
            calibration_data=str(calibration_path),
        )
    entry = model.metadata_props.add()
    entry.key, entry.value = METADATA_KEY, low
    return model.SerializeToString()


def mixed_precision(model: onnx.ModelProto) -> str | None:
    """The data type of an ONNX from autocast ("mixed-fp16" or "mixed-bf16"); None
    for any other."""
    for entry in model.metadata_props:
        if entry.key == METADATA_KEY:
            return f"mixed-{entry.value}"
    return None
