"""A video model's ONNX bundle: the bytes of the `OnnxModel` Convert To ONNX gives for an
RDUAL, RSR or SMoVSR model at one input size. It holds rdual_export.py's, rsr_export.py's
or smovsr_export.py's output (each part's model.onnx and weights.bin, and the manifest
export.json, whose "architecture" names RSR or SMoVSR; a manifest without it is RDUAL's) in the engine set's
container layout (rdual_engine_set.pack_container) with its own magic (which names
RDUAL, the first). Save Model writes it as one file, Load Model reads it, and only Build
Engine uses it (the architecture's engine worker builds the engine set from the files
written out again); every other ONNX user refuses it.

This module must not import ONNX, TensorRT or PyTorch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..tensorrt.rdual_engine_set import EngineSetError, pack_container, unpack_container
from .model import MIXED, OnnxGeneric, OnnxInfo, OnnxModel

MAGIC = b"chaiNNer-C RDUAL ONNX\n"
MANIFEST = "export.json"
# The files of each part's folder (rdual_export.save)
PART_FILES = ("model.onnx", "weights.bin")
_WHAT = "ONNX bundle"
# The architecture of a manifest that names none
FIRST = "RDUAL"
_REMEDY = "Convert the model to ONNX again."


def is_bundle(data: bytes) -> bool:
    return data.startswith(MAGIC)


def pack(folder: Path, opset: int) -> bytes:
    """The bundle of an export `folder` (rdual_export.py's OUTPUT), whose ONNX parts
    have `opset`."""
    manifest = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
    files = {
        f"{part}/{name}": (folder / part / name).read_bytes()
        for part in manifest["parts"]
        for name in PART_FILES
    }
    return pack_container(MAGIC, {"manifest": manifest, "opset": opset}, files)


def architecture(manifest: dict[str, Any]) -> str:
    """The architecture a bundle's manifest is of."""
    return str(manifest.get("architecture", FIRST))


def unpack(data: bytes) -> tuple[dict[str, Any], int, dict[str, bytes]]:
    """A bundle's manifest, opset and files (relative path -> bytes), all checked."""
    header, files = unpack_container(MAGIC, data, _WHAT, "file", _REMEDY)
    try:
        manifest = header["manifest"]
        opset = int(header["opset"])
        expected = [f"{p}/{n}" for p in manifest["parts"] for n in PART_FILES]
        size = (int(manifest["height"]), int(manifest["width"]), int(manifest["scale"]))
    except (KeyError, TypeError, ValueError) as error:
        raise EngineSetError(f"This {_WHAT} is damaged") from error
    if list(files) != expected or min(size) < 1:
        raise EngineSetError(f"This {_WHAT} is damaged")
    return manifest, opset, files


def write(data: bytes, folder: Path) -> dict[str, Any]:
    """The bundle's files written out as rdual_export.py wrote them in `folder`; its
    manifest."""
    manifest, _, files = unpack(data)
    for path, content in files.items():
        target = folder / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (folder / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load(data: bytes) -> OnnxGeneric:
    """The ONNX model of a bundle: its architecture's fixed input size and scale, in
    mixed precision (RDUAL) or FP16 with FP32 images (the others)."""
    manifest, opset, _ = unpack(data)
    arch = architecture(manifest)
    height, width = int(manifest["height"]), int(manifest["width"])
    scale = int(manifest["scale"])
    info = OnnxInfo(
        opset=opset,
        dtype=MIXED if arch == FIRST else "fp16",
        scale_width=scale,
        scale_height=scale,
        fixed_input_width=width,
        fixed_input_height=height,
        input_channels=3,
        output_channels=3,
        arch=arch,
    )
    return OnnxGeneric(data, info)


def refuse(model: OnnxModel, user: str) -> None:
    """Raises if `model` is a bundle, which `user` (a node's name) cannot run."""
    if is_bundle(model.bytes):
        arch = architecture(unpack(model.bytes)[0])
        raise ValueError(
            f"{user} cannot use an {arch} ONNX: it holds a video model's TensorRT parts,"
            " which only TensorRT's Build Engine reads."
        )


__all__ = [
    "MAGIC",
    "architecture",
    "is_bundle",
    "load",
    "pack",
    "refuse",
    "unpack",
    "write",
]
