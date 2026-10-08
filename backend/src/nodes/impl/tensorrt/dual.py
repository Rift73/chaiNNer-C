"""DUAL's specialized TensorRT path: the exporter vendored from traiNNer-redux
(vendor/dual_tensorrt, see its README.md) and the engine build with DUAL's plugins as
AOT Python plugins (dual_aot.py), each run as a child process as that guide requires.
The chaiNNer worker never imports Triton or builds plugins; only the finished engine,
which embeds the plugins' kernels, is loaded."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# backend/src, and backend/src/vendor/dual_tensorrt (its own layout, so the guide's
# commands run as is)
SOURCE = Path(__file__).resolve().parents[3]
VENDOR = SOURCE / "vendor" / "dual_tensorrt"
# spandrel's DUAL preset tag -> the guide's canonical factory
FACTORIES = {
    "Light": "dual_light",
    "XS": "dual_xs",
    "S": "dual_s",
    "M": "dual_m",
    "L": "dual_l",
    "XL": "dual_xl",
}
# The exporter's manifest (export.json), carried in the ONNX model's metadata.
METADATA_KEY = "chainner_c.dual_tensorrt.export"


def factory_of(tags: list[str]) -> str:
    """The canonical factory of a spandrel DUAL model, from its preset tag."""
    preset = tags[0] if tags else ""
    if preset not in FACTORIES:
        raise ValueError(
            "TensorRT export supports the six DUAL presets (Light, XS, S, M, L, XL);"
            " this model is not one of them."
        )
    return FACTORIES[preset]


def _child(
    arguments: list[str], label: str, cwd: Path, *, cpu_only: bool = False
) -> str:
    """Run this Python in a child process; its output, or an error with the output's
    end."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    if cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, *arguments],
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(
            f"DUAL {label} failed (exit {result.returncode}):\n{output[-3000:]}"
        )
    return output


def export_onnx(
    state_dict: dict[str, Any], tags: list[str], scale: int, height: int, width: int
) -> bytes:
    """The TensorRT-specific ONNX of a folded DUAL model (spandrel folds on load), with
    its weights embedded and the exporter's manifest in its metadata."""
    import onnx
    from safetensors.torch import save_file

    factory = factory_of(tags)
    with tempfile.TemporaryDirectory(prefix="chainner-dual-export-") as temporary:
        root = Path(temporary)
        checkpoint = root / "folded.safetensors"
        save_file({k: v.contiguous() for k, v in state_dict.items()}, str(checkpoint))
        export = root / "export"
        _child(
            [
                "-m",
                "scripts.dual_tensorrt.export",
                f"--factory={factory}",
                f"--scale={scale}",
                f"--height={height}",
                f"--width={width}",
                f"--checkpoint={checkpoint}",
                "--folded",
                f"--output={export}",
            ],
            "export",
            VENDOR,
            cpu_only=True,
        )
        manifest = json.loads((export / "export.json").read_text(encoding="utf-8"))
        model = onnx.load(str(export / "model.onnx"), load_external_data=True)
    # The guide records default geometry; spandrel cannot read context windows either.
    manifest["geometry"] = "default (C32/C64 context windows assumed)"
    entry = model.metadata_props.add()
    entry.key, entry.value = METADATA_KEY, json.dumps(manifest)
    return model.SerializeToString()


def export_manifest(onnx_bytes: bytes) -> dict[str, Any] | None:
    """The exporter's manifest of a DUAL ONNX; None for any other model."""
    import onnx

    model = onnx.load_from_string(onnx_bytes)
    for entry in model.metadata_props:
        if entry.key == METADATA_KEY:
            return json.loads(entry.value)
    return None


def build_engine(onnx_bytes: bytes, manifest: dict[str, Any], gpu_index: int) -> bytes:
    """A TensorRT engine of a DUAL ONNX with its plugins (AOT Python plugins, Triton
    kernels compiled for this GPU) embedded, built by chaiNNer-C's TensorRT, the version
    that loads it, in dual_engine_worker.py."""
    with tempfile.TemporaryDirectory(prefix="chainner-dual-engine-") as temporary:
        model = Path(temporary) / "model.onnx"
        model.write_bytes(onnx_bytes)
        engine = Path(temporary) / "model.engine"
        _child(
            [
                "-m",
                "nodes.impl.tensorrt.dual_engine_worker",
                str(model),
                str(engine),
                str(gpu_index),
                manifest["specialization"]["plugin_key"],
            ],
            "engine build",
            SOURCE,
        )
        return engine.read_bytes()


__all__ = ["build_engine", "export_manifest", "export_onnx", "factory_of"]
