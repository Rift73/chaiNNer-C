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
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil

from ..pytorch.vosr2_process import OwnedJob

if TYPE_CHECKING:
    import onnx

    from api import NodeContext

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
# A dynamic DUAL ONNX takes input sizes that are multiples of this (as the PyTorch
# model pads its input), so images need no more padding than PyTorch gives them. Every
# DUAL engine's input is named for it (aligned_input_name; a fixed size is a multiple of
# it too): the name marks the engine as DUAL's, in mixed precision, when it is loaded
# again, and Upscale Image's tiler pads to it instead of its default 64 px; Build
# Engine's profile must keep to it.
ALIGNMENT = 4
ALIGNED_INPUT = "input_aligned_"


def aligned_input_name(alignment: int) -> str:
    return f"{ALIGNED_INPUT}{alignment}"


def input_alignment(name: str) -> int | None:
    """The alignment an engine input's name carries; None for other engines."""
    suffix = name.removeprefix(ALIGNED_INPUT)
    return int(suffix) if suffix != name and suffix.isdigit() else None


def factory_of(tags: list[str]) -> str:
    """The canonical factory of a spandrel DUAL model, from its preset tag."""
    preset = tags[0] if tags else ""
    if preset not in FACTORIES:
        raise ValueError(
            "TensorRT export supports the six DUAL presets (Light, XS, S, M, L, XL);"
            " this model is not one of them."
        )
    return FACTORIES[preset]


def run_child(
    arguments: list[str],
    label: str,
    cwd: Path,
    context: NodeContext | None = None,
    *,
    cpu_only: bool = False,
) -> str:
    """Run this Python in a child process; its output, or an error with the output's
    end.

    The child ends with the call on any exit: Stop (context.check_aborted(), polled
    while it runs) or an error kills it, and on Windows it runs in a kill-on-close job,
    so a worker that dies takes it along (the host's /kill restarts a worker that does
    not stop within 3 s, and an engine build kept the GPU until it finished)."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    if cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""
    executable = sys.executable
    creationflags = 0
    job = None
    if os.name == "nt":
        # As vosr2_engine_cache.compile_worker: own the interpreter, not the venv
        # redirector, and join the job before it can start anything.
        executable = getattr(sys, "_base_executable", sys.executable)
        if executable != sys.executable:
            env["__PYVENV_LAUNCHER__"] = sys.executable
        creationflags = subprocess.CREATE_NO_WINDOW | 0x00000004  # CREATE_SUSPENDED
        job = OwnedJob()
    process = None
    try:
        with tempfile.TemporaryFile() as log:
            # A file, not pipes: no output volume can block the child.
            process = subprocess.Popen(
                [executable, *arguments],
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
            if job is not None:
                job.assign(process)
                psutil.Process(process.pid).resume()
            while process.poll() is None:
                if context is not None:
                    context.check_aborted()
                time.sleep(0.05)
            returncode = process.returncode
            log.seek(0)
            output = log.read().decode("utf-8", errors="replace")
    finally:
        if job is not None:
            job.close()
        elif process is not None and process.poll() is None:
            process.kill()
        if process is not None:
            process.wait(timeout=20)
    if returncode != 0:
        raise RuntimeError(
            f"DUAL {label} failed (exit {returncode}):\n{output[-3000:]}"
        )
    return output


def export_onnx(
    state_dict: dict[str, Any],
    tags: list[str],
    scale: int,
    size: tuple[int, int] | None,
    *,
    context: NodeContext | None = None,
) -> bytes:
    """The TensorRT-specific ONNX of a folded DUAL model (spandrel folds on load), with
    its weights embedded and the exporter's manifest in its metadata: for one input
    size (height, width), or for every size (None; scripts/dual_tensorrt/dynamic.py)."""
    import onnx
    from safetensors.torch import save_file

    factory = factory_of(tags)
    with tempfile.TemporaryDirectory(prefix="chainner-dual-export-") as temporary:
        root = Path(temporary)
        checkpoint = root / "folded.safetensors"
        save_file({k: v.contiguous() for k, v in state_dict.items()}, str(checkpoint))
        export = root / "export"
        if size is None:
            exporter = ["scripts.dual_tensorrt.dynamic", f"--alignment={ALIGNMENT}"]
        else:
            height, width = size
            exporter = [
                "scripts.dual_tensorrt.export",
                f"--height={height}",
                f"--width={width}",
            ]
        run_child(
            [
                "-m",
                *exporter,
                f"--factory={factory}",
                f"--scale={scale}",
                f"--checkpoint={checkpoint}",
                "--folded",
                f"--output={export}",
            ],
            "export",
            VENDOR,
            context,
            cpu_only=True,
        )
        manifest = json.loads((export / "export.json").read_text(encoding="utf-8"))
        model = onnx.load(str(export / "model.onnx"), load_external_data=True)
    # The guide records default geometry; spandrel cannot read context windows either.
    manifest["geometry"] = "default (C32/C64 context windows assumed)"
    entry = model.metadata_props.add()
    entry.key, entry.value = METADATA_KEY, json.dumps(manifest)
    return model.SerializeToString()


def export_manifest(model: bytes | onnx.ModelProto) -> dict[str, Any] | None:
    """The exporter's manifest of a DUAL ONNX (its bytes or the parsed model); None for
    any other model."""
    import onnx

    if isinstance(model, bytes):
        model = onnx.load_from_string(model)
    for entry in model.metadata_props:
        if entry.key == METADATA_KEY:
            return json.loads(entry.value)
    return None


Size = tuple[int, int]


def check_profile(manifest: dict[str, Any], low: Size, opt: Size, high: Size) -> None:
    """Refuse a dynamic DUAL engine's (height, width) profile that the ONNX is not
    exact for, or that is not ordered."""
    alignment = manifest["specialization"]["alignment"]
    for name, (h, w) in (("minimum", low), ("optimal", opt), ("maximum", high)):
        if h % alignment or w % alignment:
            raise ValueError(
                f"A dynamic DUAL engine's {name} size must be a multiple of"
                f" {alignment} px (got {h}x{w})."
            )
    if not (low[0] <= opt[0] <= high[0] and low[1] <= opt[1] <= high[1]):
        raise ValueError(
            "A dynamic DUAL engine needs minimum <= optimal <= maximum sizes."
        )


def build_engine(
    onnx_bytes: bytes,
    manifest: dict[str, Any],
    gpu_index: int,
    profile: tuple[Size, Size, Size] | None = None,
    *,
    context: NodeContext | None = None,
) -> bytes:
    """A TensorRT engine of a DUAL ONNX with its plugins (AOT Python plugins, Triton
    kernels compiled for this GPU) embedded, built by chaiNNer-C's TensorRT, the version
    that loads it, in dual_engine_worker.py. A dynamic ONNX takes its engine's
    (minimum, optimal, maximum) (height, width) profile (see check_profile)."""
    spec = manifest["specialization"]
    if spec.get("dynamic", False) != (profile is not None):
        raise ValueError("Only a dynamic DUAL ONNX takes a size profile.")
    sizes = (
        []
        if profile is None
        else [*(f"{h}x{w}" for h, w in profile), str(spec["alignment"])]
    )
    with tempfile.TemporaryDirectory(prefix="chainner-dual-engine-") as temporary:
        model = Path(temporary) / "model.onnx"
        model.write_bytes(onnx_bytes)
        engine = Path(temporary) / "model.engine"
        run_child(
            [
                "-m",
                "nodes.impl.tensorrt.dual_engine_worker",
                str(model),
                str(engine),
                str(gpu_index),
                manifest["specialization"]["plugin_key"],
                *sizes,
            ],
            "engine build",
            SOURCE,
            context,
        )
        return engine.read_bytes()


__all__ = [
    "ALIGNMENT",
    "aligned_input_name",
    "build_engine",
    "check_profile",
    "export_manifest",
    "export_onnx",
    "factory_of",
    "input_alignment",
]
