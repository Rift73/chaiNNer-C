"""DUAL's specialized TensorRT path: the exporter, AOT kernels, plugin build and engine
build vendored from traiNNer-redux (vendor/dual_tensorrt, see its README.md), each run
as a child process as that guide requires. The chaiNNer worker never imports Triton or
loads the plugin libraries; only the finished engine, which embeds them, is loaded."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from sanic.log import logger

# backend/src/vendor/dual_tensorrt: its own layout, so the guide's commands run as is.
VENDOR = Path(__file__).resolve().parents[3] / "vendor" / "dual_tensorrt"
# TensorRT 11.2's public headers (NVIDIA's open-source release, Apache-2.0).
TENSORRT_HEADERS = VENDOR.parent / "tensorrt"
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
# What a plugin bundle depends on besides its key and SM: the canonical source, the
# kernels and the plugin sources (the guide caches bundles by these identities).
BUNDLE_SOURCES = (
    "traiNNer/archs/dual_arch.py",
    "scripts/dual_tensorrt/kernels.py",
    "scripts/dual_tensorrt/aot.py",
    "tensorrt_plugins/dual_tensorrt/AttentionBarrier.cpp",
    "tensorrt_plugins/dual_tensorrt/CMakeLists.txt",
    "tensorrt_plugins/dual_tensorrt/Core.cpp",
    "tensorrt_plugins/dual_tensorrt/Export.h",
    "tensorrt_plugins/dual_tensorrt/Norm.cu",
    "tensorrt_plugins/dual_tensorrt/Project.cpp",
)


def factory_of(tags: list[str]) -> str:
    """The canonical factory of a spandrel DUAL model, from its preset tag."""
    preset = tags[0] if tags else ""
    if preset not in FACTORIES:
        raise ValueError(
            "TensorRT export supports the six DUAL presets (Light, XS, S, M, L, XL);"
            " this model is not one of them."
        )
    return FACTORIES[preset]


def _run(module: str, *arguments: str, cpu_only: bool = False) -> str:
    """Run a vendored module in a child process."""
    command = [sys.executable, "-m", f"scripts.dual_tensorrt.{module}", *arguments]
    return _child(command, module, cpu_only=cpu_only)


def _child(command: list[str], label: str, *, cpu_only: bool = False) -> str:
    """Run a child process from the vendored tree; its output, or an error with the
    output's end."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    if cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        command,
        cwd=VENDOR,
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
        _run(
            "export",
            f"--factory={factory}",
            f"--scale={scale}",
            f"--height={height}",
            f"--width={width}",
            f"--checkpoint={checkpoint}",
            "--folded",
            f"--output={export}",
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def headers_root(sdk: Path | None) -> Path:
    """The TensorRT root the plugins compile against: an SDK given by the user (its
    headers, and its import library when it has one), else the bundled TensorRT 11.2
    headers (the plugins call no TensorRT function, so they need no library)."""
    if sdk is None:
        return TENSORRT_HEADERS
    if not (sdk / "include" / "NvInfer.h").is_file():
        raise ValueError(f"{sdk} is not a TensorRT SDK: it has no include/NvInfer.h")
    return sdk


def bundle_id(root: Path) -> str:
    """The identity of what a plugin bundle is built from, besides its key and SM: the
    canonical source, kernels and plugin sources, and the TensorRT headers."""
    digest = hashlib.sha256()
    for name in BUNDLE_SOURCES:
        digest.update(name.encode() + b"\0" + _sha256(VENDOR / name).encode())
    for header in sorted((root / "include").glob("NvInfer*.h")):
        digest.update(header.name.encode() + b"\0" + _sha256(header).encode())
    return digest.hexdigest()[:12]


def plugin_bundle(export_dir: Path, sm: int, cache: Path, root: Path) -> Path:
    """plugins.json of the bundle for this export's specialization and SM: cached, or
    built (AOT kernels, then the four plugin libraries) into a new folder."""
    key = json.loads((export_dir / "export.json").read_text(encoding="utf-8"))[
        "specialization"
    ]["plugin_key"]
    bundle = cache / f"{key}_sm{sm}_{bundle_id(root)}"
    plugins = bundle / "plugins" / "plugins.json"
    if plugins.is_file():
        record = json.loads(plugins.read_text(encoding="utf-8"))
        if all(
            Path(entry["path"]).is_file()
            and _sha256(Path(entry["path"])) == entry["sha256"]
            for entry in record["libraries"]
        ):
            logger.info("DUAL plugins %s: cached in %s", key, bundle)
            return plugins
        raise RuntimeError(f"The cached DUAL plugins in {bundle} were modified")
    # A short build folder: nvcc's intermediate files fail past Windows' 260-character
    # path limit, which a deep cache folder reaches inside CMake's build tree.
    staging = Path(tempfile.mkdtemp(prefix="cdt-"))
    logger.info("DUAL plugins %s for SM %d: building in %s", key, sm, staging)
    _run(
        "aot", f"--export-dir={export_dir}", f"--output={staging / 'aot'}", f"--sm={sm}"
    )
    _run(
        "build",
        "plugins",
        f"--export-dir={export_dir}",
        f"--aot-dir={staging / 'aot'}",
        f"--trt-root={root}",
        f"--output={staging / 'plugins'}",
        "--execute",
    )
    # plugins.json records absolute library paths: rewrite them for the final folder.
    record = json.loads((staging / "plugins" / "plugins.json").read_text("utf-8"))
    for entry in record["libraries"]:
        relative = Path(entry["path"]).relative_to(staging)
        entry["path"] = str(bundle / relative)
    (staging / "plugins" / "plugins.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    shutil.move(staging, bundle)
    return plugins


def build_engine(
    onnx_bytes: bytes,
    manifest: dict[str, Any],
    sm: int,
    gpu_index: int,
    cache: Path,
    sdk: Path | None = None,
) -> bytes:
    """A TensorRT engine of a DUAL ONNX with its plugins serialized into it, built by
    chaiNNer-C's TensorRT (the version that loads it) in dual_engine_worker.py."""
    root = headers_root(sdk)
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="chainner-dual-engine-") as temporary:
        export_dir = Path(temporary) / "export"
        export_dir.mkdir()
        (export_dir / "model.onnx").write_bytes(onnx_bytes)
        # The weights are embedded here: the manifest's file identity is this file's.
        record = {
            **manifest,
            "files": {"model.onnx": _sha256(export_dir / "model.onnx")},
        }
        (export_dir / "export.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        plugins = plugin_bundle(export_dir, sm, cache, root)
        libraries = [
            entry["path"]
            for entry in json.loads(plugins.read_text(encoding="utf-8"))["libraries"]
        ]
        engine = Path(temporary) / "model.engine"
        _child(
            [
                sys.executable,
                str(Path(__file__).with_name("dual_engine_worker.py")),
                str(export_dir / "model.onnx"),
                str(engine),
                str(gpu_index),
                *libraries,
            ],
            "engine build",
        )
        return engine.read_bytes()


__all__ = ["build_engine", "export_manifest", "export_onnx", "factory_of"]
