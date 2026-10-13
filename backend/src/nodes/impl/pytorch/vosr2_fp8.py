"""TensorRT FP8: calibrated ONNX graphs downloaded once, then compiled like BF16."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import requests
import torch

from api import NodeContext

from .vosr2 import BundlePaths, read_configs
from .vosr2_download import fetch_files, unreachable
from .vosr2_engine_cache import (
    cache_lock,
    compile_worker,
    ensure_engine_set,
    file_hash,
)
from .vosr2_trt_models import STAGES
from .vosr2_trt_plugin import load_plugin, plugin_path

# The public folder is named after its manifest; this pin makes the set immutable.
HOST = "https://steropes.feralhosting.com/rift/"
MANIFEST_SHA256 = "f9e142df5846d2da6b0d3c780022f23942094e44497b5b848d709dfb2d9234c6"
# FP8 E4M3 tensor cores: Ada (8.9), Hopper and Blackwell.
FP8_CAPABILITY = (8, 9)
# Room kept for the ten compiled engines next to the graphs.
ENGINE_BYTES = 2_000_000_000


def folder_name() -> str:
    return "vosr2-fp8-" + MANIFEST_SHA256[:12]


def fp8_identity(device: torch.device) -> dict:
    """What determines the engines: graphs, TensorRT, GPU, plugin, builder settings."""
    import tensorrt as trt

    properties = torch.cuda.get_device_properties(device)
    return {
        "format": 2,
        "graphs": MANIFEST_SHA256,
        "tensorrt": trt.__version__,
        "gpu": properties.name,
        "capability": [properties.major, properties.minor],
        "plugin_sha256": file_hash(plugin_path()),
        "builder_sha256": file_hash(Path(__file__).with_name("vosr2_trt_compile.py")),
    }


def fetch_manifest(folder: Path) -> dict:
    path = folder / "manifest.json"
    data = path.read_bytes() if path.is_file() else b""
    if hashlib.sha256(data).hexdigest() != MANIFEST_SHA256:
        url = f"{HOST}{folder_name()}/manifest.json"
        try:
            with requests.get(url, timeout=(15, 60)) as response:
                response.raise_for_status()
                data = response.content
        except requests.RequestException as error:
            raise unreachable(url, error) from error
        if hashlib.sha256(data).hexdigest() != MANIFEST_SHA256:
            raise RuntimeError(
                f"The VOSR FP8 manifest at {url} does not match the SHA-256 this "
                "version of chaiNNer-C expects, so nothing was downloaded."
            )
        folder.mkdir(parents=True, exist_ok=True)
        part = folder / "manifest.json.part"
        part.write_bytes(data)
        os.replace(part, path)
    manifest = json.loads(data)
    stages, files = manifest["stages"], manifest["files"]
    if (
        manifest["format"] != 1
        or set(stages) != set(STAGES)
        or any(
            entry["graph"] != f"{name}.onnx"
            or entry.get("data", f"{name}.onnx.data") != f"{name}.onnx.data"
            or not {entry["graph"], entry.get("data", entry["graph"])} <= set(files)
            for name, entry in stages.items()
        )
        or any(Path(name).name != name or name in ("", ".", "..") for name in files)
    ):
        raise RuntimeError("The VOSR FP8 manifest does not describe this app's stages.")
    return manifest


def check_runtime(manifest: dict) -> None:
    import tensorrt as trt

    major = manifest["tensorrt_major"]
    if trt.__version__.split(".")[0] != str(major):
        raise RuntimeError(
            f"The VOSR FP8 graphs need TensorRT {major}; "
            f"TensorRT {trt.__version__} is installed."
        )
    load_plugin()
    registry = trt.get_plugin_registry()
    for plugin in manifest["plugins"]:
        if (
            registry.get_creator(plugin["name"], plugin["version"], plugin["namespace"])
            is None
        ):
            raise RuntimeError(
                f"The VOSR TensorRT plugin {plugin_path()} does not provide "
                f"{plugin['name']} version {plugin['version']}."
            )


def fetch_graphs(context: NodeContext, folder: Path, manifest: dict) -> Path:
    fetch_files(
        context,
        folder,
        {
            name: {**entry, "url": f"{HOST}{folder_name()}/{name}"}
            for name, entry in manifest["files"].items()
        },
        file_hash,
        "the VOSR FP8 graphs",
        ENGINE_BYTES,
    )
    return folder


def ensure_fp8_engine(
    context: NodeContext, paths: BundlePaths, device: torch.device
) -> Path:
    """Download the hosted graph set on first use, then compile and cache it."""
    read_configs(paths)
    properties = torch.cuda.get_device_properties(device)
    if (properties.major, properties.minor) < FP8_CAPABILITY:
        raise ValueError(
            "TensorRT FP8 requires an NVIDIA GPU with FP8 support (RTX 40 series or "
            f"newer); {properties.name} does not have it. Select the TensorRT backend."
        )
    root = context.storage_dir / "vosr2-tensorrt-fp8"
    folder = context.storage_dir / "vosr2-fp8-graphs" / folder_name()
    root.mkdir(parents=True, exist_ok=True)
    with cache_lock(root / "compile.lock", context):
        manifest = fetch_manifest(folder)

        def compile_into(directory: Path) -> None:
            check_runtime(manifest)
            graphs = fetch_graphs(context, folder, manifest)
            compile_worker(context, "vosr2_fp8_build.py", graphs, directory, device)

        return ensure_engine_set(
            context, root, fp8_identity(device), "FP8", compile_into
        )
