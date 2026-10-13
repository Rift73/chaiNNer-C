"""TensorRT FP8: calibrated ONNX graphs downloaded once, then compiled like BF16."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import requests
import torch
from sanic.log import logger

from api import NodeContext

from .vosr2 import BundlePaths, read_configs
from .vosr2_engine_cache import (
    cache_lock,
    compile_worker,
    ensure_engine_set,
    file_hash,
    stored_hash,
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
CHUNK = 4 * 1024 * 1024


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


def unreachable(url: str, error: requests.RequestException) -> RuntimeError:
    if isinstance(error, requests.HTTPError) and error.response is not None:
        return RuntimeError(
            f"The VOSR FP8 graph server answered HTTP {error.response.status_code} "
            f"for {url}."
        )
    return RuntimeError(
        f"Could not download the VOSR FP8 graphs from {url}: {error}. Check the "
        "internet connection and run again; finished and partial files are kept."
    )


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


def check_bundle(paths: BundlePaths, manifest: dict, root: Path) -> None:
    """The engines carry the official weights; refuse a model folder with others."""
    source = manifest["source"]
    for path in paths.files():
        name = path.relative_to(paths.root).as_posix()
        if source["files"].get(name) != stored_hash(path, root):
            raise ValueError(
                "TensorRT FP8 uses graphs calibrated from the official VOSR 2.0 "
                f"weights ({source['repository']} at {source['revision'][:12]}), "
                f"but {name} in this model folder differs. Use the TensorRT or "
                "PyTorch backend for other weights."
            )


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


def download(
    context: NodeContext,
    url: str,
    target: Path,
    entry: dict,
    advance: Callable[[int], None],
) -> None:
    """Resume a partial file when the server honours ranges; verify, then publish."""
    part = target.with_name(target.name + ".part")
    size = entry["size"]
    offset = part.stat().st_size if part.is_file() else 0
    if offset > size:
        part.unlink()
        offset = 0
    digest = hashlib.sha256()
    if offset:
        with part.open("rb") as existing:
            while chunk := existing.read(CHUNK):
                context.suspend()
                digest.update(chunk)
    try:
        if offset < size:
            headers = {"Range": f"bytes={offset}-"} if offset else None
            with requests.get(
                url, headers=headers, stream=True, timeout=(15, 60)
            ) as response:
                response.raise_for_status()
                resumed = response.status_code == 206 and response.headers.get(
                    "Content-Range", ""
                ).startswith(f"bytes {offset}-")
                if offset and not resumed:
                    if response.status_code != 200:
                        raise RuntimeError(
                            f"The VOSR FP8 graph server answered HTTP "
                            f"{response.status_code} for a partial {url}."
                        )
                    # The server ignored the range: start this file again.
                    advance(-offset)
                    offset = 0
                    digest = hashlib.sha256()
                with part.open("ab" if offset else "wb") as output:
                    for chunk in response.iter_content(CHUNK):
                        context.suspend()
                        output.write(chunk)
                        digest.update(chunk)
                        offset += len(chunk)
                        if offset > size:
                            break
                        advance(len(chunk))
    except requests.RequestException as error:
        raise unreachable(url, error) from error
    except OSError as error:
        if error.errno == errno.ENOSPC:
            raise RuntimeError(
                f"The disk holding {target.parent} is full. Free some space and run "
                "again; the download resumes where it stopped."
            ) from error
        raise
    if offset < size:
        raise RuntimeError(
            f"The download of {target.name} stopped at {offset:,} of {size:,} bytes. "
            "Run again to resume it."
        )
    if offset > size or digest.hexdigest() != entry["sha256"]:
        part.unlink()
        raise RuntimeError(
            f"The downloaded {target.name} does not match its SHA-256, so it was "
            "deleted. Run again to download it again."
        )
    os.replace(part, target)


def fetch_graphs(context: NodeContext, folder: Path, manifest: dict) -> Path:
    pending: dict[str, dict] = {}
    for name, entry in manifest["files"].items():
        target = folder / name
        if target.is_file():
            if (
                target.stat().st_size == entry["size"]
                and file_hash(target) == entry["sha256"]
            ):
                continue
            target.unlink()
        pending[name] = entry
    if not pending:
        return folder
    total = sum(entry["size"] for entry in pending.values())
    done = sum(
        min(part.stat().st_size, entry["size"])
        for name, entry in pending.items()
        if (part := folder / f"{name}.part").is_file()
    )
    free = shutil.disk_usage(folder).free
    if free < total - done + ENGINE_BYTES:
        raise RuntimeError(
            f"The VOSR FP8 graphs need {(total - done) / 1e9:.1f} GB more in {folder}, "
            f"plus about {ENGINE_BYTES / 1e9:.0f} GB for the compiled engines; "
            f"{free / 1e9:.1f} GB are free."
        )
    logger.info(
        "Downloading the VOSR FP8 graphs once: %.2f GB into %s", total / 1e9, folder
    )
    progress = [done]

    def advance(count: int) -> None:
        progress[0] += count
        context.set_progress(progress[0] / total)

    for name, entry in pending.items():
        download(
            context, f"{HOST}{folder_name()}/{name}", folder / name, entry, advance
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
        check_bundle(paths, manifest, root)

        def compile_into(directory: Path) -> None:
            check_runtime(manifest)
            graphs = fetch_graphs(context, folder, manifest)
            compile_worker(context, "vosr2_fp8_build.py", graphs, directory, device)

        return ensure_engine_set(
            context, root, fp8_identity(device), "FP8", compile_into
        )
