"""TensorRT FP8: calibrated ONNX graphs downloaded once, then compiled like BF16.

The graphs are only the source of the engines, so the set stays compressed
(Zstandard, 2.7 GB instead of 3.4 GB): a build unpacks it, at about 1 GB/s, into a
folder it deletes afterwards."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import requests
import torch
from compression import zstd
from sanic.log import logger

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
MANIFEST_SHA256 = "3ebb2423591d5fe56f457d3275a88d9d1e478b77feb09788f4201bb285383585"
# The graphs themselves: the format 1 set, which the format 2 manifest above compresses
# file by file. The engines are built from these bytes, so they key the engine cache.
GRAPHS_SHA256 = "f9e142df5846d2da6b0d3c780022f23942094e44497b5b848d709dfb2d9234c6"
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
        "graphs": GRAPHS_SHA256,
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
        manifest["format"] != 2
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


def stored_name(name: str, entry: dict) -> str:
    """A file's name in the graph folder and on the host: a compressed one ends in
    .zst."""
    return f"{name}.zst" if "zstd" in entry else name


def stage_files(manifest: dict) -> list[str]:
    """The files a build reads: each stage's graph and its external data."""
    return [
        name
        for stage in manifest["stages"].values()
        for name in (stage["graph"], stage.get("data"))
        if name
    ]


def unpacked_bytes(manifest: dict) -> int:
    return sum(manifest["files"][name]["size"] for name in stage_files(manifest))


def fetch_graphs(context: NodeContext, folder: Path, manifest: dict) -> Path:
    fetch_files(
        context,
        folder,
        {
            stored_name(name, entry): {
                **entry.get("zstd", entry),
                "url": f"{HOST}{folder_name()}/{stored_name(name, entry)}",
            }
            for name, entry in manifest["files"].items()
        },
        file_hash,
        "the VOSR FP8 graphs",
        # The build unpacks the graphs next to the set, then writes the engines.
        ENGINE_BYTES + unpacked_bytes(manifest),
    )
    return folder


def remove_tree(path: Path) -> None:
    def failed(_function: object, name: str, error: BaseException) -> None:
        logger.warning("Could not delete %s: %s", name, error)

    if path.exists():
        shutil.rmtree(path, onexc=failed)


@contextmanager
def unpacked(context: NodeContext, folder: Path, manifest: dict) -> Iterator[Path]:
    """The stage graphs of a fetched set as plain files, each checked against its
    SHA-256, for one build; the folder is deleted afterwards (and before, after a build
    that was interrupted)."""
    target = folder.with_name(f"{folder.name}-unpacked")
    remove_tree(target)
    total = unpacked_bytes(manifest)
    free = shutil.disk_usage(folder).free
    if free < total + ENGINE_BYTES:
        raise RuntimeError(
            f"Unpacking the VOSR FP8 graphs needs {total / 1e9:.1f} GB in {folder}, plus"
            f" about {ENGINE_BYTES / 1e9:.0f} GB for the compiled engines;"
            f" {free / 1e9:.1f} GB are free."
        )
    target.mkdir(parents=True)
    try:
        logger.info("Unpacking the VOSR FP8 graphs for the build: %.1f GB", total / 1e9)
        done = 0
        for name in stage_files(manifest):
            entry = manifest["files"][name]
            source = folder / stored_name(name, entry)
            digest = hashlib.sha256()
            reader = zstd.open(source) if "zstd" in entry else source.open("rb")
            with reader, (target / name).open("wb") as writer:
                while chunk := reader.read(16 << 20):
                    context.check_aborted()
                    digest.update(chunk)
                    writer.write(chunk)
                    done += len(chunk)
                    context.set_progress(done / total)
            if digest.hexdigest() != entry["sha256"]:
                raise RuntimeError(
                    f"The unpacked VOSR FP8 graph {name} does not match its SHA-256;"
                    f" delete {source} and run again to download it again."
                )
        yield target
    finally:
        remove_tree(target)


def remove_other_sets(graphs: Path) -> None:
    """Delete the graph sets of other manifests, which nothing uses again (earlier
    chaiNNer-C versions kept theirs unpacked: 3.4 GB)."""
    if not graphs.is_dir():
        return
    for path in graphs.iterdir():
        if path.is_dir() and path.name != folder_name():
            logger.info("Deleting unused VOSR FP8 graphs: %s", path)
            remove_tree(path)


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
        remove_other_sets(folder.parent)
        manifest = fetch_manifest(folder)

        def compile_into(directory: Path) -> None:
            check_runtime(manifest)
            fetch_graphs(context, folder, manifest)
            with unpacked(context, folder, manifest) as graphs:
                compile_worker(context, "vosr2_fp8_build.py", graphs, directory, device)

        return ensure_engine_set(
            context, root, fp8_identity(device), "FP8", compile_into
        )
