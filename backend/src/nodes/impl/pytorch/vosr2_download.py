"""Pinned downloads: the official VOSR 2.0 model, the YuNet face detector and the
TensorRT FP8 graph set; and the BF16 copy of the model's DiT."""

from __future__ import annotations

import errno
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import requests
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from sanic.log import logger

from api import NodeContext

from ...utils.utils import file_identity
from .vosr2 import BundlePaths
from .vosr2_engine_cache import file_lock, forget_hash, stored_hash, write_json
from .vosr2_official import FILES, REVISION

MODEL_HOST = "https://huggingface.co/CSWRY/VOSR/resolve/"
CHUNK = 4 * 1024 * 1024


def unreachable(url: str, error: requests.RequestException) -> RuntimeError:
    if isinstance(error, requests.HTTPError) and error.response is not None:
        return RuntimeError(
            f"The server answered HTTP {error.response.status_code} for {url}."
        )
    return RuntimeError(
        f"Could not download {url}: {error}. Check the internet connection and run "
        "again; finished and partial files are kept."
    )


def download(
    context: NodeContext,
    target: Path,
    entry: dict,
    advance: Callable[[int], None],
    file_digest: Callable[[Path], str],
    forget_digest: Callable[[Path], None],
) -> None:
    """Resume a partial file when the server honours ranges; publish, then verify the
    published file on disk. The caller holds the folder's download lock."""
    url = entry["url"]
    part = target.with_name(target.name + ".part")
    size = entry["size"]
    offset = part.stat().st_size if part.is_file() else 0
    if offset > size:
        part.unlink()
        offset = 0
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
                            f"The server answered HTTP {response.status_code} "
                            f"for a partial {url}."
                        )
                    # The server ignored the range: start this file again.
                    advance(-offset)
                    offset = 0
                with part.open("ab" if offset else "wb") as output:
                    for chunk in response.iter_content(CHUNK):
                        context.suspend()
                        output.write(chunk)
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
    mismatch = RuntimeError(
        f"The downloaded {target.name} does not match its SHA-256, so it was "
        "deleted. Run again to download it again."
    )
    if offset > size:
        part.unlink()
        raise mismatch
    forget_digest(target)
    os.replace(part, target)
    # The bytes on disk are what later runs load, not the bytes received.
    if file_digest(target) != entry["sha256"]:
        forget_digest(target)
        target.unlink()
        raise mismatch


def fetch_files(
    context: NodeContext,
    folder: Path,
    files: dict[str, dict],
    file_digest: Callable[[Path], str],
    forget_digest: Callable[[Path], None],
    label: str,
    reserve: int = 0,
) -> None:
    """Download each missing or mismatched file of files ({name: size, sha256, url}).

    One process at a time: another chaiNNer process (a second window, a CLI run) that
    shares the storage folder waits for the folder's lock, then finds the published
    files; two never write one partial file together."""
    folder.mkdir(parents=True, exist_ok=True)
    with file_lock(
        folder / "download.lock",
        context,
        f"another chaiNNer process to download {label}",
    ):
        pending: dict[str, dict] = {}
        for name, entry in files.items():
            target = folder / name
            if target.is_file():
                if (
                    target.stat().st_size == entry["size"]
                    and file_digest(target) == entry["sha256"]
                ):
                    continue
                forget_digest(target)
                target.unlink()
            pending[name] = entry
        if not pending:
            return
        total = sum(entry["size"] for entry in pending.values())
        done = sum(
            min(part.stat().st_size, entry["size"])
            for name, entry in pending.items()
            if (part := folder / f"{name}.part").is_file()
        )
        free = shutil.disk_usage(folder).free
        if free < total - done + reserve:
            engines = (
                f", plus about {reserve / 1e9:.0f} GB for the compiled engines"
                if reserve
                else ""
            )
            raise RuntimeError(
                f"Downloading {label} needs {(total - done) / 1e9:.1f} GB more in "
                f"{folder}{engines}; {free / 1e9:.1f} GB are free."
            )
        size = f"{total / 1e9:.2f} GB" if total >= 1e8 else f"{total / 1e6:.2f} MB"
        logger.info("Downloading %s once: %s into %s", label, size, folder)
        progress = [done]

        def advance(count: int) -> None:
            progress[0] += count
            context.set_progress(progress[0] / total)

        for name, entry in pending.items():
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            download(context, target, entry, advance, file_digest, forget_digest)


def model_folder(storage_dir: Path) -> Path:
    return storage_dir / "vosr2-model" / REVISION


# OpenCV Zoo's YuNet face detector (MIT licence), the export with a dynamic input size that
# OpenCV 5 needs. Faces that a window seam would split get a window of their own.
FACE_MODEL = "face_detection_yunet_2026may.onnx"
FACE_FILE = {
    "size": 229738,
    "sha256": "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
    "url": "https://media.githubusercontent.com/media/opencv/opencv_zoo/"
    "26cc381e4d2594bb9f47a26eb8fd96c94a13660d/models/face_detection_yunet/"
    + FACE_MODEL,
}


def ensure_face_model(context: NodeContext) -> Path:
    """The YuNet face detector: downloaded on first use, then verified once."""
    folder = context.storage_dir / "vosr2-faces"
    folder.mkdir(parents=True, exist_ok=True)
    fetch_files(
        context,
        folder,
        {FACE_MODEL: FACE_FILE},
        lambda path: stored_hash(path, folder),
        lambda path: forget_hash(path, folder),
        "the YuNet face detector",
    )
    return folder / FACE_MODEL


def ensure_model(context: NodeContext) -> BundlePaths:
    """The official VOSR 2.0 model: downloaded on first use, then verified once.

    Present files are hashed once and remembered while unchanged (stored_hash).
    """
    folder = model_folder(context.storage_dir)
    folder.mkdir(parents=True, exist_ok=True)
    fetch_files(
        context,
        folder,
        {
            local: {
                "size": size,
                "sha256": digest,
                "url": f"{MODEL_HOST}{REVISION}/{remote}",
            }
            for remote, local, size, digest in FILES
        },
        lambda path: stored_hash(path, folder.parent),
        lambda path: forget_hash(path, folder.parent),
        "the VOSR 2.0 model",
    )
    return BundlePaths(folder)


# The BF16 copy's format (bf16_dit): bump it when the conversion changes.
BF16_FORMAT = 1


def partial(target: Path) -> Path:
    """Where convert_checkpoint writes target until it is complete."""
    return target.with_name(f"{target.name}.tmp")


def convert_checkpoint(source: Path, target: Path, dtype: torch.dtype) -> None:
    """Write the safetensors checkpoint at source with every tensor converted to dtype
    (tensor.to, as load_weights converts) into target, which appears only when complete.
    One source tensor at a time: the converted ones are all that stays in memory."""
    with safe_open(source, framework="pt") as checkpoint:
        names = checkpoint.keys()
        state = {name: checkpoint.get_tensor(name).to(dtype=dtype) for name in names}
    part = partial(target)
    try:
        save_file(state, part)
        os.replace(part, target)
    finally:
        part.unlink(missing_ok=True)


def bf16_dit(context: NodeContext, paths: BundlePaths) -> BundlePaths:
    """paths with the DiT read from its BF16 copy beside the official checkpoint, which
    loads without converting 5.6 GB of FP32 weights. The first BF16 load makes it; later
    loads check it by size and SHA-256 (remembered while the file is unchanged, as for
    the official files: stored_hash) and their sidecar's record of the source and
    format, and make it again on any difference. Without room for the copy and 1 GB
    more, paths as they are (the load converts, as before)."""
    source = paths.dit
    target = source.with_name("ema_model.bf16.safetensors")
    sidecar = source.with_name("ema_model.bf16.json")
    with file_lock(
        paths.root / "download.lock",
        context,
        "another chaiNNer process to convert the VOSR 2.0 model",
    ):
        expected = {
            "format": BF16_FORMAT,
            "dtype": "bfloat16",
            "source_sha256": stored_hash(source, paths.root.parent),
            "source_identity": list(file_identity(source)),
        }
        try:
            record = json.loads(sidecar.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            record = None  # never made, or damaged: make the copy again
        if (
            isinstance(record, dict)
            and all(record.get(name) == value for name, value in expected.items())
            and target.is_file()
            and target.stat().st_size == record.get("size")
            and stored_hash(target, paths.root.parent) == record.get("sha256")
        ):
            return replace(paths, dit_copy=target)
        try:
            # Unusable now: the copy, its sidecar and a killed conversion's part go
            # before the room for a new copy is counted.
            forget_hash(target, paths.root.parent)
            for stale in (sidecar, target, partial(target)):
                stale.unlink(missing_ok=True)
            size = source.stat().st_size // 2
            free = shutil.disk_usage(target.parent).free
            if free < size + 1_000_000_000:
                logger.info(
                    "Not keeping a BF16 copy of the VOSR 2.0 DiT: it needs %.1f GB and "
                    "1 GB more in %s; %.1f GB are free. Each cold load converts it.",
                    size / 1e9,
                    target.parent,
                    free / 1e9,
                )
                return paths
            logger.info("Converting the VOSR 2.0 DiT to BF16 once: %s", target)
            convert_checkpoint(source, target, torch.bfloat16)
            write_json(
                sidecar,
                {
                    **expected,
                    "sha256": stored_hash(target, paths.root.parent),
                    "size": target.stat().st_size,
                },
            )
        except OSError:
            # The upscale goes on as before the copy: this load converts the weights.
            logger.warning(
                "Could not keep a BF16 copy of the VOSR 2.0 DiT in %s; this load "
                "converts the FP32 weights.",
                target.parent,
                exc_info=True,
            )
            return paths
    return replace(paths, dit_copy=target)
