"""Pinned downloads: the official VOSR 2.0 model and the TensorRT FP8 graph set."""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import requests
from sanic.log import logger

from api import NodeContext

from .vosr2 import BundlePaths
from .vosr2_engine_cache import stored_hash
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
) -> None:
    """Resume a partial file when the server honours ranges; verify, then publish."""
    url = entry["url"]
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
                            f"The server answered HTTP {response.status_code} "
                            f"for a partial {url}."
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


def fetch_files(
    context: NodeContext,
    folder: Path,
    files: dict[str, dict],
    file_digest: Callable[[Path], str],
    label: str,
    reserve: int = 0,
) -> None:
    """Download each missing or mismatched file of files ({name: size, sha256, url})."""
    pending: dict[str, dict] = {}
    for name, entry in files.items():
        target = folder / name
        if target.is_file():
            if (
                target.stat().st_size == entry["size"]
                and file_digest(target) == entry["sha256"]
            ):
                continue
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
    logger.info("Downloading %s once: %.2f GB into %s", label, total / 1e9, folder)
    progress = [done]

    def advance(count: int) -> None:
        progress[0] += count
        context.set_progress(progress[0] / total)

    for name, entry in pending.items():
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        download(context, target, entry, advance)


def model_folder(storage_dir: Path) -> Path:
    return storage_dir / "vosr2-model" / REVISION


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
        "the VOSR 2.0 model",
    )
    return BundlePaths(folder)
