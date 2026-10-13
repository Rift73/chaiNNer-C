"""Fetch the official, matched VOSR 2.0 bundle without a hub runtime dependency."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def verify(path: Path, size: int, digest: str) -> None:
    if path.stat().st_size != size:
        raise ValueError(f"Unexpected size: {path}")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != digest:
        raise ValueError(f"SHA-256 mismatch: {path}")


def fetch(root: Path, revision: str, asset: tuple[str, str, int, str]) -> None:
    remote, local, size, digest = asset
    destination = root / local
    if destination.exists():
        verify(destination, size, digest)
        print(f"Verified existing {local}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=destination.name + ".",
            suffix=".part",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            print(f"Downloading {local} ({size:,} bytes)", flush=True)
            url = f"https://huggingface.co/CSWRY/VOSR/resolve/{revision}/{remote}"
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                for chunk in response.iter_content(4 * 1024 * 1024):
                    output.write(chunk)
        verify(temporary, size, digest)
        # Do not overwrite a concurrent writer's output; verify it instead.
        if destination.exists():
            verify(destination, size, digest)
        else:
            os.rename(temporary, destination)
        print(f"Verified {local}", flush=True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    # The pins live in the app, whose Load VOSR Model checks the sizes.
    sys.path.append(str(Path(__file__).resolve().parents[2] / "backend/src"))
    from nodes.impl.pytorch.vosr2_official import FILES, REVISION

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda asset: fetch(args.destination, REVISION, asset), FILES))
    print(f"VOSR 2.0 bundle ready: {args.destination.resolve()}")


if __name__ == "__main__":
    main()
