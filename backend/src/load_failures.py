"""The node modules that failed to load for a reason other than a missing dependency,
kept from one start of the worker to the next in the storage folder: a module that
fails the same way at two starts in a row is taken to fail at every start, and its
failure is settled for the node list as a missing dependency's is (NodeLoading)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from sanic.log import logger

import api

FILE = "node-load-failures.json"


def failure(error: Exception) -> str:
    """How a module failed, as it is compared from one start to the next."""
    return f"{type(error).__name__}: {error}"


def failures_of(errors: list[api.LoadErrorInfo]) -> dict[str, str]:
    """The failures other than a missing dependency, by module."""
    return {
        e.module: failure(e.error)
        for e in errors
        if not isinstance(e.error, ModuleNotFoundError)
    }


def read(storage_dir: Path) -> dict[str, str]:
    """The failures the previous start recorded; none when it recorded none, or when the
    file cannot be read (logged)."""
    path = storage_dir / FILE
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        logger.warning(f"Unable to read {path}", exc_info=True)
        return {}
    if not isinstance(recorded, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in recorded.items()
    ):
        logger.warning(f"Ignoring {path}: not a record of failures")
        return {}
    return recorded


def record(storage_dir: Path, failures: dict[str, str]) -> None:
    """Keeps this start's failures for the next one (none: the file is removed). A
    record that cannot be written is logged."""
    path = storage_dir / FILE
    try:
        if not failures:
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temp.write_text(
            json.dumps(failures, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temp, path)
    except OSError:
        logger.warning(
            f"Unable to record the node modules that failed in {path}", exc_info=True
        )
