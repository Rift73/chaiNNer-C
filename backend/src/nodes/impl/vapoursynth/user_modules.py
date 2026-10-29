"""The VapourSynth user modules: the pip packages `<storage>\\vapoursynth\\user_modules.toml`
lists (`modules = ["vs_colorfix", "vs-denoise-extra>=2"]`, each with a version spec if
it has one). The host installs the missing ones when chaiNNer-C starts, with the worker
stopped, before the worker loads its nodes: an install can take minutes, and the node
import, which the app waits for, scans them only (catalog.ensure_catalog). The host
imports this module: nothing here imports VapourSynth, the worker or numpy."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from sanic.log import logger

from custom_types import UpdateProgressFn
from dependencies.store import ENV as PIP_ENV
from dependencies.store import (
    PIP_REASON_LINES,
    distribution_name,
    installed_packages,
    scan_installed_packages,
)

from .scan import user_module_names

MODULES_FILE = "user_modules.toml"
# The longest a user module's pip install may take at startup (s).
PIP_TIMEOUT = 900
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def pip_command() -> list[str]:
    """chaiNNer's own pip, as the dependency manager runs it."""
    return [sys.executable, "-m", "chainner_pip"]


def requirement_met(requirement: str) -> bool:
    """Whether pip has nothing to install for a requirement: its project is installed
    at a version its spec allows (pre-releases included). Raises ValueError for text
    that is not a requirement."""
    # pip's own reading of requirements, as pip installs them.
    from pip._vendor.packaging.requirements import Requirement

    parsed = Requirement(requirement)
    if parsed.marker is not None and not parsed.marker.evaluate():
        # For another system ("; sys_platform == 'linux'"): pip installs nothing.
        return True
    version = installed_packages.get(distribution_name(parsed.name))
    return version is not None and parsed.specifier.contains(version, prereleases=True)


def missing_user_modules(storage_dir: str | None) -> list[str]:
    """The requirements of `<storage_dir>\\vapoursynth\\user_modules.toml` (the folder
    the worker names in CHAINNER_VS_DIR) that pip has to install. None without a
    storage folder (a start without --storage-dir), or while VapourSynth is not
    installed (they follow its install)."""
    if (
        storage_dir is None
        or distribution_name("VapourSynth") not in installed_packages
    ):
        return []
    try:
        names = user_module_names(Path(storage_dir) / "vapoursynth" / MODULES_FILE)
    except ValueError as error:
        logger.warning("VapourSynth user modules: %s", error)
        return []
    missing: list[str] = []
    for name in names:
        try:
            if not requirement_met(name):
                missing.append(name)
        except ValueError as error:
            logger.warning(
                "Skipping the VapourSynth user module %r of %s: %s",
                name,
                MODULES_FILE,
                error,
            )
    return missing


def install_user_module(name: str) -> None:
    """Installs one requirement with pip, as listed. A failed install, or one that takes
    longer than PIP_TIMEOUT, is logged and skips that module only (the scan lists it
    under errors); it is tried again at the next start."""
    logger.info("Installing the VapourSynth user module %s with pip.", name)
    try:
        done = subprocess.run(
            [
                *pip_command(),
                "install",
                name,
                "--disable-chainner_pip-version-check",
                "--no-warn-script-location",
                "--no-cache-dir",
            ],
            env=PIP_ENV,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=PIP_TIMEOUT,
            check=False,
            creationflags=_NO_WINDOW,
        )
    except OSError as error:
        logger.warning("Could not run pip to install %s: %s", name, error)
        return
    except subprocess.TimeoutExpired:
        logger.warning(
            "Installing the VapourSynth user module %s took longer than %d s;"
            " pip was stopped and its functions are skipped.",
            name,
            PIP_TIMEOUT,
        )
        return
    if done.returncode == 0:
        logger.info("Installed the VapourSynth user module %s.", name)
        return
    lines = (done.stdout + done.stderr).strip().splitlines()
    logger.warning(
        "Could not install the VapourSynth user module %s; its functions are"
        " skipped. pip said:\n%s",
        name,
        "\n".join(lines[-PIP_REASON_LINES:]),
    )


async def install_user_modules(
    names: Sequence[str], update_progress: UpdateProgressFn
) -> None:
    """Installs each requirement in turn (install_user_module), showing which in the
    app's start progress."""
    if not names:
        return
    try:
        for index, name in enumerate(names):
            await update_progress(
                f"Installing the VapourSynth user module {name}...",
                index / len(names),
                None,
            )
            await asyncio.to_thread(install_user_module, name)
    finally:
        # pip may change other packages too (a module's dependencies).
        scan_installed_packages()
