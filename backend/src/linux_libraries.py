"""The system libraries the Python wheels need on Linux (OpenCV's Qt, VapourSynth's plugins,
ROCm's where PyTorch runs on an AMD GPU), from linux_libraries.txt beside this module, the
table the launcher (chaiNNer.sh) reads for Electron's libraries too. A minimal install lacks
some, and a wheel then fails to import with a loader error that names one library; this names
them all, with the command that installs them."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

from sanic.log import logger

from system import glibc_version

TABLE = Path(__file__).with_name("linux_libraries.txt")
# The columns of the table after the groups, by package manager, with the command that installs
INSTALL_COMMANDS = {
    "apt": "apt-get install",
    "dnf": "dnf install",
    "pacman": "pacman -S --needed",
    "zypper": "zypper install",
}
# The distribution ids in os-release (ID, then ID_LIKE) by the package manager of their family
FAMILIES = {
    "debian": "apt",
    "ubuntu": "apt",
    "rhel": "dnf",
    "fedora": "dnf",
    "centos": "dnf",
    "rocky": "dnf",
    "almalinux": "dnf",
    "arch": "pacman",
    "suse": "zypper",
    "sles": "zypper",
}
# Where the dynamic loader looks when there is no ldconfig to ask
STANDARD_DIRECTORIES = (
    "/lib",
    "/lib64",
    "/usr/lib",
    "/usr/lib64",
    "/lib/x86_64-linux-gnu",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/local/lib",
)
# Debian 13 and Ubuntu 24.04 (glibc 2.38 and newer) renamed some packages with a t64 suffix
T64_GLIBC = (2, 38)
# The AMD kernel driver's compute interface, which ROCm runs on, and NVIDIA's driver, whose GPU
# PyTorch runs on instead where both are loaded: as the launcher tells the table's amd group.
KFD_TOPOLOGY = Path("/sys/class/kfd/kfd/topology/nodes")
NVIDIA_DRIVER = Path("/proc/driver/nvidia")


class Entry(NamedTuple):
    file: str
    groups: frozenset[str]
    # By package manager; an apt name ends with "(t64)" where the suffix applies
    packages: dict[str, str]


def read_table(text: str) -> list[Entry]:
    entries: list[Entry] = []
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        entries.append(
            Entry(
                fields[0],
                frozenset(fields[1].split(",")),
                dict(zip(INSTALL_COMMANDS, fields[2:])),
            )
        )
    return entries


def package_manager(release: dict[str, str]) -> str | None:
    """The package manager of the distribution family that os-release (ID, ID_LIKE) names."""
    for word in f"{release.get('ID', '')} {release.get('ID_LIKE', '')}".split():
        if word in FAMILIES:
            return FAMILIES[word]
    return None


def cached_libraries(ldconfig_output: str) -> set[str]:
    """The 64-bit libraries that `ldconfig -p` lists."""
    names: set[str] = set()
    for line in ldconfig_output.splitlines():
        name, _, rest = line.strip().partition(" (")
        if rest.startswith("libc6,x86-64"):
            names.add(name)
    return names


def read_library_cache() -> set[str] | None:
    """The libraries in ldconfig's cache; None where there is no ldconfig (a non-root PATH of
    Debian lacks /sbin, so its path is tried too)."""
    for tool in ("ldconfig", "/sbin/ldconfig", "/usr/sbin/ldconfig"):
        try:
            result = subprocess.run(
                [tool, "-p"], capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        return cached_libraries(result.stdout)
    return None


def is_present(library: str, cache: set[str] | None) -> bool:
    """Whether the dynamic loader finds the library: in ldconfig's cache, in a directory of
    LD_LIBRARY_PATH, or, where there is no ldconfig, in one of the usual directories."""
    if cache is not None and library in cache:
        return True
    directories = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if cache is None:
        directories.extend(STANDARD_DIRECTORIES)
    return any(d and (Path(d) / library).exists() for d in directories)


def install_command(
    missing: Iterable[Entry],
    manager: str,
    glibc: tuple[int, int] | None,
    is_root: bool,
) -> str:
    suffix = "t64" if glibc is not None and glibc >= T64_GLIBC else ""
    packages: list[str] = []
    for entry in missing:
        package = entry.packages[manager]
        if package.endswith("(t64)"):
            package = package.removesuffix("(t64)") + suffix
        if package not in packages:
            packages.append(package)
    command = INSTALL_COMMANDS[manager]
    return " ".join([command if is_root else f"sudo {command}", *packages])


def wheel_groups() -> frozenset[str]:
    """The table's groups the Python wheels need on this system: base and vapoursynth, and
    amd where PyTorch runs on ROCm (an AMD GPU's compute driver, and no NVIDIA driver)."""
    if KFD_TOPOLOGY.is_dir() and not NVIDIA_DRIVER.exists():
        return frozenset({"base", "vapoursynth", "amd"})
    return frozenset({"base", "vapoursynth"})


def log_missing_libraries() -> None:
    """Logs the libraries of the Python wheels (the table's base and vapoursynth groups, and
    amd where it applies) that this system lacks, and the command that installs them;
    elsewhere than on Linux, nothing. OpenCV, VapourSynth (or some of its plugins) and ROCm's
    PyTorch fail to import without them."""
    if sys.platform != "linux":
        return
    cache = read_library_cache()
    groups = wheel_groups()
    missing = [
        entry
        for entry in read_table(TABLE.read_text(encoding="utf-8"))
        if entry.groups & groups and not is_present(entry.file, cache)
    ]
    if not missing:
        return
    try:
        manager = package_manager(platform.freedesktop_os_release())
    except OSError as error:
        logger.warning(f"Could not read os-release: {error}")
        manager = None
    if manager is None:
        how = "Install the packages of this distribution that provide them."
    else:
        command = install_command(missing, manager, glibc_version(), os.geteuid() == 0)
        how = f"Install them with: {command}"
    logger.error(
        "This system lacks libraries that OpenCV, VapourSynth or ROCm need: "
        f"{', '.join(entry.file for entry in missing)}. {how}"
    )
