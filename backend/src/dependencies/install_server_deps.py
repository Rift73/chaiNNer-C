from __future__ import annotations

import subprocess
from json import loads as json_parse

from .store import (
    ENV,
    DependencyInfo,
    install_dependencies_sync,
    installed_packages,
    python_path,
)

# Get the list of installed packages
# We can't rely on using the package's __version__ attribute because not all packages actually have it
try:
    pip_list = subprocess.check_output(
        [
            python_path,
            "-m",
            "pip",
            "list",
            "--format=json",
            "--disable-pip-version-check",
        ],
        env=ENV,
    )
    for p in json_parse(pip_list):
        installed_packages[p["name"]] = p["version"]
except Exception as e:
    print(f"Failed to get installed packages: {e}")


deps: list[DependencyInfo] = [
    DependencyInfo(
        package_name="sanic",
        display_name="Sanic",
        version="25.12.1",
    ),
    # Sanic's downstream deps that are py3-non-any
    DependencyInfo(
        package_name="aiofiles",
        version="25.1.0",
    ),
    DependencyInfo(
        package_name="html5tagger",
        version="2.0.0",
    ),
    DependencyInfo(
        package_name="sanic-routing",
        version="23.12.0",
    ),
    DependencyInfo(
        package_name="tracerite",
        version="2.6.5",
    ),
    # Sanic's downstream deps that we want to pin anyway
    DependencyInfo(
        package_name="websockets",
        version="17.2",
    ),
    # Other deps necessary for general use
    DependencyInfo(
        package_name="typing_extensions",
        version="4.16.0",
    ),
    # Provides the pynvml module gpu.py imports; the pynvml distribution is now a
    # deprecated shim whose import hook warns on every import.
    DependencyInfo(
        package_name="nvidia-ml-py",
        version="13.615.71",
    ),
    DependencyInfo(
        package_name="chainner-pip",
        version="23.2.0",
        from_file="chainner_pip-23.2.0-py3-none-any.whl",
    ),
    DependencyInfo(
        package_name="psutil",
        version="7.2.2",
    ),
    DependencyInfo(
        package_name="aiohttp",
        version="3.14.4",
    ),
]

install_dependencies_sync(deps)
