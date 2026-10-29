"""Prints the GPUs a HIP runtime numbers, in HIP's order, as JSON: a list of
{"name", "gfx", "luid", "total", "compute_units", "integrated"}. On Windows chaiNNer
runs it in a child process (listing), so the runtime it loads (the ROCm SDK's or the
AMD driver's) never loads into chaiNNer's own processes, where PyTorch loads its own HIP
runtime of the same file name, and a failing driver cannot take them down. Exits with 1
and a message when HIP fails.

The listing is made once: the process that makes it keeps it, or its error, in its
environment (ENVIRONMENT), which the processes it starts inherit and read instead of
listing the GPUs again. The host makes it before it starts the worker (run.py), where
NVIDIA's driver is not installed. Standard library only: run.py uses it before the
host's dependencies load.

usage: python -I -S hip_devices.py HIP_RUNTIME_DLL
"""

from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# The environment variable that hands a listing (JSON: the devices, or {"error": ...})
# to the processes started after it was made
ENVIRONMENT = "CHAINNER_C_AMD_GPUS"
# Seconds the child may take: a working driver lists its GPUs in well under a second
# (0.08 s here, without an AMD GPU); a hung one costs this once.
TIMEOUT = 10

# hipDeviceProp_tR0600 (hip_runtime_api.h; one layout from HIP 6.0 to 7.x): the offsets
# of the fields read and the struct's size, as compiled for 64-bit Windows
_NAME = (0, 256)
_LUID = (272, 8)
_TOTAL_GLOBAL_MEM = (288, 8)
_MULTI_PROCESSOR_COUNT = (388, 4)
_INTEGRATED = (396, 4)
_GCN_ARCH_NAME = (1160, 256)
_PROPERTIES_SIZE = 1472

_HIP_SUCCESS = 0
_HIP_ERROR_NO_DEVICE = 100


def _text(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", "replace")


def _field(properties: bytes, field: tuple[int, int]) -> bytes:
    offset, size = field
    return properties[offset : offset + size]


def _compute_units(processors: int, gfx: str) -> int:
    """HIP's multiProcessorCount in compute units: from gfx10 to before gfx12.5, HIP
    counts workgroup processors (two compute units) while its WGP mode is on
    (GPU_ENABLE_WGP_MODE, on unless set to 0 or false)."""
    match = re.fullmatch(r"gfx(\d+)([0-9a-f])[0-9a-f]", gfx)
    if match is None:
        return processors
    version = (int(match[1]), int(match[2], 16))
    wgp = os.environ.get("GPU_ENABLE_WGP_MODE", "1").lower() not in ("0", "false")
    return processors * 2 if wgp and (10, 0) <= version < (12, 5) else processors


def list_devices(runtime: str) -> list[dict[str, str | int]]:
    hip = ctypes.CDLL(runtime)
    count = ctypes.c_int()
    status = hip.hipGetDeviceCount(ctypes.pointer(count))
    if status == _HIP_ERROR_NO_DEVICE:
        return []
    if status != _HIP_SUCCESS:
        raise RuntimeError(f"hipGetDeviceCount failed with HIP error {status}.")
    devices: list[dict[str, str | int]] = []
    for index in range(count.value):
        # Room past the struct, should a runtime write more than this layout
        buffer = ctypes.create_string_buffer(_PROPERTIES_SIZE * 2)
        status = hip.hipGetDevicePropertiesR0600(buffer, index)
        if status != _HIP_SUCCESS:
            raise RuntimeError(
                f"hipGetDevicePropertiesR0600({index}) failed with HIP error {status}."
            )
        properties = buffer.raw
        # "gfx90a:sramecc+:xnack-" names the processor and its features
        gfx = _text(_field(properties, _GCN_ARCH_NAME)).split(":", 1)[0]
        processors = int.from_bytes(
            _field(properties, _MULTI_PROCESSOR_COUNT), "little"
        )
        devices.append(
            {
                "name": _text(_field(properties, _NAME)),
                "gfx": gfx,
                "luid": int.from_bytes(_field(properties, _LUID), "little"),
                "total": int.from_bytes(
                    _field(properties, _TOTAL_GLOBAL_MEM), "little"
                ),
                "compute_units": _compute_units(processors, gfx),
                "integrated": _field(properties, _INTEGRATED) != bytes(4),
            }
        )
    return devices


def _newest_runtime(folder: Path) -> Path | None:
    """folder's newest HIP runtime from HIP 6 on (amdhip64_<major>.dll)."""
    runtimes: dict[int, Path] = {}
    for path in folder.glob("amdhip64_*.dll"):
        match = re.fullmatch(r"amdhip64_(\d+)\.dll", path.name, re.IGNORECASE)
        if match and int(match[1]) >= 6:
            runtimes[int(match[1])] = path
    return runtimes[max(runtimes)] if runtimes else None


def runtime() -> Path | None:
    """The HIP runtime that numbers Windows' AMD GPUs: the ROCm SDK's, which PyTorch
    loads, when it is installed, else the newest one the AMD driver puts in System32."""
    spec = importlib.util.find_spec("_rocm_sdk_core")
    for folder in (spec.submodule_search_locations or []) if spec else []:
        found = _newest_runtime(Path(folder, "bin"))
        if found is not None:
            return found
    return _newest_runtime(
        Path(os.environ.get("SystemRoot", "C:\\Windows"), "System32")
    )


def nvidia_driver() -> bool:
    """Whether NVIDIA's driver is installed (its CUDA driver in System32)."""
    system32 = Path(os.environ.get("SystemRoot", "C:\\Windows"), "System32")
    return (system32 / "nvcuda.dll").is_file()


def _probe() -> str:
    """The listing, as the JSON that ENVIRONMENT holds, made by this module in a child
    process with the runtime found (none: no GPU)."""
    found = runtime()
    if found is None:
        return "[]"
    result = subprocess.run(
        [sys.executable, "-I", "-S", __file__, str(found)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=TIMEOUT,
        creationflags=subprocess.CREATE_NO_WINDOW,
        check=False,
    )
    if result.returncode != 0:
        return json.dumps(
            {
                "error": f"Listing the GPUs with {found} failed (exit status"
                f" {result.returncode}): {result.stderr.strip()}"
            }
        )
    return result.stdout.strip()


def record() -> None:
    """Make the listing, unless this process has one, and keep it in ENVIRONMENT. A
    child that outlives TIMEOUT is recorded as that error."""
    if ENVIRONMENT in os.environ:
        return
    try:
        value = _probe()
    except subprocess.TimeoutExpired as error:
        value = json.dumps({"error": f"Listing the GPUs timed out: {error}"})
    os.environ[ENVIRONMENT] = value


def listing() -> list[dict[str, Any]]:
    """The GPUs a HIP runtime numbers (as list_devices gives them), from this process's
    listing, made if it has none (record). Raises OSError with the listing's error."""
    record()
    value = json.loads(os.environ[ENVIRONMENT])
    if isinstance(value, dict):
        raise OSError(value["error"])
    return value


def main() -> int:
    try:
        devices = list_devices(sys.argv[1])
    except (OSError, RuntimeError) as error:
        print(f"{sys.argv[1]}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(devices))
    return 0


if __name__ == "__main__":
    sys.exit(main())
