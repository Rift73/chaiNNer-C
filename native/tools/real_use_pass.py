"""Run U3-3's real-use CPU pass against a package (Consult 6 P3).

Launch. The package's chaiNNer.exe starts as a child this tool owns: created
suspended in a kill-on-close job, so every process it starts is in the job, and
driven by no input. TEMP and TMP point into the run (the app's quit handler
deletes every chaiNNer-* folder in its temp folder), and PIP_REQUIRE_VIRTUALENV
makes an install attempt fail instead of writing into the package. The app does
not inherit the tool's PYTHONDONTWRITEBYTECODE: a user's launch writes bytecode.
%APPDATA%\\chaiNNer is hashed before and after (read only; the portable package
must not touch it), and the package tree is diffed against the ruled allow-list
(launch_change). The package's logs must show the version line, the integrated
Python accepted, the host started and a dependency check that installs nothing.
The app is closed as Process.CloseMainWindow() closes it: a WM_CLOSE posted with
ctypes (psutil has no window API) to the first visible unowned top-level window
of the started process. After 30 s every PID still running that we own (the
job's members, and any descendant outside it) is terminated by PID, never by
name; none may survive.

Dependency manager. The package's host (run.py on the package runtime, started
by bench_backend.Backend as the bench starts it) must report every pin the
dependency manager shows satisfied, and its start must install nothing, which
covers chaiNNer_standard's pins (/packages hides that package).

Host pass. CHAINS run on that host and on the oracle (a per-run copy of
make_oracle.py's tree on the provisioned runtime, behind
verify_runtime.oracle_source's guard), each converted as the UI converts a saved
chain, with every Save node writing into the run. Save outputs are compared by
decoded pixels and verify_runtime.sse_contract must be identical. The longest
chain is then stopped with /kill mid-run, the UI-state invariant is read from
the SSE stream, and one /run of it must complete. Pixel differences of a chain
that runs PyTorch on the CPU are counted, not gated. The package is hashed again
after the host pass, and that diff meets the launch's allow-list too.

Everything is written below native/reports/real-use-<UTC>; every input is only
read. The exit code is 0 only if every gate holds.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from ctypes import wintypes
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import bench_fixtures
import independent_ui
import package_files
import package_manifest
import psutil
from bench_backend import Backend, BenchEvents, job_accounting
from bench_oracle import FFMPEG, FFPROBE
from verify_framework_runtime import OPTIONS
from verify_runtime import (
    ORACLE,
    PROJECT,
    OwnedJob,
    canonical,
    check_success,
    compare_images,
    oracle_source,
    request,
    sse_contract,
    write_json,
)
from verify_video_runtime import job_processes

Json = dict[str, Any]
MODELS = Path(r"C:\Executables\models\ESRGAN")


@dataclass(frozen=True)
class Chain:
    """An owner chain of the pass, read in place.

    torch_cpu: it runs PyTorch on the CPU, so its pixel differences are counted,
    not gated. fallback: a saved image path missing on this machine takes a
    bench image instead of skipping the chain.
    """

    name: str
    path: Path
    torch_cpu: bool = False
    fallback: bool = False


CHAINS = (
    Chain("resize", MODELS / "resize.chn"),
    Chain("Blend", MODELS / "Chain/Blend.chn"),
    Chain("Getnative", MODELS / "Chain/Getnative.chn"),
    Chain("default", MODELS / "default.chn", torch_cpu=True, fallback=True),
)
# The node schemas reviewed for the pass: of these, only Save writes a file, and
# the driver sets its directory. A chain holding any other node is refused.
REVIEWED = frozenset(
    {
        "chainner:image:load",
        "chainner:image:save",
        "chainner:image:resize",
        "chainner:image:blend",
        "chainner:image:opacity",
        "chainner:image:split_transparency",
        "chainner:utility:text_append",
        "chainner:pytorch:load_model",
        "chainner:pytorch:upscale_image",
    }
)
SAVE = "chainner:image:save"
SAVE_DIRECTORY = 1
SAVE_SUBDIRECTORY = 2
LOAD_IMAGE = "chainner:image:load"
LOAD_IMAGE_PATH = 0
# Upstream's save-file migrations (src/common/migrations.ts): the installed app
# has 45, and a save records how many it had. The driver applies none, so a
# chain that a pending one would change is refused; each entry is that test.
CURRENT_MIGRATION = 45
PENDING_MIGRATIONS: dict[int, tuple[str, Callable[[list[Json]], bool]]] = {
    43: (
        "newIteratorToGenerator",
        lambda nodes: any(node.get("type") == "newIterator" for node in nodes),
    ),
    44: (
        "splitLoadImagePairs",
        lambda nodes: any(
            node["data"]["schemaId"] == "chainner:image:load_image_pairs"
            for node in nodes
        ),
    ),
}

# Launch.
LAUNCH_TIMEOUT = 300.0  # until the host is ready and the main window shows
SETTLE = 5.0  # the renderer's first requests, before the close
CLOSE_TIMEOUT = 30.0
TERMINATE_TIMEOUT = 15.0
HASH_THREADS = 16
DIRECTORY = "<dir>"
CREATE_SUSPENDED = 0x00000004
WM_CLOSE = 0x0010
GW_OWNER = 4
WINDOW_VISITOR = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
CLOSE_METHOD = (
    "WM_CLOSE posted with ctypes (user32.PostMessageW; psutil has no window API) "
    "to the first visible unowned top-level window of the started process, as "
    "Process.CloseMainWindow() does"
)
# The tool's own settings, recorded, and the ones the launch sets. The app
# inherits CUDA_VISIBLE_DEVICES and VK_LOADER_DRIVERS_DISABLE (U3-3's launch is a
# CPU launch by design; U4's runs without them) but not PYTHONDONTWRITEBYTECODE:
# a user's launch writes bytecode, and the pass is a user's launch (Consult 8 R-i).
TOOL_ENVIRONMENT = (
    "PYTHONDONTWRITEBYTECODE",
    "CUDA_VISIBLE_DEVICES",
    "VK_LOADER_DRIVERS_DISABLE",
)
WITHHELD = frozenset({"PYTHONDONTWRITEBYTECODE"})
LAUNCH_ENVIRONMENT = ("TEMP", "TMP", "PIP_REQUIRE_VIRTUALENV", *TOOL_ENVIRONMENT)
# What a launch may change at the package ROOT, as tree_state keys (a directory
# ends in "/" and covers what is below it). Frozen: an addition is a consult with
# its evidence. The same names below the root are not admitted.
# - logs/, backend-storage/ and settings.json: Consult 6 P2 and P3.
# - Chromium's profile (Consult 8 R-i): in portable mode the packaged main.js sets
#   userData to the package directory (resources/app/.vite/build/main.js:
#   app.setPath("userData",zs())), and out\chaiNNer-C-py311's root held exactly
#   these 11 entries after its launches (read 2026-10-06).
RULED_ROOT = ("logs/", "backend-storage/", "settings.json")
CHROMIUM_PROFILE = (
    "Cache/",
    "Code Cache/",
    "DawnCache/",
    "GPUCache/",
    "Local Storage/",
    "Session Storage/",
    "Network/",
    "blob_storage/",
    "Local State",
    "Preferences",
    "electron-log-preload.js",
)
ROOT_ENTRIES = frozenset(entry.casefold() for entry in RULED_ROOT + CHROMIUM_PROFILE)
# Under the backend the only admissible change is bytecode: a __pycache__
# directory or a .pyc file in one (Consult 8 R-i); the backend is not precompiled,
# so a launch writes its timestamp .pyc. The runtime's Lib, stdlib and
# site-packages, ships compiled (package_port.BYTECODE, Consult 10 D-9), so any
# bytecode write there is a module the build missed, and it fails, named
# (bytecode_module). An install writes .dist-info, .py, .pyd or .dll anyway.
CACHE = "__pycache__"
BYTECODE_ROOTS = (package_manifest.BACKEND.casefold(),)
RUNTIME_LIB = package_manifest.RUNTIME + "Lib/"
SITE_PACKAGES = "site-packages/"
# The package's log lines (the patched main bundle and the host).
LOG_CHECKS = ("version_line", "integrated_python", "host_start", "dependency_check")
VERSION_LINE = independent_ui.VERSION_LOG_LINE
PYTHON_FOUND = "Final Python binary: "
PYTHON_DOWNLOAD = (
    "Integrated Python not found",
    "Downloading integrated Python",
    "Extracting integrated Python",
)
HOST_SPAWN = "Attempting to spawn backend..."
HOST_ALMOST = "Backend almost ready..."
HOST_DONE = re.compile(r"\bDone\.$")
HOST_EXITED = "Python subprocess exited with code"
DEPENDENCY_CHECK = (
    "Checking dependencies...",
    "No dependencies to install. Skipping worker restart.",
    "Done checking dependencies...",
)
INSTALLING = (
    "Error installing dependencies",
    "Collecting ",
    "Installing collected packages",
    "Successfully installed",
    "Progress: ",
)
# semver.coerce, as the UI (common/version.ts) and the host
# (dependencies/store.py) read a version: its first one to three numbers.
VERSION_NUMBERS = re.compile(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?")

# Host pass.
RUN_TIMEOUT = 4 * 3600.0  # one /run; PyTorch on the CPU is slow
QUIET = 3.0  # after a stopped run, for events that arrive late
# The SSE events that put the UI into a running state (ExecutionContext): a node
# runs only from its node-start, and a chain-start marks the chain's nodes to run.
RUNNING_EVENTS = frozenset({"chain-start", "node-start"})
# Backend.info entries that show a host left nothing behind and changed nothing.
HOST_CLEAN = ("owned_process_exited", "backend_unchanged", "interpreter_unchanged")


def failure(error: BaseException) -> Json:
    """An error as a record, its traceback printed in full."""
    trace = traceback.format_exc()
    print(trace, file=sys.stderr, flush=True)
    return {"error": repr(error), "traceback": trace}


def attempt(function: Callable[..., Json], *arguments: Any) -> Json:
    """function's record, or its failure: one step's error never stops the pass."""
    try:
        return function(*arguments)
    except Exception as error:
        return failure(error)


def wait_until(condition: Callable[[], bool], timeout: float) -> bool:
    limit = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= limit:
            return False
        time.sleep(0.25)
    return True


# Trees.


def tree_state(root: Path) -> dict[str, str]:
    """Every file below root by SHA-256 and every directory as DIRECTORY.

    Empty for a missing root. Only reads, on HASH_THREADS threads.
    """
    if not root.exists():
        return {}
    entries = package_files.walk(root, directories=True)
    files = [entry for entry in entries if not entry.endswith("/")]
    with ThreadPoolExecutor(HASH_THREADS) as pool:
        hashes = dict(
            zip(
                files,
                pool.map(lambda name: package_files.hash_file(root / name), files),
                strict=True,
            )
        )
    return {entry: hashes.get(entry, DIRECTORY) for entry in entries}


def tree_diff(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
    return {
        "added": sorted(after.keys() - before.keys()),
        "removed": sorted(before.keys() - after.keys()),
        "modified": sorted(
            key for key in before.keys() & after.keys() if before[key] != after[key]
        ),
    }


def launch_change(path: str) -> str:
    """How the ruled allow-list takes a change to a package path.

    "bytecode" (under BYTECODE_ROOTS, a __pycache__ directory or a .pyc file in
    one), "allowed" (a ROOT_ENTRIES entry or below a root directory entry) or
    "disallowed". path is a tree_state key: POSIX, relative to the package, a
    directory ending in "/".
    """
    folded = path.casefold()
    parts = PurePosixPath(folded).parts
    if folded.startswith(BYTECODE_ROOTS):
        if folded.endswith("/"):
            cached = parts[-1] == CACHE
        else:
            cached = len(parts) > 1 and parts[-2] == CACHE and folded.endswith(".pyc")
        return "bytecode" if cached else "disallowed"
    first, slash, _ = folded.partition("/")
    return "allowed" if first + slash in ROOT_ENTRIES else "disallowed"


def bytecode_module(path: str) -> str | None:
    """The module whose .pyc path is, in the runtime's Lib (stdlib or
    site-packages), as Python names it; None for any other path."""
    if not path.startswith(RUNTIME_LIB) or not path.endswith(".pyc"):
        return None
    parts = PurePosixPath(path.removeprefix(RUNTIME_LIB).removeprefix(SITE_PACKAGES))
    if len(parts.parts) < 2 or parts.parts[-2] != CACHE:
        return None
    name = parts.name.split(".")[0]
    package = list(parts.parts[:-2])
    return ".".join(package if name == "__init__" else [*package, name])


def launch_tree(diff: dict[str, list[str]]) -> Json:
    """A package diff (the launch's, or the host pass's) sorted by launch_change,
    and its verdict. A bytecode write into the runtime's Lib names its module."""
    classes: dict[str, list[str]] = {
        "allowed": [],
        "bytecode": [],
        "disallowed": [],
    }
    for kind, paths in diff.items():
        for path in paths:
            module = bytecode_module(path)
            missed = f" (module {module}, not compiled by the build)" if module else ""
            classes[launch_change(path)].append(f"{kind}: {path}{missed}")
    return {**classes, "pass": not classes["disallowed"]}


# Logs.


def log_offsets(logs: Path) -> dict[str, int]:
    if not logs.is_dir():
        return {}
    return {
        path.relative_to(logs).as_posix(): path.stat().st_size
        for path in logs.rglob("*.log")
    }


def new_log_text(logs: Path, offsets: dict[str, int]) -> str:
    """What the package's logs gained since offsets; a shrunk (rotated) log is
    read whole."""
    if not logs.is_dir():
        return ""
    parts = []
    for path in sorted(logs.rglob("*.log")):
        data = path.read_bytes()
        start = offsets.get(path.relative_to(logs).as_posix(), 0)
        if len(data) < start:
            start = 0
        parts.append(data[start:].decode("utf-8", "replace"))
    return "\n".join(parts)


def host_ready(lines: list[str]) -> bool:
    """The host finished its setup: a "Done." after "Backend almost ready..."."""
    almost = next((i for i, line in enumerate(lines) if HOST_ALMOST in line), None)
    return almost is not None and any(
        HOST_DONE.search(line.rstrip()) for line in lines[almost + 1 :]
    )


def install_check(lines: list[str]) -> Json:
    """The host's dependency check ran to its end and installed nothing."""
    missing = [
        marker for marker in DEPENDENCY_CHECK if not any(marker in x for x in lines)
    ]
    installing = [line for line in lines if any(m in line for m in INSTALLING)]
    return {
        "missing_lines": missing,
        "install_lines": installing[:20],
        "pass": not missing and not installing,
    }


def log_checks(text: str, before_close: str, python: Path, interpreter: str) -> Json:
    """The launch's four log checks (LOG_CHECKS).

    text: everything the launch logged; before_close: what it had logged when
    the close was sent, so the host's own exit at the close does not count.
    python: the package's interpreter; interpreter: its version.
    """
    lines = text.splitlines()
    early = before_close.splitlines()
    found = [
        line.split(PYTHON_FOUND, 1)[1].strip() for line in lines if PYTHON_FOUND in line
    ]
    refused = [
        line
        for line in lines
        if independent_ui.PYTHON_MISSING in line
        or any(marker in line for marker in PYTHON_DOWNLOAD)
    ]
    version = f"version: '{interpreter}'"
    expected = os.path.normcase(str(python))
    spawned = any(HOST_SPAWN in line for line in early)
    ready = host_ready(early)
    exited = [line for line in early if HOST_EXITED in line]
    return {
        "version_line": {
            "expected": VERSION_LINE,
            "pass": any(line.rstrip().endswith(VERSION_LINE) for line in lines),
        },
        "integrated_python": {
            "expected": str(python),
            "found": found,
            "version_logged": version in text,
            "refused_or_download_lines": refused[:20],
            "pass": bool(found)
            and all(os.path.normcase(path) == expected for path in found)
            and version in text
            and not refused,
        },
        "host_start": {
            "spawned": spawned,
            "ready": ready,
            "exited_before_close": exited[:20],
            "pass": spawned and ready and not exited,
        },
        "dependency_check": install_check(lines),
    }


# The app's window and processes.


def user32() -> ctypes.WinDLL:
    """user32 with the signatures of the calls the close makes."""
    library = ctypes.WinDLL("user32", use_last_error=True)
    library.EnumWindows.argtypes = [WINDOW_VISITOR, wintypes.LPARAM]
    library.EnumWindows.restype = wintypes.BOOL
    library.GetWindowThreadProcessId.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    ]
    library.GetWindowThreadProcessId.restype = wintypes.DWORD
    library.IsWindowVisible.argtypes = [wintypes.HWND]
    library.IsWindowVisible.restype = wintypes.BOOL
    library.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
    library.GetWindow.restype = wintypes.HWND
    library.PostMessageW.argtypes = [
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    ]
    library.PostMessageW.restype = wintypes.BOOL
    return library


def main_window(pid: int) -> int | None:
    """The process's main window as Process.MainWindowHandle finds it: the first
    visible top-level window of the process that has no owner window."""
    library = user32()
    found: list[int] = []

    def visit(window: int | None, _parameter: int) -> bool:
        process = wintypes.DWORD()
        library.GetWindowThreadProcessId(window, ctypes.byref(process))
        if (
            window
            and process.value == pid
            and library.IsWindowVisible(window)
            and not library.GetWindow(window, GW_OWNER)
        ):
            found.append(window)
            return False
        return True

    # A visitor that stops the walk makes EnumWindows return FALSE; not an error.
    library.EnumWindows(WINDOW_VISITOR(visit), 0)
    return found[0] if found else None


def active_processes(job: OwnedJob) -> int:
    return job_accounting(job).ActiveProcesses


def running_from(directories: list[Path]) -> tuple[list[str], list[Json]]:
    """Processes whose executable lies below one of directories: the running ones
    ("<pid> <exe>"), and the exited ones, skipped and recorded.

    An exited process keeps its object while another process holds a handle to
    it, with 0 threads: it cannot execute or write, so it is skipped and recorded
    (pid, exe, creation time). Any other one is running, a suspended one with
    threads included, and so is one whose thread count cannot be read.
    """
    roots = tuple(os.path.normcase(str(path)) + os.sep for path in directories)
    running, exited = [], []
    for process in psutil.process_iter(["pid", "exe", "num_threads", "create_time"]):
        info = process.info
        if not info["exe"] or not os.path.normcase(info["exe"]).startswith(roots):
            continue
        if info["num_threads"] == 0:
            created = info["create_time"]
            exited.append(
                {
                    "pid": info["pid"],
                    "exe": info["exe"],
                    "created": None
                    if created is None
                    else datetime.fromtimestamp(created, UTC).isoformat(),
                }
            )
        else:
            running.append(f"{info['pid']} {info['exe']}")
    return running, exited


def close_app(job: OwnedJob, pid: int) -> Json:
    """CloseMainWindow, up to CLOSE_TIMEOUT for the app to end, then every PID we
    own terminated one by one, never by name: the job's members and any
    descendant of the started process outside the job (none should be)."""
    try:
        tree = psutil.Process(pid).children(recursive=True)
    except psutil.NoSuchProcess:
        tree = []  # The app has already exited; the job holds what it left.
    window = main_window(pid)
    record: Json = {
        "method": CLOSE_METHOD,
        "main_window": window is not None,
        "terminated": [],
        "exited_before_termination": [],
    }

    def outside(members: dict[str, str]) -> list[str]:
        return [
            str(p.pid) for p in tree if p.is_running() and str(p.pid) not in members
        ]

    def ended() -> bool:
        return active_processes(job) == 0 and not outside({})

    if window is not None:
        if not user32().PostMessageW(window, WM_CLOSE, 0, 0):
            raise ctypes.WinError(ctypes.get_last_error())
    record["closed_by_window"] = window is not None and wait_until(ended, CLOSE_TIMEOUT)
    if not record["closed_by_window"]:
        members = job_processes(job)
        record["terminated"] = [*members, *outside(members)]
        for owned in record["terminated"]:
            try:
                psutil.Process(int(owned)).kill()
            except psutil.NoSuchProcess:
                # It exited between the listing and the kill.
                record["exited_before_termination"].append(owned)
        wait_until(ended, TERMINATE_TIMEOUT)
    record["survivors"] = job_processes(job)
    record["outside_job"] = outside(record["survivors"])
    record["active_after"] = active_processes(job)
    record["no_survivors"] = (
        not record["survivors"]
        and not record["outside_job"]
        and record["active_after"] == 0
    )
    return record


def wait_started(
    process: subprocess.Popen[bytes], logs: Path, offsets: dict[str, int]
) -> Json:
    """Until the host is ready and the main window shows, then SETTLE more."""
    began = time.monotonic()
    while time.monotonic() - began < LAUNCH_TIMEOUT:
        if process.poll() is not None:
            return {"started": False, "exit_code": process.returncode}
        lines = new_log_text(logs, offsets).splitlines()
        if host_ready(lines) and main_window(process.pid) is not None:
            seconds = round(time.monotonic() - began, 1)
            time.sleep(SETTLE)
            return {"started": True, "seconds": seconds}
        time.sleep(0.5)
    return {"started": False, "timeout": LAUNCH_TIMEOUT}


def launch_environment(temp: Path) -> dict[str, str]:
    """The app's environment: the tool's own but WITHHELD, TEMP and TMP set to
    temp, and PIP_REQUIRE_VIRTUALENV."""
    return {
        **{key: value for key, value in os.environ.items() if key not in WITHHELD},
        "TEMP": str(temp),
        "TMP": str(temp),
        "PIP_REQUIRE_VIRTUALENV": "1",
    }


def launch(package: Path, directory: Path, interpreter: str) -> Json:
    """The launch check (see the module docstring)."""
    temp = directory / "temp"
    temp.mkdir(parents=True)
    appdata = Path(os.environ["APPDATA"], "chaiNNer")
    logs = package / "logs"
    print("Launch: hashing %APPDATA%\\chaiNNer and the package", flush=True)
    appdata_before = tree_state(appdata)
    package_before = tree_state(package)
    offsets = log_offsets(logs)
    env = launch_environment(temp)
    command = [str(package / "chaiNNer.exe")]
    record: Json = {
        "command": command,
        "environment": {name: env.get(name) for name in LAUNCH_ENVIRONMENT},
    }
    before_close = ""
    job = OwnedJob()
    process: subprocess.Popen[bytes] | None = None
    with (directory / "app-output.txt").open("wb") as output:
        try:
            # Suspended until the job holds it, so every process it starts is ours.
            process = subprocess.Popen(
                command,
                cwd=package,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                creationflags=CREATE_SUSPENDED,
            )
            job.assign(process)
            psutil.Process(process.pid).resume()
            record["pid"] = process.pid
            print(f"Launch: chaiNNer.exe started as {process.pid}", flush=True)
            record["start"] = wait_started(process, logs, offsets)
            before_close = new_log_text(logs, offsets)
            record["processes_before_close"] = job_processes(job)
            record["close"] = close_app(job, process.pid)
        finally:
            job.close()  # Kill-on-close: nothing of ours outlives an error above.
            if process is not None:
                record["exit_code"] = process.wait(timeout=TERMINATE_TIMEOUT)
    text = new_log_text(logs, offsets)
    (directory / "log.txt").write_text(text, encoding="utf-8")
    record["log"] = log_checks(
        text, before_close, package / "python/python/python.exe", interpreter
    )
    print("Launch: hashing again", flush=True)
    appdata_diff = tree_diff(appdata_before, tree_state(appdata))
    package_diff = tree_diff(package_before, tree_state(package))
    write_json(directory / "appdata-diff.json", appdata_diff)
    write_json(directory / "package-diff.json", package_diff)
    record["appdata"] = {
        "path": str(appdata),
        "entries": len(appdata_before),
        "changes": {kind: len(paths) for kind, paths in appdata_diff.items()},
        "unchanged": not any(appdata_diff.values()),
    }
    record["package_tree"] = launch_tree(package_diff)
    return record


# Saved chains, converted as the UI converts them.


def parse_handle(handle: str) -> tuple[str, int]:
    """parseSourceHandle / parseTargetHandle: a 36-character node id, "-", an id."""
    return handle[:36], int(handle[37:])


def read_chain(path: Path, schemas: dict[str, Json]) -> Json:
    """The save file's nodes and edges, once the driver can run them as saved.

    Refused: a save with no migration count or a newer one, one that a pending
    migration would change, an unknown or unreviewed node, and a passthrough
    node (the UI's passthrough map needs its type system).
    """
    save = json.loads(path.read_text(encoding="utf-8"))
    migration = save.get("migration")
    if not isinstance(migration, int) or migration > CURRENT_MIGRATION:
        raise ValueError(
            f"{path}: migration {migration!r} is not one this driver reads (the "
            f"installed app has {CURRENT_MIGRATION})"
        )
    nodes = save["content"]["nodes"]
    for index in range(migration, CURRENT_MIGRATION):
        if index not in PENDING_MIGRATIONS:
            raise ValueError(f"{path} needs upstream migration {index}; none applies")
        name, changes = PENDING_MIGRATIONS[index]
        if changes(nodes):
            raise ValueError(f"{path}: upstream migration {name} would change it")
    schema_ids = {node["data"]["schemaId"] for node in nodes}
    if unknown := sorted(schema_ids - schemas.keys()):
        raise ValueError(f"{path}: nodes unknown to the backend: {unknown}")
    if unreviewed := sorted(schema_ids - REVIEWED):
        raise ValueError(f"{path}: nodes not reviewed for the pass: {unreviewed}")
    if passthrough := [n["id"] for n in nodes if n["data"].get("isPassthrough")]:
        raise ValueError(f"{path}: passthrough nodes: {passthrough}")
    return {"nodes": nodes, "edges": save["content"]["edges"]}


def resolve_inputs(
    content: Json,
    schemas: dict[str, Json],
    images: list[dict[str, str]],
    fallback: bool,
) -> tuple[Json, list[Json], list[Json]]:
    """Points the chain's file inputs at files on this machine.

    A connected input stays, and a saved path that exists is read in place. A
    Load Image path the save left unset (a template slot), or one missing here
    when fallback is set, takes the next bench image: images are the bench's
    hashed dataset entries, each checked before use. Any other unset or missing
    file is listed, which skips the chain. Returns the new content, the
    substitutions and the missing inputs.
    """
    content = copy.deepcopy(content)
    connected = {
        parse_handle(edge["targetHandle"])
        for edge in content["edges"]
        if edge.get("targetHandle")
    }
    unused = iter(images)
    substitutions: list[Json] = []
    missing: list[Json] = []
    for node in content["nodes"]:
        schema_id = node["data"]["schemaId"]
        values = node["data"].setdefault("inputData", {})
        for item in schemas[schema_id]["inputs"]:
            if item.get("kind") != "file" or (node["id"], item["id"]) in connected:
                continue
            saved = values.get(str(item["id"]))
            if saved and Path(saved).is_file():
                continue
            entry = {"node": node["id"], "input": item["label"], "saved": saved}
            image_input = (schema_id, item["id"]) == (LOAD_IMAGE, LOAD_IMAGE_PATH)
            if not image_input or (saved and not fallback):
                missing.append(entry)
                continue
            image = next(unused, None)
            if image is None:
                missing.append({**entry, "reason": "no bench image left"})
                continue
            try:
                bench_fixtures.verify_inputs([image])
            except FileNotFoundError as error:
                missing.append({**entry, "reason": str(error)})
                continue
            values[str(item["id"])] = image["path"]
            substitutions.append({**entry, "used": image["path"]})
    return content, substitutions, missing


def confined(subdirectory: str) -> bool:
    """A Save subdirectory that stays inside the directory the driver sets."""
    path = PureWindowsPath(subdirectory)
    return not path.drive and not path.root and ".." not in path.parts


def rewrite_saves(
    content: Json, schemas: dict[str, Json], directory: Path
) -> tuple[list[Json], list[Json]]:
    """Copies of the nodes and edges with every Save writing into directory.

    An edge into a Save's directory is dropped and the value set; a connected
    subdirectory, or one that leaves directory, is refused.
    """
    nodes = copy.deepcopy(content["nodes"])
    saves = {node["id"] for node in nodes if node["data"]["schemaId"] == SAVE}
    if saves:
        kinds = {item["id"]: item.get("kind") for item in schemas[SAVE]["inputs"]}
        if (kinds.get(SAVE_DIRECTORY), kinds.get(SAVE_SUBDIRECTORY)) != (
            "directory",
            "text",
        ):
            raise ValueError(f"Unreviewed {SAVE} inputs: {kinds}")
    edges = []
    for edge in content["edges"]:
        handle = edge.get("targetHandle")
        target = parse_handle(handle) if handle else None
        if target is not None and target[0] in saves:
            if target[1] == SAVE_DIRECTORY:
                continue
            if target[1] == SAVE_SUBDIRECTORY:
                raise ValueError(f"Save {target[0]}'s subdirectory is connected")
        edges.append(copy.deepcopy(edge))
    for node in nodes:
        if node["id"] in saves:
            values = node["data"].setdefault("inputData", {})
            subdirectory = values.get(str(SAVE_SUBDIRECTORY))
            if subdirectory and not confined(subdirectory):
                raise ValueError(
                    f"Save {node['id']}'s subdirectory {subdirectory!r} escapes"
                )
            values[str(SAVE_DIRECTORY)] = str(directory)
    return nodes, edges


def with_defaults(node: Json, schemas: dict[str, Json]) -> Json:
    """setStateFromJSON: each input's default under the saved non-null values."""
    node = copy.deepcopy(node)
    data = node["data"]
    saved = {k: v for k, v in (data.get("inputData") or {}).items() if v is not None}
    inputs = schemas[data["schemaId"]]["inputs"]
    data["inputData"] = {
        **{str(item["id"]): item.get("def") for item in inputs},
        **saved,
    }
    return node


def trim_edges(nodes: list[Json], edges: list[Json]) -> list[Json]:
    ids = {node["id"] for node in nodes}
    return [edge for edge in edges if edge["source"] in ids and edge["target"] in ids]


def enabled_nodes(nodes: list[Json], edges: list[Json]) -> list[Json]:
    """The nodes getEffectivelyDisabledNodes leaves: neither disabled nor fed by
    a disabled node."""
    by_id = {node["id"]: node for node in nodes}
    incoming: dict[str, list[str]] = {}
    for edge in trim_edges(nodes, edges):
        incoming.setdefault(edge["target"], []).append(edge["source"])
    cache: dict[str, bool] = {}

    def disabled(node_id: str) -> bool:
        if node_id not in cache:
            cache[node_id] = bool(by_id[node_id]["data"].get("isDisabled")) or any(
                disabled(source) for source in incoming.get(node_id, [])
            )
        return cache[node_id]

    return [node for node in nodes if not disabled(node["id"])]


def effective_nodes(
    nodes: list[Json], edges: list[Json], schemas: dict[str, Json]
) -> list[Json]:
    """removeSideEffectFreeNodes: the nodes with side effects and those feeding
    one, less a node with side effects that has no edge and needs one."""
    by_id = {node["id"]: node for node in nodes}
    outgoing: dict[str, list[str]] = {}
    for edge in trim_edges(nodes, edges):
        outgoing.setdefault(edge["source"], []).append(edge["target"])
    cache: dict[str, bool] = {}

    def schema(node_id: str) -> Json:
        return schemas[by_id[node_id]["data"]["schemaId"]]

    def effect(node_id: str) -> bool:
        if node_id not in cache:
            cache[node_id] = bool(schema(node_id)["hasSideEffects"]) or any(
                effect(target) for target in outgoing.get(node_id, [])
            )
        return cache[node_id]

    kept = [node for node in nodes if effect(node["id"])]
    connected = {
        end
        for edge in trim_edges(kept, edges)
        for end in (edge["source"], edge["target"])
    }

    def used(node_id: str) -> bool:
        needs_edge = any(
            not item.get("optional") and item.get("def") is None
            for item in schema(node_id)["inputs"]
        )
        return (
            node_id in connected
            or not schema(node_id)["hasSideEffects"]
            or not needs_edge
        )

    return [node for node in kept if used(node["id"])]


def optimize_chain(
    nodes: list[Json], edges: list[Json], schemas: dict[str, Json]
) -> tuple[list[Json], list[Json]]:
    """optimizeChain (common/nodes/optimize.ts) for a chain with no passthrough
    node: up to ten passes, until a pass changes nothing."""
    for _ in range(10):
        size = len(nodes), len(edges)
        nodes = enabled_nodes(nodes, edges)
        nodes = effective_nodes(nodes, edges, schemas)
        edges = trim_edges(nodes, edges)
        if (len(nodes), len(edges)) == size:
            break
    return nodes, edges


def to_backend_json(
    nodes: list[Json], edges: list[Json], schemas: dict[str, Json]
) -> list[Json]:
    """toBackendJson (common/nodes/toBackendJson.ts): each node's inputs in schema
    order, an edge as its source's output index, a value otherwise."""
    by_id = {node["id"]: node for node in nodes}
    handles: dict[tuple[str, int], Json] = {}
    for edge in edges:
        if not edge.get("sourceHandle") or not edge.get("targetHandle"):
            continue
        source, output = parse_handle(edge["sourceHandle"])
        if source not in by_id:
            raise ValueError(f"Invalid handle: no node {source}")
        schema = schemas[by_id[source]["data"]["schemaId"]]
        outputs = [item["id"] for item in schema["outputs"]]
        if output not in outputs:
            raise ValueError(f"Invalid handle: node {source} has no output {output}")
        handles[parse_handle(edge["targetHandle"])] = {
            "type": "edge",
            "id": source,
            "index": outputs.index(output),
        }
    result = []
    for node in nodes:
        data = node["data"]
        if not node.get("type"):
            raise ValueError(f"Node {node['id']} has no node type")
        inputs = [
            handles.get(
                (node["id"], item["id"]),
                {"type": "value", "value": data["inputData"].get(str(item["id"]))},
            )
            for item in schemas[data["schemaId"]]["inputs"]
        ]
        result.append(
            {
                "id": node["id"],
                "schemaId": data["schemaId"],
                "inputs": inputs,
                "nodeType": node["type"],
            }
        )
    return result


def backend_request(
    content: Json, schemas: dict[str, Json], directory: Path
) -> list[Json]:
    """The /run data the UI sends for content, every Save writing into directory.

    As the UI opens a save (with_defaults) and runs it (optimize_chain, then
    to_backend_json), after the Save rewrite.
    """
    nodes, edges = rewrite_saves(content, schemas, directory)
    nodes = [with_defaults(node, schemas) for node in nodes]
    nodes, edges = optimize_chain(nodes, edges, schemas)
    if not nodes:
        raise ValueError("No node of the chain has an effect")
    return to_backend_json(nodes, edges, schemas)


def prepare(
    chain: Chain, schemas: dict[str, Json], images: list[dict[str, str]]
) -> Json:
    """The chain's record: "ready" with its content, "skipped" for a missing
    input, or "refused" with the reason."""
    record: Json = {"source": str(chain.path), "torch_cpu": chain.torch_cpu}
    if not chain.path.is_file():
        return {**record, "status": "skipped", "missing": [{"chain": str(chain.path)}]}
    try:
        content = read_chain(chain.path, schemas)
        content, substitutions, missing = resolve_inputs(
            content, schemas, images, chain.fallback
        )
    except Exception as error:
        return {**record, "status": "refused", **failure(error)}
    record.update(substitutions=substitutions, missing=missing)
    if missing:
        return {**record, "status": "skipped"}
    return {**record, "status": "ready", "content": content}


# Dependency manager.


def coerce_version(version: str) -> tuple[int, ...] | None:
    match = VERSION_NUMBERS.search(version)
    return None if match is None else tuple(int(part or 0) for part in match.groups())


def dependency_report(packages: list[Json], installed: dict[str, str]) -> Json:
    """Every pin /packages lists against /installed-dependencies, as the UI's
    dependency manager reads them: satisfied when installed and not older."""
    pins = []
    for package in packages:
        for dependency in package["dependencies"]:
            name, pin = dependency["pypiName"], dependency["version"]
            have = installed.get(name)
            wanted = coerce_version(pin)
            found = None if have is None else coerce_version(have)
            pins.append(
                {
                    "package": package["id"],
                    "pypi_name": name,
                    "pin": pin,
                    "installed": have,
                    "satisfied": wanted is not None
                    and found is not None
                    and found >= wanted,
                }
            )
    return {
        "pins": pins,
        "unsatisfied": [pin for pin in pins if not pin["satisfied"]],
        "pass": bool(pins) and all(pin["satisfied"] for pin in pins),
    }


def check_dependencies(backend: Backend) -> Json:
    packages = request(backend.port, "/packages", timeout=60)
    installed = request(backend.port, "/installed-dependencies", timeout=60)
    write_json(backend.directory / "packages.json", packages)
    write_json(backend.directory / "installed-dependencies.json", installed)
    report = dependency_report(packages, installed)
    log = (backend.directory / "backend.log").read_text("utf-8", errors="replace")
    report["host_start"] = install_check(log.splitlines())
    report["pass"] = report["pass"] and report["host_start"]["pass"]
    return report


# Host pass.


def saved_files(directory: Path) -> list[str]:
    return sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    )


def stream(backend: Backend) -> BenchEvents:
    if backend.events is None:
        raise RuntimeError(f"{backend.label} has no SSE stream")
    return backend.events


def run_chain(backend: Backend, content: Json, directory: Path) -> Json:
    """One /run of content on backend, Save writing into directory/saved."""
    saved = directory / "saved"
    saved.mkdir(parents=True)
    events = stream(backend)
    nodes = backend_request(content, backend.schemas, saved)
    payload = {"data": nodes, "options": OPTIONS, "sendBroadcastData": True}
    write_json(directory / "request.json", payload)
    start = len(events.events)
    began = time.perf_counter()
    response = request(backend.port, "/run", payload, timeout=RUN_TIMEOUT)
    seconds = time.perf_counter() - began
    check_success(response)
    received = events.wait_for(
        {node["id"] for node in nodes},
        start,
        expected_broadcasts={
            node["id"] for node in nodes if backend.schemas[node["schemaId"]]["outputs"]
        },
    )
    write_json(directory / "events.json", received)
    return {
        "directory": str(directory),
        "seconds": round(seconds, 3),
        "response": response,
        "request": canonical(nodes, directory),
        "saves": sum(node["schemaId"] == SAVE for node in nodes),
        "files": saved_files(saved),
        "sse_contract": sse_contract(received, directory),
    }


def wait_mid_run(events: BenchEvents, start: int, running: Future[Any]) -> set[str]:
    """The nodes finished once the run is under way: its chain-start and a first
    node-finish have arrived and /run has not returned."""
    while not running.done():
        if events.error:
            raise RuntimeError(f"SSE failure: {events.error}")
        received = events.events[start:]
        finished = {
            x["data"].get("nodeId") for x in received if x["event"] == "node-finish"
        }
        if finished and any(x["event"] == "chain-start" for x in received):
            return finished
        time.sleep(0.01)
    raise RuntimeError(f"The run ended before the kill: {running.result()!r}")


def ui_state(late: list[Json], executor: str | None, stream_error: str | None) -> Json:
    """The UI state after a stopped run: once /run has returned, no running
    event (RUNNING_EVENTS) may arrive, the status poll must read "ready" and the
    SSE stream must live."""
    running = [event for event in late if event["event"] in RUNNING_EVENTS]
    return {
        "late_running_events": running,
        "late_event_kinds": sorted({event["event"] for event in late}),
        "executor_after": executor,
        "stream_error": stream_error,
        "pass": not running and executor == "ready" and stream_error is None,
    }


def stop_chain(backend: Backend, content: Json, directory: Path) -> Json:
    """/kill mid-run on content, the UI state after it, then one /run that completes."""
    events = stream(backend)
    killed = directory / "kill"
    saved = killed / "saved"
    saved.mkdir(parents=True)
    nodes = backend_request(content, backend.schemas, saved)
    payload = {"data": nodes, "options": OPTIONS, "sendBroadcastData": True}
    write_json(killed / "request.json", payload)
    ids = {node["id"] for node in nodes}
    start = len(events.events)
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(request, backend.port, "/run", payload, RUN_TIMEOUT)
        finished_before = wait_mid_run(events, start, running)
        kill = request(backend.port, "/kill", {}, timeout=RUN_TIMEOUT)
        response = running.result()
    end = len(events.events)
    time.sleep(QUIET)
    late = events.events[end:]
    status = request(backend.port, "/status", timeout=15)
    during = events.events[start:end]
    write_json(killed / "events.json", {"until_returned": during, "late": late})
    finished = {x["data"].get("nodeId") for x in during if x["event"] == "node-finish"}
    worker = status.get("worker") or {}
    return {
        "kill_response": kill,
        "run_response": response,
        "finished_before_kill": len(finished_before & ids),
        "finished": len(finished & ids),
        "nodes": len(ids),
        "interrupted": not ids <= finished,
        "ui_state": ui_state(late, worker.get("executor"), events.error),
        "rerun": run_chain(backend, content, directory / "rerun"),
    }


def host_side(
    backend: Backend, prepared: dict[str, Json], directory: Path
) -> dict[str, Json]:
    """Every ready chain run once on backend; a failing run is recorded."""
    runs = {}
    for name, chain in prepared.items():
        if chain["status"] == "ready":
            print(f"{backend.label}: {name}", flush=True)
            runs[name] = attempt(run_chain, backend, chain["content"], directory / name)
    return runs


def longest_chain(runs: dict[str, Json]) -> str | None:
    done = [name for name, run in runs.items() if "error" not in run]
    return max(done, key=lambda name: runs[name]["seconds"], default=None)


def run_comparison(
    port: Json, oracle: Json, images: list[Json], torch_cpu: bool
) -> Json:
    """Port against oracle for one run: responses, the /run data, Save files,
    decoded pixels (verify_runtime.compare_images rows) and sse_contract. A
    pixel difference fails Save only without torch_cpu."""
    errors = [side["error"] for side in (port, oracle) if "error" in side]
    if errors:
        return {"errors": errors}
    mismatched = [row for row in images if not row["pass"]]
    files_equal = (
        port["files"] == oracle["files"]
        and len(port["files"]) == port["saves"] == oracle["saves"] > 0
    )
    result: Json = {
        "response_equal": port["response"] == oracle["response"],
        "request_equal": port["request"] == oracle["request"],
        "files_equal": files_equal,
        "only_port": sorted(set(port["files"]) - set(oracle["files"])),
        "only_oracle": sorted(set(oracle["files"]) - set(port["files"])),
        "images": len(images),
        "shapes_equal": all(row["same_shape_dtype"] for row in images),
        "pixels_equal": not mismatched,
        "pixel_differences": [
            {
                key: row[key]
                for key in ("name", "max_channel_difference", "differing_components")
            }
            for row in mismatched
        ],
        "torch_cpu": torch_cpu,
        "sse_contract_equal": port["sse_contract"] == oracle["sse_contract"],
    }
    result["run_pass"] = result["response_equal"] and result["request_equal"]
    result["save_pass"] = (
        files_equal
        and len(images) == len(port["files"])
        and result["shapes_equal"]
        and (not mismatched or torch_cpu)
    )
    return result


def compare(port: Json, oracle: Json, torch_cpu: bool) -> Json:
    """run_comparison, with the Save outputs decoded (in this interpreter, -B)."""
    images = []
    if "error" not in port and "error" not in oracle:
        pairs = [
            {
                "name": name,
                "baseline": str(Path(oracle["directory"], "saved", name)),
                "converted": str(Path(port["directory"], "saved", name)),
            }
            for name in sorted(set(port["files"]) & set(oracle["files"]))
        ]
        images = compare_images(Path(sys.executable), pairs) if pairs else []
    return run_comparison(port, oracle, images, torch_cpu)


def compared(comparison: Json) -> bool:
    """Both sides ran and were compared (a failed run or comparison is not)."""
    return "error" not in comparison and "errors" not in comparison


def stop_comparison(port: Json, oracle: Json, rerun: Json) -> Json:
    """The two stops: equal responses, both interrupted, both UI states held,
    and the rerun after them compared as any run."""
    errors = [side["error"] for side in (port, oracle) if "error" in side]
    if errors:
        return {"errors": errors, "pass": False}
    result: Json = {
        "kill_response_equal": port["kill_response"] == oracle["kill_response"],
        "run_response_equal": port["run_response"] == oracle["run_response"],
        "interrupted": port["interrupted"] and oracle["interrupted"],
        "ui_state": port["ui_state"]["pass"] and oracle["ui_state"]["pass"],
        "rerun": rerun,
    }
    result["pass"] = (
        result["kill_response_equal"]
        and result["run_response_equal"]
        and result["interrupted"]
        and result["ui_state"]
        and compared(rerun)
        and rerun["run_pass"]
        and rerun["save_pass"]
        and rerun["sse_contract_equal"]
    )
    return result


def brief(run: Json) -> Json:
    """A run as the summary shows it; request.json and events.json hold the rest."""
    if "error" in run:
        return run
    keys = ("directory", "seconds", "response", "files")
    return {key: run[key] for key in keys}


def host_pass(
    summary: Json,
    package: Path,
    python: Path,
    oracle: Path,
    oracle_python: Path,
    directory: Path,
) -> None:
    """The dependency manager and the host pass, recorded into summary as they go."""
    directory.mkdir()
    print("Host pass: hashing the package", flush=True)
    before = tree_state(package)
    images = bench_fixtures.load_inputs()
    chains: dict[str, Json] = {}
    hosts: Json = {}
    stop: Json = {}
    summary.update(chains=chains, hosts=hosts, stop=stop)
    port = Backend(
        "port", package / "resources/src", directory, FFMPEG, FFPROBE, {}, python
    )
    with port:
        summary["dependencies"] = attempt(check_dependencies, port)
        prepared = {
            chain.name: prepare(chain, port.schemas, images) for chain in CHAINS
        }
        for name, record in prepared.items():
            chains[name] = {
                key: value for key, value in record.items() if key != "content"
            }
        port_runs = host_side(port, prepared, directory / "runs/port")
        for name, run in port_runs.items():
            chains[name]["port"] = brief(run)
        longest = longest_chain(port_runs)
        stop["chain"] = longest
        port_stop: Json = {"error": "no chain ran"}
        oracle_stop: Json = port_stop
        if longest is not None:
            print(f"port: stop {longest}", flush=True)
            port_stop = attempt(
                stop_chain, port, prepared[longest]["content"], directory / "stop/port"
            )
    hosts["port"] = port.info
    tree = directory / "oracle-src"
    shutil.copytree(oracle, tree, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    backend = Backend("oracle", tree, directory, FFMPEG, FFPROBE, {}, oracle_python)
    with backend:
        oracle_runs = host_side(backend, prepared, directory / "runs/oracle")
        if longest is not None:
            print(f"oracle: stop {longest}", flush=True)
            oracle_stop = attempt(
                stop_chain,
                backend,
                prepared[longest]["content"],
                directory / "stop/oracle",
            )
    hosts["oracle"] = backend.info
    print("Host pass: hashing the package again", flush=True)
    hosts["package_diff"] = tree_diff(before, tree_state(package))
    hosts["package_tree"] = launch_tree(hosts["package_diff"])
    torch: Json = {}
    summary["torch_cpu_pixel_differences"] = torch
    for name, port_run in port_runs.items():
        oracle_run = oracle_runs.get(name, {"error": "not run on the oracle"})
        torch_cpu = prepared[name]["torch_cpu"]
        comparison = attempt(compare, port_run, oracle_run, torch_cpu)
        chains[name].update(
            oracle=brief(oracle_run),
            comparison=comparison,
            status="compared" if compared(comparison) else "failed",
        )
        if torch_cpu and comparison.get("pixel_differences"):
            torch[name] = comparison["pixel_differences"]
    if longest is not None:
        torch_cpu = prepared[longest]["torch_cpu"]
        rerun = attempt(
            compare,
            port_stop.get("rerun", port_stop),
            oracle_stop.get("rerun", oracle_stop),
            torch_cpu,
        )
        stop["comparison"] = stop_comparison(port_stop, oracle_stop, rerun)
        for side, record in (("port", port_stop), ("oracle", oracle_stop)):
            stop[side] = (
                {**record, "rerun": brief(record["rerun"])}
                if "rerun" in record
                else record
            )
        if torch_cpu and rerun.get("pixel_differences"):
            torch[f"{longest} (the run after the stop)"] = rerun["pixel_differences"]


# Summary.


def gates(summary: Json) -> dict[str, bool]:
    """Every gate of the pass from its summary; a missing record fails its gate."""
    launch_record = summary.get("launch") or {}
    log = launch_record.get("log") or {}
    close = launch_record.get("close") or {}
    chains = summary.get("chains") or {}
    results = [
        c["comparison"] for c in chains.values() if c.get("status") == "compared"
    ]
    hosts = summary.get("hosts") or {}

    def passed(record: object) -> bool:
        return isinstance(record, dict) and record.get("pass") is True

    return {
        "launch_started": (launch_record.get("start") or {}).get("started") is True,
        "launch_appdata_unchanged": (launch_record.get("appdata") or {}).get(
            "unchanged"
        )
        is True,
        "launch_package_tree": passed(launch_record.get("package_tree")),
        **{f"launch_log_{key}": passed(log.get(key)) for key in LOG_CHECKS},
        "launch_closed": close.get("main_window") is True
        and close.get("no_survivors") is True,
        "dependencies": passed(summary.get("dependencies")),
        "chains_run": bool(results)
        and all(c.get("status") in ("compared", "skipped") for c in chains.values())
        and all(result["run_pass"] for result in results),
        "save_outputs": bool(results)
        and all(result["save_pass"] for result in results),
        "sse_contract": bool(results)
        and all(result["sse_contract_equal"] for result in results),
        "stop": passed((summary.get("stop") or {}).get("comparison")),
        "hosts_clean": all(
            (hosts.get(side) or {}).get(key) is True
            for side in ("port", "oracle")
            for key in HOST_CLEAN
        )
        and passed(hosts.get("package_tree")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", type=Path, default=PROJECT / "out/chaiNNer-C")
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    manifest = package_manifest.load(package / package_manifest.MANIFEST)
    identity = manifest["identity"]
    if (
        manifest.get("state") != "complete"
        or identity["destination"] != str(package)
        or identity["project"] != str(PROJECT)
    ):
        raise ValueError(
            "A completed package manifest matching this project/location is required"
        )
    python = package / "python/python/python.exe"
    if not all(
        path.is_file()
        for path in (python, package / "portable", package / "chaiNNer.exe")
    ):
        raise ValueError(
            "The package's chaiNNer.exe, Python runtime or portable marker is missing"
        )
    oracle, oracle_python, record = oracle_source(manifest, ORACLE)
    busy, exited = running_from(
        [
            package,
            Path(os.environ["LOCALAPPDATA"], "chaiNNer"),
            Path(os.environ["APPDATA"], "chaiNNer"),
        ]
    )
    if busy:
        raise ValueError(f"Close these first; nothing was started: {busy}")
    if exited:
        print(f"Skipping exited process objects (0 threads): {exited}", flush=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run = PROJECT / "native/reports" / f"real-use-{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    summary: Json = {
        "utc": datetime.now(UTC).isoformat(),
        "package": str(package),
        "run": str(run),
        "oracle": record,
        "exited_processes_skipped": exited,
        "options": OPTIONS,
        "tool": {
            "python": sys.executable,
            "cpu_affinity": psutil.Process().cpu_affinity(),
            "environment": {name: os.environ.get(name) for name in TOOL_ENVIRONMENT},
        },
    }
    began = time.monotonic()
    try:
        summary["launch"] = attempt(
            launch,
            package,
            run / "launch",
            manifest["python_stack"]["interpreter_version"],
        )
        try:
            host_pass(summary, package, python, oracle, oracle_python, run / "host")
        except Exception as error:
            summary["host_error"] = failure(error)
    finally:
        summary["elapsed_seconds"] = round(time.monotonic() - began)
        summary["gates"] = gates(summary)
        summary["success"] = all(summary["gates"].values())
        write_json(run / "summary.json", summary)
        print(
            json.dumps(
                {
                    "summary": str(run / "summary.json"),
                    "success": summary["success"],
                    "gates": summary["gates"],
                }
            ),
            flush=True,
        )
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"Real-use pass refused: {error}", file=sys.stderr)
        raise SystemExit(2) from None
