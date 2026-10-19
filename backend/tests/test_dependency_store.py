from __future__ import annotations

import asyncio
import importlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dependencies import store
from dependencies.store import DependencyInfo, filter_necessary_to_install, pin


def fake_environment(
    monkeypatch: pytest.MonkeyPatch, packages: dict[str, str]
) -> dict[str, str]:
    """Makes `packages` (name as the metadata spells it -> version) the distributions
    importlib.metadata lists for this Python; a fake pip run changes the dict."""
    monkeypatch.setattr(
        store.importlib.metadata,
        "distributions",
        lambda: [
            SimpleNamespace(metadata={"Name": name, "Version": version})
            for name, version in packages.items()
        ],
    )
    return packages


FORK = DependencyInfo(
    package_name="spandrel",
    version="0.4.2+c2",
    url="https://example.invalid/spandrel-0.4.2%2Bc2-py3-none-any.whl",
)


def test_a_dependency_with_a_url_installs_that_wheel():
    assert pin(FORK) == f"spandrel @ {FORK.url}"
    assert pin(DependencyInfo(package_name="einops", version="0.8.2")) == (
        "einops==0.8.2"
    )


@pytest.mark.parametrize(
    ("installed", "needed"),
    [
        (None, True),
        ("0.4.2", True),  # PyPI's build, not the fork's
        ("0.4.3", True),  # a newer PyPI release lacks the fork's architectures
        ("0.4.2+c1", True),  # the fork's previous build
        ("0.4.2+c2", False),
    ],
)
def test_a_local_label_needs_exactly_that_build(
    monkeypatch: pytest.MonkeyPatch, installed: str | None, needed: bool
):
    packages = {} if installed is None else {"spandrel": installed}
    monkeypatch.setattr(store, "installed_packages", packages)
    assert (filter_necessary_to_install([FORK]) == [FORK]) is needed


@pytest.mark.parametrize(
    ("installed", "needed"), [("0.8.1", True), ("0.8.2", False), ("0.9.0", False)]
)
def test_a_plain_version_is_a_floor(
    monkeypatch: pytest.MonkeyPatch, installed: str, needed: bool
):
    dependency = DependencyInfo(package_name="einops", version="0.8.2")
    monkeypatch.setattr(store, "installed_packages", {"einops": installed})
    assert (filter_necessary_to_install([dependency]) == [dependency]) is needed


def test_a_package_with_extras_is_listed_by_its_name(monkeypatch: pytest.MonkeyPatch):
    # pip installs "vsjetpack[deinterlace]" but lists (and uninstalls) "vsjetpack".
    jetpack = DependencyInfo("vsjetpack[deinterlace]", "2.2.6")
    assert pin(jetpack) == "vsjetpack[deinterlace]==2.2.6"
    monkeypatch.setattr(store, "installed_packages", {"vsjetpack": "2.2.6"})
    assert filter_necessary_to_install([jetpack]) == []

    commands: list[list[str]] = []
    environment = fake_environment(monkeypatch, {})

    def check_call(command: list[str], **_kwargs: object) -> int:
        commands.append(command)
        if command[3] == "install":
            environment["vsjetpack"] = "2.2.6"
        else:
            environment.pop("vsjetpack")
        return 0

    monkeypatch.setattr(store, "installed_packages", {})
    monkeypatch.setattr(store.subprocess, "check_call", check_call)
    store.install_dependencies_sync([jetpack])
    assert "vsjetpack[deinterlace]==2.2.6" in commands[-1]
    assert store.installed_packages == {"vsjetpack": "2.2.6"}

    store.uninstall_dependencies_sync([jetpack])
    assert commands[-1][-2:] == ["vsjetpack", "-y"]
    assert store.installed_packages == {}


def test_startup_finds_a_package_pip_lists_under_another_spelling(
    monkeypatch: pytest.MonkeyPatch,
):
    # pip lists a package by the name in its own metadata: onnx-graphsurgeon's is
    # "onnx_graphsurgeon", and ModelOpt's package showed it missing after every
    # restart.
    environment = fake_environment(
        monkeypatch, {"onnx_graphsurgeon": "0.6.1", "Pillow": "12.0.0"}
    )

    def check_call(command: list[str], **_kwargs: object) -> int:
        if command[3] == "uninstall":
            environment.pop("onnx_graphsurgeon")
        return 0

    monkeypatch.setattr(store, "installed_packages", {})
    monkeypatch.setattr(subprocess, "check_call", check_call)
    monkeypatch.delitem(sys.modules, "dependencies.install_server_deps", raising=False)
    importlib.import_module("dependencies.install_server_deps")

    surgeon = DependencyInfo("onnx-graphsurgeon", "0.6.1")
    pillow = DependencyInfo("pillow", "12.0.0")
    assert filter_necessary_to_install([surgeon, pillow]) == []
    store.uninstall_dependencies_sync([surgeon])
    assert surgeon.distribution not in store.installed_packages


def test_each_extra_index_url_gets_its_own_flag(monkeypatch: pytest.MonkeyPatch):
    # pip reads a comma-joined list as one URL, so a batch of packages from two
    # indexes (Install All) would find neither's wheels.
    commands: list[list[str]] = []

    def check_call(command: list[str], **_kwargs: object) -> int:
        commands.append(command)
        return 0

    monkeypatch.setattr(store, "installed_packages", {})
    monkeypatch.setattr(store.subprocess, "check_call", check_call)
    store.install_dependencies_sync(
        [
            DependencyInfo("torch", "2.14.1", extra_index_url="https://b.invalid/whl"),
            DependencyInfo(
                "onnxruntime", "1.30.0", extra_index_url="https://a.invalid"
            ),
            DependencyInfo("einops", "0.8.2"),
        ]
    )
    [command] = commands
    first = command.index("--extra-index-url")
    assert command[first:] == [
        "--extra-index-url",
        "https://a.invalid",
        "--extra-index-url",
        "https://b.invalid/whl",
    ]


# chainner_pip's output for an unreachable index (shortened); a failed run's
# message carries the last retry and the errors.
RETRY = (
    "WARNING: Retrying (Retry(total={}, connect=None, read=None, redirect=None,"
    " status=None)) after connection broken by 'NewConnectionError('<HTTPConnection"
    " object>: Failed to establish a new connection: [WinError 10061] No connection"
    " could be made because the target machine actively refused it')':"
    " /simple/ncnn/"
)
NO_INDEX = [
    "Looking in indexes: https://pypi.org/simple",
    *[RETRY.format(n) for n in (2, 1, 0)],
    (
        "ERROR: Could not find a version that satisfies the requirement"
        " ncnn==1.0.20240410 (from versions: none)"
    ),
    "ERROR: No matching distribution found for ncnn==1.0.20240410",
]
# A crash prints a traceback longer than the lines kept: its end is the reason.
TRACEBACK = [
    "Collecting torch==2.14.1+cu132",
    'Progress: {"current": 10240, "total": 2001146779}',
    "ERROR: Exception:",
    "Traceback (most recent call last):",
    *[f'  File "chainner_pip/_internal/frame{n}.py", line {n}, in f' for n in range(9)],
    (
        "chainner_pip._vendor.urllib3.exceptions.ReadTimeoutError:"
        " HTTPSConnectionPool(host='download.pytorch.org', port=443): Read timed out."
    ),
]
UNINSTALL_LOCKED = [
    "Found existing installation: ncnn 1.0.20240410",
    "Uninstalling ncnn-1.0.20240410:",
    "ERROR: Exception:",
    "Traceback (most recent call last):",
    '  File "chainner_pip/_internal/utils/misc.py", line 122, in rmtree',
    "PermissionError: [WinError 5] Access is denied: 'C:/ncnn/ncnn.pyd'",
]
NCNN = DependencyInfo("ncnn", "1.0.20240410", display_name="NCNN")


def run_like_pip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    output: list[str],
    silence: float = 0,
    exit_code: int = 1,
) -> None:
    """Replaces store's pip process with one that prints `output` and exits with
    `exit_code` (1 by default: a failed run), silent for `silence` seconds after the
    first line. A surrogate escape in `output` is printed as that raw byte."""
    output_file = tmp_path / "pip_output.txt"
    output_file.write_text(
        "".join(f"{line}\n" for line in output),
        encoding="utf-8",
        errors="surrogateescape",
    )
    fake_pip = (
        "import sys, time;"
        " lines = open(sys.argv[1], 'rb').readlines();"
        " sys.stdout.buffer.write(lines[0]); sys.stdout.buffer.flush();"
        " time.sleep(float(sys.argv[2]));"
        " sys.stdout.buffer.write(b''.join(lines[1:])); sys.exit(int(sys.argv[3]))"
    )
    popen = subprocess.Popen

    def fake_popen(_command: list[str], **kwargs: Any) -> subprocess.Popen[str]:
        return popen(
            [
                sys.executable,
                "-c",
                fake_pip,
                str(output_file),
                str(silence),
                str(exit_code),
            ],
            **kwargs,
        )

    monkeypatch.setattr(store, "installed_packages", {})
    monkeypatch.setattr(store.subprocess, "Popen", fake_popen)


async def ignore_progress(*_args: object) -> None:
    pass


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        (NO_INDEX, NO_INDEX[-3:]),
        (TRACEBACK, TRACEBACK[-10:]),
    ],
    ids=["no-index", "crash"],
)
def test_a_failed_install_reports_pips_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    output: list[str],
    reason: list[str],
):
    run_like_pip(monkeypatch, tmp_path, output)
    with pytest.raises(ValueError) as error:
        asyncio.run(store.install_dependencies([NCNN], ignore_progress))
    assert str(error.value).splitlines() == [
        "An error occurred while installing dependencies.",
        *(line.strip() for line in reason),
    ]


def test_the_event_loop_runs_while_pip_is_silent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # pip is silent for long stretches (a slow index, its retries); the host's
    # loop must keep serving progress events and requests meanwhile.
    silence = 1.5
    run_like_pip(monkeypatch, tmp_path, NO_INDEX, silence)

    async def install_with_a_ticker() -> tuple[float, BaseException | None]:
        install = asyncio.create_task(
            store.install_dependencies([NCNN], ignore_progress)
        )
        longest_gap = 0.0
        last = time.monotonic()
        while not install.done():
            await asyncio.sleep(0.02)
            now = time.monotonic()
            longest_gap = max(longest_gap, now - last)
            last = now
        return longest_gap, install.exception()

    longest_gap, error = asyncio.run(install_with_a_ticker())
    assert longest_gap < silence / 3
    assert isinstance(error, ValueError)
    assert str(error).splitlines()[1:] == NO_INDEX[-3:]


def test_a_failed_uninstall_reports_pips_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    run_like_pip(monkeypatch, tmp_path, UNINSTALL_LOCKED)
    with pytest.raises(ValueError) as error:
        asyncio.run(store.uninstall_dependencies([NCNN], ignore_progress))
    assert str(error.value).splitlines() == [
        "An error occurred while uninstalling dependencies.",
        *(line.strip() for line in UNINSTALL_LOCKED[2:]),
    ]


@pytest.mark.parametrize(
    "error",
    [
        "[Errno 28] No space left on device",
        "[WinError 112] There is not enough space on the disk: 'C:/Temp/pip-unpack-1'",
    ],
)
def test_a_full_disk_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: str
):
    # pip's report names the full drive's path; the first line says what to do.
    output = [
        "Installing collected packages: ncnn",
        f"ERROR: Could not install packages due to an OSError: {error}",
    ]
    run_like_pip(monkeypatch, tmp_path, output)
    with pytest.raises(ValueError) as error_info:
        asyncio.run(store.install_dependencies([NCNN], ignore_progress))
    assert str(error_info.value).splitlines() == [
        (
            "Disk space ran out while installing dependencies."
            " Please free up disk space and try again."
        ),
        output[-1],
    ]


@pytest.mark.parametrize("action", ["installing", "uninstalling"])
def test_output_that_is_not_utf8_is_shown_escaped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, action: str
):
    # A line in the console's code page (a build tool's message) must neither end
    # the read with a UnicodeDecodeError while pip still runs nor hide pip's reason.
    output = [
        "Collecting ncnn==1.0.20240410",
        NO_INDEX[-2],
        "Microsoft (R) C/C++ caf\udce9",
        NO_INDEX[-1],
    ]
    run_like_pip(monkeypatch, tmp_path, output)
    run = (
        store.install_dependencies
        if action == "installing"
        else store.uninstall_dependencies
    )
    with pytest.raises(ValueError) as error:
        asyncio.run(run([NCNN], ignore_progress))
    assert str(error.value).splitlines() == [
        f"An error occurred while {action} dependencies.",
        NO_INDEX[-2],
        "Microsoft (R) C/C++ caf\\xe9",
        NO_INDEX[-1],
    ]


def test_an_uninstall_skips_dependencies_that_were_never_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # The uninstall route passes every dependency of a package; pip skips those that
    # are not installed, and a successful run must not fail on them afterwards.
    output = [
        "Found existing installation: ncnn 1.0.20240410",
        "Uninstalling ncnn-1.0.20240410:",
        "  Successfully uninstalled ncnn-1.0.20240410",
        "WARNING: Skipping onnx as it is not installed.",
    ]
    run_like_pip(monkeypatch, tmp_path, output, exit_code=0)
    fake_environment(monkeypatch, {})
    monkeypatch.setattr(store, "installed_packages", {"ncnn": "1.0.20240410"})
    never_installed = DependencyInfo("onnx", "1.20.0")
    asyncio.run(store.uninstall_dependencies([NCNN, never_installed], ignore_progress))
    assert store.installed_packages == {}


def test_an_uninstall_without_progress_forgets_the_packages(
    monkeypatch: pytest.MonkeyPatch,
):
    # The installed list must not keep reporting what was just removed.
    commands: list[list[str]] = []

    def check_call(command: list[str], **_kwargs: object) -> int:
        commands.append(command)
        return 0

    fake_environment(monkeypatch, {})
    monkeypatch.setattr(store, "installed_packages", {"ncnn": "1.0.20240410"})
    monkeypatch.setattr(store.subprocess, "check_call", check_call)
    asyncio.run(store.uninstall_dependencies([NCNN]))
    assert [command[3:] for command in commands] == [["uninstall", "ncnn", "-y"]]
    assert store.installed_packages == {}


def test_the_installed_packages_are_read_as_pip_lists_them(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(store, "installed_packages", {})
    store.scan_installed_packages()
    listed = json.loads(
        subprocess.check_output(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            env=store.ENV,
            encoding="utf-8",
        )
    )
    assert store.installed_packages == {
        store.distribution_name(p["name"]): p["version"] for p in listed
    }


ONNX = DependencyInfo("onnx", "1.20.0")


def test_an_install_lists_the_dependencies_pip_installed_with_it(
    monkeypatch: pytest.MonkeyPatch,
):
    # pip installs a package's own dependencies too; without them in the list, the
    # next install that needs them installs them again until a restart.
    environment = fake_environment(monkeypatch, {"numpy": "2.5.3"})

    def check_call(_command: list[str], **_kwargs: object) -> int:
        environment.update({"onnx": "1.20.0", "protobuf": "6.33.0"})
        return 0

    monkeypatch.setattr(store, "installed_packages", {"numpy": "2.5.3"})
    monkeypatch.setattr(store.subprocess, "check_call", check_call)
    assert store.install_dependencies_sync([ONNX]) == 1
    assert store.installed_packages == {
        "numpy": "2.5.3",
        "onnx": "1.20.0",
        "protobuf": "6.33.0",
    }


def test_a_failed_install_lists_what_pip_installed_before_it_failed(
    monkeypatch: pytest.MonkeyPatch,
):
    environment = fake_environment(monkeypatch, {})

    def check_call(command: list[str], **_kwargs: object) -> int:
        environment["protobuf"] = "6.33.0"
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(store, "installed_packages", {})
    monkeypatch.setattr(store.subprocess, "check_call", check_call)
    with pytest.raises(subprocess.CalledProcessError):
        store.install_dependencies_sync([ONNX])
    assert store.installed_packages == {"protobuf": "6.33.0"}


@pytest.mark.parametrize("action", ["installing", "uninstalling"])
def test_a_failed_run_with_progress_reads_the_list_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, action: str
):
    # What pip changed before it failed (here: protobuf in, ncnn out) is listed.
    run_like_pip(monkeypatch, tmp_path, NO_INDEX)
    fake_environment(monkeypatch, {"protobuf": "6.33.0"})
    monkeypatch.setattr(store, "installed_packages", {"ncnn": "1.0.20240410"})
    run = (
        store.install_dependencies
        if action == "installing"
        else store.uninstall_dependencies
    )
    with pytest.raises(ValueError):
        asyncio.run(run([NCNN, ONNX], ignore_progress))
    assert store.installed_packages == {"protobuf": "6.33.0"}


def test_a_rescan_finds_what_pip_added_and_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    def add(name: str, version: str) -> Path:
        info = tmp_path / f"{name}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
            encoding="utf-8",
        )
        return info

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(store, "installed_packages", {})
    first = add("chainner_fake_first", "1.0")
    store.scan_installed_packages()
    assert store.installed_packages["chainner-fake-first"] == "1.0"

    add("chainner_fake_second", "2.0")
    shutil.rmtree(first)
    store.scan_installed_packages()
    assert store.installed_packages["chainner-fake-second"] == "2.0"
    assert "chainner-fake-first" not in store.installed_packages
