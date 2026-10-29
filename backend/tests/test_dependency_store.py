from __future__ import annotations

import asyncio
import dataclasses
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


def test_no_pip_runs_when_every_pin_is_installed(monkeypatch: pytest.MonkeyPatch):
    # A start where the installed packages' metadata, read in this process, meets every
    # pin: neither install path starts pip (a pip process takes 0.4-0.6 s).
    fake_environment(monkeypatch, {"einops": "0.9.0", "spandrel": "0.4.2+c2"})
    monkeypatch.setattr(store, "installed_packages", {})
    store.scan_installed_packages()

    def pip(*arguments: Any, **_options: Any):
        raise AssertionError(f"pip ran: {arguments}")

    monkeypatch.setattr(store.subprocess, "check_call", pip)
    monkeypatch.setattr(store.subprocess, "Popen", pip)

    async def progress(*_arguments: Any) -> None:
        raise AssertionError("progress reported")

    dependencies = [DependencyInfo(package_name="einops", version="0.8.2"), FORK]
    assert store.install_dependencies_sync(dependencies) == 0
    assert asyncio.run(store.install_dependencies(dependencies, progress)) == 0


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


def test_a_download_is_named_by_its_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # chainner_pip's output for the PyTorch package on ROCm (shortened): the wheels it
    # resolved from their metadata files are downloaded after the last Collecting
    # step, which showed the 1.4 GB torch download as "Downloading tzdata...".
    r2 = "https://download-r2.pytorch.org/whl"
    output = [
        "Collecting torch==2.14.1+rocm7.14",
        f"  Downloading {r2}/rocm7.14/torch-2.14.1%2Brocm7.14-cp314-cp314-manylinux_2_28_x86_64.whl.metadata (37 kB)",
        "Collecting tzdata (from python-dateutil>=2.7->matplotlib->filterpy->facexlib==0.3.0)",
        "  Downloading tzdata-2026.6-py2.py3-none-any.whl (348 kB)",
        f"Downloading {r2}/rocm7.14/torch-2.14.1%2Brocm7.14-cp314-cp314-manylinux_2_28_x86_64.whl (1393.8 MB)",
        'Progress: {"current": 10240, "total": 1461506662}',
        f"Downloading {r2}/triton_rocm-3.8.0-cp314-cp314-linux_x86_64.whl (361.6 MB)",
        "Installing collected packages: tzdata, triton-rocm, torch",
    ]
    run_like_pip(monkeypatch, tmp_path, output, exit_code=0)
    fake_environment(monkeypatch, {})
    messages: list[str] = []

    async def record(message: str, *_args: object) -> None:
        messages.append(message)

    torch = DependencyInfo("torch", "2.14.1+rocm7.14", display_name="PyTorch")
    asyncio.run(store.install_dependencies([torch], record))
    assert messages == [
        "Collecting PyTorch...",
        "Downloading PyTorch...",
        "Collecting tzdata...",
        "Downloading tzdata...",
        "Downloading PyTorch...",
        "Downloading PyTorch...",
        "Downloading triton_rocm...",
        "Installing collected dependencies...",
        "Finished installing dependencies...",
    ]


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


def test_the_installed_versions_are_listed_in_their_canonical_form(
    monkeypatch: pytest.MonkeyPatch,
):
    # Metadata may spell a version as pip does not list it; an invalid one stays.
    monkeypatch.setattr(store, "installed_packages", {})
    fake_environment(
        monkeypatch,
        {
            "vapoursynth-mlrt-ov": "v17.1",
            "Odd_Pkg": "1.0RC1",
            "spandrel": "0.4.2+c2",
            "custom": "2.0-custom",
        },
    )
    store.scan_installed_packages()
    assert store.installed_packages == {
        "vapoursynth-mlrt-ov": "17.1",
        "odd-pkg": "1.0rc1",
        "spandrel": "0.4.2+c2",
        "custom": "2.0-custom",
    }


def test_a_fresh_python_without_packaging_lists_its_packages():
    # The integrated Python has no packaging until the server's dependencies are
    # installed; run.py lists the installed packages before installing them.
    script = (
        "import sys\n"
        "sys.modules['packaging'] = None\n"
        "from dependencies import store\n"
        "store.scan_installed_packages()\n"
        "assert 'pip' in store.installed_packages, store.installed_packages\n"
        "print(store.listed_version('1.0RC1'), store.listed_version('2.0-custom'))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(store.__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["1.0rc1", "2.0-custom"]


MIGRAPHX = DependencyInfo(
    "onnxruntime-migraphx",
    "1.30.0+rocm7.14",
    display_name="ONNX Runtime (MIGraphX)",
    url="https://example.invalid/onnxruntime_migraphx-1.30.0%2Brocm7.14-py3-none-any.whl",
    replaces=("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml"),
)


def pip_runs(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    installs: tuple[DependencyInfo, ...] = (MIGRAPHX,),
) -> list[list[str]]:
    """Replaces store's pip runs (with and without progress) with ones that print what
    pip prints, make its change to `environment` (an install adds `installs`, an
    uninstall removes the distributions it names) and succeed. Returns each run's
    arguments."""
    runs: list[list[str]] = []

    def run(command: list[str]) -> str:
        # The arguments after "-m pip", "-m chainner_pip" or the run_chainner_pip script
        runs.append(command[2 if command[1] == store.run_chainner_pip else 3 :])
        action, *arguments = runs[-1]
        lines: list[str] = []
        if action == "uninstall":
            for name in arguments[:-1]:
                version = environment.pop(name)
                lines += [
                    f"Found existing installation: {name} {version}",
                    f"Uninstalling {name}-{version}:",
                    f"  Successfully uninstalled {name}-{version}",
                ]
        else:
            for dependency in installs:
                environment[dependency.package_name] = dependency.version
                lines.append(f"Collecting {pin(dependency)}")
            names = ", ".join(d.package_name for d in installs)
            lines.append(f"Installing collected packages: {names}")
        return "".join(f"{line}\n" for line in lines)

    def check_call(command: list[str], **_kwargs: object) -> int:
        run(command)
        return 0

    popen = subprocess.Popen

    def fake_popen(command: list[str], **kwargs: Any) -> subprocess.Popen[str]:
        printing = "import sys; sys.stdout.write(sys.argv[1])"
        return popen([sys.executable, "-c", printing, run(command)], **kwargs)

    monkeypatch.setattr(store.subprocess, "check_call", check_call)
    monkeypatch.setattr(store.subprocess, "Popen", fake_popen)
    return runs


def installed(
    monkeypatch: pytest.MonkeyPatch, packages: dict[str, str]
) -> dict[str, str]:
    """`packages` as this Python's distributions, listed as the host lists them."""
    environment = fake_environment(monkeypatch, packages)
    monkeypatch.setattr(store, "installed_packages", {})
    store.scan_installed_packages()
    return environment


@pytest.mark.parametrize("progress", [False, True], ids=["sync", "with-progress"])
def test_an_install_uninstalls_the_builds_it_replaces_first(
    monkeypatch: pytest.MonkeyPatch, progress: bool
):
    # pip would install the MIGraphX build over DirectML's files and list both; an
    # uninstall of either would then delete files the other needs.
    environment = installed(
        monkeypatch, {"onnxruntime-directml": "1.24.4", "onnx": "1.23.2"}
    )
    runs = pip_runs(monkeypatch, environment)
    messages: list[str] = []

    async def record(message: str, *_args: object) -> None:
        messages.append(message)

    count = asyncio.run(
        store.install_dependencies([MIGRAPHX], record if progress else None)
    )
    assert count == 1
    assert [run[0] for run in runs] == ["uninstall", "install"]
    assert runs[0][1:] == ["onnxruntime-directml", "-y"]
    assert store.installed_packages == {
        "onnx": "1.23.2",
        "onnxruntime-migraphx": "1.30.0+rocm7.14",
    }
    if progress:
        assert messages.index("Uninstalling onnxruntime-directml...") < (
            messages.index("Collecting ONNX Runtime (MIGraphX)...")
        )


@pytest.mark.parametrize("progress", [False, True], ids=["sync", "with-progress"])
def test_an_install_uninstalls_nothing_when_no_replaced_build_is_installed(
    monkeypatch: pytest.MonkeyPatch, progress: bool
):
    environment = installed(monkeypatch, {"onnx": "1.23.2"})
    runs = pip_runs(monkeypatch, environment)
    asyncio.run(
        store.install_dependencies([MIGRAPHX], ignore_progress if progress else None)
    )
    assert [run[0] for run in runs] == ["install"]


def test_an_installed_dependency_uninstalls_nothing(monkeypatch: pytest.MonkeyPatch):
    # Already at its version: no pip run at all. The other build listed beside it
    # shares its files, so uninstalling that would break the installed one.
    environment = installed(
        monkeypatch,
        {"onnxruntime-migraphx": "1.30.0+rocm7.14", "onnxruntime": "1.30.0"},
    )
    runs = pip_runs(monkeypatch, environment)
    assert asyncio.run(store.install_dependencies([MIGRAPHX], ignore_progress)) == 0
    assert store.install_dependencies_sync([MIGRAPHX]) == 0
    assert runs == []


def test_a_failed_uninstall_of_a_replaced_build_stops_the_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Reported as any failed uninstall, and the old build stays usable.
    run_like_pip(monkeypatch, tmp_path, UNINSTALL_LOCKED)
    actions: list[str] = []
    popen = store.subprocess.Popen

    def recording_popen(command: list[str], **kwargs: Any) -> subprocess.Popen[str]:
        actions.append(command[3])
        return popen(command, **kwargs)

    monkeypatch.setattr(store.subprocess, "Popen", recording_popen)
    installed(monkeypatch, {"onnxruntime-gpu": "1.30.0+trt11.4"})
    with pytest.raises(ValueError) as error:
        asyncio.run(store.install_dependencies([MIGRAPHX], ignore_progress))
    assert str(error.value).splitlines()[0] == (
        "An error occurred while uninstalling dependencies."
    )
    assert actions == ["uninstall"]


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


# PyTorch's CPU build, which replaces its ROCm build of the same name, the Triton for ROCm
# that one brought and ROCm's runtime (packages/chaiNNer_pytorch, dependencies/rocm.py)
CPU_TORCH = DependencyInfo(
    "torch",
    "2.14.1",
    display_name="PyTorch",
    extra_index_url="https://download.pytorch.org/whl/cpu",
    replaces=(
        "torch+rocm",
        "triton-rocm",
        "rocm",
        "rocm-sdk-core",
        "rocm-sdk-libraries",
        "rocm-sdk-device-*",
        "rocm-bootstrap",
        "amd-torch-device-*",
    ),
)


@pytest.mark.parametrize(
    ("name", "version", "result"),
    [
        ("torch", "2.14.1+rocm7.14", True),
        ("torch", "2.12.0+rocm7.14.1", True),
        ("torch", "2.14.1+cu132", False),
        ("torch", "2.14.1+cpu", False),
        ("torch", "2.14.1", False),
        # A name without a label names every build.
        ("triton-rocm", "3.8.0", True),
        ("triton-rocm", "3.8.0+git1", True),
        ("triton", "3.8.0", False),
        ("torchvision", "0.29.1+rocm7.14", False),
        ("rocm", "7.14.1+chainner", True),
        # A name ending in * names every distribution it starts.
        ("rocm-sdk-device-gfx1100", "7.14.1", True),
        ("amd-torch-device-gfx12-0", "2.12.0+rocm7.14.1", True),
        ("rocm-sdk-devel", "7.14.1", False),
        ("amd-torch", "2.12.0", False),
    ],
)
def test_a_replaced_build_is_named_by_its_distribution_and_label(
    name: str, version: str, result: bool
):
    assert store.replaced(CPU_TORCH.replaces, name, version) is result


@pytest.mark.parametrize(
    ("installed", "needed"),
    [
        ("2.14.1+rocm7.14", True),
        ("2.14.1+cpu", False),
        ("2.14.1", False),
        ("2.14.1+cu132", False),
    ],
)
def test_an_installed_build_the_dependency_replaces_never_satisfies_it(
    monkeypatch: pytest.MonkeyPatch, installed: str, needed: bool
):
    monkeypatch.setattr(store, "installed_packages", {"torch": installed})
    assert (filter_necessary_to_install([CPU_TORCH]) == [CPU_TORCH]) is needed


@pytest.mark.parametrize("progress", [False, True], ids=["sync", "with-progress"])
def test_an_install_uninstalls_a_replaced_build_of_its_own_distribution_first(
    monkeypatch: pytest.MonkeyPatch, progress: bool
):
    # pip would find torch==2.14.1 met by the ROCm build, 2.14.1+rocm7.14, and keep it;
    # ROCm's runtime would stay with nothing using it.
    environment = installed(
        monkeypatch,
        {
            "torch": "2.14.1+rocm7.14",
            "triton-rocm": "3.8.0",
            "rocm-sdk-core": "7.14.1",
            "rocm-sdk-device-gfx1100": "7.14.1",
            "einops": "0.8.2",
        },
    )
    runs = pip_runs(monkeypatch, environment, (CPU_TORCH,))
    count = asyncio.run(
        store.install_dependencies([CPU_TORCH], ignore_progress if progress else None)
    )
    assert count == 1
    assert [run[0] for run in runs] == ["uninstall", "install"]
    assert runs[0][1:] == [
        "rocm-sdk-core",
        "rocm-sdk-device-gfx1100",
        "torch",
        "triton-rocm",
        "-y",
    ]
    assert runs[1][1] == "torch==2.14.1"
    assert store.installed_packages == {"torch": "2.14.1", "einops": "0.8.2"}


@pytest.mark.parametrize("torch", ["2.14.1+cpu", "2.14.1+cu126", "2.10.0"])
def test_an_install_without_amd_builds_uninstalls_nothing_first(
    monkeypatch: pytest.MonkeyPatch, torch: str
):
    # NVIDIA's build in place of CUDA 12.6's, the CPU build's or an older one: pip
    # replaces it in one run, as before.
    nvidia_torch = dataclasses.replace(CPU_TORCH, version="2.14.1+cu132")
    environment = installed(monkeypatch, {"torch": torch, "triton": "3.8.0"})
    runs = pip_runs(monkeypatch, environment, (nvidia_torch,))
    assert asyncio.run(store.install_dependencies([nvidia_torch], ignore_progress)) == 1
    assert [run[0] for run in runs] == ["install"]


def dist_info(
    site: Path,
    name: str,
    requires: tuple[str, ...] = (),
    extras: tuple[str, ...] = (),
    requested: bool = False,
    files: tuple[str, ...] | None = None,
) -> None:
    """An installed distribution's metadata; `requested`: installed by name (pip's
    REQUESTED mark), not pulled in as a dependency. Its RECORD lists `files` (by
    default, None, a module of its own name; no RECORD for ())."""
    folder = site / f"{name.replace('-', '_')}-1.0.dist-info"
    folder.mkdir(parents=True)
    if files != ():
        module = name.lower().replace("-", "_")
        listed = [*(files or (f"{module}/__init__.py",)), f"{folder.name}/METADATA"]
        # importlib.metadata lists only the files that exist.
        for path in listed[:-1]:
            (site / path).parent.mkdir(parents=True, exist_ok=True)
            (site / path).write_text("", encoding="utf-8")
        (folder / "RECORD").write_text(
            "".join(f"{path},,\n" for path in listed), encoding="utf-8"
        )
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {name}",
        "Version: 1.0",
        *[f"Provides-Extra: {extra}" for extra in extras],
        *[f"Requires-Dist: {requirement}" for requirement in requires],
    ]
    (folder / "METADATA").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if requested:
        (folder / "REQUESTED").write_text("", encoding="utf-8")


def test_an_uninstall_takes_the_dependencies_nothing_else_needs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Uninstalling only the listed distributions of the VapourSynth package left about
    # 1.8 GB behind: vsview's PySide6, CUDA wheels, the extras' plugins.
    site = tmp_path / "site"
    dist_info(site, "VapourSynth", requested=True)
    dist_info(
        site,
        "vsjetpack",
        (
            "typing_extensions>=4",
            "jetpack-helper",
            'deinterlace-plugin; extra == "deinterlace"',
            'other-plugin; extra == "other"',
        ),
        ("deinterlace", "other"),
        requested=True,
    )
    dist_info(
        site,
        "vsview",
        ("PySide6>=6.10", "numpy", "my-tool", 'old-dep; python_version < "3.0"'),
        requested=True,
    )
    dist_info(site, "dfttest2", ('nvidia-cufft; extra == "cufft"',), ("cufft",))
    dist_info(site, "PySide6", ("shiboken6==6.10", "PySide6_Addons"))
    for name in ("shiboken6", "PySide6_Addons", "other-plugin", "old-dep", "numpy"):
        dist_info(site, name)
    dist_info(site, "typing_extensions")
    dist_info(site, "deinterlace-plugin", ("shared-lib",))
    dist_info(site, "shared-lib")
    dist_info(site, "jetpack-helper")
    dist_info(site, "nvidia-cufft")
    # Staying: a package of the user's that needs shared-lib, a user module that
    # needs jetpack-helper, one that needs the CUDA wheel under an extra, and a tool
    # the user installed by name.
    dist_info(site, "chainner-thing", ("shared-lib",), requested=True)
    dist_info(site, "vs-colorfix", ("jetpack-helper",), requested=True)
    dist_info(
        site, "torchlike", ('nvidia-cufft; extra == "gpu"',), ("gpu",), requested=True
    )
    dist_info(site, "my-tool", requested=True)
    distributions = store.importlib.metadata.distributions
    monkeypatch.setattr(
        store.importlib.metadata,
        "distributions",
        lambda: distributions(path=[str(site)]),
    )
    leaving = [
        DependencyInfo(package_name=name, version="1.0")
        for name in (
            "VapourSynth",
            "vsjetpack[deinterlace]",
            "vsview",
            "dfttest2[cufft]",
        )
    ]
    # The server lists typing_extensions; numpy is always kept.
    unneeded = store.unneeded_dependencies(leaving, {"typing-extensions"})
    assert [(dep.package_name, dep.version) for dep in unneeded] == [
        ("deinterlace-plugin", "1.0"),
        ("pyside6", "1.0"),
        ("pyside6-addons", "1.0"),
        ("shiboken6", "1.0"),
    ]

    # What a distribution needs is unknown when its requirements cannot be read.
    dist_info(site, "broken", ("not a requirement !",))
    with pytest.raises(ValueError):
        store.unneeded_dependencies(leaving, {"typing-extensions"})


def test_an_uninstall_keeps_a_dependency_that_shares_files_with_one_that_stays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # vs-jetpack's niquests needs urllib3-future, whose wheel installs into urllib3/
    # over urllib3's own files: pip would delete urllib3/__init__.py with it, and
    # `import requests` (downloads, Align Image, the web UI) would fail.
    site = tmp_path / "site"
    dist_info(site, "vsjetpack", ("niquests", "no-record"), requested=True)
    dist_info(site, "niquests", ("urllib3-future", "wassima"))
    dist_info(
        site,
        "urllib3-future",
        ("h11",),
        files=("urllib3/__init__.py", "urllib3/contrib/hface/__init__.py"),
    )
    dist_info(site, "urllib3", files=("urllib3/__init__.py", "urllib3/util/retry.py"))
    dist_info(site, "requests", ("urllib3",), requested=True)
    for name in ("wassima", "h11"):
        dist_info(site, name)
    dist_info(site, "no-record", files=())
    distributions = store.importlib.metadata.distributions
    monkeypatch.setattr(
        store.importlib.metadata,
        "distributions",
        lambda: distributions(path=[str(site)]),
    )
    leaving = [DependencyInfo(package_name="vsjetpack", version="1.0")]
    # urllib3-future stays, and so does what it needs (h11); a dependency without a
    # RECORD stays too: what pip would delete with it is unknown.
    unneeded = store.unneeded_dependencies(leaving, set())
    assert [dep.package_name for dep in unneeded] == ["niquests", "wassima"]
