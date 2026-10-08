from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from dependencies import store
from dependencies.store import DependencyInfo, filter_necessary_to_install, pin

FORK = DependencyInfo(
    package_name="spandrel",
    version="0.4.2+c1",
    url="https://example.invalid/spandrel-0.4.2%2Bc1-py3-none-any.whl",
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
        ("0.4.2+c0", True),
        ("0.4.2+c1", False),
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


def fail_like_pip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, output: list[str]
) -> None:
    """Replaces store's pip process with one that prints `output` and exits 1."""
    output_file = tmp_path / "pip_output.txt"
    output_file.write_text("".join(f"{line}\n" for line in output), encoding="utf-8")
    fake_pip = (
        "import sys; sys.stdout.write(open(sys.argv[1], encoding='utf-8').read());"
        " sys.exit(1)"
    )
    popen = subprocess.Popen

    def fake_popen(_command: list[str], **kwargs: Any) -> subprocess.Popen[str]:
        return popen([sys.executable, "-c", fake_pip, str(output_file)], **kwargs)

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
    fail_like_pip(monkeypatch, tmp_path, output)
    with pytest.raises(ValueError) as error:
        asyncio.run(store.install_dependencies([NCNN], ignore_progress))
    assert str(error.value).splitlines() == [
        "An error occurred while installing dependencies.",
        *(line.strip() for line in reason),
    ]


def test_a_failed_uninstall_reports_pips_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    fail_like_pip(monkeypatch, tmp_path, UNINSTALL_LOCKED)
    with pytest.raises(ValueError) as error:
        asyncio.run(store.uninstall_dependencies([NCNN], ignore_progress))
    assert str(error.value).splitlines() == [
        "An error occurred while uninstalling dependencies.",
        *(line.strip() for line in UNINSTALL_LOCKED[2:]),
    ]
