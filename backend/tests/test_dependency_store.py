from __future__ import annotations

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
