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
