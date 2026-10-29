"""What the VapourSynth package installs on Linux by the system's glibc: PyPI's wheels
from glibc 2.35; below that, a glibc 2.28 build of vapoursynth-akarin in place of PyPI's;
below 2.34 also one of vapoursynth-nlm-ispc, and no vsview, whose Qt (PySide6 6.11 and
newer) publishes no wheels for an older glibc. Needs no VapourSynth."""

from __future__ import annotations

import re

import pytest

from api import Dependency
from packages.chaiNNer_vapoursynth import (
    AKARIN_GLIBC_2_28,
    NLM_ISPC_GLIBC_2_28,
    vapoursynth_dependencies,
    vsview_unavailable_reason,
)


def urls(dependencies: list[Dependency]) -> list[Dependency]:
    return [dep for dep in dependencies if dep.url]


@pytest.mark.parametrize("glibc", [None, (2, 35), (2, 39), (3, 0)])
def test_newer_glibc_takes_every_wheel_from_the_indexes(glibc: tuple[int, int] | None):
    dependencies = vapoursynth_dependencies(glibc)

    assert urls(dependencies) == []
    names = [dep.pypi_name for dep in dependencies]
    assert "vapoursynth-akarin" not in names
    assert "vapoursynth-nlm-ispc" not in names
    assert names[-1] == "vsview"


def test_glibc_2_34_takes_the_glibc_2_28_build_of_akarin():
    dependencies = vapoursynth_dependencies((2, 34))

    assert urls(dependencies) == [AKARIN_GLIBC_2_28]
    assert dependencies[-1].pypi_name == "vsview"


@pytest.mark.parametrize("glibc", [(2, 28), (2, 31), (2, 33)])
def test_older_glibc_takes_both_builds_and_leaves_vsview_out(glibc: tuple[int, int]):
    dependencies = vapoursynth_dependencies(glibc)

    assert urls(dependencies) == [AKARIN_GLIBC_2_28, NLM_ISPC_GLIBC_2_28]
    assert "vsview" not in [dep.pypi_name for dep in dependencies]


@pytest.mark.parametrize(
    ("build", "wheel_name"),
    [
        (AKARIN_GLIBC_2_28, "vapoursynth_akarin"),
        (NLM_ISPC_GLIBC_2_28, "vapoursynth_nlm_ispc"),
    ],
)
def test_build_is_its_version_for_glibc_2_28_pinned_by_hash(
    build: Dependency, wheel_name: str
):
    assert build.url is not None
    location, _, fragment = build.url.partition("#")
    wheel = location.rsplit("/", 1)[1]
    name, version, _python, _abi, platforms = wheel.removesuffix(".whl").split("-")

    assert (name, version) == (wheel_name, build.version)
    assert name.replace("_", "-") == build.pypi_name
    assert f"/releases/download/v{version}-glibc2.28/" in location
    assert "manylinux_2_28_x86_64" in platforms.split(".")
    assert re.fullmatch(r"sha256=[0-9a-f]{64}", fragment)


@pytest.mark.parametrize("glibc", [None, (2, 34), (2, 35), (2, 39)])
def test_vsview_installs_from_glibc_2_34(glibc: tuple[int, int] | None):
    assert vsview_unavailable_reason(glibc) is None


@pytest.mark.parametrize("glibc", [(2, 28), (2, 31), (2, 33)])
def test_vsview_reason_names_the_found_glibc_and_the_floor(glibc: tuple[int, int]):
    reason = vsview_unavailable_reason(glibc)

    assert reason is not None
    assert f"has glibc {glibc[0]}.{glibc[1]}." in reason
    assert "glibc 2.34 or newer" in reason
    assert "The other VapourSynth nodes work" in reason
