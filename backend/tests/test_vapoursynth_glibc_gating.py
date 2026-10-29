"""The VapourSynth package is disabled on Linux systems with a glibc older than 2.35,
for which vapoursynth-akarin and PySide6 publish no wheels. Needs no VapourSynth."""

from __future__ import annotations

import pytest

from packages.chaiNNer_vapoursynth import _unsupported_reason


@pytest.mark.parametrize("glibc", [None, (2, 35), (2, 39), (3, 0)])
def test_supported_glibc_leaves_the_package_enabled(glibc: tuple[int, int] | None):
    assert _unsupported_reason(glibc) is None


@pytest.mark.parametrize("glibc", [(2, 17), (2, 28), (2, 31), (2, 34)])
def test_older_glibc_names_the_found_version_and_the_floor(glibc: tuple[int, int]):
    reason = _unsupported_reason(glibc)

    assert reason is not None
    assert f"glibc {glibc[0]}.{glibc[1]}" in reason
    assert "glibc 2.35 or newer" in reason
    assert "vapoursynth-akarin" in reason
    assert "Every other package works" in reason
