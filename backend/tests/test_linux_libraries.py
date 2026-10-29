"""The table of system files that the launcher and the backend read, and the backend's
reading of it (the launcher's is tested in tests/main/linuxLauncher.test.ts)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import linux_libraries
import system
from linux_libraries import (
    TABLE,
    Entry,
    cached_libraries,
    install_command,
    is_present,
    package_manager,
    read_table,
)

LDCONFIG_OUTPUT = """\
3 libs found in cache `/etc/ld.so.cache'
\tlibGL.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libGL.so.1
\tlibGL.so.1 (libc6) => /lib/i386-linux-gnu/libGL.so.1
\tlibSM.so.6 (libc6,x86-64, OS ABI: Linux 3.2.0) => /usr/lib64/libSM.so.6
\tlibICE.so.6 (libc6) => /lib/i386-linux-gnu/libICE.so.6
"""


def test_table_rows_are_complete():
    entries = read_table(TABLE.read_text(encoding="utf-8"))

    assert {e.file for e in entries} >= {"libGL.so.1", "libstdc++.so.6", "xvfb-run"}
    for entry in entries:
        assert entry.groups <= {"base", "electron", "display"}, entry.file
        assert list(entry.packages) == list(linux_libraries.INSTALL_COMMANDS)
        assert ".so" in entry.file or entry.groups == {"display"}, entry.file
        for manager, package in entry.packages.items():
            if package.endswith("(t64)"):
                assert manager == "apt", entry.file


def test_read_table_skips_comments_and_blank_lines():
    entries = read_table("# a comment\n\nlibA.so.1  base,electron  a  b  c  d\n")

    assert entries == [
        Entry(
            "libA.so.1",
            frozenset({"base", "electron"}),
            {"apt": "a", "dnf": "b", "pacman": "c", "zypper": "d"},
        )
    ]


@pytest.mark.parametrize(
    ("release", "manager"),
    [
        ({"ID": "ubuntu", "ID_LIKE": "debian"}, "apt"),
        ({"ID": "debian"}, "apt"),
        ({"ID": "linuxmint", "ID_LIKE": "ubuntu debian"}, "apt"),
        ({"ID": "rocky", "ID_LIKE": "rhel centos fedora"}, "dnf"),
        ({"ID": "almalinux", "ID_LIKE": "rhel centos fedora"}, "dnf"),
        ({"ID": "fedora"}, "dnf"),
        ({"ID": "arch"}, "pacman"),
        ({"ID": "manjaro", "ID_LIKE": "arch"}, "pacman"),
        ({"ID": "opensuse-leap", "ID_LIKE": "suse opensuse"}, "zypper"),
        ({"ID": "opensuse-tumbleweed", "ID_LIKE": "suse opensuse"}, "zypper"),
        ({"ID": "gentoo"}, None),
        ({}, None),
    ],
)
def test_package_manager_of_the_family(release: dict[str, str], manager: str | None):
    assert package_manager(release) == manager


def test_cache_lists_64_bit_libraries_only():
    assert cached_libraries(LDCONFIG_OUTPUT) == {"libGL.so.1", "libSM.so.6"}


def test_library_is_present_in_the_cache_or_ld_library_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    (tmp_path / "libX.so.1").touch()
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{tmp_path}:")

    assert is_present("libGL.so.1", {"libGL.so.1"})
    assert is_present("libX.so.1", {"libGL.so.1"})
    assert not is_present("libY.so.1", {"libGL.so.1"})


def test_library_without_a_cache_is_looked_for_in_the_usual_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    (tmp_path / "libX.so.1").touch()
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setattr(linux_libraries, "STANDARD_DIRECTORIES", (str(tmp_path),))

    assert is_present("libX.so.1", None)
    assert not is_present("libY.so.1", None)


def entry(file: str, **packages: str) -> Entry:
    return Entry(file, frozenset({"base"}), packages)


@pytest.mark.parametrize(
    ("manager", "glibc", "is_root", "command"),
    [
        ("apt", (2, 35), False, "sudo apt-get install libgl1 libglib2.0-0"),
        ("apt", (2, 39), True, "apt-get install libgl1 libglib2.0-0t64"),
        ("apt", None, True, "apt-get install libgl1 libglib2.0-0"),
        ("dnf", (2, 39), False, "sudo dnf install libglvnd-glx glib2"),
        ("pacman", (2, 39), False, "sudo pacman -S --needed libglvnd glib2"),
        ("zypper", (2, 39), True, "zypper install libglvnd libglib-2_0-0"),
    ],
)
def test_install_command_of_the_family(
    manager: str, glibc: tuple[int, int] | None, is_root: bool, command: str
):
    missing = [
        entry(
            "libGL.so.1",
            apt="libgl1",
            dnf="libglvnd-glx",
            pacman="libglvnd",
            zypper="libglvnd",
        ),
        entry(
            "libglib-2.0.so.0",
            apt="libglib2.0-0(t64)",
            dnf="glib2",
            pacman="glib2",
            zypper="libglib-2_0-0",
        ),
    ]

    assert install_command(missing, manager, glibc, is_root) == command


def test_install_command_names_a_package_once():
    missing = [
        entry("libatk-bridge-2.0.so.0", pacman="at-spi2-core"),
        entry("libatspi.so.0", pacman="at-spi2-core"),
    ]

    assert (
        install_command(missing, "pacman", (2, 39), True)
        == "pacman -S --needed at-spi2-core"
    )


@pytest.mark.parametrize(
    ("confstr", "version"),
    [
        ("glibc 2.39", (2, 39)),
        ("glibc 2.17", (2, 17)),
        ("glibc 2.41.9000", (2, 41)),
        ("musl 1.2", None),
        (None, None),
    ],
)
def test_glibc_version(
    monkeypatch: pytest.MonkeyPatch,
    confstr: str | None,
    version: tuple[int, int] | None,
):
    monkeypatch.setattr(system, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(os, "confstr_names", {"CS_GNU_LIBC_VERSION": 2}, raising=False)
    monkeypatch.setattr(os, "confstr", lambda _name: confstr, raising=False)

    assert system.glibc_version() == version


def test_glibc_version_is_none_off_linux(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(system, "sys", SimpleNamespace(platform="win32"))

    assert system.glibc_version() is None
