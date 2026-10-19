"""Load Images listings stay byte-identical on edge trees.

The walker takes a folder's identity from its parent's plus its name, calling realpath
only for the root and for reparse points. The oracle is the walk that rule replaced:
WCMatch 11.0.1 (the frozen source's list_glob) plus one visit per directory by
os.path.normcase(os.path.realpath()). Listings compare as (type, str) in order, so
equal-key ties (set order) and Windows' case-insensitive Path equality hide nothing.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from test_execution_ops import capture, error_key
from test_file_sequence import EXTENSIONS, namespace
from wcmatch import glob

from nodes.impl.image_formats import get_available_image_formats

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows")

LONG = "\\\\?\\"
EVERYONE = "*S-1-1-0"


class OracleGlob(glob.Glob[str]):
    """WCMatch 11.0.1's walk; a deep walk enters each directory once, by its
    normcase(realpath()), starting with its own."""

    def _identity(self, path: str) -> str:
        return os.path.normcase(os.path.realpath(self._prepend_base(path)))

    def _glob_dir(
        self,
        curdir: str,
        matcher: Callable[..., Any] | None,
        dir_only: bool = False,
        deep: bool = False,
        globstar_follow: bool = False,
        visited: set[str] | None = None,
    ) -> Iterator[tuple[str, bool]]:
        if deep and visited is None:
            visited = {self._identity(curdir)}
        files = list(self._iter(curdir, dir_only, deep))
        for file, is_dir, hidden, is_link in files:
            if file in self.specials:
                if matcher is not None and matcher(file):
                    yield os.path.join(curdir, file), True
                continue
            path = os.path.join(curdir, file)
            if (matcher is None and not hidden) or (matcher and matcher(file)):
                yield path, is_dir
            follow = not is_link or self.follow_links or globstar_follow
            if deep and not hidden and is_dir and follow:
                assert visited is not None
                key = self._identity(path)
                if key not in visited:
                    visited.add(key)
                    yield from self._glob_dir(
                        path, matcher, dir_only, deep, globstar_follow, visited
                    )


def oracle() -> dict[str, Any]:
    shim = types.SimpleNamespace(**vars(glob))
    shim.iglob = lambda patterns, *, flags=0, root_dir=None: OracleGlob(
        patterns, flags=flags, root_dir=root_dir
    ).glob()
    ns = namespace("source")
    ns["glob"] = shim
    return ns


def listing(ns, directory, expression, extensions):
    value, error = capture(lambda: ns["list_glob"](directory, expression, extensions))
    if value is None:
        return None, error_key(error)
    return [(type(path), str(path)) for path in value], None


def write(path: Path, data: bytes = b"not decoded") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def junction(link: Path, target: Path) -> None:
    import _winapi

    _winapi.CreateJunction(str(target), str(link))


def attributes(path: Path, value: int) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel32.SetFileAttributesW(str(path), value):
        raise ctypes.WinError(ctypes.get_last_error())


def icacls(path: Path, *arguments: str) -> None:
    subprocess.run(["icacls", str(path), *arguments], check=True, capture_output=True)


class EdgeTree:
    """Builds the edge tree; remembers what needs more than rmtree to remove."""

    def __init__(self, base: Path):
        self.base = base
        self.root = base / "images"
        # Folders whose names Windows rewrites in a path (os.scandir lists them by the
        # rewritten path): the native listing leaves such a tree to the WCMatch walker.
        self.odd = base / "odd"
        self.links: list[Path] = []
        self.locked: list[Path] = []
        self.symlinks = True
        self.case_sensitive = True

    def build(self) -> None:
        root, base = self.root, self.base
        for name in [
            "a.png",
            "B.PNG",
            "c.Jpg",
            "d.TIFF",
            "e.WebP",
            "f.txt",
            "noext",
            ".hidden.png",
            ".png",
            "image1.png",
            "image01.png",
            "image001.png",
            "IMAGE1.PNG.png",
            "世界٣.png",
            "Straße2.png",
            "STRASSE02.png",
            "\u0661.png",
            "①.png",
            "k.png",
            "\u212a.png",  # KELVIN SIGN: lowercases to "k.png"
            "ω.png",
            "\u2126.png",  # OHM SIGN: lowercases to "ω.png"
            "x.t\u0130f",  # re's IGNORECASE matches U+0130 and U+0131 to i
            "y.t\u0131f",
            "z.\u017fgi",  # and U+017F to s
            "\ud800x.png",  # a lone surrogate
            "\U0001d7ce42.png",  # a decimal digit outside the BMP
            "😀1.png",
            "n" * 240 + ".png",
            "folder.png/inside.png",
            "sub/b.png",
            "sub/deeper/c.png",
            "z_real/r.png",
            "hs/x.png",
        ]:
            write(root / name)
        (root / "empty").mkdir()
        (root / "empty_nested" / "inner").mkdir(parents=True)
        for name in ["hidden_attr.png", "system_attr.png"]:
            write(root / name)
        attributes(root / "hidden_attr.png", 0x2)
        attributes(root / "system_attr.png", 0x4)
        attributes(root / "hs", 0x2 | 0x4 | 0x10)
        # Deep and long: past the classic MAX_PATH, with junctions in and out.
        deep = root / ("L" * 100) / ("M" * 100) / ("N" * 100)
        write(deep / "deep.png")
        write(deep / ("O" * 100) / "deeper.png")
        # Names Windows rewrites in a path: trailing dots and spaces, device names,
        # beside the plain names they come to.
        write(self.odd / "a.png")
        for name in [
            "dot",
            "dot.",
            "dots..",
            "dot. .",
            "space",
            "space ",
            "con",
            "nul.d",
        ]:
            write(Path(LONG + str(self.odd / name)) / f"in_{name}.png")
        # Junctions: a cycle to the root, back to an ancestor, a second way into a
        # folder, one sorted before the folder it targets, a chain, one outside the
        # tree whose target holds a junction back in, a dangling one, long ones.
        other = base / "other"
        write(other / "d.png")
        write(base / "chain_target" / "t.png")
        locked = base / "locked_target"
        write(locked / "l.png")
        links = [
            (root / "loop", root),
            (root / "sub" / "deeper" / "up", root / "sub"),
            (root / "again", root / "sub"),
            (root / "a_first", root / "z_real"),
            (base / "chain_middle", base / "chain_target"),
            (root / "chain", base / "chain_middle"),
            (root / "outside", other),
            (other / "inner", root / "sub"),
            (root / "dangling", base / "missing"),
            (root / "into_long", deep),
            (deep / "back", root / "sub"),
            (root / "to_locked", locked),
            (self.odd / "back", root / "sub"),
        ]
        (base / "missing").mkdir()  # CreateJunction needs its target to exist
        for link, target in links:
            junction(link, target)
            self.links.append(link)
        (base / "missing").rmdir()
        # Inaccessible: a folder nobody may list, directly and through a junction (only
        # listing is denied, so the owner can still give it back).
        write(root / "locked" / "x.png")
        for folder in (root / "locked", locked):
            icacls(folder, "/deny", f"{EVERYONE}:(RD)")
            self.locked.append(folder)
        write(base / "elsewhere" / "e.png")
        try:
            for link, target, is_dir in [
                (root / "file_link.png", root / "a.png", False),
                (root / "dir_link.png", root / "sub", True),
                (root / "outside_link", base / "elsewhere", True),  # never followed
                (root / "dangling_link.png", root / "missing.png", False),
                (root / "loop_a.png", root / "loop_b.png", False),
                (root / "loop_b.png", root / "loop_a.png", False),
            ]:
                os.symlink(target, link, target_is_directory=is_dir)
                self.links.append(link)
        except OSError:
            self.symlinks = False
        sensitive = root / "cs"
        sensitive.mkdir()
        result = subprocess.run(
            ["fsutil", "file", "setCaseSensitiveInfo", str(sensitive), "enable"],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            for name in ["a.png", "A.png", "Q.PNG", "q.png", "dir/x.png", "DIR/x.png"]:
                write(sensitive / name)
        else:
            self.case_sensitive = False

    def remove(self) -> None:
        for folder in self.locked:
            icacls(folder, "/remove:d", EVERYONE)
        for link in self.links:
            if link.is_dir() or link.is_junction():
                os.rmdir(link)  # the link only, never its target
            else:
                link.unlink(missing_ok=True)
        # The literal path form reaches the names Windows would rewrite.
        shutil.rmtree(LONG + str(self.base))


@pytest.fixture(scope="module")
def edge(tmp_path_factory) -> Iterator[EdgeTree]:
    tree = EdgeTree(tmp_path_factory.mktemp("edge"))
    try:
        tree.build()
        yield tree
    finally:
        tree.remove()


PATTERNS = [
    "*",
    "**/*",
    "**/*.png",
    "**",
    "**/",
    "*/**/*.png",
    "sub/**/*",
    "**/deeper/*",
    "{*,**/*}",
]


def directories(edge: EdgeTree) -> list[Path]:
    root = edge.root
    return [
        root,
        Path(LONG + str(root)),
        root / "sub",
        root / "outside",
        root / "into_long",
        root / "locked",
        root / "missing",
        root / "a.png",
        root / "cs",
        edge.odd,
        Path(LONG + str(edge.odd)),
        edge.odd / "space ",
        edge.odd / "dots..",
    ]


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("formats", ["fixture", "available"])
def test_listing_matches_the_one_visit_walk(edge, pattern, formats):
    extensions = EXTENSIONS if formats == "fixture" else get_available_image_formats()
    old, new = oracle(), namespace(current=True)
    for directory in directories(edge):
        expected = listing(old, directory, pattern, extensions)
        assert listing(new, directory, pattern, extensions) == expected, directory


def test_edge_tree_exercises_its_edges(edge):
    extensions = get_available_image_formats()
    paths, error = listing(namespace(current=True), edge.root, "**/*", extensions)
    assert error is None and paths is not None
    names = [Path(text).relative_to(edge.root).as_posix() for _, text in paths]
    # Entered once: sub (the cycle, a second way in, a junction back from outside),
    # z_real (through the junction listed before it) and the long folder.
    assert [name for name in names if name.endswith("/b.png")] == ["again/b.png"]
    assert "a_first/r.png" in names and "z_real/r.png" not in names
    assert sum(name.endswith("/deeper.png") for name in names) == 1
    assert "outside/d.png" in names and "chain/t.png" in names
    # Python's lower() folds KELVIN SIGN and OHM SIGN onto names listed before them.
    assert "\u212a.png" not in names and "\u2126.png" not in names
    assert {"x.t\u0130f", "y.t\u0131f", "z.\u017fgi", "\ud800x.png"} <= set(names)
    assert {"hidden_attr.png", "system_attr.png", "hs/x.png", ".png"} <= set(names)
    assert "locked/x.png" not in names and "to_locked/l.png" not in names
    if edge.symlinks:
        assert {"file_link.png", "dir_link.png", "dangling_link.png"} <= set(names)
    if edge.case_sensitive:
        assert {"cs/a.png", "cs/A.png"} <= set(names)


@pytest.mark.parametrize("pattern", ["*", "**/*"])
def test_relative_and_text_directories(edge, monkeypatch, pattern):
    old, new = oracle(), namespace(current=True)
    for cwd, directory in [
        (edge.base, Path("images")),
        (edge.root, Path(".")),
        (edge.root / "sub", Path("..")),
        (edge.base, str(edge.root)),
    ]:
        monkeypatch.chdir(cwd)
        expected = listing(old, directory, pattern, EXTENSIONS)
        assert listing(new, directory, pattern, EXTENSIONS) == expected, directory


@pytest.mark.parametrize("pattern", ["*", "**/*"])
def test_plain_listing_is_native(edge, monkeypatch, pattern):
    """Load Images' own expressions list without os.scandir (FindFirstFileExW in C++);
    a walk that would list a folder os.scandir lists by a rewritten path goes to the
    WCMatch walker."""
    new = namespace(current=True)
    expected = [
        listing(oracle(), root, pattern, EXTENSIONS) for root in (edge.root, edge.odd)
    ]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("os.scandir called")

    monkeypatch.setattr(os, "scandir", forbidden)
    assert listing(new, edge.root, pattern, EXTENSIONS) == expected[0]
    if pattern == "*":
        assert listing(new, edge.odd, pattern, EXTENSIONS) == expected[1]
    else:
        with pytest.raises(AssertionError, match=r"os\.scandir called"):
            new["list_glob"](edge.odd, pattern, EXTENSIONS)


@pytest.mark.parametrize("recursive", [False, True])
def test_load_images_snapshot_matches(edge, recursive):
    ns = namespace(current=True)
    generator, _ = ns["load_images_node"](
        edge.root, False, recursive, "", False, 0, False
    )
    expected, _ = listing(oracle(), edge.root, "**/*" if recursive else "*", EXTENSIONS)
    assert [(type(p), str(p)) for p in generator.source_paths] == expected
