"""Exact Load Images discovery/state against independently frozen source.

WCMatch 11.0.1 grammar compilation remains explicit; native walking/matching,
sorting and iterator state must work with the original walkers disabled.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_execution_ops import (
    capture,
    declarations,
    equivalent,
    error_key,
    frozen_api,
)
from wcmatch import glob

from nodes.impl.native_graph import graph

ROOT = Path(__file__).resolve().parents[2]
FROZEN = Path(__file__).with_name("reference_file_sequence")
NODE = "packages/chaiNNer_standard/image/batch_processing/load_images.py"
EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".webp"]


def namespace(
    variant="installed", current=False, loader=None, extensions=None
) -> dict[str, Any]:
    base = ROOT / "backend/src" if current else FROZEN / variant
    helper = declarations(
        FROZEN / variant / "nodes/utils/utils.py",
        {"alphanumeric_sort"},
        {"NUMBERS": re.compile(r"(\d+)")},
    )["alphanumeric_sort"]

    def default_loader(p):
        return (str(p), p.parent, p.stem)

    return declarations(
        base / NODE,
        {"extension_filter", "list_glob", "load_images_node"},
        {
            "Path": Path,
            "os": os,
            "glob": glob,
            "graph": graph,
            "alphanumeric_sort": helper,
            "Generator": frozen_api(variant).Generator,
            "load_image_node": default_loader if loader is None else loader,
            "get_available_image_formats": lambda: (
                EXTENSIONS if extensions is None else extensions
            ),
        },
    )


@pytest.fixture
def tree(tmp_path):
    names = [
        "image2.png",
        "image10.png",
        "image01.png",
        "image1.png",
        "image001.PNG",
        "IMAGE12.JPG",
        ".hidden.png",
        "space image3.tif",
        "a[1].png",
        "literal{a}.png",
        "世界٣.png",
        "世界12.png",
        "Straße2.png",
        "STRASSE02.png",
        "①.png",
        "\u0661.png",
        "1.png",
        "2.png",
        "3.png",
        "4.png",
        "plain.txt",
        "no_extension",
        "folder/child1.png",
        "folder/child20.jpg",
        "folder/.child3.png",
        "folder/deep/child02.png",
        ".private/hidden4.png",
        "folder.png/inside.png",
        "other/x.webp",
    ]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"known fixture; decoding deliberately stubbed")
    (tmp_path / "empty").mkdir()
    return tmp_path


PATTERNS = [
    "*",
    "**/*",
    "*.png",
    "**/*.png",
    "**",
    "folder/**",
    "folder/**/*",
    "folder/",
    "folder/*/",
    "folder/*/*.png",
    "**/child*.png",
    "**/.*",
    ".*/*",
    ".hidden.png",
    "./*.png",
    "./**/*.png",
    "folder/../*.png",
    "image?.png",
    "image[0-9].png",
    "image[!0-9].png",
    "[[:digit:]].png",
    "@(image1|image2|image10).png",
    "+(image1|image2).png",
    "?(image)1.png",
    "*(image)1.png",
    "!(image2).png",
    "**/!(*20).@(png|jpg)",
    "{image1,image2,image10}.png",
    "image{01,1,001}.png",
    "{1..4}.png",
    "{01..04}.png",
    "{4..1}.png",
    "{a,b}{1,2}.png",
    "{folder,other}/*",
    "{image1.png,folder/*,**/*.png}",
    "{**/*.png,!**/child*}",
    "!**/child*",
    "!*.txt",
    "{*.png,*.png}",
    "a[[]1].png",
    r"folder\*.png",
    "missing/*",
    "",
    "[]",
    "[",
    "@(",
    "**/*.{png,JPG,jpeg}",
    "**/世界*.png",
    "**/Straße*.png",
]


@pytest.mark.parametrize("variant", ["installed", "source"])
@pytest.mark.parametrize("pattern", PATTERNS)
def test_full_grammar_discovery_exact(tree, variant, pattern):
    results = []
    for current in (False, True):
        ns = namespace(variant, current)
        results.append(capture(partial(ns["list_glob"], tree, pattern, EXTENSIONS)))
    equivalent(*results)


@pytest.mark.parametrize("variant", ["installed", "source"])
def test_absolute_patterns_and_pattern_lists(tree, variant):
    patterns = [
        str(tree / "folder" / "*.png"),
        str(tree / "**" / "*.png"),
        ["*.png", "folder/*", "!image2.png"],
        ["folder/*", "**/*.png"],
        ["*.png", "*.png"],
        [],
        (),
        ["!**/child*"],
    ]
    for pattern in patterns:
        equivalent(
            *(
                capture(
                    partial(
                        namespace(variant, current)["list_glob"],
                        tree,
                        pattern,
                        EXTENSIONS,
                    )
                )
                for current in (False, True)
            )
        )


@pytest.mark.parametrize(
    "extensions",
    [
        [],
        [".PNG"],
        [".png", ".png"],
        [".png", ".jpg"],
        [".*"],
        [".@(png|jpg)"],
        [".p?g"],
        [".txt"],
        [None],
        None,
    ],
)
def test_extension_expression_and_filter_edges(tree, extensions):
    old, new = namespace(), namespace(current=True)
    equivalent(
        capture(lambda: old["extension_filter"](extensions)),
        capture(lambda: new["extension_filter"](extensions)),
    )
    equivalent(
        capture(lambda: old["list_glob"](tree, "**/*", extensions)),
        capture(lambda: new["list_glob"](tree, "**/*", extensions)),
    )


@pytest.mark.parametrize("pattern", [None, 7, b"*.png", "{1..1001}.png", [None], [7]])
def test_pattern_errors(tree, pattern):
    old, new = namespace(), namespace(current=True)
    equivalent(
        capture(lambda: old["list_glob"](tree, pattern, EXTENSIONS)),
        capture(lambda: new["list_glob"](tree, pattern, EXTENSIONS)),
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "0001",
        "a0001b23",
        "Straße12世界٣",
        "\uff11٢3",
        "a²3",
        "\U0001d7ce42",
        "iİ\u0131i\N{COMBINING DOT ABOVE}12",
        "a\x00b2",
        "\ud8002",
        "12" * 2100,
    ],
)
def test_natural_key_unicode_integer_exact(text):
    old = namespace()["alphanumeric_sort"]
    equivalent(
        capture(lambda: old(text)),
        capture(lambda: graph().file_sequence_alphanumeric(text)),
    )


def test_natural_key_digit_limit_error():
    text = "x" + "1" * 5000
    equivalent(
        capture(lambda: namespace()["alphanumeric_sort"](text)),
        capture(lambda: graph().file_sequence_alphanumeric(text)),
    )


def test_native_walk_does_not_delegate_to_old_algorithm(tree, monkeypatch):
    old, new = namespace(), namespace(current=True)
    expected = old["list_glob"](tree, "{**/*.png,!**/child*}", EXTENSIONS)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("old Python walking/filtering/natural algorithm executed")

    for method in (
        "glob",
        "_iter",
        "_glob_dir",
        "_glob",
        "_get_starting_paths",
        "_format_path",
        "_is_unique",
        "_get_matcher",
        "_match_excluded",
        "_is_excluded",
    ):
        monkeypatch.setattr(glob.Glob, method, forbidden)
    monkeypatch.setattr(glob, "iglob", forbidden)
    monkeypatch.setattr(glob, "globfilter", forbidden)
    new["alphanumeric_sort"] = forbidden
    assert new["list_glob"](tree, "{**/*.png,!**/child*}", EXTENSIONS) == expected


@pytest.mark.parametrize("variant", ["installed", "source"])
@pytest.mark.parametrize(
    "use_glob,recursive,use_limit,fail_fast", itertools.product([False, True], repeat=4)
)
@pytest.mark.parametrize("limit", [0, 1, 4, 100, -1])
def test_node_generator_options(
    tree, variant, use_glob, recursive, use_limit, fail_fast, limit
):
    results = []
    for current in (False, True):
        ns = namespace(variant, current)
        generator, directory = ns["load_images_node"](
            tree, use_glob, recursive, "**/child*.png", use_limit, limit, fail_fast
        )
        assert directory is tree
        assert generator.fail_fast is fail_fast and generator.metadata is None
        first, second = generator.supplier(), generator.supplier()
        assert first is not second and iter(first) is first
        values = list(first)
        assert list(second) == values and list(first) == []
        results.append((generator.expected_length, values))
    assert results[0] == results[1]


@pytest.mark.parametrize("limit", [0, -1, 2, None, 1.5])
def test_empty_discovery_precedes_limit_validation(tmp_path, limit):
    equivalent(
        *(
            capture(
                partial(
                    namespace(current=current)["load_images_node"],
                    tmp_path,
                    False,
                    False,
                    "*",
                    True,
                    limit,
                    False,
                )
            )
            for current in (False, True)
        )
    )


@pytest.mark.parametrize(
    "error_type",
    [ValueError, FileNotFoundError, StopIteration, KeyboardInterrupt, SystemExit],
)
def test_lazy_errors_and_closed_state(tree, error_type):
    outcomes = []
    for current in (False, True):
        calls = []

        def loader(path, calls=calls):
            calls.append(path)
            if len(calls) == 1:
                raise error_type("first file")
            return str(path), path.parent, path.stem

        ns = namespace(current=current, loader=loader)
        generator, _ = ns["load_images_node"](tree, False, False, "*", True, 3, False)
        assert calls == []
        iterator = generator.supplier()
        assert calls == []
        first, thrown = capture(partial(next, iterator))
        rest = list(iterator)
        if issubclass(error_type, Exception):
            assert thrown is None and isinstance(first, error_type)
            outcomes.append((error_key(first), rest, len(calls)))
        else:
            assert first is None and isinstance(thrown, error_type) and rest == []
            outcomes.append((error_key(thrown), rest, len(calls)))
    assert outcomes[0] == outcomes[1]


def test_discovery_snapshot_contents_stay_lazy(tmp_path):
    outcomes = []
    for current in (False, True):
        folder = tmp_path / str(current)
        folder.mkdir()
        first, second = folder / "1.png", folder / "2.png"
        first.write_text("before")
        second.write_text("removed later")

        def loader(path):
            return path.read_text(), path.parent, path.stem

        ns = namespace(current=current, loader=loader)
        generator, _ = ns["load_images_node"](
            folder, False, False, "*", False, 0, False
        )
        first.write_text("after")
        second.unlink()
        (folder / "3.png").write_text("not in snapshot")
        values = list(generator.supplier())
        assert generator.expected_length == 2
        assert values[0] == ("after", ".", "1", 0)
        assert isinstance(values[1], FileNotFoundError)
        outcomes.append(type(values[1]))
    assert outcomes[0] is outcomes[1]


@pytest.mark.parametrize("value", [None, (), (1,), (1, 2), (1, 2, 3, 4)])
def test_loader_unpack_errors_are_yielded(tree, value):
    outcomes = []
    for current in (False, True):
        ns = namespace(current=current, loader=lambda _: value)
        generator, _ = ns["load_images_node"](tree, False, False, "*", True, 1, False)
        result = next(generator.supplier())
        assert isinstance(result, Exception)
        outcomes.append(error_key(result))
    assert outcomes[0] == outcomes[1]


def test_missing_root_and_directory_with_image_suffix(tree):
    for directory in [tree / "missing", tree / "plain.txt", tree]:
        equivalent(
            *(
                capture(
                    partial(
                        namespace(current=current)["list_glob"],
                        directory,
                        "*",
                        EXTENSIONS,
                    )
                )
                for current in (False, True)
            )
        )
    assert tree / "folder.png" in namespace(current=True)["list_glob"](
        tree, "*", EXTENSIONS
    )


def test_independent_concurrent_discovery_and_iteration(tree):
    old, new = namespace(), namespace(current=True)
    expected = old["list_glob"](tree, "**/*", EXTENSIONS)

    def run(_):
        paths = new["list_glob"](tree, "**/*", EXTENSIONS)
        generator, _ = new["load_images_node"](tree, False, True, "", False, 0, False)
        return paths, list(generator.supplier())

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run, range(12)))
    for paths, values in results:
        assert paths == expected
        assert len(values) == len(expected)
        assert [x[3] for x in values] == list(range(len(values)))


def test_walk_preserves_scandir_order(tree, monkeypatch):
    scan = os.scandir
    observations = []
    for current in (False, True):
        events = []

        def logged(path, events=events):
            events.append(str(path))
            return scan(path)

        with monkeypatch.context() as patch:
            patch.setattr(os, "scandir", logged)
            result = namespace(current=current)["list_glob"](
                tree, "**/deep/*.png", EXTENSIONS
            )
        observations.append((result, events))
    assert observations[0] == observations[1]


@pytest.mark.parametrize("action", ["next", "close"])
def test_lazy_iterator_reentry_matches_generator(tree, action):
    outcomes = []
    for current in (False, True):
        state = {}

        def loader(path, state=state):
            if action == "next":
                next(state["iterator"])
            else:
                state["iterator"].close()
            return str(path), path.parent, path.stem

        ns = namespace(current=current, loader=loader)
        generator, _ = ns["load_images_node"](tree, False, False, "*", True, 1, False)
        iterator = generator.supplier()
        state["iterator"] = iterator
        value, error = capture(partial(next, iterator))
        outcomes.append(
            (
                error_key(value) if isinstance(value, Exception) else value,
                error_key(error),
            )
        )
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("variant", ["source", "installed"])
def test_frozen_source_and_public_metadata(variant):
    relative = f"{variant}/{NODE}"
    manifest = json.loads((FROZEN / "sources.json").read_text())
    assert (
        hashlib.sha256((FROZEN / relative).read_bytes()).hexdigest()
        == manifest["sha256"][relative]
    )
    old = ast.parse((FROZEN / relative).read_text())
    new = ast.parse((ROOT / "backend/src" / NODE).read_text())
    for function in ("extension_filter", "list_glob", "load_images_node"):
        a = next(
            n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == function
        )
        b = next(
            n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == function
        )
        assert ast.dump(a.args) == ast.dump(b.args)
        assert a.returns is not None and b.returns is not None
        assert ast.dump(a.returns) == ast.dump(b.returns)
        assert [ast.dump(x) for x in a.decorator_list] == [
            ast.dump(x) for x in b.decorator_list
        ]


@pytest.mark.parametrize("use_limit", [False, True])
def test_load_images_declares_its_post_limit_snapshot(tree, use_limit):
    ns = namespace(current=True)
    generator, _ = ns["load_images_node"](tree, False, True, "", use_limit, 4, False)
    discovered = ns["list_glob"](tree, "**/*", EXTENSIONS)
    assert generator.source_paths == tuple(discovered[:4] if use_limit else discovered)
    assert generator.expected_length == len(generator.source_paths)


# SP3b iterator split: describe() advances the cursor under the reentry guard and
# returns (path, index); materialize(token) loads that item with no guard and no
# iterator state. __next__ is both phases under one guard.


def image_folder(folder, names):
    for name in names:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"pixels of {name}".encode())


def recursive_images(ns, folder):
    generator, _ = ns["load_images_node"](folder, False, True, "", False, 0, False)
    return generator


def described_items(iterator):
    items = []
    while True:
        try:
            token = iterator.describe()
        except StopIteration:
            return items
        items.append(iterator.materialize(token))


def item_key(value):
    if isinstance(value, Exception):
        return type(value), str(value)
    image, relative, basename, index = value
    return image.dtype, image.shape, image.tobytes(), relative, basename, index


def test_describe_then_materialize_equals_next(tmp_path):
    image_folder(tmp_path, ["1.png", "2.png", "3.png", "sub/4.png", "sub/5.png"])

    def loader(path):
        if path.name == "2.png":
            raise ValueError(f"cannot decode {path.name}")
        return np.frombuffer(path.read_bytes(), np.uint8), path.parent, path.stem

    generator = recursive_images(namespace(current=True, loader=loader), tmp_path)
    token = generator.supplier().describe()
    assert type(token) is tuple and token == (generator.source_paths[0], 0)
    assert isinstance(token[0], Path) and type(token[1]) is int
    split = [item_key(x) for x in described_items(generator.supplier())]
    plain = [item_key(x) for x in generator.supplier()]
    installed = recursive_images(namespace(loader=loader), tmp_path)
    assert split == plain == [item_key(x) for x in installed.supplier()]
    assert split[1] == (ValueError, "cannot decode 2.png")
    assert [x[3:] for x in split if len(x) == 6] == [
        (".", "1", 0),
        (".", "3", 2),
        ("sub", "4", 3),
        ("sub", "5", 4),
    ]


class FailingWalk:
    """Fails once at the second path, then would go on with the rest."""

    def __init__(self, paths):
        self.paths = paths
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.position += 1
        if self.position == 2:
            raise OSError("listing failed")
        return next(self.paths)


class FailingListing(list):
    armed = False

    def __iter__(self):
        walk = super().__iter__()
        return FailingWalk(walk) if self.armed else walk


def test_describe_raises_stop_iteration_and_closes_at_end(tmp_path):
    image_folder(tmp_path, ["1.png", "2.png", "3.png"])
    calls = []

    def loader(path):
        calls.append(path)
        return str(path), path.parent, path.stem

    ns = namespace(current=True, loader=loader)
    generator = recursive_images(ns, tmp_path)
    paths = generator.source_paths
    iterator = generator.supplier()
    # describe() and next() advance one cursor and one index.
    assert iterator.describe() == (paths[0], 0)
    assert next(iterator) == (str(paths[1]), ".", "2", 1)
    assert iterator.describe() == (paths[2], 2)
    for _ in range(2):
        with pytest.raises(StopIteration):
            iterator.describe()
    assert list(iterator) == [] and calls == [paths[1]]
    closed = generator.supplier()
    closed.close()
    with pytest.raises(StopIteration):
        closed.describe()

    # An error while advancing closes the iterator, for describe() as for next():
    # the walk would otherwise go on with the third path.
    snapshot = FailingListing(ns["list_glob"](tmp_path, "**/*", EXTENSIONS))
    ns["list_glob"] = lambda *_: snapshot
    failing = recursive_images(ns, tmp_path)
    snapshot.armed = True
    for advance in (lambda it: it.describe(), next):
        iterator = failing.supplier()
        advance(iterator)
        with pytest.raises(OSError, match="listing failed"):
            advance(iterator)
        with pytest.raises(StopIteration):
            advance(iterator)
    assert calls == [paths[1], paths[0]]


def test_materialize_calls_load_image_node_once_per_token(monkeypatch, tmp_path):
    image_folder(tmp_path, ["1.png", "2.png"])
    calls = []

    def loader(path):
        calls.append(("loader", path))
        return str(path), path.parent, path.stem

    def replacement(path):
        calls.append(("replacement", path))
        return str(path), path.parent, path.stem

    ns = namespace(current=True, loader=loader)
    generator = recursive_images(ns, tmp_path)
    paths = generator.source_paths
    iterator = generator.supplier()
    first, second = iterator.describe(), iterator.describe()
    assert calls == []
    assert iterator.materialize(second) == (str(paths[1]), ".", "2", 1)
    assert calls == [("loader", paths[1])]
    # Looked up per call, so a monkeypatched load_image_node takes over.
    monkeypatch.setitem(ns, "load_image_node", replacement)
    for _ in range(2):
        assert iterator.materialize(first) == (str(paths[0]), ".", "1", 0)
    assert calls == [("loader", paths[1])] + [("replacement", paths[0])] * 2
    # A malformed token is the caller's error, raised rather than yielded.
    with pytest.raises(ValueError, match="not enough values to unpack"):
        iterator.materialize((paths[0],))
    assert len(calls) == 3


def test_materialize_runs_outside_the_guard(monkeypatch, tmp_path):
    image_folder(tmp_path, ["1.png", "2.png", "3.png"])
    ns = namespace(current=True)
    generator = recursive_images(ns, tmp_path)
    paths = generator.source_paths
    iterator = generator.supplier()
    inner = []

    def reentrant(path):
        inner.append(iterator.describe())
        return str(path), path.parent, path.stem

    monkeypatch.setitem(ns, "load_image_node", reentrant)
    assert iterator.materialize(iterator.describe()) == (str(paths[0]), ".", "1", 0)
    assert inner == [(paths[1], 1)]
    # __next__ holds the guard across both phases, as a generator does.
    value = next(iterator)
    assert isinstance(value, ValueError)
    assert str(value) == "generator already executing"
    assert inner == [(paths[1], 1)]


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_materialize_never_closes_the_iterator(monkeypatch, tmp_path, error_type):
    image_folder(tmp_path, ["1.png", "2.png", "3.png"])
    ns = namespace(current=True)
    generator = recursive_images(ns, tmp_path)
    paths = generator.source_paths
    iterator = generator.supplier()
    token = iterator.describe()

    def interrupted(path):
        raise error_type(f"decode of {path.name} interrupted")

    with monkeypatch.context() as patch:
        patch.setitem(ns, "load_image_node", interrupted)
        with pytest.raises(error_type, match=r"decode of 1\.png interrupted"):
            iterator.materialize(token)
    first = (str(paths[0]), ".", "1", 0)
    assert iterator.describe() == (paths[1], 1)
    assert next(iterator) == (str(paths[2]), ".", "3", 2)
    assert iterator.materialize(token) == first
    # Neither the end nor close() reaches materialize: a run's cleanup closes the
    # iterator while materializations may still be running.
    with pytest.raises(StopIteration):
        iterator.describe()
    assert iterator.materialize(token) == first
    iterator.close()
    assert iterator.materialize(token) == first
    assert iterator.materialize((paths[2], 2)) == (str(paths[2]), ".", "3", 2)
    with pytest.raises(StopIteration):
        next(iterator)


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_next_still_closes_on_a_non_exception(monkeypatch, tmp_path, error_type):
    image_folder(tmp_path, ["1.png", "2.png"])
    ns = namespace(current=True)
    iterator = recursive_images(ns, tmp_path).supplier()

    def interrupted(path):
        raise error_type(f"decode of {path.name} interrupted")

    with monkeypatch.context() as patch:
        patch.setitem(ns, "load_image_node", interrupted)
        with pytest.raises(error_type, match=r"decode of 1\.png interrupted"):
            next(iterator)
    with pytest.raises(StopIteration):
        iterator.describe()
    assert list(iterator) == []
