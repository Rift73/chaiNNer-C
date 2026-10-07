"""Exact installed tile plans, callbacks, retries and pixels."""

from __future__ import annotations

import importlib
import itertools
import sys
import types
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from reference_buffers import tile_blending as old_blend

from nodes.impl.upscale import tile_blending as new_blend

ROOT = Path(__file__).resolve().parents[2]
FROZEN = Path(__file__).with_name("reference_tiling")


def load(name, directory, blend):
    package = types.ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package
    sys.modules[name + ".tile_blending"] = blend
    return tuple(
        importlib.import_module(name + "." + module)
        for module in ["auto_split", "exact_split", "tiler"]
    )


OLD = load("nodes.impl._tiling_reference", FROZEN / "installed", old_blend)
NEW = load(
    "nodes.impl._tiling_converted", ROOT / "backend/src/nodes/impl/upscale", new_blend
)


def state(value):
    if isinstance(value, np.ndarray):
        return (value.shape, value.dtype.str, value.strides, value.tobytes())
    if isinstance(value, (list, tuple)):
        return [state(v) for v in value]
    if hasattr(value, "__dict__"):
        return (type(value).__name__, {k: state(v) for k, v in vars(value).items()})
    return value


def outcome(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = ("ok", state(call()))
        except Exception as error:
            context = error.__context__
            value = (
                type(error).__name__,
                str(error),
                error.args,
                (type(context).__name__, str(context), context.args)
                if context
                else None,
            )
    return value, [(type(w.message).__name__, str(w.message)) for w in caught]


@pytest.mark.parametrize(
    "length,exact,overlap",
    itertools.product(
        [0, 1, 7, 16, 17, 32, 39, 127, 513], [0, 1, 8, 16, 32], [0, 1, 3, 7, 8, 16]
    ),
)
def test_segment_geometry(length, exact, overlap):
    assert outcome(
        lambda: NEW[1]._exact_split_into_segments(length, exact, overlap)
    ) == outcome(lambda: OLD[1]._exact_split_into_segments(length, exact, overlap))


@pytest.mark.parametrize(
    "dimensions",
    [
        (31, 39, 16, 16, 3),
        (31, 39, 24, 16, 3),
        (16, 16, 16, 16, 0),
        (7, 17, 8, 8, 1),
        (17, 7, 8, 8, 1),
        (32, 31, 16, 16, 8),
    ],
)
def test_region_geometry(dimensions):
    assert outcome(lambda: NEW[1]._exact_split_into_regions(*dimensions)) == outcome(
        lambda: OLD[1]._exact_split_into_regions(*dimensions)
    )


def run(
    modules,
    image,
    strategy,
    tile_size,
    overlap,
    scale,
    split_at=(),
    fail=False,
):
    auto, _, tiler = modules
    events = []
    calls = 0

    def upscale(tile, region):
        nonlocal calls
        events.append(
            ("tile", state(region), tile.shape, tile.dtype.str, tile.tobytes())
        )
        calls += 1
        if calls in split_at:
            return auto.Split()
        if fail:
            raise RuntimeError("inference failed")
        return np.repeat(np.repeat(tile, scale, axis=0), scale, axis=1)

    if strategy == "exact":
        selected = tiler.ExactTileSize(tile_size)
    elif strategy == "max":
        selected = tiler.MaxTileSize(tile_size[0])
    else:
        selected = tiler.NoTiling()
    result = outcome(lambda: auto.auto_split(image, upscale, selected, overlap))
    return result, events


@pytest.mark.parametrize("shape", [(5, 7), (5, 7, 1), (19, 23, 3), (31, 37, 4)])
@pytest.mark.parametrize("strategy", ["none", "exact", "max"])
@pytest.mark.parametrize("overlap", [0, 1, 3])
@pytest.mark.parametrize("scale", [1, 2, 3])
@pytest.mark.parametrize("layout", ["C", "F", "reverse"])
def test_complete_plans_and_pixels(shape, strategy, overlap, scale, layout):
    image = np.random.default_rng(8).random(shape, dtype=np.float32)
    if layout == "F":
        image = np.asfortranarray(image)
    elif layout == "reverse":
        image = image[::-1, ::-1]
    args = (image, strategy, (16, 16), overlap, scale)
    assert run(NEW, *args) == run(OLD, *args)


@pytest.mark.parametrize("shape", [(15, 19, 3), (67, 79, 3), (99, 131, 4)])
@pytest.mark.parametrize("split_at", [(1,), (2,), (5,), (1, 2), (1, 3), (3, 7)])
@pytest.mark.parametrize("overlap", [0, 1, 4])
def test_restart_and_completed_row_reuse(shape, split_at, overlap):
    image = np.random.default_rng(18).random(shape, dtype=np.float32)
    args = (image, "max", (64, 64), overlap, 2, split_at)
    assert run(NEW, *args) == run(OLD, *args)


@pytest.mark.parametrize("strategy", ["none", "exact", "max"])
@pytest.mark.parametrize("shape", [(0, 7, 3), (7, 0, 3), (9, 11, 3)])
@pytest.mark.parametrize("failure", [False, True])
def test_failures_and_context(strategy, shape, failure):
    image = np.zeros(shape, np.float32)
    args = (
        image,
        strategy,
        (16, 16),
        3,
        1,
        (1,) if not failure else (),
        failure,
    )
    assert run(NEW, *args) == run(OLD, *args)


def test_exact_retry_limit():
    image = np.zeros((9, 11, 3), np.float32)

    def perform(modules):
        calls = []

        def upscale(tile, region):
            calls.append(state(region))
            return modules[0].Split()

        def shrink(size):
            assert isinstance(sys.exception(), modules[0]._SplitEx)
            calls.append(size)
            return size

        value = outcome(
            lambda: modules[0]._exact_split(image, upscale, (16, 16), shrink, 3)
        )
        return value, calls

    assert perform(NEW) == perform(OLD)


def test_concurrent_independent_plans():
    image = np.zeros((35, 39, 3), np.float32)

    def task(seed):
        args = (image + np.float32(seed / 30), "max", (16, 16), 3, 2)
        assert run(NEW, *args) == run(OLD, *args)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(task, range(24)))


@pytest.mark.parametrize("kind", ["NoTiling", "MaxTileSize", "ExactTileSize"])
@pytest.mark.parametrize("size", [(1, 2), (16, 16), (31, 97), (1024, 4096)])
def test_tiler_contracts(kind, size):
    def perform(modules):
        cls = getattr(modules[2], kind)
        obj = (
            cls(size if kind == "ExactTileSize" else 37)
            if kind != "NoTiling"
            else cls()
        )
        return (
            state(vars(obj)),
            obj.allow_smaller_tile_size(),
            obj.starting_tile_size(3, 53, 4),
            outcome(lambda: obj.split(size)),
        )

    assert perform(NEW) == perform(OLD)


@pytest.mark.parametrize("length", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("fail_at", [None, 0, 1, 2])
def test_unpack_consumption_and_errors(length, fail_at):
    def perform(modules):
        events = []

        def values():
            for i in range(length):
                events.append(i)
                if i == fail_at:
                    raise LookupError("iterator failure")
                yield 32 + i

        obj = modules[2].MaxTileSize(64)
        result = outcome(lambda: obj.split(values()))
        return result, events

    assert perform(NEW) == perform(OLD)


@pytest.mark.parametrize("value", [None, 17, 2.5, [], [16], [16, 32, 64]])
def test_unpack_argument_errors(value):
    assert outcome(lambda: NEW[2].MaxTileSize().split(value)) == outcome(
        lambda: OLD[2].MaxTileSize().split(value)
    )


@pytest.mark.parametrize("shape", [(0, 7, 3), (7, 0, 3), (0, 0, 3)])
def test_empty_border_terminates_and_keeps_identity(shape):
    from nodes.impl.image_utils import BorderType, create_border
    from nodes.utils.utils import Padding

    image = np.empty(shape, np.float32)
    for kind in BorderType:
        assert create_border(image, kind, Padding.all(0)) is image
        if kind.value in {1, 2, 3, 4}:
            with pytest.raises(
                ValueError, match="Cannot extend a border from an empty image"
            ):
                create_border(image, kind, Padding.all(2))
    black = create_border(image, BorderType.BLACK, Padding.all(2))
    assert black.shape == (shape[0] + 4, shape[1] + 4, 3)
    assert not np.any(black)
