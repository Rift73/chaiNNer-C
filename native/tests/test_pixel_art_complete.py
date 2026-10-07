"""Independent comparisons against the installed Rust pixel-art implementation."""

from __future__ import annotations

import ast
import ctypes as ct
import importlib.util
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import chainner_ext
from nodes.impl import native_pixel_art as native
from nodes.impl.native import ptr

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).parent / "reference_pixel_art"
NODE = (
    ROOT
    / "backend/src/packages/chaiNNer_standard/image_dimension/resize/resize_pixel_art.py"
)
MODES = tuple(native._MODES)


def exact(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(
        actual[finite].view(np.uint32), expected[finite].view(np.uint32)
    )


def compare(source, mode):
    before = source.copy()
    with np.errstate(all="ignore"):
        expected = chainner_ext.pixel_art_upscale(source, *mode)
        result = native.pixel_art_upscale(source, *mode)
    exact(result, expected)
    exact(source, before)
    assert result.flags.c_contiguous and result.flags.writeable
    assert not np.shares_memory(result, source)
    return result


def layout(source, kind):
    if kind == "fortran":
        return np.asfortranarray(source)
    if kind == "reverse":
        return source[::-1, ::-1]
    if kind == "stride":
        output = np.zeros(
            (source.shape[0] * 2, source.shape[1] * 2, *source.shape[2:]), np.float32
        )
        output[::2, ::2] = source
        return output[::2, ::2]
    if kind == "readonly":
        source.flags.writeable = False
    if kind == "unaligned":
        output = np.ndarray(
            source.shape, np.float32, buffer=bytearray(source.nbytes + 1), offset=1
        )
        output[...] = source
        return output
    return source


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("channels", [0, 1, 3, 4])
@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (2, 2), (5, 7), (17, 19)])
@pytest.mark.parametrize(
    "kind", ["contiguous", "fortran", "reverse", "stride", "readonly", "unaligned"]
)
def test_shapes_layouts(mode, channels, shape, kind):
    dims = (*shape, channels) if channels else shape
    source = np.random.default_rng(193).integers(0, 4, dims).astype(np.float32) / 3
    compare(layout(source, kind), mode)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("channels", [0, 1, 3, 4])
@pytest.mark.parametrize(
    "values", ["continuous", "special", "threshold", "bitpatterns"]
)
def test_float_semantics(mode, channels, values):
    dims = (17, 19, channels) if channels else (17, 19)
    rng = np.random.default_rng(201)
    if values == "continuous":
        source = (rng.random(dims, dtype=np.float32) - 0.5) * 4
    elif values == "special":
        palette = np.array(
            [
                0,
                -0.0,
                1,
                -1,
                np.inf,
                -np.inf,
                np.nan,
                np.finfo(np.float32).max,
                np.finfo(np.float32).tiny,
                np.nextafter(np.float32(0), np.float32(1)),
            ],
            np.float32,
        )
        source = rng.choice(palette, dims)
    elif values == "threshold":
        palette = np.array([0, 1 / 255, 3 / 255, 6 / 255, 7 / 255, 0.5], np.float32)
        palette = np.concatenate(
            (
                palette,
                np.nextafter(palette, np.float32(np.inf)),
                np.nextafter(palette, np.float32(-np.inf)),
            )
        )
        source = rng.choice(palette, dims)
    else:
        source = rng.integers(0, 2**32, dims, dtype=np.uint32).view(np.float32)
    compare(source, mode)


@pytest.mark.parametrize("scale", [2, 3, 4])
@pytest.mark.parametrize("channels", [0, 3, 4])
@pytest.mark.parametrize("pattern", range(256))
def test_every_hq_pattern(scale, channels, pattern):
    source = np.zeros((3, 3, channels) if channels else (3, 3), np.float32)
    neighbors = [(y, x) for y in range(3) for x in range(3) if (y, x) != (1, 1)]
    for bit, (y, x) in enumerate(neighbors):
        source[y, x] = float(bool(pattern & (1 << bit)))
    compare(source, ("hqx", scale))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("channels", [0, 3, 4])
def test_sai_ties_and_large_pool(mode, channels):
    # Random binary neighborhoods exercise equality/tie rules; the large image
    # crosses native row scheduling grains for every algorithm and channel count.
    dims = (133, 139, channels) if channels else (133, 139)
    source = np.random.default_rng(71).integers(0, 2, dims).astype(np.float32)
    compare(source, mode)


def test_concurrent_calls():
    source = (
        np.random.default_rng(53).integers(0, 5, (97, 131, 4)).astype(np.float32) / 4
    )
    expected = {mode: chainner_ext.pixel_art_upscale(source, *mode) for mode in MODES}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for mode, result in zip(
            MODES * 3,
            pool.map(lambda mode: native.pixel_art_upscale(source, *mode), MODES * 3),
            strict=False,
        ):
            exact(result, expected[mode])


def load_node(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        item
        for item in tree.body
        if not isinstance(item, ast.ImportFrom)
        or (not item.level and not (item.module or "").startswith("nodes.properties"))
    ]
    for item in tree.body:
        if isinstance(item, ast.FunctionDef):
            item.decorator_list = []
    module = types.ModuleType(path.stem)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def test_whole_node_and_metadata(monkeypatch):
    original = load_node(REFERENCE / "resize_pixel_art.py")
    current = load_node(NODE)
    source = np.random.default_rng(27).random((7, 11, 3), dtype=np.float32)
    expected = [
        original.resize_pixel_art_node(source, mode)
        for mode in original.ResizeAlgorithm
    ]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Runtime must not call the Rust pixel-art implementation")

    monkeypatch.setattr(chainner_ext, "pixel_art_upscale", forbidden)
    for mode, output in zip(current.ResizeAlgorithm, expected, strict=False):
        exact(current.resize_pixel_art_node(source, mode), output)
    # All enum values, labels and registered UI/typing declarations are unchanged.
    assert [(item.name, item.value) for item in original.ResizeAlgorithm] == [
        (item.name, item.value) for item in current.ResizeAlgorithm
    ]
    a = ast.parse((REFERENCE / "resize_pixel_art.py").read_text())
    b = ast.parse(NODE.read_text())
    aa = next(
        item
        for item in a.body
        if isinstance(item, ast.FunctionDef) and item.name == "resize_pixel_art_node"
    )
    bb = next(
        item
        for item in b.body
        if isinstance(item, ast.FunctionDef) and item.name == "resize_pixel_art_node"
    )
    assert ast.dump(aa) == ast.dump(bb)


def test_generated_table_is_reproducible(tmp_path, monkeypatch):
    path = ROOT / "native/tools/generate_pixel_art_tables.py"
    spec = importlib.util.spec_from_file_location("pixel_tables", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generated = tmp_path / "tables.inc"
    monkeypatch.setattr(module, "OUTPUT", generated)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    module.main()
    assert (
        generated.read_bytes()
        == (ROOT / "native/include/pixel_art_tables.inc").read_bytes()
    )


def test_abi_rejects_bad_buffers_before_writing():
    call = native._api().cn_pixel_art_f32
    source = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    output = np.full(192, -9, np.float32)
    null = ct.POINTER(ct.c_float)()
    size_max = ct.c_size_t(-1).value
    invalid = [
        (null, ptr(output), 2, 2, 3, 0),
        (ptr(source), null, 2, 2, 3, 0),
        (ptr(source), ptr(source), 2, 2, 3, 0),
        (ptr(source), ptr(output), 0, 2, 3, 0),
        (ptr(source), ptr(output), 2, 2, 2, 0),
        (ptr(source), ptr(output), 2, 2, 3, -1),
        (ptr(source), ptr(output), 2, 2, 3, 11),
        (ptr(source), ptr(output), size_max, 2, 3, 0),
        (ptr(source), ptr(output), 2, size_max // 2, 3, 0),
        (
            ct.cast(source.ctypes.data + 1, ct.POINTER(ct.c_float)),
            ptr(output),
            2,
            2,
            3,
            0,
        ),
        (
            ptr(source),
            ct.cast(output.ctypes.data + 1, ct.POINTER(ct.c_float)),
            2,
            2,
            3,
            0,
        ),
        (ct.cast(size_max - 3, ct.POINTER(ct.c_float)), ptr(output), 2, 2, 3, 0),
    ]
    for args in invalid:
        assert call(*args) != 0
        np.testing.assert_array_equal(output, -9)
        np.testing.assert_array_equal(source.reshape(-1), np.arange(12))


@pytest.mark.parametrize("mode", [("missing", 2), ("sai", 3), ("hqx", 5), ("eagle", 4)])
def test_invalid_algorithm_errors(mode):
    source = np.zeros((3, 3), np.float32)
    with pytest.raises(ValueError) as expected:
        chainner_ext.pixel_art_upscale(source, *mode)
    with pytest.raises(
        type(expected.value), match="^" + str(expected.value).replace(".", r"\.") + "$"
    ):
        native.pixel_art_upscale(source, *mode)
