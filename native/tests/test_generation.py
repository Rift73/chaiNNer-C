"""Compare real node/helper bodies with frozen pre-port implementations."""

from __future__ import annotations

import ast
import ctypes as ct
import importlib.util
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import reference_gradients

from nodes.impl import native_generation as native
from nodes.impl import noise
from nodes.impl.color.color import Color
from nodes.impl.noise_functions.simplex import SimplexNoise
from nodes.impl.noise_functions.value import ValueNoise
from nodes.utils.seed import Seed

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).with_name("reference_generation")
NODES = ROOT / "backend/src/packages/chaiNNer_standard/image/create_images"


def load_helper(name, package):
    spec = importlib.util.spec_from_file_location(
        package + "._reference_" + name, REFERENCE / (name + ".py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_node(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, ast.ImportFrom)
        or (
            node.level == 0
            and node.module not in {"api", "nodes.groups"}
            and not (node.module or "").startswith("nodes.properties")
        )
    ]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    module = types.ModuleType(path.stem)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


ORIGINAL_VALUE = load_helper("value", "nodes.impl.noise_functions").ValueNoise
ORIGINAL_SIMPLEX = load_helper("simplex", "nodes.impl.noise_functions").SimplexNoise
ORIGINAL_NOISE = load_helper("noise", "nodes.impl")
CHECKER = load_node(NODES / "create_checkerboard.py")
ORIGINAL_CHECKER = load_node(REFERENCE / "checkerboard.py")
GRADIENT = load_node(NODES / "create_gradient.py")
ORIGINAL_GRADIENT = load_node(REFERENCE / "gradient_node.py")
for _gradient_name in (
    "horizontal_gradient",
    "vertical_gradient",
    "diagonal_gradient",
    "radial_gradient",
    "conic_gradient",
):
    # The frozen node must not import the production C gradient as its oracle.
    setattr(
        ORIGINAL_GRADIENT, _gradient_name, getattr(reference_gradients, _gradient_name)
    )
CREATE_NOISE = load_node(NODES / "create_noise.py")
ORIGINAL_CREATE_NOISE = load_node(REFERENCE / "create_noise_node.py")
ORIGINAL_CREATE_NOISE.__dict__.update(
    SimplexNoise=ORIGINAL_SIMPLEX,
    ValueNoise=ORIGINAL_VALUE,
    create_blue_noise=load_helper(
        "blue", "nodes.impl.noise_functions"
    ).create_blue_noise,
)
COLORS = [
    Color.gray(0.312345),
    Color.bgr((0.07, 0.73, 0.99)),
    Color.bgra((0.1, 0.2, 0.3, 0.417)),
]


@pytest.mark.parametrize("color", COLORS)
@pytest.mark.parametrize("shape", [(1, 1), (7, 11), (257, 263)])
def test_color_fill(color, shape):
    h, w = shape
    np.testing.assert_array_equal(
        native.image_fill(w, h, color.value), color.to_image(w, h)
    )


@pytest.mark.parametrize("a", COLORS)
@pytest.mark.parametrize("b", COLORS)
@pytest.mark.parametrize(
    "shape,square", [((1, 1), 2), ((7, 11), 3), ((65, 17), 1), ((263, 257), 31)]
)
def test_checkerboard(a, b, shape, square):
    h, w = shape
    actual = CHECKER.create_checkerboard_node(w, h, a, b, square)
    expected = ORIGINAL_CHECKER.create_checkerboard_node(w, h, a, b, square)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype


@pytest.mark.parametrize("a", COLORS)
@pytest.mark.parametrize("b", COLORS)
@pytest.mark.parametrize(
    "style", ["HORIZONTAL", "VERTICAL", "DIAGONAL", "RADIAL", "CONIC"]
)
@pytest.mark.parametrize("reverse", [False, True])
def test_gradient_interpolation(a, b, style, reverse):
    args = (31, 17, a, b, reverse)
    tail = (137, 23, 10, 89, 76)
    expected = ORIGINAL_GRADIENT.create_gradient_node(
        *args, ORIGINAL_GRADIENT.GradientStyle[style], *tail
    )
    actual = GRADIENT.create_gradient_node(*args, GRADIENT.GradientStyle[style], *tail)
    np.testing.assert_array_equal(actual, expected)


def coordinates(dimensions, layout):
    points = np.random.default_rng(713).uniform(-100, 100, (129, dimensions))
    points[:4] = [
        np.zeros(dimensions),
        np.ones(dimensions),
        -np.ones(dimensions),
        np.full(dimensions, 0.5),
    ]
    if layout == "strided":
        points = points[::2, ::-1]
    elif layout == "readonly":
        points.setflags(write=False)
    elif layout == "float32":
        points = points.astype(np.float32)
    return points


@pytest.mark.parametrize("dimensions", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("seed", [0, 314159])
@pytest.mark.parametrize("layout", ["plain", "strided", "readonly", "float32"])
def test_value_noise(dimensions, smooth, seed, layout):
    points = coordinates(dimensions, layout)
    before = points.copy()
    expected = ORIGINAL_VALUE(dimensions, seed, smooth).evaluate(points)
    actual = ValueNoise(dimensions, seed, smooth).evaluate(points)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(points, before)
    assert actual.dtype == expected.dtype


@pytest.mark.parametrize("dimensions", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("seed", [None, 0, 314159])
@pytest.mark.parametrize("layout", ["plain", "strided", "readonly", "float32"])
@pytest.mark.parametrize("r2", [0.5, 0.6])
def test_simplex_noise(dimensions, seed, layout, r2):
    points = coordinates(dimensions, layout)
    before = points.copy()
    expected = ORIGINAL_SIMPLEX(dimensions, seed, r2).evaluate(points)
    actual = SimplexNoise(dimensions, seed, r2).evaluate(points)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(points, before)
    assert actual.dtype == expected.dtype


@pytest.mark.parametrize(
    "name",
    [
        "gaussian_noise",
        "uniform_noise",
        "salt_and_pepper_noise",
        "poisson_noise",
        "speckle_noise",
    ],
)
@pytest.mark.parametrize("channels", [1, 3, 4, 5])
@pytest.mark.parametrize("color", ["GRAY", "RGB"])
@pytest.mark.parametrize("amount", [0, 0.3, 1])
@pytest.mark.parametrize("layout", ["plain", "strided", "readonly"])
def test_add_noise(name, channels, color, amount, layout):
    shape = (17, 13) if channels == 1 else (17, 13, channels)
    image = np.random.default_rng(92).uniform(-0.5, 1.5, shape).astype(np.float32)
    image.flat[:3] = [np.nan, np.inf, -np.inf]
    if layout == "strided":
        image = image[::2, ::-2]
    elif layout == "readonly":
        image.setflags(write=False)
    before = image.copy()
    with np.errstate(all="ignore"):
        expected = getattr(ORIGINAL_NOISE, name)(
            image, amount, ORIGINAL_NOISE.NoiseColor[color], 701
        )
        actual = getattr(noise, name)(image, amount, noise.NoiseColor[color], 701)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, before)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape


@pytest.mark.parametrize("method", ["SIMPLEX", "VALUE_NOISE", "SMOOTH_VALUE_NOISE"])
@pytest.mark.parametrize(
    "tiling",
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, False),
        (False, False, True),
    ],
)
@pytest.mark.parametrize("fractal", ["NONE", "PINK_NOISE"])
@pytest.mark.parametrize("seed", [0, 714])
def test_create_noise_workflow(method, tiling, fractal, seed):
    def execute(module):
        return module.create_noise_node(
            37,
            29,
            Seed(seed),
            module.NoiseMethod[method],
            17.3,
            87,
            *tiling,
            module.FractalMethod[fractal],
            4,
            2,
            2,
            True,
            1.5,
        )

    expected, actual = execute(ORIGINAL_CREATE_NOISE), execute(CREATE_NOISE)
    np.testing.assert_array_equal(actual, expected)


def test_concurrent_noise_and_construction():
    points = np.random.default_rng(312).uniform(-1, 1, (40000, 4))
    image = np.random.default_rng(4).random((513, 257, 4), dtype=np.float32)
    generator = SimplexNoise(4, 197)
    functions = [
        lambda: generator.evaluate(points),
        lambda: noise.speckle_noise(image, 0.3, noise.NoiseColor.RGB, 198),
        lambda: native.image_fill(
            517, 257, (0.1, 0.2, 0.3, 0.4), second=(1, 0, 0.7, 0.5), square=31
        ),
    ]
    expected = [f() for f in functions]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda n: functions[n % 3](), range(24)))
    for n, result in enumerate(results):
        np.testing.assert_array_equal(result, expected[n % 3])


def test_checked_boundaries():
    with pytest.raises(ValueError):
        native.image_fill(0, 1, (1,))
    with pytest.raises(ValueError):
        native.image_fill(1, 1, (1,), second=(1, 2, 3), square=1)
    with pytest.raises(ValueError):
        native.image_fill(1, 1, (1,), second=(0,), gradient=np.ones((2, 2), np.float32))
    with pytest.raises(ValueError):
        native.combine_noise(
            np.ones((2, 2, 2), np.float32), [np.ones((2, 2), np.float32)], 0
        )
    with pytest.raises(ValueError):
        native.combine_noise(
            np.ones((2, 2), np.float32), [np.ones((1, 1), np.float32)], 0
        )
    with pytest.raises(ValueError):
        native.procedural_noise(
            np.ones((2, 2)), np.array([-1, 0]), values=np.ones(2, np.float32)
        )
    api = native._api()  # exercise malformed direct ABI calls
    assert api.cn_image_fill(None, 1, 1, 1, None, None, None, 0, 0) == 1
    bad = np.array([[np.nan, 0]], np.float64)
    table = np.arange(4, dtype=np.int32)
    values = np.ones(2, np.float32)
    out = np.zeros(1)
    dp = ct.POINTER(ct.c_double)
    assert (
        api.cn_procedural_noise(
            bad.ctypes.data_as(dp),
            1,
            2,
            table.ctypes.data_as(ct.POINTER(ct.c_int32)),
            4,
            native.ptr(values),
            2,
            None,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            out.ctypes.data_as(dp),
        )
        == 1
    )
    assert out[0] == 0


def test_empty_and_nonfinite_simplex_compatibility():
    for generator in (SimplexNoise(2, 91), ORIGINAL_SIMPLEX(2, 91)):
        with pytest.raises(ValueError):
            generator.evaluate(np.empty((0, 2)))
    for radius in (np.nan, np.inf, -np.inf):
        with np.errstate(all="ignore"):
            actual = SimplexNoise(2, 91, radius).evaluate(np.zeros((5, 2)))
            expected = ORIGINAL_SIMPLEX(2, 91, radius).evaluate(np.zeros((5, 2)))
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("layout", ["reversed", "unaligned"])
@pytest.mark.parametrize("mode", ["solid", "checker", "gradient"])
def test_fill_foreign_color_buffers(channels, layout, mode):
    def color(values):
        value = np.array(values, dtype=np.float32)
        if layout == "reversed":
            return value[::-1]
        buffer = np.ndarray(
            value.shape, np.float32, buffer=bytearray(value.nbytes + 1), offset=1
        )
        buffer[:] = value
        assert not buffer.flags.aligned
        return buffer

    first = color([0.2, 0.4, 0.7, 0.9][:channels])
    second = color([0.1, 0.5, 0.3, 0.8][:channels])
    saved_first, saved_second = first.copy(), second.copy()
    options = {}
    if mode == "checker":
        options["square"] = 2
    if mode == "gradient":
        options["gradient"] = np.linspace(0, 1, 20, dtype=np.float32).reshape(4, 5)
    if mode != "solid":
        options["second"] = second
    actual = native.image_fill(5, 4, first, **options)
    if "second" in options:
        options["second"] = tuple(second)
    expected = native.image_fill(5, 4, tuple(first), **options)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(first, saved_first)
    np.testing.assert_array_equal(second, saved_second)
