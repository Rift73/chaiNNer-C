"""Independent frozen oracles for the final stage-one generation/material audit.

Every finite result is compared by bits, including signed zero. NaN payloads
are not part of the image contract, but their positions must match exactly.
The original node bodies explicitly bind pre-port resize/gradient helpers.
"""

from __future__ import annotations

import ast
import importlib.util
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import reference_gradients

from nodes.impl import gradients, native_geometry
from nodes.impl.color.color import Color

ROOT = Path(__file__).resolve().parents[2]
TESTS = Path(__file__).parent
BASE = ROOT / "backend/src/packages/chaiNNer_standard"


def load_node(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        item
        for item in tree.body
        if not isinstance(item, ast.ImportFrom)
        or (
            item.level == 0
            and item.module not in {"api", "nodes.groups"}
            and not (item.module or "").startswith("nodes.properties")
        )
    ]
    for item in tree.body:
        if isinstance(item, ast.FunctionDef):
            item.decorator_list = []
    module = types.ModuleType(path.stem)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def load_helper(name, path):
    spec = importlib.util.spec_from_file_location("nodes.impl." + name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ORIGINAL_RESIZE = load_helper(
    "stage1_reference_resize", TESTS / "reference_resample/resize.py"
)
ORIGINAL_IMAGE = load_helper(
    "stage1_reference_image", TESTS / "reference_buffers/image_utils.py"
)
PATHS = {
    "metal_to_specular": "material_textures/conversion/metal_to_specular.py",
    "specular_to_metal": "material_textures/conversion/specular_to_metal.py",
    "premultiplied_alpha": "image_adjustment/arithmetic/premultiplied_alpha.py",
    "quantize_to_reference": "image_filter/quantize/quantize_to_reference.py",
    "create_gradient": "image/create_images/create_gradient.py",
    "create_checkerboard": "image/create_images/create_checkerboard.py",
    "create_color": "image/create_images/create_color.py",
}
ACTUAL = {name: load_node(BASE / path) for name, path in PATHS.items()}
ORIGINAL = {
    name: load_node(TESTS / "reference_color_ops" / (name + ".py"))
    for name in ("metal_to_specular", "specular_to_metal")
}
ORIGINAL["premultiplied_alpha"] = load_node(
    TESTS / "reference_adjustments/premultiplied_alpha.py"
)
ORIGINAL["quantize_to_reference"] = load_node(
    TESTS / "reference_filters/quantize/quantize_to_reference.py"
)
ORIGINAL["create_gradient"] = load_node(TESTS / "reference_generation/gradient_node.py")
ORIGINAL["create_checkerboard"] = load_node(
    TESTS / "reference_generation/checkerboard.py"
)
for _name in ("metal_to_specular", "specular_to_metal"):
    ORIGINAL[_name].__dict__.update(resize=ORIGINAL_RESIZE.resize)
    ORIGINAL[_name].__dict__.update(ResizeFilter=ORIGINAL_RESIZE.ResizeFilter)
for _name in ("create_gradient", "create_checkerboard"):
    ORIGINAL[_name].__dict__.update(
        as_target_channels=ORIGINAL_IMAGE.as_target_channels
    )
for _name in ("horizontal", "vertical", "diagonal", "radial", "conic"):
    setattr(
        ORIGINAL["create_gradient"],
        _name + "_gradient",
        getattr(reference_gradients, _name + "_gradient"),
    )


def exact(actual, expected):
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        for a, b in zip(actual, expected, strict=True):
            exact(a, b)
        return
    assert actual.dtype == expected.dtype == np.float32
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    valid = ~np.isnan(expected)
    np.testing.assert_array_equal(
        actual[valid].view(np.uint32), expected[valid].view(np.uint32)
    )


def foreign(image, layout):
    if layout == "strided":
        return image[::-1, ::-1]
    if layout == "transposed":
        return image.swapaxes(0, 1)
    if layout == "readonly":
        image.flags.writeable = False
    if layout == "unaligned":
        result = np.ndarray(
            image.shape, np.float32, buffer=bytearray(image.nbytes + 1), offset=1
        )
        result[...] = image
        return result
    return image


@pytest.mark.parametrize("angle", range(361))
@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (7, 1), (17, 31), (32, 32)])
def test_every_ui_angle_exact(angle, shape):
    for name, args in [("conic_gradient", (angle * np.pi / 180,))] + [
        ("diagonal_gradient", (angle * np.pi / 180, width)) for width in (0, 1, 23, 100)
    ]:
        actual = np.empty(shape, np.float32)
        expected = np.empty_like(actual)
        with np.errstate(all="ignore"):
            getattr(gradients, name)(actual, *args)
            getattr(reference_gradients, name)(expected, *args)
        exact(actual, expected)


@pytest.mark.parametrize("inner", [0, 1, 25, 50, 99, 100])
@pytest.mark.parametrize("outer", [0, 1, 25, 50, 99, 100])
@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (19, 1), (31, 37)])
def test_radial_intervals_exact(inner, outer, shape):
    actual = np.empty(shape, np.float32)
    expected = np.empty_like(actual)
    with np.errstate(all="ignore"):
        gradients.radial_gradient(actual, inner / 100, outer / 100)
        reference_gradients.radial_gradient(expected, inner / 100, outer / 100)
    exact(actual, expected)


@pytest.mark.parametrize("angle", [0, 1, 45, 89, 90, 137, 180, 270, 359, 360])
@pytest.mark.parametrize(
    "width", [65535, 65536, 65537, 2**31, 2**53 - 1, 2**64, 10**100, 10**308, 1e39]
)
@pytest.mark.parametrize("layout", ["plain", "strided", "transposed", "unaligned"])
def test_diagonal_scalar_promotion_stays_native(angle, width, layout):
    actual = foreign(np.empty((17, 31), np.float32), layout)
    expected = np.empty_like(actual)
    radians = angle * np.pi / 180
    reference_gradients.diagonal_gradient(expected, radians, width)
    assert native_geometry.fill_gradient(actual, 2, radians, width)
    exact(actual, expected)


def test_diagonal_unrepresentable_integer_preserves_error():
    for module in (gradients, reference_gradients):
        image = np.full((3, 5), 0.125, np.float32)
        with pytest.raises(OverflowError):
            module.diagonal_gradient(image, 0.75, 10**309)
        exact(image, np.full_like(image, 0.125))


@pytest.mark.parametrize(
    "name,args",
    [
        ("horizontal_gradient", ()),
        ("vertical_gradient", ()),
        ("diagonal_gradient", (1.23, 29)),
        ("radial_gradient", (0.12, 0.78)),
        ("conic_gradient", (0.72,)),
    ],
)
@pytest.mark.parametrize("layout", ["plain", "strided", "transposed", "unaligned"])
def test_gradient_mutation_foreign_buffers(name, args, layout):
    actual = foreign(np.empty((17, 31), np.float32), layout)
    expected = np.empty_like(actual)
    assert getattr(reference_gradients, name)(expected, *args) is None
    assert getattr(gradients, name)(actual, *args) is None
    exact(actual, expected)


COLORS = [
    Color.gray(-0.0),
    Color.gray(0.312345),
    Color.bgr((0.07, 0.73, 0.99)),
    Color.bgra((0.1, 0.2, 0.3, 0.417)),
]


@pytest.mark.parametrize("color1", COLORS)
@pytest.mark.parametrize("color2", COLORS)
@pytest.mark.parametrize(
    "style", ["HORIZONTAL", "VERTICAL", "DIAGONAL", "RADIAL", "CONIC"]
)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("shape", [(1, 1), (17, 31)])
def test_complete_gradient_node(color1, color2, style, reverse, shape):
    results = []
    for module in (ACTUAL["create_gradient"], ORIGINAL["create_gradient"]):
        with np.errstate(all="ignore"):
            results.append(
                module.create_gradient_node(
                    shape[1],
                    shape[0],
                    color1,
                    color2,
                    reverse,
                    module.GradientStyle[style],
                    137,
                    23,
                    10,
                    89,
                    76,
                )
            )
    exact(*results)


@pytest.mark.parametrize("color1", COLORS)
@pytest.mark.parametrize("color2", COLORS)
@pytest.mark.parametrize(
    "shape,square", [((1, 1), 1), ((7, 11), 3), ((257, 263), 1000)]
)
def test_color_and_checkerboard(color1, color2, shape, square):
    h, w = shape
    exact(ACTUAL["create_color"].create_color_node(color1, w, h), color1.to_image(w, h))
    exact(
        ACTUAL["create_checkerboard"].create_checkerboard_node(
            w, h, color1, color2, square
        ),
        ORIGINAL["create_checkerboard"].create_checkerboard_node(
            w, h, color1, color2, square
        ),
    )


@pytest.mark.parametrize("name", ["metal_to_specular", "specular_to_metal"])
@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize(
    "layout", ["plain", "strided", "transposed", "readonly", "unaligned"]
)
@pytest.mark.parametrize("kind", ["random", "signed_zero", "nonfinite"])
@pytest.mark.parametrize("resized", [False, True])
def test_material_complete(name, channels, layout, kind, resized):
    rng = np.random.default_rng(294)
    a = rng.random((7, 11, channels), dtype=np.float32)
    b_shape = (3, 5) if resized else a.shape[:2]
    b = rng.random(
        (*b_shape, 3) if name == "specular_to_metal" else b_shape, dtype=np.float32
    )
    extra = rng.random((2, 3), dtype=np.float32)
    if kind == "signed_zero":
        for array in (a, b, extra):
            array[:] = 0
            array.flat[::2] = -0.0
    elif kind == "nonfinite":
        for array in (a, b, extra):
            array.flat[::7] = np.nan
            array.flat[1::7] = np.inf
            array.flat[2::7] = -np.inf
    args = (foreign(a, layout), foreign(b, layout), foreign(extra, layout))
    before = [array.copy() for array in args]
    if name == "specular_to_metal":
        args += (0, 100)
    with np.errstate(all="ignore"):
        exact(
            getattr(ACTUAL[name], name + "_node")(*args),
            getattr(ORIGINAL[name], name + "_node")(*args),
        )
    for array, saved in zip(args[:3], before, strict=True):
        exact(array, saved)


@pytest.mark.parametrize("operation", ["PREMULTIPLY_RGB", "UNPREMULTIPLY_RGB"])
@pytest.mark.parametrize(
    "layout", ["plain", "strided", "transposed", "readonly", "unaligned"]
)
def test_premultiplied_all_special_pairs(operation, layout):
    values = np.array(
        [
            -np.inf,
            -1,
            -0.0,
            0,
            np.nextafter(np.float32(0), np.float32(1)),
            0.5,
            1,
            np.inf,
            np.nan,
        ],
        np.float32,
    )
    image = np.empty((len(values), len(values), 4), np.float32)
    image[..., :3] = values[:, None, None]
    image[..., 3] = values[None, :]
    image = foreign(image, layout)
    saved = image.copy()
    results = []
    with np.errstate(all="ignore"):
        for module in (ACTUAL["premultiplied_alpha"], ORIGINAL["premultiplied_alpha"]):
            results.append(
                module.premultiplied_alpha_node(
                    image, module.AlphaAssociation[operation]
                )
            )
    exact(*results)
    exact(image, saved)


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize("kind", ["random", "ties", "signed_zero", "nonfinite"])
@pytest.mark.parametrize("radius", [1, 3, 5])
@pytest.mark.parametrize("spatial", [0, 35.7, 100])
def test_quantize_to_reference_exact(channels, kind, radius, spatial):
    rng = np.random.default_rng(851)
    palette = rng.random((3, 5, channels), dtype=np.float32)
    image = rng.random((6, 10, channels), dtype=np.float32)
    if kind == "ties":
        palette[:] = (np.indices(palette.shape[:2]).sum(axis=0) % 2)[..., None]
        image[:] = 0.5
    elif kind == "signed_zero":
        palette[:] = 0
        palette.flat[::3] = -0.0
        image[:] = -0.0
    elif kind == "nonfinite":
        palette.flat[::7] = np.nan
        palette.flat[1::11] = np.inf
        palette.flat[2::13] = -np.inf
        image.flat[::11] = np.nan
    with np.errstate(all="ignore"):
        results = [
            module.quantize_to_reference_node(image, palette, radius, spatial)
            for module in (
                ACTUAL["quantize_to_reference"],
                ORIGINAL["quantize_to_reference"],
            )
        ]
    exact(*results)


def test_large_gradients_concurrent():
    tasks = [
        (name, args)
        for name, args in [
            ("horizontal_gradient", ()),
            ("vertical_gradient", ()),
            ("diagonal_gradient", (137 * np.pi / 180, 213)),
            ("radial_gradient", (0.1, 0.89)),
            ("conic_gradient", (76 * np.pi / 180,)),
        ]
        for _ in range(2)
    ]

    def run(task):
        name, args = task
        expected = np.empty((257, 263), np.float32)
        actual = foreign(np.empty_like(expected), "unaligned")
        with np.errstate(all="ignore"):
            getattr(reference_gradients, name)(expected, *args)
            getattr(gradients, name)(actual, *args)
        exact(actual, expected)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, tasks))
