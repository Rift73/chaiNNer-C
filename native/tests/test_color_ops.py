"""CPU differential coverage for C color, threshold and material algorithms.

The references are frozen upstream node bodies. Only registration decorators
and their unused UI imports are stripped; numerical helpers remain unchanged.
"""

from __future__ import annotations

import ast
import ctypes as ct
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest

from nodes.impl import native_color_ops as native
from nodes.impl.image_utils import to_uint8

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/src/packages/chaiNNer_standard"
REFERENCE = Path(__file__).with_name("reference_color_ops")
PATHS = {
    "hue_and_saturation": "image_adjustment/adjustments/hue_and_saturation.py",
    "threshold": "image_adjustment/threshold/threshold.py",
    "generate_threshold": "image_adjustment/threshold/generate_threshold.py",
    "threshold_adaptive": "image_adjustment/threshold/threshold_adaptive.py",
    "metal_to_specular": "material_textures/conversion/metal_to_specular.py",
    "specular_to_metal": "material_textures/conversion/specular_to_metal.py",
    "normal_map_generator": "material_textures/normal_map/normal_map_generator.py",
}


def load_body(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "utils.utils":
            node.level = 0
            node.module = "nodes.utils.utils"
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


MODULES = {
    name: (load_body(BASE / relative), load_body(REFERENCE / Path(relative).name))
    for name, relative in PATHS.items()
}
HEIGHT_REFERENCE = load_body(REFERENCE / "height.py")


def image(channels=3, layout="contiguous", shape=(13, 17), seed=327):
    rng = np.random.default_rng(seed)
    img = rng.random((*shape, channels) if channels != 1 else shape, dtype=np.float32)
    img.flat[: min(6, img.size)] = [0, 1, 0.5, 0.001, 0.9999, 0.3333][
        : min(6, img.size)
    ]
    if layout == "strided":
        img = img[::2, ::-2]
    elif layout == "transpose":
        img = img.swapaxes(0, 1)
    elif layout == "readonly":
        img.flags.writeable = False
    elif layout == "unaligned":
        unaligned = np.ndarray(
            img.shape, dtype=np.float32, buffer=bytearray(img.nbytes + 1), offset=1
        )
        unaligned[...] = img
        img = unaligned
    return img


def equal(actual, expected, *, atol=0):
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        for a, e in zip(actual, expected, strict=True):
            equal(a, e, atol=atol)
        return
    if not isinstance(expected, np.ndarray):
        assert actual == expected
        return
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol, equal_nan=True)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize(
    "layout", ["contiguous", "strided", "transpose", "readonly", "unaligned"]
)
@pytest.mark.parametrize(
    "adjustments",
    [
        (0, 0, 0),
        (180, 0, 0),
        (-180, 43.3, 13.7),
        (0, -100, -100),
        (33, 100, 100),
        (0, 0, -33.3),
    ],
)
def test_hue_and_saturation(channels, layout, adjustments):
    current, reference = MODULES["hue_and_saturation"]
    img = image(channels, layout)
    saved = img.copy()
    actual = current.hue_and_saturation_node(img, *adjustments)
    expected = reference.hue_and_saturation_node(img, *adjustments)
    equal(actual, expected)
    np.testing.assert_array_equal(img, saved)
    if adjustments == (0, 0, 0):
        assert actual is img


@pytest.mark.parametrize("kind", range(5))
@pytest.mark.parametrize("channels", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "readonly"])
@pytest.mark.parametrize("antialias", [False, True])
def test_threshold(kind, channels, layout, antialias):
    current, reference = MODULES["threshold"]
    img = image(channels, layout)
    saved = img.copy()
    actual = current.threshold_node(
        img, 43.3, current.ThresholdType(kind), 78.9, antialias, 3
    )
    expected = reference.threshold_node(
        img, 43.3, reference.ThresholdType(kind), 78.9, antialias, 3
    )
    equal(actual, expected)
    np.testing.assert_array_equal(img, saved)


@pytest.mark.parametrize("kind", range(5))
@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (17, 1), (2, 2)])
def test_threshold_boundaries_and_nonfinite(kind, shape):
    current, reference = MODULES["threshold"]
    img = np.resize(np.array([np.nan, np.inf, -np.inf, 0.5, 0, 1], np.float32), shape)
    with np.errstate(all="ignore"):
        actual = current.threshold_node(
            img, 50, current.ThresholdType(kind), 100, False, 0
        )
        expected = reference.threshold_node(
            img, 50, reference.ThresholdType(kind), 100, False, 0
        )
    if kind == 2:
        # The C correction preserves NaNs consistently instead of depending
        # on the alignment of OpenCV/IPP's newly allocated output.
        nan_source = np.isnan(img)
        equal(actual[~nan_source], expected[~nan_source])
        assert np.all(np.isnan(actual[nan_source]))
    else:
        equal(actual, expected)


@pytest.mark.parametrize("channels", [1, 2, 3, 4, 5, 9])
@pytest.mark.parametrize("method", [0, 1])
@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (17, 13), (257, 513)])
def test_auto_threshold(channels, method, shape):
    current, reference = MODULES["generate_threshold"]
    img = image(channels, "readonly", shape)
    actual = current.generate_threshold_node(img, current.AutoThreshold(method))
    expected = reference.generate_threshold_node(img, reference.AutoThreshold(method))
    equal(actual, expected)


@pytest.mark.parametrize("value", [0, 1, 0.5, -0.1, 1.2, np.nan, np.inf])
@pytest.mark.parametrize("method", [0, 1])
def test_auto_threshold_constant_and_nonfinite(value, method):
    current, reference = MODULES["generate_threshold"]
    img = np.full((7, 11), value, dtype=np.float32)
    with np.errstate(all="ignore"):
        equal(
            current.generate_threshold_node(img, current.AutoThreshold(method)),
            reference.generate_threshold_node(img, reference.AutoThreshold(method)),
        )


def test_quantization_rounding_and_wrapping():
    values = np.array(
        [0, 1, -1, 2, -10, 10, np.nan, np.inf, -np.inf, 1e30, -1e30], np.float32
    )
    ties = (np.arange(-30, 280, dtype=np.float32) + 0.5) / 255
    img = np.concatenate([values, ties]).reshape((1, -1))
    with np.errstate(all="ignore"):
        equal(native.quantize_u8(img), to_uint8(img, normalized=True))


@pytest.mark.parametrize("kind", [0, 1])
@pytest.mark.parametrize("method", [0, 1])
@pytest.mark.parametrize("radius", [1, 3, 8])
@pytest.mark.parametrize(
    "maximum,delta", [(100, 0), (37.5, -3.3), (0, 100), (150, -100)]
)
def test_adaptive_threshold(kind, method, radius, maximum, delta):
    current, reference = MODULES["threshold_adaptive"]
    img = image(1, "strided")
    actual = current.threshold_adaptive_node(
        img,
        current.AdaptiveThresholdType(kind),
        maximum,
        current.AdaptiveMethod(method),
        radius,
        delta,
    )
    expected = reference.threshold_adaptive_node(
        img,
        reference.AdaptiveThresholdType(kind),
        maximum,
        reference.AdaptiveMethod(method),
        radius,
        delta,
    )
    equal(actual, expected)


@pytest.mark.parametrize("method", [0, 1])
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1)])
def test_adaptive_tiny(method, shape):
    img = image(1, "readonly", shape)
    equal(
        native.adaptive_threshold(img, 0, 255, method, 1, 0),
        cv2.adaptiveThreshold(to_uint8(img, normalized=True), 255, method, 0, 3, 0),
    )


@pytest.mark.parametrize("name", ["metal_to_specular", "specular_to_metal"])
@pytest.mark.parametrize("alpha", [False, True])
@pytest.mark.parametrize("resized", [False, True])
@pytest.mark.parametrize("optional", [False, True])
def test_material_nodes(name, alpha, resized, optional):
    current, reference = MODULES[name]
    a = image(4 if alpha else 3, "strided")
    shape = (9, 11) if resized else a.shape[:2]
    b = image(1 if name == "metal_to_specular" else 3, "readonly", shape, seed=49)
    extra = image(1, "readonly", (3, 4)) if optional else None
    arguments = (a, b, extra) + ((23, 30) if name == "specular_to_metal" else ())
    snapshots = [arg.copy() for arg in arguments if isinstance(arg, np.ndarray)]
    actual = getattr(current, name + "_node")(*arguments)
    expected = getattr(reference, name + "_node")(*arguments)
    equal(actual, expected)
    for arg, saved in zip(
        (arg for arg in arguments if isinstance(arg, np.ndarray)),
        snapshots,
        strict=True,
    ):
        np.testing.assert_array_equal(arg, saved)


@pytest.mark.parametrize("minimum,maximum", [(23, 23), (80, 20), (0, 100)])
def test_material_nonfinite_and_equal_cutoffs(minimum, maximum):
    current, reference = MODULES["specular_to_metal"]
    a, b = image(), image(seed=75)
    b.flat[:3] = [np.nan, np.inf, -np.inf]
    with np.errstate(all="ignore"):
        equal(
            current.specular_to_metal_node(a, b, None, minimum, maximum),
            reference.specular_to_metal_node(a, b, None, minimum, maximum),
        )


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("mode", range(7))
def test_height_map(channels, mode):
    img = image(channels, "strided")
    equal(
        native.height_map(img, mode),
        HEIGHT_REFERENCE.get_height_map(img, HEIGHT_REFERENCE.HeightSource(mode)),
    )


@pytest.mark.parametrize("mode", range(7))
def test_height_nonfinite_and_derivative_output(mode):
    img = image(4)
    img.flat[:4] = [np.nan, np.inf, -np.inf, 0.5]
    with np.errstate(all="ignore"):
        equal(
            native.height_map(img, mode),
            HEIGHT_REFERENCE.get_height_map(img, HEIGHT_REFERENCE.HeightSource(mode)),
        )
        dx = np.array([[np.nan, np.inf, -np.inf, 0, 0.3]], dtype=np.float32)
        dy = np.array([[0, 0, np.inf, 0, -0.5]], dtype=np.float32)
        x, y, z = MODULES["normal_map_generator"][1].normalize(dx, dy)
        expected = np.dstack((np.abs(z), (y + 1) * 0.5, (-x + 1) * 0.5))
        equal(native.normal_output(dx, dy, None, True, False, False), expected)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("mode", range(7))
@pytest.mark.parametrize("alpha", ["none", "unchanged", "height", "one"])
def test_normal_generator(channels, mode, alpha):
    current, reference = MODULES["normal_map_generator"]
    img = image(channels, "readonly", shape=(5, 7))
    arguments = (
        False,
        current.HeightSource(mode),
        0,
        0.2,
        1.7,
        current.EdgeFilter.SOBEL,
        0.25,
        0.5,
        0.3,
        0.25,
        0.2,
        0.15,
        0.1,
        0.1,
        True,
        False,
        current.AlphaOutput(alpha),
    )
    expected_args = (*arguments[:-1], reference.AlphaOutput(alpha))
    equal(
        current.normal_map_generator_node(img, *arguments),
        reference.normal_map_generator_node(img, *expected_args),
    )


@pytest.mark.parametrize("tileable", [False, True])
@pytest.mark.parametrize("blur", [-1.3, 0, 1.7])
@pytest.mark.parametrize("scale", [0, 0.3, 2])
def test_normal_generator_filters(tileable, blur, scale):
    current, reference = MODULES["normal_map_generator"]
    img = image(4, "strided", (7, 11))
    arguments = (
        tileable,
        current.HeightSource.SCREEN_RGB,
        blur,
        0,
        scale,
        current.EdgeFilter.SCHARR,
        0.25,
        0.5,
        0.3,
        0.25,
        0.2,
        0.15,
        0.1,
        0.1,
        False,
        True,
        current.AlphaOutput.HEIGHT,
    )
    equal(
        current.normal_map_generator_node(img, *arguments),
        reference.normal_map_generator_node(
            img, *arguments[:-1], reference.AlphaOutput.HEIGHT
        ),
    )


def test_parallel_concurrent_pipelines():
    def run(seed):
        img = image(4, "readonly", (257, 513), seed=seed)
        current, reference = MODULES["hue_and_saturation"]
        equal(
            current.hue_and_saturation_node(img, 31.3, 15.7, -25.3),
            reference.hue_and_saturation_node(img, 31.3, 15.7, -25.3),
        )
        current, reference = MODULES["generate_threshold"]
        equal(
            current.generate_threshold_node(img, current.AutoThreshold.OTSU),
            reference.generate_threshold_node(img, reference.AutoThreshold.OTSU),
        )
        current, reference = MODULES["metal_to_specular"]
        metal = img[:, :, 0]
        equal(
            current.metal_to_specular_node(img, metal, None),
            reference.metal_to_specular_node(img, metal, None),
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(run, range(4)))


@pytest.mark.parametrize("shape", [(0, 2), (2, 0), (2,), (1, 2, 3, 4)])
def test_malformed_images_rejected(shape):
    with pytest.raises(ValueError):
        native.linear(np.empty(shape, dtype=np.float32), 1)


def test_dtype_and_shape_validation():
    with pytest.raises(TypeError):
        native.linear(np.zeros((2, 2), dtype=np.float64), 1)
    with pytest.raises(ValueError):
        native.normal_output(
            np.zeros((2, 2), np.float32),
            np.zeros((3, 2), np.float32),
            None,
            False,
            False,
            False,
        )
    with pytest.raises(ValueError):
        native.material(
            np.zeros((2, 2, 3), np.float32),
            4,
            b=np.zeros((2, 2, 3), np.float32),
            mask=np.zeros((1, 2), np.float32),
        )


def test_c_rejects_invalid_sizes():
    library = native._api()  # check the raw ABI boundary
    maximum = ct.c_size_t(-1).value
    assert library.cn_color_linear(None, None, maximum, 1, 0) == 2
    assert library.cn_hls_adjust(None, maximum, 0, 1) == 2
    assert library.cn_threshold_f32(None, None, None, 0, 5, 0, 1) == 1
    assert library.cn_height_map(None, None, 1, 2, 0) == 1
    pointer = ct.pointer(ct.c_float())
    assert (
        library.cn_normal_output(pointer, pointer, None, pointer, maximum, 0, 0, 3) == 2
    )


def test_nonfinite_threshold_uses_c_and_preserves_nan(monkeypatch):
    img = np.full((1, 17), np.nan, np.float32)

    def threshold(*_args, **_kwargs):
        raise AssertionError("Threshold must execute its image arithmetic in C")

    monkeypatch.setattr(cv2, "threshold", threshold)
    assert np.isnan(native.threshold(img, 0.5, 1, 2)).all()


def test_missing_library_not_hidden(monkeypatch):
    def fail():
        raise RuntimeError("broken C library")

    monkeypatch.setattr(native, "_api", fail)
    with pytest.raises(RuntimeError, match="broken C library"):
        native.threshold(np.full((1, 1), np.nan, np.float32), 0.5, 1, 2)


@pytest.mark.parametrize("relative", PATHS.values())
def test_node_schema_and_signature_unchanged(relative):
    current = ast.parse((BASE / relative).read_text(encoding="utf-8"))
    original = ast.parse((REFERENCE / Path(relative).name).read_text(encoding="utf-8"))

    def contracts(tree):
        return {
            function.name: (
                ast.dump(function.args),
                [ast.dump(d) for d in function.decorator_list],
            )
            for function in tree.body
            if isinstance(function, ast.FunctionDef) and function.name.endswith("_node")
        }

    assert contracts(current) == contracts(original)
