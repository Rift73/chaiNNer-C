"""Differential tests against frozen upstream channel/layout node functions."""

from __future__ import annotations

import ast
import ctypes as ct
import itertools
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from nodes.impl import native_channels as native
from nodes.impl.color.color import Color
from nodes.impl.image_utils import FlipAxis, ShiftFill
from nodes.impl.native import lib, ptr

ROOT = Path(__file__).resolve().parents[2]
NODES = ROOT / "backend/src/packages/chaiNNer_standard"
REFERENCE = Path(__file__).with_name("reference_channels")
PATHS = {
    "combine_rgba": "image_channel/all/combine_rgba.py",
    "merge_channels": "image_channel/all/merge_channels.py",
    "merge_transparency": "image_channel/transparency/merge_transparency.py",
    "flip": "image_utility/modification/flip.py",
    "shift": "image_utility/modification/shift.py",
    "crop_to_content": "image_dimension/crop/crop_to_content.py",
    "get_bounding_box": "image_dimension/utility/get_bounding_box.py",
}
# The frozen bodies keep the node names they were captured under.
FROZEN_NAMES = {
    "combine_rgba": "combine_r_g_b_a",
    "merge_transparency": "combine_rgb_a",
}


def load_body(path):
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


MODULES = {
    name: (
        load_body(NODES / path),
        load_body(REFERENCE / (FROZEN_NAMES.get(name, name) + ".py")),
    )
    for name, path in PATHS.items()
}
LAYOUTS = ("contiguous", "strided", "transpose", "readonly", "unaligned")


def source(channels=1, layout="contiguous", size=(11, 13), seed=103):
    shape = size if channels == 1 else (*size, channels)
    image = np.random.default_rng(seed).random(shape, dtype=np.float32)
    image.flat[: min(image.size, 4)] = [0, 1, 0.5, 0.99999][: min(image.size, 4)]
    image.flat[-1] = 1
    if layout == "strided":
        image = image[::2, ::-2]
    elif layout == "transpose":
        image = image.swapaxes(0, 1)
    elif layout == "readonly":
        image.setflags(write=False)
    elif layout == "unaligned":
        target = np.ndarray(
            shape, dtype=np.float32, buffer=bytearray(image.nbytes + 1), offset=1
        )
        target[:] = image
        image = target
    return image


def node_functions(name: str):
    current, original = MODULES[name]
    return (
        getattr(current, name + "_node"),
        getattr(original, FROZEN_NAMES.get(name, name) + "_node"),
    )


def compare(name, args, reference_args=None):
    current, original = node_functions(name)
    saved = [(value, value.copy()) for value in args if isinstance(value, np.ndarray)]
    with np.errstate(all="ignore"):
        expected = original(*(reference_args or args))
        actual = current(*args)
    if isinstance(expected, np.ndarray):
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype == np.float32
        # Upstream's memory order, as np.clip's order K passes it on (D-16).
        assert np.empty_like(actual, order="K").strides == (
            np.empty_like(expected, order="K").strides
        )
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected
    for image, before in saved:
        np.testing.assert_array_equal(image.view(np.uint32), before.view(np.uint32))
    return actual, expected


@pytest.mark.parametrize("mask", range(1, 16))
@pytest.mark.parametrize("layout", LAYOUTS)
def test_combine_rgba(mask, layout):
    inputs = tuple(
        source(layout=layout, seed=107 + i)
        if mask & (1 << i)
        else Color.gray(0.173 + i / 10)
        for i in range(4)
    )
    compare("combine_rgba", inputs)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_optional_alpha(layout):
    image = source(layout=layout)
    result, _ = compare("combine_rgba", (image, Color.gray(0.2), image, None))
    assert np.all(result[:, :, 3] == 1)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("kinds", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_combine_rgb_alpha(channels, kinds, layout):
    rgb = (
        Color(tuple(0.173 + i / 10 for i in range(channels)))
        if kinds[0]
        else source(channels, layout)
    )
    alpha = Color.gray(0.4321) if kinds[1] else source(layout=layout, seed=179)
    compare("merge_transparency", (rgb, alpha))


@pytest.mark.parametrize(
    "channels",
    [
        (1,),
        (2,),
        (3,),
        (4,),
        (5,),
        (1, 1),
        (1, 1, 1),
        (1, 1, 1, 1),
        (3, 1),
        (1, 3),
        (3, 3),
        (4, 4, 4, 4),
    ],
)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_merge(channels, layout):
    images = [source(c, layout, seed=181 + i) for i, c in enumerate(channels)]
    images += [None] * (4 - len(images))
    result, _ = compare("merge_channels", tuple(images))
    assert result.ndim == 3
    assert not any(
        np.shares_memory(result, image) for image in images if image is not None
    )


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("channels", [1, 2, 3, 4, 7])
@pytest.mark.parametrize("axis", list(FlipAxis))
def test_flip(layout, channels, axis):
    image = source(channels, layout)
    actual, _ = compare("flip", (image, axis))
    assert (actual is image) == (axis == FlipAxis.NONE)
    if axis != FlipAxis.NONE:
        assert not np.shares_memory(actual, image)


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("fill", list(ShiftFill))
@pytest.mark.parametrize(
    "offset",
    [(-20, -17), (-1, 0), (0, -1), (0, 0), (1, 0), (0, 1), (2, -3), (13, 11), (20, 17)],
)
@pytest.mark.parametrize("layout", ["contiguous", "strided", "readonly"])
def test_shift(channels, fill, offset, layout):
    if channels == 2 and fill == ShiftFill.TRANSPARENT:
        return
    current, original = MODULES["shift"]
    image = source(channels, layout)
    args = (image, current.ShiftCoordMode.ABSOLUTE, 0, 0, *offset, fill)
    original_args = (image, original.ShiftCoordMode.ABSOLUTE, 0, 0, *offset, fill)
    actual, _ = compare("shift", args, original_args)
    identity = (
        fill == ShiftFill.WRAP
        and offset[0] % image.shape[1] == 0
        and offset[1] % image.shape[0] == 0
    )
    assert (actual is image) == identity


@pytest.mark.parametrize(
    "offset", [(-100, -100), (-50, 50), (-1, 1), (33, 67), (100, 100)]
)
@pytest.mark.parametrize("fill", list(ShiftFill))
def test_relative_shift(offset, fill):
    current, original = MODULES["shift"]
    image = source(4)
    compare(
        "shift",
        (image, current.ShiftCoordMode.RELATIVE, *offset, 29, 31, fill),
        (image, original.ShiftCoordMode.RELATIVE, *offset, 29, 31, fill),
    )


@pytest.mark.parametrize("size", [(1, 1), (1, 7), (5, 1)])
@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("fill", list(ShiftFill))
def test_degenerate_shift(size, channels, fill):
    current, original = MODULES["shift"]
    image = source(channels, size=size)
    for offset in itertools.product((-1, 0, 1), repeat=2):
        compare(
            "shift",
            (image, current.ShiftCoordMode.ABSOLUTE, 0, 0, *offset, fill),
            (image, original.ShiftCoordMode.ABSOLUTE, 0, 0, *offset, fill),
        )


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("fill", list(ShiftFill))
@pytest.mark.parametrize("special", [np.nan, np.inf, -np.inf, -0.0])
def test_nonfinite_shift(channels, fill, special):
    current, original = MODULES["shift"]
    image = source(channels, size=(4, 5))
    image[0, 0] = special
    image[1, 2] = special
    image[-1, -1] = special
    for offset in itertools.product((-1, 0, 1), repeat=2):
        actual, expected = compare(
            "shift",
            (image, current.ShiftCoordMode.ABSOLUTE, 0, 0, *offset, fill),
            (image, original.ShiftCoordMode.ABSOLUTE, 0, 0, *offset, fill),
        )
        if special == 0:
            np.testing.assert_array_equal(
                actual.view(np.uint32), expected.view(np.uint32)
            )


@pytest.mark.parametrize("channels", [2, 5, 128])
def test_nonfinite_shift_remap_channels(channels):
    # OpenCV 5.0's warp_kernels lerps serve 1, 3 and 4 channels; any other count keeps
    # the remap invoker's weighted taps, whose inf, NaN and zero-sign results differ.
    current, original = MODULES["shift"]
    image = source(channels, size=(4, 5))
    for index, special in enumerate([np.inf, -np.inf, np.nan, -0.0]):
        image[index, index] = special
    for offset in itertools.product((-1, 0, 1), repeat=2):
        actual, expected = compare(
            "shift",
            (image, current.ShiftCoordMode.ABSOLUTE, 0, 0, *offset, ShiftFill.BLACK),
            (image, original.ShiftCoordMode.ABSOLUTE, 0, 0, *offset, ShiftFill.BLACK),
        )
        valid = ~np.isnan(expected)
        np.testing.assert_array_equal(
            actual[valid].view(np.uint32), expected[valid].view(np.uint32)
        )


@pytest.mark.parametrize("channels", [1, 3, 4, 5])
@pytest.mark.parametrize("threshold", [0, 1, 25, 50, 99.999, 100, np.nan])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_crop_content(channels, threshold, layout):
    image = source(channels, layout)
    if channels < 4:
        actual, _ = compare("crop_to_content", (image, threshold))
        assert actual is image
        return
    # A content rectangle separated from each boundary and with holes.
    image = image.copy()
    image[:, :, 3] = 0
    image[1:-1, 1:-1, 3] = 1
    image[2, 2, 3] = np.nan
    if layout == "readonly":
        image.setflags(write=False)
    if np.isnan(threshold):
        for module in MODULES["crop_to_content"]:
            with pytest.raises(RuntimeError, match="Crop results in empty image"):
                module.crop_to_content_node(image, threshold)
    else:
        actual, _ = compare("crop_to_content", (image, threshold))
        assert not np.shares_memory(actual, image)


@pytest.mark.parametrize("threshold", [0, 0.1, 50, 99.999, 100, -1])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_bbox(layout, threshold):
    image = source(layout=layout)
    compare("get_bounding_box", (image, threshold))


@pytest.mark.parametrize("name", ["get_bounding_box", "crop_to_content"])
@pytest.mark.parametrize("value", [0, 0.5, np.nan, -np.inf])
def test_empty_bbox(name, value):
    image = np.full(
        (7, 9) if name == "get_bounding_box" else (7, 9, 4), value, dtype=np.float32
    )
    for module in MODULES[name]:
        with pytest.raises(RuntimeError, match="empty"):
            getattr(module, name + "_node")(image, 50)


@pytest.mark.parametrize("threshold", [0.1, 25.1, 99.999, 100])
def test_bbox_threshold_rounding(threshold):
    boundary = np.float32(min(threshold / 100, 0.99999))
    image = np.array(
        [
            [
                np.nextafter(boundary, -np.inf, dtype=np.float32),
                boundary,
                np.nextafter(boundary, np.inf, dtype=np.float32),
            ]
        ],
        dtype=np.float32,
    )
    assert compare("get_bounding_box", (image, threshold))[0] == (2, 0, 1, 1)


def test_singleton_channel_shape():
    image = source()[..., None]
    compare("merge_channels", (image, None, None, None))
    compare("flip", (image, FlipAxis.HORIZONTAL))
    compare("get_bounding_box", (image, 0))
    current, original = MODULES["shift"]
    for fill in ShiftFill:
        compare(
            "shift",
            (image, current.ShiftCoordMode.ABSOLUTE, 0, 0, 1, 1, fill),
            (image, original.ShiftCoordMode.ABSOLUTE, 0, 0, 1, 1, fill),
        )


@pytest.mark.parametrize(
    "name,args,message",
    [
        ("combine_rgba", (Color.gray(0),) * 4, "At least one channels"),
        ("merge_transparency", (Color.gray(0), Color.gray(1)), "At least one input"),
        (
            "combine_rgba",
            (source(), source(size=(3, 4)), source(), None),
            "same resolution",
        ),
        ("merge_transparency", (source(3), source(size=(3, 4))), "same resolution"),
        (
            "merge_channels",
            (source(), source(size=(3, 4)), None, None),
            "same resolution",
        ),
    ],
)
def test_original_failures(name, args, message):
    for function in node_functions(name):
        with pytest.raises((ValueError, AssertionError), match=message):
            function(*args)


def test_constant_colors_do_not_materialize_images(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("An assembly kernel must broadcast constants directly")

    monkeypatch.setattr(Color, "to_image", forbidden)
    image = source()
    native.combine_rgba(image, Color.gray(0.2), Color.gray(0.3), None)
    native.combine_rgb_alpha(Color.bgra((0.1, 0.2, 0.3, 0.4)), image)


BAD_IMAGES = [
    np.array(0, dtype=np.float32),
    np.zeros(3, dtype=np.float32),
    np.zeros((2, 3, 0), dtype=np.float32),
    np.zeros((0, 3), dtype=np.float32),
    np.zeros((2, 0, 3), dtype=np.float32),
    np.zeros((2, 3, 4, 5), dtype=np.float32),
    np.zeros((2, 3), dtype=np.float64),
    np.zeros((2, 3), dtype=">f4"),
]


@pytest.mark.parametrize("image", BAD_IMAGES)
def test_bad_shapes_never_dispatch(image, monkeypatch):
    def forbidden(*args):
        raise AssertionError("Malformed image reached native code")

    for name in ("assemble", "flip", "shift", "bbox", "crop"):
        monkeypatch.setattr(lib(), "cn_channels_" + name, forbidden)
    calls = [
        lambda: native.combine_rgba(image, image, image, None),
        lambda: native.combine_rgb_alpha(image, Color.gray(1)),
        lambda: native.merge_channels(image),
        lambda: native.flip(image, 1),
        lambda: native.shift(image, 0, 0, 0),
        lambda: native.bounding_box(image, 0),
        lambda: native.crop_to_content(image, 0),
    ]
    for call in calls:
        with pytest.raises((TypeError, ValueError)):
            call()


def test_c_abi_rejects_invalid_extents():
    library = lib()
    value = np.ones(1, dtype=np.float32)
    pointer = ptr(value)
    huge = ct.c_size_t(-1).value
    assert library.cn_channels_flip(pointer, pointer, huge, 2, 4, 0) == 2
    assert library.cn_channels_flip(pointer, pointer, 0, 2, 4, 0) == 1
    assert library.cn_channels_shift(pointer, pointer, 1, 1, 1, -2, 0, 2) == 1
    assert library.cn_channels_crop(pointer, pointer, 1, 1, 1, 0, 0, huge, 1, huge) == 1
    # The output's rows can't be shorter than the crop's.
    assert library.cn_channels_crop(pointer, pointer, 1, 2, 1, 0, 0, 2, 1, 1) == 1
    bounds = (ct.c_size_t * 4)()
    assert library.cn_channels_bbox(pointer, 1, 1, 1, 1, 0, bounds) == 1
    desc = (native._Channel * 1)(native._Channel(pointer, huge, 0, 0))
    assert library.cn_channels_assemble(desc, pointer, 2, 1) == 2


def test_parallel_calls_match_originals():
    image = source(4, size=(257, 263))
    image[0, :, 3] = 0
    image[:, 0, 3] = 0
    image.setflags(write=False)
    current, original = MODULES["shift"]
    cases = [
        ("flip", (image, FlipAxis.BOTH), None),
        (
            "shift",
            (image, current.ShiftCoordMode.ABSOLUTE, 0, 0, 7, -11, ShiftFill.WRAP),
            (image, original.ShiftCoordMode.ABSOLUTE, 0, 0, 7, -11, ShiftFill.WRAP),
        ),
        (
            "shift",
            (image, current.ShiftCoordMode.ABSOLUTE, 0, 0, -13, 17, ShiftFill.BLACK),
            (image, original.ShiftCoordMode.ABSOLUTE, 0, 0, -13, 17, ShiftFill.BLACK),
        ),
        ("merge_transparency", (image, image[:, :, 3]), None),
        (
            "combine_rgba",
            (image[:, :, 0], image[:, :, 1], image[:, :, 2], None),
            None,
        ),
        (
            "merge_channels",
            (image[:, :, 0], image[:, :, 1], image[:, :, 2], None),
            None,
        ),
        ("crop_to_content", (image, 0), None),
        ("get_bounding_box", (image[:, :, 3], 0), None),
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(compare, *case) for case in cases * 3]
        for future in futures:
            future.result()
