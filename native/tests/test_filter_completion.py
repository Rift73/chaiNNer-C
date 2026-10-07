"""Final Box coefficient and filter channel-assembly CPU contracts."""

from __future__ import annotations

import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from test_adjustment_complete import ORIGINAL_UNSHARP, exact, layout, load_body
from test_normalized_outputs import RAW_BOX, assert_enforced_equal

from nodes.impl import native_box, native_channels
from nodes.impl.native import ptr

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/src/packages/chaiNNer_standard/image_filter"
REFERENCES = Path(__file__).parent
BOX = load_body(BASE / "blur/box_blur.py")
ORIGINAL_BOX = load_body(
    REFERENCES / "reference_convolution/blur/box_blur_installed.py"
)
EDGE = load_body(BASE / "miscellaneous/edge_detection.py")
ORIGINAL_EDGE = load_body(
    REFERENCES / "reference_filters/miscellaneous/edge_detection.py"
)
HIGH = load_body(BASE / "miscellaneous/high_pass.py")
ORIGINAL_HIGH = load_body(
    REFERENCES
    / "reference_analysis/packages/chaiNNer_standard/image_filter/miscellaneous/high_pass.py"
)
vars(ORIGINAL_EDGE).update(fast_gaussian_blur=ORIGINAL_UNSHARP.fast_gaussian_blur)
vars(ORIGINAL_HIGH).update(fast_gaussian_blur=ORIGINAL_UNSHARP.fast_gaussian_blur)


@pytest.mark.parametrize("rx", [0, 1e-50, 0.1, 1, 1.7, 69.99, 70.1, 200, 200.1, 1000])
@pytest.mark.parametrize("ry", [0, 0.2, 1, 35.3, 70.1, 200.9])
def test_box_coefficients_exact(rx, ry):
    exact(native_box.kernel_2d(rx, ry), ORIGINAL_BOX.get_kernel_2d(rx, ry))


@pytest.mark.parametrize(
    "radii",
    [
        (69.9, 69.9),
        (70.1, 70.1),
        (73.2, 169.7),
        (77.1, 102.3),
        (199.9, 200),
        (200.1, 169.7),
    ],
)
@pytest.mark.parametrize("channels", [1, 3, 7])
@pytest.mark.parametrize("pattern", ["random", "half", "negative_zero"])
def test_box_branch_boundaries(radii, channels, pattern):
    shape = (13, 17) if channels == 1 else (13, 17, channels)
    image = np.random.default_rng(591).random(shape, dtype=np.float32)
    if pattern == "half":
        image.fill(0.5)
    elif pattern == "negative_zero":
        image.fill(-0.0)
    # As the user receives it (ImageOutput.enforce of both returns), and the helpers'
    # raw results (RAW_BOX) for kernel parity, each bitwise (SP4b Task 3b2).
    expected = ORIGINAL_BOX.box_blur_node(image, *radii)
    assert_enforced_equal(BOX.box_blur_node(image, *radii), expected)
    exact(RAW_BOX.box_blur_node(image, *radii), expected)


@pytest.mark.parametrize("channels", [1, 3, 4, 7, 513])
@pytest.mark.parametrize("algorithm", range(1, 10))
@pytest.mark.parametrize("kind", ["plain", "reverse", "unaligned"])
def test_edge_all_channels(channels, algorithm, kind):
    shape = (7, 11) if channels == 1 else (7, 11, channels)
    image = layout(np.random.default_rng(914).random(shape, dtype=np.float32), kind)
    before = image.copy()
    expected = ORIGINAL_EDGE.edge_detection_node(
        image,
        1.3,
        ORIGINAL_EDGE.Algorithm(algorithm),
        ORIGINAL_EDGE.GradientComponent.MAGNITUDE,
        0.7,
        1.3,
    )
    actual = EDGE.edge_detection_node(
        image,
        1.3,
        EDGE.Algorithm(algorithm),
        EDGE.GradientComponent.MAGNITUDE,
        0.7,
        1.3,
    )
    exact(actual, expected)
    exact(image, before)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("kind", ["plain", "reverse", "unaligned"])
@pytest.mark.parametrize("radius", [0, 0.1, 1.7, 19.5])
def test_high_pass_alpha(channels, kind, radius):
    shape = (11, 19) if channels == 1 else (11, 19, channels)
    image = layout(np.random.default_rng(934).random(shape, dtype=np.float32), kind)
    exact(
        HIGH.high_pass_node(image, HIGH.BlurMode.GAUSSIAN, radius, None, 0.7),
        ORIGINAL_HIGH.high_pass_node(
            image, ORIGINAL_HIGH.BlurMode.GAUSSIAN, radius, None, 0.7
        ),
    )


@pytest.mark.parametrize("channels", [1, 3, 4, 7, 65, 513])
@pytest.mark.parametrize(
    "kind", ["plain", "rows", "columns", "reverse", "fortran", "unaligned"]
)
def test_concatenation_preserves_bits(channels, kind):
    values = np.array([-0.0, 0.0, np.nan, np.inf, -np.inf, 1], np.float32)
    rng = np.random.default_rng(983)
    image = layout(values[rng.integers(0, len(values), (5, 7, channels))], kind)
    alpha = layout(values[rng.integers(0, len(values), (5, 7))], kind)
    expected = np.dstack((image, alpha))
    exact(native_channels.concatenate_channels(image, alpha), expected)


def test_coefficient_and_assembly_concurrency_without_numpy_loops(monkeypatch):
    coefficients = ORIGINAL_BOX.get_kernel_2d(73.2, 169.7)
    image = np.random.default_rng(241).random((131, 137, 17), dtype=np.float32)
    expected = np.dstack((image, image[:, :, :1]))

    def forbidden(*_args, **_kwargs):
        pytest.fail("Coefficient/assembly image loops must remain in C")

    monkeypatch.setattr(np, "ones", forbidden)
    monkeypatch.setattr(np, "dstack", forbidden)
    with ThreadPoolExecutor(max_workers=4) as pool:
        output = list(
            pool.map(
                lambda _: (
                    BOX.get_kernel_2d(73.2, 169.7),
                    native_channels.concatenate_channels(image, image[:, :, :1]),
                ),
                range(8),
            )
        )
    for kernel, combined in output:
        exact(kernel, coefficients)
        exact(combined, expected)


def test_box_and_assembly_checked_boundaries():
    for radius in (-1, 1000.1, np.inf, np.nan):
        with pytest.raises(ValueError):
            native_box.kernel_2d(radius, 1)
    result = np.empty((3, 3), np.float32)
    call = native_box._lib.cn_box_kernel_2d
    assert call(None, 9, 1, 1) == 1
    assert call(ptr(result), 8, 1, 1) == 1
    assert call(ct.cast(result.ctypes.data + 1, ct.POINTER(ct.c_float)), 9, 1, 1) == 1
    source = np.ones((5, 7), np.float32)
    descriptors = (native_channels._Channel * 1)(
        native_channels._Channel(ptr(source), 1, 0, 0)
    )
    assemble = native_channels._lib.cn_channels_assemble
    assert assemble(descriptors, ptr(source), source.size, 1) == 1
    unaligned = ct.cast(source.ctypes.data + 1, ct.POINTER(ct.c_float))
    assert assemble(descriptors, unaligned, source.size, 1) == 1
    assert assemble(descriptors, ptr(source), ct.c_size_t(-1).value, 1) == 2
    descriptors[0].data = unaligned
    assert assemble(descriptors, ptr(result), source.size, 1) == 1
