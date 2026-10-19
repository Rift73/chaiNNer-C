"""Native image_filter entry points (native/src/run/nodes/image_filters.cpp) against their Python bodies."""

import numpy as np
import pytest
from native_differential import assert_native_matches
from native_images import CHANNELS, EMPTY_SHAPES, SIZES, image, seed_of

from nodes.utils.seed import Seed

CASES = [(size, channels) for size in SIZES for channels in CHANNELS]


def make(key: object, size: tuple[int, int], channels: int) -> np.ndarray:
    height, width = size
    return image(seed_of(key, size, channels), height, width, channels)


NOISE_TYPES = ["gaussian", "uniform", "salt_and_pepper", "speckle", "poisson"]


@pytest.mark.parametrize(("size", "channels"), CASES)
@pytest.mark.parametrize("color", ["rgb", "gray"])
@pytest.mark.parametrize("noise_type", NOISE_TYPES)
@pytest.mark.parametrize(("amount", "seed"), [(50.0, 0), (0.0, 7), (100.0, 2**32 - 1), (12.3, 2**32), (-0.0, 5)])
def test_add_noise_matches(amount, seed, noise_type, color, size, channels):
    img = make(("noise", noise_type, color, amount, seed), size, channels)
    assert_native_matches("chainner:image:add_noise", [img, noise_type, color, amount, seed])


def test_add_noise_edge_cases():
    img = image(3, 5, 6, 3)
    for seed in (-1, -(2**40), 2**62 + 12345):
        assert_native_matches("chainner:image:add_noise", [img, "uniform", "rgb", 50.0, seed])
    assert_native_matches("chainner:image:add_noise", [img, "uniform", "rgb", 50.0, Seed(2**70)], declines=True)
    for shape in EMPTY_SHAPES:
        assert_native_matches(
            "chainner:image:add_noise", [np.zeros(shape, np.float32), "gaussian", "gray", 50.0, 1], declines=True
        )


@pytest.mark.parametrize(("size", "channels"), CASES)
@pytest.mark.parametrize(("size_x", "size_y"), [(1, 1), (10, 10), (3, 7), (1024, 1), (40, 1024)])
def test_pixelate_matches(size_x, size_y, size, channels):
    assert_native_matches("chainner:image:pixelate", [make(("pixelate", size_x, size_y), size, channels), size_x, size_y])


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_pixelate_empty_image(shape):
    assert_native_matches("chainner:image:pixelate", [np.zeros(shape, np.float32), 10, 10])


@pytest.mark.parametrize(("size", "channels"), CASES)
@pytest.mark.parametrize(("radius", "amount", "threshold"), [(0.0, 1.0, 0.0), (3.0, 0.0, 0.0), (3.0, 1.0, 0.0),
                                                             (0.5, 100.0, 0.0), (2.5, 2.0, 10.0), (12.0, 1.0, 0.0),
                                                             (1.5, 50.0, 100.0)])
def test_unsharp_mask_matches(radius, amount, threshold, size, channels):
    img = make(("unsharp", radius, amount, threshold), size, channels)
    # fast_gaussian_blur downsamples (cv2.resize) for sigmas of at least 11 on images wider or taller than one pixel.
    declines = radius >= 11 and size != (1, 1)
    assert_native_matches("chainner:image:sharpen", [img, radius, amount, threshold], declines=declines)


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_unsharp_mask_empty_image(shape):
    for radius in (0.0, 3.0, 12.0):
        assert_native_matches("chainner:image:sharpen", [np.zeros(shape, np.float32), radius, 1.0, 0.0])


def test_unsharp_mask_large_sigma_declines():
    assert_native_matches("chainner:image:sharpen", [image(1, 40, 40, 3), 20.0, 1.0, 0.0], declines=True)


@pytest.mark.parametrize(("size", "channels"), CASES)
@pytest.mark.parametrize(("mode", "radius", "contrast"), [(0, 3.0, 1.0), (0, 0.0, 2.5), (0, 7.5, 0.0), (1, 3.0, 1.0),
                                                          (1, 3.0, 57.25)])
def test_high_pass_matches(mode, radius, contrast, size, channels):
    height, width = size
    img = make(("high pass", mode, radius, contrast), size, channels)
    blurred = image(seed_of("blurred", size, channels), height, width, min(channels, 3))
    assert_native_matches("chainner:image:high_pass", [img, mode, radius, blurred, contrast])


def test_high_pass_errors():
    img = image(2, 5, 6, 4)
    cases: list[list[object]] = [
        [img, 1, 3.0, None, 1.0],
        [img, 1, 3.0, image(3, 5, 6, 4), 1.0],
        [img, 1, 3.0, image(3, 6, 5, 3), 1.0],
        [img, 0, 3.0, None, 1.0],
        [np.zeros((0, 5, 3), np.float32), 1, 3.0, np.zeros((0, 5, 3), np.float32), 1.0],
        [np.zeros((0, 5, 4), np.float32), 0, 3.0, None, 1.0],
        [np.zeros((4, 0), np.float32), 0, 0.0, None, 1.0],
    ]
    for inputs in cases:
        assert_native_matches("chainner:image:high_pass", inputs)


KERNELS = [
    "0 0 0\n0 1 0\n0 0 0",
    "1 2 1\n2 4 2\n1 2 1",
    "-1 -1 -1\r\n-1 8 -1\r\n-1 -1 -1\n",
    "0.5 +.25\r1. -2e-1",
    "1e40 -1e39 3.4028236e38",
    "\t1  2 3 4 5 6 7\n",
    "1",
    "\n".join(" ".join(str((i * 7 + j) % 5 - 2) for j in range(7)) for i in range(7)),
]


@pytest.mark.parametrize(("size", "channels"), CASES)
@pytest.mark.parametrize("padding", [0, 1, 5])
@pytest.mark.parametrize("kernel", KERNELS)
def test_convolve_matches(kernel, padding, size, channels):
    img = make(("convolve", kernel, padding), size, channels)
    assert_native_matches("chainner:image:image_convolve", [img, kernel, padding])


def test_convolve_edge_cases():
    img = image(1, 6, 7, 3)
    for text in ("1 2\n3","\n", "a b", "1_0", "inf", "1\n\n2", "0x10", "1 2\n3 4 "):
        assert_native_matches("chainner:image:image_convolve", [img, text, 0], declines=True)
    # 50 coefficients and more: OpenCV's DFT path (without SSE3) or spatial path, both cv2 here.
    big = "\n".join(" ".join("1" for _ in range(12)) for _ in range(12))
    assert_native_matches("chainner:image:image_convolve", [img, big, 0], declines=True)
    assert_native_matches("chainner:image:image_convolve", [img, "1", 2**31])
    for shape in EMPTY_SHAPES:
        assert_native_matches("chainner:image:image_convolve", [np.zeros(shape, np.float32), "1 2", 0])


@pytest.mark.parametrize("text", ["1e400", "-1e400", "-1e-400", "1e-0400", "+0.1e310 -00.10e-323\n1 -1e400"])
def test_convolve_coefficients_beyond_the_doubles(text):
    # float() gives +-inf above the doubles' range and +-0 below it.
    assert_native_matches("chainner:image:image_convolve", [image(2, 6, 7, 3), text, 0])


@pytest.mark.parametrize(("size", "channels"), CASES)
@pytest.mark.parametrize("kernel_type", [0, 1])
@pytest.mark.parametrize(("amount", "adaptive", "bias"), [(0.0, False, 2.0), (2.0, False, 2.0), (37.5, False, 2.0),
                                                          (2.0, True, 2.0), (100.0, True, 1.0), (0.5, True, 3.0)])
def test_high_boost_filter_matches(amount, adaptive, bias, kernel_type, size, channels):
    img = make(("high boost", amount, adaptive, bias, kernel_type), size, channels)
    assert_native_matches("chainner:image:sharpen_hbf", [img, kernel_type, amount, adaptive, bias])


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_high_boost_filter_empty_image(shape):
    for adaptive in (False, True):
        assert_native_matches("chainner:image:sharpen_hbf", [np.zeros(shape, np.float32), 1, 2.0, adaptive, 2.0])


@pytest.mark.parametrize(("size", "channels"), CASES)
@pytest.mark.parametrize("component", [1, 2, 3])
@pytest.mark.parametrize("algorithm", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_edge_detection_matches(algorithm, component, size, channels):
    if component != 1 and algorithm > 5:
        return
    img = make(("edges", algorithm, component), size, channels)
    for amount, radius_1, radius_2 in ((1.0, 1.0, 2.0), (3.75, 0.0, 4.5), (10.0, 2.5, 0.0)):
        if algorithm != 9 and (radius_1, radius_2) != (1.0, 2.0):
            continue
        assert_native_matches("chainner:image:edge_detection", [img, amount, algorithm, component, radius_1, radius_2])


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
@pytest.mark.parametrize("algorithm", [1, 6, 8, 9])
def test_edge_detection_empty_image(algorithm, shape):
    for radius in (0.0, 1.0):
        assert_native_matches("chainner:image:edge_detection", [np.zeros(shape, np.float32), 1.0, algorithm, 1, radius, radius])


BOX_RADII = [(0.0, 0.0), (1.0, 1.0), (3.0, 0.0), (0.5, 0.5), (2.5, 7.0), (16.0, 2.5), (1.5, 30.0), (20.0, 15.0),
             (200.5, 1.0), (250.5, 3.5), (1000.0, 1000.0), (99.5, 99.5), (69.9, 69.9)]


@pytest.mark.parametrize(("size", "channels"), CASES)
@pytest.mark.parametrize(("radius_x", "radius_y"), BOX_RADII)
def test_box_blur_matches(radius_x, radius_y, size, channels):
    img = make(("box", radius_x, radius_y), size, channels)
    # An average radius of 70 and more takes the spectral filter (cv2.dft).
    declines = radius_x == radius_y == 99.5
    assert_native_matches("chainner:image:blur", [img, radius_x, radius_y], declines=declines)


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
@pytest.mark.parametrize(("radius_x", "radius_y"), BOX_RADII)
def test_box_blur_empty_image(radius_x, radius_y, shape):
    assert_native_matches("chainner:image:blur", [np.zeros(shape, np.float32), radius_x, radius_y])


@pytest.mark.parametrize(("size", "channels"), CASES)
@pytest.mark.parametrize("radius", [0, 1, 2, 3, 7, 40])
def test_median_blur_matches(radius, size, channels):
    assert_native_matches("chainner:image:median_blur", [make(("median", radius), size, channels), radius])


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_median_blur_empty_image(shape):
    for radius in (0, 1, 5):
        assert_native_matches("chainner:image:median_blur", [np.zeros(shape, np.float32), radius])


# (algorithm, threshold map, diffusion map, history length)
DITHERS = [("None", "B16", "FS", 16), ("Ordered", "B2", "FS", 16), ("Ordered", "B4", "FS", 16),
           ("Ordered", "B8", "FS", 16), ("Ordered", "B16", "FS", 16), ("Riemersma", "B16", "FS", 2),
           ("Riemersma", "B16", "FS", 37)] + [("Diffusion", "B16", m, 16) for m in ("FS", "JJN", "ST", "A", "B", "S", "S2", "SL")]


@pytest.mark.parametrize(("size", "channels"), CASES)
@pytest.mark.parametrize("colors", [2, 8, 255])
@pytest.mark.parametrize("dither", DITHERS)
def test_dither_matches(dither, colors, size, channels):
    img = make(("dither", dither, colors), size, channels)
    assert_native_matches("chainner:image:dither", [img, colors, *dither])


def test_dither_edge_cases():
    img = image(5, 6, 7, 3)
    for colors in (2**32 - 1, 2**32, 2**70):
        assert_native_matches("chainner:image:dither", [img, colors, "Ordered", "B4", "FS", 16])
    for history in (2**32 - 1, 2**32, 2**70):
        assert_native_matches("chainner:image:dither", [img, 4, "Riemersma", "B4", "FS", history])
    for shape in EMPTY_SHAPES:
        assert_native_matches("chainner:image:dither", [np.zeros(shape, np.float32), 2**70, "Riemersma", "B4", "FS", 2**70])


@pytest.mark.parametrize(("size", "channels"), CASES)
@pytest.mark.parametrize(("lower", "upper"), [(100, 300), (0, 0), (50, 60), (300, 100), (2**40, 5)])
def test_canny_matches(lower, upper, size, channels):
    assert_native_matches("chainner:image:canny_edge_detection", [make(("canny", lower, upper), size, channels), lower, upper])


def test_canny_empty_image_declines():
    for shape in EMPTY_SHAPES:
        assert_native_matches("chainner:image:canny_edge_detection", [np.zeros(shape, np.float32), 100, 300], declines=True)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("spread", [1, 4, 37])
def test_distance_transform_matches(spread, size):
    height, width = size
    img = image(seed_of("sdf", spread, size), height, width, 1)
    img = np.where(img > 0.5, 1.0, img).astype(np.float32)
    assert_native_matches("chainner:image:distance_transform", [img, spread, True])
    # The binary SDF's distances come from cv2.distanceTransform under IPP (OpenCV's default).
    assert_native_matches("chainner:image:distance_transform", [img, spread, False], declines=True)


def test_distance_transform_empty_image():
    for shape in [(0, 5), (4, 0)]:
        for esdf in (False, True):
            assert_native_matches("chainner:image:distance_transform", [np.zeros(shape, np.float32), 4, esdf])


@pytest.mark.parametrize(("size", "channels"), CASES)
@pytest.mark.parametrize(("strength", "color", "patch", "search"), [(3.0, 3.0, 3, 10), (0.0, 0.0, 1, 1),
                                                                    (12.5, 50.0, 2, 4), (50.0, 7.5, 30, 2)])
def test_denoise_matches(strength, color, patch, search, size, channels):
    if size == (64, 257) and patch > 3:
        return
    img = make(("denoise", strength, patch), size, channels)
    assert_native_matches("chainner:image:fast_nlmeans", [img, strength, color, patch, search])


@pytest.mark.parametrize("shape", EMPTY_SHAPES)
def test_denoise_empty_image(shape):
    assert_native_matches("chainner:image:fast_nlmeans", [np.zeros(shape, np.float32), 3.0, 3.0, 3, 10])


@pytest.mark.parametrize("channels", [3, 4])
@pytest.mark.parametrize(("size", "reference"), [((4, 6), (2, 3)), ((17, 31), (17, 31)), ((64, 64), (8, 16)),
                                                 ((1, 1), (1, 1))])
@pytest.mark.parametrize(("radius", "scale"), [(1, 0.0), (2, 50.0), (0, 100.0), (3, 12.5)])
def test_quantize_to_reference_matches(radius, scale, size, reference, channels):
    img = make(("quantize", size), size, channels)
    ref = make(("reference", reference), reference, channels)
    assert_native_matches("chainner:image:quantize_to_referece", [img, ref, radius, scale])


def test_quantize_to_reference_errors():
    a = image(1, 6, 8, 3)
    cases: list[list[object]] = [
        [a, image(2, 3, 4, 4), 1, 10.0],
        [a, image(2, 7, 4, 3), 1, 10.0],
        [a, image(2, 4, 4, 3), 1, 10.0],
        [a, image(2, 3, 9, 3), 1, 10.0],
        [a, image(2, 3, 3, 3), 1, 10.0],
        [a, np.zeros((0, 4, 3), np.float32), 1, 10.0],
        [a, np.zeros((3, 0, 3), np.float32), 1, 10.0],
        [np.zeros((0, 8, 3), np.float32), np.zeros((0, 4, 3), np.float32), 1, 10.0],
        [np.zeros((6, 0, 3), np.float32), np.zeros((3, 0, 3), np.float32), 1, 10.0],
    ]
    for inputs in cases:
        assert_native_matches("chainner:image:quantize_to_referece", inputs)


@pytest.mark.parametrize("size", SIZES[1:])
@pytest.mark.parametrize(("channels", "reference_channels"), [(3, 3), (4, 3), (3, 4), (4, 4)])
@pytest.mark.parametrize("colorspace", ["L*a*b*", "RGB"])
@pytest.mark.parametrize(("overflow", "reciprocal"), [(1, True), (0, True), (1, False), (0, False)])
def test_color_transfer_matches(overflow, reciprocal, colorspace, channels, reference_channels, size):
    img = make(("transfer", size), size, channels)
    ref = make(("transfer reference", size), (size[1], size[0]), reference_channels)
    if channels == 4:
        img[: size[0] // 2, :, 3] = 0
    assert_native_matches("chainner:image:color_transfer", [img, ref, "mean_std", colorspace, overflow, reciprocal])


def test_color_transfer_declines():
    img, ref = image(1, 9, 8, 3), image(2, 7, 6, 4)
    for algorithm in ("linear_histogram", "principal_color"):
        assert_native_matches("chainner:image:color_transfer", [img, ref, algorithm, "RGB", 1, True], declines=True)
    empty = np.zeros((0, 5, 3), np.float32)
    assert_native_matches("chainner:image:color_transfer", [empty, ref, "mean_std", "RGB", 1, True], declines=True)
    clear = image(3, 5, 6, 4)
    clear[:, :, 3] = 0
    assert_native_matches("chainner:image:color_transfer", [clear, ref, "mean_std", "L*a*b*", 0, True])
