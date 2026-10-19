"""The run engine's registry and Python boundary (native/src/run): run_native, native_schema_ids and the Value
conversion (run_engine_value: a value through the engine and back)."""

import gc
import os
import struct
import subprocess
import sys
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

import api
from nodes.impl.color.color import Color
from nodes.impl.image_utils import FlipAxis, normalize
from nodes.impl.native_buffers import freeze_normalized
from nodes.impl.native_graph import graph
from nodes.properties.outputs import ImageOutput
from nodes.utils.seed import Seed

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"

def through(value, normalize=False):
    return graph().run_engine_value(value, normalize)


def address(array: np.ndarray) -> int:
    return array.__array_interface__["data"][0]


def bits(array: np.ndarray) -> tuple:
    return array.dtype.str, array.shape, array.tobytes()


def engine_image(height=5, width=7, channels=3) -> np.ndarray:
    """An array over engine storage: Flip's result."""
    img = np.random.default_rng(height * width).random((height, width, channels), dtype=np.float32)
    return graph().run_native("chainner:image:flip", [img[:, :, 0] if channels == 1 else img, 0])[0]


# Registry


def test_schema_ids_are_sorted_unique_and_registered():
    ids = graph().native_schema_ids()
    assert ids == sorted(set(ids))
    assert "chainner:image:flip" in ids


def test_unknown_schema_is_a_key_error():
    with pytest.raises(KeyError, match="No native entry point for chainner:image:nothing"):
        graph().run_native("chainner:image:nothing", [])


# Images, enforced (run_native's inputs)


@pytest.mark.parametrize("shape", [(4, 6, 3), (4, 6, 4), (4, 6), (4, 6, 1)])
def test_contiguous_float32_is_borrowed_as_itself(shape):
    img = np.zeros(shape, np.float32)
    assert through(img) is img


def test_strided_float32_is_copied_with_its_bits():
    planar = np.random.default_rng(3).random((3, 9, 5), dtype=np.float32)
    planar[0, 0, 0] = -0.0
    planar[1, 2, 3] = np.uint32(0x7FC00042).view(np.float32)
    permuted = planar.transpose(1, 2, 0)
    out = through(permuted)
    assert out is not permuted and out.flags.c_contiguous and not out.flags.writeable
    assert bits(out) == bits(np.ascontiguousarray(permuted))


def test_engine_images_are_read_only_views_kept_alive_by_a_capsule():
    out = engine_image()
    assert type(out) is np.ndarray and out.dtype == np.float32 and out.flags.c_contiguous
    assert not out.flags.writeable and not out.flags.owndata
    assert type(out.base).__name__ == "PyCapsule"
    with pytest.raises(ValueError):
        out[0, 0, 0] = 1
    with pytest.raises(ValueError):
        out.setflags(write=True)


def test_grey_engine_image_is_two_dimensional():
    out = engine_image(channels=1)
    assert out.shape == (5, 7) and out.flags.c_contiguous


def test_engine_image_crosses_back_without_a_copy():
    out = engine_image()
    back = through(out)
    assert address(back) == address(out) and bits(back) == bits(out)
    assert address(through(out, normalize=True)) == address(out)


def test_engine_storage_outlives_the_array_that_produced_it():
    out = engine_image(31, 17, 4)
    expected = bits(out)
    view = out[3:, ::2]
    back = through(out)
    del out
    gc.collect()
    assert bits(back) == expected
    assert bits(view) == bits(np.frombuffer(expected[2], np.float32).reshape(31, 17, 4)[3:, ::2])


POOLED = """
import numpy as np
from nodes.impl.native_graph import graph
g = graph()
assert g.numpy_pool_install()[0] == "installed"
img = np.zeros((256, 256, 4), np.float32)  # 1 MiB, above the pool's threshold
out = g.run_native("chainner:image:flip", [img, 0])[0]
address = out.__array_interface__["data"][0]
assert address % 65536 == 0, address
del out
assert address in [block for block, _ in g.numpy_pool_idle_blocks()]
again = g.run_native("chainner:image:flip", [img, 1])[0]
assert again.__array_interface__["data"][0] == address
del again
assert g.numpy_pool_release()[0] >= 1 << 20
assert g.numpy_pool_idle_blocks() == []
print("pooled")
"""


def test_engine_images_live_in_the_numpy_pool():
    # A fresh interpreter: the pool reads its variables at the first install (test_numpy_pool.py).
    environment = {k: v for k, v in os.environ.items() if k not in ("CHAINNER_C_NUMPY_POOL", "CHAINNER_C_PROFILE")}
    result = subprocess.run(
        [sys.executable, "-B", "-c", POOLED],
        cwd=BACKEND,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0 and result.stdout.strip() == "pooled", result.stderr


def test_borrowed_inputs_are_released():
    img = np.ones((8, 8, 3), np.float32)
    before = sys.getrefcount(img)
    for axis in (0, 1, -1, 2):
        graph().run_native("chainner:image:flip", [img, axis])
    gc.collect()
    assert sys.getrefcount(img) == before


@pytest.mark.parametrize("shape", [(4, 6, 2), (4, 6, 5), (4,), (2, 3, 4, 3)])
def test_unsupported_shapes_are_type_errors(shape):
    with pytest.raises(TypeError, match="1, 3 or 4 channels"):
        through(np.zeros(shape, np.float32))


def test_enforced_images_must_be_float32():
    with pytest.raises(TypeError, match="float32"):
        through(np.zeros((2, 2, 3), np.float64))


# Images, normalized (as ImageOutput.enforce)


def enforced(value: np.ndarray) -> np.ndarray:
    return ImageOutput().enforce(value)


@pytest.mark.parametrize(
    "values",
    [
        np.array([[[-0.5, 1.5, np.nan], [-0.0, 0.25, np.inf]]], np.float32),
        np.array([[[-0.5, 1.5, np.nan], [-0.0, 0.25, -np.inf]]], np.float64),
        np.arange(24, dtype=np.uint8).reshape(2, 4, 3) * 10,
        np.arange(24, dtype=np.uint16).reshape(2, 4, 3) * 2500,
        np.arange(-12, 12, dtype=np.int16).reshape(2, 4, 3),
        np.array([[True, False], [False, True]]),
    ],
)
def test_normalized_conversion_matches_image_output_enforce(values):
    with np.errstate(all="ignore"):
        assert bits(through(values, normalize=True)) == bits(enforced(values))


def test_normalized_strided_input_matches_enforce():
    planar = np.random.default_rng(5).random((4, 7, 6)) * 2 - 0.5
    permuted = planar.transpose(1, 2, 0)
    assert bits(through(permuted, normalize=True)) == bits(enforced(permuted))


def test_normalized_conversion_raises_no_floating_point_errors_as_normalize():
    values = np.array([[[1e300, -1e300, np.nan], [np.inf, 5e-324, -0.0]]], np.float64)
    with np.errstate(all="raise"):
        assert bits(through(values, normalize=True)) == bits(normalize(values))


@pytest.mark.parametrize("shape", [(0, 4, 3), (5, 0), (0, 0, 4)])
def test_empty_images_have_data(shape):
    # An empty engine image still has a data block: kernels refuse a null pointer even for a zero count.
    values = np.zeros(shape, np.uint8)
    out = through(values, normalize=True)
    assert out.dtype == np.float32 and out.shape == shape[:2] + shape[2:] and not out.flags.writeable
    assert bits(out) == bits(normalize(values))
    empty = np.zeros(shape, np.float32)
    assert through(empty) is empty


def test_known_normalized_array_is_borrowed():
    img = freeze_normalized(np.random.default_rng(7).random((6, 5, 4), dtype=np.float32))
    assert through(img, normalize=True) is img


def test_unknown_normalized_array_is_copied_with_its_bits():
    img = np.random.default_rng(8).random((6, 5, 3), dtype=np.float32)
    out = through(img, normalize=True)
    assert out is not img and bits(out) == bits(img)


def test_engine_image_out_of_range_is_converted():
    img = np.array([[[2.0, -1.0, 0.5]]], np.float32)
    out = graph().run_native("chainner:image:flip", [img, 1])[0]
    assert bits(through(out, normalize=True)) == bits(enforced(img))


# Other values


class Kind(Enum):
    A = "first"
    B = 7


@pytest.mark.parametrize(
    "value",
    [None, True, False, 0, -(2**63), 2**63 - 1, "", "naïve ✓ 日本語 \U0001f600", "a\x00b"],
)
def test_scalars_round_trip_exactly(value):
    out = through(value)
    assert type(out) is type(value) and out == value


@pytest.mark.parametrize("value", [0.0, -0.0, 1.5, float("inf"), float("nan"), 5e-324])
def test_floats_round_trip_by_bits(value):
    out = through(value)
    assert type(out) is float and struct.pack("<d", out) == struct.pack("<d", value)


@pytest.mark.parametrize("value", [2**63, -(2**63) - 1, "\ud800 lone", object(), [1, 2], b"bytes"])
def test_other_objects_are_kept_as_themselves(value):
    assert through(value) is value


@pytest.mark.parametrize(
    "path", [Path("C:/images/in.png"), Path("D:/données/日本/ü 😀.png"), Path("relative/dir"), Path(".")]
)
def test_paths_round_trip(path):
    out = through(path)
    assert type(out) is type(path) and out == path and str(out) == str(path)


def test_lone_surrogate_path_round_trips():
    path = Path("C:/odd/\udc80name")
    assert str(through(path)) == str(path)


def test_enum_members_become_their_values():
    assert through(FlipAxis.BOTH) == -1 and type(through(FlipAxis.BOTH)) is int
    assert through(Kind.A) == "first" and through(Kind.B) == 7


@pytest.mark.parametrize("value", [(0.25,), (0.0, 0.5, 1.0), (0.1, 0.2, 0.3, 1.0)])
def test_colors_round_trip(value):
    out = through(Color(value))
    assert type(out) is Color and out.value == value


def test_colors_keep_their_int_channels():
    # Color's _norm keeps an int at the bounds: Color.gray(0).value == (0,); the channel types cross unchanged.
    for color in (Color.gray(0), Color((0, 0.5, 1)), Color((0.25, 1, 0.0))):
        out = through(color)
        assert type(out) is Color and out.value == color.value
        assert [type(c) for c in out.value] == [type(c) for c in color.value]


def test_colors_with_other_channel_types_stay_themselves():
    for color in (Color((True, 0.5)), Color((2**60, 0.5))):
        assert through(color) is color


@pytest.mark.parametrize("value", [0, -1, 42, 2**32, 2**63 - 1, -(2**63)])
def test_seeds_in_int64_cross_as_seeds(value):
    out = through(Seed(value))
    assert type(out) is Seed and out.value == value and type(out.value) is int


@pytest.mark.parametrize("value", [2**63, -(2**63) - 1, 10**40])
def test_seeds_beyond_int64_cross_as_themselves(value):
    seed = Seed(value)
    assert through(seed) is seed


def test_lazy_inputs_cross_as_themselves():
    lazy = api.Lazy.ready(np.zeros((2, 2), np.float32))
    assert through(lazy) is lazy
