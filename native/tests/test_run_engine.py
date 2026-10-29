"""The run engine's registry and Python boundary (native/src/run): run_native, native_schema_ids and the Value
conversion (run_engine_value: a value through the engine and back)."""

import gc
import os
import struct
import subprocess
import sys
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pytest

import api
from nodes.impl.color.color import Color
from nodes.impl.image_utils import FlipAxis
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


@pytest.mark.parametrize("settings", [[("key", 1)], "key", 3])
def test_settings_must_be_a_dict(settings):
    with pytest.raises(TypeError, match="settings must be a dict"):
        graph().run_native("chainner:utility:number", [1], settings=settings)


@pytest.mark.parametrize("storage_dir", ["C:/storage", b"C:/storage", PurePosixPath("storage")])
def test_storage_dir_must_be_a_path(storage_dir):
    with pytest.raises(TypeError, match=r"storage_dir must be a pathlib.Path"):
        graph().run_native("chainner:utility:number", [1], storage_dir=storage_dir)
    assert graph().run_native("chainner:utility:number", [1], settings={}, storage_dir=Path("C:/storage")) == [1]


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
import mmap
import numpy as np
from nodes.impl.native_graph import graph
g = graph()
assert g.numpy_pool_install()[0] == "installed"
img = np.zeros((256, 256, 4), np.float32)  # 1 MiB, above the pool's threshold
out = g.run_native("chainner:image:flip", [img, 0])[0]
address = out.__array_interface__["data"][0]
assert address % mmap.ALLOCATIONGRANULARITY == 0, address  # VirtualAlloc's, or mmap's page
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


# Images, normalized (as the executor normalizes a native image output: normalized_output)


def enforced(value: np.ndarray) -> np.ndarray:
    return ImageOutput().enforce(value)


def special_float32(channels=3) -> np.ndarray:
    """float32 pixels around the clip: out of range, +-0, NaN payloads, infinities, subnormals, exact bounds."""
    values = np.array(
        [-0.5, 1.5, np.nan, -0.0, 0.25, np.inf, -np.inf, 1.0, 0.0, 1e-45, -1e-45, np.nextafter(1, 2, dtype=np.float32)],
        np.float32,
    )
    raw = values.view(np.uint32).copy()
    raw[2] = 0x7FC00042  # a quiet NaN with a payload
    raw = np.concatenate([raw, np.array([0xFFC00001, 0x7F800001], np.uint32)])  # negative and signalling NaN
    flat = raw.view(np.float32)
    count = len(flat) - len(flat) % channels
    return flat[:count].reshape(1, count // channels, channels).copy()


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_normalized_output_matches_image_output_enforce(channels):
    values = special_float32(channels)
    if channels == 1:
        values = values[:, :, 0]
    with np.errstate(all="ignore"):
        expected = bits(enforced(values.copy()))
        assert bits(through(values, normalize=True)) == expected
        engine = graph().run_native("chainner:image:flip", [graph().run_native("chainner:image:flip", [values, 1])[0], 1])[0]
        assert type(engine.base).__name__ == "PyCapsule"
        assert bits(through(engine, normalize=True)) == expected


def test_normalized_strided_input_matches_enforce():
    planar = (np.random.default_rng(5).random((4, 7, 6)) * 2 - 0.5).astype(np.float32)
    permuted = planar.transpose(1, 2, 0)
    assert bits(through(permuted, normalize=True)) == bits(enforced(permuted))


def test_normalized_output_takes_only_engine_images():
    # Every native image output is float32 (an Image); other dtypes never reach the engine's normalization.
    for dtype in (np.float64, np.uint8, np.uint16, np.int32, np.int64, np.bool_, np.float16, ">f4"):
        with pytest.raises(TypeError, match="float32"):
            through(np.zeros((2, 2, 3), dtype), normalize=True)


@pytest.mark.parametrize("shape", [(0, 4, 3), (5, 0), (0, 0, 4)])
def test_empty_images_cross_as_themselves(shape):
    empty = np.zeros(shape, np.float32)
    assert through(empty) is empty
    assert through(empty, normalize=True) is empty


def test_normalized_engine_image_is_itself():
    out = engine_image()
    assert address(through(out, normalize=True)) == address(out)


def test_known_normalized_array_is_borrowed():
    img = freeze_normalized(np.random.default_rng(7).random((6, 5, 4), dtype=np.float32))
    assert through(img, normalize=True) is img


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


class TextSubclass(str):
    pass


@pytest.mark.parametrize(
    "value",
    [np.float64(1.5), np.float64(-0.0), np.str_("x"), TextSubclass("y"), PurePosixPath("a/b"), PureWindowsPath("C:/a")],
)
def test_subclasses_are_kept_as_themselves(value):
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


# The differential harness's comparison (native_differential.comparable)


def test_comparable_is_exact_and_recursive():
    from native_differential import comparable

    nan = struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000001))[0]
    assert comparable([(nan, -0.0)]) == comparable([(nan, -0.0)])
    assert comparable((0.0,)) != comparable((-0.0,)) and comparable({"a": [0.0]}) != comparable({"a": [-0.0]})
    assert comparable(np.float64(1.5)) != comparable(1.5) and comparable(np.float32(np.nan)) == comparable(np.float32(np.nan))
    assert comparable(Path("C:/A")) != comparable(Path("c:/a")) and comparable(Path("a/b")) != comparable(PurePosixPath("a/b"))
    assert comparable([Path("x")]) == comparable([Path("x")]) and comparable("1") != comparable(1)
    frozen = np.zeros((2, 2), np.float32)
    frozen.setflags(write=False)
    assert comparable(frozen) != comparable(np.zeros((2, 2), np.float32)) != comparable(np.zeros((2, 2, 1), np.float32))
    assert comparable(np.zeros(4, np.float32)) != comparable(np.zeros(4, np.int32))
    assert comparable(Color((0.25,))) == comparable(Color((0.25,))) != comparable(Color((0.5,)))
