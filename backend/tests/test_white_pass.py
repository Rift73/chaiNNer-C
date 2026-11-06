"""convenient_upscale's transparency path: the white pass takes the black pass's output on
a tile that gives the model the same input bits, and the image comes out the same bits as
from two full passes."""

from __future__ import annotations

import zlib
from collections import deque
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from spandrel import ImageModelDescriptor

# The blend runs in the native backend, which GitHub's checks do not build.
pytest.importorskip("nodes.impl._chainner_graph")

from api import Progress
from nodes.impl.native_framework_shared import (
    assemble_alpha,
    clip_image,
    recover_alpha,
    recovered_alpha_image,
)
from nodes.impl.native_tensors import cast_numpy
from nodes.impl.onnx.auto_split import onnx_auto_split
from nodes.impl.pytorch import auto_split as torch_split
from nodes.impl.tensorrt.tiling import ShapeBounds, tiled_upscale
from nodes.impl.upscale import white_pass
from nodes.impl.upscale.convenient_upscale import (
    convenient_upscale,
    denoise_and_flatten_alpha,
    with_black_and_white_backgrounds,
)
from nodes.impl.upscale.tiler import ExactTileSize, MaxTileSize, NoTiling, Tiler
from nodes.impl.upscale.white_pass import current_white_pass

SCALE = 2
SIZE = 512
# 256 px tiles at 0, 208 and 256 along each axis: nine tiles
BOUNDS = ShapeBounds(64, 64, 256, 256)
TILES = 9


def float32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """bfloat16's cast: round to nearest, ties to even; a NaN becomes the quiet NaN with
    its sign (the native pass rounds the same way)."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    out = (rounded >> np.uint32(16)).astype(np.uint16)
    nan = (bits & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    out[nan] = np.where(bits[nan] >> np.uint32(31), 0xFFC0, 0x7FC0)
    return out


def bf16_bits_to_float32(x: np.ndarray) -> np.ndarray:
    """Exact: a bfloat16 value is the top half of a float32."""
    return (x.astype(np.uint32) << np.uint32(16)).view(np.float32)


def as_float16(tile: np.ndarray) -> np.ndarray:
    return cast_numpy(tile, np.dtype(np.float16))


def model(input_bits: np.ndarray) -> np.ndarray:
    """Depends on every bit of its input: a tile that differs anywhere comes out shifted."""
    values = (
        bf16_bits_to_float32(input_bits)
        if input_bits.dtype == np.uint16
        else input_bits.astype(np.float32)
    )
    shift = np.float32(zlib.crc32(input_bits.tobytes()) / 2**32 / 8)
    return values.repeat(SCALE, axis=0).repeat(SCALE, axis=1) * 0.75 + shift


class Engine:
    """A TileRunner on an FP16, BF16 or FP32 input."""

    def __init__(self, model_input: Callable[[np.ndarray], np.ndarray]) -> None:
        self.model_input = model_input
        self.queue: deque[np.ndarray] = deque()
        self.runs = 0

    def submit(self, tile: np.ndarray) -> None:
        assert len(self.queue) < 2
        self.runs += 1
        self.queue.append(model(self.model_input(tile)))

    def collect(self, height: int, width: int) -> np.ndarray:
        return self.queue.popleft()[:height, :width]


def picture(low: float = 0.0, seed: int = 0) -> np.ndarray:
    """Opaque 8-bit colours from `low` up."""
    rng = np.random.default_rng(seed)
    levels = rng.integers(round(low * 255), 256, (SIZE, SIZE, 4)).astype(np.float32)
    rgba = levels / 255
    rgba[:, :, 3] = 1
    return rgba


def assert_same_bits(new: np.ndarray, old: np.ndarray) -> None:
    assert new.shape == old.shape
    assert new.tobytes() == old.tobytes()


def tensorrt_upscale(engine: Engine, bounds: ShapeBounds = BOUNDS):
    return lambda i: tiled_upscale(
        i, SCALE, bounds, engine, model_input=engine.model_input
    )


def two_full_passes(
    img: np.ndarray, upscale: Callable[[np.ndarray], np.ndarray], clip: bool = True
) -> np.ndarray:
    """The transparency path with nothing kept, as it ran before."""
    assert current_white_pass() is None
    black, white = with_black_and_white_backgrounds(img)
    black_up, white_up = upscale(black), upscale(white)
    if clip:
        black_up, white_up = clip_image(black_up), clip_image(white_up)
    image = recovered_alpha_image(white_up, black_up)
    if image is not None:
        return image
    alpha = denoise_and_flatten_alpha(recover_alpha(white_up, black_up))
    return assemble_alpha(black_up, alpha, frozen=True)


def run_tensorrt(
    img: np.ndarray,
    model_input: Callable[[np.ndarray], np.ndarray] = as_float16,
    bounds: ShapeBounds = BOUNDS,
) -> int:
    """How many tiles the model ran, checking the image's bits against two full passes."""
    engine = Engine(model_input)
    new = convenient_upscale(img, 3, 3, tensorrt_upscale(engine, bounds))
    assert_same_bits(
        new, two_full_passes(img, tensorrt_upscale(Engine(model_input), bounds))
    )
    return engine.runs


def hole(y: int, x: int, size: int = 8, alpha: float = 0.0) -> np.ndarray:
    img = picture()
    img[y : y + size, x : x + size, 3] = alpha
    return img


@pytest.mark.parametrize(
    "model_input",
    [as_float16, float32_to_bf16_bits],
    ids=["fp16", "bf16"],
)
@pytest.mark.parametrize(
    ("y", "x", "tiles_with_it"),
    [
        (8, 8, 1),  # one corner tile
        (8, 230, 2),  # the band two tiles share, along x
        (230, 230, 4),  # the band four tiles share
        (500, 500, 1),
        (511, 511, 1),
    ],
)
def test_opaque_tiles_run_once(
    model_input: Callable[[np.ndarray], np.ndarray],
    y: int,
    x: int,
    tiles_with_it: int,
):
    runs = run_tensorrt(hole(y, x, size=1), model_input)
    assert runs == TILES + tiles_with_it


def test_alpha_over_every_tile_runs_both_passes():
    img = picture()
    img[:, :, 3] = np.linspace(0.2, 1, SIZE, dtype=np.float32)
    assert run_tensorrt(img) == 2 * TILES


def test_a_hole_in_every_tile_runs_both_passes():
    img = picture()
    for y in (100, 240, 400):
        for x in (100, 240, 400):
            img[y, x, 3] = 0.5
    assert run_tensorrt(img) == 2 * TILES


def test_alpha_near_one_is_judged_by_the_bits():
    # 16-bit transparency a model's FP16 input may or may not hold: the output is the same
    for alpha in (65534 / 65535, 255 / 256, 1 - 2**-24):
        run_tensorrt(hole(8, 8, alpha=alpha))


def test_negative_zero_is_not_zero():
    # Black keeps -0 where white makes +0: equal values, different input bits
    img = hole(8, 8)
    img[400, 400, 0] = -0.0  # in the four tiles at 208 and 256
    assert run_tensorrt(img) == TILES + 1 + 4


def test_nan_and_values_out_of_range_keep_their_bits():
    img = hole(8, 8)
    img[400, 400, 0] = np.nan
    img[300, 400, 1] = 2.5
    img[400, 300, 2] = -0.5
    new = convenient_upscale(img, 3, 3, tensorrt_upscale(Engine(as_float16)))
    assert_same_bits(new, two_full_passes(img, tensorrt_upscale(Engine(as_float16))))


def test_fp32_input_runs_both_passes_where_a_low_bit_differs():
    # A dark colour's white background comes back with its lowest bits lost
    assert run_tensorrt(hole(8, 8), lambda i: i) == 2 * TILES


def fp32_into_fp16(engine: Engine):
    return lambda i: tiled_upscale(
        i, SCALE, BOUNDS, engine, model_input=engine.model_input, opaque_is_same=True
    )


def test_fp32_into_fp16_reuses_a_dark_opaque_tile():
    engine = Engine(lambda i: i)
    convenient_upscale(hole(8, 8), 3, 3, fp32_into_fp16(engine))
    assert engine.runs == TILES + 1


@pytest.mark.parametrize("alpha", [65534 / 65535, 1 - 2**-24])
def test_fp32_into_fp16_runs_both_passes_where_alpha_is_near_one(alpha: float):
    engine = Engine(lambda i: i)
    convenient_upscale(hole(8, 8, alpha=alpha), 3, 3, fp32_into_fp16(engine))
    assert engine.runs == TILES + 1


def test_fp32_input_reuses_bright_tiles():
    # From 0.5 up the white background is the colour itself
    img = picture(low=0.5)
    img[8, 8, 3] = 0
    assert run_tensorrt(img, lambda i: i) == TILES + 1


def test_opaque_image_in_rgba_form_runs_one_pass():
    engine = Engine(as_float16)
    img = picture()
    out = convenient_upscale(img, 3, 3, tensorrt_upscale(engine))
    assert engine.runs == TILES
    assert np.all(out[:, :, 3] == 1)


def test_an_image_of_one_tile_with_a_hole_runs_two():
    bounds = ShapeBounds(64, 64, SIZE, SIZE)
    assert run_tensorrt(hole(8, 8), bounds=bounds) == 2


# 1024 x 1024 output pixels: 7 floats each for the white and RGBA results
RESULTS = 1024 * 1024 * 7 * 4
TILE_OUTPUT = 512 * 512 * 3 * 4
OPAQUE_TILES = TILES - 1


@pytest.mark.parametrize(
    ("tiles_of_room", "spare", "runs"),
    [
        (1, -1, 2 * TILES),
        (1, 0, 2 * TILES - 1),
        (3, 0, 2 * TILES - 3),
        (OPAQUE_TILES, -1, TILES + 2),
        (OPAQUE_TILES, 0, TILES + 1),
    ],
)
def test_every_tile_kept_counts_against_the_memory_read(
    monkeypatch: pytest.MonkeyPatch, tiles_of_room: int, spare: int, runs: int
):
    # One reading for all of them: a burst of tiles keeps no more than that reading has room
    # for, however many come before the next one.
    room = RESULTS + tiles_of_room * TILE_OUTPUT + spare
    available = SimpleNamespace(available=room)
    calls = []

    def memory() -> SimpleNamespace:
        calls.append(1)
        return available

    monkeypatch.setattr(white_pass.psutil, "virtual_memory", memory)
    monkeypatch.setattr(white_pass, "MEMORY_QUERY_SECONDS", 3600.0)
    engine = Engine(as_float16)
    convenient_upscale(hole(8, 8), 3, 3, tensorrt_upscale(engine))
    assert engine.runs == runs
    assert len(calls) == 1


def test_memory_that_falls_stops_the_keeping_at_the_next_reading(
    monkeypatch: pytest.MonkeyPatch,
):
    # Another program takes the memory: seen at the reading after the interval
    clock = [0.0]

    def monotonic() -> float:
        clock[0] += 0.6
        return clock[0]

    readings = [2**40, 0]

    def memory() -> SimpleNamespace:
        return SimpleNamespace(available=readings.pop(0) if readings else 0)

    monkeypatch.setattr(white_pass.time, "monotonic", monotonic)
    monkeypatch.setattr(white_pass.psutil, "virtual_memory", memory)
    engine = Engine(as_float16)
    convenient_upscale(hole(8, 8), 3, 3, tensorrt_upscale(engine))
    # Kept at 0.6 s, the first reading, and at 1.2 s; refused from the reading at 1.8 s
    assert engine.runs == 2 * TILES - 2


def test_each_image_reads_the_memory_again(monkeypatch: pytest.MonkeyPatch):
    # What a batch of images keeps ends with each image: none starts from an old reading
    calls = []

    def memory() -> SimpleNamespace:
        calls.append(1)
        return SimpleNamespace(available=2**40)

    monkeypatch.setattr(white_pass.psutil, "virtual_memory", memory)
    monkeypatch.setattr(white_pass, "MEMORY_QUERY_SECONDS", 3600.0)
    for _ in range(3):
        engine = Engine(as_float16)
        convenient_upscale(hole(8, 8), 3, 3, tensorrt_upscale(engine))
        assert engine.runs == TILES + 1
    assert len(calls) == 3


def test_a_grey_engine_runs_both_passes():
    engine = Engine(as_float16)
    img = hole(8, 8)

    def upscale(i: np.ndarray) -> np.ndarray:
        return tiled_upscale(i, SCALE, BOUNDS, engine, model_input=engine.model_input)

    out = convenient_upscale(img, 1, 1, upscale)
    assert out.shape == (SIZE * SCALE, SIZE * SCALE, 4)
    assert engine.runs == 2 * TILES


def test_a_runner_that_does_not_say_its_input_runs_both_passes():
    engine = Engine(as_float16)
    img = hole(8, 8)
    convenient_upscale(img, 3, 3, lambda i: tiled_upscale(i, SCALE, BOUNDS, engine))
    assert engine.runs == 2 * TILES


def test_the_white_pass_ends_with_the_upscale():
    def failing(_: np.ndarray) -> np.ndarray:
        assert current_white_pass() is not None
        raise OSError("stop here")

    with pytest.raises(OSError, match="stop here"):
        convenient_upscale(hole(8, 8), 3, 3, failing)
    assert current_white_pass() is None


class PyTorchModel:
    """The members of a model pytorch_auto_split uses. Runs out of memory on a tile
    over `limit` pixels."""

    dtype = torch.float32
    device = torch.device("cpu")
    architecture = SimpleNamespace(id="Stand-in", name="Stand-in")
    model = torch.nn.Identity()

    def __init__(self, limit: int = 1024 * 1024) -> None:
        self.limit = limit
        self.runs = 0

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        pixels = t.shape[2] * t.shape[3]
        if pixels > self.limit:
            raise torch.OutOfMemoryError("out of memory")
        self.runs += 1
        out = torch.nn.functional.interpolate(t, scale_factor=SCALE, mode="nearest")
        return (out * 0.75 + zlib.crc32(t.numpy().tobytes()) / 2**32 / 8).float()


def pytorch_upscale(pytorch: PyTorchModel, tiler: Tiler):
    descriptor = cast(ImageModelDescriptor, pytorch)
    return lambda i: torch_split.pytorch_auto_split(
        i, descriptor, torch.device("cpu"), False, tiler, Progress.noop_progress()
    )


def bright_hole(y: int = 8, x: int = 8) -> np.ndarray:
    img = picture(low=0.5)
    img[y, x, 3] = 0
    return img


@pytest.mark.parametrize(
    ("tiler", "fewer"),
    [
        (NoTiling(), False),
        (MaxTileSize(256), True),
        (ExactTileSize((256, 256)), True),
        (ExactTileSize((768, 768)), False),  # padded to the exact size
    ],
    ids=["none", "max256", "exact256", "exact768"],
)
def test_pytorch_tile_modes(tiler: Tiler, fewer: bool):
    img = bright_hole()
    new_model = PyTorchModel()
    new = convenient_upscale(img, 3, 3, pytorch_upscale(new_model, tiler), clip=False)
    old_model = PyTorchModel()
    assert_same_bits(
        new, two_full_passes(img, pytorch_upscale(old_model, tiler), False)
    )
    assert (new_model.runs < old_model.runs) == fewer


def test_pytorch_dark_colours_run_both_passes_in_fp32():
    img = hole(8, 8)
    pytorch = PyTorchModel()
    convenient_upscale(
        img, 3, 3, pytorch_upscale(pytorch, MaxTileSize(256)), clip=False
    )
    assert pytorch.runs == 2 * 4


def test_pytorch_tiles_kept_before_a_split_are_not_used_for_other_tiles():
    # Out of memory halves the tiles: the two passes see the same layout only because
    # the limit is on the tile size, and the output is the two full passes' bits.
    img = bright_hole()
    new_model = PyTorchModel(limit=300 * 300)
    new = convenient_upscale(
        img, 3, 3, pytorch_upscale(new_model, MaxTileSize(SIZE)), clip=False
    )
    old_model = PyTorchModel(limit=300 * 300)
    assert_same_bits(
        new, two_full_passes(img, pytorch_upscale(old_model, MaxTileSize(SIZE)), False)
    )
    assert new_model.runs < old_model.runs


def test_pytorch_a_new_layout_in_the_white_pass_runs_the_model():
    # The white pass runs out of memory where the black pass did not: its smaller tiles
    # find nothing kept, and the image is still made.
    img = bright_hole()
    pytorch = PyTorchModel()
    upscale = pytorch_upscale(pytorch, MaxTileSize(256))
    passes: list[int] = []

    def shrinking(i: np.ndarray) -> np.ndarray:
        passes.append(pytorch.runs)
        if len(passes) == 2:
            pytorch.limit = 200 * 200
        return upscale(i)

    out = convenient_upscale(img, 3, 3, shrinking, clip=False)
    assert out.shape == (SIZE * SCALE, SIZE * SCALE, 4)
    assert pytorch.runs - passes[1] > 4


class Input:
    name = "input"

    def __init__(self, element: str) -> None:
        self.type = f"tensor({element})"


class Output:
    name = "output"

    def __init__(self, element: str = "float") -> None:
        self.type = f"tensor({element})"


class OnnxSession:
    """The members of a session onnx_auto_split uses."""

    def __init__(self, element: str, output: str = "float") -> None:
        self.input = Input(element)
        self.output = Output(output)
        self.runs = 0

    def get_inputs(self):
        return [self.input]

    def get_outputs(self):
        return [self.output]

    def run(self, _: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.runs += 1
        t = feed["input"]
        out = t.repeat(SCALE, axis=2).repeat(SCALE, axis=3).astype(np.float32) * 0.75
        return [out + np.float32(zlib.crc32(t.tobytes()) / 2**32 / 8)]


@pytest.mark.parametrize(
    ("element", "picture_with_hole", "runs"),
    [
        ("float16", lambda: hole(8, 8), 4 + 1),
        ("float", lambda: hole(8, 8), 2 * 4),
        ("float", bright_hole, 4 + 1),
    ],
    ids=["fp16", "fp32-dark", "fp32-bright"],
)
def test_onnx(element: str, picture_with_hole: Callable[[], np.ndarray], runs: int):
    img = picture_with_hole()

    def upscale(session: OnnxSession):
        return lambda i: onnx_auto_split(
            i, cast(Any, session), False, MaxTileSize(256), collect_after=False
        )

    session = OnnxSession(element)
    new = convenient_upscale(img, 3, 3, upscale(session), clip=False)
    assert_same_bits(new, two_full_passes(img, upscale(OnnxSession(element)), False))
    assert session.runs == runs


def test_onnx_fp32_into_fp16_reuses_opaque_tiles():
    session = OnnxSession("float", "float16")
    convenient_upscale(
        hole(8, 8),
        3,
        3,
        lambda i: onnx_auto_split(
            i, cast(Any, session), False, MaxTileSize(256), collect_after=False
        ),
        clip=False,
    )
    assert session.runs == 4 + 1


def test_a_tile_of_another_size_where_one_was_kept_drops_every_kept_tile():
    # Out of memory changed the layout between the passes: nothing kept fits the new
    # tiles, and keeping them would hold a second output image for nothing.
    white = np.full((512, 512, 3), 0.5, np.float32)
    tiles = white_pass.WhitePass(white, np.ones(white.shape[:2], bool))
    output = np.zeros((4, 4, 3), np.float32)
    tiles.keep(0, 0, white[:256, :256], output, lambda t: t)
    tiles.keep(256, 0, white[:256, 256:], output, lambda t: t)
    tiles.start()
    assert tiles.reuse(256, 0, white[:128, 256:384]) is None
    assert tiles.reuse(0, 0, white[:256, :256]) is None


def test_an_edge_tile_of_another_size_keeps_the_other_tiles():
    # The layout's own smaller edge tile, where nothing was kept, shows no change.
    white = np.full((300, 512, 3), 0.5, np.float32)
    tiles = white_pass.WhitePass(white, np.ones(white.shape[:2], bool))
    output = np.zeros((4, 4, 3), np.float32)
    tiles.keep(0, 0, white[:256, :256], output, lambda t: t)
    tiles.start()
    assert tiles.reuse(0, 256, white[256:, :256]) is None
    assert tiles.reuse(0, 0, white[:256, :256]) is output
