"""Critical CPU checks for Guided training counts and exact patch preparation."""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from api import Aborted, Progress
from nodes.impl.pytorch.pix_transform import pix_transform as guided


@pytest.mark.parametrize(
    ("iterations", "batch_size", "expected_steps"),
    [(1, 32, 1), (3, 7, 3), (4, 24, 3)],
)
def test_guided_optimizer_steps(monkeypatch, iterations, batch_size, expected_steps):
    class TinyNet(torch.nn.Module):
        def __init__(self, channels_in, weights_regularizer):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.params_with_regularizer = [{"params": [self.scale]}]

        def forward(self, value):
            assert value.device.type == "cpu"
            return value[:, :1] * self.scale

    steps = []
    original_step = guided.optim.Adam.step

    def step(optimizer, *args, **kwargs):
        steps.append(1)
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(guided, "PixTransformNet", TinyNet)
    monkeypatch.setattr(guided.optim.Adam, "step", step)
    source = np.arange(64, dtype=np.float32).reshape(8, 8)
    guide = np.arange(256, dtype=np.float32).reshape(1, 16, 16)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        result = guided.pix_transform(
            source,
            guide,
            torch.device("cpu"),
            guided.Params(iteration=iterations, batch_size=batch_size),
            Progress.noop_progress(),
        )
    assert len(steps) == expected_steps
    assert result.shape == (16, 16)


@pytest.mark.parametrize(("m", "d"), [(1, 2), (2, 1), (3, 2)])
def test_guided_patch_order_bits_and_rng(m, d):
    # Isolate the actual preparation block so m=1 does not hit the unrelated
    # source.squeeze() scalar limitation, and no network or loader is run.
    function = ast.parse(inspect.getsource(guided.pix_transform)).body[0]
    assert isinstance(function, ast.FunctionDef)
    assignments = {
        target.id: index
        for index, statement in enumerate(function.body)
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
    }
    preparation = ast.Module(
        body=function.body[assignments["guide_patches"] : assignments["train_data"]],
        type_ignores=[],
    )
    guide = torch.arange(2 * (m * d) ** 2, dtype=torch.float32).reshape(2, m * d, m * d)
    source = torch.arange(m * m, dtype=torch.float32).reshape(m, m)
    guide.view(torch.int32).flatten()[:2] = torch.tensor(
        [0x7FC12345, -2147483648], dtype=torch.int32
    )
    source.view(torch.int32).flatten()[0] = -2147483648
    expected_guide = torch.zeros((m * m, 2, d, d))
    expected_source = torch.zeros((m * m, 1))
    before = torch.get_rng_state().clone()
    for i in range(m):
        for j in range(m):
            expected_guide[j + i * m] = guide[
                :, i * d : (i + 1) * d, j * d : (j + 1) * d
            ]
            expected_source[j + i * m] = source[i : i + 1, j : j + 1]
    namespace: dict[str, Any] = {
        "torch": torch,
        "guide_tensor": guide,
        "source_tensor": source,
        "m": m,
        "d": d,
        "device": torch.device("cpu"),
    }
    exec(compile(preparation, guided.__file__, "exec"), namespace)
    for name, expected in (
        ("guide_patches", expected_guide),
        ("source_pixels", expected_source),
    ):
        actual = namespace[name]
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("channels", [1, 3], ids=["grayscale", "rgb-control"])
def test_guided_non_square_tile_assembly(monkeypatch, channels):
    from nodes.impl.pytorch.pix_transform import auto_split as guided_split
    from nodes.impl.upscale.grayscale import SplitMode
    from nodes.properties.outputs import ImageOutput

    calls = []

    def transform(source_img, guide_img, device, params, progress):
        assert device.type == "cpu"
        assert source_img.ndim == 2
        assert guide_img.shape == (channels, *[size * 2 for size in source_img.shape])
        calls.append(source_img.shape)
        return source_img.repeat(2, axis=0).repeat(2, axis=1)

    monkeypatch.setattr(guided_split, "pix_transform", transform)
    y, x, c = np.indices((64, 96, channels))
    source = ((3 * y + 5 * x + 17 * c) % 251).astype(np.float32) / 256
    expected = source.repeat(2, axis=0).repeat(2, axis=1)
    result = guided_split.pix_transform_auto_split(
        source,
        expected,
        torch.device("cpu"),
        guided.Params(),
        SplitMode.RGB,
        2048,
        Progress.noop_progress(),
    )
    # Exercise the public grayscale output convention as well as the tiler boundary.
    result = ImageOutput().enforce(result)
    assert len(calls) >= 2 * channels
    if channels == 1:
        expected = expected[:, :, 0]
    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-7)


class StopAtStep(Progress):
    """A Stop pressed while the `step`th optimizer step is due; every step is counted."""

    def __init__(self, step: int):
        self.step = step
        self.steps = 0

    @property
    def aborted(self) -> bool:
        return self.steps >= self.step

    @property
    def paused(self) -> bool:
        return False

    def suspend(self) -> None:
        self.steps += 1
        self.check_aborted()

    def set_progress(self, progress: float) -> None:
        pass

    def sub_progress(self, offset: float, length: float) -> Progress:
        return self


def test_stop_ends_the_training():
    from nodes.impl.pytorch.pix_transform.auto_split import pix_transform_auto_split
    from nodes.impl.upscale.grayscale import SplitMode

    rng = np.random.default_rng(5)
    source = rng.random((16, 16, 3), dtype=np.float32)
    guide = rng.random((32, 32, 3), dtype=np.float32)
    stop = StopAtStep(3)
    with pytest.raises(Aborted):
        pix_transform_auto_split(
            source,
            guide,
            torch.device("cpu"),
            guided.Params(iteration=100_000),
            SplitMode.RGB,
            2048,
            stop,
        )
    assert stop.steps == 3


def test_stop_reaches_the_prediction():
    # The one optimizer step, then the first band of the prediction.
    rng = np.random.default_rng(6)
    stop = StopAtStep(2)
    with pytest.raises(Aborted):
        guided.pix_transform(
            rng.random((16, 16), dtype=np.float32),
            rng.random((3, 32, 32), dtype=np.float32),
            torch.device("cpu"),
            guided.Params(iteration=1),
            stop,
        )
    assert stop.steps == 2


def test_the_node_hands_its_context_to_the_training(monkeypatch):
    from nodes.impl.upscale.grayscale import SplitMode
    from packages.chaiNNer_pytorch.pytorch.processing import guided_upscale

    seen = []

    def auto_split(source, guide, device, params, split_mode, tile_size, progress):
        seen.append((tile_size, progress))
        return guide

    class Context:
        def add_cleanup(self, fn, after="chain"):
            pass

    settings = SimpleNamespace(device=torch.device("cpu"), force_cache_wipe=False)
    monkeypatch.setattr(guided_upscale, "pix_transform_auto_split", auto_split)
    monkeypatch.setattr(guided_upscale, "get_settings", lambda _: settings)
    context: Any = Context()
    image = np.zeros((4, 4, 3), np.float32)
    guided_upscale.guided_upscale_node(context, image, image, 1, SplitMode.RGB, 123)
    assert seen == [(123, context)]


def test_the_tile_size_defaults_to_80_up_to_2048_and_old_chains_get_it():
    from nodes.properties.inputs import NumberInput
    from packages.chaiNNer_pytorch.pytorch import processing_group

    data = next(
        node
        for node in processing_group.nodes
        if node.schema_id == "chainner:pytorch:guided_upscale"
    )
    tile = next(i for i in data.inputs if i.id == 4)
    assert isinstance(tile, NumberInput)
    assert (tile.label, tile.default, tile.min, tile.max) == ("Tile Size", 80, 16, 2048)
    assert [i.id for i in data.inputs] == [0, 1, 2, 3, 4]


@pytest.mark.parametrize(("tile", "calls"), [(2048, 1), (64, 1), (48, 4), (16, 49)])
def test_the_tile_size_limits_the_tiles(monkeypatch, tile, calls):
    from nodes.impl.pytorch.pix_transform import auto_split as guided_split
    from nodes.impl.upscale.grayscale import SplitMode

    sizes = []

    def transform(source_img, guide_img, device, params, progress):
        sizes.append(source_img.shape)
        return source_img.repeat(2, axis=0).repeat(2, axis=1)

    monkeypatch.setattr(guided_split, "pix_transform", transform)
    y, x = np.indices((64, 64))
    source = ((3 * y + 5 * x) % 251).astype(np.float32) / 256
    expected = source.repeat(2, axis=0).repeat(2, axis=1)
    result = guided_split.pix_transform_auto_split(
        source,
        expected,
        torch.device("cpu"),
        guided.Params(),
        SplitMode.RGB,
        tile,
        Progress.noop_progress(),
    )
    assert max(max(size) for size in sizes) <= tile
    np.testing.assert_allclose(result[:, :, 0], expected, rtol=0, atol=2e-7)
    assert len(sizes) == calls


def test_the_prediction_runs_a_bounded_number_of_pixels_at_a_time(monkeypatch):
    # The network holds 24 KB a pixel at its widest: a whole 1024x1024 guide took 25 GiB,
    # after the training, whose work an out-of-memory then threw away.
    pixels = []

    class Recording(torch.nn.Module):
        def __init__(self, channels_in, weights_regularizer):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.params_with_regularizer = [{"params": [self.scale]}]

        def forward(self, value):
            if not self.training:
                pixels.append(value.shape[2] * value.shape[3])
            return value[:, :1] * self.scale

    monkeypatch.setattr(guided, "PixTransformNet", Recording)
    rng = np.random.default_rng(9)
    result = guided.pix_transform(
        rng.random((512, 512), dtype=np.float32),
        rng.random((1, 1024, 1024), dtype=np.float32),
        torch.device("cpu"),
        guided.Params(iteration=1),
        Progress.noop_progress(),
    )
    assert result.shape == (1024, 1024)
    assert sum(pixels) == 1024 * 1024
    assert max(pixels) <= 1 << 14


@pytest.mark.parametrize("side", [64, 18])
def test_the_bands_of_the_prediction_match_the_whole_guide(monkeypatch, side):
    # Every layer of the network is 1x1, so no pixel depends on its neighbours; a band's
    # size can still change a value by a float32 step (5.96e-8), through the vector paths.
    rng = np.random.default_rng(11)
    source = rng.random((side // 2, side // 2), dtype=np.float32)
    guide = rng.random((3, side, side), dtype=np.float32)

    def run(band_pixels: int) -> np.ndarray:
        monkeypatch.setattr(guided, "PREDICTION_PIXELS", band_pixels)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(23)
            return guided.pix_transform(
                source,
                guide,
                torch.device("cpu"),
                guided.Params(iteration=2),
                Progress.noop_progress(),
            )

    whole = run(side * side)
    for band_pixels in (1, side, 3 * side + 1):
        np.testing.assert_allclose(run(band_pixels), whole, rtol=0, atol=1e-7)
