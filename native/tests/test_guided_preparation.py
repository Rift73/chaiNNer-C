"""Critical CPU checks for Guided training counts and exact patch preparation."""

from __future__ import annotations

import ast
import inspect
from typing import Any

import numpy as np
import pytest
import torch

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
