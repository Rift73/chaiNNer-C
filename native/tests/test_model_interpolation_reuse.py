"""Tiny CPU tensor tests; validation deliberately mutates its model weights."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_framework_optimizations import INTERP, REF, SOURCE, functions


@pytest.mark.parametrize("amount", [0, 25, 50, 75, 100])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("mutate", [False, True])
@pytest.mark.parametrize("budget", [0, 1])
def test_exact_weights_and_fresh_final_model(amount, dtype, mutate, budget):
    outcomes, counts = [], []
    for root in (REF, SOURCE):
        models, blends = [], []

        class Model:
            def __init__(self, state):
                self.model = self
                self.state = state
                self.device = torch.device("cpu")
                self.input_channels = 3
                self.size_requirements = SimpleNamespace(minimum=3, multiple_of=1)
                self.forward_calls = 0

            def state_dict(self):
                return self.state

            def __call__(self, image):
                self.forward_calls += 1
                if mutate:
                    for weight in self.state.values():
                        weight.add_(123)
                return image

        class Loader:
            def __init__(self, device):
                assert device == torch.device("cpu")

            def load_from_state_dict(self, state, models=models):
                value = Model(state)
                models.append(value)
                return value

        def blend(a, b, x, y, blends=blends):
            blends.append((x, y))
            return a * x + b * y

        module = functions(
            root / INTERP,
            torch=torch,
            ImageModelDescriptor=Model,
            MaskedImageModelDescriptor=type("Masked", (), {}),
            ModelLoader=Loader,
            interpolate_torch=blend,
            native_mean=np.mean,
            np2tensor=lambda a, **k: torch.from_numpy(a),
            tensor2np=lambda a, **k: a.numpy(),
            gc=SimpleNamespace(collect=lambda: None),
            get_settings=lambda c: SimpleNamespace(
                device=torch.device("cpu"), budget_limit=budget
            ),
        )
        a = Model({"w": torch.tensor([0.25, 0.75, 1.25], dtype=dtype)})
        b = Model({"w": torch.tensor([0.5, 1.0, 1.5], dtype=dtype)})
        result, x, y = module.interpolate_models_node(None, a, b, amount)
        outcomes.append((result.state["w"].clone(), x, y, result.forward_calls))
        counts.append((len(blends), len(models)))
        torch.testing.assert_close(
            a.state["w"], torch.tensor([0.25, 0.75, 1.25], dtype=dtype), rtol=0, atol=0
        )
        torch.testing.assert_close(
            b.state["w"], torch.tensor([0.5, 1.0, 1.5], dtype=dtype), rtol=0, atol=0
        )
    torch.testing.assert_close(outcomes[0][0], outcomes[1][0], rtol=0, atol=0)
    assert outcomes[0][1:] == outcomes[1][1:]
    assert counts[1] == ((1, 2) if amount == 50 and budget == 0 else counts[0])
    if amount == 50:
        assert counts[0] == (2, 2)


@pytest.mark.parametrize("keys_match", [False, True])
def test_rejection_contract(keys_match):
    errors = []
    for root in (REF, SOURCE):

        class Model:
            def __init__(self, state):
                self.model = self
                self.state = state
                self.device = torch.device("cpu")
                self.input_channels = 3
                self.size_requirements = SimpleNamespace(minimum=3, multiple_of=1)

            def state_dict(self):
                return self.state

            def __call__(self, value):
                return torch.zeros_like(value)

        module = functions(
            root / INTERP,
            torch=torch,
            ImageModelDescriptor=Model,
            MaskedImageModelDescriptor=type("Masked", (), {}),
            ModelLoader=lambda d: SimpleNamespace(load_from_state_dict=Model),
            interpolate_torch=lambda a, b, x, y: a * x + b * y,
            native_mean=np.mean,
            np2tensor=lambda a, **k: torch.from_numpy(a),
            tensor2np=lambda a, **k: a.numpy(),
            gc=SimpleNamespace(collect=lambda: None),
            get_settings=lambda c: SimpleNamespace(
                device=torch.device("cpu"), budget_limit=0
            ),
        )
        a = Model({"w": torch.ones(2)})
        b = Model({("w" if keys_match else "other"): torch.ones(2)})
        try:
            module.interpolate_models_node(None, a, b, 50)
        except Exception as error:
            errors.append((type(error), error.args))
    assert errors[0] == errors[1]


def test_cached_constructor_stops_repeated_global_rng_draws_explicitly():
    from test_framework_optimizations import Context

    from nodes.impl.pytorch.resource_cache import lease_resource

    # Setup reuse intentionally removes the old constructor's repeated random
    # draws. Fully loaded models ignore these random initial values; incomplete
    # checkpoints must opt out of retention instead of changing their output.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(713)
        draws = []

        def create():
            draws.append(torch.rand(1))
            return object(), True

        context = Context()
        states = []
        for _ in range(3):
            with lease_resource(context, lambda: 1, create, reuse=True):
                pass
            states.append(torch.get_rng_state().clone())
        assert len(draws) == 1
        assert all(torch.equal(states[0], state) for state in states)
        context.finish()
        for _ in range(3):
            with lease_resource(
                context, lambda: 1, lambda: (create()[0], False), reuse=True
            ):
                pass
        assert len(draws) == 4
        context.finish()
