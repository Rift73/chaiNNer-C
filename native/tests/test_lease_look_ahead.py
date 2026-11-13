"""A node that leases a model (resource_cache.lease_resource) before a node that looks
ahead: the engines compute it ahead with a copy of its context, and it keeps one model
for the run, which is gone after the run."""

from __future__ import annotations

import gc
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
from engine_differential import execute
from test_engine_scenarios import Builder, items

import api
from api import NodeContext
from nodes.impl.pytorch.resource_cache import lease_resource

ITEMS = 12


class Model:
    """Stands for a leased model bundle."""


@pytest.fixture(autouse=True)
def node_package(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        api.registry, "get_package", lambda _: SimpleNamespace(id="test")
    )


@pytest.mark.parametrize("kind", ["python", "native"])
@pytest.mark.parametrize("look_ahead", [False, True], ids=["in-order", "look-ahead"])
def test_a_model_leased_before_a_look_ahead_is_built_once(
    kind: str, look_ahead: bool, tmp_path: Path
):
    made: list[weakref.ref[Model]] = []
    alive_at_the_end: list[int] = []
    b = Builder()
    b.node("frames", lambda: items(list(range(ITEMS))), kind="generator", inputs=0)

    def restore(context: NodeContext, x: int) -> int:
        def create() -> tuple[Model, bool]:
            model = Model()
            made.append(weakref.ref(model))
            return model, True

        with lease_resource(context, lambda: "same key", create, reuse=True):
            return x * 100

    def temporal(context: NodeContext, frame: int) -> int:
        view = context.iteration
        assert view is not None
        if look_ahead:
            for offset in range(1, view.lookahead(0) + 1):
                view.peek(0, offset)
        if view.index == ITEMS - 1:
            gc.collect()
            alive_at_the_end.append(sum(ref() is not None for ref in made))
        return frame

    b.node("restore", restore, context=True)
    b.node("temporal", temporal, context=True, side_effects=True)
    b.edge("frames", 0, "restore", 0)
    b.edge("restore", 0, "temporal", 0)
    result = execute(kind, b.chain, storage=tmp_path / "storage")
    assert result.error is None, result.error
    gc.collect()
    assert (len(made), alive_at_the_end) == (1, [1])
    assert all(ref() is None for ref in made)
