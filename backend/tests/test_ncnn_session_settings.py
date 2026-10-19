"""An NCNN net is kept per model and per the settings it was built from: a changed
GPU, Winograd, SGEMM, thread count or block time builds the net again, other
settings keep it. Vulkan stays hidden (conftest), so the real nets are CPU nets."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from test_ncnn_errors import SETTINGS, tiny_model

from nodes.impl.ncnn import session
from packages.chaiNNer_ncnn.settings import NcnnSettings

CPU_CHANGES = {
    "winograd": (True, "use_winograd_convolution"),
    "sgemm": (True, "use_sgemm_convolution"),
    "threads": (2, "num_threads"),
    "blocktime": (7, "openmp_blocktime"),
}


@pytest.mark.parametrize("name", CPU_CHANGES)
def test_a_changed_cpu_setting_builds_the_net_from_it(tmp_path: Path, name: str):
    value, option = CPU_CHANGES[name]
    model = tiny_model(tmp_path)
    first = session.get_ncnn_net(model, SETTINGS)
    assert session.get_ncnn_net(model, SETTINGS) is first
    changed = dataclasses.replace(SETTINGS, **{name: value})
    second = session.get_ncnn_net(model, changed)
    assert second is not first
    assert getattr(second.opt, option) == value != getattr(first.opt, option)
    assert session.get_ncnn_net(model, changed) is second


@pytest.mark.parametrize("name", ["gpu_index", "budget_limit"])
def test_settings_a_cpu_net_does_not_use_keep_it(tmp_path: Path, name: str):
    model = tiny_model(tmp_path)
    first = session.get_ncnn_net(model, SETTINGS)
    changed = dataclasses.replace(SETTINGS, **{name: 1})
    assert session.get_ncnn_net(model, changed) is first


class Net:
    """Stands in for a Vulkan net, which the tests cannot build."""

    def __init__(self, settings: NcnnSettings) -> None:
        self.settings = settings


@pytest.mark.parametrize(
    ("name", "value", "rebuilt"),
    [
        ("gpu_index", 1, True),
        ("winograd", True, False),
        ("sgemm", True, False),
        ("threads", 2, False),
        ("blocktime", 7, False),
        ("budget_limit", 1, False),
    ],
)
def test_on_the_gpu_only_the_gpu_setting_builds_the_net_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: object,
    rebuilt: bool,
):
    monkeypatch.setattr(session, "use_gpu", True)
    monkeypatch.setattr(
        session, "create_ncnn_net", lambda _model, settings: Net(settings)
    )
    model = tiny_model(tmp_path)
    first = session.get_ncnn_net(model, SETTINGS)
    second = session.get_ncnn_net(model, dataclasses.replace(SETTINGS, **{name: value}))
    assert (second is not first) == rebuilt
