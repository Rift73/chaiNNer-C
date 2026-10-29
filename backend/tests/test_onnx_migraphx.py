"""ONNX Runtime's MIGraphX provider (AMD GPUs): the provider options a session is
created with, the settings a session is kept for, and the ONNX settings a MIGraphX
build of ONNX Runtime offers. Runs without an AMD GPU: sessions are stand-ins that
report the providers they were created with, and the settings module runs against
stand-in provider lists, GPUs and installed distributions."""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper
from test_onnx_session_providers import (
    CPU,
    CUDA,
    DML,
    TENSORRT,
    identity_model,
    ort_creates,
    ort_registers,
)

import gpu
import nodes.impl.onnx.session as onnx_session
from api import CacheSetting, DropdownSetting, SettingsParser, ToggleSetting
from api.settings import DropdownOption
from nodes.impl.onnx.model import OnnxGeneric, OnnxInfo
from nodes.impl.onnx.session import get_onnx_session
from packages.chaiNNer_onnx import package as onnx_package

MIGRAPHX = "MIGraphXExecutionProvider"
SETTINGS_FILE = (
    Path(__file__).parents[1] / "src" / "packages" / "chaiNNer_onnx" / "settings.py"
)


def conv_model(seed: int) -> OnnxGeneric:
    """A 3x3 convolution with dynamic height and width, its weights from `seed`."""
    rng = np.random.default_rng(seed)
    weight = numpy_helper.from_array(
        rng.standard_normal((3, 3, 3, 3)).astype(np.float32), "weight"
    )
    image = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "h", "w"])
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3, "h", "w"]
    )
    node = helper.make_node("Conv", ["input", "weight"], ["output"], pads=[1, 1, 1, 1])
    graph = helper.make_graph([node], "conv", [image], [output], [weight])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    )
    return OnnxGeneric(model.SerializeToString(), OnnxInfo(13, "fp32"))


@pytest.fixture(autouse=True)
def migraphx_loads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """MIGraphX sessions' library loads on Windows (_load_migraphx), recorded instead
    of made: the test environment's ONNX Runtime has no MIGraphX."""
    loads: list[str] = []
    monkeypatch.setattr(onnx_session, "_load_migraphx", lambda: loads.append("load"))
    return loads


def migraphx_options(providers: Sequence[object]) -> dict[str, object]:
    first = providers[0]
    assert isinstance(first, tuple) and first[0] == MIGRAPHX
    return first[1]


def test_a_migraphx_session_gets_its_options_and_falls_back_to_the_cpu(
    monkeypatch: pytest.MonkeyPatch,
):
    created = ort_creates(monkeypatch)
    get_onnx_session(
        identity_model(),
        1,
        MIGRAPHX,
        False,
        None,
        should_migraphx_fp16=True,
        migraphx_exhaustive_tune=True,
    )
    # Without a cache folder no program file is written: no cache option at all.
    assert created == [
        [
            (
                MIGRAPHX,
                {
                    "device_id": 1,
                    "migraphx_fp16_enable": True,
                    "migraphx_exhaustive_tune": True,
                },
            ),
            CPU,
        ]
    ]


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_on_windows_a_migraphx_session_loads_migraphx_first(
    monkeypatch: pytest.MonkeyPatch, migraphx_loads: list[str], platform: str
):
    # ORT's provider and MIGraphX find HIP and MIGraphX's GPU target only once they
    # are loaded; Linux's run paths need nothing. Other providers load nothing.
    monkeypatch.setattr(onnx_session.sys, "platform", platform)
    created = ort_creates(monkeypatch)
    for provider in (DML, MIGRAPHX):
        get_onnx_session(identity_model(), 0, provider, False)
    assert [providers[0] for providers in created] == [
        (DML, {"device_id": 0}),
        (MIGRAPHX, migraphx_options(created[1])),
    ]
    assert migraphx_loads == (["load"] if platform == "win32" else [])


def test_the_options_reach_onnx_runtime_in_its_text_form(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # ORT's binding passes str() of each value; MIGraphX parses "True"/"False" and an
    # integer device id, and takes the cache option as a path.
    created = ort_creates(monkeypatch)
    get_onnx_session(conv_model(1), 0, MIGRAPHX, False, None, False, str(tmp_path))
    options = migraphx_options(created[0])
    assert {
        k: str(v) for k, v in options.items() if k != "migraphx_model_cache_dir"
    } == {
        "device_id": "0",
        "migraphx_fp16_enable": "False",
        "migraphx_exhaustive_tune": "False",
    }
    assert Path(str(options["migraphx_model_cache_dir"])).parent == tmp_path


def test_programs_are_cached_per_model_precision_and_tuning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # MIGraphX names a program by tensor names and shapes: two models of one
    # architecture with other weights, or another precision or tuning, need their own
    # folders.
    created = ort_creates(monkeypatch)
    cache = str(tmp_path)

    def folder(model: OnnxGeneric, fp16: bool, tune: bool) -> str:
        get_onnx_session(model, 0, MIGRAPHX, False, None, fp16, cache, tune)
        return str(migraphx_options(created[-1])["migraphx_model_cache_dir"])

    first = folder(conv_model(1), False, False)
    assert os.path.isdir(first) and Path(first).parent == tmp_path
    # The same bytes in a model loaded again
    assert folder(conv_model(1), False, False) == first
    others = [
        folder(conv_model(2), False, False),
        folder(conv_model(1), True, False),
        folder(conv_model(1), False, True),
        folder(conv_model(1), True, True),
    ]
    assert len({first, *others}) == 5
    assert all(os.path.isdir(f) for f in others)


# (GPU index, MIGraphX FP16, MIGraphX cache path, MIGraphX exhaustive tuning)
MigraphxSettings = tuple[int, bool, str | None, bool]


@pytest.mark.parametrize(
    "after",
    [
        (1, False, None, False),
        (0, True, None, False),
        (0, False, "cache", False),
        (0, False, None, True),
    ],
    ids=["gpu", "fp16", "cache", "exhaustive-tune"],
)
def test_a_changed_migraphx_setting_gets_a_session_created_from_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, after: MigraphxSettings
):
    created = ort_creates(monkeypatch)
    model = conv_model(1)

    def session(gpu_index: int, fp16: bool, cache: str | None, tune: bool) -> object:
        path = None if cache is None else str(tmp_path / cache)
        return get_onnx_session(
            model, gpu_index, MIGRAPHX, False, None, fp16, path, tune
        )

    first = session(0, False, None, False)
    assert session(0, False, None, False) is first
    second = session(*after)
    assert second is not first
    assert len(created) == 2
    assert session(*after) is second
    assert len(created) == 2


@pytest.mark.parametrize(
    "tensorrt", [(True, None), (False, "cache")], ids=["fp16", "cache"]
)
def test_tensorrt_settings_keep_a_migraphx_session(
    monkeypatch: pytest.MonkeyPatch, tensorrt: tuple[bool, str | None]
):
    created = ort_creates(monkeypatch)
    model = conv_model(1)
    first = get_onnx_session(model, 0, MIGRAPHX, False, None)
    assert get_onnx_session(model, 0, MIGRAPHX, *tensorrt) is first
    assert len(created) == 1


@pytest.mark.parametrize("provider", [TENSORRT, CUDA, DML, CPU])
def test_migraphx_settings_keep_other_sessions_and_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, provider: str
):
    created = ort_creates(monkeypatch)
    model = conv_model(1)
    first = get_onnx_session(model, 0, provider, False, None)
    assert (
        get_onnx_session(model, 0, provider, False, None, True, str(tmp_path), True)
        is first
    )
    assert len(created) == 1
    assert MIGRAPHX not in str(created[0])
    assert os.listdir(tmp_path) == []


def test_migraphx_that_cannot_start_is_reported_not_run_on_the_cpu(
    monkeypatch: pytest.MonkeyPatch,
):
    ort_registers(monkeypatch, [CPU])
    model = identity_model()
    for _ in range(2):  # not cached: the next run tries again
        with pytest.raises(RuntimeError) as raised:
            get_onnx_session(model, 0, MIGRAPHX, False)
        message = str(raised.value)
        assert f"({MIGRAPHX})" in message
        assert f"on {CPU} instead" in message
        assert "Choose another execution provider" in message


def amd_device(
    index: int, compute_units: int = 48, luid: int | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        index=index,
        name=f"AMD Radeon {index}",
        gfx="gfx1100",
        supports_fp16=True,
        compute_units=compute_units,
        luid=luid,
    )


def nvidia_device(index: int) -> SimpleNamespace:
    return SimpleNamespace(index=index, name=f"NVIDIA GeForce {index}")


def load_settings(
    monkeypatch: pytest.MonkeyPatch,
    providers: Sequence[str],
    *,
    amd: Sequence[SimpleNamespace] = (),
    nvidia: Sequence[SimpleNamespace] = (),
    migraphx_version: str | None = None,
) -> ModuleType:
    """The ONNX settings module as it loads with ONNX Runtime offering `providers`,
    these GPUs, and onnxruntime-migraphx `migraphx_version` installed (None: not
    installed). Its settings go to a fresh list (`.package.settings`)."""
    monkeypatch.setattr(ort, "get_available_providers", lambda: list(providers))
    monkeypatch.setattr(
        gpu,
        "amd",
        SimpleNamespace(
            devices=list(amd),
            is_available=bool(amd),
            all_support_fp16=all(d.supports_fp16 for d in amd),
            # As gpu.AmdInfo: some GPU with 8 compute units or more; the strongest
            eligible=any(d.compute_units >= 8 for d in amd),
            default_index=max(
                amd, key=lambda d: (d.compute_units, -d.index), default=amd_device(0)
            ).index,
        ),
    )
    monkeypatch.setattr(
        gpu,
        "nvidia",
        SimpleNamespace(
            devices=list(nvidia), is_available=bool(nvidia), all_support_fp16=True
        ),
    )
    real_version = metadata.version

    def version(name: str) -> str:
        if name != "onnxruntime-migraphx":
            return real_version(name)
        if migraphx_version is None:
            raise metadata.PackageNotFoundError(name)
        return migraphx_version

    monkeypatch.setattr(metadata, "version", version)
    monkeypatch.setattr(onnx_package, "settings", [])
    # A module of its own beside the loaded one (dataclasses look their module up)
    spec = importlib.util.spec_from_file_location(
        "packages.chaiNNer_onnx.settings_under_test", SETTINGS_FILE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def registered(module: ModuleType) -> dict[str, object]:
    return {s.key: s for s in module.package.settings}


def gpu_options(module: ModuleType) -> list[DropdownOption]:
    setting = registered(module)["gpu_index"]
    assert isinstance(setting, DropdownSetting)
    return setting.options


def test_a_cuda_build_offers_what_it_did(monkeypatch: pytest.MonkeyPatch):
    # An AMD GPU beside the NVIDIA ones changes nothing for a CUDA build.
    module = load_settings(
        monkeypatch,
        [TENSORRT, CUDA, CPU],
        nvidia=[nvidia_device(0), nvidia_device(1)],
        amd=[amd_device(0)],
    )
    settings = registered(module)
    assert list(settings) == [
        "gpu_index",
        "execution_provider",
        "onnx_tensorrt_cache",
        "tensorrt_fp16_mode",
    ]
    assert gpu_options(module) == [
        {"label": "NVIDIA GeForce 0", "value": "0"},
        {"label": "NVIDIA GeForce 1", "value": "1"},
    ]
    assert module.default_provider == CUDA
    tensorrt_fp16 = settings["tensorrt_fp16_mode"]
    assert isinstance(tensorrt_fp16, ToggleSetting)
    assert tensorrt_fp16.default is True and not tensorrt_fp16.disabled


@pytest.mark.parametrize(
    ("providers", "default"),
    [([DML, CPU], CPU), ([CPU], CPU)],
    ids=["directml", "cpu"],
)
def test_builds_without_migraphx_offer_what_they_did(
    monkeypatch: pytest.MonkeyPatch, providers: list[str], default: str
):
    module = load_settings(monkeypatch, providers, amd=[amd_device(0)])
    assert list(registered(module)) == [
        "gpu_index",
        "execution_provider",
        "onnx_tensorrt_cache",
        "tensorrt_fp16_mode",
    ]
    assert gpu_options(module) == []
    assert module.default_provider == default


def test_a_migraphx_build_on_an_amd_gpu_defaults_to_migraphx(
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_settings(
        monkeypatch,
        [MIGRAPHX, CPU],
        amd=[amd_device(0), amd_device(1)],
        migraphx_version="1.30.0+rocm7.14",
    )
    settings = registered(module)
    assert list(settings) == [
        "gpu_index",
        "execution_provider",
        "onnx_tensorrt_cache",
        "tensorrt_fp16_mode",
        "onnx_migraphx_cache",
        "migraphx_fp16_mode",
        "migraphx_exhaustive_tune",
    ]
    # HIP's device indices, which MIGraphX takes
    assert gpu_options(module) == [
        {"label": "AMD Radeon 0", "value": "0"},
        {"label": "AMD Radeon 1", "value": "1"},
    ]
    assert module.default_provider == MIGRAPHX
    providers = settings["execution_provider"]
    assert isinstance(providers, DropdownSetting)
    assert list(providers.options) == [
        {"label": "MIGraphX", "value": MIGRAPHX},
        {"label": "CPU", "value": CPU},
    ]
    cache = settings["onnx_migraphx_cache"]
    assert isinstance(cache, CacheSetting) and not cache.disabled
    fp16 = settings["migraphx_fp16_mode"]
    assert isinstance(fp16, ToggleSetting) and fp16.default is True
    tune = settings["migraphx_exhaustive_tune"]
    assert isinstance(tune, ToggleSetting) and tune.default is False
    for key in ("onnx_tensorrt_cache", "tensorrt_fp16_mode"):
        tensorrt = settings[key]
        assert isinstance(tensorrt, (CacheSetting, ToggleSetting)) and tensorrt.disabled


@pytest.mark.parametrize(
    "amd", [[], [amd_device(0, compute_units=2)]], ids=["none", "ryzen-graphics"]
)
def test_without_an_amd_gpu_worth_it_a_migraphx_build_offers_no_migraphx(
    monkeypatch: pytest.MonkeyPatch, amd: list[SimpleNamespace]
):
    # No AMD GPU, or only a Ryzen's small graphics: the CPU, as before.
    module = load_settings(
        monkeypatch, [MIGRAPHX, CPU], amd=amd, migraphx_version="1.30.0+rocm7.14"
    )
    assert list(registered(module)) == [
        "gpu_index",
        "execution_provider",
        "onnx_tensorrt_cache",
        "tensorrt_fp16_mode",
    ]
    assert gpu_options(module) == []
    assert module.default_provider == CPU
    assert module.can_cache_migraphx is False


def test_with_an_nvidia_gpu_a_migraphx_build_offers_what_it_did(
    monkeypatch: pytest.MonkeyPatch,
):
    # An NVIDIA GPU keeps its choices and defaults; MIGraphX's apply without one.
    module = load_settings(
        monkeypatch,
        [MIGRAPHX, CPU],
        nvidia=[nvidia_device(0)],
        amd=[amd_device(0)],
        migraphx_version="1.30.0+rocm7.14",
    )
    assert list(registered(module)) == [
        "gpu_index",
        "execution_provider",
        "onnx_tensorrt_cache",
        "tensorrt_fp16_mode",
    ]
    assert gpu_options(module) == [{"label": "NVIDIA GeForce 0", "value": "0"}]
    assert module.default_provider == CPU
    assert module.can_cache_migraphx is False
    settings = node_settings(module, STORED)
    assert settings.migraphx_cache_path is None


def test_the_gpu_setting_defaults_to_the_strongest_amd_gpu(
    monkeypatch: pytest.MonkeyPatch,
):
    # A Ryzen's graphics often come first in HIP's order.
    module = load_settings(
        monkeypatch,
        [MIGRAPHX, CPU],
        amd=[amd_device(0, compute_units=2), amd_device(1)],
        migraphx_version="1.30.0+rocm7.14",
    )
    setting = registered(module)["gpu_index"]
    assert isinstance(setting, DropdownSetting)
    assert setting.default == "1"
    assert len(gpu_options(module)) == 2


@pytest.mark.parametrize(
    ("version", "cache"),
    [
        ("1.30.0+rocm7.14", True),
        ("1.31.0+rocm7.15", True),
        ("1.27.1", False),
        ("1.30.0", False),
        (None, False),
    ],
)
def test_the_program_cache_needs_a_build_that_names_programs_by_shape(
    monkeypatch: pytest.MonkeyPatch, version: str | None, cache: bool
):
    module = load_settings(
        monkeypatch, [MIGRAPHX, CPU], amd=[amd_device(0)], migraphx_version=version
    )
    setting = registered(module)["onnx_migraphx_cache"]
    assert isinstance(setting, CacheSetting)
    assert setting.disabled is not cache
    assert module.can_cache_migraphx is cache


def node_settings(module: ModuleType, values: dict[str, object]) -> Any:
    """The node settings `module` makes of the stored settings `values`."""
    return module.get_settings(SimpleNamespace(settings=SettingsParser(values)))


STORED: dict[str, object] = {
    "execution_provider": MIGRAPHX,
    "gpu_index": "1",
    "onnx_migraphx_cache": "cache",
    "migraphx_fp16_mode": True,
    "migraphx_exhaustive_tune": True,
}


@pytest.mark.parametrize(
    ("version", "cache"), [("1.30.0+rocm7.14", "cache"), ("1.27.1", None)]
)
def test_the_node_settings_carry_the_migraphx_ones(
    monkeypatch: pytest.MonkeyPatch, version: str, cache: str | None
):
    # A cache location kept from before is not used where the cache gives other
    # shapes' programs.
    module = load_settings(
        monkeypatch, [MIGRAPHX, CPU], amd=[amd_device(0)], migraphx_version=version
    )
    settings = node_settings(module, STORED)
    assert settings == module.OnnxSettings(
        gpu_index=1,
        execution_provider=MIGRAPHX,
        tensorrt_cache_path=None,
        tensorrt_fp16_mode=False,
        migraphx_cache_path=cache,
        migraphx_fp16_mode=True,
        migraphx_exhaustive_tune=True,
    )


def test_a_cuda_build_passes_no_migraphx_cache(monkeypatch: pytest.MonkeyPatch):
    module = load_settings(monkeypatch, [CUDA, CPU], nvidia=[nvidia_device(0)])
    settings = node_settings(module, {**STORED, "execution_provider": CUDA})
    assert settings.migraphx_cache_path is None


# ONNX Runtime's Windows MIGraphX build lists MIGraphX before DirectML.
WINDOWS_BUILD = [MIGRAPHX, DML, CPU]


def test_the_windows_migraphx_build_defaults_to_directml(
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_settings(
        monkeypatch,
        WINDOWS_BUILD,
        amd=[amd_device(0, compute_units=2), amd_device(1)],
        migraphx_version="1.30.0+rocm7.14.dml",
    )
    settings = registered(module)
    assert list(settings) == [
        "gpu_index",
        "execution_provider",
        "onnx_tensorrt_cache",
        "tensorrt_fp16_mode",
        "onnx_migraphx_cache",
        "migraphx_fp16_mode",
        "migraphx_exhaustive_tune",
    ]
    assert module.default_provider == DML
    providers = settings["execution_provider"]
    assert isinstance(providers, DropdownSetting)
    assert [o["value"] for o in providers.options] == WINDOWS_BUILD
    assert providers.default == DML
    # The AMD GPUs by HIP's index, the strongest the default, as on Linux
    gpu_setting = settings["gpu_index"]
    assert isinstance(gpu_setting, DropdownSetting) and gpu_setting.default == "1"
    assert len(gpu_options(module)) == 2
    assert module.can_cache_migraphx is True


def test_directml_beside_migraphx_runs_on_the_gpus_dxgi_adapter(
    monkeypatch: pytest.MonkeyPatch,
):
    # HIP lists the Ryzen's graphics first; DXGI the Radeon driving the display.
    adapters = {0xA: 1, 0xB: 0}
    module = load_settings(
        monkeypatch,
        WINDOWS_BUILD,
        amd=[amd_device(0, compute_units=2, luid=0xA), amd_device(1, luid=0xB)],
        migraphx_version="1.30.0+rocm7.14.dml",
    )
    monkeypatch.setattr(module, "dxgi_adapter_index", adapters.__getitem__)
    directml = {**STORED, "execution_provider": DML}
    assert node_settings(module, directml).gpu_index == 0
    assert node_settings(module, {**directml, "gpu_index": "0"}).gpu_index == 1
    # MIGraphX takes HIP's index itself.
    assert node_settings(module, STORED).gpu_index == 1
    with pytest.raises(RuntimeError, match="names AMD GPU 5, which DirectML cannot"):
        node_settings(module, {**directml, "gpu_index": "5"})


def test_directml_without_migraphx_keeps_the_index_as_it_was(
    monkeypatch: pytest.MonkeyPatch,
):
    # onnxruntime-directml: the GPU setting is DirectML's own index, as before.
    module = load_settings(
        monkeypatch, [DML, CPU], amd=[amd_device(0, luid=0xA), amd_device(1, luid=0xB)]
    )
    monkeypatch.setattr(module, "dxgi_adapter_index", pytest.fail)
    stored: dict[str, object] = {"execution_provider": DML, "gpu_index": "1"}
    assert node_settings(module, stored).gpu_index == 1


@pytest.mark.skipif(sys.platform != "win32", reason="DXGI is Windows'")
def test_dxgi_has_no_adapter_of_an_unknown_luid():
    # The Microsoft Basic Render Driver is always listed.
    with pytest.raises(
        OSError, match=r"no adapter with the LUID 0x7FFFFFFF00000001 \(\d+ adapters\)"
    ) as raised:
        gpu.dxgi_adapter_index(0x7FFF_FFFF_0000_0001)
    assert "(0 adapters)" not in str(raised.value)
