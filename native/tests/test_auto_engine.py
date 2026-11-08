"""PyTorch Upscale Image's Use TensorRT Engine (nodes/impl/pytorch/auto_engine.py).

Fakes for TensorRT and the build: no GPU is needed. Off, the node runs the path it always
ran; on, a model gets one engine, kept on disk by its weights, or runs on PyTorch with a
reason.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import onnx
import pytest
import torch
from onnx import helper

from api import Aborted, ExecutionOptions, NodeContext, NodeId, SettingsParser
from nodes.impl.onnx import load as onnx_load
from nodes.impl.pytorch import auto_engine as auto
from nodes.impl.tensorrt import build_cache
from nodes.impl.tensorrt import cache as tensorrt_cache
from nodes.impl.tensorrt.model import TensorRTEngine, TensorRTEngineInfo
from nodes.impl.upscale.auto_split_tiles import (
    ESTIMATE,
    MAX_TILE_SIZE,
    NO_TILING,
    TileSize,
)
from nodes.impl.upscale.tile_report import TileReport

# The fixture replaces it; a test of the node's own route puts it back.
REAL_ENGINE_OP = auto._engine_op
NODE = NodeId("upscale")
SCHEMA = "chainner:pytorch:upscale_image"
GPU = torch.device("cuda:0")


def engine_of(scale=4):
    info = TensorRTEngineInfo(
        precision="fp16",
        input_channels=3,
        output_channels=3,
        scale=scale,
        gpu_architecture="sm_90",
        tensorrt_version="11",
        has_dynamic_shapes=True,
        min_shape=(1, 3, 64, 64),
        opt_shape=(1, 3, 512, 512),
        max_shape=(1, 3, 512, 512),
    )
    return TensorRTEngine(b"engine bytes", info)


class Context(NodeContext):
    def __init__(self, root: Path, settings: dict[str, object] | None = None):
        super().__init__()
        self.storage = root / "backend-storage"
        self.parser = SettingsParser(settings or {})
        self.warnings: list[str] = []
        self.chain_cleanups: list = []
        self.stopped = False

    @property
    def aborted(self):
        return self.stopped

    @property
    def paused(self):
        return False

    def set_progress(self, progress):
        self.check_aborted()

    @property
    def settings(self):
        return self.parser

    @property
    def node_id(self):
        return NODE

    @property
    def storage_dir(self):
        return self.storage

    def add_cleanup(self, fn, after="chain"):
        assert after == "chain"
        self.chain_cleanups.append(fn)

    def warn(self, message):
        self.warnings.append(message)

    def end_chain(self):
        for fn in self.chain_cleanups:
            fn()
        self.chain_cleanups.clear()


class Model:
    def __init__(self, arch="ESRGAN", weight=1.0, scale=4, tags=()):
        self.architecture = SimpleNamespace(name=arch, id=arch)
        self.tags = list(tags)
        self.scale = scale
        self.size_requirements = SimpleNamespace(minimum=0, multiple_of=1)
        self.model = torch.nn.Conv2d(1, 1, 1)
        torch.nn.init.constant_(self.model.weight, weight)
        torch.nn.init.constant_(self.model.bias, 0.0)


class FakeOutOfMemory(RuntimeError):
    pass


class Fakes:
    def __init__(self):
        self.builds = 0
        self.tiles: list[int] = []
        self.build_error: Exception | None = None
        self.engine_error: Exception | None = None
        self.engine_output: np.ndarray | None = None
        self.engine_runs = 0
        self.released: list[NodeId] = []


@pytest.fixture
def fakes(monkeypatch):
    """Everything TensorRT behind a fake GPU, with the real memory and disk caches."""
    fake = Fakes()

    def build(_context, _model, _tile, _whole=False):
        fake.tiles.append(_tile)
        fake.builds += 1
        if fake.build_error is not None:
            raise fake.build_error
        return engine_of()

    def engine_op(_context, _model, _engine, _report=None):
        def run(img):
            fake.engine_runs += 1
            if fake.engine_error is not None:
                raise fake.engine_error
            return img * 2 if fake.engine_output is None else fake.engine_output

        return run

    monkeypatch.setattr(auto, "_build", build)
    monkeypatch.setattr(auto, "_engine_op", engine_op)
    monkeypatch.setattr(auto, "_no_engine_here", lambda _device: None)
    monkeypatch.setattr(tensorrt_cache, "release_node", fake.released.append)
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace(__version__="11.4"))
    monkeypatch.setitem(
        sys.modules,
        "nodes.impl.tensorrt.memory",
        SimpleNamespace(
            get_cuda_device_name=lambda _i: "GPU",
            get_cuda_compute_capability=lambda _i: (10, 0),
            CudaOutOfMemory=FakeOutOfMemory,
        ),
    )
    auto._outcomes.clear()
    auto._engine_nodes.clear()
    auto._unproven.clear()
    auto._rebuilt.clear()
    build_cache._memory.clear()
    yield fake
    auto._outcomes.clear()
    auto._engine_nodes.clear()
    auto._unproven.clear()
    auto._rebuilt.clear()
    build_cache._memory.clear()


IMAGE = np.full((4, 4, 3), 0.25, dtype=np.float32)


def kept_files(root: Path) -> list[Path]:
    folder = root / "cache" / "tensorrt_engine_cache"
    return list(folder.glob(f"*{build_cache.SUFFIX}"))


def pytorch(img):
    return img * 3


def run(context, model, device=GPU, gpu=0, image=IMAGE, tile=ESTIMATE):
    return auto.upscale_through_engine(
        context, model, device, gpu, tile, lambda op: op(image), pytorch
    )


def test_a_model_uses_its_engine_and_builds_once(fakes, tmp_path):
    context, model = Context(tmp_path), Model()
    for _ in range(3):
        np.testing.assert_array_equal(run(context, model), IMAGE * 2)
    assert fakes.builds == 1
    assert fakes.engine_runs == 3
    assert context.warnings == []


def test_a_later_process_reads_the_engine_from_disk_by_the_weights(fakes, tmp_path):
    context = Context(tmp_path)
    run(context, Model(weight=1.0))
    assert len(kept_files(tmp_path)) == 1

    # A restart: nothing in memory, the model loaded anew from the same file.
    auto._outcomes.clear()
    build_cache._memory.clear()
    run(context, Model(weight=1.0))
    assert fakes.builds == 1

    run(context, Model(weight=2.0))
    assert fakes.builds == 2
    assert len(kept_files(tmp_path)) == 2


def test_with_the_setting_on_the_engine_stays_in_the_folder_it_names(fakes, tmp_path):
    chosen = tmp_path / "chosen"
    options = ExecutionOptions(
        {
            "chaiNNer_pytorch": {"use_fp16": True},
            "chaiNNer_tensorrt": {"engine_cache": str(chosen)},
        }
    )
    context = Context(tmp_path)
    context.parser = options.get_package_settings("chaiNNer_pytorch")
    run(context, Model())
    assert len(list(chosen.glob(f"*{build_cache.SUFFIX}"))) == 1
    assert kept_files(tmp_path) == []


def test_with_the_setting_off_the_engine_stays_where_the_setting_would_keep_it(
    fakes, tmp_path
):
    options = ExecutionOptions({"chaiNNer_tensorrt": {"engine_cache": ""}})
    context = Context(tmp_path)
    context.parser = options.get_package_settings("chaiNNer_pytorch")
    run(context, Model())
    assert len(kept_files(tmp_path)) == 1
    # Nothing in the backend's storage folder, whose files have other owners.
    assert not context.storage.exists()


def test_build_engine_and_this_feature_share_one_limit(fakes, tmp_path, monkeypatch):
    context = Context(tmp_path)
    folder = tmp_path / "cache" / "tensorrt_engine_cache"
    folder.mkdir(parents=True)
    record = len(build_cache._encode(engine_of()))
    monkeypatch.setattr(build_cache, "DISK_BYTES", record)

    # A Build Engine file goes for this feature's engine, the least recently used first.
    build_cache.cached_engine("b" * 64, engine_of, str(folder))
    built = folder / f"{'b' * 64}{build_cache.SUFFIX}"
    os.utime(built, (1, 1))
    run(context, Model())
    assert not built.exists()
    kept = kept_files(tmp_path)
    assert len(kept) == 1

    # And this feature's goes for a Build Engine file.
    os.utime(kept[0], (1, 1))
    build_cache.cached_engine("c" * 64, engine_of, str(folder))
    assert kept_files(tmp_path) == [folder / f"{'c' * 64}{build_cache.SUFFIX}"]


def test_another_packages_settings_read_through_the_run(tmp_path):
    options = ExecutionOptions({"a": {"n": 1}, "b": {"folder": str(tmp_path)}})
    parser = options.get_package_settings("a")
    assert parser.of_package("b").get_cache_location("folder") == str(tmp_path)
    assert parser.of_package("b").of_package("a").get_int("n", 0) == 1
    assert parser.of_package("missing").get_cache_location("folder") is None
    # A parser outside a run has no other package's settings to read.
    assert SettingsParser({}).of_package("b").get_cache_location("folder") is None


def test_the_key_follows_everything_the_engine_depends_on(fakes, monkeypatch):
    base = auto._engine_key(Model(), "fp16", 0, 512)
    assert base == auto._engine_key(Model(), "fp16", 0, 512)
    assert base != auto._engine_key(Model(weight=2.0), "fp16", 0, 512)
    assert base != auto._engine_key(Model(arch="SPAN"), "fp16", 0, 512)
    assert base != auto._engine_key(Model(scale=2), "fp16", 0, 512)
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace(__version__="12"))
    assert base != auto._engine_key(Model(), "fp16", 0, 512)
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace(__version__="11.4"))
    monkeypatch.setattr(build_cache, "FORMAT_VERSION", build_cache.FORMAT_VERSION + 1)
    assert base != auto._engine_key(Model(), "fp16", 0, 512)


def test_the_recipe_sources_exist():
    # A source renamed away would turn every run into a fallback.
    for source in auto._RECIPE_SOURCES:
        assert source.is_file(), source


@pytest.mark.parametrize(
    "arch", ["ESRGAN", "HAT", "DAT", "RealESRGAN Compact", "SPAN", "PLKSR", "OmniSR"]
)
def test_the_automatic_fp16_architectures_have_a_recipe(arch):
    assert auto._recipe(Model(arch=arch)) == "fp16"


@pytest.mark.parametrize(
    ("arch", "tags", "recipe"),
    [
        ("RCAN", [], "fp16"),
        ("RetinexFormer", [], "fp16"),
        ("SwinIR", ["small", "60dim"], "fp16"),
        ("SwinIR", ["medium"], "fp16"),
        ("SwinIR", [], None),
        ("FDAT", ["medium"], "fp16"),
        ("FDAT", ["xl"], "fp16"),
        ("FDAT", ["tiny"], None),
        ("SeemoRe", ["Tiny"], None),
        ("NAFNet", [], None),
    ],
)
def test_a_recipe_goes_by_the_list_build_engine_follows(arch, tags, recipe):
    # One model gets one precision, through this setting or through the nodes.
    assert auto._recipe(Model(arch=arch, tags=tags)) == recipe


@pytest.mark.parametrize(
    ("tile", "size"),
    [
        (TileSize(128), 128),
        (TileSize(1024), 1024),
        (TileSize(100), 100),
        (TileSize(16), 64),
        (ESTIMATE, 512),
        (MAX_TILE_SIZE, 512),
        (NO_TILING, 512),
    ],
)
def test_the_engine_goes_up_to_the_tile_size(tile, size):
    assert auto._engine_tile(Model(), tile, 0) == size


@pytest.mark.parametrize("arch", ["DAT", "HAT", "FIGSR"])
@pytest.mark.parametrize(
    ("tile", "size"),
    [
        (ESTIMATE, 1408),
        (MAX_TILE_SIZE, 1408),
        (NO_TILING, 1408),
        (TileSize(256), 256),
        (TileSize(4096), 4096),
    ],
)
def test_dat_and_hat_engines_take_whole_images(arch, tile, size, monkeypatch):
    monkeypatch.setattr(auto, "migraphx_here", lambda: False)
    monkeypatch.setattr(auto.torch.cuda, "is_available", lambda: True)
    memory = {1: SimpleNamespace(total_memory=auto.WHOLE_IMAGE_MEMORY)}
    monkeypatch.setattr(auto.torch.cuda, "get_device_properties", memory.__getitem__)
    assert auto._engine_tile(Model(arch=arch), tile, 1) == size


@pytest.mark.parametrize("tile", [ESTIMATE, MAX_TILE_SIZE, NO_TILING])
def test_a_pytorch_without_cuda_keeps_the_default_quietly(tile, monkeypatch):
    # A chain saved with the engine on, opened where PyTorch has no CUDA, runs on PyTorch.
    monkeypatch.setattr(auto, "migraphx_here", lambda: False)
    monkeypatch.setattr(auto.torch.cuda, "is_available", lambda: False)
    assert auto._engine_tile(Model(arch="HAT"), tile, 0) == 512


@pytest.mark.parametrize(
    ("channels", "size"), [(64, 1920), (128, 1920), (192, 1536), (256, 1344)]
)
def test_a_wider_dual_takes_a_smaller_whole_image(channels, size, monkeypatch):
    # Its build's 8 GiB workspace held 1920 px at 128 channels, not 2048 px.
    monkeypatch.setattr(auto, "migraphx_here", lambda: False)
    monkeypatch.setattr(auto.torch.cuda, "is_available", lambda: True)
    memory = SimpleNamespace(total_memory=auto.WHOLE_IMAGE_MEMORY)
    monkeypatch.setattr(auto.torch.cuda, "get_device_properties", lambda _: memory)
    model = Model(arch="DUAL")
    model.model.embed_dim = channels
    assert auto._engine_tile(model, ESTIMATE, 0) == size
    assert auto._engine_tile(model, TileSize(2048), 0) == 2048


@pytest.mark.parametrize("tile", [ESTIMATE, MAX_TILE_SIZE, NO_TILING])
def test_a_gpu_with_less_memory_keeps_the_default_for_dat(tile, monkeypatch):
    monkeypatch.setattr(auto, "migraphx_here", lambda: False)
    monkeypatch.setattr(auto.torch.cuda, "is_available", lambda: True)
    memory = SimpleNamespace(total_memory=auto.WHOLE_IMAGE_MEMORY - 1)
    monkeypatch.setattr(auto.torch.cuda, "get_device_properties", lambda _: memory)
    assert auto._engine_tile(Model(arch="DAT"), tile, 0) == 512


@pytest.mark.parametrize("tile", [ESTIMATE, MAX_TILE_SIZE, NO_TILING])
def test_a_program_of_one_size_keeps_the_default_for_dat(tile, monkeypatch):
    monkeypatch.setattr(auto, "migraphx_here", lambda: True)
    assert auto._engine_tile(Model(arch="DAT"), tile, 0) == 512


@pytest.mark.parametrize(
    ("tile", "size"),
    [(TileSize(128), 128), (TileSize(256), 256), (TileSize(512), 256), (ESTIMATE, 256)],
)
def test_rgt_stops_where_its_memory_holds(tile, size):
    # Its Dynamic engine at 512 px nearly filled a 32 GB card.
    assert auto._engine_tile(Model(arch="RGT"), tile, 0) == size


def test_rgt_has_a_recipe_and_the_setting():
    assert auto._recipe(Model(arch="RGT")) == "fp16"
    assert '"RGT"' in auto.ENGINE_MODEL_NAVI


def test_each_tile_size_has_its_own_engine(fakes, tmp_path):
    context, model = Context(tmp_path), Model()
    for tile in (TileSize(256), TileSize(1024), TileSize(256), ESTIMATE):
        run(context, model, tile=tile)
    assert fakes.tiles == [256, 1024, 512]
    assert fakes.engine_runs == 4
    assert len(kept_files(tmp_path)) == 3
    assert context.warnings == []


def test_a_tile_rgt_cannot_take_says_so_once(fakes, tmp_path):
    context, model = Context(tmp_path), Model(arch="RGT")
    for _ in range(2):
        run(context, model, tile=TileSize(512))
    assert fakes.tiles == [256]
    assert len(context.warnings) == 1
    assert "256 px" in context.warnings[0]
    assert "512 px" in context.warnings[0]


def test_auto_says_nothing_about_the_size_rgt_takes(fakes, tmp_path):
    context, model = Context(tmp_path), Model(arch="RGT")
    run(context, model)
    assert fakes.tiles == [256]
    assert context.warnings == []


def test_dual_has_a_recipe():
    assert auto._recipe(Model(arch="DUAL")) == "dual"


def test_a_model_without_a_recipe_runs_on_pytorch_without_a_warning(fakes, tmp_path):
    context = Context(tmp_path)
    np.testing.assert_array_equal(run(context, Model(arch="Swin2SR")), IMAGE * 3)
    assert fakes.builds == 0
    assert context.warnings == []


def test_a_drft_that_is_not_legacy_says_why(fakes, tmp_path):
    context = Context(tmp_path)
    np.testing.assert_array_equal(run(context, Model(arch="DRFT")), IMAGE * 3)
    assert fakes.builds == 0
    assert "only legacy DRFT" in context.warnings[0]


def test_no_nvidia_stays_on_pytorch_quietly(monkeypatch):
    """The input is hidden there: a saved On could not be turned off."""
    monkeypatch.setattr(auto, "nvidia", SimpleNamespace(is_available=False))
    assert auto._no_engine_here(GPU) == ""


def test_rocm_stays_on_pytorch_quietly(monkeypatch):
    monkeypatch.setattr(auto, "nvidia", SimpleNamespace(is_available=True))
    monkeypatch.setattr(torch.version, "hip", "7.14")
    assert auto._no_engine_here(GPU) == ""


def test_a_chain_saved_on_warns_nothing_where_the_input_is_hidden(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(auto, "nvidia", SimpleNamespace(is_available=False))
    auto._outcomes.clear()
    context, model = Context(tmp_path), Model()
    for _ in range(2):
        np.testing.assert_array_equal(run(context, model), IMAGE * 3)
        context.end_chain()
    assert context.warnings == []
    auto._outcomes.clear()


def test_cpu_mode_stays_on_pytorch(monkeypatch):
    monkeypatch.setattr(auto, "nvidia", SimpleNamespace(is_available=True))
    monkeypatch.setattr(torch.version, "hip", None)
    assert "CPU" in str(auto._no_engine_here(torch.device("cpu")))


def test_a_disabled_tensorrt_package_says_why(monkeypatch):
    monkeypatch.setattr(auto, "nvidia", SimpleNamespace(is_available=True))
    monkeypatch.setattr(torch.version, "hip", None)
    package = ModuleType("packages.chaiNNer_tensorrt")
    package.package = SimpleNamespace(disabled=True, disabled_reason="an old driver")
    monkeypatch.setitem(sys.modules, "packages.chaiNNer_tensorrt", package)
    assert auto._no_engine_here(GPU) == "an old driver"


def test_a_missing_tensorrt_says_so(monkeypatch):
    monkeypatch.setattr(auto, "nvidia", SimpleNamespace(is_available=True))
    monkeypatch.setattr(torch.version, "hip", None)
    package = ModuleType("packages.chaiNNer_tensorrt")
    package.package = SimpleNamespace(disabled=False, disabled_reason=None)
    monkeypatch.setitem(sys.modules, "packages.chaiNNer_tensorrt", package)
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    assert "not installed" in str(auto._no_engine_here(GPU))


def test_a_reason_warns_once_per_chain_and_the_next_chain_tries_again(
    fakes, tmp_path, monkeypatch
):
    context, model = Context(tmp_path), Model()
    reasons = ["PyTorch is set to the CPU."]
    monkeypatch.setattr(auto, "_no_engine_here", lambda _d: reasons[0])
    for _ in range(3):
        np.testing.assert_array_equal(run(context, model), IMAGE * 3)
    assert len(context.warnings) == 1
    assert "PyTorch is set to the CPU." in context.warnings[0]

    context.end_chain()
    reasons[0] = None
    np.testing.assert_array_equal(run(context, model), IMAGE * 2)


def test_a_failed_build_falls_back_says_why_and_is_not_retried_in_the_chain(
    fakes, tmp_path
):
    context, model = Context(tmp_path), Model()
    fakes.build_error = RuntimeError("Failed to build TensorRT engine.")
    for _ in range(3):
        np.testing.assert_array_equal(run(context, model), IMAGE * 3)
    assert fakes.builds == 1
    assert len(context.warnings) == 1
    assert "Failed to build TensorRT engine." in context.warnings[0]
    assert kept_files(tmp_path) == []

    context.end_chain()
    fakes.build_error = None
    np.testing.assert_array_equal(run(context, model), IMAGE * 2)
    assert fakes.builds == 2


def test_a_whole_image_engine_that_does_not_build_leaves_images_to_512_px(
    fakes, tmp_path, monkeypatch
):
    monkeypatch.setattr(auto, "migraphx_here", lambda: False)
    monkeypatch.setattr(auto.torch.cuda, "is_available", lambda: True)
    memory = SimpleNamespace(total_memory=auto.WHOLE_IMAGE_MEMORY)
    monkeypatch.setattr(auto.torch.cuda, "get_device_properties", lambda _: memory)
    fake_build = auto._build
    wholes = []

    def build(context, model, tile, whole=False):
        wholes.append(whole)
        if tile == auto.WHOLE_IMAGE_TILE["HAT"]:
            fakes.tiles.append(tile)
            raise RuntimeError("Failed to build TensorRT engine.")
        return fake_build(context, model, tile, whole)

    monkeypatch.setattr(auto, "_build", build)
    context, model = Context(tmp_path), Model(arch="HAT")
    for _ in range(2):
        np.testing.assert_array_equal(run(context, model), IMAGE * 2)
    assert fakes.tiles == [1408, 512]
    assert wholes == [True, False]
    assert context.warnings == []
    assert len(kept_files(tmp_path)) == 1
    # A later process tries the whole-image engine again (the failure may have been the
    # GPU's memory at the time), then reads the 512 px engine without building it.
    auto._outcomes.clear()
    build_cache._memory.clear()
    np.testing.assert_array_equal(run(Context(tmp_path), model), IMAGE * 2)
    assert fakes.tiles == [1408, 512, 1408]


def test_a_tile_size_of_the_whole_image_size_is_not_a_whole_image_engine(
    fakes, tmp_path, monkeypatch
):
    monkeypatch.setattr(auto, "migraphx_here", lambda: False)
    monkeypatch.setattr(auto.torch.cuda, "is_available", lambda: True)
    memory = SimpleNamespace(total_memory=auto.WHOLE_IMAGE_MEMORY)
    monkeypatch.setattr(auto.torch.cuda, "get_device_properties", lambda _: memory)
    wholes = []
    fake_build = auto._build

    def build(context, model, tile, whole=False):
        wholes.append(whole)
        return fake_build(context, model, tile, whole)

    monkeypatch.setattr(auto, "_build", build)
    context, model = Context(tmp_path), Model(arch="HAT")
    run(context, model, tile=TileSize(1408))
    run(context, model)
    assert fakes.tiles == [1408, 1408]
    assert wholes == [False, True]


def test_a_missing_tensorrt_falls_back(fakes, tmp_path, monkeypatch):
    context = Context(tmp_path)
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    np.testing.assert_array_equal(run(context, Model()), IMAGE * 3)
    assert "ModuleNotFoundError" in context.warnings[0] or (
        "ImportError" in context.warnings[0]
    )


def test_stop_during_the_build_is_stop_and_is_not_remembered(fakes, tmp_path):
    context, model = Context(tmp_path), Model()
    fakes.build_error = Aborted()
    with pytest.raises(Aborted):
        run(context, model)
    assert context.warnings == []
    fakes.build_error = None
    np.testing.assert_array_equal(run(context, model), IMAGE * 2)


def test_stop_during_the_engine_is_stop(fakes, tmp_path):
    context = Context(tmp_path)
    fakes.engine_error = Aborted()
    with pytest.raises(Aborted):
        run(context, Model())
    assert context.warnings == []


def test_an_engine_that_fails_gives_the_whole_image_to_pytorch(fakes, tmp_path):
    context, model = Context(tmp_path), Model()
    np.testing.assert_array_equal(run(context, model), IMAGE * 2)
    fakes.engine_error = RuntimeError("CUDA error 2: out of memory")
    np.testing.assert_array_equal(run(context, model), IMAGE * 3)
    assert "out of memory" in context.warnings[0]
    # Its session goes, so that PyTorch has the GPU's memory.
    assert fakes.released == [NODE]

    runs = fakes.engine_runs
    np.testing.assert_array_equal(run(context, model), IMAGE * 3)
    assert fakes.engine_runs == runs
    assert len(context.warnings) == 1


def test_the_whole_image_starts_over_when_the_second_pass_fails(
    fakes, tmp_path, monkeypatch
):
    """A transparent image's two passes must come from one backend."""
    context, model = Context(tmp_path), Model()
    used: list[str] = []
    calls = {"n": 0}

    def engine_op(_context, _model, _engine, _report=None):
        def op(img):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("second pass")
            used.append("engine")
            return img

        return op

    def two_passes(op):
        return op(op(IMAGE))

    def on_pytorch(img):
        used.append("pytorch")
        return img

    monkeypatch.setattr(auto, "_engine_op", engine_op)
    auto.upscale_through_engine(
        context, model, GPU, 0, ESTIMATE, two_passes, on_pytorch
    )
    assert used == ["engine", "pytorch", "pytorch"]


def test_the_engine_path_hands_the_nodes_report_to_the_engine(
    fakes, tmp_path, monkeypatch
):
    """The Tile Size output tells of the engine's tiles when the engine ran."""
    reports = []

    def engine_op(_context, _model, _engine, report=None):
        reports.append(report)
        return lambda img: img

    monkeypatch.setattr(auto, "_engine_op", engine_op)
    report = TileReport()
    auto.upscale_through_engine(
        Context(tmp_path),
        Model(),
        GPU,
        0,
        ESTIMATE,
        lambda op: op(IMAGE),
        pytorch,
        report,
    )
    assert reports == [report]


def fake_tensorrt_node(monkeypatch, output):
    module = ModuleType("packages.chaiNNer_tensorrt.tensorrt.processing.upscale_image")
    calls = []

    def engine_upscale(
        context, img, engine, separate_alpha, balanced_tiles, before_clip=None
    ):
        calls.append((context, img, engine, separate_alpha, balanced_tiles))
        if before_clip is not None:
            before_clip(output)
        # As the real function: the image and the tiles it ran
        return output, "Whole image"

    module.engine_upscale = engine_upscale
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


def test_the_engine_runs_through_tensorrts_own_node(monkeypatch):
    calls = fake_tensorrt_node(monkeypatch, IMAGE)
    engine = engine_of()
    context = object()
    report = TileReport()
    output = auto._engine_op(context, Model(), engine, report)(IMAGE)
    assert output is IMAGE
    assert report.text == "Whole image"
    # The image already has the model's channels: basic_upscale's separate alpha applies.
    assert calls == [(context, IMAGE, engine, False, False)]


def test_the_call_fits_tensorrts_real_node():
    """A call that no longer fits would only show as a fallback on every image."""
    module = pytest.importorskip(
        "packages.chaiNNer_tensorrt.tensorrt.processing.upscale_image"
    )
    inspect.signature(module.engine_upscale).bind(
        object(), IMAGE, engine_of(), False, False, before_clip=print
    )


def test_invalid_values_from_a_finite_image_are_a_failure(monkeypatch):
    nan = IMAGE.copy()
    nan[0, 0, 0] = np.nan
    fake_tensorrt_node(monkeypatch, nan)
    with pytest.raises(RuntimeError, match="invalid"):
        auto._engine_op(object(), Model(), engine_of())(IMAGE)
    # NaN already in the input passes through, as on PyTorch.
    np.testing.assert_array_equal(
        auto._engine_op(object(), Model(), engine_of())(nan), nan
    )


def test_invalid_values_fall_back(fakes, tmp_path, monkeypatch):
    context, model = Context(tmp_path), Model()

    def checked(_context, _model, _engine, _report=None):
        def op(_img):
            raise RuntimeError("returned invalid values (NaN or infinity)")

        return op

    monkeypatch.setattr(auto, "_engine_op", checked)
    np.testing.assert_array_equal(run(context, model), IMAGE * 3)
    assert "invalid values" in context.warnings[0]


def test_a_different_gpu_asks_again(fakes, tmp_path):
    context, model = Context(tmp_path), Model()
    run(context, model, gpu=0)
    run(context, model, gpu=1)
    # The fake GPU's name and capability are the same, so the second GPU finds the engine.
    assert fakes.builds == 1
    assert auto._outcomes[model][0] == (GPU, 1, 512, False)


def test_release_only_touches_a_node_that_used_an_engine(fakes, tmp_path):
    auto.release_engine(NODE)
    assert fakes.released == []
    run(Context(tmp_path), Model())
    assert fakes.released == []
    auto.release_engine(NODE)
    auto.release_engine(NODE)
    assert fakes.released == [NODE]


def tiny_onnx():
    relu = helper.make_node("Relu", ["input"], ["middle"])
    both = helper.make_node("Add", ["middle", "input"], ["output"])
    tensor = helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT16, [1, 3, 8, 8]
    )
    out = helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT16, [1, 3, 8, 8]
    )
    graph = helper.make_graph([relu, both], "tiny", [tensor], [out])
    model = helper.make_model(graph)
    model.metadata_props.add(key="kept", value="yes")
    return model.SerializeToString()


def test_the_aligned_input_is_named_everywhere_and_the_rest_is_kept():
    model = onnx.load_from_string(auto._with_aligned_input(tiny_onnx(), 4))
    assert model.graph.input[0].name == "input_aligned_4"
    assert model.graph.output[0].name == "output"
    assert [list(node.input) for node in model.graph.node] == [
        ["input_aligned_4"],
        ["middle", "input_aligned_4"],
    ]
    assert model.metadata_props[0].value == "yes"
    onnx.checker.check_model(model)


def test_build_wires_convert_to_onnx_and_build_engine(monkeypatch):
    """The real nodes' signatures take these arguments, in the automatic recipe's values."""
    convert_module = pytest.importorskip(
        "packages.chaiNNer_pytorch.pytorch.utility.convert_to_onnx"
    )
    build_module = pytest.importorskip(
        "packages.chaiNNer_tensorrt.tensorrt.utility.build_engine"
    )
    seen: dict[str, inspect.BoundArguments] = {}
    onnx_model = SimpleNamespace(bytes=tiny_onnx())
    renamed = object()
    engine = engine_of(scale=4)

    def fake_convert(*args):
        seen["convert"] = inspect.signature(real_convert).bind(*args)
        return onnx_model, "fp16", "opset20"

    def fake_build(*args):
        seen["build"] = inspect.signature(real_build).bind(*args)
        return engine

    real_convert = convert_module.convert_to_onnx_node
    real_build = build_module.build_engine_node
    monkeypatch.setattr(convert_module, "convert_to_onnx_node", fake_convert)
    monkeypatch.setattr(build_module, "build_engine_node", fake_build)
    loaded = []

    def fake_load(onnx_bytes):
        loaded.append(onnx_bytes)
        return renamed

    monkeypatch.setattr(onnx_load, "load_onnx_model", fake_load)

    context = Context(Path("."))
    assert auto._build(context, Model(arch="SPAN"), 512) is engine
    convert = seen["convert"].arguments
    assert convert["data_type"] == convert_module.OnnxDataType.FP16
    assert convert["opset"].value == auto.OPSET
    assert convert["verify"] is False
    assert convert["dual_shape"] == convert_module.DualShape.DYNAMIC
    build = seen["build"].arguments
    # The input names the model's own multiple, so an image pads only as PyTorch pads it.
    assert build["onnx_model"] is renamed
    assert onnx.load_from_string(loaded[0]).graph.input[0].name == "input_aligned_1"
    assert build["shape_mode"] == build_module.ShapeMode.DYNAMIC
    assert (build["min_height"], build["min_width"]) == (16, 16)
    assert (build["opt_height"], build["opt_width"]) == (512, 512)
    assert (build["max_height"], build["max_width"]) == (512, 512)
    auto._build(context, Model(arch="SPAN"), 1024)
    assert (
        seen["build"].arguments["opt_height"],
        seen["build"].arguments["max_width"],
    ) == (1024, 1024)
    # The largest size keeps to the model's multiple, and is at least the smallest.
    aligned = Model(arch="ESRGAN")
    aligned.size_requirements = SimpleNamespace(minimum=128, multiple_of=64)
    auto._build(context, aligned, 100)
    assert seen["build"].arguments["min_height"] == 128
    assert seen["build"].arguments["max_height"] == 128
    assert build["workspace"] == 0
    assert build["allow_tf32"] is False
    assert build["optimization_level"] == 3

    # Models of convolutions take small sizes; any other keeps Build Engine's default,
    # and any model keeps to its requirement.
    for arch in ("ESRGAN", "RealESRGAN Compact", "PLKSR"):
        auto._build(context, Model(arch=arch), 512)
        assert seen["build"].arguments["min_height"] == 16, arch
    for arch in ("HAT", "DAT", "OmniSR"):
        auto._build(context, Model(arch=arch), 512)
        assert seen["build"].arguments["min_height"] == 64, arch
    # An engine for whole images keeps its optimum at the default size; a Tile Size of
    # another size keeps it at its own.
    for arch in ("HAT", "DAT"):
        auto._build(context, Model(arch=arch), auto.WHOLE_IMAGE_TILE[arch], True)
        whole = seen["build"].arguments
        assert (whole["opt_height"], whole["opt_width"]) == (512, 512), arch
        assert (whole["max_height"], whole["max_width"]) == (1408, 1408), arch
        auto._build(context, Model(arch=arch), 1408)
        assert seen["build"].arguments["opt_height"] == 1408, arch
    required = Model(arch="ESRGAN")
    required.size_requirements = SimpleNamespace(minimum=20, multiple_of=8)
    auto._build(context, required, 512)
    assert seen["build"].arguments["min_height"] == 24
    assert onnx.load_from_string(loaded[-1]).graph.input[0].name == "input_aligned_8"

    # A legacy DRFT's smallest side is two windows, and its input names itself.
    monkeypatch.setattr(auto, "_recipe", lambda _m: "drft")
    model = Model()
    model.model.window_size = 64
    count = len(loaded)
    auto._build(context, model, 512)
    assert seen["build"].arguments["min_height"] == 128
    assert seen["build"].arguments["onnx_model"] is onnx_model
    assert len(loaded) == count

    wrong = engine_of(scale=2)
    monkeypatch.setattr(build_module, "build_engine_node", lambda *_a: wrong)
    with pytest.raises(ValueError, match="upscales 2x"):
        auto._build(context, model, 512)


def tiny_esrgan():
    from spandrel import ModelLoader
    from spandrel.architectures.ESRGAN import ESRGAN

    torch.manual_seed(0)
    network = ESRGAN(in_nc=3, out_nc=3, num_filters=8, num_blocks=1, scale=4)
    return ModelLoader("cpu").load_from_state_dict(network.state_dict()).eval()


@pytest.fixture
def node_module():
    return pytest.importorskip(
        "packages.chaiNNer_pytorch.pytorch.processing.upscale_image"
    )


# The node's input, imported as the server imports it, on a PC of each kind.
THE_NODES_INPUT = """
import json, sys
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
import torch, spandrel, gpu
gpu.nvidia = SimpleNamespace(
    is_available={nvidia}, any_needs_legacy_cuda=False, all_support_fp16=True, devices=[]
)
from api.group import Group
from nodes.condition import Condition
from nodes.impl.pytorch import auto_engine
auto_engine.migraphx_here = lambda: {migraphx}
import packages.chaiNNer_pytorch
__import__("packages.chaiNNer_pytorch.pytorch.processing.upscale_image", fromlist=["x"])
from packages.chaiNNer_pytorch.pytorch import processing_group
node = next(n for n in processing_group.nodes if n.schema_id == "{schema}")
wanted = (
    Condition.const({shown})
    & Condition.type(0, auto_engine.ENGINE_MODEL_NAVI, if_not_connected=True)
).to_json()
def holds_it(item):
    return isinstance(item, Group) and any(i == 8 or holds_it(i) for i in item.items)
groups = [g for g in node.group_layout if holds_it(g)]
input_8 = next(i for i in node.inputs if i.id == 8)
print("RESULT" + json.dumps(dict(
    label=input_8.label,
    default=input_8.default,
    matches=groups[0].info.options["condition"] == wanted,
)))
"""


@pytest.mark.parametrize(
    ("nvidia", "migraphx", "label", "shown"),
    [
        (True, False, "Use TensorRT Engine", True),
        (False, True, "Use MIGraphX Program", True),
        # A PC with neither shows no input, whatever a saved chain holds.
        (False, False, "Use TensorRT Engine", False),
    ],
    ids=["nvidia", "amd", "neither"],
)
def test_the_same_input_on_either_kind_of_pc(nvidia, migraphx, label, shown):
    backend = Path(auto.__file__).parents[3]
    done = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            THE_NODES_INPUT.format(
                nvidia=nvidia, migraphx=migraphx, schema=SCHEMA, shown=shown
            ),
            str(backend),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    line = next(x for x in done.stdout.splitlines() if x.startswith("RESULT"))
    result = json.loads(line.removeprefix("RESULT"))
    assert result["label"] == label
    assert result["default"] == 0
    assert result["matches"]


def test_the_setting_is_off_by_default_and_its_id_is_new(node_module):
    from packages.chaiNNer_pytorch.pytorch import processing_group

    node = next(n for n in processing_group.nodes if n.schema_id == SCHEMA)
    inputs = {i.id: i for i in node.inputs}
    assert inputs[8].label == "Use TensorRT Engine"
    assert inputs[8].default == 0
    assert sorted(inputs) == list(range(9))


def test_off_is_the_path_the_node_always_ran(node_module, tmp_path, monkeypatch):
    from nodes.impl.upscale.auto_split_tiles import NO_TILING
    from nodes.impl.upscale.basic_upscale import PaddingType, UpscaleInfo, basic_upscale

    def never(*_args):
        raise AssertionError("the engine path ran with the setting off")

    monkeypatch.setattr(node_module, "upscale_through_engine", never)
    model = tiny_esrgan()
    context = Context(tmp_path, {"use_cpu": True})
    image = np.random.default_rng(0).random((16, 16, 3), dtype=np.float32)
    node = node_module.upscale_image_node
    output, tiles = node(
        context, image, model, False, 4, NO_TILING, 256, PaddingType.NONE, False, False
    )
    assert tiles == "Whole image"
    expected = basic_upscale(
        image,
        lambda i: node_module.upscale(
            i, model, NO_TILING, node_module.get_settings(context), context
        ),
        upscale_info=UpscaleInfo(in_nc=3, out_nc=3, scale=4),
        scale=4,
        separate_alpha=False,
        padding=PaddingType.NONE,
        clip=False,
    )
    assert output.shape == (64, 64, 3)
    assert output.tobytes() == expected.tobytes()
    # Off keeps nothing on disk.
    assert not (tmp_path / "cache").exists()


def test_on_in_cpu_mode_gives_pytorchs_output_and_says_why(
    node_module, tmp_path, monkeypatch
):
    from nodes.impl.upscale.auto_split_tiles import NO_TILING
    from nodes.impl.upscale.basic_upscale import PaddingType

    monkeypatch.setattr(auto, "nvidia", SimpleNamespace(is_available=True))
    monkeypatch.setattr(torch.version, "hip", None)
    auto._outcomes.clear()
    model = tiny_esrgan()
    context = Context(tmp_path, {"use_cpu": True})
    image = np.random.default_rng(1).random((16, 16, 3), dtype=np.float32)
    node = node_module.upscale_image_node
    args = (context, image, model, False, 4, NO_TILING, 256, PaddingType.NONE, False)
    on, _ = node(*args, True)
    off, _ = node(*args, False)
    assert on.tobytes() == off.tobytes()
    assert len(context.warnings) == 1
    assert "PyTorch is set to the CPU." in context.warnings[0]
    auto._outcomes.clear()


# The node's module imported as the server imports it, in a process of its own.
IMPORT_THE_NODE = """
import sys
sys.path.insert(0, sys.argv[1])
{before}
import torch, spandrel
import packages.chaiNNer_pytorch
before = set(sys.modules)
module = __import__(
    "packages.chaiNNer_pytorch.pytorch.processing.upscale_image", fromlist=["x"]
)
new = set(sys.modules) - before
print("loaded", hasattr(module, "upscale_image_node"))
print("heavy", sorted(n for n in new if n == "onnx" or n.startswith("nodes.impl.onnx.load")
    or n.startswith("nodes.impl.tensorrt.")))
"""


def import_the_node(before: str) -> str:
    backend = Path(auto.__file__).parents[3]
    done = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            IMPORT_THE_NODE.format(before=before),
            str(backend),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout


def test_the_node_loads_without_the_onnx_package():
    """A node module that fails to import loses the node, and chains drop it."""
    output = import_the_node('sys.modules["onnx"] = None')
    assert "loaded True" in output


def test_the_off_path_imports_neither_onnx_nor_the_tensorrt_modules():
    assert "heavy []" in import_the_node("")


def test_without_the_onnx_package_the_engine_falls_back(fakes, tmp_path, monkeypatch):
    context = Context(tmp_path)
    source = tiny_onnx()
    monkeypatch.setitem(sys.modules, "onnx", None)
    monkeypatch.setattr(
        auto, "_build", lambda _c, _m: auto._with_aligned_input(source, 1)
    )
    np.testing.assert_array_equal(run(context, Model()), IMAGE * 3)
    assert "onnx" in context.warnings[0]


def test_the_tile_size_output_tells_of_the_engines_tiles(
    node_module, fakes, tmp_path, monkeypatch
):
    from nodes.impl.upscale.auto_split_tiles import NO_TILING
    from nodes.impl.upscale.basic_upscale import PaddingType

    monkeypatch.setattr(auto, "_engine_op", REAL_ENGINE_OP)
    fake_tensorrt_node(monkeypatch, np.zeros((32, 32, 3), dtype=np.float32))
    image = np.full((8, 8, 3), 0.25, dtype=np.float32)
    _, tiles = node_module.upscale_image_node(
        Context(tmp_path, {"use_cpu": True}),
        image,
        tiny_esrgan(),
        False,
        4,
        NO_TILING,
        256,
        PaddingType.NONE,
        False,
        True,
    )
    assert tiles == "Whole image"


def test_a_kept_engine_does_not_outlive_the_cpu_setting(fakes, tmp_path, monkeypatch):
    """The app keeps Load Model's output between runs: one model sees the setting change."""
    monkeypatch.setattr(
        auto,
        "_no_engine_here",
        lambda d: None if d.type == "cuda" else "PyTorch is set to the CPU.",
    )
    context, model = Context(tmp_path), Model()
    run(context, model, device=GPU)
    assert fakes.engine_runs == 1
    context.end_chain()
    np.testing.assert_array_equal(
        run(context, model, device=torch.device("cpu")), IMAGE * 3
    )
    assert fakes.engine_runs == 1
    assert "PyTorch is set to the CPU." in context.warnings[0]
    assert fakes.released == [NODE]

    # And back: the engine again, without another build.
    context.end_chain()
    np.testing.assert_array_equal(run(context, model, device=GPU), IMAGE * 2)
    assert fakes.builds == 1


def test_recipe_sources_cover_the_export_path():
    """A step of the export left out would keep an engine made by its older code."""
    here = Path(auto.__file__).parent
    listed = {source.resolve() for source in auto._RECIPE_SOURCES}
    tree = ast.parse((here / "convert_to_onnx_impl.py").read_text(encoding="utf-8"))
    siblings = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None
        for alias in node.names
    ]
    assert "input_export" in siblings
    for name in siblings:
        assert (here / f"{name}.py").resolve() in listed, name
    # Build Engine's rewrites of the ONNX
    tensorrt = here.parent / "tensorrt"
    for name in ("dense_blocks", "engine_builder", "group_norm"):
        assert (tensorrt / f"{name}.py").resolve() in listed, name


def test_an_image_that_fails_on_both_keeps_the_engine_for_the_next_image(
    fakes, tmp_path
):
    """With Stop on First Error off, an image bad on PyTorch too says nothing of the engine."""
    context, model = Context(tmp_path), Model()

    def bad(_op):
        raise ValueError("bad image")

    run(context, model)
    assert fakes.engine_runs == 1
    with pytest.raises(ValueError, match="bad image"):
        auto.upscale_through_engine(context, model, GPU, 0, ESTIMATE, bad, pytorch)
    np.testing.assert_array_equal(run(context, model), IMAGE * 2)
    assert fakes.engine_runs == 2
    assert context.warnings == []


def test_stop_on_the_way_to_pytorch_leaves_the_engine_kept(fakes, tmp_path):
    context, model = Context(tmp_path), Model()
    run(context, model)
    fakes.engine_error = RuntimeError("the engine failed")

    def stopped(_img):
        raise Aborted()

    with pytest.raises(Aborted):
        auto.upscale_through_engine(
            context, model, GPU, 0, ESTIMATE, lambda op: op(IMAGE), stopped
        )
    assert context.warnings == []
    fakes.engine_error = None
    np.testing.assert_array_equal(run(context, model), IMAGE * 2)


def test_the_fp16_setting_toggled_is_one_esrgan_engine(fakes, tmp_path):
    """An ESRGAN's ONNX is byte-identical either way: export_copy halves its weights."""
    context = Context(tmp_path)
    run(context, Model())
    context.end_chain()
    half = Model()
    half.model.half()
    run(context, half)
    assert fakes.builds == 1
    assert len(kept_files(tmp_path)) == 1


def test_another_architecture_keeps_its_weights_as_loaded():
    """RealPLKSR's sampler stays as loaded, so the FP16 setting changes its ONNX."""
    half = Model(arch="PLKSR")
    half.model.half()
    assert auto._weights_digest(half) != auto._weights_digest(Model(arch="PLKSR"))


def a_restart():
    auto._outcomes.clear()
    build_cache._memory.clear()


def test_an_engine_file_that_tensorrt_refuses_is_built_again(fakes, tmp_path):
    """The record is whole, so its SHA-256 passes, but TensorRT cannot start it (a driver
    update, say). The file goes, once; a new engine it refuses too is a plain fallback."""
    context = Context(tmp_path)
    run(context, Model())
    assert fakes.builds == 1
    first = kept_files(tmp_path)
    context.end_chain()

    a_restart()
    fakes.engine_error = RuntimeError("Failed to deserialize the TensorRT engine.")
    model = Model()
    np.testing.assert_array_equal(run(context, model), IMAGE * 3)
    assert kept_files(tmp_path) == []
    assert "built again" in context.warnings[0]

    # The next image builds, and this engine is refused too: PyTorch for the chain.
    np.testing.assert_array_equal(run(context, model), IMAGE * 3)
    assert fakes.builds == 2
    assert len(context.warnings) == 2
    assert kept_files(tmp_path) == first
    np.testing.assert_array_equal(run(context, model), IMAGE * 3)
    assert fakes.builds == 2
    assert len(context.warnings) == 2
    context.end_chain()

    # Later chains read the new file and do not build a third time.
    a_restart()
    np.testing.assert_array_equal(run(context, Model()), IMAGE * 3)
    assert fakes.builds == 2


def test_a_rebuilt_engine_that_starts_runs_for_the_next_image(fakes, tmp_path):
    context = Context(tmp_path)
    run(context, Model())
    a_restart()
    model = Model()
    fakes.engine_error = RuntimeError("Failed to deserialize the TensorRT engine.")
    run(context, model)
    fakes.engine_error = None
    np.testing.assert_array_equal(run(context, model), IMAGE * 2)
    assert fakes.builds == 2


def test_a_kept_engine_that_ran_is_not_built_again_when_it_fails_later(fakes, tmp_path):
    context = Context(tmp_path)
    run(context, Model())
    a_restart()
    model = Model()
    run(context, model)
    fakes.engine_error = RuntimeError("the session failed")
    np.testing.assert_array_equal(run(context, model), IMAGE * 3)
    assert fakes.builds == 1
    assert len(kept_files(tmp_path)) == 1


def test_a_kept_engine_short_of_memory_is_not_built_again(fakes, tmp_path):
    context = Context(tmp_path)
    run(context, Model())
    a_restart()
    fakes.engine_error = FakeOutOfMemory("CUDA error: out of memory")
    np.testing.assert_array_equal(run(context, Model()), IMAGE * 3)
    assert fakes.builds == 1
    assert len(kept_files(tmp_path)) == 1


def test_a_bad_image_does_not_cost_a_kept_engine_its_file(fakes, tmp_path):
    context = Context(tmp_path)
    run(context, Model())
    a_restart()

    def bad(_op):
        raise ValueError("bad image")

    with pytest.raises(ValueError, match="bad image"):
        auto.upscale_through_engine(context, Model(), GPU, 0, ESTIMATE, bad, pytorch)
    assert len(kept_files(tmp_path)) == 1
    assert fakes.builds == 1


def test_forgetting_an_engine_takes_its_file_and_its_copy_in_memory(tmp_path):
    build_cache._memory.clear()
    engine = engine_of()
    build_cache.cached_engine("a" * 64, lambda: engine, str(tmp_path))
    build_cache.forget_engine("a" * 64, str(tmp_path))
    assert list(tmp_path.iterdir()) == []
    assert build_cache._recalled("a" * 64) is None
    # A file another program holds is logged: its next read fails again, and falls back.
    build_cache.forget_engine("a" * 64, str(tmp_path))


def test_infinity_from_a_finite_image_is_a_failure(monkeypatch):
    """The node's clip turns it to white or black, so the check comes before the clip."""
    infinite = IMAGE.copy()
    infinite[0, 0, 0] = np.inf
    fake_tensorrt_node(monkeypatch, infinite)
    with pytest.raises(RuntimeError, match="invalid"):
        auto._engine_op(object(), Model(), engine_of())(IMAGE)


class InfiniteRunner:
    """A TileRunner whose tiles hold infinity and minus infinity."""

    def submit(self, tile):
        self.shape = tile.shape

    def collect(self, height, width):
        out = np.full((height, width, self.shape[2]), 0.5, dtype=np.float32)
        out[0, 0, 0] = np.inf
        out[1, 1, 0] = -np.inf
        return out


# Infinity times a blend weight of zero, on the way to the check
@pytest.mark.filterwarnings("ignore:invalid value encountered")
@pytest.mark.parametrize("size", [64, 200], ids=["one tile", "several tiles"])
@pytest.mark.parametrize("channels", [None, 3], ids=["grey", "colour"])
def test_before_clip_sees_what_the_clip_hides(size, channels):
    from nodes.impl.tensorrt.tiling import ShapeBounds, tiled_upscale

    shape = (size, 64) if channels is None else (size, 64, channels)
    finite = []
    output = tiled_upscale(
        np.zeros(shape, dtype=np.float32),
        1,
        ShapeBounds(64, 64, 64, 64),
        InfiniteRunner(),
        clip=True,
        before_clip=lambda raw: finite.append(bool(np.isfinite(raw).all())),
    )
    assert finite == [False]
    # What the check is there for: after the clip, the infinity is white or black.
    assert not np.isinf(output).any()
