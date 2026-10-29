"""Cache lifecycle, publication and geometry boundaries without a CUDA dependency."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch

from api import NodeContext
from nodes.impl.pytorch import vosr2_cache
from nodes.impl.pytorch.vosr2 import BundlePaths, VosrBundle
from nodes.impl.pytorch.vosr2_export import rotary_tables
from nodes.impl.pytorch.vosr2_temporal import coordinates, remap
from nodes.impl.tensorrt.cache import SharedCache
from vendor.vosr2.lightningdit import LightningDiT


@pytest.fixture
def bundle_cache(monkeypatch):
    created = []

    class Bundle:
        def __init__(self, *args, **kwargs):
            self.offloads = 0
            created.append(self)

        def offload(self):
            self.offloads += 1

    cache: SharedCache[Bundle] = SharedCache(lambda bundle: bundle.offload())
    monkeypatch.setattr(vosr2_cache, "VosrBundle", Bundle)
    monkeypatch.setattr(vosr2_cache, "_BUNDLES", cache)
    return created, cache


def test_vosr_bundle_survives_chain_context_and_clear_releases_it(
    bundle_cache, tmp_path
):
    created, cache = bundle_cache
    paths = BundlePaths(tmp_path)
    # A new chain gets a new context object, but retains the same stable node ID.
    for _ in range(2):
        context = cast(NodeContext, SimpleNamespace(node_id="vosr"))
        with vosr2_cache.lease_bundle(
            context, paths, torch.float32, reuse=True
        ) as value:
            assert value is created[0]
    assert len(created) == 1
    assert created[0].offloads == 2
    cache.release_node("vosr")
    assert len(cache) == 0
    with vosr2_cache.lease_bundle(context, paths, torch.float32, reuse=True):
        pass
    assert len(created) == 2


def test_changed_bundle_and_cache_wipe_never_reuse_old_weights(bundle_cache, tmp_path):
    created, cache = bundle_cache
    context = cast(NodeContext, SimpleNamespace(node_id="vosr"))
    paths = BundlePaths(tmp_path)
    with vosr2_cache.lease_bundle(context, paths, torch.float32, reuse=True):
        pass
    paths.args.write_text("changed", encoding="utf-8")
    with vosr2_cache.lease_bundle(context, paths, torch.float32, reuse=True):
        pass
    assert len(created) == 2
    for _ in range(2):
        with vosr2_cache.lease_bundle(context, paths, torch.float32, reuse=False):
            pass
    assert len(created) == 4
    assert len(cache) == 0


def test_cancelled_bundle_is_discarded(bundle_cache, tmp_path):
    created, _ = bundle_cache
    context = cast(NodeContext, SimpleNamespace(node_id="vosr"))
    with pytest.raises(InterruptedError):
        with vosr2_cache.lease_bundle(
            context, BundlePaths(tmp_path), torch.float32, reuse=True
        ):
            raise InterruptedError("cancel")
    with vosr2_cache.lease_bundle(
        context, BundlePaths(tmp_path), torch.float32, reuse=True
    ):
        pass
    assert len(created) == 2


def test_cpu_master_storage_survives_module_device_replacement():
    model: torch.nn.Module = torch.nn.Linear(3, 4).eval().requires_grad_(False)
    expected = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    bundle = VosrBundle.__new__(VosrBundle)
    bundle._cpu_weights = {}
    bundle._cpu_weights[model] = {
        name: tensor.detach() for name, tensor in model.state_dict().items()
    }
    model.to("meta")
    bundle.offload_model(model)
    for name, tensor in model.state_dict().items():
        assert tensor.device.type == "cpu"
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("width", [64, 72])
def test_export_rotary_tables_match_original_geometry(width):
    model = LightningDiT(
        input_size=64, hidden_size=1536, depth=0, num_heads=24, use_rope=True
    )
    x = torch.randn(1, 24, 32 * (width // 2), 64).half()
    rope = (
        model.feat_rope
        if width == 64
        else model._get_dynamic_rope((32, width // 2), x.device, x.dtype)
    )
    assert rope is not None
    cos, sin = rotary_tables(width, x.device, torch.float16)
    paired = x.reshape(*x.shape[:-1], 32, 2)
    rotated = torch.stack((-paired[..., 1], paired[..., 0]), dim=-1).reshape_as(x)
    torch.testing.assert_close(x * cos + rotated * sin, rope(x), rtol=0, atol=0)


def test_cached_coordinates_are_immutable_bounded_and_preserve_warp():
    import cv2

    coordinates.cache_clear()
    grid = coordinates(13, 17)
    assert coordinates(13, 17) is grid
    with pytest.raises(ValueError):
        grid[0, 0] = 4
    image = np.arange(13 * 17, dtype=np.float32).reshape(13, 17)
    flow = np.zeros((13, 17, 2), dtype=np.float32)
    np.testing.assert_array_equal(remap(image, flow, cv2.INTER_LINEAR), image)
    coordinates(5, 6)
    coordinates(7, 8)
    assert coordinates.cache_info().currsize == 2


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="a Windows job object, its suspended start, and files a process holds open",
)
def test_compiler_job_releases_descendant_file_handles_before_cleanup(tmp_path):
    import subprocess
    import time

    import psutil

    from nodes.impl.pytorch.vosr2_process import OwnedJob

    held = tmp_path / "compiler.log"
    ready = tmp_path / "child-ready"
    child_code = (
        "import time;from pathlib import Path;"
        f"held=open({str(held)!r},'wb');"
        f"Path({str(ready)!r}).write_text('ready');time.sleep(30)"
    )
    parent_code = (
        "import subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}]);time.sleep(30)"
    )
    job = OwnedJob()
    process = None
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", parent_code],
            creationflags=subprocess.CREATE_NO_WINDOW | 0x00000004,
        )
        job.assign(process)
        psutil.Process(process.pid).resume()
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "Owned descendant did not start"
        job.close()
        process.wait(timeout=10)
        held.rename(tmp_path / "finished.log")
    finally:
        job.close()
        if process is not None:
            process.wait(timeout=10)


@pytest.mark.skipif(sys.platform != "linux", reason="the job is a Linux process group")
def test_compiler_job_ends_the_compiler_and_its_descendants(tmp_path):
    import subprocess
    import time

    import psutil

    from nodes.impl.pytorch.vosr2_process import OwnedJob

    started = tmp_path / "descendant.pid"
    child_code = (
        "import os,time;from pathlib import Path;"
        f"Path({str(started)!r}).write_text(str(os.getpid()));time.sleep(30)"
    )
    parent_code = (
        "import subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}]);time.sleep(30)"
    )
    job = OwnedJob()
    process = subprocess.Popen(
        [sys.executable, "-c", parent_code],
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        job.assign(process)
        deadline = time.monotonic() + 10
        while not (started.exists() and started.read_text()):
            assert time.monotonic() < deadline, "Owned descendant did not start"
            time.sleep(0.01)
        descendant = psutil.Process(int(started.read_text()))
        job.close()
        process.wait(timeout=10)
        # Orphaned by the compiler's end, then reaped: gone within the timeout
        descendant.wait(timeout=10)
    finally:
        job.close()
        process.wait(timeout=10)


def test_compiler_owns_real_interpreter_and_preserves_virtual_environment(
    monkeypatch, tmp_path
):
    from contextlib import nullcontext
    from pathlib import Path

    from nodes.impl.pytorch import vosr2_engine_cache as cache

    calls = []
    launched = {}
    base = tmp_path / "runtime" / "python.exe"
    launcher = tmp_path / "venv" / "Scripts" / "python.exe"
    monkeypatch.setattr(cache.sys, "executable", str(launcher))
    monkeypatch.setattr(cache.sys, "_base_executable", str(base))
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    gpu = SimpleNamespace(
        index=0,
        get_current_vram_usage=lambda: SimpleNamespace(used=0, total=32_000_000_000),
    )
    monkeypatch.setattr(cache, "nvidia", SimpleNamespace(devices=[gpu]))

    class Job:
        def assign(self, process):
            calls.append("assign")

        def close(self):
            calls.append("close")

    def popen(command, **kwargs):
        launched.update(command=command, **kwargs)
        return SimpleNamespace(
            pid=123,
            poll=lambda: 0,
            returncode=0,
            wait=lambda timeout: calls.append("wait"),
        )

    monkeypatch.setattr(cache, "OwnedJob", Job)
    monkeypatch.setattr(cache.subprocess, "Popen", popen)
    monkeypatch.setattr(
        cache.psutil,
        "Process",
        lambda pid: SimpleNamespace(resume=lambda: calls.append("resume")),
    )
    context = cast(NodeContext, SimpleNamespace())
    cache.build(context, BundlePaths(tmp_path), tmp_path, torch.device("cpu"))
    if sys.platform == "win32":
        assert Path(launched["command"][0]) == base
        assert launched["env"]["__PYVENV_LAUNCHER__"] == str(launcher)
        assert launched["creationflags"] & 0x00000004
        assert calls == ["assign", "resume", "close", "wait"]
    else:
        # The current interpreter (a venv's own, no redirector) leads a session of
        # its own, the job's process group, started running
        assert Path(launched["command"][0]) == launcher
        assert "__PYVENV_LAUNCHER__" not in launched["env"]
        assert launched["start_new_session"] is True
        assert launched["creationflags"] == 0
        assert calls == ["assign", "close", "wait"]
