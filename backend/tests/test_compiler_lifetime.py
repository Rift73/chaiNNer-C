"""A compiler or engine build that a dead worker started ends before its slow imports.

On Linux the kernel ends a child only when its parent dies after the child asked for
that (vosr2_process.end_with_parent), and the compilers' first imports (onnx,
tensorrt, torch) take seconds. So the worker names itself in the environment, and each
entry point asks for the signal first and exits at once if that process is already gone.
Stub onnx and tensorrt modules, which record that they were imported, stand in for the
real ones; nothing here needs a GPU.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from api import NodeContext
from nodes.impl.pytorch.vosr2_process import PARENT_PID_VARIABLE

SRC = Path(__file__).resolve().parents[1] / "src"
PYTORCH = SRC / "nodes" / "impl" / "pytorch"

# the command line of each entry point, and the module that it imports first
ENTRY_POINTS = [
    pytest.param(
        [str(PYTORCH / "vosr2_trt_build.py"), *"bundle directory 0 cache".split()],
        "onnx",
        id="vosr2_trt_build",
    ),
    pytest.param(
        [str(PYTORCH / "vosr2_fp8_build.py"), *"graphs directory 0 cache".split()],
        "tensorrt",
        id="vosr2_fp8_build",
    ),
    pytest.param(
        [
            *"-m nodes.impl.tensorrt.drft_engine_worker".split(),
            *"model.onnx model.engine 0 library 0 3 0 1 none".split(),
            *"64x64 64x64 64x64 16".split(),
        ],
        "tensorrt",
        id="drft_engine_worker",
    ),
    pytest.param(
        [
            *"-m nodes.impl.tensorrt.dual_engine_worker".split(),
            *"model.onnx model.engine 0 c32_h64_w64 8 3 0 0 none".split(),
        ],
        "onnx",
        id="dual_engine_worker",
    ),
]


def run_entry_point(
    arguments: list[str], stub: str, tmp_path: Path, parent: int
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run an entry point as its worker does, told that `parent` started it, with a
    stub `stub` module that records its import; the marker file is that record."""
    marker = tmp_path / "imported"
    stubs = tmp_path / "stubs"
    stubs.mkdir(exist_ok=True)
    (stubs / f"{stub}.py").write_text(
        "import os\n"
        "open(os.environ['STUB_MARKER'], 'w').close()\n"
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        PARENT_PID_VARIABLE: str(parent),
        "PYTHONPATH": os.pathsep.join(
            [str(stubs), *filter(None, [os.environ.get("PYTHONPATH")])]
        ),
        "STUB_MARKER": str(marker),
    }
    result = subprocess.run(
        [sys.executable, "-B", *arguments],
        cwd=SRC,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return result, marker


@pytest.mark.skipif(
    sys.platform != "linux", reason="the parent-death signal is Linux only"
)
@pytest.mark.parametrize(("arguments", "stub"), ENTRY_POINTS)
def test_an_entry_point_goes_on_to_its_imports_while_its_parent_lives(
    arguments: list[str], stub: str, tmp_path: Path
):
    result, marker = run_entry_point(arguments, stub, tmp_path, os.getpid())

    assert result.returncode == 7, result.stderr
    assert marker.exists()


@pytest.mark.skipif(
    sys.platform != "linux", reason="the parent-death signal is Linux only"
)
@pytest.mark.parametrize(("arguments", "stub"), ENTRY_POINTS)
def test_an_entry_point_exits_before_its_imports_when_its_parent_is_gone(
    arguments: list[str], stub: str, tmp_path: Path
):
    # a process that is not this child's parent, as a dead worker's pid is not
    result, marker = run_entry_point(arguments, stub, tmp_path, os.getppid())

    assert result.returncode == 1
    assert "has ended" in result.stderr
    assert not marker.exists()


def test_the_vosr_compiler_is_given_its_parents_pid_outside_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from contextlib import nullcontext

    import torch

    from nodes.impl.pytorch import vosr2_engine_cache as cache
    from nodes.impl.pytorch.vosr2 import BundlePaths

    launched = {}
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        cache,
        "get_vram_usage",
        lambda index: SimpleNamespace(used=0, total=32_000_000_000),
    )
    monkeypatch.setattr(
        cache,
        "OwnedJob",
        lambda: SimpleNamespace(assign=lambda process: None, close=lambda: None),
    )
    monkeypatch.setattr(
        cache.psutil,
        "Process",
        lambda pid: SimpleNamespace(resume=lambda: None),
    )

    def popen(command: list[str], **kwargs: object):
        launched.update(command=command, **kwargs)
        return SimpleNamespace(
            pid=123, poll=lambda: 0, returncode=0, wait=lambda timeout: None
        )

    monkeypatch.setattr(cache.subprocess, "Popen", popen)

    context = cast(NodeContext, SimpleNamespace())
    cache.build(context, BundlePaths(tmp_path), tmp_path, torch.device("cpu"))

    assert launched["env"].get(PARENT_PID_VARIABLE) == (
        None if os.name == "nt" else str(os.getpid())
    )
