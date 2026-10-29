"""The VRAM offload, which PyTorch's Upscale Image imports, loads with a CPU build of torch,
which has none of the CUDA host allocator's calls."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def test_the_offload_loads_without_the_host_allocators_calls():
    script = (
        "import torch\n"
        "if hasattr(torch._C, '_host_emptyCache'):\n"
        "    del torch._C._host_emptyCache\n"
        "from nodes.impl.pytorch import offload\n"
        "offload.release_pinned_memory()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=SRC,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
