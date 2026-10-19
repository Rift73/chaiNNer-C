"""PyTorch's process settings (torch_runtime) reach every process that loads a
PyTorch model, VapourSynth's AI Upscale child included, which imports load_model
without the worker's folder walk."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "src"


def test_importing_load_model_applies_the_process_settings():
    # A fresh interpreter: the test process may have imported torch_runtime already.
    code = "\n".join(
        [
            "import sys",
            f"sys.path[:0] = [{str(BACKEND)!r}]",
            "import torch",
            "from torch._C import _get_sdp_priority_order",
            "from torch.nn.attention import SDPBackend",
            "from packages.chaiNNer_pytorch.pytorch.io.load_model import load_model",
            "print(torch.backends.cuda.matmul.fp32_precision)",
            "print(_get_sdp_priority_order()[0] == SDPBackend.CUDNN_ATTENTION.value)",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-2:] == ["tf32", "True"]
