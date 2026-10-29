"""PyTorch's process settings (torch_runtime) reach every process that loads a
PyTorch model, VapourSynth's AI Upscale child included, which imports load_model
without the worker's folder walk."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "src"


def test_importing_load_model_applies_the_process_settings():
    # A fresh interpreter: the test process may have imported torch_runtime already.
    code = "\n".join(
        [
            "import os, sys",
            f"sys.path[:0] = [{str(BACKEND)!r}]",
            "import torch",
            "from torch._C import _get_sdp_priority_order",
            "from torch.nn.attention import SDPBackend",
            "from packages.chaiNNer_pytorch.pytorch.io.load_model import load_model",
            "print(torch.backends.cuda.matmul.fp32_precision)",
            # cuDNN attention first, or on a ROCm torch the AOTriton switch
            "cudnn = _get_sdp_priority_order()[0] == SDPBackend.CUDNN_ATTENTION.value",
            "switch = 'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL' in os.environ",
            "print(cudnn if torch.version.hip is None else switch)",
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


def _rocm_settings(environment: dict[str, str]) -> list[str]:
    """The attention settings torch_runtime leaves in a fresh interpreter whose torch
    reports a ROCm build (torch.version.hip set, as on a ROCm wheel): whether the
    backend order is PyTorch's own, and the AOTriton switch."""
    code = "\n".join(
        [
            "import os, sys",
            f"sys.path[:0] = [{str(BACKEND)!r}]",
            "import torch",
            "from torch._C import _get_sdp_priority_order",
            "torch.version.hip = torch.version.hip or '7.14.60850'",
            "order = _get_sdp_priority_order()",
            "import packages.chaiNNer_pytorch.torch_runtime",
            "print(_get_sdp_priority_order() == order)",
            "print(os.environ.get('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'))",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()[-2:]


def test_rocm_keeps_the_attention_order_and_enables_aotriton_everywhere():
    environment = {
        k: v
        for k, v in os.environ.items()
        if k != "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"
    }
    assert _rocm_settings(environment) == ["True", "1"]


def test_rocm_keeps_the_users_aotriton_switch():
    environment = {**os.environ, "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "0"}
    assert _rocm_settings(environment) == ["True", "0"]
