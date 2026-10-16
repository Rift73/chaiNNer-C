"""Lazy loading of the packaged TensorRT Q/K normalization and RoPE plugin."""

from __future__ import annotations

import ctypes
import sys
from functools import lru_cache
from pathlib import Path


def plugin_path() -> Path:
    if sys.platform != "win32":
        raise RuntimeError(
            "The VOSR TensorRT rewrite plugin currently supports Windows."
        )
    path = Path(__file__).with_name("vosr2_qkv.dll")
    if not path.is_file():
        raise RuntimeError(
            f"The VOSR TensorRT plugin {path} is missing: this build was made "
            "without it. Use the PyTorch backend, or build it with native/Build.ps1 "
            "-Define CHAINNER_C_TENSORRT_ROOT=<TensorRT 11 SDK>,"
            "CHAINNER_C_CUDA_ROOT=<CUDA 13 toolkit> (native/README.md, Build)."
        )
    return path


@lru_cache(maxsize=1)
def load_plugin() -> ctypes.CDLL:
    # TensorRT must load its runtime DLL before the plugin's static registration.
    import tensorrt

    if not tensorrt.__version__.startswith("11."):
        raise RuntimeError("The VOSR Q/K plugin requires TensorRT 11.")
    return ctypes.CDLL(str(plugin_path()))
