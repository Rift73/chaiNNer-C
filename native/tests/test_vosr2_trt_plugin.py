"""The VOSR TensorRT plugin's loader: where it looks and what a build without it says."""

from __future__ import annotations

import sys

import pytest

from nodes.impl.pytorch import vosr2_trt_plugin


def test_a_missing_plugin_names_itself_the_build_option_and_the_alternative(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        vosr2_trt_plugin, "__file__", str(tmp_path / "vosr2_trt_plugin.py")
    )
    windows = sys.platform == "win32"
    plugin = tmp_path / ("vosr2_qkv.dll" if windows else "vosr2_qkv.so")
    with pytest.raises(RuntimeError) as error:
        vosr2_trt_plugin.plugin_path()
    for part in (
        str(plugin),
        "PyTorch backend",
        "native/Build.ps1" if windows else "native/build_linux.sh",
        "CHAINNER_C_TENSORRT_ROOT=<TensorRT 11 SDK>",
        "CHAINNER_C_CUDA_ROOT=<CUDA 13 toolkit>",
    ):
        assert part in str(error.value)
    plugin.write_bytes(b"plugin fixture")
    assert vosr2_trt_plugin.plugin_path() == plugin
