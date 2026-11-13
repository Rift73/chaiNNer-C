"""Triton for Windows on NVIDIA GPUs, which two packages list: the PyTorch package for
spandrel's eager kernels (DUAL's and RDUAL's attention, the window attention and the
normalization kernels) and the TensorRT package for its engine builds' plugins. One pin
for both, so neither ever asks for another version than the other installed."""

from __future__ import annotations

from api import MB, Dependency


def triton_windows_nvidia() -> Dependency:
    """3.9.0.post30, the version the TensorRT package's engine builds compile their AOT
    plugins with on Python 3.14. Triton for Windows on AMD GPUs is rocm.triton_windows."""
    return Dependency(
        display_name="Triton",
        pypi_name="triton-windows",
        version="3.9.0.post30",
        size_estimate=45 * MB,
        import_name="triton",
    )
