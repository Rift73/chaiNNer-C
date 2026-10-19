"""The TensorRT package is disabled on GPUs and drivers CUDA 13 does not support
(upstream chaiNNer #3347). Runs without a GPU: NVML is a stand-in."""

from __future__ import annotations

import pytest

import gpu
from packages.chaiNNer_tensorrt import _supports_cuda_13


def _device(
    monkeypatch: pytest.MonkeyPatch, cc: tuple[int, int] | None
) -> gpu.NvDevice:
    def compute_capability(_handle: object) -> tuple[int, int]:
        if cc is None:
            raise gpu.nv.NVMLError(gpu.nv.NVML_ERROR_NOT_SUPPORTED)
        return cc

    monkeypatch.setattr(
        gpu.nv, "nvmlDeviceGetCudaComputeCapability", compute_capability
    )
    return gpu.NvDevice(index=0, handle=gpu.nv.c_nvmlDevice_t(), name="Test GPU")


@pytest.mark.parametrize(
    ("cc", "supported"),
    [((6, 1), False), ((7, 0), False), ((7, 5), True), ((12, 0), True), (None, True)],
)
def test_compute_capability_decides_support(
    monkeypatch: pytest.MonkeyPatch, cc: tuple[int, int] | None, supported: bool
):
    device = _device(monkeypatch, cc)

    assert device.compute_capability == cc
    assert _supports_cuda_13(device) is supported


@pytest.mark.parametrize(
    ("version", "major"), [("610.47", 610), ("576.88", 576), (None, None)]
)
def test_driver_major_version(
    monkeypatch: pytest.MonkeyPatch, version: str | None, major: int | None
):
    def driver_version() -> str:
        if version is None:
            raise gpu.nv.NVMLError(gpu.nv.NVML_ERROR_UNINITIALIZED)
        return version

    monkeypatch.setattr(gpu.nv, "nvmlSystemGetDriverVersion", driver_version)

    assert gpu.NvInfo([], lambda: None).driver_major_version == major
