"""The NVIDIA GPU list numbers devices as CUDA does, since the index chosen in the
GPU settings goes to CUDA (upstream chaiNNer #1899). Runs without a GPU: NVML and
the CUDA driver are stand-ins."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable

import pytest

import gpu

if TYPE_CHECKING:
    from ctypes import _Pointer

Address = tuple[int, int, int]
Devices = list[tuple[str, Address]]
InstallNvml = Callable[[Devices], None]

# NVML's order: a 3060 on bus 1, then a 4090 on bus 2.
NVML: Devices = [
    ("NVIDIA GeForce RTX 3060", (0, 1, 0)),
    ("NVIDIA GeForce RTX 4090", (0, 2, 0)),
]


@pytest.fixture
def nvml(monkeypatch: pytest.MonkeyPatch) -> InstallNvml:
    """Installs an NVML listing `devices` (name, PCI address), whose handles are
    their NVML indexes."""

    def install(devices: Devices) -> None:
        def pci(handle: int) -> SimpleNamespace:
            domain, bus, device = devices[handle][1]
            return SimpleNamespace(domain=domain, bus=bus, device=device)

        monkeypatch.setattr(gpu.nv, "nvmlInit", lambda: None)
        monkeypatch.setattr(gpu.nv, "nvmlShutdown", lambda: None)
        monkeypatch.setattr(gpu.nv, "nvmlDeviceGetCount", lambda: len(devices))
        monkeypatch.setattr(gpu.nv, "nvmlDeviceGetHandleByIndex", lambda i: i)
        monkeypatch.setattr(gpu.nv, "nvmlDeviceGetName", lambda h: devices[h][0])
        monkeypatch.setattr(gpu.nv, "nvmlDeviceGetPciInfo", pci)

    return install


def cuda_lists(monkeypatch: pytest.MonkeyPatch, addresses: list[Address]) -> None:
    monkeypatch.setattr(gpu, "cuda_pci_addresses", lambda: addresses)


def listed() -> list[tuple[int, str]]:
    # The NvInfo dies here, while NVML is still the stand-in its __del__ shuts down.
    return [(d.index, d.name) for d in gpu.get_nvidia_info().devices]


def test_fastest_first_cuda_order_numbers_the_list(
    nvml: InstallNvml, monkeypatch: pytest.MonkeyPatch
):
    nvml(NVML)
    cuda_lists(monkeypatch, [(0, 2, 0), (0, 1, 0)])
    assert listed() == [(0, "NVIDIA GeForce RTX 4090"), (1, "NVIDIA GeForce RTX 3060")]


def test_pci_order_keeps_nvml_order(nvml: InstallNvml, monkeypatch: pytest.MonkeyPatch):
    nvml(NVML)
    cuda_lists(monkeypatch, [(0, 1, 0), (0, 2, 0)])
    assert listed() == [(0, "NVIDIA GeForce RTX 3060"), (1, "NVIDIA GeForce RTX 4090")]


def test_devices_cuda_does_not_list_follow(
    nvml: InstallNvml, monkeypatch: pytest.MonkeyPatch
):
    nvml(NVML)
    cuda_lists(monkeypatch, [(0, 2, 0)])  # CUDA_VISIBLE_DEVICES shows only the 4090
    assert listed() == [(0, "NVIDIA GeForce RTX 4090"), (1, "NVIDIA GeForce RTX 3060")]


def test_without_the_cuda_driver_nvml_order_stays(
    nvml: InstallNvml, monkeypatch: pytest.MonkeyPatch
):
    nvml(NVML)
    cuda_lists(monkeypatch, [])
    assert listed() == [(0, "NVIDIA GeForce RTX 3060"), (1, "NVIDIA GeForce RTX 4090")]


def test_one_device_does_not_ask_cuda(
    nvml: InstallNvml, monkeypatch: pytest.MonkeyPatch
):
    nvml(NVML[:1])

    def unexpected() -> list[Address]:
        raise AssertionError("CUDA queried for a single device")

    monkeypatch.setattr(gpu, "cuda_pci_addresses", unexpected)
    assert listed() == [(0, "NVIDIA GeForce RTX 3060")]


def fake_driver(addresses: list[Address], fail: str = "") -> SimpleNamespace:
    """The nvcuda.dll calls cuda_pci_addresses makes, over `addresses` by CUDA
    index; `fail` names a call that returns an error."""

    def init(flags: int) -> int:
        return 100 if fail == "init" else 0  # CUDA_ERROR_NO_DEVICE

    def get_count(count: _Pointer[ctypes.c_int]) -> int:
        count.contents.value = len(addresses)
        return 0

    def get(device: _Pointer[ctypes.c_int], ordinal: int) -> int:
        device.contents.value = 1000 + ordinal  # an opaque CUdevice
        return 0

    def get_attribute(
        value: _Pointer[ctypes.c_int], attribute: int, device: ctypes.c_int
    ) -> int:
        if fail == "attribute":
            return 1
        domain, bus, slot = addresses[device.value - 1000]
        value.contents.value = {50: domain, 33: bus, 34: slot}[attribute]
        return 0

    return SimpleNamespace(
        cuInit=init,
        cuDeviceGetCount=get_count,
        cuDeviceGet=get,
        cuDeviceGetAttribute=get_attribute,
    )


@pytest.mark.parametrize(
    ("driver", "expected"),
    [
        (fake_driver([(0, 2, 0), (1, 0x41, 3)]), [(0, 2, 0), (1, 0x41, 3)]),
        (fake_driver([(0, 2, 0)], fail="init"), []),
        (fake_driver([(0, 2, 0)], fail="attribute"), []),
    ],
)
def test_cuda_pci_addresses_reads_the_driver(
    monkeypatch: pytest.MonkeyPatch, driver: SimpleNamespace, expected: list[Address]
):
    monkeypatch.setattr(ctypes, "CDLL", lambda name: driver)
    assert gpu.cuda_pci_addresses() == expected


def test_cuda_pci_addresses_without_the_driver(monkeypatch: pytest.MonkeyPatch):
    def missing(name: str):
        raise OSError(f"{name} not found")

    monkeypatch.setattr(ctypes, "CDLL", missing)
    assert gpu.cuda_pci_addresses() == []
