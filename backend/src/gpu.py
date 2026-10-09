from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Callable, Sequence

import pynvml as nv
from sanic.log import logger

if TYPE_CHECKING:
    from ctypes import _Pointer

_FP16_ARCH_ABILITY_MAP = {
    nv.NVML_DEVICE_ARCH_KEPLER: False,
    nv.NVML_DEVICE_ARCH_MAXWELL: False,
    nv.NVML_DEVICE_ARCH_PASCAL: False,
    nv.NVML_DEVICE_ARCH_VOLTA: True,
    nv.NVML_DEVICE_ARCH_TURING: True,
    nv.NVML_DEVICE_ARCH_AMPERE: True,
    nv.NVML_DEVICE_ARCH_ADA: True,
    nv.NVML_DEVICE_ARCH_HOPPER: True,
    nv.NVML_DEVICE_ARCH_UNKNOWN: False,
}

# CUdevice_attribute values (cuda.h) CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, _PCI_BUS_ID and
# _PCI_DEVICE_ID: the device's PCI domain, bus and device (slot), as NVML's PCI info.
_CU_PCI_ADDRESS_ATTRIBUTES = (50, 33, 34)


@dataclass
class MemoryUsage:
    total: int
    used: int
    free: int


@dataclass(frozen=True)
class NvDevice:
    index: int
    """CUDA's index of the device, which PyTorch, ONNX Runtime and TensorRT take."""
    handle: _Pointer[nv.struct_c_nvmlDevice_t]  # nv.c_nvmlDevice_t's type
    name: str

    @staticmethod
    def from_handle(index: int, handle: _Pointer[nv.struct_c_nvmlDevice_t]) -> NvDevice:
        return NvDevice(
            index=index,
            handle=handle,
            name=nv.nvmlDeviceGetName(handle),
        )

    @cached_property
    def architecture(self) -> int:
        # We catch and ignore errors to support older drivers that don't have nvmlDeviceGetArchitecture
        try:
            return nv.nvmlDeviceGetArchitecture(self.handle)
        except Exception:
            return nv.NVML_DEVICE_ARCH_UNKNOWN

    @property
    def supports_fp16(self):
        arch = self.architecture

        # This generation also contains the GTX 1600 cards, which do not support FP16.
        if arch == nv.NVML_DEVICE_ARCH_TURING:
            return "RTX" in self.name

        # Future proofing. We can be reasonably sure that future architectures will support FP16.
        return _FP16_ARCH_ABILITY_MAP.get(arch, arch > nv.NVML_DEVICE_ARCH_HOPPER)

    def get_current_vram_usage(self) -> MemoryUsage:
        info = nv.nvmlDeviceGetMemoryInfo(self.handle)
        return MemoryUsage(info.total, info.used, info.free)  # type: ignore


class NvInfo:
    def __init__(self, devices: Sequence[NvDevice], clean_up: Callable[[], None]):
        self.__devices: Sequence[NvDevice] = devices
        self.__clean_up = clean_up

    @staticmethod
    def unavailable():
        return NvInfo([], lambda: None)

    def __del__(self):
        self.__clean_up()

    @property
    def devices(self) -> Sequence[NvDevice]:
        return self.__devices

    @property
    def is_available(self):
        return len(self.devices) > 0

    @property
    def all_support_fp16(self) -> bool:
        return all(gpu.supports_fp16 for gpu in self.devices)

    @property
    def any_needs_legacy_cuda(self) -> bool:
        """
        Check if any device needs legacy CUDA version (12.6 instead of 13.2).
        CUDA 13 dropped Maxwell, Pascal and Volta; those get cu126.
        """
        for gpu in self.devices:
            arch = gpu.architecture
            # Volta and older architectures need CUDA 12.6
            if arch in (
                nv.NVML_DEVICE_ARCH_VOLTA,
                nv.NVML_DEVICE_ARCH_PASCAL,
                nv.NVML_DEVICE_ARCH_MAXWELL,
                nv.NVML_DEVICE_ARCH_KEPLER,
            ):
                return True
        return False


def _try_nvml_init():
    try:
        nv.nvmlInit()
        return True
    except Exception as e:
        if isinstance(e, nv.NVMLError):
            logger.info("No Nvidia GPU found, or invalid driver installed.")
        else:
            logger.info(
                f"Unknown error occurred when trying to initialize Nvidia GPU: {e}"
            )
        return False


def _try_nvml_shutdown():
    try:
        nv.nvmlShutdown()
    except Exception:
        logger.warn("Failed to shut down Nvidia GPU.", exc_info=True)


def cuda_pci_addresses() -> list[tuple[int, int, int]]:
    """The PCI address (domain, bus, device) of each CUDA device, by CUDA index.
    Empty when the CUDA driver is missing or fails."""
    try:
        cuda = ctypes.CDLL("nvcuda.dll")
    except OSError:
        return []
    count = ctypes.c_int()
    if cuda.cuInit(0) != 0 or cuda.cuDeviceGetCount(ctypes.pointer(count)) != 0:
        return []
    addresses: list[tuple[int, int, int]] = []
    for index in range(count.value):
        device = ctypes.c_int()
        if cuda.cuDeviceGet(ctypes.pointer(device), index) != 0:
            return []
        address: list[int] = []
        for attribute in _CU_PCI_ADDRESS_ATTRIBUTES:
            value = ctypes.c_int()
            if cuda.cuDeviceGetAttribute(ctypes.pointer(value), attribute, device) != 0:
                return []
            address.append(value.value)
        domain, bus, slot = address
        addresses.append((domain, bus, slot))
    return addresses


def _in_cuda_order(
    handles: list[_Pointer[nv.struct_c_nvmlDevice_t]],
) -> list[_Pointer[nv.struct_c_nvmlDevice_t]]:
    """NVML's devices in CUDA's order. CUDA numbers devices by CUDA_DEVICE_ORDER
    (fastest first by default) and CUDA_VISIBLE_DEVICES, not as NVML lists them.
    Devices CUDA does not list follow in NVML's order."""
    cuda = cuda_pci_addresses()

    def position(nvml_index: int) -> int:
        pci = nv.nvmlDeviceGetPciInfo(handles[nvml_index])
        address = (int(pci.domain), int(pci.bus), int(pci.device))
        return cuda.index(address) if address in cuda else len(cuda) + nvml_index

    return [handles[i] for i in sorted(range(len(handles)), key=position)]


def get_nvidia_info() -> NvInfo:
    if not _try_nvml_init():
        return NvInfo.unavailable()

    try:
        device_count = nv.nvmlDeviceGetCount()
        handles = [nv.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
        if len(handles) > 1:
            # The GPU settings list these devices, and the index chosen there goes
            # to CUDA, so the list takes CUDA's numbering.
            handles = _in_cuda_order(handles)
        devices = [NvDevice.from_handle(i, h) for i, h in enumerate(handles)]
        return NvInfo(devices, _try_nvml_shutdown)
    except Exception as e:
        logger.info(f"Unknown error occurred when trying to initialize Nvidia GPU: {e}")
        _try_nvml_shutdown()
        return NvInfo.unavailable()


nvidia = get_nvidia_info()


__all__ = ["MemoryUsage", "NvDevice", "NvInfo", "nvidia"]
