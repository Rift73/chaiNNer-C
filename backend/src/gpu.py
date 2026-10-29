from __future__ import annotations

import ctypes
import importlib.util
import os
import re
import sys
import threading
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Sequence

import pynvml as nv
from sanic.log import logger

import hip_devices
from system import is_linux, is_windows

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

# The NVIDIA driver's CUDA library. pynvml finds the driver's NVML library itself.
_CUDA_DRIVER_LIBRARY = "nvcuda.dll" if is_windows else "libcuda.so.1"


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

    @cached_property
    def compute_capability(self) -> tuple[int, int] | None:
        try:
            return nv.nvmlDeviceGetCudaComputeCapability(self.handle)
        except Exception:
            return None

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

    @cached_property
    def driver_major_version(self) -> int | None:
        try:
            return int(nv.nvmlSystemGetDriverVersion().split(".")[0])
        except Exception:
            return None


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
        cuda = ctypes.CDLL(_CUDA_DRIVER_LIBRARY)
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


# Linux's sysfs; tests point it at a fake tree.
_SYSFS = Path("/sys")
# libdrm's AMD product names by PCI device and revision, which ROCm names GPUs by: the
# system's copy, then the one AMD's driver packages install.
_AMDGPU_IDS = (
    Path("/usr/share/libdrm/amdgpu.ids"),
    Path("/opt/amdgpu/share/libdrm/amdgpu.ids"),
)


@dataclass(frozen=True)
class AmdDevice:
    index: int
    """HIP's index of the device, which PyTorch (ROCm), ONNX Runtime and MIGraphX take."""
    name: str
    gfx: str
    """The LLVM processor HIP runs the device as, like "gfx1100" (with
    HSA_OVERRIDE_GFX_VERSION applied); empty when the driver does not report one."""
    sysfs: Path | None = None
    """Linux: the device's PCI folder in sysfs, which holds its memory counters."""
    luid: int | None = None
    """Windows: the adapter's LUID, which names its memory counter."""
    vram_total: int = 0
    """Windows: the device's memory, as HIP reports it."""
    compute_units: int = 0
    """The device's compute units; 0 when unknown."""
    is_apu: bool = False
    """Whether the device is a processor's graphics, sharing its memory."""

    @property
    def eligible(self) -> bool:
        """Whether the GPU is worth running models on: a discrete GPU or a large APU
        (Strix Halo), not a Ryzen's small graphics."""
        return self.compute_units >= _ELIGIBLE_COMPUTE_UNITS

    @property
    def supports_fp16(self) -> bool:
        # Vega (gfx9) and later run FP16 at twice the FP32 rate.
        version = _gfx_version(self.gfx)
        return version is not None and version[0] >= 9

    @property
    def supports_fp8(self) -> bool:
        return self.fp8_format is not None

    @property
    def fp8_format(self) -> Literal["fnuz", "ocp"] | None:
        """The FP8 formats the device's matrix math takes: the FNUZ ones on CDNA3
        (gfx942), the OCP ones on CDNA4 (gfx950), RDNA4 (gfx120x) and gfx1250. None
        elsewhere: MIGraphX computes FP8 in float there, and HIP has no FP8 type."""
        version = _gfx_version(self.gfx)
        if version == (9, 4, 2):
            return "fnuz"
        if version is not None and (
            version in ((9, 5, 0), (12, 5, 0)) or version[:2] == (12, 0)
        ):
            return "ocp"
        return None

    def get_current_vram_usage(self) -> MemoryUsage:
        """Raises OSError when the device's counters cannot be read."""
        if self.luid is not None:
            used = _dedicated_usage(self.luid)
            return MemoryUsage(self.vram_total, used, self.vram_total - used)
        if self.sysfs is None:
            raise OSError(f"{self.name} has no memory counters.")
        total = int((self.sysfs / "mem_info_vram_total").read_text())
        used = int((self.sysfs / "mem_info_vram_used").read_text())
        return MemoryUsage(total, used, total - used)


class AmdInfo:
    def __init__(self, devices: Sequence[AmdDevice]):
        self.__devices: Sequence[AmdDevice] = devices

    @staticmethod
    def unavailable():
        return AmdInfo([])

    @property
    def devices(self) -> Sequence[AmdDevice]:
        return self.__devices

    @property
    def is_available(self):
        return len(self.devices) > 0

    @property
    def all_support_fp16(self) -> bool:
        return all(gpu.supports_fp16 for gpu in self.devices)

    @property
    def eligible(self) -> bool:
        """Whether an AMD GPU is worth running models on (see AmdDevice.eligible)."""
        return any(gpu.eligible for gpu in self.devices)

    @property
    def default_index(self) -> int:
        """HIP's index of the strongest GPU: the most compute units, a discrete GPU
        before an APU, then the first. A Ryzen's graphics often come first in HIP's
        order. 0 without AMD GPUs."""
        if not self.devices:
            return 0
        return max(
            self.devices,
            key=lambda gpu: (gpu.compute_units, not gpu.is_apu, -gpu.index),
        ).index


# The fewest compute units of a GPU worth running models on (Radeon 780M graphics have
# 12, a desktop Ryzen's 2)
_ELIGIBLE_COMPUTE_UNITS = 8


def _gfx_version(gfx: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"gfx(\d+)([0-9a-f])([0-9a-f])", gfx)
    return (int(match[1]), int(match[2], 16), int(match[3], 16)) if match else None


def _gfx_name(major: int, minor: int, stepping: int) -> str:
    # LLVM's names: the major version in decimal, then a hex digit each for the others
    return f"gfx{major}{minor:x}{stepping:x}"


def _kfd_properties(path: Path) -> dict[str, int]:
    """A KFD topology node's properties file: one "name value" line per property."""
    properties: dict[str, int] = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) == 2:
            properties[fields[0]] = int(fields[1])
    return properties


def _kfd_gfx(node: int, properties: dict[str, int]) -> str:
    """The processor ROCm runs a KFD GPU node as: its gfx_target_version (major * 10000 +
    minor * 100 + stepping), or the "major.minor.stepping" of HSA_OVERRIDE_GFX_VERSION_<node>
    or HSA_OVERRIDE_GFX_VERSION, as ROCm's libhsakmt reads them. Raises ValueError for an
    override ROCm rejects, with which ROCm finds no GPU at all."""
    target = properties.get("gfx_target_version", 0)
    if not target:
        return ""
    major = target // 10000
    override = os.environ.get(f"HSA_OVERRIDE_GFX_VERSION_{node}") or os.environ.get(
        "HSA_OVERRIDE_GFX_VERSION"
    )
    if not override:
        return _gfx_name(major, target // 100 % 100, target % 100)
    match = re.fullmatch(r"\s*(\d+)\.\s*(\d+)\.\s*(\d+)", override)
    if not match or int(match[1]) != major or max(int(match[2]), int(match[3])) > 255:
        raise ValueError(
            f"ROCm rejects the GFX version override {override} for a gfx{major} GPU,"
            " and then finds no GPU."
        )
    return _gfx_name(major, int(match[2]), int(match[3]))


def _pci_folder(properties: dict[str, int]) -> Path:
    # location_id is the PCI bus number above the device/function byte (Linux's PCI_DEVID)
    location = properties.get("location_id", 0)
    domain = properties.get("domain", 0)
    address = (
        f"{domain:04x}:{location >> 8:02x}:{location >> 3 & 0x1F:02x}.{location & 7:x}"
    )
    return _SYSFS / "bus/pci/devices" / address


def _amdgpu_ids_name(device: int, revision: int) -> str | None:
    for path in _AMDGPU_IDS:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            # "744C,\tC8,\tAMD Radeon RX 7900 XTX"
            fields = [field.strip() for field in line.split(",", 2)]
            ids = [field.lower() for field in fields[:2]]
            if len(fields) == 3 and ids == [f"{device:04x}", f"{revision:02x}"]:
                return fields[2]
    return None


def _amd_name(pci: Path, gfx: str) -> str:
    """ROCm's name for a GPU: libdrm's product name for its PCI device and revision, else
    the one boards with a FRU memory store (product_name), else its processor's."""
    name = None
    try:
        name = _amdgpu_ids_name(
            int((pci / "device").read_text(), 16),
            int((pci / "revision").read_text(), 16),
        )
        if name is None and (pci / "product_name").is_file():
            name = (pci / "product_name").read_text().strip() or None
    except (OSError, ValueError, UnicodeDecodeError) as error:
        logger.warning(f"Could not read the name of the AMD GPU at {pci}: {error}")
    return name or f"AMD GPU {gfx}".rstrip()


def _tokens(value: str) -> list[str]:
    """A visible-devices list's comma-separated tokens as ROCm reads them: an empty last
    token is none."""
    tokens = value.split(",")
    if tokens[-1] == "":
        tokens.pop()
    return tokens


def _c_integer(token: str) -> int | None:
    """strtol(token, &end, 0) when it reads all of token: C's decimal, 0x hex and 0 octal."""
    match = re.fullmatch(r"([+-]?)(0[xX][0-9a-fA-F]+|0[0-7]*|[1-9][0-9]*)", token)
    if match is None:
        return None
    digits = match[2]
    if digits[:2] in ("0x", "0X"):
        value = int(digits, 16)
    else:
        value = int(digits, 8 if digits.startswith("0") else 10)
    return -value if match[1] == "-" else value


def _rocr_visible(unique_ids: list[int]) -> list[int]:
    """The positions of the GPUs ROCm's runtime shows, in its order: all, or those that
    ROCR_VISIBLE_DEVICES lists by index or by a unique prefix of their UUID
    (GPU-<unique_id in hex>), up to the first token it cannot use or that repeats one."""
    if "ROCR_VISIBLE_DEVICES" not in os.environ:
        return list(range(len(unique_ids)))
    uuids = [f"GPU-{u:016X}" if u else "Invalid-UUID" for u in unique_ids]
    chosen: list[int] = []
    for raw in _tokens(os.environ["ROCR_VISIBLE_DEVICES"])[: len(unique_ids)]:
        token = raw.strip().upper()
        if token.startswith("G"):
            matches = [i for i, uuid in enumerate(uuids) if uuid.startswith(token)]
            position = matches[0] if len(matches) == 1 and len(token) >= 5 else None
        else:
            position = _c_integer(token)
        if position is None or not 0 <= position < len(uuids) or position in chosen:
            break
        chosen.append(position)
    return chosen


def _hip_visible(uuids: list[str]) -> list[int]:
    """The positions of the GPUs HIP shows out of the runtime's (`uuids`, HIP's UUID of
    each), in HIP's order: all, or those that HIP_VISIBLE_DEVICES (CUDA_VISIBLE_DEVICES
    while it is empty) lists by index or by a part of their UUID, up to the first token
    it cannot use; a repeated one is skipped."""
    value = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get(
        "CUDA_VISIBLE_DEVICES", ""
    )
    if not value:
        return list(range(len(uuids)))
    chosen: list[int] = []
    for token in _tokens(value):
        number = token
        if "GPU-" in token:
            number = next((str(i) for i, u in enumerate(uuids) if token in u), token)
        if not re.fullmatch(r"0|[1-9][0-9]*", number) or int(number) >= len(uuids):
            break
        if int(number) not in chosen:
            chosen.append(int(number))
    return chosen


def _linux_amd_devices() -> list[AmdDevice]:
    """The AMD GPUs by HIP's index, from the KFD topology, which has a node for each CPU
    and GPU (a GPU node has SIMDs; an APU's node has both): ROCm's runtime takes the GPU
    nodes in node order, then ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES choose and
    order them. Raises OSError when the topology cannot be read."""
    nodes = _SYSFS / "class/kfd/kfd/topology/nodes"
    if not nodes.is_dir():
        return []
    gpus: list[tuple[int, dict[str, int]]] = []
    for node in sorted(
        int(path.name) for path in nodes.iterdir() if path.name.isdigit()
    ):
        properties = _kfd_properties(nodes / str(node) / "properties")
        if properties.get("simd_count", 0) > 0:
            gpus.append((node, properties))
    # ROCm reads every GPU's processor, the hidden ones' too.
    gfx = [_kfd_gfx(node, properties) for node, properties in gpus]
    unique_ids = [properties.get("unique_id", 0) for _, properties in gpus]
    runtime = _rocr_visible(unique_ids)
    hip = _hip_visible(
        [f"GPU-{unique_ids[i]:016x}" if unique_ids[i] else "GPU-XX" for i in runtime]
    )
    devices: list[AmdDevice] = []
    for index, position in enumerate(runtime[i] for i in hip):
        properties = gpus[position][1]
        pci = _pci_folder(properties)
        simds_per_unit = properties.get("simd_per_cu", 0)
        devices.append(
            AmdDevice(
                index,
                _amd_name(pci, gfx[position]),
                gfx[position],
                sysfs=pci,
                compute_units=properties["simd_count"] // simds_per_unit
                if simds_per_unit
                else 0,
                # An APU's GPU shares the CPU's node where the BIOS's topology (CRAT)
                # describes them together. Kernels that give it a node of its own
                # leave it unmarked, and its compute units alone rank it.
                is_apu=properties.get("cpu_cores_count", 0) > 0,
            )
        )
    return devices


def _rocm_installed() -> bool:
    # The ROCm SDK's core package, which PyTorch's ROCm builds load HIP from
    return importlib.util.find_spec("_rocm_sdk_core") is not None


def _windows_amd_devices() -> list[AmdDevice]:
    """The AMD GPUs by HIP's index, as a HIP runtime lists them in a child process,
    once for this process and those it starts (hip_devices.listing; the host lists them
    before it starts the worker). The AMD driver's runtime numbers them as the ROCm
    SDK's does. Raises OSError when the runtime fails."""
    return [
        AmdDevice(
            index,
            device["name"],
            device["gfx"],
            luid=device["luid"],
            vram_total=device["total"],
            compute_units=device["compute_units"],
            is_apu=device["integrated"],
        )
        for index, device in enumerate(hip_devices.listing())
    ]


class _PdhValue(ctypes.Structure):
    # PDH_FMT_COUNTERVALUE, its union read as largeValue
    _fields_ = (("status", ctypes.c_ulong), ("large", ctypes.c_longlong))


_PDH_FMT_LARGE = 0x400
# PDH_CSTATUS_VALID_DATA and PDH_CSTATUS_NEW_DATA
_PDH_GOOD_DATA = (0, 1)
# Each adapter's PDH query and counter by LUID, opened at its first read
_pdh_counters: dict[int, tuple[ctypes.c_void_p, ctypes.c_void_p]] = {}
_pdh_lock = threading.Lock()


@cache
def _pdh() -> ctypes.WinDLL:
    return ctypes.WinDLL("pdh")


def _pdh_check(status: int, call: str) -> None:
    if status != 0:
        raise OSError(f"{call} failed with PDH status 0x{status & 0xFFFFFFFF:08X}.")


def _dedicated_usage(luid: int) -> int:
    """The memory in use on the adapter with this LUID, by all processes: Windows' GPU
    Adapter Memory counter, which Task Manager shows. Raises OSError on failure."""
    pdh = _pdh()
    with _pdh_lock:
        if luid not in _pdh_counters:
            query, counter = ctypes.c_void_p(), ctypes.c_void_p()
            # The counter's instance: luid_0x<high part>_0x<low part>_phys_0
            path = (
                f"\\GPU Adapter Memory(luid_0x{luid >> 32:08X}_0x{luid & 0xFFFFFFFF:08X}"
                "_phys_0)\\Dedicated Usage"
            )
            _pdh_check(
                pdh.PdhOpenQueryW(None, None, ctypes.byref(query)), "PdhOpenQuery"
            )
            status = pdh.PdhAddEnglishCounterW(query, path, None, ctypes.byref(counter))
            if status != 0:
                pdh.PdhCloseQuery(query)
            _pdh_check(status, f"PdhAddEnglishCounter for {path}")
            _pdh_counters[luid] = (query, counter)
        query, counter = _pdh_counters[luid]
        _pdh_check(pdh.PdhCollectQueryData(query), "PdhCollectQueryData")
        value = _PdhValue()
        _pdh_check(
            pdh.PdhGetFormattedCounterValue(
                counter, _PDH_FMT_LARGE, None, ctypes.byref(value)
            ),
            "PdhGetFormattedCounterValue",
        )
        if value.status not in _PDH_GOOD_DATA:
            _pdh_check(value.status, "The GPU Adapter Memory counter")
        return value.large


class _Guid(ctypes.Structure):
    _fields_ = (
        ("data1", ctypes.c_uint32),
        ("data2", ctypes.c_uint16),
        ("data3", ctypes.c_uint16),
        ("data4", ctypes.c_ubyte * 8),
    )


# IID_IDXGIFactory1, {770aae78-f26f-4dba-a829-253c83d1b387}
_IID_DXGI_FACTORY1 = _Guid(
    0x770AAE78,
    0xF26F,
    0x4DBA,
    (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
)
# Vtable slots: IUnknown::Release, IDXGIFactory1::EnumAdapters1, IDXGIAdapter1::GetDesc1
_COM_RELEASE, _DXGI_ENUM_ADAPTERS1, _DXGI_GET_DESC1 = 2, 12, 10
_DXGI_ERROR_NOT_FOUND = 0x887A0002
# DXGI_ADAPTER_DESC1: 312 bytes, the adapter's LUID in bytes 296 to 304
_DXGI_ADAPTER_DESC1_SIZE, _DXGI_ADAPTER_LUID = 312, slice(296, 304)


def _com_call(
    instance: ctypes.c_void_p, slot: int, argtypes: tuple[type, ...], *args: object
) -> int:
    """Method `slot` of the COM object `instance`, called with `args`; its HRESULT (or
    reference count) as an unsigned 32-bit value."""
    vtable = ctypes.cast(instance, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))
    method = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p, *argtypes)(
        vtable.contents[slot]
    )
    return method(instance, *args)


def _hresult_check(status: int, call: str) -> None:
    if status & 0x80000000:
        raise OSError(f"{call} failed with HRESULT 0x{status:08X}.")


def dxgi_adapter_index(luid: int) -> int:
    """Windows: the position of the adapter with this LUID in DXGI's adapter order, by
    which DirectML numbers its devices (ONNX Runtime's device_id). Raises OSError when
    DXGI fails or has no such adapter."""
    dxgi = ctypes.WinDLL("dxgi")
    dxgi.CreateDXGIFactory1.restype = ctypes.c_ulong
    dxgi.CreateDXGIFactory1.argtypes = (
        ctypes.POINTER(_Guid),
        ctypes.POINTER(ctypes.c_void_p),
    )
    factory = ctypes.c_void_p()
    _hresult_check(
        dxgi.CreateDXGIFactory1(
            ctypes.byref(_IID_DXGI_FACTORY1), ctypes.byref(factory)
        ),
        "CreateDXGIFactory1",
    )
    try:
        index = 0
        while True:
            adapter = ctypes.c_void_p()
            status = _com_call(
                factory,
                _DXGI_ENUM_ADAPTERS1,
                (ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)),
                index,
                ctypes.byref(adapter),
            )
            if status == _DXGI_ERROR_NOT_FOUND:
                raise OSError(
                    f"DXGI lists no adapter with the LUID 0x{luid:016X} ({index}"
                    " adapters)."
                )
            _hresult_check(status, "IDXGIFactory1::EnumAdapters1")
            description = (ctypes.c_ubyte * _DXGI_ADAPTER_DESC1_SIZE)()
            try:
                _hresult_check(
                    _com_call(
                        adapter, _DXGI_GET_DESC1, (ctypes.c_void_p,), description
                    ),
                    "IDXGIAdapter1::GetDesc1",
                )
            finally:
                _com_call(adapter, _COM_RELEASE, ())
            if int.from_bytes(bytes(description[_DXGI_ADAPTER_LUID]), "little") == luid:
                return index
            index += 1
    finally:
        _com_call(factory, _COM_RELEASE, ())


def get_amd_info() -> AmdInfo:
    """The AMD GPUs as HIP will number them, found without ROCm installed, so the
    dependency manager can choose ROCm builds by them."""
    if not (is_linux or is_windows):
        return AmdInfo.unavailable()
    try:
        if is_windows and nvidia.is_available and not _rocm_installed():
            # The AMD driver's HIP runtime stays out of an NVIDIA machine's start (a
            # laptop's Radeon graphics beside its GeForce) until ROCm is installed.
            logger.info("AMD GPUs not looked for: an NVIDIA GPU and no ROCm installed.")
            return AmdInfo.unavailable()
        devices = _linux_amd_devices() if is_linux else _windows_amd_devices()
    except Exception as e:
        logger.warning(f"Could not list the AMD GPUs: {e}", exc_info=True)
        return AmdInfo.unavailable()
    if not devices:
        logger.info("No AMD GPU found.")
    return AmdInfo(devices)


amd = get_amd_info()


def get_vram_usage(index: int) -> MemoryUsage | None:
    """The memory usage of the GPU PyTorch numbers `index`: an AMD GPU (HIP's index) when
    the loaded PyTorch is a ROCm build, else an NVIDIA GPU (CUDA's index). None when that
    vendor has no GPU with the index; raises when the GPU's counters cannot be read."""
    torch = sys.modules.get("torch")
    rocm = torch is not None and getattr(getattr(torch, "version", None), "hip", None)
    devices: Sequence[NvDevice | AmdDevice] = amd.devices if rocm else nvidia.devices
    device = next((d for d in devices if d.index == index), None)
    return None if device is None else device.get_current_vram_usage()


__all__ = [
    "AmdDevice",
    "AmdInfo",
    "MemoryUsage",
    "NvDevice",
    "NvInfo",
    "amd",
    "dxgi_adapter_index",
    "get_vram_usage",
    "nvidia",
]
