"""The AMD GPU list numbers devices as HIP does and is found without ROCm (Linux: the
KFD topology and the GPUs' PCI folders in sysfs; Windows: a HIP runtime in a child
process, and Windows' GPU memory counters). Runs without an AMD GPU: sysfs is a fake
tree, HIP and the child process are stand-ins."""

from __future__ import annotations

import asyncio
import ctypes
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

import gpu
import hip_devices
from gpu import _dedicated_usage, _windows_amd_devices
from gpu import _pci_folder as pci_folder

if TYPE_CHECKING:
    from ctypes import _Pointer

# The variables that choose and number HIP's GPUs
VARIABLES = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "HSA_OVERRIDE_GFX_VERSION_1",
    "HSA_OVERRIDE_GFX_VERSION_2",
)

IDS = """\
# List of AMDGPU IDs
#
# Syntax:
# device_id,\trevision_id,\tproduct_name        <-- single tab after comma

1.0.0
164E,\tC1,\tAMD Radeon(TM) Graphics
744C,\tC8,\tAMD Radeon RX 7900 XTX
7480,\tC0,\tAMD Radeon RX 7600
"""


def file_name(address: str) -> str:
    # Windows file names cannot hold a PCI address's colons.
    return address.replace(":", "_") if sys.platform == "win32" else address


class Sysfs:
    """A fake /sys: KFD topology nodes and the PCI folders of their GPUs."""

    def __init__(self, root: Path):
        self.root = root
        (root / "class/kfd/kfd/topology/nodes").mkdir(parents=True)

    def cpu(self, node: int) -> None:
        self.node(node, cpu_cores_count=16, simd_count=0)

    def gpu(
        self,
        node: int,
        gfx: int,
        bus: int,
        unique_id: int,
        device: int = 0x744C,
        revision: int = 0xC8,
        vram: tuple[int, int] = (24 << 30, 1 << 30),
        cpu_cores: int = 0,
        compute_units: int = 96,
        simds_per_unit: int = 2,
    ) -> Path:
        """A GPU node on PCI bus `bus` (device 0, function 0); its PCI folder."""
        self.node(
            node,
            cpu_cores_count=cpu_cores,
            simd_count=compute_units * simds_per_unit,
            simd_per_cu=simds_per_unit,
            gfx_target_version=gfx,
            vendor_id=0x1002,
            device_id=device,
            location_id=bus << 8,
            domain=0,
            unique_id=unique_id,
        )
        pci = self.root / "bus/pci/devices" / file_name(f"0000:{bus:02x}:00.0")
        pci.mkdir(parents=True)
        (pci / "device").write_text(f"0x{device:04x}\n")
        (pci / "revision").write_text(f"0x{revision:02x}\n")
        (pci / "mem_info_vram_total").write_text(f"{vram[0]}\n")
        (pci / "mem_info_vram_used").write_text(f"{vram[1]}\n")
        return pci

    def node(self, node: int, **properties: int) -> None:
        folder = self.root / f"class/kfd/kfd/topology/nodes/{node}"
        folder.mkdir()
        lines = [f"{name} {value}\n" for name, value in properties.items()]
        (folder / "properties").write_text("".join(lines))


@pytest.fixture
def sysfs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Sysfs:
    ids = tmp_path / "amdgpu.ids"
    ids.write_text(IDS)
    monkeypatch.setattr(gpu, "_SYSFS", tmp_path / "sys")
    monkeypatch.setattr(gpu, "_AMDGPU_IDS", (tmp_path / "missing.ids", ids))
    monkeypatch.setattr(gpu, "is_linux", True)
    monkeypatch.setattr(gpu, "is_windows", False)

    def fake_pci_folder(properties: dict[str, int]) -> Path:
        folder = pci_folder(properties)
        return folder.with_name(file_name(folder.name))

    monkeypatch.setattr(gpu, "_pci_folder", fake_pci_folder)
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)
    return Sysfs(tmp_path / "sys")


def listed() -> list[tuple[int, str, str]]:
    return [(d.index, d.name, d.gfx) for d in gpu.get_amd_info().devices]


# Two discrete GPUs: a 7900 XTX on bus 3 (node 1), a 7600 on bus 8 (node 2)
XTX = (0, "AMD Radeon RX 7900 XTX", "gfx1100")
RX7600 = (0, "AMD Radeon RX 7600", "gfx1102")


@pytest.fixture
def two(sysfs: Sysfs) -> Sysfs:
    sysfs.cpu(0)
    sysfs.gpu(1, 110000, bus=3, unique_id=0xAAAA000000000001)
    sysfs.gpu(
        2, 110002, bus=8, unique_id=0xBBBB000000000002, device=0x7480, revision=0xC0
    )
    return sysfs


def numbered(*devices: tuple[int, str, str]) -> list[tuple[int, str, str]]:
    return [(i, name, gfx) for i, (_, name, gfx) in enumerate(devices)]


def test_no_kfd_topology(sysfs: Sysfs):
    (sysfs.root / "class/kfd/kfd/topology/nodes").rmdir()
    info = gpu.get_amd_info()
    assert not info.is_available
    assert info.devices == []


def test_cpu_nodes_only(sysfs: Sysfs):
    sysfs.cpu(0)
    sysfs.cpu(1)
    assert listed() == []


def test_one_gpu(sysfs: Sysfs):
    sysfs.cpu(0)
    pci = sysfs.gpu(1, 110000, bus=3, unique_id=1, vram=(24 << 30, 3 << 30))
    [device] = gpu.get_amd_info().devices
    assert (device.index, device.name, device.gfx) == XTX
    assert device.sysfs == pci
    assert device.supports_fp16
    assert not device.supports_fp8
    usage = device.get_current_vram_usage()
    assert (usage.total, usage.used, usage.free) == (24 << 30, 3 << 30, 21 << 30)


def test_two_gpus_in_node_order(two: Sysfs):
    assert listed() == numbered(XTX, RX7600)


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        ({"HIP_VISIBLE_DEVICES": "1,0"}, [RX7600, XTX]),
        ({"HIP_VISIBLE_DEVICES": "1"}, [RX7600]),
        ({"CUDA_VISIBLE_DEVICES": "1"}, [RX7600]),
        # HIP_VISIBLE_DEVICES wins; an empty one leaves CUDA_VISIBLE_DEVICES
        ({"HIP_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "1"}, [XTX]),
        ({"HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "1"}, [RX7600]),
        # The test suite's own CUDA_VISIBLE_DEVICES=-1 hides every GPU from HIP too.
        ({"CUDA_VISIBLE_DEVICES": "-1"}, []),
        # HIP: the first token it cannot use ends the list; a repeat is skipped
        ({"HIP_VISIBLE_DEVICES": "1,5,0"}, [RX7600]),
        ({"HIP_VISIBLE_DEVICES": "1,1,0"}, [RX7600, XTX]),
        ({"HIP_VISIBLE_DEVICES": "01"}, []),
        ({"HIP_VISIBLE_DEVICES": "0,"}, [XTX]),
        ({"HIP_VISIBLE_DEVICES": "GPU-bbbb"}, [RX7600]),
        # ROCm's runtime: indices into all GPUs, then HIP's into the runtime's
        ({"ROCR_VISIBLE_DEVICES": "1"}, [RX7600]),
        ({"ROCR_VISIBLE_DEVICES": "1,0", "HIP_VISIBLE_DEVICES": "1"}, [XTX]),
        ({"ROCR_VISIBLE_DEVICES": ""}, []),
        ({"ROCR_VISIBLE_DEVICES": " 0x1 "}, [RX7600]),
        ({"ROCR_VISIBLE_DEVICES": "GPU-BBBB,gpu-aaaa"}, [RX7600, XTX]),
        # ...where a repeat, an ambiguous or unknown UUID or a stray character ends it
        ({"ROCR_VISIBLE_DEVICES": "1,1"}, [RX7600]),
        ({"ROCR_VISIBLE_DEVICES": "GPU-,0"}, []),
        ({"ROCR_VISIBLE_DEVICES": "GPU-C,0"}, []),
        ({"ROCR_VISIBLE_DEVICES": "1x,0"}, []),
    ],
)
def test_visible_devices(
    two: Sysfs,
    monkeypatch: pytest.MonkeyPatch,
    variables: dict[str, str],
    expected: list[tuple[int, str, str]],
):
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    assert listed() == numbered(*expected)


def test_apu_and_discrete_gpu(sysfs: Sysfs):
    # A Ryzen's graphics share node 0 with its CPU cores; HIP numbers them first.
    sysfs.gpu(
        0,
        100306,
        bus=0x7C,
        unique_id=0,
        device=0x164E,
        revision=0xC1,
        cpu_cores=16,
        compute_units=2,
    )
    sysfs.gpu(1, 110000, bus=3, unique_id=1)
    info = gpu.get_amd_info()
    assert [
        (d.index, d.name, d.gfx, d.compute_units, d.is_apu) for d in info.devices
    ] == [
        (0, "AMD Radeon(TM) Graphics", "gfx1036", 2, True),
        (1, "AMD Radeon RX 7900 XTX", "gfx1100", 96, False),
    ]
    # The discrete GPU decides, and is PyTorch's default.
    assert info.eligible
    assert info.default_index == 1


def test_compute_units(sysfs: Sysfs):
    # CDNA has four SIMDs per compute unit, RDNA two; unknown without simd_per_cu.
    sysfs.gpu(1, 90402, bus=3, unique_id=1, compute_units=304, simds_per_unit=4)
    sysfs.node(2, simd_count=64, gfx_target_version=110000, unique_id=2)
    assert [d.compute_units for d in gpu.get_amd_info().devices] == [304, 0]


def amd_gpu(index: int, compute_units: int, is_apu: bool = False) -> gpu.AmdDevice:
    return gpu.AmdDevice(
        index, "AMD GPU", "gfx1100", compute_units=compute_units, is_apu=is_apu
    )


@pytest.mark.parametrize(
    ("devices", "eligible", "default_index"),
    [
        ([], False, 0),
        # A desktop Ryzen's graphics alone
        ([amd_gpu(0, 2, is_apu=True)], False, 0),
        ([amd_gpu(0, 2, is_apu=True), amd_gpu(1, 96)], True, 1),
        # Radeon 780M and Strix Halo's 8060S graphics are worth using.
        ([amd_gpu(0, 12, is_apu=True)], True, 0),
        ([amd_gpu(0, 40, is_apu=True), amd_gpu(1, 32)], True, 0),
        # Ties: a discrete GPU before an APU, then the first
        ([amd_gpu(0, 12, is_apu=True), amd_gpu(1, 12)], True, 1),
        ([amd_gpu(0, 60), amd_gpu(1, 60)], True, 0),
        ([amd_gpu(0, 0), amd_gpu(1, 0)], False, 0),
    ],
)
def test_eligible_and_default_index(
    devices: list[gpu.AmdDevice], eligible: bool, default_index: int
):
    info = gpu.AmdInfo(devices)
    assert (info.eligible, info.default_index) == (eligible, default_index)


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        ({"HSA_OVERRIDE_GFX_VERSION": "11.0.0"}, ["gfx1100", "gfx1100"]),
        ({"HSA_OVERRIDE_GFX_VERSION_2": "11.0.1"}, ["gfx1100", "gfx1101"]),
        # The node's own wins
        (
            {
                "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
                "HSA_OVERRIDE_GFX_VERSION_1": "11.0.3",
            },
            ["gfx1103", "gfx1100"],
        ),
        # ROCm rejects a malformed or cross-generation override and finds no GPU.
        ({"HSA_OVERRIDE_GFX_VERSION": "11.0"}, []),
        ({"HSA_OVERRIDE_GFX_VERSION": "10.3.0"}, []),
    ],
)
def test_gfx_override(
    two: Sysfs,
    monkeypatch: pytest.MonkeyPatch,
    variables: dict[str, str],
    expected: list[str],
):
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    assert [d.gfx for d in gpu.get_amd_info().devices] == expected


@pytest.mark.parametrize(
    ("target", "gfx", "fp16", "fp8"),
    [
        (80003, "gfx803", False, None),
        (90000, "gfx900", True, None),
        (90010, "gfx90a", True, None),
        (90012, "gfx90c", True, None),
        # CDNA3 before gfx942 is not a ROCm target.
        (90400, "gfx940", True, None),
        (90402, "gfx942", True, "fnuz"),
        (90500, "gfx950", True, "ocp"),
        (100300, "gfx1030", True, None),
        (110000, "gfx1100", True, None),
        (110501, "gfx1151", True, None),
        (120000, "gfx1200", True, "ocp"),
        (120001, "gfx1201", True, "ocp"),
        (120500, "gfx1250", True, "ocp"),
    ],
)
def test_processor(sysfs: Sysfs, target: int, gfx: str, fp16: bool, fp8: str | None):
    sysfs.gpu(1, target, bus=3, unique_id=1, device=0x1234)
    [device] = gpu.get_amd_info().devices
    assert (device.gfx, device.supports_fp16, device.fp8_format) == (gfx, fp16, fp8)
    assert device.supports_fp8 == (fp8 is not None)
    # Not in libdrm's table and no product_name: named by its processor
    assert device.name == f"AMD GPU {gfx}"


def test_product_name_without_libdrm_entry(sysfs: Sysfs):
    pci = sysfs.gpu(1, 90402, bus=3, unique_id=1, device=0x74A1, revision=0)
    (pci / "product_name").write_text("AMD Instinct MI300X\n")
    assert listed() == [(0, "AMD Instinct MI300X", "gfx942")]


def test_unreadable_topology_reports_no_gpu(sysfs: Sysfs):
    sysfs.gpu(1, 110000, bus=3, unique_id=1)
    # A folder where the node's properties file belongs cannot be read as one.
    (sysfs.root / "class/kfd/kfd/topology/nodes/2/properties").mkdir(parents=True)
    assert not gpu.get_amd_info().is_available


def test_unreadable_name_files_keep_the_gpu(sysfs: Sysfs):
    pci = sysfs.gpu(1, 110000, bus=3, unique_id=1)
    (pci / "revision").unlink()
    (pci / "revision").mkdir()
    assert listed() == [(0, "AMD GPU gfx1100", "gfx1100")]


def test_unreadable_memory_counters_raise(sysfs: Sysfs):
    pci = sysfs.gpu(1, 110000, bus=3, unique_id=1)
    (pci / "mem_info_vram_used").unlink()
    [device] = gpu.get_amd_info().devices
    with pytest.raises(OSError):
        device.get_current_vram_usage()


def test_device_without_counters():
    device = gpu.AmdDevice(index=0, name="AMD Radeon RX 7900 XTX", gfx="gfx1100")
    with pytest.raises(OSError):
        device.get_current_vram_usage()


def test_vram_usage_follows_the_vendor_pytorch_runs_on(
    two: Sysfs, monkeypatch: pytest.MonkeyPatch
):
    # A machine with both vendors: NVIDIA's list from NVML, AMD's from sysfs
    nvidia_device = SimpleNamespace(
        index=0,
        get_current_vram_usage=lambda: gpu.MemoryUsage(32 << 30, 2 << 30, 30 << 30),
    )
    monkeypatch.setattr(gpu, "nvidia", SimpleNamespace(devices=[nvidia_device]))
    monkeypatch.setattr(gpu, "amd", gpu.get_amd_info())

    def torch(hip: str | None) -> SimpleNamespace:
        return SimpleNamespace(version=SimpleNamespace(hip=hip, cuda=None))

    monkeypatch.setitem(sys.modules, "torch", torch("7.14.1"))
    assert gpu.get_vram_usage(0) == gpu.MemoryUsage(24 << 30, 1 << 30, 23 << 30)
    assert gpu.get_vram_usage(1) == gpu.MemoryUsage(24 << 30, 1 << 30, 23 << 30)
    assert gpu.get_vram_usage(2) is None

    monkeypatch.setitem(sys.modules, "torch", torch(None))
    assert gpu.get_vram_usage(0) == gpu.MemoryUsage(32 << 30, 2 << 30, 30 << 30)
    assert gpu.get_vram_usage(1) is None

    # Without PyTorch loaded: NVIDIA's numbering, as CUDA builds have
    monkeypatch.delitem(sys.modules, "torch")
    assert gpu.get_vram_usage(0) == gpu.MemoryUsage(32 << 30, 2 << 30, 30 << 30)


# Windows: the child process's HIP runtime and its listing

# hipDeviceProp_tR0600 as HIP 6.0 to 7.x compile it for 64-bit Windows (offsetof)
NAME, LUID, TOTAL, GCN_ARCH_NAME = 0, 272, 288, 1160
PROCESSORS, INTEGRATED = 388, 396  # multiProcessorCount, integrated


def fake_hip(
    devices: list[tuple[str, str, int, int, int, bool]], count_status: int = 0
):
    """A HIP runtime listing `devices` (name, gcnArchName, LUID, memory,
    multiProcessorCount, integrated)."""

    def get_count(count: _Pointer[ctypes.c_int]) -> int:
        count.contents.value = len(devices)
        return count_status

    def get_properties(buffer: ctypes.Array[ctypes.c_char], index: int) -> int:
        name, arch, luid, total, processors, integrated = devices[index]
        buffer[NAME : NAME + len(name)] = name.encode()
        buffer[LUID : LUID + 8] = luid.to_bytes(8, "little")
        buffer[TOTAL : TOTAL + 8] = total.to_bytes(8, "little")
        buffer[PROCESSORS : PROCESSORS + 4] = processors.to_bytes(4, "little")
        buffer[INTEGRATED : INTEGRATED + 4] = int(integrated).to_bytes(4, "little")
        start = GCN_ARCH_NAME
        buffer[start : start + len(arch)] = arch.encode()
        return 0

    return SimpleNamespace(
        hipGetDeviceCount=get_count, hipGetDevicePropertiesR0600=get_properties
    )


HIP_GPUS = [
    # HIP counts RDNA's workgroup processors: 48 for 96 compute units.
    ("AMD Radeon RX 7900 XTX", "gfx1100", 0x42EBC, 24 << 30, 48, False),
    (
        "AMD Radeon(TM) 8060S Graphics",
        "gfx1151:xnack-",
        0x1_0000_0002,
        96 << 30,
        20,
        True,
    ),
    ("AMD Instinct MI210", "gfx90a:sramecc+:xnack-", 0x3, 64 << 30, 104, False),
]


def test_hip_devices_reads_the_properties(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GPU_ENABLE_WGP_MODE", raising=False)
    monkeypatch.setattr(ctypes, "CDLL", lambda name: fake_hip(HIP_GPUS))
    assert hip_devices.list_devices("amdhip64_7.dll") == [
        {"name": "AMD Radeon RX 7900 XTX", "gfx": "gfx1100", "luid": 0x42EBC}
        | {"total": 24 << 30, "compute_units": 96, "integrated": False},
        {"name": "AMD Radeon(TM) 8060S Graphics", "gfx": "gfx1151"}
        | {"luid": 0x1_0000_0002, "total": 96 << 30}
        | {"compute_units": 40, "integrated": True},
        {"name": "AMD Instinct MI210", "gfx": "gfx90a", "luid": 0x3}
        | {"total": 64 << 30, "compute_units": 104, "integrated": False},
    ]


@pytest.mark.parametrize(
    ("gfx", "processors", "wgp_mode", "compute_units"),
    [
        ("gfx1100", 48, None, 96),
        ("gfx1201", 32, "1", 64),
        ("gfx1100", 96, "0", 96),
        ("gfx1100", 96, "false", 96),
        # GCN and CDNA have no workgroup processors; gfx1250 runs without WGP mode.
        ("gfx906", 60, None, 60),
        ("gfx1250", 64, None, 64),
    ],
)
def test_hip_compute_units(
    monkeypatch: pytest.MonkeyPatch,
    gfx: str,
    processors: int,
    wgp_mode: str | None,
    compute_units: int,
):
    if wgp_mode is None:
        monkeypatch.delenv("GPU_ENABLE_WGP_MODE", raising=False)
    else:
        monkeypatch.setenv("GPU_ENABLE_WGP_MODE", wgp_mode)
    hip = fake_hip([("AMD GPU", gfx, 1, 1 << 30, processors, False)])
    monkeypatch.setattr(ctypes, "CDLL", lambda name: hip)
    [device] = hip_devices.list_devices("amdhip64_7.dll")
    assert device["compute_units"] == compute_units


def test_hip_devices_without_a_gpu(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ctypes, "CDLL", lambda name: fake_hip([], count_status=100))
    assert hip_devices.list_devices("amdhip64_7.dll") == []


def test_hip_devices_failure(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ctypes, "CDLL", lambda name: fake_hip([], count_status=101))
    with pytest.raises(RuntimeError, match="HIP error 101"):
        hip_devices.list_devices("amdhip64_7.dll")


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A Windows machine without an NVIDIA GPU or ROCm whose HIP runtime lists
    `listing.devices` from a child process that exits with `listing.status`, in a
    process that was handed no listing."""
    listing = SimpleNamespace(devices=[], status=0, calls=[], kwargs=[])
    runtime = Path("C:/Windows/System32/amdhip64_7.dll")

    def run(args: list[str], **kwargs: object) -> SimpleNamespace:
        listing.calls.append(args)
        listing.kwargs.append(kwargs)
        stdout = json.dumps(listing.devices) if listing.status == 0 else ""
        return SimpleNamespace(
            returncode=listing.status, stdout=stdout, stderr="hipErrorInvalidDevice"
        )

    monkeypatch.setattr(gpu, "is_linux", False)
    monkeypatch.setattr(gpu, "is_windows", True)
    monkeypatch.setattr(gpu, "nvidia", gpu.NvInfo.unavailable())
    monkeypatch.setattr(gpu, "_rocm_installed", lambda: False)
    monkeypatch.setattr(hip_devices, "runtime", lambda: runtime)
    monkeypatch.setattr(subprocess, "run", run)
    # Restored after the test, whatever the code under test records
    monkeypatch.delenv(hip_devices.ENVIRONMENT, raising=False)
    # Windows' flag, for runs of this test on Linux
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    return listing


def test_windows_lists_hip_devices(
    windows: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
):
    windows.devices = [
        {"name": "AMD Radeon(TM) Graphics", "gfx": "gfx1036", "luid": 0x3E1C0}
        | {"total": 512 << 20, "compute_units": 2, "integrated": True},
        {"name": "AMD Radeon RX 9070 XT", "gfx": "gfx1201", "luid": 0x51234}
        | {"total": 16 << 30, "compute_units": 64, "integrated": False},
    ]
    info = gpu.get_amd_info()
    devices = info.devices
    assert [
        (d.index, d.name, d.gfx, d.luid, d.compute_units, d.is_apu) for d in devices
    ] == [
        (0, "AMD Radeon(TM) Graphics", "gfx1036", 0x3E1C0, 2, True),
        (1, "AMD Radeon RX 9070 XT", "gfx1201", 0x51234, 64, False),
    ]
    assert devices[1].supports_fp8
    assert (info.eligible, info.default_index) == (True, 1)
    # The child: this Python, isolated, running hip_devices.py on the runtime, within
    # the timeout a working driver needs with margin
    [args] = windows.calls
    assert args[:3] == [sys.executable, "-I", "-S"]
    assert Path(args[3]) == Path(hip_devices.__file__)
    assert args[4] == str(Path("C:/Windows/System32/amdhip64_7.dll"))
    assert windows.kwargs[0]["timeout"] == hip_devices.TIMEOUT == 10
    # Once: the listing stays in the environment, which the worker inherits.
    assert json.loads(os.environ[hip_devices.ENVIRONMENT]) == windows.devices
    assert [d.luid for d in gpu.get_amd_info().devices] == [0x3E1C0, 0x51234]
    assert len(windows.calls) == 1

    used = {0x3E1C0: 128 << 20, 0x51234: 6 << 30}
    monkeypatch.setattr(gpu, "_dedicated_usage", lambda luid: used[luid])
    assert devices[1].get_current_vram_usage() == gpu.MemoryUsage(
        16 << 30, 6 << 30, 10 << 30
    )


def test_windows_runtime_failure_reports_no_gpu(windows: SimpleNamespace):
    windows.status = 0xC0000005  # an access violation in the driver
    assert not gpu.get_amd_info().is_available
    # The failure is handed on too: the worker does not run a failing driver again.
    assert "exit status 3221225477" in os.environ[hip_devices.ENVIRONMENT]
    with pytest.raises(OSError, match="hipErrorInvalidDevice"):
        hip_devices.listing()
    assert len(windows.calls) == 1


def test_windows_hung_runtime_is_recorded_as_timed_out(
    windows: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
):
    def hang(args: list[str], **kwargs: Any) -> None:
        windows.calls.append(args)
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", hang)
    assert not gpu.get_amd_info().is_available
    with pytest.raises(OSError, match="timed out"):
        hip_devices.listing()
    assert len(windows.calls) == 1


def test_a_handed_on_listing_is_used_without_a_child(windows: SimpleNamespace):
    device = {"name": "AMD Radeon RX 7900 XTX", "gfx": "gfx1100", "luid": 9}
    device |= {"total": 24 << 30, "compute_units": 96, "integrated": False}
    os.environ[hip_devices.ENVIRONMENT] = json.dumps([device])
    assert [d.name for d in gpu.get_amd_info().devices] == [device["name"]]
    os.environ[hip_devices.ENVIRONMENT] = json.dumps({"error": "HIP hung"})
    assert not gpu.get_amd_info().is_available
    assert windows.calls == []


def test_the_worker_inherits_the_hosts_listing(monkeypatch: pytest.MonkeyPatch):
    import worker_spawn

    started: list[dict[str, str]] = []

    def popen(args: list[str], **kwargs: Any) -> None:
        started.append(kwargs["env"])

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.delenv(hip_devices.ENVIRONMENT, raising=False)
    worker_spawn.spawn_worker(["8000"])
    # Without a listing (NVIDIA machines, Linux) the environment is as before.
    assert started[0] is worker_spawn.ENV
    monkeypatch.setenv(hip_devices.ENVIRONMENT, "[]")
    worker_spawn.spawn_worker(["8000"])
    assert started[1] == {**worker_spawn.ENV, hip_devices.ENVIRONMENT: "[]"}


def test_the_nvidia_driver_is_its_cuda_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    system32 = tmp_path / "System32"
    system32.mkdir()
    monkeypatch.setenv("SystemRoot", str(tmp_path))
    assert not hip_devices.nvidia_driver()
    (system32 / "nvcuda.dll").touch()
    assert hip_devices.nvidia_driver()


def test_windows_nvidia_machine_without_rocm_is_not_probed(
    windows: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(gpu, "nvidia", SimpleNamespace(is_available=True))
    windows.devices = [
        {"name": "AMD Radeon(TM) Graphics", "gfx": "gfx1036", "luid": 7}
        | {"total": 512 << 20, "compute_units": 2, "integrated": True}
    ]
    assert not gpu.get_amd_info().is_available
    assert windows.calls == []

    # With ROCm installed (PyTorch's ROCm build), both vendors are listed.
    monkeypatch.setattr(gpu, "_rocm_installed", lambda: True)
    assert [d.gfx for d in gpu.get_amd_info().devices] == ["gfx1036"]


def test_hip_runtime_choice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    system32 = tmp_path / "Windows/System32"
    system32.mkdir(parents=True)
    for name in ("amdhip64.dll", "amdhip64_6.dll", "amdhip64_7.dll", "amdhip64_x.dll"):
        (system32 / name).touch()
    monkeypatch.setenv("SystemRoot", str(tmp_path / "Windows"))
    find_spec = importlib.util.find_spec
    sdk = tmp_path / "site-packages/_rocm_sdk_core"

    def fake_find_spec(name: str, package: str | None = None):
        if name != "_rocm_sdk_core":
            return find_spec(name, package)
        if not sdk.exists():
            return None
        return SimpleNamespace(submodule_search_locations=[str(sdk)])

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    # The driver's newest from HIP 6 on
    assert hip_devices.runtime() == system32 / "amdhip64_7.dll"
    # The ROCm SDK's, which PyTorch loads, once installed
    (sdk / "bin").mkdir(parents=True)
    (sdk / "bin/amdhip64_7.dll").touch()
    assert hip_devices.runtime() == sdk / "bin/amdhip64_7.dll"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows' processes and PDH")
def test_windows_child_process_and_counters(monkeypatch: pytest.MonkeyPatch):
    # The real child process: a runtime that cannot load is an error, not a crash.
    missing = Path(hip_devices.__file__).with_name("amdhip64_99.dll")
    monkeypatch.setattr(hip_devices, "runtime", lambda: missing)
    monkeypatch.delenv(hip_devices.ENVIRONMENT, raising=False)
    with pytest.raises(OSError, match=r"amdhip64_99\.dll"):
        _windows_amd_devices()
    # The real counter of an adapter Windows does not have
    with pytest.raises(OSError, match="PDH"):
        _dedicated_usage(0x7FFF_FFFF_0000_0001)


@pytest.fixture
def server_host(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # The module parses its command line when imported.
    monkeypatch.setattr(sys, "argv", ["server_host.py"])
    return importlib.import_module("server_host")


def vram_labels(
    server_host: ModuleType, monkeypatch: pytest.MonkeyPatch, nvidia: int, amd: int
) -> list[str]:
    usage = gpu.MemoryUsage(8 << 30, 2 << 30, 6 << 30)

    def vendor(count: int) -> SimpleNamespace:
        device = SimpleNamespace(
            get_current_vram_usage=lambda: usage, eligible=True, name="GPU"
        )
        return SimpleNamespace(
            devices=[SimpleNamespace(**vars(device), index=i) for i in range(count)]
        )

    monkeypatch.setattr(server_host, "nvidia", vendor(nvidia))
    monkeypatch.setattr(server_host, "amd", vendor(amd))
    response = asyncio.run(server_host.system_usage(None))
    vram = [s for s in json.loads(response.body) if s["label"].startswith("VRAM")]
    assert all(s["percent"] == 25 for s in vram)
    return [s["label"] for s in vram]


@pytest.mark.parametrize(
    ("nvidia", "amd", "labels"),
    [
        (1, 0, ["VRAM"]),
        (2, 0, ["VRAM 0", "VRAM 1"]),
        (0, 1, ["VRAM"]),
        (0, 2, ["VRAM 0", "VRAM 1"]),
        # Both vendors: NVIDIA's, as before (a GeForce laptop's Radeon graphics)
        (1, 1, ["VRAM"]),
    ],
)
def test_host_vram_line(
    server_host: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    nvidia: int,
    amd: int,
    labels: list[str],
):
    assert vram_labels(server_host, monkeypatch, nvidia, amd) == labels


def test_host_vram_line_leaves_out_small_and_unreadable_amd_gpus(
    server_host: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    # A Ryzen's small graphics (not eligible) stay out; a GPU whose counters fail is left
    # out, and the CPU and RAM stats still show.
    def unreadable() -> gpu.MemoryUsage:
        raise OSError("no counter")

    devices = [
        SimpleNamespace(
            index=0,
            name="iGPU",
            eligible=False,
            get_current_vram_usage=lambda: gpu.MemoryUsage(1 << 30, 1 << 29, 1 << 29),
        ),
        SimpleNamespace(
            index=1, name="dGPU", eligible=True, get_current_vram_usage=unreadable
        ),
    ]
    monkeypatch.setattr(server_host, "nvidia", SimpleNamespace(devices=[]))
    monkeypatch.setattr(server_host, "amd", SimpleNamespace(devices=devices))
    response = asyncio.run(server_host.system_usage(None))
    assert [s["label"] for s in json.loads(response.body)] == ["CPU", "RAM"]


def test_pytorch_default_gpu_on_rocm():
    """PyTorch's GPU setting starts at the strongest AMD GPU on a ROCm build, and at
    GPU 0 on a CUDA build."""
    import torch

    settings = importlib.import_module("packages.chaiNNer_pytorch.settings")
    registered = list(settings.package.settings)

    def default(hip: str | None) -> str:
        with pytest.MonkeyPatch.context() as m:
            m.setattr(torch.version, "hip", hip)
            m.setattr(torch.cuda, "device_count", lambda: 2)
            m.setattr(
                torch.cuda,
                "get_device_properties",
                lambda i: SimpleNamespace(name=f"GPU {i}"),
            )
            m.setattr(
                gpu, "amd", gpu.AmdInfo([amd_gpu(0, 2, is_apu=True), amd_gpu(1, 96)])
            )
            settings.package.settings[:] = []
            importlib.reload(settings)
            [dropdown] = [s for s in settings.package.settings if s.key == "gpu_index"]
            return dropdown.default

    try:
        assert default("7.14.60850") == "1"
        assert default(None) == "0"
    finally:
        # The module as this machine's PyTorch makes it, its settings registered once
        importlib.reload(settings)
        settings.package.settings[:] = registered
