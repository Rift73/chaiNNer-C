"""What the packages install for each GPU (packages/chaiNNer_pytorch, chaiNNer_onnx and
chaiNNer_tensorrt, dependencies/rocm.py): NVIDIA and CPU selections as before, ROCm's
wheels for AMD GPUs; the bundled ROCm meta package (native/tools/build_rocm_meta_wheel.py);
the host's uninstall keeping what another installed package lists; the other builds the
AMD ones replace, and the AMD builds the NVIDIA and CPU ones replace in turn (a folder
that ran on an AMD GPU, on a computer with an NVIDIA GPU or none). Runs without a GPU:
the GPUs are stand-ins."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import importlib
import sys
import zipfile
from types import ModuleType, SimpleNamespace

import build_rocm_meta_wheel
import pytest

import packages.chaiNNer_onnx as onnx_package
import packages.chaiNNer_pytorch as pytorch_package
import packages.chaiNNer_tensorrt as tensorrt_package
from api import Dependency
from dependencies import rocm, store
from dependencies.store import filter_necessary_to_install, replaced_distributions
from gpu import AmdDevice, AmdInfo

NVIDIA = SimpleNamespace(is_available=True, any_needs_legacy_cuda=False, devices=[1])
NVIDIA_LEGACY = SimpleNamespace(
    is_available=True, any_needs_legacy_cuda=True, devices=[1]
)
NO_NVIDIA = SimpleNamespace(is_available=False, any_needs_legacy_cuda=False, devices=[])


def amd(*gpus: str | tuple[str, int]) -> AmdInfo:
    """AMD GPUs by architecture, with 48 compute units (a discrete GPU) unless given."""
    return AmdInfo(
        [
            AmdDevice(i, f"AMD GPU {g}", g, compute_units=48)
            if isinstance(g, str)
            else AmdDevice(i, f"AMD GPU {g[0]}", g[0], compute_units=g[1])
            for i, g in enumerate(gpus)
        ]
    )


def computer(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    *,
    nvidia: SimpleNamespace,
    gfx: tuple[str | tuple[str, int], ...] = (),
    system: str = "linux",
) -> None:
    monkeypatch.setattr(module, "nvidia", nvidia)
    monkeypatch.setattr(rocm, "amd", amd(*gfx))
    monkeypatch.setattr(module, "is_linux", system == "linux")
    monkeypatch.setattr(module, "is_windows", system == "windows")
    monkeypatch.setattr(rocm, "is_windows", system == "windows")
    if hasattr(module, "is_arm_mac"):
        monkeypatch.setattr(module, "is_arm_mac", False)


def pins(dependencies: list) -> list[tuple[str, str]]:
    return [(d.pypi_name, d.version) for d in dependencies]


# ROCm's runtime, which the NVIDIA and CPU builds of PyTorch and ONNX Runtime replace
RUNTIME_BUILDS = [
    "rocm",
    "rocm-sdk-core",
    "rocm-sdk-libraries",
    "rocm-sdk-device-*",
    "rocm-bootstrap",
    "amd-torch-device-*",
]
ROCM_RUNTIME = [
    ("rocm", "7.14.1+chainner"),
    ("rocm-sdk-core", "7.14.1"),
    ("rocm-sdk-libraries", "7.14.1"),
]


@pytest.mark.parametrize("system", ["linux", "windows"])
@pytest.mark.parametrize("gfx", [(), ("gfx1100",)], ids=["cpu", "nvidia-and-amd"])
def test_pytorch_on_nvidia_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, system: str, gfx: tuple[str, ...]
):
    computer(monkeypatch, pytorch_package, nvidia=NVIDIA, gfx=gfx, system=system)
    assert [d.to_dict() for d in pytorch_package.get_pytorch()] == [
        {
            "displayName": "PyTorch",
            "pypiName": "torch",
            "version": "2.14.1+cu132",
            "sizeEstimate": 2 * 1024**3,
            "autoUpdate": False,
            "findLink": "https://download.pytorch.org/whl/cu132",
            "url": None,
            "replaces": ["torch+rocm", "triton-rocm", *RUNTIME_BUILDS],
        },
        {
            "displayName": "TorchVision",
            "pypiName": "torchvision",
            "version": "0.29.1+cu132",
            "sizeEstimate": 2 * 1024**2,
            "autoUpdate": False,
            "findLink": "https://download.pytorch.org/whl/cu132",
            "url": None,
            "replaces": ["torchvision+rocm"],
        },
    ]
    monkeypatch.setattr(pytorch_package, "nvidia", NVIDIA_LEGACY)
    assert pins(pytorch_package.get_pytorch()) == [
        ("torch", "2.14.1+cu126"),
        ("torchvision", "0.29.1+cu126"),
    ]


# A GPU ROCm has no kernels for (gfx906, Radeon VII) or none reported: the CPU build.
@pytest.mark.parametrize("gfx", [(), ("gfx906",), ("",)])
@pytest.mark.parametrize("system", ["linux", "windows"])
def test_pytorch_without_a_gpu_it_runs_on_is_the_cpu_build(
    monkeypatch: pytest.MonkeyPatch, gfx: tuple[str, ...], system: str
):
    computer(monkeypatch, pytorch_package, nvidia=NO_NVIDIA, gfx=gfx, system=system)
    assert [d.to_dict() for d in pytorch_package.get_pytorch()] == [
        {
            "displayName": "PyTorch",
            "pypiName": "torch",
            "version": "2.14.1",
            "sizeEstimate": 140 * 1024**2,
            "autoUpdate": False,
            "findLink": "https://download.pytorch.org/whl/cpu",
            "url": None,
            "replaces": ["torch+rocm", "triton-rocm", *RUNTIME_BUILDS],
        },
        {
            "displayName": "TorchVision",
            "pypiName": "torchvision",
            "version": "0.29.1",
            "sizeEstimate": 800 * 1024,
            "autoUpdate": False,
            "findLink": "https://download.pytorch.org/whl/cpu",
            "url": None,
            "replaces": ["torchvision+rocm"],
        },
    ]
    assert pytorch_package.amd_architectures()[0] == []


def test_pytorch_on_amd_linux_is_rocm_with_the_gpus_kernels_only(
    monkeypatch: pytest.MonkeyPatch,
):
    # Two GPUs of one architecture and one of another; gfx906 has no kernels.
    computer(
        monkeypatch,
        pytorch_package,
        nvidia=NO_NVIDIA,
        gfx=("gfx1201", "gfx1100", "gfx1201", "gfx906"),
    )
    dependencies = pytorch_package.get_pytorch()
    assert pins(dependencies) == [
        ("torch", "2.14.1+rocm7.14"),
        ("torchvision", "0.29.1+rocm7.14"),
        ("triton-rocm", "3.8.0"),
        *ROCM_RUNTIME,
        ("rocm-sdk-device-gfx1100", "7.14.1"),
        ("rocm-sdk-device-gfx1201", "7.14.1"),
    ]
    assert pytorch_package.amd_architectures() == (["gfx1100", "gfx1201"], ["gfx906"])
    for dependency in dependencies:
        assert not dependency.auto_update or dependency.pypi_name == "triton-rocm"
        if dependency.pypi_name == "rocm":
            assert dependency.extra_index_url is None
            assert dependency.url == rocm.META_WHEEL.as_uri()
        else:
            assert dependency.extra_index_url == rocm.PYTORCH_ROCM_INDEX
    # Without device-all: a download of gigabytes, not of every architecture's kernels.
    assert 2.1 * 1024**3 < sum(d.size_estimate for d in dependencies) < 2.8 * 1024**3


def test_pytorch_on_amd_windows_is_amds_build_with_its_kernel_wheels(
    monkeypatch: pytest.MonkeyPatch,
):
    computer(
        monkeypatch,
        pytorch_package,
        nvidia=NO_NVIDIA,
        gfx=("gfx1101", "gfx1030", "gfx1151"),
        system="windows",
    )
    dependencies = pytorch_package.get_pytorch()
    assert pins(dependencies) == [
        ("torch", "2.12.0+rocm7.14.1"),
        ("torchvision", "0.27.0+rocm7.14.1"),
        ("rocm-bootstrap", "0.1.0"),
        ("amd-torch-device-gfx1030", "2.12.0+rocm7.14.1"),
        ("amd-torch-device-gfx1101", "2.12.0+rocm7.14.1"),
        ("amd-torch-device-gfx1151", "2.12.0+rocm7.14.1"),
        ("amd-torch-device-gfx110x", "2.12.0+rocm7.14.1"),
        ("amd-torch-device-gfx115x", "2.12.0+rocm7.14.1"),
        *ROCM_RUNTIME,
        ("rocm-sdk-device-gfx1030", "7.14.1"),
        ("rocm-sdk-device-gfx1101", "7.14.1"),
        ("rocm-sdk-device-gfx1151", "7.14.1"),
    ]
    # Each of AMD's wheels by its URL with the SHA-256 it was pinned at (AMD publishes
    # none); no index, and no Triton (Triton for Windows is not for AMD GPUs).
    for dependency in dependencies:
        assert dependency.extra_index_url is None
        if dependency.pypi_name == "rocm":
            assert dependency.url == rocm.META_WHEEL.as_uri()
            continue
        file, sha256, size = rocm.WINDOWS_WHEELS[dependency.pypi_name]
        assert dependency.url == f"{rocm.AMD_INDEX}{file}#sha256={sha256}"
        assert dependency.size_estimate == size
        assert file.endswith(("-win_amd64.whl", "-py3-none-any.whl"))
        assert len(sha256) == 64
    assert dependencies[0].url == (
        "https://repo.amd.com/rocm/whl-multi-arch/"
        "torch-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl"
        "#sha256=ba7dad6e7c8514fc55d642accfa52569a679472aecc58d9e4428eb76fab2e9bb"
    )


@pytest.mark.parametrize("system", ["linux", "windows"])
@pytest.mark.parametrize(
    ("gpus", "kernels"),
    [
        # A desktop Ryzen's graphics (2 compute units) alone: the CPU builds, no note.
        ((("gfx1036", 2),), []),
        ((("gfx906", 2),), []),
        # Unknown compute units count as too few.
        ((("gfx1100", 0),), []),
        # Radeon 780M graphics (12) and Strix Halo (40) are worth it.
        ((("gfx1103", 12),), ["gfx1103"]),
        ((("gfx1151", 40),), ["gfx1151"]),
        # The small graphics beside a discrete GPU get no kernel wheel.
        ((("gfx1036", 2), ("gfx1100", 48)), ["gfx1100"]),
    ],
)
def test_amd_stack_only_for_gpus_with_eight_compute_units(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    gpus: tuple[tuple[str, int], ...],
    kernels: list[str],
):
    for module in (pytorch_package, onnx_package, tensorrt_package):
        computer(monkeypatch, module, nvidia=NO_NVIDIA, gfx=gpus, system=system)
    assert rocm.gpu_architectures() == (kernels, [])
    assert pytorch_package.amd_architectures() == (kernels, [])
    torch = pytorch_package.get_pytorch()
    names = [d.pypi_name for d in torch]
    assert [n.removeprefix("rocm-sdk-device-") for n in names if "sdk-device" in n] == (
        kernels
    )
    expected_torch = (
        ("2.12.0+rocm7.14.1" if system == "windows" else "2.14.1+rocm7.14")
        if kernels
        else "2.14.1"
    )
    assert torch[0].version == expected_torch
    onnx = onnx_package.get_onnx_runtime().pypi_name
    expected_onnx = "onnxruntime-directml" if system == "windows" else "onnxruntime"
    if kernels:
        expected_onnx = "onnxruntime-migraphx"
    assert onnx == expected_onnx
    triton = pins(tensorrt_package.get_triton())
    if not kernels:
        assert triton == [
            ("triton-windows", "3.5.1.post22")
            if system == "windows"
            else ("triton", "3.8.0")
        ]


@pytest.mark.parametrize("gfx", [(), ("gfx1100",)], ids=["cpu", "nvidia-and-amd"])
def test_onnx_runtime_on_nvidia_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, gfx: tuple[str, ...]
):
    computer(monkeypatch, onnx_package, nvidia=NVIDIA, gfx=gfx, system="linux")
    linux = onnx_package.get_onnx_runtime()
    assert (linux.pypi_name, linux.version, linux.size_estimate) == (
        "onnxruntime-gpu",
        "1.30.0+trt11.4",
        250 * 1024**2,
    )
    assert linux.url == (
        "https://github.com/Rift73/onnxruntime/releases/download/v1.30.0-trt11/"
        "onnxruntime_gpu-1.30.0%2Btrt11.4-cp314-cp314-manylinux_2_28_x86_64.whl"
        "#sha256=b67a9a5b03df5fd4687d33d41dc035542d4ec2c4da8da57741749ee21b783778"
    )
    assert onnx_package.get_rocm() == []
    computer(monkeypatch, onnx_package, nvidia=NVIDIA, gfx=gfx, system="windows")
    windows = onnx_package.get_onnx_runtime()
    assert (windows.version, windows.size_estimate) == ("1.30.0+trt11.4", 160 * 1024**2)
    assert windows.url == (
        "https://github.com/Rift73/onnxruntime/releases/download/v1.30.0-trt11/"
        "onnxruntime_gpu-1.30.0%2Btrt11.4-cp314-cp314-win_amd64.whl"
        "#sha256=3ab12b436d5f6463867b0635366c29fd4a26c3ae489d2b3911c00391411d1ca6"
    )


@pytest.mark.parametrize(
    ("system", "gfx", "runtime"),
    [
        ("linux", (), ("onnxruntime", "1.30.0")),
        ("linux", ("gfx906",), ("onnxruntime", "1.30.0")),
        # Intel and other GPUs, and AMD GPUs without ROCm's Windows kernels (gfx942
        # has none there, gfx906 none at all): DirectML as before.
        ("windows", (), ("onnxruntime-directml", "1.24.4")),
        ("windows", ("gfx906",), ("onnxruntime-directml", "1.24.4")),
        ("windows", ("gfx942",), ("onnxruntime-directml", "1.24.4")),
    ],
)
def test_onnx_runtime_without_migraphx_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    gfx: tuple[str, ...],
    runtime: tuple[str, str],
):
    computer(monkeypatch, onnx_package, nvidia=NO_NVIDIA, gfx=gfx, system=system)
    dependency = onnx_package.get_onnx_runtime()
    assert (dependency.pypi_name, dependency.version, dependency.url) == (
        *runtime,
        None,
    )
    assert onnx_package.get_rocm() == []


def test_onnx_runtime_on_amd_linux_is_the_migraphx_build_on_rocm(
    monkeypatch: pytest.MonkeyPatch,
):
    computer(monkeypatch, onnx_package, nvidia=NO_NVIDIA, gfx=("gfx942",))
    dependency = onnx_package.get_onnx_runtime()
    assert (dependency.pypi_name, dependency.version, dependency.import_name) == (
        "onnxruntime-migraphx",
        "1.30.0+rocm7.14",
        "onnxruntime",
    )
    assert (
        dependency.url == f"{rocm.ORT_MIGRAPHX_URL}#sha256={rocm.ORT_MIGRAPHX_SHA256}"
    )
    # The same ROCm wheels as PyTorch's, so the packages never ask for two versions.
    assert pins(onnx_package.get_rocm()) == [
        *ROCM_RUNTIME,
        ("rocm-sdk-device-gfx942", "7.14.1"),
    ]


def test_onnx_runtime_on_amd_windows_is_the_directml_and_migraphx_build(
    monkeypatch: pytest.MonkeyPatch,
):
    computer(
        monkeypatch, onnx_package, nvidia=NO_NVIDIA, gfx=("gfx1201",), system="windows"
    )
    dependency = onnx_package.get_onnx_runtime()
    assert (dependency.pypi_name, dependency.version, dependency.import_name) == (
        "onnxruntime-migraphx",
        "1.30.0+rocm7.14.dml",
        "onnxruntime",
    )
    # The Linux wheel's release, its Windows asset
    assert dependency.url == (
        "https://github.com/Rift73/onnxruntime/releases/download/v1.30.0-rocm7.14/"
        "onnxruntime_migraphx-1.30.0%2Brocm7.14.dml-cp314-cp314-win_amd64.whl"
        f"#sha256={rocm.ORT_MIGRAPHX_WINDOWS_SHA256}"
    )
    # AMD's Windows ROCm wheels by their pinned URLs, as PyTorch's
    rocm_wheels = onnx_package.get_rocm()
    assert pins(rocm_wheels) == [*ROCM_RUNTIME, ("rocm-sdk-device-gfx1201", "7.14.1")]
    assert all(d.url and d.url.startswith(rocm.AMD_INDEX) for d in rocm_wheels[1:])


@pytest.mark.parametrize(
    ("system", "nvidia", "gfx", "triton"),
    [
        ("linux", NVIDIA, (), ("triton", "3.8.0")),
        ("linux", NVIDIA, ("gfx1100",), ("triton", "3.8.0")),
        ("linux", NO_NVIDIA, (), ("triton", "3.8.0")),
        ("linux", NO_NVIDIA, ("gfx1100",), ("triton-rocm", "3.8.0")),
        ("windows", NVIDIA, (), ("triton-windows", "3.5.1.post22")),
        ("windows", NVIDIA, ("gfx1100",), ("triton-windows", "3.5.1.post22")),
        ("windows", NO_NVIDIA, (), ("triton-windows", "3.5.1.post22")),
        # Triton for Windows is not for AMD GPUs.
        ("windows", NO_NVIDIA, ("gfx1100",), None),
    ],
)
def test_tensorrt_triton(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    nvidia: SimpleNamespace,
    gfx: tuple[str, ...],
    triton: tuple[str, str] | None,
):
    computer(monkeypatch, tensorrt_package, nvidia=nvidia, gfx=gfx, system=system)
    dependencies = tensorrt_package.get_triton()
    assert pins(dependencies) == ([triton] if triton else [])
    for dependency in dependencies:
        assert dependency.import_name == "triton" or dependency.pypi_name == "triton"


@pytest.mark.parametrize("system", ["linux", "windows"])
def test_kernel_wheels_are_each_systems_rocm_architectures(
    monkeypatch: pytest.MonkeyPatch, system: str
):
    linux = set(rocm.LINUX_KERNEL_WHEEL_SIZES)
    assert len(linux) == 25
    assert linux - rocm.WINDOWS_ARCHITECTURES == {"gfx942", "gfx950", "gfx1250"}
    # Every Windows architecture has PyTorch's kernel wheel too.
    for architecture in rocm.WINDOWS_ARCHITECTURES:
        assert f"amd-torch-device-{architecture}" in rocm.WINDOWS_WHEELS
    monkeypatch.setattr(rocm, "amd", amd("gfx942", "gfx1100"))
    monkeypatch.setattr(rocm, "is_windows", system == "windows")
    supported, unsupported = rocm.gpu_architectures()
    kernels = rocm.rocm_runtime(supported)[3:]
    if system == "windows":
        assert (supported, unsupported) == (["gfx1100"], ["gfx942"])
        assert kernels[0].size_estimate == 179_338_768
        assert (kernels[0].url or "").startswith(rocm.AMD_INDEX)
    else:
        assert (supported, unsupported) == (["gfx1100", "gfx942"], [])
        assert all(d.extra_index_url == rocm.PYTORCH_ROCM_INDEX for d in kernels)
        assert kernels[0].size_estimate == 205.0 * 1024**2


def test_the_bundled_meta_package_requires_no_kernel_wheel():
    assert rocm.META_WHEEL.name == "rocm-7.14.1+chainner-py3-none-any.whl"
    with zipfile.ZipFile(rocm.META_WHEEL) as wheel:
        dist_info = "rocm-7.14.1+chainner.dist-info"
        metadata = wheel.read(f"{dist_info}/METADATA").decode().splitlines()
        # RECORD is whole and right, as pip checks it on install.
        record = wheel.read(f"{dist_info}/RECORD").decode().splitlines()
        assert sorted(line.split(",")[0] for line in record) == sorted(wheel.namelist())
        for line in record:
            name, digest, size = line.rsplit(",", 2)
            if name.endswith("RECORD"):
                continue
            data = wheel.read(name)
            expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            assert digest == "sha256=" + expected.rstrip(b"=").decode()
            assert int(size) == len(data)
        assert "rocm_sdk/__init__.py" in wheel.namelist()
    assert "Version: 7.14.1+chainner" in metadata
    assert "Provides-Extra: device-all" in metadata
    assert "Provides-Extra: device" not in metadata
    requirements = [line for line in metadata if line.startswith("Requires-Dist: ")]
    assert "Requires-Dist: rocm-sdk-core==7.14.1" in requirements
    assert (
        'Requires-Dist: rocm-sdk-libraries==7.14.1; extra == "libraries"'
        in requirements
    )
    assert not [r for r in requirements if r.endswith(('"device-all"', '"device"'))]
    assert (
        'Requires-Dist: rocm-sdk-device-gfx1100==7.14.1; extra == "device-gfx1100"'
        in requirements
    )


def amd_pkg_info(device_all: int) -> str:
    lines = [
        "Metadata-Version: 2.4",
        "Name: rocm",
        "Version: 7.14.1",
        "Requires-Dist: rocm-sdk-core==7.14.1",
        "Provides-Extra: device",
        'Requires-Dist: rocm-sdk-device-gfx1010==7.14.1; extra == "device"',
        "Provides-Extra: device-gfx1100",
        'Requires-Dist: rocm-sdk-device-gfx1100==7.14.1; extra == "device-gfx1100"',
        "Provides-Extra: device-all",
        *(
            f'Requires-Dist: rocm-sdk-device-gfx{n}==7.14.1; extra == "device-all"'
            for n in range(device_all)
        ),
        "Dynamic: requires-dist",
    ]
    return "\n".join(lines) + "\n"


def test_the_builder_empties_device_all_and_refuses_another_layout():
    assert build_rocm_meta_wheel.wheel_metadata(amd_pkg_info(25)) == (
        "Metadata-Version: 2.4\n"
        "Name: rocm\n"
        "Version: 7.14.1+chainner\n"
        "Requires-Dist: rocm-sdk-core==7.14.1\n"
        "Provides-Extra: device-gfx1100\n"
        'Requires-Dist: rocm-sdk-device-gfx1100==7.14.1; extra == "device-gfx1100"\n'
        "Provides-Extra: device-all\n"
    )
    with pytest.raises(ValueError, match="24 device-all requirements"):
        build_rocm_meta_wheel.wheel_metadata(amd_pkg_info(24))
    with pytest.raises(ValueError, match=r"not version 7.14.1"):
        build_rocm_meta_wheel.wheel_metadata(
            amd_pkg_info(25).replace("7.14.1\n", "7.14.2\n", 1)
        )


@pytest.fixture
def server_host(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # The module parses its command line when imported.
    monkeypatch.setattr(sys, "argv", ["server_host.py"])
    return importlib.import_module("server_host")


def package(id: str, *names: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=id, dependencies=[SimpleNamespace(pypi_name=n) for n in names]
    )


def test_uninstall_keeps_what_another_installed_package_lists(
    server_host: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    pytorch = package("pytorch", "torch", "rocm", "rocm-sdk-device-gfx1100", "einops")
    onnx = package(
        "onnx", "onnx", "onnxruntime-migraphx", "rocm", "rocm-sdk-device-gfx1100"
    )
    ncnn = package("ncnn", "ncnn")
    everything = [pytorch, onnx, ncnn]
    installed = dict.fromkeys(
        [
            "torch",
            "rocm",
            "rocm-sdk-device-gfx1100",
            "einops",
            "onnx",
            "onnxruntime-migraphx",
        ],
        "1",
    )
    monkeypatch.setattr(server_host, "installed_packages", installed)

    shared = {"rocm", "rocm-sdk-device-gfx1100"}
    assert server_host.shared_distributions([onnx], everything) == shared
    assert server_host.shared_distributions([pytorch], everything) == shared
    # Both leave, or the other one is not installed: nothing is kept.
    assert server_host.shared_distributions([pytorch, onnx], everything) == set()
    for name in ("onnx", "onnxruntime-migraphx"):
        del installed[name]
    assert server_host.shared_distributions([pytorch], everything) == set()
    # No package shares a dependency with another on NVIDIA GPUs and CPUs.
    assert server_host.shared_distributions([ncnn], everything) == set()


def selected_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    nvidia: SimpleNamespace,
    gfx: tuple[str, ...],
    system: str,
) -> list[Dependency]:
    """What the PyTorch, ONNX and TensorRT packages choose for this computer."""
    for module in (pytorch_package, onnx_package, tensorrt_package):
        computer(monkeypatch, module, nvidia=nvidia, gfx=gfx, system=system)
    return [
        *pytorch_package.get_pytorch(),
        onnx_package.get_onnx_runtime(),
        *onnx_package.get_rocm(),
        *tensorrt_package.get_triton(),
    ]


OTHER_COMPUTERS = pytest.mark.parametrize(
    ("nvidia", "gfx"),
    [(NVIDIA, ()), (NVIDIA, ("gfx1100",)), (NO_NVIDIA, ()), (NO_NVIDIA, ("gfx906",))],
    ids=["nvidia", "nvidia-and-amd", "cpu", "amd-without-rocm-kernels"],
)


@pytest.mark.parametrize("system", ["linux", "windows"])
@OTHER_COMPUTERS
def test_nvidia_and_cpu_dependencies_replace_the_amd_builds_alone(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    nvidia: SimpleNamespace,
    gfx: tuple[str, ...],
):
    dependencies = selected_dependencies(
        monkeypatch, nvidia=nvidia, gfx=gfx, system=system
    )
    runtime = onnx_package.get_onnx_runtime().pypi_name
    expected: dict[str, tuple[str, ...]] = {
        "torch": ("torch+rocm", "triton-rocm", *RUNTIME_BUILDS),
        "torchvision": ("torchvision+rocm",),
        runtime: ("onnxruntime-migraphx", *RUNTIME_BUILDS),
    }
    if system == "linux":
        expected["triton"] = ("triton-rocm",)
    assert {d.pypi_name: d.replaces for d in dependencies if d.replaces} == expected


@pytest.mark.parametrize("system", ["linux", "windows"])
def test_on_amd_onnx_runtime_and_triton_replace_the_other_builds(
    monkeypatch: pytest.MonkeyPatch, system: str
):
    dependencies = selected_dependencies(
        monkeypatch, nvidia=NO_NVIDIA, gfx=("gfx1100",), system=system
    )
    builds = ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml")
    expected: dict[str, tuple[str, ...]] = {"onnxruntime-migraphx": builds}
    if system == "linux":
        expected["triton-rocm"] = ("triton",)
    assert {d.pypi_name: d.replaces for d in dependencies if d.replaces} == expected


def test_the_host_installs_with_what_the_workers_dependency_replaces(
    server_host: ModuleType,
):
    # The host reads the worker's packages from their dicts (Package.from_dict).
    received = Dependency.from_dict(rocm.triton_rocm().to_dict())
    assert received.replaces == ("triton",)
    (info,) = server_host.deps_to_dep_info([received])
    assert info.replaces == ("triton",)
    assert Dependency.from_dict(rocm.onnxruntime_migraphx().to_dict()).replaces == (
        rocm.ORT_BUILDS
    )


def listing(
    monkeypatch: pytest.MonkeyPatch, server_host: ModuleType, packages: dict[str, str]
) -> None:
    """`packages` (name -> version) as the installed distributions, as the host lists
    them for the installer and for the dependency manager."""
    monkeypatch.setattr(store, "installed_packages", packages)
    monkeypatch.setattr(server_host, "installed_packages", packages)


@pytest.mark.parametrize("system", ["linux", "windows"])
@pytest.mark.parametrize("nvidia", [NVIDIA, NO_NVIDIA], ids=["nvidia", "cpu"])
def test_a_folder_from_an_amd_computer_swaps_its_amd_builds_for_this_computers(
    monkeypatch: pytest.MonkeyPatch,
    server_host: ModuleType,
    system: str,
    nvidia: SimpleNamespace,
):
    # What the packages installed on an AMD GPU, then what they list on this computer
    amd_stack = selected_dependencies(
        monkeypatch, nvidia=NO_NVIDIA, gfx=("gfx1100",), system=system
    )
    listing(monkeypatch, server_host, {d.pypi_name: d.version for d in amd_stack})
    dependencies = selected_dependencies(
        monkeypatch, nvidia=nvidia, gfx=(), system=system
    )
    to_install = filter_necessary_to_install(server_host.deps_to_dep_info(dependencies))
    # Each is installed, PyTorch's CPU build too where it has the ROCm build's version
    # (2.14.1 on Linux).
    assert [d.package_name for d in to_install] == [d.pypi_name for d in dependencies]
    # First every AMD distribution leaves: the builds that own the same modules, and
    # ROCm's runtime, which nothing here uses.
    assert {d.package_name for d in replaced_distributions(to_install)} == {
        d.pypi_name for d in amd_stack
    }
    # The dependency manager shows them missing, not PyTorch installed at the ROCm
    # build's version.
    package = SimpleNamespace(dependencies=dependencies)
    assert server_host.installed_versions([package]) == {}


@pytest.mark.parametrize("system", ["linux", "windows"])
@OTHER_COMPUTERS
def test_a_computer_without_amd_builds_installs_as_before(
    monkeypatch: pytest.MonkeyPatch,
    server_host: ModuleType,
    system: str,
    nvidia: SimpleNamespace,
    gfx: tuple[str, ...],
):
    dependencies = selected_dependencies(
        monkeypatch, nvidia=nvidia, gfx=gfx, system=system
    )
    before = [dataclasses.replace(d, replaces=()) for d in dependencies]
    # Each dependency as pip lists it after an install (PyTorch's CPU builds with +cpu),
    # at another build of its version (CUDA 12.6's, PyPI's), and older.
    states = [
        {
            d.pypi_name: d.version
            if "+" in d.version or not d.pypi_name.startswith("torch")
            else f"{d.version}+cpu"
            for d in dependencies
        },
        {d.pypi_name: f"{d.version.split('+')[0]}+cu126" for d in dependencies},
        {d.pypi_name: "0.1" for d in dependencies},
    ]
    for state in states:
        listing(monkeypatch, server_host, state)
        to_install = filter_necessary_to_install(
            server_host.deps_to_dep_info(dependencies)
        )
        assert [d.package_name for d in to_install] == [
            d.package_name
            for d in filter_necessary_to_install(server_host.deps_to_dep_info(before))
        ]
        assert replaced_distributions(to_install) == []
        assert server_host.installed_versions(
            [SimpleNamespace(dependencies=dependencies)]
        ) == server_host.installed_versions([SimpleNamespace(dependencies=before)])


@pytest.mark.parametrize("system", ["linux", "windows"])
def test_an_amd_computer_keeps_rocms_runtime(
    monkeypatch: pytest.MonkeyPatch, server_host: ModuleType, system: str
):
    dependencies = selected_dependencies(
        monkeypatch, nvidia=NO_NVIDIA, gfx=("gfx1100",), system=system
    )
    runtime = {
        d.pypi_name
        for d in dependencies
        if store.replaced(RUNTIME_BUILDS, d.pypi_name, d.version)
    }
    assert {"rocm", "rocm-sdk-core", "rocm-sdk-device-gfx1100"} <= runtime
    # Installed at their versions, and at older ones (an update of every dependency):
    # the runtime is never uninstalled first, and counts as installed.
    for state in [
        {d.pypi_name: d.version for d in dependencies},
        {d.pypi_name: "0.1" for d in dependencies},
    ]:
        listing(monkeypatch, server_host, state)
        to_install = filter_necessary_to_install(
            server_host.deps_to_dep_info(dependencies)
        )
        replaced = {d.package_name for d in replaced_distributions(to_install)}
        assert not runtime & replaced
        assert runtime <= set(
            server_host.installed_versions([SimpleNamespace(dependencies=dependencies)])
        )
