"""The dependencies the packages share on AMD GPUs: ROCm, ONNX Runtime with the MIGraphX
provider and Triton for ROCm.

ROCm 7.14.1 ships as pip wheels: rocm-sdk-core (the HIP runtime), rocm-sdk-libraries
(rocBLAS, hipBLASLt, MIOpen and the other math libraries) and one GPU kernel wheel per
architecture, rocm-sdk-device-gfx<arch>, which unpacks into the libraries' folder. PyTorch's
Linux ROCm build requires AMD's meta package as rocm[device-all,libraries], and AMD's
device-all is every architecture's kernel wheel (5.8 GB). The packages install the bundled
build of the meta package instead, whose device-all extra requires none
(native/tools/build_rocm_meta_wheel.py), and the kernel wheels of the GPUs found.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from api import KB, MB, Dependency
from gpu import amd
from system import is_windows

ROCM_VERSION = "7.14.1"
# PyTorch's ROCm index (Linux), which serves ROCm's wheels and Triton for ROCm too, and
# AMD's own, where AMD's Windows builds of PyTorch and ROCm's Windows wheels are.
PYTORCH_ROCM_INDEX = "https://download.pytorch.org/whl/rocm7.14"
AMD_INDEX = "https://repo.amd.com/rocm/whl-multi-arch/"
META_WHEEL = (
    Path(__file__).resolve().parent / "whls/rocm/rocm-7.14.1+chainner-py3-none-any.whl"
)
# The architectures ROCm 7.14.1 has a GPU kernel wheel for on Linux, with the wheel's size.
LINUX_KERNEL_WHEEL_SIZES = {
    "gfx908": 290.4 * MB,
    "gfx90a": 450.7 * MB,
    "gfx942": 1527.5 * MB,
    "gfx950": 649.9 * MB,
    "gfx1010": 60.6 * MB,
    "gfx1011": 60.3 * MB,
    "gfx1012": 60.3 * MB,
    "gfx1030": 458.5 * MB,
    "gfx1031": 65.0 * MB,
    "gfx1032": 65.0 * MB,
    "gfx1033": 65.0 * MB,
    "gfx1034": 65.0 * MB,
    "gfx1035": 65.0 * MB,
    "gfx1036": 65.0 * MB,
    "gfx1100": 205.0 * MB,
    "gfx1101": 202.0 * MB,
    "gfx1102": 200.0 * MB,
    "gfx1103": 52.5 * MB,
    "gfx1150": 141.5 * MB,
    "gfx1151": 164.7 * MB,
    "gfx1152": 141.3 * MB,
    "gfx1153": 141.5 * MB,
    "gfx1200": 383.8 * MB,
    "gfx1201": 322.2 * MB,
    "gfx1250": 75.1 * MB,
}
# AMD's Windows wheels the packages install, by distribution: the file on AMD_INDEX, its
# SHA-256 and its size, as downloaded on 2026-10-29 (AMD publishes no hashes). The ROCm
# kernel wheels' architectures (no gfx942, gfx950 or gfx1250 on Windows) are ROCm's there.
WINDOWS_WHEELS = {
    "torch": (
        "torch-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "ba7dad6e7c8514fc55d642accfa52569a679472aecc58d9e4428eb76fab2e9bb",
        114519695,
    ),
    "torchvision": (
        "torchvision-0.27.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "2f437ecf57e1c9ac1272643a7231389e759035fac3da785065c846ae4188e0fc",
        900748,
    ),
    "rocm-bootstrap": (
        "rocm_bootstrap-0.1.0-py3-none-any.whl",
        "33e3d4dad086b1f0e06bc049a618ca874d4fb72f758fff00676e23d2a644b652",
        27015,
    ),
    "rocm-sdk-core": (
        "rocm_sdk_core-7.14.1-py3-none-win_amd64.whl",
        "49eb83cb89d50562944a1f0eaf65f4e21f4557813de53e78d022fee81bbfb6e7",
        758745858,
    ),
    "rocm-sdk-libraries": (
        "rocm_sdk_libraries-7.14.1-py3-none-win_amd64.whl",
        "f5f639b488e3e78a44bebba786e917a98c4386001a3fa2a27393a9ce97e2a32d",
        115912267,
    ),
    "amd-torch-device-gfx90a": (
        "amd_torch_device_gfx90a-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "c9cf18fc5b95d4460b58dc07a23d325b1d4b3b4a9a6a2f0a657d031e71a9ed83",
        149614559,
    ),
    "amd-torch-device-gfx110x": (
        "amd_torch_device_gfx110x-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "37fd7a79923f7a379f623684b8ae72fd2a937c27ead91eeebd76844b607ade58",
        243089281,
    ),
    "amd-torch-device-gfx115x": (
        "amd_torch_device_gfx115x-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "2059256a3329df325ca9a29a28f723f4cea1badb7acf7d632d360ef7c250b41b",
        187800925,
    ),
    "amd-torch-device-gfx12-0": (
        "amd_torch_device_gfx12_0-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "26afd0c602569146fb3818e036a1674fcffc83283a16fac9e45d1f3ef3d0d9d7",
        157950129,
    ),
    "amd-torch-device-gfx908": (
        "amd_torch_device_gfx908-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "65d179491530dc231aa9347d2d395732f89d710f5d4f71c7453806332f09e181",
        44154885,
    ),
    "amd-torch-device-gfx1010": (
        "amd_torch_device_gfx1010-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "55015e9af5b42cdf7b14d4b14cf044e4c33dafdf41aca230f80f2bbefa87110d",
        44844301,
    ),
    "amd-torch-device-gfx1011": (
        "amd_torch_device_gfx1011-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "6808adfd16919ca7acab0f5380adbed61bc2cf8c4052f7934df8a962f4570bc9",
        44844604,
    ),
    "amd-torch-device-gfx1012": (
        "amd_torch_device_gfx1012-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "e0a6896fc8663c38dfaa0cf0648e0513c2f5d4cb6a569701de82ff298a89e197",
        44844574,
    ),
    "amd-torch-device-gfx1030": (
        "amd_torch_device_gfx1030-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "cde85e31889098e04e264dd48578f49448efa857923233f3d387aca20d9240b6",
        44927380,
    ),
    "amd-torch-device-gfx1031": (
        "amd_torch_device_gfx1031-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "0a6662e461799881733593588f5ce01ab9b69424c83e1eadbb14ab776fd13003",
        45026617,
    ),
    "amd-torch-device-gfx1032": (
        "amd_torch_device_gfx1032-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "b306f776ce2e50b18906422283c6196b41063d3cf8bbf04b65832fa5f7b269dd",
        45026679,
    ),
    "amd-torch-device-gfx1033": (
        "amd_torch_device_gfx1033-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "84e1625a04e7171de02561db39439558894628cbd7dbd53e092459daadd837ce",
        45026813,
    ),
    "amd-torch-device-gfx1034": (
        "amd_torch_device_gfx1034-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "a12fa7e575a5d2816dfdc70ab5a7ced7a0d8ce3ef0ff9d0d638d7bdf85f3de62",
        45026762,
    ),
    "amd-torch-device-gfx1035": (
        "amd_torch_device_gfx1035-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "0ab545cc0c17a0be592899a02a13e71acc1fe560786b5616050676bc327a04df",
        45026774,
    ),
    "amd-torch-device-gfx1036": (
        "amd_torch_device_gfx1036-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "88a4b86c9eb1be9e9b39428a1617ebc6ec3dcb456c9a7ac5e09e495f7bed018e",
        45026696,
    ),
    "amd-torch-device-gfx1100": (
        "amd_torch_device_gfx1100-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "7a25ec176fb06c542cc832c4d5fa92635ea3e9303422b8791913fb8ecf05c4f9",
        50544482,
    ),
    "amd-torch-device-gfx1101": (
        "amd_torch_device_gfx1101-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "1363cac39e0baa65b25addb8529ec89b8bac994c3ae9634d3a700244fb176d23",
        50505780,
    ),
    "amd-torch-device-gfx1102": (
        "amd_torch_device_gfx1102-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "74bd992e6086b4e2c5f2327de7bd80287947558b7a82dbdbbe95818941350ecd",
        50766787,
    ),
    "amd-torch-device-gfx1103": (
        "amd_torch_device_gfx1103-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "a2051b33db8b0fff7fb1a53db565cbee83784969f1877997581efe372258510e",
        50716682,
    ),
    "amd-torch-device-gfx1150": (
        "amd_torch_device_gfx1150-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "6e0e949ea5af715cdd5cf175479ae213fcbdd5f167ce2fbe9d5efa9db6387dca",
        50277517,
    ),
    "amd-torch-device-gfx1151": (
        "amd_torch_device_gfx1151-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "8b09fca3b5a7afc3e64dc8273b2ec93cac418b5d38226c47ab55a932d32c53bb",
        50076585,
    ),
    "amd-torch-device-gfx1152": (
        "amd_torch_device_gfx1152-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "11d66241ffc53003c90cf2cf6c7b0ec76d9ad2347bed5a73da4a7d8fb7803d4f",
        50277347,
    ),
    "amd-torch-device-gfx1153": (
        "amd_torch_device_gfx1153-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "f7914af753eaca86ac1dab77c2b0e226ac7d1475c8fd6e19f4c163379977aacd",
        50277625,
    ),
    "amd-torch-device-gfx1200": (
        "amd_torch_device_gfx1200-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "da84361af1da450a3096c858ca4dcf6a6403f0d5853f7c53169db0ce24d847f7",
        51168407,
    ),
    "amd-torch-device-gfx1201": (
        "amd_torch_device_gfx1201-2.12.0%2Brocm7.14.1-cp314-cp314-win_amd64.whl",
        "173b9d013ebe0c98ba8c455bf93710929ec015daba346c7a5e9134b79ea98257",
        51186573,
    ),
    "rocm-sdk-device-gfx90a": (
        "rocm_sdk_device_gfx90a-7.14.1-py3-none-win_amd64.whl",
        "9dafa999d061e22c834310aa48074d3f40b2e3717d94aa8dfa5c2c3d4e559dec",
        273421801,
    ),
    "rocm-sdk-device-gfx908": (
        "rocm_sdk_device_gfx908-7.14.1-py3-none-win_amd64.whl",
        "c809f6cfa65d9db0e89c24f9e3054a385ef053a006d9d0bb4c7b9d17e0249d93",
        159309542,
    ),
    "rocm-sdk-device-gfx1010": (
        "rocm_sdk_device_gfx1010-7.14.1-py3-none-win_amd64.whl",
        "ed392a44da05a6483fddab740ee52db87f2c9cdbc88d46ed3530f2ce8c888057",
        47204905,
    ),
    "rocm-sdk-device-gfx1011": (
        "rocm_sdk_device_gfx1011-7.14.1-py3-none-win_amd64.whl",
        "99b452bbc2398d8cda4fd5df95eeba366ccb45237cabf1131df48f93c594b476",
        46858014,
    ),
    "rocm-sdk-device-gfx1012": (
        "rocm_sdk_device_gfx1012-7.14.1-py3-none-win_amd64.whl",
        "da93e7fcc4e81c9ed3e7361a29de2f5de64da9c25910bff31a52ee2d2967e411",
        46853629,
    ),
    "rocm-sdk-device-gfx1030": (
        "rocm_sdk_device_gfx1030-7.14.1-py3-none-win_amd64.whl",
        "24394a281c11eb520faa8102e84c2b97811fa06c7c5f24775837a3c7cc982e2d",
        51618639,
    ),
    "rocm-sdk-device-gfx1031": (
        "rocm_sdk_device_gfx1031-7.14.1-py3-none-win_amd64.whl",
        "51a41d2fe38c3e2f2382c05427812ad13b3eda72187050bb78069ab39bc3f625",
        51648923,
    ),
    "rocm-sdk-device-gfx1032": (
        "rocm_sdk_device_gfx1032-7.14.1-py3-none-win_amd64.whl",
        "e99823260ceb197c82b6a65409f3ba01e508c14ea8e281b46c883e76bf960af7",
        51645340,
    ),
    "rocm-sdk-device-gfx1033": (
        "rocm_sdk_device_gfx1033-7.14.1-py3-none-win_amd64.whl",
        "1c5b4e2b28d5db19895715f4012757d2a8148d032c5f9258c92a73dccba5cb19",
        51647040,
    ),
    "rocm-sdk-device-gfx1034": (
        "rocm_sdk_device_gfx1034-7.14.1-py3-none-win_amd64.whl",
        "bdf74b1d65265a2e0d4bd25554b20f710edf04a35860a9cd07a25069efc6d64e",
        51647648,
    ),
    "rocm-sdk-device-gfx1035": (
        "rocm_sdk_device_gfx1035-7.14.1-py3-none-win_amd64.whl",
        "121c63a6d02328afa9433b8306a5e9a38e56bd3c3a8321c0b996f30b6c632c57",
        51646829,
    ),
    "rocm-sdk-device-gfx1036": (
        "rocm_sdk_device_gfx1036-7.14.1-py3-none-win_amd64.whl",
        "6e353f297a2433da38137af7b0dcbe629a6128fab441ac4b14c453957c261a4e",
        51649103,
    ),
    "rocm-sdk-device-gfx1100": (
        "rocm_sdk_device_gfx1100-7.14.1-py3-none-win_amd64.whl",
        "64eeb74d6ad30816e356f46c9d784204f1ca78eb728dc4299507d8a94ee50bcf",
        179338768,
    ),
    "rocm-sdk-device-gfx1101": (
        "rocm_sdk_device_gfx1101-7.14.1-py3-none-win_amd64.whl",
        "e33b0fd156aacf2dd547deed43cdcedfb249045293f768fb5d4058145a015db3",
        176226399,
    ),
    "rocm-sdk-device-gfx1102": (
        "rocm_sdk_device_gfx1102-7.14.1-py3-none-win_amd64.whl",
        "33b7d0c7723500a6acd2b23733817555eff7717ccdd6623c4fc7265d770893cb",
        174222935,
    ),
    "rocm-sdk-device-gfx1103": (
        "rocm_sdk_device_gfx1103-7.14.1-py3-none-win_amd64.whl",
        "a7d516b30adbb0968c05858a5b9d949d49021f4a61e9b3da9b6a59a432364b4d",
        55138684,
    ),
    "rocm-sdk-device-gfx1150": (
        "rocm_sdk_device_gfx1150-7.14.1-py3-none-win_amd64.whl",
        "8995a7c9dc00d3776a599ca896bd9c06c73629f6428e2dcae471502875486060",
        133874245,
    ),
    "rocm-sdk-device-gfx1151": (
        "rocm_sdk_device_gfx1151-7.14.1-py3-none-win_amd64.whl",
        "8ea906238b6247ce8230bc7a02ab99d410fbae524e3c1ce57e6457c9229ff79a",
        138414354,
    ),
    "rocm-sdk-device-gfx1152": (
        "rocm_sdk_device_gfx1152-7.14.1-py3-none-win_amd64.whl",
        "8434ea45f9fffd6e3f6d49beaface0fdf2e3e95b657c612edc9bce965bebc271",
        133778073,
    ),
    "rocm-sdk-device-gfx1153": (
        "rocm_sdk_device_gfx1153-7.14.1-py3-none-win_amd64.whl",
        "c4532a48282a9b0004408e3e89c2c0aee817c148893dbb903300b36fdc050395",
        133945357,
    ),
    "rocm-sdk-device-gfx1200": (
        "rocm_sdk_device_gfx1200-7.14.1-py3-none-win_amd64.whl",
        "0b08cdddc9c124ef94a55021d9d677b585aa1765ed2cc6ac119fd9994b3a00af",
        376874157,
    ),
    "rocm-sdk-device-gfx1201": (
        "rocm_sdk_device_gfx1201-7.14.1-py3-none-win_amd64.whl",
        "a75a22d5dc198e9e821fb360d52bb23b04101d8f459676ace803787a9172b619",
        312013049,
    ),
}
WINDOWS_ARCHITECTURES = frozenset(
    name.removeprefix("rocm-sdk-device-")
    for name in WINDOWS_WHEELS
    if name.startswith("rocm-sdk-device-")
)

# onnxruntime-migraphx 1.30.0 built with its MIGraphX provider against ROCm 7.14.1
# (manylinux_2_28), MIGraphX's libraries inside it; PyPI's last onnxruntime-migraphx is
# 1.27.1, for glibc 2.34 and a system ROCm 7.2.
ORT_MIGRAPHX_VERSION = "1.30.0+rocm7.14"
ORT_MIGRAPHX_URL = (
    "https://github.com/Rift73/onnxruntime/releases/download/v1.30.0-rocm7.14/"
    "onnxruntime_migraphx-1.30.0%2Brocm7.14-cp314-cp314-manylinux_2_28_x86_64.whl"
)
ORT_MIGRAPHX_SHA256 = "d9dc6fbe7acb8d7b4a40cf33bb7345bae92b154bd9da7023e67c66c1d11cf153"
ORT_MIGRAPHX_SIZE = 100 * MB
# Its Windows build (win_amd64), in the same release: the DirectML provider beside
# MIGraphX's, MIGraphX 2.16.1's DLLs and the MSVC runtime inside it, ROCm's from the ROCm
# pip packages.
ORT_MIGRAPHX_WINDOWS_VERSION = "1.30.0+rocm7.14.dml"
ORT_MIGRAPHX_WINDOWS_URL = (
    "https://github.com/Rift73/onnxruntime/releases/download/v1.30.0-rocm7.14/"
    "onnxruntime_migraphx-1.30.0%2Brocm7.14.dml-cp314-cp314-win_amd64.whl"
)
ORT_MIGRAPHX_WINDOWS_SHA256 = (
    "9a8c2d1fddfd9a00e429d21fe8f50a32496d0dfe797db1675d3ef299782b870c"
)
ORT_MIGRAPHX_WINDOWS_SIZE = 74 * MB
# The other builds of ONNX Runtime, which own its onnxruntime module too: the CPU build
# (Linux), DirectML's (Windows) and NVIDIA's (a computer whose NVIDIA GPU gave way to an
# AMD one). The installer uninstalls an installed one first.
ORT_BUILDS = ("onnxruntime", "onnxruntime-gpu", "onnxruntime-directml")
# The AMD builds that NVIDIA's and the CPU builds replace in turn (a folder that ran on an
# AMD GPU, on a computer whose GPU is now an NVIDIA one or none): ONNX Runtime's MIGraphX
# build; Triton for ROCm, which PyTorch's ROCm build brings on Linux and NVIDIA's PyTorch
# would install Triton over; PyTorch's ROCm builds, which keep its distributions' names
# (Dependency.replaces' "+rocm"); and ROCm's runtime, which PyTorch's and ONNX Runtime's
# ROCm builds run on and nothing else uses (2 GB and more, and on Windows HIP's GPU listing
# at every start while it is installed): the meta package, the SDK and its GPU kernel
# wheels, and on Windows AMD's GPU detection helper and PyTorch's GPU kernel wheels
# (Dependency.replaces' "*"). The PyTorch and ONNX builds each remove the runtime: where
# the ONNX package changes first, PyTorch's ROCm build no longer loads until the PyTorch
# package changes too, as the dependency manager asks (it shows PyTorch missing).
ROCM_RUNTIME_BUILDS = (
    "rocm",
    "rocm-sdk-core",
    "rocm-sdk-libraries",
    "rocm-sdk-device-*",
    "rocm-bootstrap",
    "amd-torch-device-*",
)
ORT_AMD_BUILDS = ("onnxruntime-migraphx", *ROCM_RUNTIME_BUILDS)
TRITON_AMD_BUILDS = ("triton-rocm",)
TORCH_AMD_BUILDS = ("torch+rocm", *TRITON_AMD_BUILDS, *ROCM_RUNTIME_BUILDS)
TORCHVISION_AMD_BUILDS = ("torchvision+rocm",)


def gpu_architectures() -> tuple[list[str], list[str]]:
    """The architectures of the AMD GPUs worth running models on (amd.eligible's: a
    discrete GPU or a large APU, not a Ryzen's small graphics), each once: those ROCm has
    GPU kernels for, and the others. A GPU whose driver reports no architecture, or a
    small one beside them, is left out; none at all keeps the CPU builds."""
    kernels = WINDOWS_ARCHITECTURES if is_windows else LINUX_KERNEL_WHEEL_SIZES.keys()
    found = {
        device.gfx
        for device in amd.devices
        if device.gfx and device.eligible
    }
    return sorted(found & kernels), sorted(found - kernels)


def windows_wheel(display_name: str, pypi_name: str, version: str) -> Dependency:
    """One of AMD's Windows wheels, by its pinned URL."""
    file, sha256, size = WINDOWS_WHEELS[pypi_name]
    return Dependency(
        display_name=display_name,
        pypi_name=pypi_name,
        version=version,
        size_estimate=size,
        url=f"{AMD_INDEX}{file}#sha256={sha256}",
        auto_update=False,
    )


def rocm_runtime(architectures: Sequence[str]) -> list[Dependency]:
    """ROCm's runtime and libraries, the bundled meta package, and the GPU kernel wheels
    of the architectures: from PyTorch's ROCm index on Linux, by their pinned URLs on
    Windows."""
    meta = Dependency(
        display_name="ROCm",
        pypi_name="rocm",
        version=f"{ROCM_VERSION}+chainner",
        size_estimate=14 * KB,
        url=META_WHEEL.as_uri(),
        auto_update=False,
    )
    wheels = [
        ("ROCm runtime", "rocm-sdk-core"),
        ("ROCm libraries", "rocm-sdk-libraries"),
        *(
            (f"ROCm GPU kernels ({architecture})", f"rocm-sdk-device-{architecture}")
            for architecture in architectures
        ),
    ]
    if is_windows:
        return [meta, *(windows_wheel(d, n, ROCM_VERSION) for d, n in wheels)]
    sizes = {
        "rocm-sdk-core": 395.4 * MB,
        "rocm-sdk-libraries": 136.4 * MB,
        **{f"rocm-sdk-device-{a}": s for a, s in LINUX_KERNEL_WHEEL_SIZES.items()},
    }
    return [
        meta,
        *(
            Dependency(
                display_name=display_name,
                pypi_name=pypi_name,
                version=ROCM_VERSION,
                size_estimate=sizes[pypi_name],
                extra_index_url=PYTORCH_ROCM_INDEX,
                auto_update=False,
            )
            for display_name, pypi_name in wheels
        ),
    ]


def onnxruntime_migraphx() -> Dependency:
    """ONNX Runtime's MIGraphX build for this system: on Windows the one that also has
    DirectML."""
    if is_windows:
        return Dependency(
            display_name="ONNX Runtime (DirectML, MIGraphX)",
            pypi_name="onnxruntime-migraphx",
            version=ORT_MIGRAPHX_WINDOWS_VERSION,
            size_estimate=ORT_MIGRAPHX_WINDOWS_SIZE,
            import_name="onnxruntime",
            url=f"{ORT_MIGRAPHX_WINDOWS_URL}#sha256={ORT_MIGRAPHX_WINDOWS_SHA256}",
            replaces=ORT_BUILDS,
        )
    return Dependency(
        display_name="ONNX Runtime (MIGraphX)",
        pypi_name="onnxruntime-migraphx",
        version=ORT_MIGRAPHX_VERSION,
        size_estimate=ORT_MIGRAPHX_SIZE,
        import_name="onnxruntime",
        url=f"{ORT_MIGRAPHX_URL}#sha256={ORT_MIGRAPHX_SHA256}",
        replaces=ORT_BUILDS,
    )


def triton_rocm() -> Dependency:
    """Triton for ROCm (Linux), the release PyTorch's ROCm build pins. It replaces
    NVIDIA's Triton (PyTorch's CUDA build installs it), which owns the triton module
    too."""
    return Dependency(
        display_name="Triton",
        pypi_name="triton-rocm",
        version="3.8.0",
        size_estimate=344.9 * MB,
        import_name="triton",
        extra_index_url=PYTORCH_ROCM_INDEX,
        replaces=("triton",),
    )
