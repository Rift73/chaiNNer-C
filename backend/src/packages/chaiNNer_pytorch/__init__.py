from __future__ import annotations

import os

from sanic.log import logger

from api import GB, KB, MB, Dependency, add_package
from dependencies.rocm import (
    PYTORCH_ROCM_INDEX,
    gpu_architectures,
    rocm_runtime,
    triton_rocm,
    windows_wheel,
)
from gpu import nvidia
from system import is_arm_mac, is_linux, is_windows

general = "PyTorch uses .pth models to upscale images."

# AMD's Windows build of PyTorch on ROCm 7.14.1 (the newest it has; Linux runs 2.14.1), by
# the pinned URLs of dependencies.rocm.WINDOWS_WHEELS. A GPU needs its architecture's PyTorch
# kernels and, for RDNA 3, 3.5 and 4, its family's flash attention kernels: what torch's
# device-gfx<arch> extra names, never its device-all.
WINDOWS_ROCM_TORCH = "2.12.0+rocm7.14.1"
WINDOWS_ROCM_TORCHVISION = "0.27.0+rocm7.14.1"
WINDOWS_TORCH_FAMILIES = {
    "gfx1100": "gfx110x",
    "gfx1101": "gfx110x",
    "gfx1102": "gfx110x",
    "gfx1103": "gfx110x",
    "gfx1150": "gfx115x",
    "gfx1151": "gfx115x",
    "gfx1152": "gfx115x",
    "gfx1153": "gfx115x",
    "gfx1200": "gfx12-0",
    "gfx1201": "gfx12-0",
}


def amd_architectures() -> tuple[list[str], list[str]]:
    """The AMD GPUs' architectures PyTorch runs on with ROCm, and the others; none where
    an NVIDIA GPU is found (CUDA is chosen) or on macOS."""
    if nvidia.is_available or not (is_linux or is_windows):
        return [], []
    return gpu_architectures()


if is_arm_mac:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    package_description = general
    inst_hint = f"{general} It is the most widely-used upscaling architecture."
else:
    package_description = (
        f"{general} It is fastest when CUDA (Nvidia GPU) or ROCm (AMD GPU) is supported."
        " Otherwise, it will install with CPU support (which is very slow)."
    )
    inst_hint = f"{general} It is the most widely-used upscaling architecture."
    _supported, _unsupported = amd_architectures()
    if _unsupported and not _supported:
        package_description += (
            " ROCm has no GPU kernels for this computer's AMD GPU"
            f" ({', '.join(_unsupported)}), so it installs with CPU support."
        )


def get_pytorch_rocm(architectures: list[str]) -> list[Dependency]:
    """PyTorch on ROCm with the GPU kernels of the architectures: PyTorch's build on
    Linux, AMD's on Windows."""
    if is_windows:
        families = sorted(
            {
                WINDOWS_TORCH_FAMILIES[a]
                for a in architectures
                if a in WINDOWS_TORCH_FAMILIES
            }
        )
        return [
            windows_wheel("PyTorch", "torch", WINDOWS_ROCM_TORCH),
            windows_wheel("TorchVision", "torchvision", WINDOWS_ROCM_TORCHVISION),
            windows_wheel("ROCm bootstrap", "rocm-bootstrap", "0.1.0"),
            *(
                windows_wheel(
                    f"PyTorch GPU kernels ({name})",
                    f"amd-torch-device-{name}",
                    WINDOWS_ROCM_TORCH,
                )
                for name in [*architectures, *families]
            ),
            *rocm_runtime(architectures),
        ]
    return [
        Dependency(
            display_name="PyTorch",
            pypi_name="torch",
            version="2.14.1+rocm7.14",
            size_estimate=1329.2 * MB,
            extra_index_url=PYTORCH_ROCM_INDEX,
            auto_update=False,
        ),
        Dependency(
            display_name="TorchVision",
            pypi_name="torchvision",
            version="0.29.1+rocm7.14",
            size_estimate=3.3 * MB,
            extra_index_url=PYTORCH_ROCM_INDEX,
            auto_update=False,
        ),
        triton_rocm(),
        *rocm_runtime(architectures),
    ]


def get_pytorch():
    if is_arm_mac:
        return [
            Dependency(
                display_name="PyTorch",
                pypi_name="torch",
                version="2.14.1",
                size_estimate=55.8 * MB,
                auto_update=False,
            ),
            Dependency(
                display_name="TorchVision",
                pypi_name="torchvision",
                version="0.29.1",
                size_estimate=1.3 * MB,
                auto_update=False,
            ),
        ]
    elif rocm_architectures := amd_architectures()[0]:
        return get_pytorch_rocm(rocm_architectures)
    else:
        # Determine CUDA version based on GPU architecture
        # Volta and older need CUDA 12.6 (CUDA 13 dropped Maxwell, Pascal and Volta),
        # newer cards use CUDA 13.2
        if nvidia.is_available and nvidia.any_needs_legacy_cuda:
            cuda_version = "cu126"
            cuda_url = "https://download.pytorch.org/whl/cu126"
        elif nvidia.is_available:
            cuda_version = "cu132"
            cuda_url = "https://download.pytorch.org/whl/cu132"
        else:
            cuda_version = None
            cuda_url = "https://download.pytorch.org/whl/cpu"

        return [
            Dependency(
                display_name="PyTorch",
                pypi_name="torch",
                version=f"2.14.1+{cuda_version}" if cuda_version else "2.14.1",
                size_estimate=2 * GB if nvidia.is_available else 140 * MB,
                extra_index_url=cuda_url,
                auto_update=False,
            ),
            Dependency(
                display_name="TorchVision",
                pypi_name="torchvision",
                version=f"0.29.1+{cuda_version}" if cuda_version else "0.29.1",
                size_estimate=2 * MB if nvidia.is_available else 800 * KB,
                extra_index_url=cuda_url,
                auto_update=False,
            ),
        ]


package = add_package(
    __file__,
    id="chaiNNer_pytorch",
    name="PyTorch",
    description=package_description,
    dependencies=[
        *get_pytorch(),
        Dependency(
            display_name="FaceXLib",
            pypi_name="facexlib",
            version="0.3.0",
            size_estimate=59.6 * KB,
        ),
        Dependency(
            display_name="Einops",
            pypi_name="einops",
            version="0.8.2",
            size_estimate=42.2 * KB,
        ),
        Dependency(
            display_name="safetensors",
            pypi_name="safetensors",
            version="0.8.0",
            size_estimate=1 * MB,
        ),
        Dependency(
            display_name="Spandrel",
            pypi_name="spandrel",
            # The fork's build of 0.4.2: adds DRFT, DUAL and SST; c2 computes fp16
            # LayerNorm statistics in fp32 and pads exported ONNX graphs per input;
            # c3 runs DUAL in fp16; c4 runs HAT and DAT in fp16; c5 lays out their
            # attention for speed (DAT2 2x in PyTorch) and for ONNX export; c6 runs it
            # through scaled_dot_product_attention (cuDNN or memory-efficient attention);
            # c7 folds DRFT's edge branches without a per-channel Python loop; c8 adds
            # WSR, GaterV3, SMoSR, FIGSR and GFISRv2; c9 loads released GaterV3 models
            # (no flash buffer, resselt's upsamplers); c10 runs DUAL's context layers
            # as neighborhood attention; c11 exports HAT's and DAT's attention in the
            # layout MIGraphX fuses, on request.
            version="0.4.2+c11",
            size_estimate=580 * KB,
            url=(
                "https://github.com/Rift73/spandrel/releases/download/v0.4.2-c11/"
                "spandrel-0.4.2%2Bc11-py3-none-any.whl"
                "#sha256=77467a3fbd2e915ee83b52ec10546714c6820304c7983d4a55d5185952b08cb2"
            ),
        ),
        Dependency(
            display_name="Spandrel extra architectures",
            pypi_name="spandrel_extra_arches",
            version="0.2.0",
            size_estimate=83 * KB,
        ),
    ],
    icon="PyTorch",
    color="#DD6B20",
)

pytorch_category = package.add_category(
    name="PyTorch",
    description="Nodes for using the PyTorch Neural Network Framework with images.",
    icon="PyTorch",
    color="#DD6B20",
    install_hint=inst_hint,
)

logger.debug(f"Loaded package {package.name}")
