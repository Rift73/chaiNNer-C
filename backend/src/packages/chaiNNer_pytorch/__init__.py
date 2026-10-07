import os

import torch
from sanic.log import logger

from api import GB, KB, MB, Dependency, add_package
from gpu import nvidia
from system import is_arm_mac

general = "PyTorch uses .pth models to upscale images."


def _enable_tf32() -> None:
    # Best effort: enable TF32 paths for FP32 matmul/conv where supported.
    try:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            matmul_backend = torch.backends.cuda.matmul
            if hasattr(matmul_backend, "fp32_precision"):
                matmul_backend.fp32_precision = "tf32"
            elif hasattr(matmul_backend, "allow_tf32"):
                matmul_backend.allow_tf32 = True

        if hasattr(torch.backends, "cudnn") and hasattr(
            torch.backends.cudnn, "allow_tf32"
        ):
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        logger.warning(
            "Failed to enable TF32 for chaiNNer PyTorch runtime.", exc_info=True
        )


_enable_tf32()

if is_arm_mac:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    package_description = general
    inst_hint = f"{general} It is the most widely-used upscaling architecture."
else:
    package_description = (
        f"{general} and is fastest when CUDA is supported (Nvidia GPU). If CUDA is"
        " unsupported, it will install with CPU support (which is very slow)."
    )
    inst_hint = (
        f"{general} It is the most widely-used upscaling architecture. However, it does"
        " not support AMD GPUs."
    )


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
            version="0.4.2",
            size_estimate=264 * KB,
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
