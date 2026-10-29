from __future__ import annotations

from sanic.log import logger

from api import GB, MB, Category, Dependency, add_package
from dependencies.rocm import gpu_architectures, triton_rocm
from gpu import NvDevice, nvidia
from system import is_arm_mac, is_linux, is_windows

package_description = (
    "TensorRT provides native NVIDIA TensorRT support for optimized GPU inference. "
    "It offers significant performance improvements over ONNX Runtime by using "
    "NVIDIA's proprietary optimization engine. Engines are built specifically for "
    "your GPU architecture for maximum performance."
)

install_hint = (
    "TensorRT uses NVIDIA's TensorRT engine for GPU-accelerated inference. "
    "Requires an NVIDIA GPU with CUDA support."
)


def get_triton() -> list[Dependency]:
    # The package is disabled without an NVIDIA GPU. Where PyTorch runs on ROCm instead, the
    # Triton its ROCm build pins on Linux (NVIDIA's would overwrite its triton module), and
    # none on Windows (Triton for Windows is not for AMD GPUs).
    on_rocm = not nvidia.is_available and bool(gpu_architectures()[0])
    if is_windows:
        if on_rocm:
            return []
        return [
            Dependency(
                display_name="Triton",
                pypi_name="triton-windows",
                version="3.5.1.post22",
                size_estimate=45 * MB,
                import_name="triton",
            )
        ]
    if is_linux and on_rocm:
        return [triton_rocm()]
    # The release PyTorch 2.14.1 pins on Linux, so one Triton serves both packages.
    return [
        Dependency(
            display_name="Triton",
            pypi_name="triton",
            version="3.8.0",
            size_estimate=250 * MB,
        )
    ]


# Always register the package so dependencies are installed in CI
package = add_package(
    __file__,
    id="chaiNNer_tensorrt",
    name="TensorRT",
    description=package_description,
    dependencies=[
        Dependency(
            display_name="TensorRT",
            pypi_name="tensorrt",
            version="11.4.0.106",
            # The Linux libraries wheel is 3.8 GB.
            size_estimate=int((1.2 if is_windows else 3.8) * GB),
        ),
        Dependency(
            display_name="CUDA Python",
            pypi_name="cuda-python",
            version="13.4.1",
            size_estimate=20 * MB,
        ),
        # Compiles DUAL's TensorRT plugin kernels (Build Engine, in a child process).
        *get_triton(),
    ],
    icon="BsNvidia",
    color="#76B900",
)


def _supports_cuda_13(gpu: NvDevice) -> bool:
    # TensorRT is installed for CUDA 13, which dropped GPUs older than Turing.
    # Assume support if the compute capability is unknown.
    cc = gpu.compute_capability
    return cc is None or cc >= (7, 5)


if not nvidia.is_available:
    package.disabled = True
    package.disabled_reason = "TensorRT requires an NVIDIA GPU with CUDA support"
elif not any(_supports_cuda_13(gpu) for gpu in nvidia.devices):
    package.disabled = True
    package.disabled_reason = (
        "TensorRT requires an NVIDIA GPU with compute capability 7.5 or newer"
        " (GTX 16 series, RTX 20 series, or newer)"
    )
elif nvidia.driver_major_version is not None and nvidia.driver_major_version < 580:
    package.disabled = True
    package.disabled_reason = "TensorRT requires NVIDIA driver version 580 or newer"

tensorrt_category: Category | None = None
if not package.disabled and not is_arm_mac:
    tensorrt_category = package.add_category(
        name="TensorRT",
        description="Nodes for using NVIDIA TensorRT for optimized GPU inference.",
        icon="BsNvidia",
        color="#76B900",
        install_hint=install_hint,
    )
    logger.debug("Loaded package %s", package.name)
