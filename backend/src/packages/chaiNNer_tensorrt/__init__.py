from __future__ import annotations

from sanic.log import logger

from api import GB, MB, Category, Dependency, add_package
from gpu import NvDevice, nvidia
from system import is_arm_mac

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
            size_estimate=int(1.2 * GB),
        ),
        Dependency(
            display_name="CUDA Python",
            pypi_name="cuda-python",
            version="13.4.1",
            size_estimate=20 * MB,
        ),
        # Compiles DUAL's TensorRT plugin kernels (Build Engine, in a child process).
        Dependency(
            display_name="Triton",
            pypi_name="triton-windows",
            version="3.5.1.post22",
            size_estimate=45 * MB,
            import_name="triton",
        ),
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
