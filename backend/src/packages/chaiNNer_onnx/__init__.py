from __future__ import annotations

from sanic.log import logger

from api import KB, MB, Dependency, add_package
from dependencies.rocm import (
    ORT_AMD_BUILDS,
    gpu_architectures,
    onnxruntime_migraphx,
    rocm_runtime,
)
from gpu import nvidia
from system import is_arm_mac, is_linux, is_windows

general = "ONNX uses .onnx models to upscale images."
conversion = "It also helps to convert between PyTorch and NCNN."
amd_providers = (
    "On AMD GPUs it uses MIGraphX on Linux; on Windows DirectML by default, with"
    " MIGraphX as the other choice."
)

if is_arm_mac:
    package_description = f"{general} {conversion} However, it does not support CoreML."
    inst_hint = general
else:
    package_description = (
        f"{general} {conversion} It is fastest when CUDA is supported. With the"
        " TensorRT package installed, it can also be configured to use TensorRT."
        f" {amd_providers}"
    )
    inst_hint = f"{general} {amd_providers}"


def amd_architectures() -> list[str]:
    """The AMD GPUs' architectures ONNX Runtime runs on with MIGraphX: on Linux and
    Windows, where no NVIDIA GPU is found (CUDA is chosen), those ROCm has GPU kernels
    for."""
    if nvidia.is_available or not (is_linux or is_windows):
        return []
    return gpu_architectures()[0]


def get_rocm():
    """ROCm for the MIGraphX build of ONNX Runtime, which runs on ROCm's pip wheels."""
    architectures = amd_architectures()
    return rocm_runtime(architectures) if architectures else []


def get_onnx_runtime():
    if nvidia.is_available and is_windows:
        # onnxruntime-gpu 1.30.0 rebuilt from the unmodified v1.30.0 tag with its
        # TensorRT provider linked against TensorRT 11.4 (PyPI's links TensorRT 10). The
        # provider uses the TensorRT package's TensorRT 11 (see onnx/session.py).
        return Dependency(
            display_name="ONNX Runtime (GPU)",
            pypi_name="onnxruntime-gpu",
            version="1.30.0+trt11.4",
            size_estimate=160 * MB,
            import_name="onnxruntime",
            url=(
                "https://github.com/Rift73/onnxruntime/releases/download/v1.30.0-trt11/"
                "onnxruntime_gpu-1.30.0%2Btrt11.4-cp314-cp314-win_amd64.whl"
                "#sha256=3ab12b436d5f6463867b0635366c29fd4a26c3ae489d2b3911c00391411d1ca6"
            ),
            replaces=ORT_AMD_BUILDS,
        )
    elif nvidia.is_available and is_linux:
        # The same build for Linux (manylinux_2_28): the unmodified v1.30.0 tag with its
        # TensorRT provider linked against TensorRT 11.4.
        return Dependency(
            display_name="ONNX Runtime (GPU)",
            pypi_name="onnxruntime-gpu",
            version="1.30.0+trt11.4",
            size_estimate=250 * MB,
            import_name="onnxruntime",
            url=(
                "https://github.com/Rift73/onnxruntime/releases/download/v1.30.0-trt11/"
                "onnxruntime_gpu-1.30.0%2Btrt11.4-cp314-cp314-manylinux_2_28_x86_64.whl"
                "#sha256=b67a9a5b03df5fd4687d33d41dc035542d4ec2c4da8da57741749ee21b783778"
            ),
            replaces=ORT_AMD_BUILDS,
        )
    elif nvidia.is_available:
        return Dependency(
            display_name="ONNX Runtime (GPU)",
            pypi_name="onnxruntime-gpu",
            version="1.30.0",
            size_estimate=120 * MB,
            import_name="onnxruntime",
            replaces=ORT_AMD_BUILDS,
        )
    elif amd_architectures():
        return onnxruntime_migraphx()
    elif is_windows:
        # onnxruntime-directml's last release is 1.24.4; it serves the public API 22
        # that the native session wrapper requests.
        return Dependency(
            display_name="ONNX Runtime (DirectMl)",
            pypi_name="onnxruntime-directml",
            version="1.24.4",
            size_estimate=15 * MB,
            import_name="onnxruntime",
            replaces=ORT_AMD_BUILDS,
        )
    else:
        return Dependency(
            display_name="ONNX Runtime",
            pypi_name="onnxruntime",
            version="1.30.0",
            size_estimate=6 * MB,
            replaces=ORT_AMD_BUILDS,
        )


package = add_package(
    __file__,
    id="chaiNNer_onnx",
    name="ONNX",
    description=package_description,
    dependencies=[
        Dependency(
            display_name="ONNX",
            pypi_name="onnx",
            version="1.23.2",
            size_estimate=12 * MB,
        ),
        Dependency(
            display_name="ONNX Optimizer",
            pypi_name="onnxoptimizer",
            version="0.4.2",
            size_estimate=300 * KB,
        ),
        get_onnx_runtime(),
        Dependency(
            display_name="Protobuf",
            pypi_name="protobuf",
            version="7.36.2",
            size_estimate=500 * KB,
        ),
        *get_rocm(),
    ],
    icon="ONNX",
    color="#63B3ED",
)


onnx_category = package.add_category(
    name="ONNX",
    description="Nodes for using the ONNX Neural Network Framework with images.",
    icon="ONNX",
    color="#63B3ED",
    install_hint=inst_hint,
)


logger.debug(f"Loaded package {package.name}")
