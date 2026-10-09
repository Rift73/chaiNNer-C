from sanic.log import logger

from api import KB, MB, Dependency, add_package
from gpu import nvidia
from system import is_arm_mac, is_windows

general = "ONNX uses .onnx models to upscale images."
conversion = "It also helps to convert between PyTorch and NCNN."

if is_arm_mac:
    package_description = f"{general} {conversion} However, it does not support CoreML."
    inst_hint = general
else:
    package_description = (
        f"{general} {conversion} It is fastest when CUDA is supported. With the"
        " TensorRT package installed, it can also be configured to use TensorRT."
    )
    inst_hint = f"{general} It does not support AMD GPUs, in linux."


def get_onnx_runtime():
    if nvidia.is_available and is_windows:
        # onnxruntime-gpu 1.30.0 rebuilt from the unmodified v1.30.0 tag with its
        # TensorRT provider linked against TensorRT 11 (PyPI's links TensorRT 10). The
        # provider uses the TensorRT package's TensorRT 11 (see onnx/session.py).
        return Dependency(
            display_name="ONNX Runtime (GPU)",
            pypi_name="onnxruntime-gpu",
            version="1.30.0+trt11",
            size_estimate=160 * MB,
            import_name="onnxruntime",
            url=(
                "https://github.com/Rift73/onnxruntime/releases/download/v1.30.0-trt11/"
                "onnxruntime_gpu-1.30.0%2Btrt11-cp314-cp314-win_amd64.whl"
                "#sha256=443287192cb0add06e1faa7022f1badf9ead0e77b82cd6deb0fc2192af5a96d3"
            ),
        )
    elif nvidia.is_available:
        return Dependency(
            display_name="ONNX Runtime (GPU)",
            pypi_name="onnxruntime-gpu",
            version="1.30.0",
            size_estimate=120 * MB,
            import_name="onnxruntime",
        )
    elif is_windows:
        # onnxruntime-directml's last release is 1.24.4; it serves the public API 22
        # that the native session wrapper requests.
        return Dependency(
            display_name="ONNX Runtime (DirectMl)",
            pypi_name="onnxruntime-directml",
            version="1.24.4",
            size_estimate=15 * MB,
            import_name="onnxruntime",
        )
    else:
        return Dependency(
            display_name="ONNX Runtime",
            pypi_name="onnxruntime",
            version="1.30.0",
            size_estimate=6 * MB,
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
