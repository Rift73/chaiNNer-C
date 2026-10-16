from sanic.log import logger

from api import MB, Dependency, add_package

# Convert To ONNX's Mixed FP16 and Mixed BF16 data types run NVIDIA ModelOpt's AutoCast
# (nodes/impl/onnx/autocast.py); this package has no nodes of its own. ModelOpt is
# installed without its [onnx] extra, which on Windows pins onnxruntime-gpu 1.22,
# onnx 1.21 and CUDA 12 CuPy over the ONNX package's: the extra's modules AutoCast
# imports are listed here instead, and pip resolves ModelOpt's own requirements.
package = add_package(
    __file__,
    id="chaiNNer_modelopt",
    name="NVIDIA ModelOpt",
    description=(
        "NVIDIA's Model Optimizer. Convert To ONNX uses its AutoCast for mixed-precision"
        " (FP16 or BF16 with FP32 where needed) ONNX models, which TensorRT builds in"
        " their own precision. Needs the PyTorch and ONNX packages."
    ),
    dependencies=[
        Dependency(
            display_name="NVIDIA ModelOpt",
            pypi_name="nvidia-modelopt",
            version="0.47.0",
            size_estimate=60 * MB,
        ),
        Dependency(
            display_name="ONNX GraphSurgeon",
            pypi_name="onnx-graphsurgeon",
            version="0.6.1",
            size_estimate=1 * MB,
        ),
        Dependency(
            display_name="Polygraphy",
            pypi_name="polygraphy",
            version="0.53.6",
            size_estimate=5 * MB,
        ),
        Dependency(
            display_name="ONNX Script",
            pypi_name="onnxscript",
            version="0.7.2",
            size_estimate=10 * MB,
        ),
        Dependency(
            display_name="OnnxSlim",
            pypi_name="onnxslim",
            version="0.1.98",
            size_estimate=3 * MB,
        ),
        Dependency(
            display_name="LIEF",
            pypi_name="lief",
            version="1.0.0",
            size_estimate=14 * MB,
        ),
        Dependency(
            display_name="cppimport",
            pypi_name="cppimport",
            version="26.4.17",
            size_estimate=1 * MB,
        ),
    ],
    icon="BsNvidia",
    color="#76B900",
)

logger.debug(f"Loaded package {package.name}")
