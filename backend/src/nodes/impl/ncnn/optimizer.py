from .model import BinaryOpTypes as BOT  # noqa
from .model import EltwiseOpTypes as EOT  # noqa
from .model import NcnnModel
from ..native_ncnn_graph import optimize


class NcnnOptimizer:
    def __init__(self, model: NcnnModel) -> None:
        self.model = model

    def __fuse_batchnorm_scale(self):
        optimize(self.model, "fuse_batchnorm_scale")

    def __fuse_x_batchnorm(self):
        """Combines fuse_convolution_batchnorm, fuse_convolutiondepthwise_batchnorm,
        fuse_deconvolution_batchnorm, fuse_deconvolutiondepthwise_batchnorm, and
        fuse_innerproduct_batchnorm"""

        optimize(self.model, "fuse_x_batchnorm")

    def __fuse_x_mul(self):
        """Combines fuse_convolution_mul, fuse_convolutiondepthwise_mul,
        and fuse_deconvolution_mul"""

        optimize(self.model, "fuse_x_mul")

    def __fuse_x_add(self):
        """Combines fuse_convolution_add, fuse_convolutiondepthwise_add,
        fuse_deconvolution_add, and fuse_innerproduct_add"""

        optimize(self.model, "fuse_x_add")

    def __fuse_innerproduct_dropout(self):
        optimize(self.model, "fuse_innerproduct_dropout")

    def __fuse_x_activation(self):
        """Combines fuse_convolution_activation, fuse_convolution1d_activation,
        fuse_convolutiondepthwise_activation, fuse_deconvolution_activation,
        fuse_deconvolutiondepthwise_activation, and fuse_innerproduct_activation"""

        optimize(self.model, "fuse_x_activation")

    def __fuse_memorydata_binaryop(self):
        optimize(self.model, "fuse_memorydata_binaryop")

    def __fuse_binaryop_eltwise(self):
        optimize(self.model, "fuse_binaryop_eltwise")

    def __eliminate_dropout(self):
        optimize(self.model, "eliminate_dropout")

    def __eliminate_pooling1x1(self):
        optimize(self.model, "eliminate_pooling1x1")

    def __eliminate_noop(self):
        optimize(self.model, "eliminate_noop")

    def __eliminate_split(self):
        optimize(self.model, "eliminate_split")

    def __eliminate_orphaned_memorydata(self):
        optimize(self.model, "eliminate_orphaned_memorydata")

    def __eliminate_reshape_after_global_pooling(self):
        optimize(self.model, "eliminate_reshape_after_global_pooling")

    def __eliminate_flatten_after_global_pooling(self):
        optimize(self.model, "eliminate_flatten_after_global_pooling")

    def __eliminate_flatten_after_innerproduct(self):
        optimize(self.model, "eliminate_flatten_after_innerproduct")

    def __eliminate_reshape_before_binaryop(self):
        optimize(self.model, "eliminate_reshape_before_binaryop")

    def __replace_reduction_with_global_pooling(self):
        optimize(self.model, "replace_reduction_with_global_pooling")

    def __replace_prelu_with_leaky_relu(self):
        optimize(self.model, "replace_prelu_with_leaky_relu")

    def __replace_convolution_with_innerproduct_after_global_pooling(self):
        optimize(
            self.model, "replace_convolution_with_innerproduct_after_global_pooling"
        )

    def __replace_convolution_with_innerproduct_after_innerproduct(self):
        optimize(self.model, "replace_convolution_with_innerproduct_after_innerproduct")

    def optimize(self) -> None:
        optimize(self.model)
