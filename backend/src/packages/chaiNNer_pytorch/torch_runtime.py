"""PyTorch's process-wide settings for chaiNNer.

The worker imports this module with the package's node modules, all before it reports
the PyTorch nodes loaded, so every PyTorch node runs with these settings. A torch that
is not installed, or fails to load (a DLL error), costs only the PyTorch nodes, which
then fail to import one by one; the package stays registered, so the dependency manager
can install or repair it.
"""

from sanic.log import logger


def _configure_torch() -> None:
    try:
        import torch
    except Exception as error:
        if not (isinstance(error, ModuleNotFoundError) and error.name == "torch"):
            logger.error(
                "Failed to import PyTorch; the PyTorch nodes are unavailable.",
                exc_info=True,
            )
        return

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

    # Best effort: try cuDNN attention before PyTorch's other attention backends. It
    # takes FP16/BF16 calls only (FP32 ones fall through as before). Windows PyTorch
    # has no Flash Attention, and for DUAL in FP16 cuDNN is 8-10 % faster than the
    # memory-efficient kernel, the calls that add a position bias included. PyTorch's
    # public sdpa_kernel(set_priority=True) restores the order when its block ends;
    # for a lasting order this calls the private setter that block uses.
    try:
        from torch._C import _get_sdp_priority_order, _set_sdp_priority_order
        from torch.nn.attention import SDPBackend

        cudnn = SDPBackend.CUDNN_ATTENTION.value
        rest = [b for b in _get_sdp_priority_order() if b != cudnn]
        _set_sdp_priority_order([cudnn, *rest])
    except Exception:
        logger.warning(
            "Failed to prefer cuDNN attention for chaiNNer PyTorch runtime.",
            exc_info=True,
        )


_configure_torch()
