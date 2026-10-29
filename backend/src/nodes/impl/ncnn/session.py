from __future__ import annotations

import atexit
import tempfile
from weakref import WeakKeyDictionary

from ncnn import ncnn

from packages.chaiNNer_ncnn.settings import CPU_DEVICE, NcnnSettings

from .model import NcnnModelWrapper

use_gpu = ncnn.get_gpu_count() > 0


def create_ncnn_net(model: NcnnModelWrapper, settings: NcnnSettings) -> ncnn.Net:
    net = ncnn.Net()

    if model.fp == "fp16":
        net.opt.use_fp16_packed = True
        net.opt.use_fp16_storage = True
        net.opt.use_fp16_arithmetic = True
    else:
        net.opt.use_fp16_packed = False
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_arithmetic = False

    if use_gpu and settings.gpu_index != CPU_DEVICE:
        # Use vulkan compute
        net.opt.use_vulkan_compute = True
        net.set_vulkan_device(settings.gpu_index)
    else:
        # Configure Winograd/SGEMM optimizations
        net.opt.use_winograd_convolution = settings.winograd
        net.opt.use_sgemm_convolution = settings.sgemm
        # Configure multithreading
        net.opt.num_threads = settings.threads
        net.opt.openmp_blocktime = settings.blocktime

    # Load model param and bin. PyPI ncnn reports a failure only by a non-zero return
    # code; it prints the reason to stderr, which the log keeps. The bin loads from a
    # file on the GPU as well: PyPI ncnn's load_model_mem returns no code to check.
    ret = net.load_param_mem(model.model.write_param())
    if ret != 0:
        raise ValueError(
            f"NCNN could not load the model's .param (error code {ret}). Its reason is in chaiNNer's log."
        )
    with tempfile.TemporaryDirectory() as tmp_model_dir:
        bin_filename = tmp_model_dir + "/ncnn-model.bin"
        model.model.write_bin(bin_filename)
        ret = net.load_model(bin_filename)
    if ret != 0:
        raise ValueError(
            f"NCNN could not load the model's .bin (error code {ret}). Its reason is in chaiNNer's log."
        )

    return net


def _net_settings(settings: NcnnSettings) -> tuple[object, ...]:
    """The settings `create_ncnn_net` builds a net from."""
    if use_gpu and settings.gpu_index != CPU_DEVICE:
        return (settings.gpu_index,)
    return (settings.winograd, settings.sgemm, settings.threads, settings.blocktime)


# One net per model: a net created from other settings is replaced, which frees its
# memory before the new one is created.
__session_cache: WeakKeyDictionary[
    NcnnModelWrapper, tuple[tuple[object, ...], ncnn.Net]
] = WeakKeyDictionary()


def get_ncnn_net(model: NcnnModelWrapper, settings: NcnnSettings) -> ncnn.Net:
    key = _net_settings(settings)
    cached = __session_cache.get(model)
    if cached is not None and cached[0] != key:
        del __session_cache[model]
        cached = None
    if cached is None:
        cached = (key, create_ncnn_net(model, settings=settings))
        __session_cache[model] = cached
    return cached[1]


def destroy_gpu_instance() -> None:
    """Frees the cached nets, then ncnn's Vulkan instance. Runs at exit.

    PyPI ncnn creates the instance at the first get_gpu_count() and otherwise frees
    it only in a static destructor, once the Vulkan driver is unloading: a process
    that exits normally with it alive crashes (0xC0000005). A Vulkan net freed after
    the instance crashes the same way, so the cached nets go first. A worker the host
    terminates never gets here.
    """
    __session_cache.clear()
    ncnn.destroy_gpu_instance()


atexit.register(destroy_gpu_instance)
