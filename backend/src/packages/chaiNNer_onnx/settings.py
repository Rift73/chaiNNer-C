from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import metadata
from typing import List, Sequence, cast

import onnxruntime as ort
from sanic.log import logger

from api import CacheSetting, DropdownSetting, NodeContext, ToggleSetting
from gpu import amd, dxgi_adapter_index, nvidia
from system import is_arm_mac

from . import package


def offers_migraphx(providers: Sequence[str]) -> bool:
    """Whether the settings offer MIGraphX's choices (AMD GPUs): where ONNX Runtime has
    the MIGraphX provider, an AMD GPU is worth running models on (not only a Ryzen's
    small graphics) and no NVIDIA GPU is detected, whose choices and defaults stay as
    they are."""
    return (
        "MIGraphXExecutionProvider" in providers
        and amd.eligible
        and not nvidia.is_available
    )


def get_providers():
    providers = cast(List[str], ort.get_available_providers())

    default = providers[0]
    if "CUDAExecutionProvider" in providers:
        default = "CUDAExecutionProvider"
    elif offers_migraphx(providers) and "DmlExecutionProvider" in providers:
        # ONNX Runtime's Windows MIGraphX build: DirectML stays the default there.
        default = "DmlExecutionProvider"
    elif offers_migraphx(providers):
        default = "MIGraphXExecutionProvider"
    elif "CPUExecutionProvider" in providers:
        default = "CPUExecutionProvider"

    return providers, default


def get_provider_label(identifier: str) -> str:
    label = identifier.replace("ExecutionProvider", "")
    if label.lower() == "tensorrt":
        label = "TensorRT"
    return label


def _names_programs_by_shape() -> bool:
    """Whether the installed ONNX Runtime names each compiled MIGraphX program file by
    its input shape. Up to 1.30.0, the MIGraphX provider hashes only the first bytes of
    the shape, so a model run at a second shape loads the program of the first from
    the cache. chaiNNer-C's onnxruntime-migraphx build (local label rocmX.Y, as in
    1.30.0+rocm7.14) hashes the whole shape."""
    try:
        version = metadata.version("onnxruntime-migraphx")
    except metadata.PackageNotFoundError:
        return False
    return version.partition("+")[2].startswith("rocm")


execution_providers, default_provider = get_providers()
migraphx_offered = offers_migraphx(execution_providers)
can_cache_migraphx = migraphx_offered and _names_programs_by_shape()
# The GPU setting's default: CUDA's first device, or HIP's strongest AMD GPU (a Ryzen's
# graphics often come first in HIP's order)
default_gpu_index = amd.default_index if migraphx_offered else 0


def provider_device(execution_provider: str, gpu_index: int) -> int:
    """The device `execution_provider` takes for the GPU setting's `gpu_index`. Where
    MIGraphX is offered, the setting lists the AMD GPUs by HIP's index; DirectML (beside
    MIGraphX in ONNX Runtime's Windows MIGraphX build) numbers adapters in DXGI's order,
    so it takes the position of the GPU's adapter there. Elsewhere the index as it is."""
    if not (migraphx_offered and execution_provider == "DmlExecutionProvider"):
        return gpu_index
    device = next((d for d in amd.devices if d.index == gpu_index), None)
    if device is None or device.luid is None:
        raise RuntimeError(
            f"The ONNX GPU setting names AMD GPU {gpu_index}, which DirectML cannot"
            " find on this computer; choose another GPU in the ONNX settings."
        )
    return dxgi_adapter_index(device.luid)


if not is_arm_mac:
    # The index goes to the GPU provider: CUDA numbers the devices for CUDA and
    # TensorRT, HIP for MIGraphX.
    gpus = amd.devices if migraphx_offered else nvidia.devices
    package.add_setting(
        DropdownSetting(
            label="GPU",
            key="gpu_index",
            description="Which GPU to use for ONNX. This is only relevant if you have multiple GPUs.",
            options=[{"label": d.name, "value": str(d.index)} for d in gpus],
            default=str(default_gpu_index),
        )
    )

    package.add_setting(
        DropdownSetting(
            label="Execution Provider",
            key="execution_provider",
            description="What provider to use for ONNX.",
            options=[
                {"label": get_provider_label(x), "value": x}
                for x in execution_providers
            ],
            default=default_provider,
        )
    )

    package.add_setting(
        CacheSetting(
            label="Cache TensorRT Engines",
            key="onnx_tensorrt_cache",
            description="Whether to cache the converted TensorRT engines. This can significantly speed up subsequent runs.",
            directory="onnx_tensorrt_cache",
            disabled="TensorrtExecutionProvider" not in execution_providers,
        )
    )

    should_fp16 = nvidia.is_available and nvidia.all_support_fp16

    package.add_setting(
        ToggleSetting(
            label="Use TensorRT FP16 Mode",
            key="tensorrt_fp16_mode",
            description="No effect with TensorRT 11, which builds each engine in the model's own precision. For FP16 speed and lower VRAM use, convert the model to fp16 (Convert To ONNX).",
            default=should_fp16,
            disabled="TensorrtExecutionProvider" not in execution_providers,
        )
    )

    if migraphx_offered:
        package.add_setting(
            CacheSetting(
                label="Cache MIGraphX Programs",
                key="onnx_migraphx_cache",
                description="Whether to keep the programs MIGraphX compiles. MIGraphX compiles a model for each input size it runs at (each tile size, and the smaller tiles at an image's edges); with the cache, later runs load a size compiled before.",
                directory="onnx_migraphx_cache",
                disabled=not can_cache_migraphx,
            )
        )

        package.add_setting(
            ToggleSetting(
                label="Use MIGraphX FP16 Mode",
                key="migraphx_fp16_mode",
                description="Runs FP32 models in FP16 on MIGraphX: faster and less VRAM, with slightly different results. FP16 models run in FP16 either way.",
                default=amd.all_support_fp16,
            )
        )

        package.add_setting(
            ToggleSetting(
                label="MIGraphX Exhaustive Tuning",
                key="migraphx_exhaustive_tune",
                description="Times every kernel candidate when MIGraphX compiles a model: compiling takes much longer, and the model may run faster. Best with the program cache on.",
                default=False,
            )
        )


@dataclass(frozen=True)
class OnnxSettings:
    gpu_index: int
    """The device index the execution provider takes (provider_device)."""
    execution_provider: str
    tensorrt_cache_path: str | None
    tensorrt_fp16_mode: bool
    migraphx_cache_path: str | None
    migraphx_fp16_mode: bool
    migraphx_exhaustive_tune: bool


def get_settings(context: NodeContext) -> OnnxSettings:
    settings = context.settings

    tensorrt_cache_path = settings.get_cache_location("onnx_tensorrt_cache")
    logger.info(f"TensorRT cache location: {tensorrt_cache_path}")
    if tensorrt_cache_path and not os.path.exists(tensorrt_cache_path):
        os.makedirs(tensorrt_cache_path)

    migraphx_cache_path = None
    if can_cache_migraphx:
        migraphx_cache_path = settings.get_cache_location("onnx_migraphx_cache")
        logger.info(f"MIGraphX cache location: {migraphx_cache_path}")

    execution_provider = settings.get_str("execution_provider", default_provider)
    return OnnxSettings(
        gpu_index=provider_device(
            execution_provider, settings.get_int("gpu_index", 0, parse_str=True)
        ),
        execution_provider=execution_provider,
        tensorrt_cache_path=tensorrt_cache_path,
        tensorrt_fp16_mode=settings.get_bool("tensorrt_fp16_mode", False),
        migraphx_cache_path=migraphx_cache_path,
        migraphx_fp16_mode=settings.get_bool("migraphx_fp16_mode", False),
        migraphx_exhaustive_tune=settings.get_bool("migraphx_exhaustive_tune", False),
    )
