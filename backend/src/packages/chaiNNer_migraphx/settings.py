from __future__ import annotations

import os
from dataclasses import dataclass

from sanic.log import logger

from api import CacheSetting, DropdownSetting, NodeContext
from gpu import amd

from . import devices, package

# The strongest GPU (gpu.amd.default_index) when MIGraphX runs on it, else the first listed
default_index = (
    amd.default_index
    if any(d.index == amd.default_index for d in devices)
    else devices[0].index
    if devices
    else 0
)

# Only where the package runs: elsewhere it would add a MIGraphX tab to the settings of
# every computer without an AMD GPU.
if not package.disabled:
    package.add_setting(
        DropdownSetting(
            label="GPU",
            key="gpu_index",
            description="Which AMD GPU to use for MIGraphX. This is only relevant if you have multiple GPUs.",
            options=[
                {"label": f"{d.name} ({d.gfx})", "value": str(d.index)} for d in devices
            ],
            default=str(default_index),
        )
    )

    package.add_setting(
        CacheSetting(
            label="Program Cache",
            key="program_cache",
            description="Directory where Build Program keeps the programs it compiles, so that a build of the same model with the same settings, GPU and MIGraphX reads its program instead of compiling it again, also after a restart. Up to 16 GiB; the least recently used programs are removed first.",
            directory="migraphx_program_cache",
        )
    )


@dataclass(frozen=True)
class MIGraphXSettings:
    gpu_index: int
    """HIP's index of the GPU, which MIGraphX takes."""
    gpu_name: str
    gfx: str
    supports_fp8: bool
    program_cache_path: str | None


def get_settings(context: NodeContext) -> MIGraphXSettings:
    settings = context.settings

    program_cache_path = settings.get_cache_location("program_cache")
    if program_cache_path and not os.path.exists(program_cache_path):
        os.makedirs(program_cache_path)
        logger.info("Created MIGraphX program cache at: %s", program_cache_path)

    gpu_index = settings.get_int("gpu_index", 0, parse_str=True)
    device = next((d for d in devices if d.index == gpu_index), None)
    if device is None:
        raise RuntimeError(
            f"The MIGraphX GPU setting names GPU {gpu_index}, which is not one of this"
            " computer's AMD GPUs that ROCm supports; choose another in the MIGraphX"
            " settings."
        )
    return MIGraphXSettings(
        gpu_index=gpu_index,
        gpu_name=device.name,
        gfx=device.gfx,
        supports_fp8=device.supports_fp8,
        program_cache_path=program_cache_path,
    )
