from __future__ import annotations

import os
from dataclasses import dataclass

from sanic.log import logger

from api import CacheSetting, DropdownSetting, NodeContext
from gpu import nvidia

from . import package

package.add_setting(
    DropdownSetting(
        label="GPU",
        key="gpu_index",
        description="Which GPU to use for TensorRT. This is only relevant if you have multiple GPUs.",
        options=[{"label": d.name, "value": str(d.index)} for d in nvidia.devices],
        default="0",
    )
)

package.add_setting(
    CacheSetting(
        label="Timing Cache",
        key="timing_cache",
        description="Directory for TensorRT timing cache. Speeds up engine building for similar models.",
        directory="tensorrt_timing_cache",
    )
)

package.add_setting(
    CacheSetting(
        label="Engine Cache",
        key="engine_cache",
        description="Directory where Build Engine keeps the engines it builds, so that a build of the same model with the same settings, GPU and TensorRT version reads its engine instead of building it again, also after a restart. Up to 16 GiB; the least recently used engines are removed first.",
        directory="tensorrt_engine_cache",
    )
)


@dataclass(frozen=True)
class TensorRTSettings:
    gpu_index: int
    timing_cache_path: str | None
    engine_cache_path: str | None


def get_settings(context: NodeContext) -> TensorRTSettings:
    settings = context.settings

    timing_cache_path = settings.get_cache_location("timing_cache")
    if timing_cache_path and not os.path.exists(timing_cache_path):
        os.makedirs(timing_cache_path)
        logger.info("Created TensorRT timing cache at: %s", timing_cache_path)
    engine_cache_path = settings.get_cache_location("engine_cache")
    if engine_cache_path and not os.path.exists(engine_cache_path):
        os.makedirs(engine_cache_path)
        logger.info("Created TensorRT engine cache at: %s", engine_cache_path)

    return TensorRTSettings(
        gpu_index=settings.get_int("gpu_index", 0, parse_str=True),
        timing_cache_path=timing_cache_path,
        engine_cache_path=engine_cache_path,
    )
