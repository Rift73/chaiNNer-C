from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

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
        label="DUAL Plugin Cache",
        key="dual_plugin_cache",
        description="Directory for DUAL's compiled TensorRT plugins (one per engine size and GPU). Without it they go to a temporary folder.",
        directory="tensorrt_dual_plugins",
    )
)


@dataclass(frozen=True)
class TensorRTSettings:
    gpu_index: int
    timing_cache_path: str | None
    dual_plugin_cache: Path


def get_settings(context: NodeContext) -> TensorRTSettings:
    settings = context.settings

    timing_cache_path = settings.get_cache_location("timing_cache")
    if timing_cache_path and not os.path.exists(timing_cache_path):
        os.makedirs(timing_cache_path)
        logger.info("Created TensorRT timing cache at: %s", timing_cache_path)

    dual_plugin_cache = settings.get_cache_location("dual_plugin_cache")
    return TensorRTSettings(
        gpu_index=settings.get_int("gpu_index", 0, parse_str=True),
        timing_cache_path=timing_cache_path,
        dual_plugin_cache=Path(dual_plugin_cache or tempfile.gettempdir()).joinpath(
            "chainner-c-dual-tensorrt"
        ),
    )
