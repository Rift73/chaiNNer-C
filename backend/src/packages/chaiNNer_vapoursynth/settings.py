from __future__ import annotations

from dataclasses import dataclass

from api import NodeContext, NumberSetting

from . import package

package.add_setting(
    NumberSetting(
        label="VapourSynth cache (MB)",
        key="cache_mb",
        description=(
            "How much memory each VapourSynth process (a preview, frames for chaiNNer,"
            " a script run) may use to cache frames. More is faster for scripts that"
            " reuse frames; less leaves more memory for everything else."
        ),
        default=4096,
        min=64,
        max=1024**2,
    )
)


@dataclass(frozen=True)
class VapourSynthSettings:
    cache_mb: int


def get_settings(context: NodeContext) -> VapourSynthSettings:
    settings = context.settings

    return VapourSynthSettings(
        cache_mb=settings.get_int("cache_mb", 4096, parse_str=True),
    )
