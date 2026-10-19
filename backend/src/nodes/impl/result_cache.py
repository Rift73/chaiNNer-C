"""The Result Cache settings: the native engine's memo cache (native/ARCHITECTURE.md section 4), which the standard
package declares (packages/chaiNNer_standard/settings.py) and native_executor applies at each run's start, where a
smaller size trims the cache and off empties it. CHAINNER_C_MEMO_BUDGET_MIB (native/src/run/executor.cpp's Memo), a
diagnostic override, wins over both settings."""

from __future__ import annotations

import psutil

from api import ExecutionOptions, SettingsJson

PACKAGE = "chaiNNer_standard"
ENABLED = "memo_cache"
SIZE_MIB = "memo_cache_mib"


def default_size_mib() -> int:
    """The engine's own budget, in MiB: the smaller of 2 GiB and 10 % of the RAM."""
    return min(2048, psutil.virtual_memory().total // 10 >> 20)


def cache_budget(options: ExecutionOptions) -> int:
    """The memo budget in bytes the run's settings ask for; 0 when the cache is off."""
    settings = options.get_package_settings(PACKAGE)
    if not settings.get_bool(ENABLED, True):
        return 0
    return max(0, settings.get_int(SIZE_MIB, default_size_mib(), parse_str=True)) << 20


def keyed_settings(package_id: str, settings: SettingsJson) -> SettingsJson:
    """A node's package settings as its memo key holds them: without the Result Cache's own, which never change a
    node's outputs, so another size keeps the cached entries' keys."""
    if package_id != PACKAGE:
        return settings
    return {
        key: value for key, value in settings.items() if key not in (ENABLED, SIZE_MIB)
    }
