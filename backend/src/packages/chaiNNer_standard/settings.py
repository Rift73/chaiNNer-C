from __future__ import annotations

import psutil

from api import NumberSetting, ToggleSetting
from nodes.impl.result_cache import ENABLED, SIZE_MIB, default_size_mib

from . import package

package.add_setting(
    ToggleSetting(
        label="Result Cache",
        key=ENABLED,
        description=(
            "Keeps the outputs of nodes that always give the same result for the same"
            " inputs (Load Image, filters, resizes and the like) in memory between runs,"
            " so running a chain again computes only what changed. A change applies at"
            " the next run; turning it off empties the cache."
        ),
        default=True,
    )
)

package.add_setting(
    NumberSetting(
        label="Result Cache Size (MiB)",
        key=SIZE_MIB,
        description=(
            "How much memory the Result Cache may use. When it is full, the least"
            " recently used outputs go first, and an output larger than an eighth of"
            " this size is not kept. Default: the smaller of 2 GiB and 10 % of this"
            f" computer's RAM ({default_size_mib()} MiB here). A smaller size applies"
            " at the next run."
        ),
        default=default_size_mib(),
        min=1,
        max=psutil.virtual_memory().total >> 20,
    )
)
