"""PyTorch's CUDA allocator settings for the worker (2026-10-24). PyTorch reads them when
it first allocates, so the worker (server.py) calls apply() at its start, before any
module imports torch; its child processes inherit them.

PyTorch reads the first of PYTORCH_CUDA_ALLOC_CONF, PYTORCH_HIP_ALLOC_CONF and
PYTORCH_ALLOC_CONF that is set, so apply() writes PYTORCH_CUDA_ALLOC_CONF: the options
of the first one the user set, without expandable_segments, and the VRAM offload's
pinned RAM options (nodes/impl/pytorch/offload.py) the user did not set.

expandable_segments is left out: in PyTorch 2.14 a mapping that runs out of memory
after its handles were created (cuMemSetAccess on Windows) keeps them, and from then on
every allocation in that range fails an internal assert (pytorch/pytorch#188008) until
the worker restarts. Upscale Image's tiling meets out-of-memory on purpose (it splits
the tile and retries), so with the option set one such tile ended the GPU's work for
every model until the app restarted.
"""

from __future__ import annotations

import os

from sanic.log import logger

# In the order PyTorch reads them
_VARIABLES = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
_LEFT_OUT = "expandable_segments"
# The VRAM offload's pinned RAM page-locked by 8 threads: a first overflow's 15-18 GiB
# took about 1 s less.
_PINNED = ("pinned_use_cuda_host_register:True", "pinned_num_register_threads:8")


def _options(config: str) -> list[str]:
    """config's comma-separated options; a bracketed value (roundup_power2_divisions's
    [256:1,>:2]) keeps its commas."""
    options: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(config):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "," and depth == 0:
            options.append(config[start:i])
            start = i + 1
    options.append(config[start:])
    return [option.strip() for option in options if option.strip()]


def _key(option: str) -> str:
    # PyTorch drops all whitespace, inside a key too
    return "".join(option.split(":", 1)[0].split())


def allocator_config(user_config: str) -> tuple[str, list[str]]:
    """The worker's PYTORCH_CUDA_ALLOC_CONF for the user's, and the user's options it
    leaves out."""
    options = _options(user_config)
    left_out = [option for option in options if _key(option) == _LEFT_OUT]
    kept = [option for option in options if _key(option) != _LEFT_OUT]
    keys = {_key(option) for option in kept}
    kept += [option for option in _PINNED if _key(option) not in keys]
    return ",".join(kept), left_out


def apply() -> None:
    variable = next((name for name in _VARIABLES if name in os.environ), None)
    config, left_out = allocator_config(os.environ[variable] if variable else "")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = config
    if left_out:
        logger.warning(
            f"{variable}'s {', '.join(left_out)} is not used: with expandable segments,"
            " one out-of-memory error can stop all PyTorch GPU work until the app"
            " restarts (pytorch/pytorch#188008)."
        )
