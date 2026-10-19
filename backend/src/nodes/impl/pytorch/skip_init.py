"""Spandrel builds a model's architecture with randomly initialised weights, then loads
the state dict into it strictly (ModelDescriptor.__init__), which overwrites every
parameter and persistent buffer. Inside `skip_random_init`, torch.nn.init's random
initialisers leave their tensors as they are, which the load then fills: the model is
bit-identical (test_skip_init.py) and builds faster: on the CPU, DRFT-L in 249 ms
instead of 727, HAT-L 176 instead of 308, DAT2 115 instead of 149.

Only the thread inside the block skips them: another thread building or initialising
a model at the same time is unaffected, and builds need no lock. The deterministic
initialisers (zeros_, ones_, constant_, eye_, dirac_) still run, as a buffer the
state dict does not hold may depend on them, and they cost little. The wrappers are
installed when this module is imported and call the original outside the block."""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch

_RANDOM_INITIALISERS = (
    "uniform_",
    "normal_",
    "trunc_normal_",
    "xavier_uniform_",
    "xavier_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
    "orthogonal_",
    "sparse_",
)

# Marks this module's wrappers: a module imported again (vsview's Reload Script)
# replaces them instead of wrapping them once more.
_MARKER = "_chainner_skip_random_init"

_skipping = threading.local()


def _unless_skipping(
    initialise: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    @functools.wraps(initialise)
    def wrapper(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        if getattr(_skipping, "active", False):
            return tensor
        return initialise(tensor, *args, **kwargs)

    setattr(wrapper, _MARKER, True)
    return wrapper


for _name in _RANDOM_INITIALISERS:
    _initialise = getattr(torch.nn.init, _name)
    if getattr(_initialise, _MARKER, False):
        _initialise = _initialise.__wrapped__
    setattr(torch.nn.init, _name, _unless_skipping(_initialise))


@contextmanager
def skip_random_init() -> Iterator[None]:
    """This thread's calls of torch.nn.init's random initialisers do nothing within the
    block. Only for building a model whose every parameter and persistent buffer is
    loaded afterwards."""
    previous = getattr(_skipping, "active", False)
    _skipping.active = True
    try:
        yield
    finally:
        _skipping.active = previous
