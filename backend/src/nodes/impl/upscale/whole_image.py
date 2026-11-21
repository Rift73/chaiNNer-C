"""The architectures whose output changes when an image is cut into tiles, for every
Upscale Image that would tile one: PyTorch's (and its engine route) and ONNX's.

Imports neither PyTorch nor ONNX, so that each package's node reads the one set."""

from __future__ import annotations

from functools import partial
from threading import Lock
from weakref import WeakSet

from api import NodeContext

# Their context reaches past a tile's overlap. WSR's context gates and attention: a
# trained 2x WSR Compact on an 800x800 image read RMSE 2.1e-3 in 256 px tiles against the
# whole image (32 codes at worst, everywhere in the image, not only at the seams), and
# RMSE 3.2e-4 in 512 px tiles (9 codes); at 3840x2160 in 1920 px tiles RMSE 1.64e-3
# against PyTorch FP32 on the whole frame (0.12 at worst), 4.1 times PyTorch FP16's own
# error.
WHOLE_IMAGE_ONLY = frozenset({"WSR"})

# The models of WHOLE_IMAGE_ONLY whose tiles this chain has warned of
_warned: WeakSet[object] = WeakSet()
_lock = Lock()


def warn_tiled(context: NodeContext, model: object, message: str) -> None:
    """`message`, that a model of WHOLE_IMAGE_ONLY ran in tiles, once per chain for
    `model`."""
    with _lock:
        if model in _warned:
            return
        _warned.add(model)
    context.add_cleanup(partial(_forget, model), after="chain")
    context.warn(message)


def _forget(model: object) -> None:
    with _lock:
        _warned.discard(model)
