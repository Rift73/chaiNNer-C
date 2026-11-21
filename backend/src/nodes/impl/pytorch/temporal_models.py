"""The temporal models Upscale Image runs on its temporal path (`temporal_window`), and
the PyTorch executor it runs each with."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any

from spandrel import ImageModelDescriptor

from ..upscale.temporal_window import LOOK_AHEAD, WARM_UP
from .rdual_window import RDUALWindows, lighter_modes, takes_tiles
from .rsr_window import RSRWindows
from .tfdat_window import TFDATWindows

if TYPE_CHECKING:
    from packages.chaiNNer_pytorch.settings import PyTorchSettings

# spandrel's ids of the temporal architectures.
TEMPORAL_ARCHITECTURES = ("RDUAL", "RSR", "TFDAT", "TFDATv2")
# Those that read a clip centred on each frame and carry nothing (tfdat_window.py).
TFDAT_ARCHITECTURES = ("TFDAT", "TFDATv2")


# Here, not in rdual_engine.py, which imports auto_engine: the TensorRT package's Upscale
# Image (which auto_engine imports for its engine route) reads it.
def runtime(architecture: str) -> tuple[ModuleType, ModuleType] | None:
    """A temporal architecture's engine sets (`CHECKED_SCALES`, `engine_set`) and their
    session (`temporal_engine_upscale`), or None for another architecture."""
    from ..tensorrt import rdual, rdual_session, rsr, rsr_session, tfdat, tfdat_session

    return {
        "RDUAL": (rdual, rdual_session),
        "RSR": (rsr, rsr_session),
        "TFDAT": (tfdat, tfdat_session),
        "TFDATv2": (tfdat, tfdat_session),
    }.get(architecture)


def is_temporal(model: object) -> bool:
    """Whether Upscale Image runs this model on its temporal path."""
    return (
        isinstance(model, ImageModelDescriptor)
        and model.architecture.id in TEMPORAL_ARCHITECTURES
    )


def context_applies(model: object) -> bool:
    """Upscale Image's gate (`temporal_if` on its model input): the context items apply
    to a temporal model, and to a value that is no model descriptor, which it cannot
    judge."""
    return not isinstance(model, ImageModelDescriptor) or is_temporal(model)


def context_of(model: object) -> tuple[int, int]:
    """Upscale Image's context items (before, after) for a model the loop knows before
    its first item (`temporal_context_of`): TFDAT's and TFDATv2's clip reads half of it
    on each side of its centre, so a Range or Run this frame gets the frames the whole
    video gives it; every other value keeps (WARM_UP, LOOK_AHEAD), RDUAL's and RSR's."""
    if (
        isinstance(model, ImageModelDescriptor)
        and model.architecture.id in TFDAT_ARCHITECTURES
    ):
        side = model.model.clip_size // 2
        return side, side
    return WARM_UP, LOOK_AHEAD


def pytorch_windows(
    model: ImageModelDescriptor, settings: PyTorchSettings, budget: int
) -> tuple[
    Callable[[], RDUALWindows | RSRWindows | TFDATWindows],
    Sequence[tuple[str, Callable[[Any], Any]]],
    bool,
]:
    """A temporal model's PyTorch executor, made by the function only where it runs (it
    puts the model on the settings' device), with its lighter memory modes and whether
    it runs tiles (`Fit.lighter`, `Fit.tiles`). `budget` is the settings' memory budget
    in bytes (0: none).

    RSR has neither lighter modes nor tiles: only a shorter window fits it. It runs in
    FP16 as the node runs any model: when the settings' FP16 is on (they turn it off on
    the CPU) and the model supports it; else in FP32. TFDAT and TFDATv2 run as RSR, but
    without a memory estimate: each frame is one call on its clip, whatever the window,
    and a tile would change what their optical flow and attention see."""
    device = settings.device
    if model.architecture.id == "RSR":
        use_fp16 = settings.use_fp16 and model.supports_half
        return lambda: RSRWindows(model, device, budget, use_fp16), (), False
    if model.architecture.id in TFDAT_ARCHITECTURES:
        use_fp16 = settings.use_fp16 and model.supports_half
        return lambda: TFDATWindows(model, device, use_fp16), (), False
    return lambda: RDUALWindows(model, device, budget), lighter_modes(), takes_tiles()
