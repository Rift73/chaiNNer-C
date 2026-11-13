"""The PyTorch executor of Upscale Image's temporal path (`temporal_window.Executor`):
RDUAL's windows through spandrel's `run_window`, eager, in its precision (BF16 autocast
on CUDA)."""

from __future__ import annotations

import gc
import inspect
from collections.abc import Callable, Hashable, Sequence
from typing import TypedDict, Unpack

import numpy as np
import torch
from spandrel import ImageModelDescriptor

from ..upscale.temporal_window import LOOK_AHEAD
from .utils import safe_cuda_cache_empty


def is_temporal(model: object) -> bool:
    """Whether Upscale Image runs this model on its temporal path."""
    return isinstance(model, ImageModelDescriptor) and model.architecture.id == "RDUAL"


def context_applies(model: object) -> bool:
    """Upscale Image's gate (`temporal_if` on its model input): the context items apply
    to a temporal model, and to a value that is no model descriptor, which it cannot
    judge."""
    return not isinstance(model, ImageModelDescriptor) or is_temporal(model)


def release_memory() -> None:
    """What a window that ran out of memory left cached, released as Upscale Image's
    tiling releases it before it splits a tile."""
    gc.collect()
    safe_cuda_cache_empty()


class MemoryOptions(TypedDict, total=False):
    """run_window's memory options; an option left out keeps its default and is not
    passed, so a spandrel whose run_window lacks it never gets it."""

    flow_volume: bool
    hold_encodes: bool


class TileOptions(TypedDict, total=False):
    """run_window's tiles, passed only for a tiled window."""

    tile: tuple[int, int]
    overlap: int


# run_window's lighter memory modes, each lighter than the one before: the flow's
# lookups computed on demand instead of from its all-pairs volume, then also the kept
# frames' encodes recomputed in the forward sweep instead of held. Each keeps the frame
# and the window and costs a few percent of time.
LIGHTER_MODES: tuple[tuple[str, MemoryOptions], ...] = (
    ("the flow's lookups computed on demand", {"flow_volume": False}),
    (
        "the flow's lookups on demand and the encodes recomputed",
        {"flow_volume": False, "hold_encodes": False},
    ),
)


def takes_tiles() -> bool:
    """Whether this spandrel's run_window runs a window in tiles (`with_tile`)."""
    from spandrel.architectures.RDUAL import run_window

    return "tile" in inspect.signature(run_window).parameters


def lighter_modes() -> list[tuple[str, Callable[[RDUALWindows], RDUALWindows]]]:
    """The LIGHTER_MODES that this spandrel's run_window takes, for `Fit.lighter`."""
    from spandrel.architectures.RDUAL import run_window

    taken = inspect.signature(run_window).parameters
    return [
        (name, lambda windows, options=options: windows.with_options(**options))
        for name, options in LIGHTER_MODES
        if all(option in taken for option in options)
    ]


MIB = 2**20
# The memory one window adds, in MiB per megapixel of its padded frames (a 2x model),
# measured in steady state at 1280x720, 1488x837 and 1920x1080: the sweeps take
# SWEEP, plus STATE for each frame ahead past two (its reverse state) and, while the
# kept frames' encodes are held, ENCODE for each kept frame past two; the flow with its
# all-pairs volume takes the volume itself plus FLOW. The window's peak is the larger.
SWEEP, STATE, ENCODE, FLOW = 2100, 52, 197, 700
# A window with no frames ahead (no reverse sweep) grows slower above 1488x837 (its
# output stage runs in bands): SWEEP up to PAST_ONLY_KNEE megapixels, then
# PAST_ONLY_SLOPE MiB for each further megapixel (measured 3025 MiB at 1920x1080).
PAST_ONLY_KNEE, PAST_ONLY_SLOPE = 1.262, 450
# What the estimate leaves free, on top of it. A process's memory cap counts the
# allocator's reserved memory, which a window takes up to 19 % above what it allocates
# (5.08 against 4.27 GiB at 1920x1080), and the estimate falls up to 1.2 % below the
# measured peaks: with 5 % and 512 MiB a 1920x1080 window two frames ahead still ran out
# of memory under a 5.5 GiB cap.
MARGIN, MARGIN_MIB = 0.20, 256


# Input pixels run_window's tiles share with each neighbour; the model places tiles on
# its grid of TILE_GRID pixels, so a tile can come out up to that much larger.
TILE_OVERLAP, TILE_GRID = 64, 128


def window_peak(
    height: int,
    width: int,
    ahead: int,
    *,
    flow_volume: bool,
    hold_encodes: bool,
    tile: tuple[int, int] | None = None,
) -> int:
    """The bytes a window with `ahead` frames ahead adds to what is allocated, for
    frames of height x width: its frames, and the larger of its sweeps and its flow.
    With a `tile` (height, width), the flow is the whole frame's and the sweeps are one
    tile's (each tile's sweeps run after the last's are freed)."""
    padded_h, padded_w = -(-height // 16) * 16, -(-width // 16) * 16
    trunk_h, trunk_w = padded_h, padded_w
    if tile is not None:
        trunk_h = min(padded_h, tile[0] + TILE_GRID)
        trunk_w = min(padded_w, tile[1] + TILE_GRID)
    megapixels = trunk_h * trunk_w / 1e6
    keep = max(1, ahead + 1 - LOOK_AHEAD)
    sweeps = megapixels * (
        SWEEP
        + STATE * max(0, ahead - 2)
        + (ENCODE * max(0, keep - 2) if hold_encodes else 0)
    )
    if ahead == 0:
        knee = SWEEP * PAST_ONLY_KNEE
        sweeps = min(sweeps, knee + PAST_ONLY_SLOPE * (megapixels - PAST_ONLY_KNEE))
    flow = 0.0
    if flow_volume:
        cells = (padded_h // 8) * (padded_w // 8)
        flow = 2 * cells * cells * 4 / MIB + FLOW * padded_h * padded_w / 1e6
    # The window's frames and the frame before it, RGB float32.
    frames = (ahead + 2) * 12 * padded_h * padded_w / MIB
    return int((max(sweeps, flow) + frames) * MIB)


class RDUALWindows:
    """Frames are RGB CHW float32 tensors on the device, outputs RGB CHW tensors left on
    the device until their items take them (a warm-up item's never reaches the host),
    and the state is the model's carried trunk state."""

    def __init__(
        self,
        model: ImageModelDescriptor,
        device: torch.device,
        budget: int = 0,
        tile: tuple[int, int] | None = None,
        **options: Unpack[MemoryOptions],
    ):
        self.model = model
        self.network = model.model
        self.device = device
        # The PyTorch settings' memory budget in bytes; 0: none.
        self.budget = budget
        # run_window's tiles (height, width), or None for the whole frame.
        self.tile = tile
        # run_window's memory options (LIGHTER_MODES); none: its defaults.
        self.options = options
        if next(self.network.parameters()).device != device:
            self.network.to(device)

    def with_options(self, **options: Unpack[MemoryOptions]) -> RDUALWindows:
        """The same model, device and tiles with these run_window memory options. Its
        key is this one's: the options change no state, so a window carries it on."""
        return RDUALWindows(self.model, self.device, self.budget, self.tile, **options)

    def with_tile(self, tile: tuple[int, int]) -> RDUALWindows:
        """The same with run_window's tiles of `tile` (width, height, as Tile Size gives
        it): the flow and its cues on the whole frame, the trunk per tile. Its key is
        its own: the state holds one state per tile."""
        return RDUALWindows(
            self.model, self.device, self.budget, (tile[1], tile[0]), **self.options
        )

    def fitting_ahead(self, shape: tuple[int, int], ahead: int) -> int:
        """The frames ahead (at most `ahead`) whose window fits the device's free memory
        by `window_peak`, with its margin, or -1 when not even a window of one frame
        fits; on a CPU, `ahead`."""
        if self.device.type != "cuda":
            return ahead
        # The settings' device may name no index: inside it, the calls take its index.
        with torch.cuda.device(self.device):
            free, total = torch.cuda.mem_get_info()
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            fraction = torch.cuda.get_per_process_memory_fraction()
        available = free + reserved - allocated
        available = min(available, int(fraction * total) - allocated)
        if self.budget > 0:
            available = min(available, self.budget - allocated)
        flow_volume = self.options.get("flow_volume", True)
        hold_encodes = self.options.get("hold_encodes", True)
        for frames in range(ahead, -1, -1):
            peak = window_peak(
                *shape,
                frames,
                flow_volume=flow_volume,
                hold_encodes=hold_encodes,
                tile=self.tile,
            )
            if peak * (1 + MARGIN) + MARGIN_MIB * MIB <= available:
                return frames
        return -1

    @property
    def key(self) -> Hashable:
        return ("pytorch", id(self.network), str(self.device), self.tile)

    def convert(self, frame: np.ndarray) -> torch.Tensor:
        # BGR HWC -> RGB CHW in 0..1 on the model's device
        tensor = torch.from_numpy(np.ascontiguousarray(frame)).to(self.device)
        return tensor.float().clamp(0, 1).permute(2, 0, 1).flip(0)

    def run(
        self,
        frames: Sequence[torch.Tensor],
        keep: int,
        past: tuple[torch.Tensor, torch.Tensor] | None,
        _first: int,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        # Every frame is computed in the window that reads it: the index is not needed.
        # spandrel before 0.4.2+c20 has no run_window: imported here, so that Upscale Image
        # loads with it and only an RDUAL model needs the newer spandrel.
        from spandrel.architectures.RDUAL import run_window

        tiles: TileOptions = (
            {} if self.tile is None else {"tile": self.tile, "overlap": TILE_OVERLAP}
        )
        with torch.inference_mode():
            outputs, state = run_window(
                self.network,
                torch.stack(list(frames))[None],
                keep,
                None if past is None else (past[0], past[1][None]),
                **self.options,
                **tiles,
            )
        return list(outputs[0]), state

    @staticmethod
    def to_image(output: torch.Tensor) -> np.ndarray:
        """RGB CHW -> BGR HWC float32 in 0..1."""
        return np.ascontiguousarray(
            output.clamp(0, 1).flip(0).permute(1, 2, 0).float().cpu().numpy()
        )
