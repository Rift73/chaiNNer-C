"""Temporal Fix split into the work of one frame and the work of one window: the parts
as PyTorch modules, which both the PyTorch route (temporal_fix_eager.py) and the export
of the engine set's two parts (tensorrt/temporal_fix_split.py) run. No ONNX or TensorRT
here: the PyTorch package installs neither.

A window's frames are each encoded (Head: four feature channels, upsampled to the frame's
size) and low-passed and resized to the aligner's three stage sizes, frames and features
alike (_resize_like); nothing else in the network depends on one frame only, and the
full-size features reach nothing but those resizes. The confidence head's half-size centre,
centre features and support features are the same resizes at the largest stage size.

`frame_part` maps one frame to its six pyramid levels: the frame at the three stage sizes
(3 channels each), then its features at the three (4 channels each), smallest first.
`window_part` runs the network's own forward on the window's frames with those levels
cached: inside `cached_pyramids` the encoder gives a placeholder that nothing reads, the
aligner's two pyramid builders hand out the cached levels, and the three half-size resizes
of the forward return theirs. Each level enters as a (7, C, h, w) stack in window order.
The vendored network is used as it is; its module and instances are patched only while the
context lasts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn

from vendor.temporalfix import temporalfix_arch as arch

# The vendored module's helpers this split reuses and replaces; it keeps them private
HELPERS = vars(arch)
RESIZE, SCALED, EXPAND = "_resize_like", "_scaled_hw", "_expand_scalar_map"
CENTRE_PYRAMIDS, SUPPORT_PYRAMIDS = "_build_center_pyramids", "_build_support_pyramids"

FRAMES = 7
CENTRE = FRAMES // 2
SUPPORTS = (0, 1, 2, 4, 5, 6)
LEVELS = 3


def stage_sizes(
    model: arch.temporalfix_arch, hw: tuple[int, int]
) -> list[tuple[int, int]]:
    return [HELPERS[SCALED](hw, s) for s in model.aligner.scale_list]


class FramePart(nn.Module):
    """One frame (1, 3, H, W) -> its frame and feature levels at the stage sizes."""

    def __init__(self, model: arch.temporalfix_arch) -> None:
        super().__init__()
        self.model = model

    def forward(self, frame: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hw = (int(frame.shape[-2]), int(frame.shape[-1]))
        features = self.model.encode(frame)
        sizes = stage_sizes(self.model, hw)
        frames = [HELPERS[RESIZE](frame, size) for size in sizes]
        return (*frames, *[HELPERS[RESIZE](features, size) for size in sizes])


@contextmanager
def cached_pyramids(
    model: arch.temporalfix_arch, levels: list[torch.Tensor]
) -> Iterator[None]:
    """`model`'s forward on a window whose six levels (frames then features, each
    (7, C, h, w)) are `levels`, while the context lasts."""
    frames, features = levels[:LEVELS], levels[LEVELS:]
    aligner = model.aligner
    supports = list(SUPPORTS)
    half = model.confidence_scale

    def encode(x: torch.Tensor) -> torch.Tensor:
        return x[:, :1].expand(-1, 4, -1, -1)

    def centre_pyramids(
        center: torch.Tensor, f_center: torch.Tensor
    ) -> tuple[list[tuple[int, int]], list[torch.Tensor], list[torch.Tensor]]:
        hw = (int(center.shape[-2]), int(center.shape[-1]))
        return (
            stage_sizes(model, hw),
            [level[CENTRE : CENTRE + 1] for level in frames],
            [level[CENTRE : CENTRE + 1] for level in features],
        )

    def support_pyramids(
        support: torch.Tensor,
        f_support: torch.Tensor,
        delta_t: torch.Tensor,
        stage_hws: list[tuple[int, int]],
    ) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
    ]:
        b, s, _, h, w = support.shape
        delta_t_flat = delta_t.reshape(b * s, 1, h, w)
        support_pyr, f_support_pyr, delta_t_pyr, support_feat_pyr = [], [], [], []
        for hw_i, frame_level, feature_level in zip(stage_hws, frames, features):
            support_i = frame_level[supports].reshape(b, s, 3, hw_i[0], hw_i[1])
            feat_i = feature_level[supports].reshape(b, s, 4, hw_i[0], hw_i[1])
            delta_i = HELPERS[EXPAND](delta_t_flat, hw_i)
            support_pyr.append(support_i)
            f_support_pyr.append(feat_i)
            delta_t_pyr.append(delta_i.reshape(b, s, 1, hw_i[0], hw_i[1]))
            support_feat_pyr.append(
                torch.cat((support_i, feat_i), dim=2).reshape(
                    b * s, 7, hw_i[0], hw_i[1]
                )
            )
        return support_pyr, f_support_pyr, delta_t_pyr, support_feat_pyr

    resize = HELPERS[RESIZE]

    def resize_like(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """The forward's half-size centre, centre features and support features are
        cached levels; its other resizes (the warped frames, the raw confidence) run."""
        full = x.shape[-2:] != size and (
            HELPERS[SCALED]((int(x.shape[-2]), int(x.shape[-1])), half) == tuple(size)
        )
        cached = {(1, 3): frames, (1, 4): features, (len(SUPPORTS), 4): features}
        stack = cached.get((int(x.shape[0]), int(x.shape[1]))) if full else None
        if stack is None:
            return resize(x, size)
        level = next(level for level in stack if tuple(level.shape[-2:]) == tuple(size))
        return level[CENTRE : CENTRE + 1] if x.shape[0] == 1 else level[supports]

    model.encode.forward = encode
    setattr(aligner, CENTRE_PYRAMIDS, centre_pyramids)
    setattr(aligner, SUPPORT_PYRAMIDS, support_pyramids)
    setattr(arch, RESIZE, resize_like)
    try:
        yield
    finally:
        setattr(arch, RESIZE, resize)
        del model.encode.forward
        delattr(aligner, CENTRE_PYRAMIDS)
        delattr(aligner, SUPPORT_PYRAMIDS)


class WindowPart(nn.Module):
    """A window (1, 7, 3, H, W) and its frames' cached levels -> the network's output."""

    def __init__(self, model: arch.temporalfix_arch) -> None:
        super().__init__()
        self.model = model

    def forward(self, frames: torch.Tensor, *levels: torch.Tensor) -> torch.Tensor:
        with cached_pyramids(self.model, list(levels)):
            return self.model(frames)


def frame_levels(
    model: arch.temporalfix_arch, window: torch.Tensor
) -> list[torch.Tensor]:
    """The six levels of a window (1, 7, 3, H, W), each frame's part run on its own."""
    part = FramePart(model)
    per_frame = [part(window[:, t]) for t in range(window.shape[1])]
    return [torch.cat(level) for level in zip(*per_frame)]


__all__ = [
    "CENTRE",
    "FRAMES",
    "LEVELS",
    "SUPPORTS",
    "FramePart",
    "WindowPart",
    "cached_pyramids",
    "frame_levels",
    "stage_sizes",
]
