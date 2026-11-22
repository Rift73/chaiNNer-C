"""vs_temporalfix's AI model (pifroggi, Apache-2.0) on the frames of a chaiNNer loop.

Follows vs_temporalfix/temporalfix_model.py `_pytorch()` at the pinned commit: the same
checkpoints and strength interpolation, the 7-frame window with `utils.gen_shifts`' mirror
padding at the clip edges, fp16 on CUDA GPUs of compute capability 7 or newer.
"""

from __future__ import annotations

import hashlib
import math
import os
import weakref
from collections import OrderedDict
from collections.abc import Callable, Hashable
from pathlib import Path
from typing import Generic, TypeVar

import requests
import torch

from vendor.temporalfix.temporalfix_arch import temporalfix_arch

COMMIT = "bc3d5d8b8bdf5ee3e785cc9a083fc6218dc91d63"
URL = f"https://raw.githubusercontent.com/pifroggi/vs_temporalfix/{COMMIT}/vs_temporalfix/models/"
# Checkpoint per integer strength: file name and SHA-256 of the file at COMMIT.
CHECKPOINTS = {
    0: (
        "temporalfix_s0_v1.pth",
        "778b32d5c117b846912a93e88ea887541c797474784f6d896c5bfe92d71dd9fd",
    ),
    1: (
        "temporalfix_s1_v1.1.pth",
        "092db03f1d47402f1cc21294afe31072f31ba7a0de76984e06f748c0b6f6b311",
    ),
    2: (
        "temporalfix_s2_v1.pth",
        "f762c05f219ed0588044179f658adc0dcf6875b1d57310d8dff69424ad8f4b1b",
    ),
    3: (
        "temporalfix_s3_v1.pth",
        "276c84c6e16a0f44df21c503118e0eaa812dc36c2500f539ed3d26934e7da303",
    ),
}
RADIUS = 3

T = TypeVar("T")


def checkpoint_path(folder: Path, strength: int) -> Path:
    """The pinned checkpoint for an integer strength, downloaded into `folder` once."""
    name, sha256 = CHECKPOINTS[strength]
    target = folder / name
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == sha256:
        return target
    url = URL + name
    try:
        with requests.get(url, timeout=(15, 60)) as response:
            response.raise_for_status()
            data = response.content
    except requests.RequestException as error:
        raise RuntimeError(
            f"Could not download the Temporal Fix weights from {url}: {error}. "
            "Check the internet connection and run again."
        ) from error
    if hashlib.sha256(data).hexdigest() != sha256:
        raise RuntimeError(
            f"The downloaded {name} does not match the SHA-256 this version of "
            "chaiNNer-C expects, so it was not saved."
        )
    folder.mkdir(parents=True, exist_ok=True)
    part = target.with_name(name + ".part")
    part.write_bytes(data)
    os.replace(part, target)
    return target


def load_state(folder: Path, strength: float) -> OrderedDict[str, torch.Tensor]:
    """Upstream's weights for `strength` (0 < strength <= 3): an integer strength loads
    its checkpoint; another lerps the floating tensors of the two neighbouring ones."""
    strength = round(float(strength), 2)
    lower, upper = math.floor(strength), math.ceil(strength)

    def load(index: int) -> OrderedDict[str, torch.Tensor]:
        return torch.load(
            checkpoint_path(folder, index), map_location="cpu", weights_only=True
        )

    if lower == upper:
        return load(lower)
    low, high, weighting = load(lower), load(upper), strength - lower
    return OrderedDict(
        (
            key,
            torch.lerp(value, high[key], weighting)
            if torch.is_floating_point(value)
            else value,
        )
        for key, value in low.items()
    )


def uses_fp16(device: torch.device) -> bool:
    return (
        device.type == "cuda"
        and torch.cuda.get_device_capability(device=device)[0] >= 7
    )


def even_size(height: int, width: int) -> tuple[int, int]:
    """The size the network runs frames of height x width at: upstream needs even sizes,
    so an odd one gains a row or column."""
    return height + height % 2, width + width % 2


def build_model(
    state: OrderedDict[str, torch.Tensor], hw: tuple[int, int], device: torch.device
) -> temporalfix_arch:
    model = temporalfix_arch(
        fixed_hw=hw, conf_thresh=0.6, min_support=1, gate_slope=12.0, count_slope=4.0
    )
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    if uses_fp16(device):
        model.half()
    return model


def mirror(index: int, first: int, last: int) -> int:
    """`utils.gen_shifts`' edge padding: reflect about the first and the last frame without
    repeating them. Upstream needs 4 frames or more; shorter spans keep reflecting."""
    span = last - first
    if span == 0:
        return first
    offset = (index - first) % (2 * span)
    return first + (offset if offset <= span else 2 * span - offset)


def window(index: int, first: int, last: int) -> list[int]:
    """The frames feeding offsets -RADIUS..RADIUS of `index` in the clip first..last."""
    return [mirror(index + k, first, last) for k in range(-RADIUS, RADIUS + 1)]


class FrameWindow(Generic[T]):
    """Per loop run: the converted frames of the last RADIUS items and of the frames read
    ahead. `ahead` is False when the loop cannot serve RADIUS frames ahead; every frame
    is then the clip's last one, so the future side mirrors the past. `further`: the loop
    serves one frame more, so the next item's window can be gathered (gather_ahead)
    while this item's is computed."""

    def __init__(self, ahead: bool, shape: tuple[int, ...], further: bool = False):
        self.ahead = ahead
        self.further = ahead and further
        self.shape = shape
        self.frames: dict[int, T] = {}
        # The next item's window, gathered ahead
        self.next: tuple[int, list[T]] | None = None
        # The PyTorch route's pyramid levels of the kept frames
        self.levels = LevelCache()

    def gather(
        self,
        index: int,
        current: Callable[[], T],
        peek: Callable[[int], T | None],
    ) -> list[T]:
        """The window of item `index`. `current` converts this item's frame; `peek(k)`
        converts the frame k items ahead, None when it does not exist (end of clip). A
        missing earlier item (one this node did not run for) starts the clip afresh."""
        gathered, self.next = self.next, None
        if gathered is not None and gathered[0] == index:
            return gathered[1]
        frames = {i: f for i, f in self.frames.items() if i >= index - RADIUS}
        if index not in frames:
            frames[index] = current()
        first = index
        while first > index - RADIUS and first - 1 in frames:
            first -= 1
        last = index
        while self.ahead and last < index + RADIUS:
            if last + 1 not in frames:
                frame = peek(last + 1 - index)
                if frame is None:
                    break
                frames[last + 1] = frame
            last += 1
        self.frames = {i: f for i, f in frames.items() if i > index - RADIUS}
        return [frames[i] for i in window(index, first, last)]

    def gather_ahead(
        self, index: int, peek: Callable[[int], T | None]
    ) -> list[T] | None:
        """The window of item `index + 1`, gathered now (its newest frame read and
        converted through `peek`, as gather's) and kept for that item's gather; None
        when the loop does not serve the frame further ahead, or `index` is the clip's
        last item."""
        following = index + 1
        if not self.further or following not in self.frames:
            return None
        frames = self.gather(
            following, lambda: self.frames[following], lambda k: peek(k + 1)
        )
        self.next = (following, frames)
        return frames


class LevelCache:
    """Per loop run on PyTorch: the six pyramid levels of each frame the window keeps,
    known by the frame's object (FrameWindow hands on the same tensor for a frame while
    it keeps it), so a sliding window computes one new frame's levels per output. The
    levels are those of the weights that computed them (`source`, compared by equality):
    other weights (a Strength that changes from item to item) start the cache afresh."""

    def __init__(self) -> None:
        self.source: Hashable = None
        self.entries: dict[
            int, tuple[weakref.ref[torch.Tensor], tuple[torch.Tensor, ...]]
        ] = {}

    def levels(
        self,
        frame: torch.Tensor,
        source: Hashable,
        compute: Callable[[torch.Tensor], tuple[torch.Tensor, ...]],
    ) -> tuple[torch.Tensor, ...]:
        if source != self.source:
            self.source = source
            self.entries = {}
        entry = self.entries.get(id(frame))
        if entry is not None and entry[0]() is frame:
            return entry[1]
        self.entries = {k: e for k, e in self.entries.items() if e[0]() is not None}
        levels = compute(frame)
        self.entries[id(frame)] = (weakref.ref(frame), levels)
        return levels
