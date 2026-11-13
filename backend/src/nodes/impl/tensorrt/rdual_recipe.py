"""What an RDUAL engine set's export takes from its input size: the padded, trunk and
flow sizes, and the flow part's precision recipe, which is chosen here and only here
(flow_precision). Imported by the export (rdual_export.py) and, for the set's key, by
rdual.py; no torch, ONNX or TensorRT."""

from __future__ import annotations

from dataclasses import dataclass


class Size:
    """The padded, trunk and flow sizes of an input size, as RDUAL pads it: to a
    multiple of 4 for the trunk (RDUAL._pad), to a multiple of 16 and at least 32 for
    the flow (RDUAL.guidance)."""

    def __init__(self, height: int, width: int) -> None:
        self.height, self.width = height, width
        self.padded_height = (height + 3) // 4 * 4
        self.padded_width = (width + 3) // 4 * 4
        self.trunk_height = self.padded_height // 2
        self.trunk_width = self.padded_width // 2
        self.flow_height = max(32, (height + 15) // 16 * 16)
        self.flow_width = max(32, (width + 15) // 16 * 16)


@dataclass(frozen=True)
class FlowPrecision:
    """The flow part's precision recipe: its name (in the set's key and info), the
    NeuFlow modules that run in FP16, and whether the correlation's features and window
    values are FP16 (as those modules make them) or FP32. The global matching, the flow
    field and its accumulation, the correlation sampling's blend and the convex
    upsampling stay FP32 in every recipe."""

    name: str
    half_modules: tuple[str, ...]
    half: bool


# Mixed precision: the encoder, cross attention, merges, refinement and upsampling
# convolutions in FP16
MIXED_FLOW = FlowPrecision(
    name="mixed",
    half_modules=(
        "backbone",
        "cross_attn_s16",
        "merge_s8",
        "context_merge_s8",
        "refine_s16",
        "refine_s8",
        "conv_s8",
        "upsample_s8",
    ),
    half=True,
)
# The whole flow in FP32
FULL_FLOW = FlowPrecision(name="full", half_modules=(), half=False)
FLOW_RECIPES = {recipe.name: recipe for recipe in (MIXED_FLOW, FULL_FLOW)}
# The most 1/16-scale tokens (the global matching's queries) the mixed recipe runs at.
# Against the flow in FP32 (mean end-point error over the clip's pairs): mixed is
# 0.004-0.016 px at 1280x720 (3,600 tokens) and 0.008-0.05 px at 1488x837 (4,929), but
# 0.016-0.18 px at 1600x900 (5,700) and 0.9-1.2 px at 1920x1080 (8,160, 17-24 % of
# positions over 1 px): FP16 rounding of the encoder's features grows through the eight
# refinement steps with the token count. FP32 holds there (0.043-0.062 px at 1080p) at
# 42.5 against 16.2 ms per pair.
# In the node's output, against PyTorch's Upscale Image on the same frames: up to 5,700
# tokens (1600x900) the two recipes give the same output (RMS 0.000267 both, largest pixel
# error 0.0127 mixed, 0.0132 FP32); at 8,160 (1920x1080) the RMS stays the same (0.000237
# mixed, 0.000233 FP32) but the largest pixel error rises from 0.012 to 0.034 with the
# mixed flow. So the flow runs in FP32 from 1920x1080 up, at 11-16 % more time per frame.
MIXED_FLOW_TOKENS = 6000


def flow_tokens(size: Size) -> int:
    """The global matching's token count: the flow input's 1/16-scale positions."""
    return (size.flow_height // 16) * (size.flow_width // 16)


def flow_precision(size: Size, name: str | None = None) -> FlowPrecision:
    """The flow part's recipe: the one place it is chosen. `name` (a FLOW_RECIPES key)
    asks for one explicitly; otherwise the mixed recipe up to MIXED_FLOW_TOKENS
    tokens, FP32 above."""
    if name:
        if name not in FLOW_RECIPES:
            raise ValueError(
                f"No flow recipe {name!r}; the recipes are {', '.join(FLOW_RECIPES)}"
            )
        return FLOW_RECIPES[name]
    return MIXED_FLOW if flow_tokens(size) <= MIXED_FLOW_TOKENS else FULL_FLOW


__all__ = [
    "FLOW_RECIPES",
    "FULL_FLOW",
    "MIXED_FLOW",
    "MIXED_FLOW_TOKENS",
    "FlowPrecision",
    "Size",
    "flow_precision",
    "flow_tokens",
]
