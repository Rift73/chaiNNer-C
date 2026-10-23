import navi
from nodes.impl.upscale.basic_upscale import PaddingType

from ...impl.color.convert_data import (
    color_spaces,
    color_spaces_or_detectors,
    get_alpha_partner,
    is_alpha_partner,
)

# pylint: disable=relative-beyond-top-level
from ...impl.image_utils import BorderType
from ...impl.linear_light import LightTransfer
from ...impl.pil_utils import RotationInterpolationMethod
from ...impl.resize import ResizeFilter
from .generic_inputs import DropDownGroup, DropDownInput, EnumInput


def ColorSpaceDetectorInput(label: str = "Color Space") -> DropDownInput:
    return DropDownInput(
        input_type="ColorSpace",
        label=label,
        options=[
            {
                "option": c.name,
                "value": c.id,
                "type": navi.named("ColorSpace", {"channels": c.channels}),
            }
            for c in color_spaces_or_detectors
        ],
    )


def ColorSpaceInput(label: str = "Color Space") -> DropDownInput:
    return DropDownInput(
        input_type="ColorSpace",
        label=label,
        options=[
            {
                "option": c.name,
                "value": c.id,
                "type": navi.named(
                    "ColorSpace",
                    {
                        "channels": c.channels,
                        "supportsAlpha": get_alpha_partner(c) is not None,
                    },
                ),
            }
            for c in color_spaces
            if not is_alpha_partner(c)
        ],
    )


def ResizeFilterInput() -> DropDownInput:
    return EnumInput(
        ResizeFilter,
        label="Interpolation Method",
        categories=[
            DropDownGroup("Basic", start_at=ResizeFilter.AUTO),
            DropDownGroup("Advanced", start_at=ResizeFilter.HERMITE),
        ],
        option_labels={
            ResizeFilter.NEAREST: "Nearest Neighbor",
            ResizeFilter.BOX: "Area (Box)",
            ResizeFilter.CATROM: "Cubic",
            ResizeFilter.BSPLINE: "B-Spline",
            ResizeFilter.DPID: "DPID (detail-preserving)",
        },
    )


def LightTransferInput(default: LightTransfer) -> DropDownInput:
    """Linear Light's curve; LightTransfer::label gives a key info its short name."""
    return EnumInput(
        LightTransfer,
        label="Transfer",
        default=default,
        option_labels={
            LightTransfer.GAMMA_24: "Gamma 2.4 (BT.1886)",
            LightTransfer.SRGB: "sRGB",
            LightTransfer.GAMMA_22: "Gamma 2.2",
            LightTransfer.BT709_CAMERA: "BT.709 camera (inverse OETF)",
        },
        extra_definitions="""
            def LightTransfer::label(transfer: LightTransfer): string {
                match transfer {
                    LightTransfer::Gamma24 => "Gamma 2.4",
                    LightTransfer::Srgb => "sRGB",
                    LightTransfer::Gamma22 => "Gamma 2.2",
                    LightTransfer::Bt709Camera => "BT.709 camera",
                }
            }
        """,
    ).with_docs(
        "The curve the image's values are encoded with: Gamma 2.4 (BT.1886) for video as displayed, sRGB for images and screen captures, Gamma 2.2 for a plain 2.2 power, BT.709 camera to undo the camera curve (scene light).",
        "The curves never clip: they continue above 1 and are mirrored below 0, so values outside 0-1 (such as Descale's ringing) survive the conversion. A resize filter that clamps its result to 0-1 still clamps it.",
    )


def RotateInterpolationInput() -> DropDownInput:
    return EnumInput(
        RotationInterpolationMethod,
        label="Interpolation Method",
        option_labels={
            RotationInterpolationMethod.NEAREST: "Nearest Neighbor",
        },
    )


def BorderInput() -> DropDownInput:
    return EnumInput(
        BorderType,
        default=BorderType.REFLECT_MIRROR,
        option_labels={
            BorderType.REFLECT_MIRROR: "Reflect (Mirror)",
            BorderType.WRAP: "Wrap (Tile)",
            BorderType.REPLICATE: "Replicate Edges",
        },
        extra_definitions="""
            def BorderType::getOutputChannels(type: BorderType, channels: uint, color: Color | null): uint {
                match type {
                    BorderType::Transparent => 4,
                    BorderType::CustomColor => match color {
                        Color => max(color.channels, channels),
                        null => never,
                    },
                    _ => channels
                }
            }
        """,
    )


def PaddingTypeInput() -> DropDownInput:
    return EnumInput(
        PaddingType,
        label="Padding",
        default=PaddingType.NONE,
        option_labels={
            PaddingType.REFLECT_MIRROR: "Reflect (Mirror)",
            PaddingType.WRAP: "Wrap (Tile)",
            PaddingType.REPLICATE: "Replicate Edges",
        },
    ).with_docs(
        "Adding padding to an image can eliminate artifacts at the edges of an image, at the cost of increasing processing time.",
        "**Always** use *Wrap (Tile)* when upscaling tiled images to avoid artifacts at the tile borders.",
        "For very small images (e.g. pixel art smaller than 100x100px), use *Reflect (Mirror)* or *Replicate Edges* to increase the upscale quality.",
        hint=True,
    )
