from __future__ import annotations

from nodes.impl.colour_info import (
    CHROMA_LOCATION_LABELS,
    MATRIX_LABELS,
    PRIMARIES_LABELS,
    RANGE_LABELS,
    TRANSFER_LABELS,
    ChromaLocation,
    Matrix,
    Primaries,
    Range,
    Transfer,
)

from .generic_inputs import EnumInput

AUTO_DOCS = (
    "Auto takes the value the video is tagged with, else the best guess from the video"
    " (the node's Colour output says which values were guessed). Choose a value when"
    " the tag is wrong or missing."
)


def colour_override_inputs(first_id: int) -> list[EnumInput]:
    """Matrix, Transfer, Primaries, Range and Chroma Siting dropdowns, Auto by default,
    with the ids first_id to first_id + 4 (ColourInfo.with_choices's arguments)."""
    return [
        EnumInput(Matrix, "Matrix", default=Matrix.AUTO, option_labels=MATRIX_LABELS)
        .with_id(first_id)
        .with_docs(
            "The matrix between YUV and RGB.",
            AUTO_DOCS,
        ),
        EnumInput(
            Transfer, "Transfer", default=Transfer.AUTO, option_labels=TRANSFER_LABELS
        )
        .with_id(first_id + 1)
        .with_docs("The transfer curve (gamma, PQ or HLG).", AUTO_DOCS),
        EnumInput(
            Primaries,
            "Primaries",
            default=Primaries.AUTO,
            option_labels=PRIMARIES_LABELS,
        )
        .with_id(first_id + 2)
        .with_docs("The colour primaries (the gamut).", AUTO_DOCS),
        EnumInput(Range, "Range", default=Range.AUTO, option_labels=RANGE_LABELS)
        .with_id(first_id + 3)
        .with_docs(
            "Limited (TV) range codes black at 16 and white at 235 in 8 bits; full"
            " range uses 0 to 255.",
            AUTO_DOCS,
        ),
        EnumInput(
            ChromaLocation,
            "Chroma Siting",
            default=ChromaLocation.AUTO,
            option_labels=CHROMA_LOCATION_LABELS,
        )
        .with_id(first_id + 4)
        .with_docs(
            "Where each subsampled chroma sample sits between the luma pixels it covers.",
            AUTO_DOCS,
        ),
    ]
