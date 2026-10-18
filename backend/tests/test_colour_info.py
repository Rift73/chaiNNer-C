"""The plane value convention and ColourInfo (nodes/impl/colour_info.py)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from nodes.impl.colour_info import (
    ChromaLocation,
    ColourInfo,
    Matrix,
    Primaries,
    Range,
    Transfer,
    chroma_size_for,
    code_to_float,
    float_to_code,
    subsampling_from_sizes,
)


@pytest.mark.parametrize("bit_depth", [8, 10, 12, 16])
@pytest.mark.parametrize("full_range", [False, True])
@pytest.mark.parametrize("chroma", [False, True])
def test_every_code_survives_a_round_trip(
    bit_depth: int, full_range: bool, chroma: bool
):
    codes = np.arange(1 << bit_depth, dtype=np.uint8 if bit_depth == 8 else np.uint16)
    values = code_to_float(codes, bit_depth, full_range, chroma)
    assert values.dtype == np.float32
    assert np.array_equal(float_to_code(values, bit_depth, full_range, chroma), codes)


def test_black_white_and_neutral_chroma():
    def value(code: int, bits: int, full: bool, chroma: bool) -> float:
        return float(code_to_float(np.array([code], np.uint16), bits, full, chroma)[0])

    assert value(16, 8, False, False) == 0 and value(235, 8, False, False) == 1
    assert value(64, 10, False, False) == 0 and value(940, 10, False, False) == 1
    assert value(128, 8, False, True) == 0.5 and value(512, 10, False, True) == 0.5
    assert value(16, 8, False, True) == 0 and value(240, 8, False, True) == 1
    assert value(0, 8, True, False) == 0 and value(255, 8, True, False) == 1
    assert value(32768, 16, True, True) == pytest.approx(0.5, abs=1e-5)
    # Codes outside the nominal range stay outside 0-1; saving clips only at the codes.
    assert value(4, 8, False, False) < 0
    assert float_to_code(
        np.array([-1.0, 2.0], np.float32), 8, False, False
    ).tolist() == [0, 255]


def test_subsampling_from_plane_sizes():
    assert subsampling_from_sizes((1920, 1080), (960, 540)) == (1, 1)
    assert subsampling_from_sizes((1920, 1080), (960, 1080)) == (1, 0)
    assert subsampling_from_sizes((1920, 1080), (1920, 1080)) == (0, 0)
    assert subsampling_from_sizes((1919, 1079), (960, 540)) == (1, 1)
    assert chroma_size_for((1919, 1079), (1, 1)) == (960, 540)
    with pytest.raises(ValueError, match="does not fit"):
        subsampling_from_sizes((1920, 1080), (1000, 540))


def info(**changes: Any) -> ColourInfo:
    base = ColourInfo(
        matrix=Matrix.BT709,
        transfer=Transfer.BT709,
        primaries=Primaries.BT709,
        range=Range.LIMITED,
        chroma_location=ChromaLocation.LEFT,
        bit_depth=10,
        luma_size=(1920, 1080),
        chroma_size=(960, 540),
        origin=(
            ("chroma_location", "file"),
            ("matrix", "guessed"),
            ("primaries", "file"),
            ("range", "file"),
            ("transfer", "file"),
        ),
    )
    return replace(base, **changes) if changes else base


def test_choices_override_and_mark_their_origin():
    chosen = info().with_choices(matrix=Matrix.BT470BG, range_=Range.FULL)
    assert (chosen.matrix, chosen.range, chosen.transfer) == (
        Matrix.BT470BG,
        Range.FULL,
        Transfer.BT709,
    )
    assert (
        chosen.origin_of("matrix") == "set" and chosen.origin_of("transfer") == "file"
    )
    base = info()
    assert base.with_choices() is base
    with pytest.raises(ValueError, match="cannot be Auto"):
        info(matrix=Matrix.AUTO)


def test_summary_names_the_guessed_values():
    assert info().summary() == (
        "Matrix BT.709, transfer BT.709 (BT.1886), primaries BT.709, limited range,"
        " 4:2:0 left-sited, 10-bit (matrix guessed)"
    )
    full = info(chroma_size=(1920, 1080), range=Range.FULL).with_choices(
        primaries=Primaries.BT2020
    )
    assert full.summary() == (
        "Matrix BT.709, transfer BT.709 (BT.1886), primaries BT.2020, full range, 4:4:4,"
        " 10-bit (matrix guessed, primaries set)"
    )
