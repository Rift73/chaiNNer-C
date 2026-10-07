from __future__ import annotations

from typing import Union

from nodes.groups import optional_list_group, seed_group
from nodes.impl.native_utility_random import derive_seed, seed_to_bytes
from nodes.properties.inputs import BaseInput, SeedInput
from nodes.properties.outputs import SeedOutput
from nodes.utils.seed import Seed
from nodes.utils.utils import ALPHABET

from .. import random_group

Source = Union[int, float, str, Seed]


def SourceInput(label: str):
    return BaseInput(
        kind="generic",
        label=label,
        input_type="number | string | Directory | Seed",
    ).make_optional()


def _to_bytes(s: Source) -> bytes:
    return seed_to_bytes(s)


@random_group.register(
    schema_id="chainner:utility:derive_seed",
    name="Derive Seed",
    description="Creates a new seed from multiple sources of randomness.",
    icon="MdCalculate",
    inputs=[
        seed_group(SeedInput(has_handle=False)),
        SourceInput("Source A"),
        optional_list_group(
            *[SourceInput(f"Source {letter}") for letter in ALPHABET[1:10]],
        ),
    ],
    outputs=[
        SeedOutput(),
    ],
)
def derive_seed_node(seed: Seed, *sources: Source | None) -> Seed:
    return derive_seed(seed, sources)
