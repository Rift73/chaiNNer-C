"""Native utility RNG control; compiled CPython MT/hash primitives are retained."""

from __future__ import annotations

from nodes.utils.seed import Seed

from .native_graph import graph


def seed_to_bytes(value: object) -> bytes:
    return graph().utility_seed_to_bytes(value, Seed)


def derive_seed(seed: Seed, sources: tuple) -> Seed:
    return graph().utility_derive_seed(seed, sources, Seed)


def random_number(minimum: int, maximum: int, seed: Seed) -> int:
    return graph().utility_random_number(minimum, maximum, seed)
