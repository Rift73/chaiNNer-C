"""Seeded test images for the native image node differentials (test_run_nodes_image_*.py)."""

import zlib

import numpy as np

SIZES = [(1, 1), (3, 5), (17, 31), (64, 257)]
CHANNELS = [1, 3, 4]
EMPTY_SHAPES = [(0, 5, 3), (4, 0), (0, 0, 4)]
# NaN with two payloads, both zeros, both infinities, a denormal, out-of-range values.
EDGES = np.concatenate(
    [
        np.array([np.nan, -0.0, 0.0, np.inf, -np.inf, 1e-45, -0.25, 1.75, 1.0], np.float32),
        np.array([0x7FC00123, 0xFFC00001], np.uint32).view(np.float32),
    ]
)


def seed_of(*key: object) -> int:
    """A stable seed for a test case's parameters."""
    return zlib.crc32(repr(key).encode())


def image(seed: int, height: int, width: int, channels: int, edges: bool = True) -> np.ndarray:
    """Seeded noise in [0, 1) as an enforced image (2-D for one channel), its first values the edge values."""
    rng = np.random.default_rng(seed)
    values = rng.random((height, width, channels), dtype=np.float32)
    if edges:
        flat = values.reshape(-1)
        flat[: min(flat.size, EDGES.size)] = EDGES[: flat.size]
    return values[:, :, 0] if channels == 1 else values

