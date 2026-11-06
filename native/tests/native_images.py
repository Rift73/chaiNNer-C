"""Seeded test images for the native entry point differential tests (test_run_nodes_*.py)."""

import zlib

import numpy as np

SIZES = [(1, 1), (3, 5), (17, 31), (64, 257)]
CHANNELS = [1, 3, 4]
EMPTY_SHAPES = [(0, 5, 3), (4, 0), (0, 0, 4)]
# NaN with two payloads, both zeros, both infinities, a denormal, out-of-range values.
EDGES = np.concatenate(
    [
        np.array(
            [np.nan, -0.0, 0.0, np.inf, -np.inf, 1e-45, -0.25, 1.75, 1.0], np.float32
        ),
        np.array([0x7FC00123, 0xFFC00001], np.uint32).view(np.float32),
    ]
)


def opaque(image: np.ndarray) -> np.ndarray:
    """A copy with a fourth channel of exactly 1, for a four-channel image: the filters with an alpha rule then run
    per channel, as the frozen originals do."""
    if image.ndim != 3 or image.shape[2] != 4:
        return image
    out = image.copy()
    out[:, :, 3] = 1
    return out


def seed_of(*key: object) -> int:
    """A stable seed for a parametrized case."""
    return zlib.crc32(repr(key).encode())


def image(
    seed: int,
    height: int,
    width: int,
    channels: int,
    layout: str = "c",
    edges: bool = True,
) -> np.ndarray:
    """Seeded noise in [0, 1) as an enforced image (2-D for one channel), its first
    values replaced by EDGES when edges is set: C order, or Lens Blur's planar
    transpose ("permuted")."""
    rng = np.random.default_rng(seed)
    values = rng.random((height, width, channels), dtype=np.float32)
    if edges:
        flat = values.reshape(-1)
        flat[: min(flat.size, EDGES.size)] = EDGES[: flat.size]
    if layout == "permuted":
        values = np.ascontiguousarray(values.transpose(2, 0, 1)).transpose(1, 2, 0)
    return values[:, :, 0] if channels == 1 else values
