"""Records native/tests/golden/vosr2_temporal.npz: temporal VOSR's float32 outputs for the
golden cases of native/tests/test_vosr2_temporal.py (stabilize on three tiny frames) and
native/tests/test_vosr2_temporal_step.py (the step over tiny clips), from the current
code. The first recording came from the code they replaced: the one-thread stabilize
(before c8a4f267) and the removed Temporal Sequence generators, which the current code
equalled bit for bit. Run it only after an intended change of those outputs, from the
repository root:

  native\\.venv\\Scripts\\python.exe -B native/tools/vosr2_temporal_golden.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "native" / "tests" / "golden" / "vosr2_temporal.npz"
# OpenCV's optical flow, blurs and remaps differ in their last bits by build and CPU,
# which moves the outputs by about 1e-7, except where a match lands on the edge of the
# valid area (an exact one-pixel motion does): with AVX2 off, a few pixels within 4 of
# the frame's edge moved by up to 0.01. So: within ATOL inside MARGIN pixels of the edges,
# and within MEAN on average over the whole frame.
ATOL = 1e-5
MARGIN = 4
MEAN = 1e-4


def golden(key: str) -> np.ndarray:
    with np.load(GOLDEN) as data:
        return data[key]


def assert_close(actual: np.ndarray, expected: np.ndarray) -> None:
    """A float32 frame (height, width, channels) within the golden's tolerances."""
    assert actual.dtype == expected.dtype == np.float32
    assert actual.shape == expected.shape
    difference = np.abs(actual.astype(np.float64) - expected)
    inner = difference[MARGIN:-MARGIN, MARGIN:-MARGIN].max()
    assert inner <= ATOL, f"{inner:.3g} apart inside the margin"
    assert difference.mean() <= MEAN, f"{difference.mean():.3g} apart on average"


def main() -> int:
    sys.path[:0] = [str(ROOT / "backend/src"), str(ROOT / "native/tests")]
    recorded = {}
    for tests in ("test_vosr2_temporal", "test_vosr2_temporal_step"):
        recorded.update(importlib.import_module(tests).golden_outputs())
    np.savez_compressed(GOLDEN, **recorded)
    print(f"{GOLDEN}: {len(recorded)} outputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
