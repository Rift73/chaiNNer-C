"""The frozen upstream transfers under chaiNNer-C's contract (ruling of 2026-10-23): one
valid pixel has zero covariance, a solid image's, where np.cov's zero degrees of freedom
give NaN and eig rejects it ("Array must not contain infs or NaNs")."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def cov(m, rowvar):
    if len(m) == 1:
        return np.zeros((m.shape[1], m.shape[1]))
    return np.cov(m, rowvar=rowvar)


def load(name, replacements=()):
    """`name`_transfer from this folder's `name`.py, its np.cov calls through cov, after
    the given (old, new) source replacements."""
    source = Path(__file__).with_name(name + ".py").read_text()
    source = source.replace("np.cov(", "cov(")
    for old, new in replacements:
        source = source.replace(old, new)
    namespace = {"cov": cov}
    exec(compile(source, name, "exec"), namespace)
    return namespace[name + "_transfer"]
