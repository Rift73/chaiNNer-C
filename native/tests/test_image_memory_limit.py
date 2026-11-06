"""Requests for images past the 16 GiB limit fail with a clear error before anything is allocated, in both executors."""

import numpy as np
import pytest
from native_differential import assert_native_matches

from nodes.impl.resize import ResizeFilter


def strip(height: int, width: int, channels: int = 3) -> np.ndarray:
    return np.zeros((height, width, channels), np.float32)


def assert_refused(schema_id: str, inputs: list[object], name: str, gigabytes: float) -> None:
    message = f"{name} would require {gigabytes} GB of memory, but only 16 GB are allowed."
    assert assert_native_matches(schema_id, inputs) == ("error", ("RuntimeError", message))


@pytest.mark.parametrize(
    "filter_",
    [f for f in ResizeFilter if f not in (ResizeFilter.AUTO, ResizeFilter.NEAREST, ResizeFilter.DPID)],
)
def test_resize_counts_its_horizontal_pass(filter_):
    # One output row, but the horizontal pass holds all 20000 source rows at the new width: 22.4 GiB for 1.1 MiB.
    inputs = [strip(20000, 1, 1), 1, 100.0, 300000, 1, filter_.value, False, 1.0, False, 1]
    assert_refused("chainner:image:resize", inputs, "Resize", 22.352)


def test_resize_auto_counts_its_horizontal_pass():
    inputs = [strip(20000, 1, 1), 1, 100.0, 300000, 1, ResizeFilter.AUTO.value, False, 1.0, False, 1]
    assert_refused("chainner:image:resize", inputs, "Resize", 22.352)


def test_resize_nearest_has_no_horizontal_pass():
    inputs = [strip(20000, 1, 1), 1, 100.0, 300000, 1, ResizeFilter.NEAREST.value, False, 1.0, False, 1]
    outcome = assert_native_matches("chainner:image:resize", inputs)
    assert outcome[0] == "outputs"


def test_resize_within_the_limit_runs():
    inputs = [strip(200, 1, 1), 1, 100.0, 3000, 1, ResizeFilter.LANCZOS.value, False, 1.0, False, 1]
    assert assert_native_matches("chainner:image:resize", inputs)[0] == "outputs"


def test_absurd_percentage_names_the_limit():
    inputs = [strip(100, 100), 0, 1e300, 1, 1, ResizeFilter.LANCZOS.value, False, 1.0, False, 1]
    message = "Resize would require inf GB of memory, but only 16 GB are allowed."
    assert assert_native_matches("chainner:image:resize", inputs) == ("error", ("RuntimeError", message))
