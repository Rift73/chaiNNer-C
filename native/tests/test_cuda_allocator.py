"""The worker's PyTorch allocator settings (cuda_allocator.py): expandable_segments left
out, the user's other options kept, the VRAM offload's pinned RAM options added."""

from __future__ import annotations

import logging
import os
import subprocess
import sys

import pytest

import cuda_allocator

PINNED = "pinned_use_cuda_host_register:True,pinned_num_register_threads:8"
VARIABLES = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF")


def test_without_user_options_only_the_pinned_ones():
    assert cuda_allocator.allocator_config("") == (PINNED, [])


def test_expandable_segments_left_out_and_the_other_options_kept():
    user = (
        "max_split_size_mb:512, expandable_segments : True,"
        "roundup_power2_divisions:[256:1,>:2]"
    )
    assert cuda_allocator.allocator_config(user) == (
        f"max_split_size_mb:512,roundup_power2_divisions:[256:1,>:2],{PINNED}",
        ["expandable_segments : True"],
    )


def test_a_pinned_option_the_user_set_wins():
    assert cuda_allocator.allocator_config("pinned_num_register_threads:4") == (
        "pinned_num_register_threads:4,pinned_use_cuda_host_register:True",
        [],
    )


@pytest.mark.parametrize(
    ("user", "expected", "warning"),
    [
        ({}, PINNED, None),
        (
            {"PYTORCH_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:256"},
            f"max_split_size_mb:256,{PINNED}",
            "PYTORCH_ALLOC_CONF's expandable_segments:True is not used",
        ),
        # PyTorch reads only the first variable set
        (
            {
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            },
            f"max_split_size_mb:256,{PINNED}",
            None,
        ),
    ],
)
def test_apply_writes_the_variable_pytorch_reads_first(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    user: dict[str, str],
    expected: str,
    warning: str | None,
):
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)
    for name, value in user.items():
        monkeypatch.setenv(name, value)
    with caplog.at_level(logging.WARNING):
        cuda_allocator.apply()
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == expected
    if warning is None:
        assert not caplog.records
    else:
        assert warning in caplog.text


def test_pytorch_accepts_the_config():
    config, _ = cuda_allocator.allocator_config(
        "expandable_segments:True,max_split_size_mb:512,"
        "roundup_power2_divisions:[256:1,>:2],garbage_collection_threshold:0.8"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys, torch; torch._C._accelerator_setAllocatorSettings(sys.argv[1])",
            config,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
