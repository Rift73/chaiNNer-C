"""run_tests.py holds a shard's libraries to its thread cap, also those that read no variable."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import psutil
import pytest
import run_tests
import shard_threads
from onnx import TensorProto, helper


def tiny_onnx() -> bytes:
    graph = helper.make_graph(
        [helper.make_node("Neg", ["x"], ["y"])],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    return helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=8
    ).SerializeToString()


def test_a_shard_with_a_cap_loads_the_plugin_and_says_the_cap(tmp_path: Path):
    env = run_tests.environment(3, tmp_path / "report.jsonl")
    assert env["CHAINNER_C_SHARD_THREADS"] == "3"
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "3"
    assert "shard_threads" in env["PYTEST_PLUGINS"].split(",")


def test_the_whole_machine_shard_has_no_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("CHAINNER_C_SHARD_THREADS", raising=False)
    monkeypatch.delenv("PYTEST_PLUGINS", raising=False)
    env = run_tests.environment(None, tmp_path / "report.jsonl")
    assert "CHAINNER_C_SHARD_THREADS" not in env
    assert "shard_threads" not in env.get("PYTEST_PLUGINS", "")


def test_a_session_keeps_to_the_cap_and_does_not_spin(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ort, "InferenceSession", ort.InferenceSession)
    shard_threads.cap_onnx_runtime(2)
    me = psutil.Process()
    before = me.num_threads()
    session = ort.InferenceSession(tiny_onnx(), providers=["CPUExecutionProvider"])
    assert isinstance(session, ort.InferenceSession)
    options = session.get_session_options()
    assert options.intra_op_num_threads == 2
    assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"
    # One thread beside the caller's, not a pool of the machine's cores
    assert me.num_threads() - before <= 2
    (output,) = session.run(None, {"x": np.array([1.0], np.float32)})
    np.testing.assert_array_equal(output, [-1.0])


def test_a_session_whose_caller_chose_its_threads_keeps_them(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(ort, "InferenceSession", ort.InferenceSession)
    shard_threads.cap_onnx_runtime(2)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 3
    session = ort.InferenceSession(
        tiny_onnx(), options, providers=["CPUExecutionProvider"]
    )
    assert session.get_session_options().intra_op_num_threads == 3


INSIDE_A_SHARD = """
import sys

import cv2
import onnxruntime as ort
import torch


def test_a_shard_caps_what_it_runs():
    assert "shard_threads" in sys.modules
    assert torch.get_num_threads() == 2
    assert cv2.getNumThreads() == 2
    assert ort.InferenceSession.__name__ == "CappedSession"
"""


def test_a_shard_s_own_process_is_capped(tmp_path: Path):
    # As run_tests.py starts one: its environment alone turns this on.
    env = run_tests.environment(2, tmp_path / "report.jsonl")
    # A shard's own run reports through it, which this nested run must not.
    env.pop("CHAINNER_C_SHARD_REPORT")
    test_file = tmp_path / "test_inside_a_shard.py"
    test_file.write_text(INSIDE_A_SHARD, encoding="utf-8")
    done = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "--rootdir",
            str(tmp_path),
            str(test_file),
        ],
        cwd=tmp_path,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr
