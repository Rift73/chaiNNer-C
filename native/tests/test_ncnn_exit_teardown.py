"""The NCNN session's at-exit teardown (U4-light item 2, Consult 8 sweep item 7).

A process holding PyPI ncnn's Vulkan instance at a normal exit crashed with
0xC0000005 (a worker started with --close-after-start, Vulkan visible). U4-light
verified the teardown on the GPU: that worker now exits 0. These tests hold its
order and its registration without a GPU.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from nodes.impl.ncnn import session
from nodes.impl.ncnn.model import NcnnModelWrapper
from packages.chaiNNer_ncnn.settings import NcnnSettings

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"
SETTINGS = NcnnSettings(
    gpu_index=0, winograd=False, sgemm=False, threads=1, blocktime=0, budget_limit=0
)


class Model(NcnnModelWrapper):
    """A cache key: only its identity and weak reference are used."""

    def __init__(self) -> None:
        pass


def test_teardown_frees_the_cached_nets_before_the_instance(monkeypatch):
    events: list[str] = []

    class Net:
        def __del__(self) -> None:
            events.append("net freed")

    model = Model()
    monkeypatch.setattr(session, "create_ncnn_net", lambda _model, settings: Net())
    net = session.get_ncnn_net(model, SETTINGS)
    assert session.get_ncnn_net(model, SETTINGS) is net
    del net
    assert events == []
    monkeypatch.setattr(
        session.ncnn,
        "destroy_gpu_instance",
        lambda: events.append("instance destroyed"),
    )
    session.destroy_gpu_instance()
    assert events == ["net freed", "instance destroyed"]


def test_importing_the_session_registers_the_teardown_at_exit():
    # Vulkan stays hidden (conftest), so ncnn creates no instance and the real
    # destroy_gpu_instance() is the no-op a CPU-only process gets.
    code = "\n".join(
        [
            "import sys",
            f"sys.path[:0] = [{str(BACKEND)!r}]",
            "from nodes.impl.ncnn import session",
            "real = session.ncnn.destroy_gpu_instance",
            "def destroy():",
            "    real()",
            "    print('instance destroyed', flush=True)",
            "session.ncnn.destroy_gpu_instance = destroy",
            "print('exiting', flush=True)",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-2:] == ["exiting", "instance destroyed"]
