"""drft_trt_plugins.dll (native/src/drft): the creators TensorRT's plugin registry finds in it.
The library is optional (native/Build.ps1 with the TensorRT and CUDA roots), so the test skips
without it. It loads in a child process with CUDA hidden: registration needs no GPU, and the
library never enters this process's registry."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from nodes.impl.tensorrt.drft import PLUGIN_LIBRARY

# Every creator of the library: version "1", empty namespace, these fields in this order.
EXPECTED = {
    "DrftOCABAttentionSafe_TRT": [
        ["q_window_size", "INT32"],
        ["kv_window_size", "INT32"],
    ],
    "DrftOCABRelativeBiasSafe_TRT": [
        ["q_window_size", "INT32"],
        ["kv_window_size", "INT32"],
    ],
    "DrftOCABQKVPackTokenMajor_TRT": [
        ["channels", "INT32"],
        ["heads", "INT32"],
        ["query_window_size", "INT32"],
        ["key_window_size", "INT32"],
    ],
    "DrftWindowNeuralBias_TRT": [],
    "DrftIdentityBarrier_TRT": [],
    "DrftFinalAffineCrop_TRT": [
        ["scale", "INT32"],
        ["mean_r", "FLOAT32"],
        ["mean_g", "FLOAT32"],
        ["mean_b", "FLOAT32"],
    ],
}

PROBE = """
import json
import sys

import tensorrt

registry = tensorrt.get_plugin_registry()
registry.load_library(sys.argv[1])
creators = {}
for creator in registry.all_creators:
    if creator.name.startswith("Drft"):
        creators[creator.name] = {
            "version": creator.plugin_version,
            "namespace": creator.plugin_namespace,
            "fields": [[field.name, field.type.name] for field in creator.field_names],
        }
print(json.dumps(creators))
"""


@pytest.mark.skipif(
    not PLUGIN_LIBRARY.is_file(), reason=f"{PLUGIN_LIBRARY.name} not built"
)
def test_the_library_registers_its_six_creators_with_their_fields():
    pytest.importorskip("tensorrt")
    result = subprocess.run(
        [sys.executable, "-B", "-c", PROBE, str(PLUGIN_LIBRARY)],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    creators = json.loads(result.stdout.strip().splitlines()[-1])
    assert creators == {
        name: {"version": "1", "namespace": "", "fields": fields}
        for name, fields in EXPECTED.items()
    }
