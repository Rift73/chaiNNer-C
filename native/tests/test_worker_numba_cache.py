"""The worker keeps numba's cache in its storage directory (U4-light item 5, D-24).

PyMatting's kernels are numba functions with cache=True. Without NUMBA_CACHE_DIR,
numba writes their .nbi/.nbc files into the runtime's site-packages (the package's
python/ tree); the worker points it at <storage_dir>/numba-cache, inside the ruled
backend-storage/ root. numba reads the variable once, at its first import, so the
worker must set it before the node modules that import numba load.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"


def test_worker_sets_numba_cache_dir_before_numba_imports(tmp_path):
    storage = tmp_path / "backend-storage"
    # The real server.py in its own process: importing it builds the Sanic app and
    # parses argv. app.run is where setup, which loads the node modules, would start.
    code = "\n".join(
        [
            "import json, os, sys",
            f"sys.argv = ['server.py', '0', '--storage-dir', {str(storage)!r}]",
            f"sys.path[:0] = [{str(BACKEND)!r}]",
            "os.environ.pop('NUMBA_CACHE_DIR', None)",
            "import server",
            "state = {'numba_after_server_import': 'numba' in sys.modules}",
            "def run(self, **_):",
            "    state['at_run'] = os.environ.get('NUMBA_CACHE_DIR')",
            "    state['numba_at_run'] = 'numba' in sys.modules",
            "type(server.app).run = run",
            "server.main()",
            "import pymatting",
            "import numba",
            "state['numba_cache_dir'] = numba.config.CACHE_DIR",
            "print(json.dumps(state))",
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
    state = json.loads(result.stdout.splitlines()[-1])
    expected = str(storage / "numba-cache")
    assert state == {
        "numba_after_server_import": False,
        "at_run": expected,
        "numba_at_run": False,
        "numba_cache_dir": expected,
    }
