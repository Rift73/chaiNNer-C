"""Records native/tests/golden/texconv.json: texconv.exe's results for the cases of
native/tests/test_texconv.py (the DDS file each writes and the pixels read back), as
SHA-256 digests. Windows only: the repository's texconv.exe is the reference that the
Linux texconv (native/texconv) is checked against. Run it after a texconv.exe update or a
change to the cases, from the repository root with backend/src on the path:

  native\\.venv\\Scripts\\python.exe -B native/tools/texconv_golden.py
"""

from __future__ import annotations

import importlib
import json
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    if platform.system() != "Windows":
        print("texconv_golden.py: texconv.exe runs on Windows only", file=sys.stderr)
        return 1
    sys.path[:0] = [str(ROOT / "backend/src"), str(ROOT / "native/tests")]
    # Found on the path set just above, as the test suite finds it
    test_texconv = importlib.import_module("test_texconv")

    from nodes.impl.dds import texconv

    texconv.__dict__["__TEXCONV_EXE"] = str(test_texconv.TEXCONV)
    recorded = {}
    with tempfile.TemporaryDirectory() as directory:
        for name, case in test_texconv.CASES.items():
            recorded[name] = test_texconv.run(case, Path(directory))
    golden = {
        "about": "texconv.exe 2026.5.8.1's results for native/tests/test_texconv.py's"
        " cases (SHA-256, first 16 hex digits), from native/tools/texconv_golden.py",
        "cases": recorded,
    }
    text = json.dumps(golden, indent=0, sort_keys=True)
    test_texconv.GOLDEN.write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"{len(recorded)} cases -> {test_texconv.GOLDEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
