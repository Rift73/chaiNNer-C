"""The backend's startup on a runtime that has only the server dependencies.

On a fresh install the integrated Python gets dependencies/install_server_deps.py's
packages, then the host starts the worker, the worker lists the packages, and only
then does the host install their dependencies. So the host, the worker and every
package's __init__ must import without any package dependency but NumPy, and a
dependency that is missing or fails to load costs only its own package's nodes.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "src"

# The top-level modules of every package Dependency, but NumPy's (a server dependency).
ABSENT = [
    "cv2",
    "PIL",
    "pillow_avif",
    "ffmpeg",
    "av",
    "requests",
    "scipy",
    "wcmatch",
    "numba",
    "pymatting",
    "torch",
    "torchvision",
    "facexlib",
    "einops",
    "safetensors",
    "spandrel",
    "spandrel_extra_arches",
    "onnx",
    "onnxoptimizer",
    "onnxruntime",
    "ncnn",
    "tensorrt",
    "cuda",
    "triton",
]

# argv: the absent modules as JSON, then the modules whose import raises OSError.
STARTUP = """
import asyncio, importlib.abc, json, logging, sys

absent, broken = json.loads(sys.argv[1]), json.loads(sys.argv[2])


class FreshRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        top = name.split(".")[0]
        if top in broken:
            raise OSError(f"[WinError 1455] The paging file is too small ({top})")
        if top in absent:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


sys.meta_path.insert(0, FreshRuntime())
sys.argv = [sys.argv[0]]
errors = []


class Errors(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            errors.append([record.getMessage(), repr(record.exc_info and record.exc_info[1])])


import server_host
import server

# After the imports: each Sanic app applies its logging config when created.
logging.getLogger("sanic.root").addHandler(Errors())
asyncio.run(server.import_packages(server.AppContext.get(server.app).config))
print(json.dumps({
    "packages": sorted(p.id for p in server.api.registry.packages.values()),
    "categories": sorted({sub.category.id for _, sub in server.api.registry.nodes.values()}),
    "errors": errors,
}))
"""


def start(absent: list[str], broken: list[str]) -> dict:
    done = subprocess.run(
        [sys.executable, "-B", "-c", STARTUP, json.dumps(absent), json.dumps(broken)],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.splitlines()[-1])


PACKAGES = [
    "chaiNNer_external",
    "chaiNNer_ncnn",
    "chaiNNer_onnx",
    "chaiNNer_pytorch",
    "chaiNNer_standard",
    "chaiNNer_tensorrt",
]


def test_the_worker_lists_every_package_without_their_dependencies():
    started = start(ABSENT, [])
    assert started["packages"] == PACKAGES
    # Not installed is not an error: the node modules' imports fail one by one.
    assert started["errors"] == []


def test_a_torch_that_fails_to_load_costs_only_the_pytorch_nodes():
    # The standard package's dependencies are installed; the GPU frameworks other
    # than torch are absent, so nothing here loads a GPU runtime.
    gpu = ["ncnn", "onnx", "onnxoptimizer", "onnxruntime", "tensorrt", "cuda", "triton"]
    started = start(gpu, ["torch"])
    assert started["packages"] == PACKAGES
    assert "image" in started["categories"]
    assert "pytorch" not in started["categories"]
    assert started["errors"] == [
        [
            "Failed to import PyTorch; the PyTorch nodes are unavailable.",
            "OSError('[WinError 1455] The paging file is too small (torch)')",
        ]
    ]


def declared(path: Path, call: str, name_keyword: str) -> dict[str, str]:
    """{name: version} of every `call(...)` in the file at path."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keywords = [
        {k.arg: k.value for k in node.keywords}
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == call
    ]
    return {
        ast.literal_eval(k[name_keyword]): ast.literal_eval(k["version"])
        for k in keywords
    }


def test_numpy_is_a_server_dependency_at_the_standard_packages_pin():
    server = declared(
        BACKEND / "dependencies/install_server_deps.py",
        "DependencyInfo",
        "package_name",
    )
    standard = declared(
        BACKEND / "packages/chaiNNer_standard/__init__.py", "Dependency", "pypi_name"
    )
    assert server["numpy"] == standard["numpy"]
