"""The downloads of the dependency manager's installer (dependencies/store.py), run with
chainner_pip in a new virtual environment against an index on 127.0.0.1 that serves
each wheel's metadata as a file of its own (PEP 658), as the PyTorch indexes do. No
network."""

from __future__ import annotations

import hashlib
import os
import re
import threading
import zipfile
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

from test_dependency_replacement import (
    BACKEND_SRC,
    build_wheel,
    isolated_env,
    new_environment,
    run,
)

# Runs in the new environment as the dependency manager runs the installer (with
# progress): a and b in one install.
INSTALL_A_AND_B = """
import asyncio
import sys

sys.path.insert(0, sys.argv[1])
from dependencies import store
from dependencies.store import DependencyInfo


async def progress(message, *_args):
    print("progress:", message)


store.scan_installed_packages()
dependencies = [DependencyInfo(n, "1.0", display_name=f"Wheel {n}") for n in "ab"]
asyncio.run(store.install_dependencies(dependencies, progress))
"""


class Index(ThreadingHTTPServer):
    """A simple index (PEP 503) of the wheels in a folder, with their metadata files
    (PEP 658), that counts the GET requests of each path."""

    def __init__(self, wheels: Path):
        super().__init__(("127.0.0.1", 0), Handler)
        self.wheels = wheels
        self.gets: Counter[str] = Counter()
        self.lock = threading.Lock()


def wheel_metadata(wheel: Path) -> bytes:
    with zipfile.ZipFile(wheel) as archive:
        return archive.read(f"{wheel.name.split('-')[0]}-1.0.dist-info/METADATA")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        index = cast(Index, self.server)
        with index.lock:
            index.gets[self.path] += 1
        if page := re.fullmatch(r"/simple/([a-z]+)/", self.path):
            wheel = index.wheels / f"{page.group(1)}-1.0-py3-none-any.whl"
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            metadata = hashlib.sha256(wheel_metadata(wheel)).hexdigest()
            body = (
                f'<a href="/files/{wheel.name}#sha256={digest}"'
                f' data-dist-info-metadata="sha256={metadata}">{wheel.name}</a>'
            ).encode()
            content_type = "text/html"
        elif file := re.fullmatch(r"/files/([^/]+\.whl)(\.metadata)?", self.path):
            wheel = index.wheels / file.group(1)
            body = wheel_metadata(wheel) if file.group(2) else wheel.read_bytes()
            content_type = "application/octet-stream"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


def test_each_wheel_is_downloaded_once_and_named_in_the_progress(tmp_path: Path):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    build_wheel(wheels, "a", {"a.py": ""})
    build_wheel(wheels, "b", {"b.py": ""})
    script = tmp_path / "install_a_and_b.py"
    script.write_text(INSTALL_A_AND_B, encoding="utf-8")
    env = isolated_env()
    python = new_environment(tmp_path / "venv", env)

    index = Index(wheels)
    threading.Thread(target=index.serve_forever, daemon=True).start()
    try:
        # The index stands in for PyPI, and no pip configuration file of this
        # machine applies.
        env = {
            **env,
            "PIP_INDEX_URL": f"http://127.0.0.1:{index.server_address[1]}/simple/",
            "PIP_CONFIG_FILE": os.devnull,
        }
        output = run([python, script, BACKEND_SRC], env)
    finally:
        index.shutdown()
        index.server_close()

    assert {path: n for path, n in index.gets.items() if "/files/" in path} == {
        f"/files/{name}-1.0-py3-none-any.whl{suffix}": 1
        for name in "ab"
        for suffix in ("", ".metadata")
    }
    # chainner_pip downloads both wheels after it has collected b: the progress names
    # the wheel each download fetches.
    progress = [line for line in output.splitlines() if line.startswith("progress:")]
    after_collecting = progress[progress.index("progress: Collecting Wheel b...") :]
    assert "progress: Downloading Wheel a..." in after_collecting
