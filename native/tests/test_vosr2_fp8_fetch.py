"""VOSR fetches its pinned files once, resumably: the official model and the TensorRT FP8
graph set, which it then compiles once."""

from __future__ import annotations

import errno
import hashlib
import json
import socket
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from api import Aborted, NodeContext
from nodes.impl.pytorch import vosr2_download, vosr2_engine_cache, vosr2_fp8
from nodes.impl.pytorch.vosr2 import BundlePaths
from nodes.impl.pytorch.vosr2_trt_models import STAGES

CHUNK = 1024


@dataclass
class Host:
    files: dict[str, bytes]
    ranges: bool = True
    drop_after: dict[str, int] = field(default_factory=dict)
    corrupt: set[str] = field(default_factory=set)
    log: list[tuple[str, str | None]] = field(default_factory=list)
    url: str = ""


class Server(ThreadingHTTPServer):
    def __init__(self, host: Host):
        super().__init__(("127.0.0.1", 0), Handler)
        self.host = host


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        host = cast(Server, self.server).host
        name = self.path.rsplit("/", 1)[-1]
        host.log.append((name, self.headers.get("Range")))
        data = host.files.get(self.path.removeprefix("/"))
        if data is None:
            self.send_error(404)
            return
        if name in host.corrupt:
            data = bytes(255 - value for value in data)
        start = 0
        requested = self.headers.get("Range")
        if requested and host.ranges:
            start = int(requested.removeprefix("bytes=").split("-")[0])
            self.send_response(206)
            self.send_header(
                "Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}"
            )
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        limit = host.drop_after.pop(name, None)
        if limit is not None:
            # A broken connection: fewer bytes than announced.
            self.wfile.write(body[:limit])
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


@dataclass
class Bundle:
    paths: BundlePaths
    hashes: dict[str, str]


def make_bundle(root: Path) -> Bundle:
    paths = BundlePaths(root)
    hashes = {}
    for path in paths.files():
        path.parent.mkdir(parents=True, exist_ok=True)
        name = path.relative_to(root).as_posix()
        path.write_bytes(name.encode())
        hashes[name] = hashlib.sha256(name.encode()).hexdigest()
    return Bundle(paths, hashes)


def make_set(bundle: Bundle) -> tuple[bytes, dict[str, bytes]]:
    files = {"LICENSE": b"Apache License", "NOTICE": b"VOSR authors"}
    stages = {}
    for index, name in enumerate(STAGES):
        files[f"{name}.onnx"] = f"graph {name}".encode()
        stages[name] = {"precision": "test", "graph": f"{name}.onnx"}
        if index < 3:
            # Several chunks, so cancellation and resumption happen mid-file.
            files[f"{name}.onnx.data"] = bytes(range(256)) * (16 + index)
            stages[name]["data"] = f"{name}.onnx.data"
    manifest = {
        "format": 1,
        "tensorrt_major": 11,
        "plugins": [{"name": "VosrQkvPack", "version": "1", "namespace": ""}],
        "source": {
            "repository": "test/VOSR",
            "revision": "f24b3061b7f350b81e1907bdcfc27f71fb4ff3f3",
            "files": bundle.hashes,
        },
        "stages": stages,
        "files": {
            name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in files.items()
        },
    }
    return json.dumps(manifest, indent=2).encode(), files


@pytest.fixture
def host(monkeypatch, tmp_path) -> Iterator[Host]:
    manifest, files = make_set(make_bundle(tmp_path / "bundle"))
    digest = hashlib.sha256(manifest).hexdigest()
    monkeypatch.setattr(vosr2_fp8, "MANIFEST_SHA256", digest)
    monkeypatch.setattr(vosr2_download, "CHUNK", CHUNK)
    folder = vosr2_fp8.folder_name()
    state = Host({f"{folder}/{name}": data for name, data in files.items()})
    state.files[f"{folder}/manifest.json"] = manifest
    server = Server(state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_address[1]}/"
    monkeypatch.setattr(vosr2_fp8, "HOST", state.url)
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def context_for(tmp_path: Path, progress: list[float], abort_after: int = -1):
    calls = [0]

    def suspend() -> None:
        calls[0] += 1
        if calls[0] == abort_after:
            raise Aborted()

    return cast(
        NodeContext,
        SimpleNamespace(
            storage_dir=tmp_path / "storage",
            node_id="fp8",
            suspend=suspend,
            check_aborted=lambda: None,
            set_progress=progress.append,
        ),
    )


def fetch(tmp_path: Path, context: NodeContext) -> Path:
    folder = tmp_path / "storage" / "graphs"
    return vosr2_fp8.fetch_graphs(context, folder, vosr2_fp8.fetch_manifest(folder))


def served(host: Host) -> dict[str, bytes]:
    return {
        path.rsplit("/", 1)[-1]: data
        for path, data in host.files.items()
        if not path.endswith("manifest.json")
    }


def assert_complete(folder: Path, host: Host) -> None:
    for name, data in served(host).items():
        assert (folder / name).read_bytes() == data
    assert not list(folder.glob("*.part"))


def test_downloads_verified_files_with_progress_and_only_once(host, tmp_path):
    progress: list[float] = []
    folder = fetch(tmp_path, context_for(tmp_path, progress))
    assert_complete(folder, host)
    assert (folder / "manifest.json").is_file()
    assert progress == sorted(progress) and progress[-1] == pytest.approx(1.0)
    requests = len(host.log)
    assert fetch(tmp_path, context_for(tmp_path, [])) == folder
    assert len(host.log) == requests


@pytest.mark.parametrize("ranges", [True, False])
def test_a_broken_download_resumes_or_restarts_by_server_support(
    host, tmp_path, ranges
):
    host.ranges = ranges
    host.drop_after["dit_cached.onnx.data"] = 2500
    with pytest.raises(RuntimeError, match="Check the internet connection"):
        fetch(tmp_path, context_for(tmp_path, []))
    folder = tmp_path / "storage" / "graphs"
    assert (folder / "dit_kv.onnx.data").is_file()
    partial = (folder / "dit_cached.onnx.data.part").stat().st_size
    assert 0 < partial <= 2500
    progress: list[float] = []
    fetch(tmp_path, context_for(tmp_path, progress))
    assert_complete(folder, host)
    assert ("dit_cached.onnx.data", f"bytes={partial}-") in host.log
    assert progress[-1] == pytest.approx(1.0)


def test_a_hash_mismatch_is_deleted_and_reported(host, tmp_path):
    host.corrupt.add("dino.onnx.data")
    with pytest.raises(
        RuntimeError, match=r"dino\.onnx\.data does not match its SHA-256"
    ):
        fetch(tmp_path, context_for(tmp_path, []))
    folder = tmp_path / "storage" / "graphs"
    assert not (folder / "dino.onnx.data").exists()
    assert not (folder / "dino.onnx.data.part").exists()
    host.corrupt.clear()
    fetch(tmp_path, context_for(tmp_path, []))
    assert_complete(folder, host)


def test_a_manifest_with_another_hash_is_refused(host, monkeypatch, tmp_path):
    expected = vosr2_fp8.folder_name()
    monkeypatch.setattr(vosr2_fp8, "MANIFEST_SHA256", "0" * 64)
    monkeypatch.setattr(vosr2_fp8, "folder_name", lambda: expected)
    with pytest.raises(RuntimeError, match="does not match the SHA-256"):
        fetch(tmp_path, context_for(tmp_path, []))
    assert not (tmp_path / "storage" / "graphs").exists()


def test_cancel_keeps_the_partial_file_for_the_next_run(host, tmp_path):
    with pytest.raises(Aborted):
        # The first check is the download lock's; the sixth falls in the middle of a file.
        fetch(tmp_path, context_for(tmp_path, [], abort_after=6))
    folder = tmp_path / "storage" / "graphs"
    assert 0 < (folder / "dit_kv.onnx.data.part").stat().st_size
    assert not (folder / "dit_kv.onnx.data").exists()
    fetch(tmp_path, context_for(tmp_path, []))
    assert_complete(folder, host)
    assert any(name == "dit_kv.onnx.data" and span for name, span in host.log)


def test_offline_fails_clearly(monkeypatch, tmp_path):
    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
    monkeypatch.setattr(vosr2_fp8, "HOST", f"http://127.0.0.1:{port}/")
    with pytest.raises(RuntimeError, match="Check the internet connection"):
        fetch(tmp_path, context_for(tmp_path, []))


def test_missing_files_report_http_status(host, tmp_path):
    del host.files[f"{vosr2_fp8.folder_name()}/NOTICE"]
    with pytest.raises(RuntimeError, match="answered HTTP 404"):
        fetch(tmp_path, context_for(tmp_path, []))


def test_disk_space_is_checked_before_and_during_the_download(
    host, monkeypatch, tmp_path
):
    context = context_for(tmp_path, [])
    usage = vosr2_download.shutil.disk_usage
    monkeypatch.setattr(
        vosr2_download.shutil, "disk_usage", lambda _: SimpleNamespace(free=1_000_000)
    )
    with pytest.raises(RuntimeError, match="GB are free"):
        fetch(tmp_path, context)
    monkeypatch.setattr(vosr2_download.shutil, "disk_usage", usage)
    original = Path.open

    def full_disk(self: Path, mode: str = "r", *args, **kwargs):
        if self.suffix == ".part" and mode in ("ab", "wb"):
            raise OSError(errno.ENOSPC, "No space left on device")
        return original(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", full_disk)
    with pytest.raises(RuntimeError, match="is full"):
        fetch(tmp_path, context)


def test_engines_are_built_once_and_keyed_by_what_determines_them(
    host, monkeypatch, tmp_path
):
    bundle = BundlePaths(tmp_path / "bundle")
    context = context_for(tmp_path, [])
    identity = {"graphs": vosr2_fp8.MANIFEST_SHA256, "plugin_sha256": "a"}
    capability = [12, 0]
    builds = []

    def compile_worker(context, worker, graphs, directory, device):
        builds.append(worker)
        assert_complete(graphs, host)
        engines = {}
        for name in STAGES:
            (directory / f"{name}.plan").write_bytes(name.encode())
            engines[name] = hashlib.sha256(name.encode()).hexdigest()
        (directory / "validation.json").write_text(json.dumps({"engines": engines}))

    monkeypatch.setattr(vosr2_fp8, "read_configs", lambda _: None)
    monkeypatch.setattr(vosr2_fp8, "check_runtime", lambda _: None)
    monkeypatch.setattr(vosr2_fp8, "compile_worker", compile_worker)
    monkeypatch.setattr(vosr2_fp8, "fp8_identity", lambda _: dict(identity))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            name="Test GPU", major=capability[0], minor=capability[1]
        ),
    )
    device = torch.device("cuda:0")
    path = vosr2_fp8.ensure_fp8_engine(context, bundle, device)
    assert {p.stem for p in path.glob("*.plan")} == set(STAGES)
    assert builds == ["vosr2_fp8_build.py"]
    requests = len(host.log)
    assert vosr2_fp8.ensure_fp8_engine(context, bundle, device) == path
    assert len(builds) == 1 and len(host.log) == requests
    identity["plugin_sha256"] = "b"
    assert vosr2_fp8.ensure_fp8_engine(context, bundle, device) != path
    assert len(builds) == 2 and len(host.log) == requests
    capability[:] = [8, 6]
    with pytest.raises(ValueError, match="FP8 support"):
        vosr2_fp8.ensure_fp8_engine(context, bundle, device)
    assert len(builds) == 2


def test_runtime_check_names_the_tensorrt_major_and_the_plugin(monkeypatch):
    trt = pytest.importorskip("tensorrt")
    monkeypatch.setattr(vosr2_fp8, "load_plugin", lambda: None)
    plugin = {"name": "VosrQkvPack", "version": "1", "namespace": ""}
    manifest = {"tensorrt_major": 99, "plugins": [plugin]}
    with pytest.raises(RuntimeError, match="need TensorRT 99"):
        vosr2_fp8.check_runtime(manifest)
    manifest["tensorrt_major"] = int(trt.__version__.split(".")[0])
    monkeypatch.setattr(vosr2_fp8, "plugin_path", lambda: Path("vosr2_qkv.dll"))
    monkeypatch.setattr(
        trt,
        "get_plugin_registry",
        lambda: SimpleNamespace(get_creator=lambda *args: None),
    )
    with pytest.raises(RuntimeError, match="does not provide VosrQkvPack version 1"):
        vosr2_fp8.check_runtime(manifest)


# The official model: the same pinned download into the storage folder.


def model_files() -> dict[str, bytes]:
    """Small stand-ins for the five files, under their remote names."""
    return {
        remote: (local * (40 if local.endswith(".safetensors") else 1)).encode()
        for remote, local, _, _ in vosr2_download.FILES
    }


@pytest.fixture
def model_host(monkeypatch) -> Iterator[Host]:
    files = model_files()
    pins = tuple(
        (remote, local, len(files[remote]), hashlib.sha256(files[remote]).hexdigest())
        for remote, local, _, _ in vosr2_download.FILES
    )
    monkeypatch.setattr(vosr2_download, "FILES", pins)
    monkeypatch.setattr(vosr2_download, "CHUNK", 64)
    state = Host(
        {f"{vosr2_download.REVISION}/{name}": data for name, data in files.items()}
    )
    server = Server(state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_address[1]}/"
    monkeypatch.setattr(vosr2_download, "MODEL_HOST", state.url)
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def assert_model(paths: BundlePaths, tmp_path: Path) -> None:
    folder = tmp_path / "storage" / "vosr2-model" / vosr2_download.REVISION
    assert paths == BundlePaths(folder)
    files = model_files()
    for remote, local, _, _ in vosr2_download.FILES:
        assert (folder / local).read_bytes() == files[remote]
    assert not list(folder.rglob("*.part"))


def test_the_model_downloads_into_the_storage_folder_with_progress(
    model_host, tmp_path
):
    progress: list[float] = []
    paths = vosr2_download.ensure_model(context_for(tmp_path, progress))
    assert_model(paths, tmp_path)
    assert {local for _, local, _, _ in vosr2_download.FILES} == {
        path.relative_to(paths.root).as_posix() for path in paths.files()
    }
    assert progress == sorted(progress) and progress[-1] == pytest.approx(1.0)


def test_a_broken_model_download_resumes(model_host, tmp_path):
    model_host.drop_after["ema_model.safetensors"] = 300
    with pytest.raises(RuntimeError, match="Check the internet connection"):
        vosr2_download.ensure_model(context_for(tmp_path, []))
    folder = tmp_path / "storage" / "vosr2-model" / vosr2_download.REVISION
    partial = (folder / "checkpoints/ema_model.safetensors.part").stat().st_size
    assert 0 < partial <= 300
    assert_model(vosr2_download.ensure_model(context_for(tmp_path, [])), tmp_path)
    assert ("ema_model.safetensors", f"bytes={partial}-") in model_host.log


def test_a_model_file_that_does_not_match_its_pin_is_deleted(model_host, tmp_path):
    model_host.corrupt.add("dinov2_vitl14_pretrain.pth")
    with pytest.raises(
        RuntimeError, match=r"dinov2_vitl14_pretrain\.pth does not match its SHA-256"
    ):
        vosr2_download.ensure_model(context_for(tmp_path, []))
    folder = tmp_path / "storage" / "vosr2-model" / vosr2_download.REVISION
    assert not list(folder.glob("dinov2_vitl14_pretrain.pth*"))
    model_host.corrupt.clear()
    assert_model(vosr2_download.ensure_model(context_for(tmp_path, [])), tmp_path)


def test_two_processes_download_the_model_once(model_host, tmp_path):
    # Two backends share the storage folder (two windows, or the app and a CLI run).
    # The second waits for the first's download, then finds the published files; the two
    # never write one partial file together, which published a mix of both streams.
    folder = tmp_path / "storage" / "vosr2-model" / vosr2_download.REVISION
    paused, finished = threading.Event(), threading.Event()
    errors: list[BaseException] = []

    def suspend() -> None:
        if not paused.is_set() and list(folder.rglob("*.part")):
            paused.set()
            finished.wait(timeout=1)  # the second's turn, if it could take one

    first = cast(
        NodeContext,
        SimpleNamespace(
            storage_dir=tmp_path / "storage",
            node_id="fp8",
            suspend=suspend,
            check_aborted=lambda: None,
            set_progress=lambda _: None,
        ),
    )

    def run(context: NodeContext, wait: threading.Event | None) -> None:
        try:
            if wait is not None:
                wait.wait(timeout=10)
            assert_model(vosr2_download.ensure_model(context), tmp_path)
        except BaseException as error:
            errors.append(error)
        finally:
            if wait is not None:
                finished.set()

    threads = [
        threading.Thread(target=run, args=(first, None)),
        threading.Thread(target=run, args=(context_for(tmp_path, []), paused)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert paused.is_set() and errors == []
    assert_model(vosr2_download.ensure_model(context_for(tmp_path, [])), tmp_path)
    names = [name for name, _ in model_host.log]
    assert sorted(names) == sorted(set(names))  # each file requested once


def test_present_model_files_are_hashed_once_and_never_downloaded(
    model_host, monkeypatch, tmp_path
):
    folder = tmp_path / "storage" / "vosr2-model" / vosr2_download.REVISION
    files = model_files()
    for remote, local, _, _ in vosr2_download.FILES:
        (folder / local).parent.mkdir(parents=True, exist_ok=True)
        (folder / local).write_bytes(files[remote])
    hashed = []
    file_hash = vosr2_engine_cache.file_hash

    def counted(path: Path) -> str:
        hashed.append(path)
        return file_hash(path)

    monkeypatch.setattr(vosr2_engine_cache, "file_hash", counted)
    for _ in range(2):
        # A new process: only the hashes stored next to the model remain.
        vosr2_engine_cache._HASHES.clear()
        assert_model(vosr2_download.ensure_model(context_for(tmp_path, [])), tmp_path)
    assert len(hashed) == len(files)
    assert model_host.log == []
