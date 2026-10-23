"""The built-in FFmpeg's setup (FFMpegEnv.get_integrated) downloads the pinned build of
the platform, checks each archive's SHA-256 before it extracts anything, and is all or
nothing and can be retried: a stopped download or extraction leaves no FFmpeg folder and
no temporary files, a failed setup is not kept until a restart, a server that stops
answering fails the download after a timeout, and threads that need FFmpeg at the same
time share one download. An older build this program installed, or the 5.1.2 build
earlier versions downloaded, stays in use while the pinned build downloads in the
background into a staging folder beside it, which replaces it at a later start once its
folder can be renamed; an FFmpeg the user put there, or a newer one, is kept. The
download is a stand-in serving a local archive, except for a server on 127.0.0.1 that
never answers and the opt-in test of the real download (CHAINNER_C_NETWORK_TESTS=1)."""

from __future__ import annotations

import errno
import hashlib
import io
import json
import logging
import os
import shutil
import socket
import subprocess
import tarfile
import threading
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest
import requests

from nodes.impl import ffmpeg
from nodes.impl.ffmpeg import (
    MARKER,
    PINS,
    Archive,
    FFMpegEnv,
    Pin,
    get_executable_path,
    get_platform,
)

FFMPEG_NAME, FFPROBE_NAME = (p.name for p in get_executable_path(Path()))
PLATFORM = get_platform()
# An FFmpeg build a user put in chaiNNer's storage (read only: the test copies it), named by
# CHAINNER_C_USER_FFMPEG_DIR; without it the case skips.
USER_FFMPEG = Path(os.environ.get("CHAINNER_C_USER_FFMPEG_DIR", "user-ffmpeg-not-set"))


def executable(name: str, version: str) -> bytes:
    return f"{name} {version} executable".encode() * 1000


def archive(
    version: str = "1", names: tuple[str, ...] = (FFMPEG_NAME, FFPROBE_NAME)
) -> bytes:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as zip_file:
        for name in names:
            zip_file.writestr(name, executable(name, version))
    return data.getvalue()


class Download:
    """Stands in for requests.get: serves `data` in chunks and counts the downloads."""

    def __init__(self) -> None:
        self.data = b""
        self.count = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def __call__(
        self, _url: str, stream: bool, timeout: tuple[float, float]
    ) -> Download:
        assert stream
        assert timeout == ffmpeg.DOWNLOAD_TIMEOUT
        self.count += 1
        self.started.set()
        assert self.release.wait(10)
        return self

    def __enter__(self) -> Download:
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        for start in range(0, len(self.data), chunk_size):
            yield self.data[start : start + chunk_size]


class Serve:
    """Pins version `version` of a stand-in build and serves `data` as its download,
    by default an archive of that version with both executables at its root."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, download: Download) -> None:
        self.monkeypatch = monkeypatch
        self.download = download

    def __call__(
        self,
        version: str = "1",
        data: bytes | None = None,
        url: str = "https://example.invalid/ffmpeg.zip",
        members: dict[str, str] | None = None,
    ) -> None:
        self.download.data = archive(version) if data is None else data
        sha256 = hashlib.sha256(self.download.data).hexdigest()
        members = members or {FFMPEG_NAME: FFMPEG_NAME, FFPROBE_NAME: FFPROBE_NAME}
        pin = Pin(version, (Archive(url, sha256, members),))
        self.monkeypatch.setitem(PINS, PLATFORM, pin)


@pytest.fixture
def download(monkeypatch: pytest.MonkeyPatch) -> Download:
    fake = Download()
    # The setup imports requests when it downloads.
    monkeypatch.setattr(requests, "get", fake)
    monkeypatch.setattr(ffmpeg, "_setup_future", None)
    monkeypatch.setattr(ffmpeg, "_checked", set())
    return fake


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch, download: Download) -> Serve:
    serve = Serve(monkeypatch, download)
    serve()
    return serve


def names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def installed(tmp_path: Path) -> list[str]:
    return names(tmp_path / "ffmpeg")


def new_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """The next use checks the folder again, as at the next start."""
    monkeypatch.setattr(ffmpeg, "_checked", set())


def staging_finished() -> None:
    """Waits for the background download of an update."""
    for thread in threading.enumerate():
        if thread.name == "FFMPEG update":
            thread.join(10)
            assert not thread.is_alive()


def staged(tmp_path: Path) -> list[str]:
    return names(tmp_path / "ffmpeg.staged")


def marker(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "ffmpeg" / MARKER).read_text(encoding="utf-8"))


def put(tmp_path: Path, version: str, with_marker: dict | None = None) -> None:
    """Puts an FFmpeg into the storage by hand, as a user or an earlier version did."""
    folder = tmp_path / "ffmpeg"
    folder.mkdir()
    for name in (FFMPEG_NAME, FFPROBE_NAME):
        (folder / name).write_bytes(executable(name, version))
    if with_marker is not None:
        (folder / MARKER).write_text(json.dumps(with_marker), encoding="utf-8")


def sha256_of(version: str) -> dict[str, str]:
    return {
        name: hashlib.sha256(executable(name, version)).hexdigest()
        for name in (FFMPEG_NAME, FFPROBE_NAME)
    }


def test_the_setup_puts_both_executables_and_the_marker_in_place(
    tmp_path: Path, download: Download, serve: Serve
):
    env = FFMpegEnv.get_integrated(tmp_path)

    assert download.count == 1
    assert Path(env.ffmpeg).read_bytes() == executable(FFMPEG_NAME, "1")
    assert Path(env.ffprobe).is_file()
    # no download or extraction folder is left beside it
    assert names(tmp_path) == ["ffmpeg"]
    assert installed(tmp_path) == sorted([FFMPEG_NAME, FFPROBE_NAME, MARKER])
    assert marker(tmp_path) == {
        "version": "1",
        "platform": PLATFORM,
        "sha256": sha256_of("1"),
    }

    FFMpegEnv.get_integrated(tmp_path)
    assert download.count == 1


def test_a_tar_xz_download_with_the_executables_in_a_folder_is_installed(
    tmp_path: Path, download: Download, serve: Serve
):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:xz") as tar:
        for name in ("README.txt", FFMPEG_NAME, FFPROBE_NAME):
            content = executable(name, "1")
            info = tarfile.TarInfo(f"ffmpeg-build/bin/{name}")
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    serve(
        data=data.getvalue(),
        url="https://example.invalid/ffmpeg.tar.xz",
        members={
            name: f"ffmpeg-build/bin/{name}" for name in (FFMPEG_NAME, FFPROBE_NAME)
        },
    )

    env = FFMpegEnv.get_integrated(tmp_path)

    assert Path(env.ffprobe).read_bytes() == executable(FFPROBE_NAME, "1")
    # the archive's other files stay out
    assert installed(tmp_path) == sorted([FFMPEG_NAME, FFPROBE_NAME, MARKER])


def test_a_download_with_another_sha256_installs_nothing(
    tmp_path: Path, download: Download, serve: Serve
):
    download.data = archive("tampered")

    with pytest.raises(RuntimeError, match=r"SHA-256.*nothing was installed"):
        FFMpegEnv.get_integrated(tmp_path)
    assert download.count == 2  # the setup and its immediate retry
    assert names(tmp_path) == []


def test_a_stopped_extraction_leaves_nothing_and_the_next_use_retries(
    tmp_path: Path, download: Download, serve: Serve, monkeypatch: pytest.MonkeyPatch
):
    copy = shutil.copyfileobj
    failures = 2  # the setup and its immediate retry

    def stopped_extraction(
        source: io.BufferedIOBase, target: io.BufferedIOBase, length: int
    ) -> None:
        nonlocal failures
        if failures == 0:
            copy(source, target, length)
            return
        failures -= 1
        target.write(b"cut off")
        raise OSError("[Errno 28] No space left on device")

    monkeypatch.setattr(shutil, "copyfileobj", stopped_extraction)

    with pytest.raises(OSError, match="No space left"):
        FFMpegEnv.get_integrated(tmp_path)
    assert download.count == 2
    assert names(tmp_path) == []

    env = FFMpegEnv.get_integrated(tmp_path)
    assert download.count == 3
    assert Path(env.ffmpeg).read_bytes() != b"cut off"
    assert names(tmp_path) == ["ffmpeg"]


def test_a_download_without_ffprobe_is_not_put_in_place(
    tmp_path: Path, download: Download, serve: Serve
):
    serve(data=archive("1", (FFMPEG_NAME,)))

    with pytest.raises(RuntimeError, match=f"lacks {FFPROBE_NAME}"):
        FFMpegEnv.get_integrated(tmp_path)
    assert names(tmp_path) == []


def test_an_incomplete_folder_of_an_earlier_version_is_replaced(
    tmp_path: Path, download: Download, serve: Serve
):
    # earlier versions extracted in place: a stopped setup left ffmpeg without ffprobe
    folder = tmp_path / "ffmpeg"
    folder.mkdir()
    (folder / FFMPEG_NAME).write_bytes(b"cut off")
    (folder / "_ffmpeg.zip").write_bytes(b"cut off")

    env = FFMpegEnv.get_integrated(tmp_path)

    assert download.count == 1
    assert Path(env.ffmpeg).read_bytes() != b"cut off"
    assert names(tmp_path) == ["ffmpeg"]
    assert installed(tmp_path) == sorted([FFMPEG_NAME, FFPROBE_NAME, MARKER])


def test_threads_that_need_ffmpeg_together_share_one_download(
    tmp_path: Path, download: Download, serve: Serve
):
    download.release.clear()
    results: list[FFMpegEnv] = []

    def use() -> None:
        results.append(FFMpegEnv.get_integrated(tmp_path))

    first = threading.Thread(target=use)
    first.start()
    assert download.started.wait(10)
    second = threading.Thread(target=use)
    second.start()
    # the second thread waits for the download the first started
    second.join(0.2)
    assert second.is_alive()
    download.release.set()
    first.join(10)
    second.join(10)

    assert download.count == 1
    assert len(results) == 2 and results[0] == results[1]
    assert names(tmp_path) == ["ffmpeg"]


# At first use, by the folder's marker and the executables' SHA-256:


def test_the_pinned_build_is_used(
    tmp_path: Path, download: Download, serve: Serve, monkeypatch: pytest.MonkeyPatch
):
    FFMpegEnv.get_integrated(tmp_path)
    new_process(monkeypatch)

    FFMpegEnv.get_integrated(tmp_path)

    assert download.count == 1


def test_an_older_build_stays_in_use_until_the_update_replaces_it_at_a_later_start(
    tmp_path: Path, download: Download, serve: Serve, monkeypatch: pytest.MonkeyPatch
):
    FFMpegEnv.get_integrated(tmp_path)
    new_process(monkeypatch)
    serve("2")

    env = FFMpegEnv.get_integrated(tmp_path)
    staging_finished()
    FFMpegEnv.get_integrated(tmp_path)

    # this run keeps the older build; the update waits beside it, downloaded once
    assert download.count == 2
    assert Path(env.ffmpeg).read_bytes() == executable(FFMPEG_NAME, "1")
    assert marker(tmp_path)["version"] == "1"
    assert names(tmp_path) == ["ffmpeg", "ffmpeg.staged"]
    assert staged(tmp_path) == sorted([FFMPEG_NAME, FFPROBE_NAME, MARKER])

    new_process(monkeypatch)
    env = FFMpegEnv.get_integrated(tmp_path)

    assert download.count == 2
    assert Path(env.ffmpeg).read_bytes() == executable(FFMPEG_NAME, "2")
    assert marker(tmp_path)["version"] == "2"
    # neither the older build nor the staging folder is left beside it
    assert names(tmp_path) == ["ffmpeg"]


def test_the_5_1_2_build_earlier_versions_downloaded_is_replaced_at_a_later_start(
    tmp_path: Path, download: Download, serve: Serve, monkeypatch: pytest.MonkeyPatch
):
    put(tmp_path, "5.1.2")
    rehost = {
        name: (len(executable(name, "5.1.2")), sha256)
        for name, sha256 in sha256_of("5.1.2").items()
    }
    monkeypatch.setitem(ffmpeg.REHOST, PLATFORM, rehost)

    env = FFMpegEnv.get_integrated(tmp_path)
    staging_finished()

    assert Path(env.ffmpeg).read_bytes() == executable(FFMPEG_NAME, "5.1.2")
    assert names(tmp_path) == ["ffmpeg", "ffmpeg.staged"]

    new_process(monkeypatch)
    env = FFMpegEnv.get_integrated(tmp_path)

    assert download.count == 1
    assert Path(env.ffmpeg).read_bytes() == executable(FFMPEG_NAME, "1")
    assert marker(tmp_path)["version"] == "1"
    assert names(tmp_path) == ["ffmpeg"]


@pytest.mark.parametrize(
    ("pin", "version", "kept"),
    [
        ("n9.0.2-17-g2a571b6068", "n9.0.2-18-g0123abc", True),
        ("n9.0.2-17-g2a571b6068", "n9.1", True),
        ("n9.0.2-17-g2a571b6068", "n10.0-3-gabcdef1", True),
        ("9.0.2", "9.0.10", True),
        ("n9.0.2-17-g2a571b6068", "n9.0.2", False),
        ("n9.0.2-17-g2a571b6068", "n9.0.2-16-g0123abc", False),
        ("n9.0.2-17-g2a571b6068", "n8.1-5-gabcdef1", False),
        ("9.0.2", "9.0.1", False),
    ],
)
def test_a_build_this_program_installed_is_replaced_only_when_older_than_the_pin(
    tmp_path: Path,
    download: Download,
    serve: Serve,
    caplog: pytest.LogCaptureFixture,
    pin: str,
    version: str,
    kept: bool,
):
    # a build from another install of this program that shares the storage
    put(
        tmp_path,
        version,
        {"version": version, "platform": PLATFORM, "sha256": sha256_of(version)},
    )
    serve(pin)

    with caplog.at_level(logging.INFO):
        env = FFMpegEnv.get_integrated(tmp_path)
    staging_finished()

    assert Path(env.ffmpeg).read_bytes() == executable(FFMPEG_NAME, version)
    assert download.count == (0 if kept else 1)
    assert names(tmp_path) == (["ffmpeg"] if kept else ["ffmpeg", "ffmpeg.staged"])
    newer = [r.message for r in caplog.records if "newer than" in r.message]
    assert len(newer) == (1 if kept else 0)


def test_a_staged_build_is_put_in_place_without_a_download(
    tmp_path: Path, download: Download, serve: Serve, monkeypatch: pytest.MonkeyPatch
):
    put(tmp_path, "0", {"version": "0", "platform": PLATFORM, "sha256": sha256_of("0")})
    FFMpegEnv.get_integrated(tmp_path)
    staging_finished()
    # the older build is gone before the next start
    shutil.rmtree(tmp_path / "ffmpeg")
    new_process(monkeypatch)

    env = FFMpegEnv.get_integrated(tmp_path)

    assert download.count == 1
    assert Path(env.ffmpeg).read_bytes() == executable(FFMPEG_NAME, "1")
    assert names(tmp_path) == ["ffmpeg"]


def test_a_blocked_swap_keeps_the_staged_build_until_the_folder_is_free(
    tmp_path: Path,
    download: Download,
    serve: Serve,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    put(tmp_path, "0", {"version": "0", "platform": PLATFORM, "sha256": sha256_of("0")})
    FFMpegEnv.get_integrated(tmp_path)
    staging_finished()
    folder = (tmp_path / "ffmpeg").resolve()
    rename = os.rename

    def blocked(source: str | Path, target: str | Path) -> None:
        # Windows refuses to rename a folder while a process runs an executable in it
        if Path(source) == folder:
            raise PermissionError(errno.EACCES, "Access is denied", str(source))
        rename(source, target)

    monkeypatch.setattr(os, "rename", blocked)
    for _start in range(2):
        new_process(monkeypatch)
        caplog.clear()
        with caplog.at_level(logging.INFO):
            env = FFMpegEnv.get_integrated(tmp_path)
            FFMpegEnv.get_integrated(tmp_path)

        assert Path(env.ffmpeg).read_bytes() == executable(FFMPEG_NAME, "0")
        # the staged build is kept, not downloaded again, and the failure logged once
        assert download.count == 1
        assert names(tmp_path) == ["ffmpeg", "ffmpeg.staged"]
        assert staged(tmp_path) == sorted([FFMPEG_NAME, FFPROBE_NAME, MARKER])
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(errors) == 1 and "stays in use" in errors[0].message
        assert errors[0].exc_info is not None

    monkeypatch.setattr(os, "rename", rename)
    new_process(monkeypatch)
    env = FFMpegEnv.get_integrated(tmp_path)

    assert download.count == 1
    assert Path(env.ffmpeg).read_bytes() == executable(FFMPEG_NAME, "1")
    assert names(tmp_path) == ["ffmpeg"]


@pytest.fixture
def stalled_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[socket.socket]:
    """A server on 127.0.0.1 that accepts connections and never answers, pinned as
    the download (a proxy setting must not take the request elsewhere)."""
    server = socket.create_server(("127.0.0.1", 0))
    url = f"http://127.0.0.1:{server.getsockname()[1]}/ffmpeg.zip"
    members = {FFMPEG_NAME: FFMPEG_NAME, FFPROBE_NAME: FFPROBE_NAME}
    monkeypatch.setitem(PINS, PLATFORM, Pin("2", (Archive(url, "0" * 64, members),)))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setattr(ffmpeg, "_setup_future", None)
    monkeypatch.setattr(ffmpeg, "_checked", set())
    # a download without a timeout would wait forever: the server stops after 20 s
    stop = threading.Timer(20, server.close)
    stop.start()
    with server:
        yield server
    stop.cancel()


def test_a_stalled_server_fails_a_new_setup_with_a_clear_error(
    tmp_path: Path, stalled_server: socket.socket, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(ffmpeg, "DOWNLOAD_TIMEOUT", (5.0, 0.5))
    start = time.perf_counter()

    with pytest.raises(
        RuntimeError, match=r"Unable to download FFMPEG from http://127"
    ):
        FFMpegEnv.get_integrated(tmp_path)

    # the setup and its retry, each after the read timeout
    assert time.perf_counter() - start < 10
    assert names(tmp_path) == []


def test_a_stalled_server_does_not_hold_up_the_older_build(
    tmp_path: Path,
    stalled_server: socket.socket,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    monkeypatch.setattr(ffmpeg, "DOWNLOAD_TIMEOUT", (5.0, 60.0))
    put(tmp_path, "1", {"version": "1", "platform": PLATFORM, "sha256": sha256_of("1")})
    start = time.perf_counter()

    with caplog.at_level(logging.INFO):
        env = FFMpegEnv.get_integrated(tmp_path)
        seconds = time.perf_counter() - start
        # the update still waits for the server's answer
        waiting = [
            t
            for t in threading.enumerate()
            if t.name == "FFMPEG update" and t.is_alive()
        ]
        # the server stops: the waiting download fails
        stalled_server.close()
        staging_finished()

    assert seconds < 5 and len(waiting) == 1
    assert Path(env.ffmpeg).read_bytes() == executable(FFMPEG_NAME, "1")
    assert names(tmp_path) == ["ffmpeg"]
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert (
        len(errors) == 1 and "Failed to prepare the FFMPEG update" in errors[0].message
    )
    assert errors[0].exc_info is not None


@pytest.fixture
def hashed(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The names of the files hashed."""
    names: list[str] = []
    file_digest = hashlib.file_digest

    def counted(file: io.BufferedReader, digest: str) -> hashlib._Hash:
        names.append(Path(file.name).name)
        return file_digest(file, digest)

    monkeypatch.setattr(hashlib, "file_digest", counted)
    return names


def test_a_build_the_user_replaced_is_kept_and_logged_once(
    tmp_path: Path,
    download: Download,
    serve: Serve,
    hashed: list[str],
    caplog: pytest.LogCaptureFixture,
):
    # the marker of an older build this program installed, the user's executables
    put(
        tmp_path,
        "user",
        {"version": "0", "platform": PLATFORM, "sha256": sha256_of("0")},
    )

    with caplog.at_level(logging.INFO):
        env = FFMpegEnv.get_integrated(tmp_path)
        FFMpegEnv.get_integrated(tmp_path)

    assert download.count == 0
    assert Path(env.ffmpeg).read_bytes() == executable(FFMPEG_NAME, "user")
    # checked once in a process: the files are hashed and the decision logged once
    assert sorted(hashed) == sorted([FFMPEG_NAME, FFPROBE_NAME])
    folder = tmp_path.resolve() / "ffmpeg"
    assert [r.message for r in caplog.records if "Keeping" in r.message] == [
        f"Keeping the FFMPEG in {folder}: the user replaced the 0 build installed there"
    ]


def test_an_ffmpeg_without_marker_the_user_put_there_is_kept(
    tmp_path: Path, download: Download, serve: Serve, hashed: list[str]
):
    put(tmp_path, "user")

    env = FFMpegEnv.get_integrated(tmp_path)

    assert download.count == 0
    assert Path(env.ffmpeg).read_bytes() == executable(FFMPEG_NAME, "user")
    assert installed(tmp_path) == sorted([FFMPEG_NAME, FFPROBE_NAME])
    # its sizes differ from the 5.1.2 build's: it is not read
    assert hashed == []


@pytest.mark.skipif(
    not all(p.is_file() for p in get_executable_path(USER_FFMPEG)),
    reason="needs CHAINNER_C_USER_FFMPEG_DIR: a folder with a user's own FFmpeg build",
)
def test_a_users_ffmpeg_8_build_is_kept(
    tmp_path: Path, download: Download, serve: Serve
):
    folder = tmp_path / "ffmpeg"
    folder.mkdir()
    for path in get_executable_path(USER_FFMPEG):
        shutil.copyfile(path, folder / path.name)

    env = FFMpegEnv.get_integrated(tmp_path)

    assert download.count == 0
    assert installed(tmp_path) == sorted([FFMPEG_NAME, FFPROBE_NAME])
    assert (
        Path(env.ffmpeg).stat().st_size
        == get_executable_path(USER_FFMPEG)[0].stat().st_size
    )


def test_a_failed_update_keeps_the_older_build_in_use_and_the_next_start_retries(
    tmp_path: Path,
    download: Download,
    serve: Serve,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    FFMpegEnv.get_integrated(tmp_path)
    new_process(monkeypatch)
    serve("2")
    download.data = archive("tampered")  # or offline: no download at all

    with caplog.at_level(logging.INFO):
        env = FFMpegEnv.get_integrated(tmp_path)
        staging_finished()
        FFMpegEnv.get_integrated(tmp_path)

    assert download.count == 2  # the update, once in this process
    assert Path(env.ffmpeg).read_bytes() == executable(FFMPEG_NAME, "1")
    assert marker(tmp_path)["version"] == "1"
    assert names(tmp_path) == ["ffmpeg"]
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1 and "Failed to prepare" in errors[0].message
    assert errors[0].exc_info is not None

    new_process(monkeypatch)
    FFMpegEnv.get_integrated(tmp_path)
    staging_finished()
    assert download.count == 3


@pytest.mark.skipif(
    os.environ.get("CHAINNER_C_NETWORK_TESTS") != "1",
    reason="downloads the pinned FFmpeg (about 190 MB on Windows)",
)
def test_the_pinned_build_downloads_verifies_and_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(ffmpeg, "_setup_future", None)
    monkeypatch.setattr(ffmpeg, "_checked", set())
    start = time.perf_counter()

    env = FFMpegEnv.get_integrated(tmp_path)

    seconds = time.perf_counter() - start
    version = subprocess.run(
        [env.ffmpeg, "-hide_banner", "-version"],
        capture_output=True,
        text=True,
        check=True,
        stdin=subprocess.DEVNULL,
    ).stdout
    print(f"set up in {seconds:.1f} s: {version.splitlines()[0]}")
    assert PINS[PLATFORM].version.lstrip("n") in version
    assert marker(tmp_path)["version"] == PINS[PLATFORM].version
    assert installed(tmp_path) == sorted([FFMPEG_NAME, FFPROBE_NAME, MARKER])
