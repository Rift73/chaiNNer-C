"""The built-in FFmpeg's setup (FFMpegEnv.get_integrated) is all or nothing and can be
retried: a stopped download or extraction leaves no FFmpeg folder and no temporary
files, a failed setup is not kept until a restart, and threads that need FFmpeg at the
same time share one download. The download is a stand-in serving a local zip."""

from __future__ import annotations

import io
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from nodes.impl import ffmpeg
from nodes.impl.ffmpeg import FFMpegEnv, get_executable_path

FFMPEG_NAME, FFPROBE_NAME = (p.name for p in get_executable_path(Path()))


def archive(names: tuple[str, ...] = (FFMPEG_NAME, FFPROBE_NAME)) -> bytes:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as zip_file:
        for name in names:
            zip_file.writestr(name, f"{name} executable".encode() * 1000)
    return data.getvalue()


class Download:
    """Stands in for requests.get: serves `data` in chunks and counts the downloads."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.count = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def __call__(self, _url: str, stream: bool) -> Download:
        assert stream
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


@pytest.fixture
def download(monkeypatch: pytest.MonkeyPatch) -> Download:
    fake = Download(archive())
    monkeypatch.setattr(ffmpeg.requests, "get", fake)
    monkeypatch.setattr(ffmpeg, "_setup_future", None)
    return fake


def names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def test_the_setup_puts_both_executables_in_place(tmp_path: Path, download: Download):
    env = FFMpegEnv.get_integrated(tmp_path)

    assert download.count == 1
    assert Path(env.ffmpeg).read_bytes() == f"{FFMPEG_NAME} executable".encode() * 1000
    assert Path(env.ffprobe).is_file()
    # no download or extraction folder is left beside it
    assert names(tmp_path) == ["ffmpeg"]
    assert names(tmp_path / "ffmpeg") == sorted([FFMPEG_NAME, FFPROBE_NAME])

    FFMpegEnv.get_integrated(tmp_path)
    assert download.count == 1


def test_a_stopped_extraction_leaves_nothing_and_the_next_use_retries(
    tmp_path: Path, download: Download, monkeypatch: pytest.MonkeyPatch
):
    extract_all = zipfile.ZipFile.extractall
    failures = 2  # the setup and its immediate retry

    def stopped_extraction(self: zipfile.ZipFile, path: Path) -> None:
        nonlocal failures
        if failures == 0:
            extract_all(self, path)
            return
        failures -= 1
        (Path(path) / FFMPEG_NAME).write_bytes(b"cut off")
        raise OSError("[Errno 28] No space left on device")

    monkeypatch.setattr(zipfile.ZipFile, "extractall", stopped_extraction)

    with pytest.raises(OSError, match="No space left"):
        FFMpegEnv.get_integrated(tmp_path)
    assert download.count == 2
    assert names(tmp_path) == []

    env = FFMpegEnv.get_integrated(tmp_path)
    assert download.count == 3
    assert Path(env.ffmpeg).read_bytes() != b"cut off"
    assert names(tmp_path) == ["ffmpeg"]


def test_a_download_without_ffprobe_is_not_put_in_place(
    tmp_path: Path, download: Download
):
    download.data = archive((FFMPEG_NAME,))

    with pytest.raises(RuntimeError, match=f"lacks {FFPROBE_NAME}"):
        FFMpegEnv.get_integrated(tmp_path)
    assert names(tmp_path) == []


def test_an_incomplete_folder_of_an_earlier_version_is_replaced(
    tmp_path: Path, download: Download
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
    assert names(folder) == sorted([FFMPEG_NAME, FFPROBE_NAME])


def test_threads_that_need_ffmpeg_together_share_one_download(
    tmp_path: Path, download: Download
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
