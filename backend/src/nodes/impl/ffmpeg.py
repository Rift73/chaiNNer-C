from __future__ import annotations

import os
import shutil
import tempfile
import threading
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from subprocess import check_call

import requests
from sanic.log import logger

from system import is_arm_mac, is_linux, is_mac, is_windows

from . import numpy_pool


def get_download_url() -> str:
    if is_windows:
        return "https://github.com/chaiNNer-org/ffmpeg-rehost/releases/download/ffmpeg/ffmpeg-win32-x64.zip"

    if is_arm_mac:
        return "https://github.com/chaiNNer-org/ffmpeg-rehost/releases/download/ffmpeg/ffmpeg-darwin-arm64.zip"
    if is_mac:
        return "https://github.com/chaiNNer-org/ffmpeg-rehost/releases/download/ffmpeg/ffmpeg-darwin-x64.zip"

    if is_linux:
        return "https://github.com/chaiNNer-org/ffmpeg-rehost/releases/download/ffmpeg/ffmpeg-linux-x64.zip"

    raise Exception("Unsupported platform")


def get_executable_path(base_dir: Path) -> tuple[Path, Path]:
    if is_windows:
        return base_dir / "ffmpeg.exe", base_dir / "ffprobe.exe"

    return base_dir / "ffmpeg", base_dir / "ffprobe"


def is_set_up(base_dir: Path) -> bool:
    """Whether the folder holds both FFmpeg executables."""
    return all(path.is_file() for path in get_executable_path(base_dir))


def _make_executable(ffmpeg_path: Path, ffprobe_path: Path) -> None:
    logger.info("Settings permissions for FFMPEG...")
    if is_mac:
        # Un-Quarantine ffmpeg on macOS
        try:
            check_call(["xattr", "-dr", "com.apple.quarantine", str(ffmpeg_path)])
            check_call(["xattr", "-dr", "com.apple.quarantine", str(ffprobe_path)])
        except Exception as e:
            logger.warn(f"Failed to un-quarantine ffmpeg: {e}")

    if is_arm_mac:
        # M1 can only run signed files, we must ad-hoc sign it
        try:
            check_call(["xattr", "-cr", str(ffmpeg_path)])
            check_call(["xattr", "-cr", str(ffprobe_path)])
            check_call(["codesign", "-s", "-", str(ffmpeg_path)])
            check_call(["codesign", "-s", "-", str(ffprobe_path)])
        except Exception as e:
            logger.warn(f"Failed to sign ffmpeg: {e}")

    if is_mac or is_linux:
        # Make the files executable
        try:
            ffmpeg_path.chmod(0o7777)
            ffprobe_path.chmod(0o7777)
        except Exception as e:
            logger.warn(f"Failed to set permissions for ffmpeg: {e}")


def _move_into_place(extracted: Path, base_dir: Path) -> None:
    """Renames the extracted folder to base_dir. A base_dir an earlier version left
    incomplete (it extracted in place) is set aside first and removed afterwards; one
    another process completed meanwhile is kept."""
    if is_set_up(base_dir):
        logger.info("FFMPEG was set up by another process")
        return
    stale: Path | None = None
    if base_dir.exists():
        stale = Path(
            tempfile.mkdtemp(
                dir=base_dir.parent, prefix=f"{base_dir.name}.", suffix=".old"
            )
        )
        os.rename(base_dir, stale / base_dir.name)
    try:
        os.rename(extracted, base_dir)
    except OSError:
        if not is_set_up(base_dir):
            raise
        logger.info("FFMPEG was set up by another process")
    if stale is not None:

        def failed(_function: object, path: str, error: BaseException) -> None:
            logger.warning(f"Unable to remove {path} of an incomplete FFMPEG: {error}")

        shutil.rmtree(stale, onexc=failed)


def setup_integrated_ffmpeg(base_dir: Path):
    """Downloads FFmpeg into base_dir, all or nothing: the archive downloads to a .part
    file and is extracted into a temporary folder next to base_dir, which becomes
    base_dir by a rename once both executables are in it. A setup that stops midway
    leaves no base_dir (the next use starts over), never a cut-off ffmpeg that exists."""
    url = get_download_url()
    parent = base_dir.parent
    parent.mkdir(exist_ok=True, parents=True)
    prefix = f"{base_dir.name}.{os.getpid()}-"
    handle, download_name = tempfile.mkstemp(
        dir=parent, prefix=prefix, suffix=".zip.part"
    )
    download_path = Path(download_name)
    extracted: Path | None = None
    try:
        logger.info("Downloading FFMPEG...")
        with os.fdopen(handle, "wb") as f, requests.get(url, stream=True) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("Extracting FFMPEG...")
        extracted = Path(tempfile.mkdtemp(dir=parent, prefix=prefix, suffix=".tmp"))
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(extracted)
        ffmpeg_path, ffprobe_path = get_executable_path(extracted)
        missing = [p.name for p in (ffmpeg_path, ffprobe_path) if not p.is_file()]
        if missing:
            raise RuntimeError(f"The FFMPEG download lacks {' and '.join(missing)}.")

        _make_executable(ffmpeg_path, ffprobe_path)
        _move_into_place(extracted, base_dir)
    finally:
        download_path.unlink(missing_ok=True)
        if extracted is not None and extracted.exists():
            shutil.rmtree(extracted)


_setup_lock = threading.Lock()
_setup_future: Future[None] | None = None


def run_setup(base_dir: Path):
    """Sets up FFmpeg in base_dir once for every thread that needs it meanwhile: the
    first starts the setup in a new thread, all of them wait for it. A failed setup is
    retried once; if that fails too, every waiting thread gets the error and the next
    use starts a new setup (nothing failed is kept until a restart)."""

    def task():
        global _setup_future
        try:
            try:
                setup_integrated_ffmpeg(base_dir)
                return
            except Exception as e:
                logger.warn(f"Failed to setup FFMPEG: {e}")
                logger.warn("Trying again...")

            setup_integrated_ffmpeg(base_dir)
        finally:
            with _setup_lock:
                _setup_future = None

    global _setup_future
    with _setup_lock:
        future = _setup_future
        if future is None:
            if is_set_up(base_dir):
                # set up by the setup this thread waited for the lock behind
                return
            executor = ThreadPoolExecutor(max_workers=1, initializer=numpy_pool.install)
            future = _setup_future = executor.submit(task)
            executor.shutdown(wait=False)

    future.result()


@dataclass(frozen=True)
class FFMpegEnv:
    """
    Paths to FFMPEG binaries.
    """

    ffmpeg: str
    ffprobe: str

    @staticmethod
    def from_env() -> FFMpegEnv:
        return FFMpegEnv(
            ffmpeg="ffmpeg",
            ffprobe="ffprobe",
        )

    @staticmethod
    def get_integrated(base_dir: Path) -> FFMpegEnv:
        base_dir = base_dir.resolve() / "ffmpeg"
        ffmpeg_path, ffprobe_path = get_executable_path(base_dir)

        if not is_set_up(base_dir):
            logger.info(f"Integrated FFMPEG not found at {base_dir}")
            run_setup(base_dir)

        return FFMpegEnv(
            ffmpeg=str(ffmpeg_path),
            ffprobe=str(ffprobe_path),
        )
