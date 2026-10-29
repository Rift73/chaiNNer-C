from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import threading
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from subprocess import check_call
from typing import BinaryIO

from sanic.log import logger

from system import is_arm_mac, is_linux, is_mac, is_windows

from . import numpy_pool


@dataclass(frozen=True)
class Archive:
    """One download of a pinned FFmpeg build: its SHA-256 and the path of each
    executable inside it, by the executable's file name."""

    url: str
    sha256: str
    members: dict[str, str]


@dataclass(frozen=True)
class Pin:
    version: str
    archives: tuple[Archive, ...]


_BTBN = "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-09-30-13-08/ffmpeg-n9.0.2-17-g2a571b6068"
_EVERMEET = "https://evermeet.cx/ffmpeg"
_MARTIN_RIEDL = "https://ffmpeg.martin-riedl.de/download/macos/arm64/1789931890_9.0.2"

# The FFmpeg each platform downloads. BtbN keeps the last build of each month for two
# years (daily builds for 14 days), so the pin is a month's last build.
PINS: dict[str, Pin] = {
    "win32-x64": Pin(
        "n9.0.2-17-g2a571b6068",
        (
            Archive(
                f"{_BTBN}-win64-gpl-9.0.zip",
                "a0e45723c72141975f51d8666302e614711745f3102b704ca3f82c897a58d278",
                {
                    "ffmpeg.exe": "ffmpeg-n9.0.2-17-g2a571b6068-win64-gpl-9.0/bin/ffmpeg.exe",
                    "ffprobe.exe": "ffmpeg-n9.0.2-17-g2a571b6068-win64-gpl-9.0/bin/ffprobe.exe",
                },
            ),
        ),
    ),
    "linux-x64": Pin(
        "n9.0.2-17-g2a571b6068",
        (
            Archive(
                f"{_BTBN}-linux64-gpl-9.0.tar.xz",
                "68ee646831adaae2495618346f3bba94ff207ff83bbd34d643e7004730d66269",
                {
                    "ffmpeg": "ffmpeg-n9.0.2-17-g2a571b6068-linux64-gpl-9.0/bin/ffmpeg",
                    "ffprobe": "ffmpeg-n9.0.2-17-g2a571b6068-linux64-gpl-9.0/bin/ffprobe",
                },
            ),
        ),
    ),
    "darwin-x64": Pin(
        "9.0.2",
        (
            Archive(
                f"{_EVERMEET}/ffmpeg-9.0.2.zip",
                "4acc0be580f9b2788029eb7bd4d645ff87968911b0a62aeeb3940d42d54558d5",
                {"ffmpeg": "ffmpeg"},
            ),
            Archive(
                f"{_EVERMEET}/ffprobe-9.0.2.zip",
                "24a9c968cd4da72d99c7245e914b921815835eb6dff01d99868031aebaf1d439",
                {"ffprobe": "ffprobe"},
            ),
        ),
    ),
    "darwin-arm64": Pin(
        "9.0.2",
        (
            Archive(
                f"{_MARTIN_RIEDL}/ffmpeg.zip",
                "c8ed4c4e6978a03c485edbfe4e0a5dc2380f8a30bba5150531b31b094492d924",
                {"ffmpeg": "ffmpeg"},
            ),
            Archive(
                f"{_MARTIN_RIEDL}/ffprobe.zip",
                "fcbe839537485eaee7a7a8bc5cbc0f90d53617e80943e8a5b2e31cb851197ea6",
                {"ffprobe": "ffprobe"},
            ),
        ),
    ),
}

# Size and SHA-256 of the executables in the unversioned FFmpeg 5.1.2 builds earlier
# versions downloaded (github.com/chaiNNer-org/ffmpeg-rehost, release "ffmpeg"), which
# carry no marker. A copy that equals one is replaced like an older pinned build; the
# sizes tell other copies apart without reading them.
REHOST: dict[str, dict[str, tuple[int, str]]] = {
    "win32-x64": {
        "ffmpeg.exe": (
            78925824,
            "98094b9736637916bd98cc8ee7546d9d3638763545f153c6daa9dcf87b43d10e",
        ),
        "ffprobe.exe": (
            78828544,
            "702c7b51e4ac3001c7f8b451c9f03dca9f0e7e5f15d127e750e7a33fee2997f7",
        ),
    },
    "linux-x64": {
        "ffmpeg": (
            78502560,
            "024e91a47bdcdaee12edbcad106c6db9543d74b791e473f1a5bbba6d5f3a5cc5",
        ),
        "ffprobe": (
            78400704,
            "b279033f6203e2d5bf210e60d81b6f164fbde55fd5e77db55e63860c1e4655d3",
        ),
    },
    "darwin-x64": {
        "ffmpeg": (
            78200980,
            "9d500ca67a32094bdc1fdf3c9820f21be5b43a571bfc8b50b4d579829a0e487d",
        ),
        "ffprobe": (
            78138996,
            "5fe92ddc819d815e5f5db9e6991384bea95d8fc9b74012ebcc128c4f6dd15627",
        ),
    },
    "darwin-arm64": {
        "ffmpeg": (
            42093320,
            "345107d72ea1142d873842ec85456f7a7e80be5458ae554ac82e3ae2725735dd",
        ),
        "ffprobe": (
            78138996,
            "5fe92ddc819d815e5f5db9e6991384bea95d8fc9b74012ebcc128c4f6dd15627",
        ),
    },
}

# Written next to the executables of an FFmpeg this program installed: the version,
# the platform and the SHA-256 of each executable.
MARKER = "chainner-ffmpeg.json"

# Seconds a download waits to connect, and for each next piece of data.
DOWNLOAD_TIMEOUT = (10.0, 30.0)


def get_platform() -> str:
    if is_windows:
        return "win32-x64"
    if is_arm_mac:
        return "darwin-arm64"
    if is_mac:
        return "darwin-x64"
    if is_linux:
        return "linux-x64"

    raise Exception("Unsupported platform")


def get_executable_path(base_dir: Path) -> tuple[Path, Path]:
    if is_windows:
        return base_dir / "ffmpeg.exe", base_dir / "ffprobe.exe"

    return base_dir / "ffmpeg", base_dir / "ffprobe"


def is_set_up(base_dir: Path) -> bool:
    """Whether the folder holds both FFmpeg executables."""
    return all(path.is_file() for path in get_executable_path(base_dir))


def _executable_sha256(base_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in get_executable_path(base_dir):
        with path.open("rb") as f:
            hashes[path.name] = hashlib.file_digest(f, "sha256").hexdigest()
    return hashes


def _read_marker(base_dir: Path) -> dict | None:
    path = base_dir / MARKER
    if not path.is_file():
        return None
    marker = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(marker, dict):
        raise ValueError(f"{path} holds no JSON object")
    return marker


def _holds_pin(base_dir: Path) -> bool:
    """Whether base_dir holds the pinned build, by its marker."""
    if not is_set_up(base_dir):
        return False
    marker = _read_marker(base_dir)
    platform = get_platform()
    return (
        marker is not None
        and marker.get("platform") == platform
        and marker.get("version") == PINS[platform].version
    )


def _version_key(version: str) -> tuple[int, ...]:
    """A build's version as numbers that compare in release order: n9.0.2-17-g2a571b6068
    (17 commits after the n9.0.2 tag) is (9, 0, 2, 17), 9.0.2 is (9, 0, 2)."""
    match = re.fullmatch(r"n?(\d+(?:\.\d+)*)(?:-(\d+)-g[0-9a-f]+)?", version)
    if match is None:
        raise ValueError(f"Unknown FFMPEG version {version!r}")
    numbers = [int(n) for n in match[1].split(".")]
    if match[2] is not None:
        numbers.append(int(match[2]))
    return tuple(numbers)


def needs_update(base_dir: Path) -> bool:
    """Whether the FFmpeg set up in base_dir gives way to the pinned build: an older one
    this program installed (its marker's hashes equal the files and its version is
    older than the pin's) or the 5.1.2 build earlier versions downloaded. One as new as
    the pin or newer, and one the user put there (a marker the files differ from, or no
    marker and other files), are kept."""
    platform = get_platform()
    version = PINS[platform].version
    try:
        if _holds_pin(base_dir):
            return False
        marker = _read_marker(base_dir)
        newer = (
            marker is not None
            and marker.get("platform") == platform
            and _version_key(str(marker.get("version"))) > _version_key(version)
        )
    except (OSError, ValueError) as e:
        logger.warning(
            f"Keeping the FFMPEG in {base_dir}: its {MARKER} is unreadable ({e})"
        )
        return False

    if marker is not None and newer:
        logger.info(
            f"Keeping the FFMPEG {marker.get('version')} in {base_dir}: it is newer"
            f" than {version}"
        )
        return False
    if marker is None:
        rehost = REHOST[platform]
        paths = get_executable_path(base_dir)
        if all(p.stat().st_size == rehost[p.name][0] for p in paths) and (
            _executable_sha256(base_dir) == {n: h for n, (_, h) in rehost.items()}
        ):
            logger.info(f"Updating the FFMPEG 5.1.2 in {base_dir} to {version}")
            return True
        logger.info(f"Keeping the FFMPEG in {base_dir}: it was put there by the user")
        return False
    if marker.get("sha256") == _executable_sha256(base_dir):
        logger.info(
            f"Updating the FFMPEG {marker.get('version')} in {base_dir} to {version}"
        )
        return True
    logger.info(
        f"Keeping the FFMPEG in {base_dir}: the user replaced the"
        f" {marker.get('version')} build installed there"
    )
    return False


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
            ffmpeg_path.chmod(0o755)
            ffprobe_path.chmod(0o755)
        except Exception as e:
            logger.warn(f"Failed to set permissions for ffmpeg: {e}")


def _move_into_place(extracted: Path, base_dir: Path) -> None:
    """Renames the extracted folder to base_dir. A base_dir in place (an older build,
    or one an earlier version left incomplete when it extracted in place) is set aside
    first and removed afterwards, and put back if the rename fails; the pinned build
    another process put there meanwhile is kept."""
    if _holds_pin(base_dir):
        logger.info("FFMPEG was set up by another process")
        return
    stale: Path | None = None
    if base_dir.exists():
        stale = Path(
            tempfile.mkdtemp(
                dir=base_dir.parent, prefix=f"{base_dir.name}.", suffix=".old"
            )
        )
        try:
            os.rename(base_dir, stale / base_dir.name)
        except OSError:
            stale.rmdir()
            raise
    try:
        os.rename(extracted, base_dir)
    except OSError:
        if not _holds_pin(base_dir):
            if stale is not None and not base_dir.exists():
                os.rename(stale / base_dir.name, base_dir)
                stale.rmdir()
            raise
        logger.info("FFMPEG was set up by another process")
    if stale is not None:

        def failed(_function: object, path: str, error: BaseException) -> None:
            logger.warning(f"Unable to remove {path} of an earlier FFMPEG: {error}")

        shutil.rmtree(stale, onexc=failed)


def _download(url: str, sha256: str, file: BinaryIO) -> None:
    """Writes the download to file and checks its SHA-256. A server that does not
    connect, or stops sending, within DOWNLOAD_TIMEOUT fails the download."""
    # Imported here, for the download alone: Load Video and Save Video import this
    # module among the nodes every start loads first, and requests takes about 80 ms.
    import requests

    digest = hashlib.sha256()
    try:
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1 << 20):
                digest.update(chunk)
                file.write(chunk)
    except requests.RequestException as e:
        raise RuntimeError(
            f"Unable to download FFMPEG from {url} ({e}); nothing was installed."
            " Check the internet connection and run again."
        ) from e
    if digest.hexdigest() != sha256:
        raise RuntimeError(
            f"The FFMPEG download {url} has the SHA-256 {digest.hexdigest()},"
            f" not the expected {sha256}; nothing was installed."
        )


def _extract(archive: Archive, path: Path, folder: Path) -> None:
    """Copies the archive's executables, alone, into folder."""
    names = {member: name for name, member in archive.members.items()}
    if archive.url.endswith(".tar.xz"):
        # read as a stream: one pass over the compressed data
        with tarfile.open(path, "r|xz") as tar:
            for info in tar:
                wanted = info.isfile() and info.name in names
                source = tar.extractfile(info) if wanted else None
                if source is not None:
                    with source, (folder / names[info.name]).open("wb") as f:
                        shutil.copyfileobj(source, f, 1 << 20)
        return
    with zipfile.ZipFile(path) as zip_file:
        for info in zip_file.infolist():
            if info.filename in names:
                target = folder / names[info.filename]
                with zip_file.open(info) as source, target.open("wb") as f:
                    shutil.copyfileobj(source, f, 1 << 20)


def setup_integrated_ffmpeg(base_dir: Path):
    """Downloads the pinned FFmpeg into base_dir (the install folder, or the staging
    folder beside it), all or nothing: each archive downloads to a .part file, is
    checked against its SHA-256 and has its executables extracted into a temporary
    folder next to base_dir, which gets the marker and becomes base_dir by a rename
    once both executables are in it. A setup that stops midway or a download whose
    SHA-256 differs installs nothing (the next use starts over) and leaves an FFmpeg in
    place as it was; the .part and .tmp of a process that a crash ended are removed at
    the backend's next start (leftovers.py)."""
    platform = get_platform()
    pin = PINS[platform]
    parent = base_dir.parent
    parent.mkdir(exist_ok=True, parents=True)
    prefix = f"{base_dir.name}.{os.getpid()}-"
    extracted = Path(tempfile.mkdtemp(dir=parent, prefix=prefix, suffix=".tmp"))
    try:
        for archive in pin.archives:
            handle, download_name = tempfile.mkstemp(
                dir=parent, prefix=prefix, suffix=".part"
            )
            download_path = Path(download_name)
            try:
                logger.info(f"Downloading FFMPEG {pin.version} from {archive.url}...")
                with os.fdopen(handle, "wb") as f:
                    _download(archive.url, archive.sha256, f)
                logger.info("Extracting FFMPEG...")
                _extract(archive, download_path, extracted)
            finally:
                download_path.unlink(missing_ok=True)

        ffmpeg_path, ffprobe_path = get_executable_path(extracted)
        missing = [p.name for p in (ffmpeg_path, ffprobe_path) if not p.is_file()]
        if missing:
            raise RuntimeError(f"The FFMPEG download lacks {' and '.join(missing)}.")

        marker = {
            "version": pin.version,
            "platform": platform,
            "sha256": _executable_sha256(extracted),
        }
        (extracted / MARKER).write_text(json.dumps(marker, indent=2), encoding="utf-8")
        _make_executable(ffmpeg_path, ffprobe_path)
        _move_into_place(extracted, base_dir)
    finally:
        if extracted.exists():
            shutil.rmtree(extracted)


def _staging_dir(base_dir: Path) -> Path:
    """Where the pinned build waits, beside base_dir, while an older one is in use."""
    return base_dir.with_name(f"{base_dir.name}.staged")


def _put_staged_in_place(base_dir: Path) -> bool:
    """Moves the pinned build staged beside base_dir into base_dir (replacing an FFmpeg
    there): whether one was staged. The rename fails while a process runs the FFmpeg in
    base_dir (Windows); the staged build then stays for the next try."""
    staged = _staging_dir(base_dir)
    if not _holds_pin(staged):
        return False
    _move_into_place(staged, base_dir)
    logger.info(f"Moved the FFMPEG staged in {staged} into {base_dir}")
    return True


def _stage(base_dir: Path) -> None:
    """Downloads the pinned build into the staging folder beside base_dir, for a later
    start to put in place; a failure is logged and the next start tries again."""
    staged = _staging_dir(base_dir)
    try:
        setup_integrated_ffmpeg(staged)
    except Exception:
        logger.exception(f"Failed to prepare the FFMPEG update in {staged}")
        return
    logger.info(f"The FFMPEG update in {staged} replaces {base_dir} at the next start")


_setup_lock = threading.Lock()
_setup_future: Future[None] | None = None
# The folders whose FFmpeg this process has checked (needs_update) and set up if needed.
_checked: set[Path] = set()


def run_setup(base_dir: Path):
    """Checks the FFmpeg in base_dir once per process, once for every thread that needs
    FFmpeg meanwhile: the first starts the check in a new thread, all of them wait for
    it. With no FFmpeg set up, the pinned build is put in place: the staged one if it
    waits beside base_dir, else a download (retried once); if that fails, every waiting
    thread gets the error and the next use starts over. An FFmpeg in place that needs
    an update stays in use: the staged build replaces it now when the folder can be
    renamed, else (or with none staged) the update waits for a later start, and with
    none staged the pinned build downloads in the background meanwhile."""

    def set_up():
        if _put_staged_in_place(base_dir):
            return
        try:
            setup_integrated_ffmpeg(base_dir)
            return
        except Exception as e:
            logger.warn(f"Failed to setup FFMPEG: {e}")
            logger.warn("Trying again...")

        setup_integrated_ffmpeg(base_dir)

    def task():
        global _setup_future
        checked = False
        try:
            if not is_set_up(base_dir):
                set_up()
            elif needs_update(base_dir):
                try:
                    if not _put_staged_in_place(base_dir):
                        threading.Thread(
                            target=_stage,
                            args=(base_dir,),
                            daemon=True,
                            name="FFMPEG update",
                        ).start()
                except Exception:
                    logger.exception(
                        f"Failed to update the FFMPEG in {base_dir}; it stays in use"
                        " until a later start"
                    )
            checked = True
        finally:
            with _setup_lock:
                _setup_future = None
                if checked:
                    _checked.add(base_dir)

    global _setup_future
    with _setup_lock:
        future = _setup_future
        if future is None:
            if base_dir in _checked and is_set_up(base_dir):
                # checked by the setup this thread waited for the lock behind
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
        elif base_dir not in _checked:
            run_setup(base_dir)

        return FFMpegEnv(
            ffmpeg=str(ffmpeg_path),
            ffprobe=str(ffprobe_path),
        )
