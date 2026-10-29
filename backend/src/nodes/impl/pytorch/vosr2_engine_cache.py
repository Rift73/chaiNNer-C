"""Persistent, content-addressed VOSR TensorRT compilation with atomic publication."""

from __future__ import annotations

import errno
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

import psutil
import torch
from sanic.log import logger

from api import NodeContext
from gpu import MemoryUsage, get_vram_usage

from ...utils.utils import file_identity
from ..owned_job import OwnedJob
from .vosr2 import BundlePaths, VosrBundle, load_weights, read_configs
from .vosr2_official import FILES
from .vosr2_plan_file import compress, compressed, plain, set_aside, stored
from .vosr2_process import PARENT_PID_VARIABLE
from .vosr2_stage_cache import migrate_stage_cache, migration_leftover, stage_identity
from .vosr2_trt_compile import STAGE_LINE
from .vosr2_trt_plugin import plugin_path

_HASHES: dict[tuple, str] = {}


def file_hash(path: Path) -> str:
    identity = file_identity(path)
    if identity not in _HASHES:
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if file_identity(path) != identity:
            raise RuntimeError(f"File changed while hashing: {path}")
        _HASHES[identity] = digest
    return _HASHES[identity]


def write_json(path: Path, value: object) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    ) as output:
        json.dump(value, output, indent=2)
        temporary = Path(output.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if sys.platform == "win32":
    import msvcrt

    def _try_lock(descriptor: int) -> None:
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)

    def _unlock(descriptor: int) -> None:
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(descriptor: int) -> None:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(descriptor: int) -> None:
        fcntl.flock(descriptor, fcntl.LOCK_UN)


@contextmanager
def file_lock(path: Path, context: NodeContext, waiting_for: str) -> Iterator[None]:
    """Hold path's lock: other processes, and other threads of this one, wait for it.
    The operating system releases it when its holder dies, so a file left behind by a
    killed process is never held."""
    with path.open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        deadline = time.monotonic() + 1200
        while True:
            context.suspend()
            lock.seek(0)
            try:
                _try_lock(lock.fileno())
                break
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for {waiting_for}."
                    ) from error
                time.sleep(0.05)
        try:
            yield
        finally:
            lock.seek(0)
            _unlock(lock.fileno())


@contextmanager
def cache_lock(path: Path, context: NodeContext) -> Iterator[None]:
    with file_lock(path, context, "the VOSR compiler cache lock"):
        yield


def stored_record(path: Path, root: Path) -> Path:
    """Where stored_hash keeps path's SHA-256."""
    path_key = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
    return root / f"checkpoint-{path_key}.json"


def stored_hash(path: Path, root: Path) -> str:
    """SHA-256 remembered in root across sessions while the file is unchanged."""
    identity = list(file_identity(path))
    digest_path = stored_record(path, root)
    try:
        previous = json.loads(digest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        previous = None  # never stored, or damaged: hash the file again
    if (
        isinstance(previous, dict)
        and previous.get("identity") == identity
        and isinstance(previous.get("sha256"), str)
    ):
        return previous["sha256"]
    digest = file_hash(path)
    write_json(digest_path, {"identity": identity, "sha256": digest})
    return digest


def forget_hash(path: Path, root: Path | None = None) -> None:
    """Drop the SHA-256 that file_hash remembers for path in this process and, with
    root, the one stored_hash keeps there. Call it before deleting or replacing the
    file: a file system gives the inode of a deleted file to the next file it creates
    and stamps both with coarse times, so the new file can have the old one's identity
    and would be taken for it."""
    absolute = str(path.absolute())
    for identity in list(_HASHES):
        if identity[0] == absolute:
            _HASHES.pop(identity, None)
    if root is not None:
        stored_record(path, root).unlink(missing_ok=True)


def identity_key(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def architecture(*builder: Path) -> str:
    """The SHA-256 of the model code the BF16 graphs are exported from, then of the
    `builder` files that compile them.

    Of vosr2.py only the bundle's construction and weight loading count; its other
    definitions (options, image preparation, messages) never reach an engine.
    """
    sources = [
        *sorted((Path(__file__).resolve().parents[3] / "vendor/vosr2").glob("*.py")),
        Path(__file__).with_name("vosr2_export.py"),
        *builder,
    ]
    model = "".join(
        inspect.getsource(code) for code in (VosrBundle.__init__, load_weights)
    )
    return hashlib.sha256(
        "".join(
            [
                *(file_hash(path) for path in sources),
                hashlib.sha256(model.encode()).hexdigest(),
            ]
        ).encode()
    ).hexdigest()


def cache_identity(device: torch.device) -> dict:
    """The model code the BF16 graphs are exported from, and what exports and builds them.
    The driver and torch's CUDA build do not decide an engine.
    """
    import tensorrt as trt

    properties = torch.cuda.get_device_properties(device)
    return {
        # With the builder's configuration (as the FP8 set's identity has it)
        "architecture": architecture(Path(__file__).with_name("vosr2_trt_compile.py")),
        "torch": str(torch.__version__),
        "onnx": metadata.version("onnx"),
        "tensorrt": trt.__version__,
        "gpu": properties.name,
        "capability": [properties.major, properties.minor],
    }


class StageLog:
    """A build worker's stages as its output names them (vosr2_trt_compile.report_stage):
    each stage's start and end are logged, with its duration, and its start is the
    node's progress."""

    def __init__(self, context: NodeContext, label: str):
        self.context = context
        self.label = label
        self.unread = b""
        # The running stage: its number, the stage count and when it started
        self.running: tuple[int, int, float] | None = None

    def read(self, output: bytes) -> None:
        """Take more of the worker's output; a stage line counts once it is whole."""
        *lines, self.unread = (self.unread + output).split(b"\n")
        for line in lines:
            stage = STAGE_LINE.fullmatch(line.decode("utf-8", "replace").rstrip("\r"))
            if stage is None:
                continue
            self.finish()
            index, total = int(stage[1]), int(stage[2])
            logger.info(
                "VOSR %s build: stage %d of %d, %s", self.label, index, total, stage[3]
            )
            self.context.set_progress((index - 1) / total)
            self.running = (index, total, time.monotonic())

    def finish(self) -> None:
        """Log the end of the running stage, if one is."""
        if self.running is not None:
            index, total, started = self.running
            logger.info(
                "VOSR %s build: stage %d of %d done in %.1f s",
                self.label,
                index,
                total,
                time.monotonic() - started,
            )
            self.running = None


def compile_worker(
    context: NodeContext,
    worker: str | Path,
    source: Path,
    directory: Path,
    device: torch.device,
    *,
    label: str = "TensorRT",
    limit: int = 1200,
    arguments: Sequence[str] = (),
) -> None:
    """Run a stage compiler, a file next to this module or at a given path: worker
    <source> <directory> <gpu> <cache> [arguments], for at most `limit` seconds. The
    cache folder, the parent of every building-* directory, keeps the worker's TensorRT
    timing cache (vosr2_trt_compile.timing_cache), which only builds that hold the
    folder's lock read or write. `label` names the compiler in messages."""
    script = worker if isinstance(worker, Path) else Path(__file__).with_name(worker)
    with torch.cuda.device(device):
        index = torch.cuda.current_device()

    def current_usage() -> MemoryUsage:
        usage = get_vram_usage(index)
        if usage is None:
            raise RuntimeError(
                f"VOSR {label} compilation requires working GPU memory monitoring."
            )
        return usage

    usage = current_usage()
    memory = usage.total
    # Compilation creates a second CUDA context and temporarily resident weights.
    # Refuse insufficient headroom before a large non-torch allocation can happen.
    if usage.used + 5_500_000_000 >= memory:
        raise RuntimeError(
            f"Not enough free GPU memory to compile VOSR {label}. "
            "Close other GPU workloads and run again."
        )
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    executable = Path(sys.executable)
    creationflags = 0
    if os.name == "nt":
        # Match multiprocessing's Windows venv workaround: own the interpreter,
        # not the redirector whose child can outlive its termination.
        executable = Path(getattr(sys, "_base_executable", sys.executable))
        if executable != Path(sys.executable):
            environment["__PYVENV_LAUNCHER__"] = sys.executable
        # Assign before the Windows venv launcher can spawn its interpreter.
        creationflags = subprocess.CREATE_NO_WINDOW | 0x00000004
    else:
        # The compiler checks, before its slow imports, that this process still lives.
        environment[PARENT_PID_VARIABLE] = str(os.getpid())
    job = OwnedJob()
    process = None
    started = time.monotonic()
    log_path = directory / "build.log"
    logger.info("Compiling VOSR %s once; progress and errors: %s", label, log_path)
    try:
        with log_path.open("wb") as log, log_path.open("rb") as progress:
            stages = StageLog(context, label)
            process = subprocess.Popen(
                [
                    str(executable),
                    "-B",
                    str(script),
                    str(source),
                    str(directory),
                    str(index),
                    str(directory.parent),
                    *arguments,
                ],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
                # Outside Windows the job is the process group the compiler leads.
                start_new_session=os.name != "nt",
            )
            job.assign(process)
            if os.name == "nt":
                psutil.Process(process.pid).resume()
            while process.poll() is None:
                # A build cannot be safely suspended inside CUDA. Keep policing
                # memory and cancellation while paused; inference waits later.
                context.check_aborted()
                stages.read(progress.read())
                if current_usage().used >= memory - 100_000_000:
                    raise RuntimeError(
                        f"VOSR {label} compilation filled the GPU memory. "
                        "Close other GPU workloads and run again."
                    )
                if time.monotonic() - started > limit:
                    raise TimeoutError(
                        f"VOSR {label} compilation exceeded {limit // 60} minutes."
                    )
                time.sleep(0.05)
            # What the worker printed after the last look, before it exited
            stages.read(progress.read())
        if process.returncode:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"VOSR {label} compilation failed:\n{tail}")
        stages.finish()
        logger.info("VOSR %s build: done in %.0f s", label, time.monotonic() - started)
    finally:
        job.close()
        if process is not None:
            process.wait(timeout=20)


def build(
    context: NodeContext, paths: BundlePaths, directory: Path, device: torch.device
) -> None:
    compile_worker(context, "vosr2_trt_build.py", paths.root, directory, device)


def ensure_engine_set(
    context: NodeContext,
    root: Path,
    identity: dict,
    label: str,
    compile_into: Callable[[Path], None],
    *,
    stages: Sequence[str] | None = None,
    compression: bool = True,
) -> Path:
    """Caller owns the cache lock. Publish a set only after every stage is validated.
    `stages` names the set's plans (default: the TensorRT engines' STAGES), which it
    keeps compressed unless `compression` is off (a plan loaded from its file)."""
    from .vosr2_trt_models import STAGES
    from .vosr2_trt_pipeline import release_full_session

    if stages is None:
        stages = STAGES
    key = identity_key(identity)
    destination = root / key
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            valid = (
                isinstance(manifest, dict)
                and manifest.get("identity") == identity
                and isinstance(manifest.get("engines"), dict)
                and set(manifest["engines"]) == set(stages)
                and all(
                    manifest["engines"][name]
                    == file_hash(stored(destination, manifest, name))
                    for name in stages
                )
            )
        except (FileNotFoundError, json.JSONDecodeError):
            valid = False
        if valid:
            if compression and manifest.get("compression") != "zstd":
                compress_set(destination, manifest, stages)
            return destination
        set_aside(destination, label)
    release_full_session(context.node_id)
    torch.cuda.empty_cache()
    destination.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="building-", dir=root) as work:
        directory = Path(work).resolve()
        if not directory.is_relative_to(root.resolve()):
            raise RuntimeError("VOSR build directory escaped its cache.")
        try:
            compile_into(directory)
            validation = json.loads(
                (directory / "validation.json").read_text(encoding="utf-8")
            )
            if set(validation["engines"]) != set(stages) or any(
                validation["engines"][name] != file_hash(directory / f"{name}.plan")
                for name in stages
            ):
                raise RuntimeError(f"VOSR {label} stages changed after validation.")
            if not compression:
                for name in stages:
                    os.replace(plain(directory, name), plain(destination, name))
                write_json(manifest_path, {"identity": identity, **validation})
                return destination
            engines = {}
            for name in stages:
                target = compressed(destination, name)
                compress(directory / f"{name}.plan", target)
                engines[name] = file_hash(target)
            write_json(
                manifest_path,
                {
                    "identity": identity,
                    **validation,
                    "engines": engines,
                    "plans": validation["engines"],
                    "compression": "zstd",
                },
            )
        finally:
            if (directory / "build.log").exists():
                os.replace(directory / "build.log", destination / "build.log")
    return destination


def remove_tree(path: Path) -> None:
    def failed(_function: object, name: str, error: BaseException) -> None:
        logger.warning("Could not delete %s: %s", name, error)

    if path.exists():
        shutil.rmtree(path, onexc=failed)


# What another GPU's set differs in (TensorRT: name and capability; MIGraphX: name and
# gfx target). A machine with two GPUs keeps a set for each.
GPU_FIELDS = ("gpu", "capability", "gfx")
# A set's mark of its last use: a file of its own, since the file identities of the
# manifest and the plans key the sessions loaded from them.
LAST_USED = "last-used"
# Another version's set stays this long after its last use: builds that share one
# storage folder (an installed release, a source checkout, the CLI) keep theirs.
UNUSED_DAYS = 14


def last_used(path: Path) -> float:
    """When the set at path was last found valid: its LAST_USED mark, or, for a set
    from before the mark, its newest file."""
    mark = path / LAST_USED
    if mark.is_file():
        return mark.stat().st_mtime
    return max(
        (file.stat().st_mtime for file in path.rglob("*") if file.is_file()),
        default=path.stat().st_mtime,
    )


def unused_reason(path: Path, identity: dict) -> str | None:
    """Why `path`, an entry of an engine cache root whose current set has `identity`,
    is no longer needed; None for what this cache did not make or may still use."""
    if not path.is_dir():
        return None
    if path.name.startswith("quarantine-"):
        return "a set moved aside"
    if path.name.startswith("building-"):
        return "an interrupted build"
    if migration_leftover(path):
        return "an interrupted migration"
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None  # no readable manifest, or not one this cache wrote
    other = manifest.get("identity") if isinstance(manifest, dict) else None
    if not isinstance(other, dict) or path.name != identity_key(other):
        return None

    def engines(value: dict) -> dict:
        return {key: field for key, field in value.items() if key not in GPU_FIELDS}

    if engines(other) == engines(identity):
        return None  # the current set, or another GPU's
    days = (time.time() - last_used(path)) / 86400
    if days < UNUSED_DAYS:
        return None  # another version's, which a build sharing this folder may use
    return f"another version's set, unused for {days:.0f} days"


def remove_unused_sets(root: Path, current: Path, identity: dict) -> None:
    """Caller owns root's cache lock and has just validated `current`, the set of
    `identity`, which this marks used. Delete the sets of other versions unused for
    UNUSED_DAYS, sets moved aside and the folders of interrupted builds and migrations;
    everything else under root stays. Cleanup never fails the run: what cannot be read
    or deleted now stays for the next check."""
    try:
        (current / LAST_USED).touch()
    except OSError as error:
        logger.warning("Could not mark %s as used: %s", current, error)
    freed = 0
    for path in sorted(root.iterdir()):
        if path == current:
            continue
        try:
            reason = unused_reason(path, identity)
            if reason is None:
                continue
            size = sum(
                file.stat().st_size for file in path.rglob("*") if file.is_file()
            )
        except OSError as error:
            logger.warning("Keeping %s for now: %s", path, error)
            continue
        # Moved aside first: Windows refuses the move while a file inside is open, and a
        # deletion that stops halfway is finished by a later check.
        target = path
        if not path.name.startswith("quarantine-"):
            target = path.with_name(f"quarantine-{path.name}-{time.time_ns()}")
            try:
                path.rename(target)
            except OSError as error:
                logger.info(
                    "Keeping %s for now, a file in it is in use: %s", path, error
                )
                continue
        logger.info("Deleting %s (%.1f GB), %s", path, size / 1e9, reason)
        remove_tree(target)
        if not target.exists():
            freed += size
    if freed:
        logger.info("Freed %.1f GB of unused VOSR engines in %s", freed / 1e9, root)


def compress_set(destination: Path, manifest: dict, stages: Sequence[str]) -> None:
    """Compress a verified set of plain plans in place. Its manifest switches only once
    every plan is compressed, and the plain plans go after it, so an interrupted
    conversion leaves the plain set valid."""
    logger.info("Compressing the cached VOSR engines in %s", destination)
    engines = {}
    for name in stages:
        target = compressed(destination, name)
        compress(destination / f"{name}.plan", target)
        engines[name] = file_hash(target)
    write_json(
        destination / "manifest.json",
        {
            **manifest,
            "engines": engines,
            "plans": manifest["engines"],
            "compression": "zstd",
        },
    )
    for name in stages:
        (destination / f"{name}.plan").unlink()


def ensure_full_engine(
    context: NodeContext, paths: BundlePaths, device: torch.device
) -> Path:
    """Publish the entire BF16 set only after every stage has been validated."""
    from .vosr2_trt_models import STAGES, profiles

    read_configs(paths)
    root = context.storage_dir / "vosr2-tensorrt-bf16"
    root.mkdir(parents=True, exist_ok=True)
    with cache_lock(root / "compile.lock", context):
        identity = cache_identity(device)
        identity.update(
            format=4,
            precision="bf16-fp32-qkv-pack-dino-bf16-attention-raw-kv-v4",
            # The whole DLL: its host code (formats, shapes) shapes the engines as much
            # as its kernels, and the build is reproducible (native/README.md, Build).
            plugin_sha256=file_hash(plugin_path()),
            # The official files' pins: Upscale Image (VOSR) verified them on disk.
            checkpoints={Path(local).name: digest for _, local, _, digest in FILES},
            stages={name: profiles(name) for name in STAGES},
            # The build worker and what it exports, rewrites and compiles; the runtime
            # modules that only run the engines do not count (stage_identity).
            pipeline_sources={
                name: file_hash(Path(__file__).with_name(name))
                for name in (
                    "vosr2_trt_models.py",
                    "vosr2_trt_build.py",
                    "vosr2_trt_compile.py",
                    "vosr2_trt_rewrite.py",
                )
            },
        )
        # Normalize tuples so manifests read back with the same identity.
        identity = stage_identity(json.loads(json.dumps(identity)))
        destination = root / identity_key(identity)
        if not destination.exists():
            migrate_stage_cache(context, root, destination, identity)
        engine = ensure_engine_set(
            context,
            root,
            identity,
            "BF16",
            lambda directory: build(context, paths, directory, device),
        )
        remove_unused_sets(root, engine, identity)
        return engine
