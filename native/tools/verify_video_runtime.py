"""Oracle versus portable CPU video HTTP/SSE contracts in owned sandboxes.

Default: validate baseline fixtures only; the baseline is the oracle
(verify_runtime.oracle_source), on the provisioned runtime the package was copied
from. --include-port requires the coordinator's completed stage-5 package, run
on the package's runtime. FFmpeg engines remain external
dependencies. Container bytes are recorded, but exact comparisons use decoded
pixels/audio and selected stream metadata because muxer identifiers/timestamps
need not be deterministic.
PNG bytes are recorded too, and PNGs compare by decoded pixels because the port
encodes default PNGs with fpng. SSE events compare by
verify_runtime.sse_event_mismatches (generator-once, coalesced previews).
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import http.client
import json
import os
import shutil
import socket
import subprocess
import time
import traceback
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from bench_oracle import decoded_png
from verify_framework_runtime import OPTIONS, file_hash, source_hashes
from verify_runtime import (
    METADATA_COMPARISON,
    METADATA_ENDPOINTS,
    ORACLE,
    PROJECT,
    REPEAT_RULES,
    SSE_EVENT_COMPARISON,
    Events,
    OwnedJob,
    broadcast_multisets,
    canonical,
    check_success,
    compare_metadata,
    edge,
    generator_node_ids,
    make_node,
    metadata_record,
    oracle_repeat_flags,
    oracle_source,
    repeat_final_value_differences,
    repeat_mismatches,
    request,
    run_owned,
    sse_event_mismatches,
    sse_record,
    write_json,
)

VIDEO_IDS = {"chainner:image:load_video", "chainner:image:save_video"}
VIDEO_EXTENSIONS = {".mkv", ".mp4", ".mov", ".webm", ".avi", ".gif"}
ENCODERS = [
    ("mkv", "libx264"),
    ("mkv", "libx265"),
    ("mkv", "libvpx-vp9"),
    ("mkv", "ffv1"),
    ("mp4", "libx264"),
    ("mp4", "libx265"),
    ("mp4", "libvpx-vp9"),
    ("mov", "libx264"),
    ("mov", "libx265"),
    ("webm", "libvpx-vp9"),
    ("avi", "libx264"),
    ("gif", "libx264"),
]


def prepare_fixtures(directory: Path, ffmpeg: Path, ffprobe: Path):
    import wave

    import numpy as np

    directory.mkdir(parents=True)
    y, x = np.indices((24, 32))
    frames = [
        np.stack(
            (
                (x * 7 + i * 41) % 256,
                (y * 11 + i * 19) % 256,
                (x + y * 3 + i * 67) % 256,
            ),
            axis=2,
        ).astype(np.uint8)
        for i in range(3)
    ]
    base = [str(ffmpeg), "-nostdin", "-v", "error", "-y"]

    def write(name: str, width: int, images: list):
        run_owned(
            [
                *base,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x24",
                "-r",
                "3",
                "-i",
                "pipe:0",
                "-c:v",
                "ffv1",
                "-pix_fmt",
                "bgr0",
                "-threads",
                "1",
                str(directory / name),
            ],
            data=b"".join(image.tobytes() for image in images),
        )

    write("rgb.mkv", 32, frames)
    write("odd.mkv", 31, [image[:, :31] for image in frames])
    with wave.open(str(directory / "stereo.wav"), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(np.repeat(np.arange(8000, dtype=np.int16), 2).tobytes())
    run_owned(
        [
            *base,
            "-i",
            str(directory / "rgb.mkv"),
            "-i",
            str(directory / "stereo.wav"),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c",
            "copy",
            str(directory / "audio.mkv"),
        ]
    )
    # Small spatial frame, sufficient iterator length to request cancellation
    # after both owned reader and writer have started. No timing is measured.
    write("cancel.mkv", 32, [frames[0]] * 1000)
    (directory / "corrupt.mkv").write_bytes(b"owned invalid video fixture")
    write_json(
        directory / "manifest.json",
        {
            "files": {p.name: file_hash(p) for p in directory.iterdir() if p.is_file()},
            "ffmpeg": str(ffmpeg),
            "ffprobe": str(ffprobe),
            "ffmpeg_sha256": file_hash(ffmpeg),
            "ffprobe_sha256": file_hash(ffprobe),
            "ffmpeg_version": run_owned([str(ffmpeg), "-version"])
            .decode()
            .splitlines()[0],
            "ffprobe_version": run_owned([str(ffprobe), "-version"])
            .decode()
            .splitlines()[0],
        },
    )


def fixture_graphs(schemas: dict, assets: Path, output: Path):
    def node(name: str, schema: str, values: dict):
        return make_node(schemas, name, schema, values)

    def make(
        name: str,
        *,
        container: str = "mkv",
        encoder: str = "ffv1",
        source: str = "rgb.mkv",
        limit: int | None = None,
        audio: str | None = None,
        resize: bool = False,
        simple: str | None = None,
        failure: bool = False,
        cancel: bool = False,
    ) -> dict:
        directory = output / name
        load = node(
            "load",
            "chainner:image:load_video",
            {0: str(assets / source), 1: int(limit is not None), 2: limit or 10},
        )
        nodes = [load]
        image = edge("load")
        if resize:
            nodes.append(
                node(
                    "resize",
                    "chainner:image:resize",
                    {0: image, 1: 1, 3: 16, 4: 12, 5: 0},
                )
            )
            image = edge("resize")
        if name == "midstream-failure":
            nodes += [
                node(
                    "amount",
                    "chainner:utility:math",
                    {0: edge("load", 1), 1: "mul", 2: 100},
                ),
                node(
                    "crop", "chainner:image:crop", {0: image, 1: 0, 2: edge("amount")}
                ),
            ]
            image = edge("crop")
        parameters = "-threads 1"
        if encoder == "libx265":
            parameters += " -x265-params pools=none:frame-threads=1:log-level=error"
        nodes.append(
            node(
                "save",
                "chainner:image:save_video",
                {
                    0: image,
                    1: str(directory),
                    2: "video",
                    16: 0 if simple else 1,
                    4: container,
                    3: encoder,
                    8: "ultrafast",
                    9: 23,
                    13: parameters,
                    17: simple or "mp4_h264",
                    18: 75,
                    14: edge("load", 4),
                    15: edge("load", 5) if audio else None,
                    10: audio or "auto",
                },
            )
        )
        if not failure and not cancel:
            nodes.append(
                node(
                    "png",
                    "chainner:image:save",
                    {
                        0: image,
                        1: str(directory),
                        2: None,
                        3: edge("load", 1),
                        4: "png",
                        15: "u8",
                    },
                )
            )
        return {
            "name": name,
            "nodes": nodes,
            "expected_error": failure,
            "cancel": cancel,
            "directory": str(directory),
        }

    for container, encoder in ENCODERS:
        yield make(container + "-" + encoder, container=container, encoder=encoder)
    for simple in ("mp4_h264", "mp4_h265", "webm", "gif"):
        yield make("simple-" + simple, simple=simple)
    yield make("limit-resize", limit=2, resize=True)
    for container, encoder, mode in (
        ("mkv", "ffv1", "auto"),
        ("mkv", "ffv1", "copy"),
        ("mkv", "ffv1", "transcode"),
        ("webm", "libvpx-vp9", "auto"),
        ("webm", "libvpx-vp9", "transcode"),
    ):
        yield make(
            "audio-" + container + "-" + mode,
            container=container,
            encoder=encoder,
            source="audio.mkv",
            audio=mode,
        )
    yield make("odd-error", source="odd.mkv", encoder="libx264", failure=True)
    yield make("corrupt-error", source="corrupt.mkv", failure=True)
    yield make("midstream-failure", failure=True)
    yield make("cancel", source="cancel.mkv", cancel=True)


def job_processes(job: OwnedJob):
    class ProcessList(ctypes.Structure):
        _fields_ = [
            ("assigned", ctypes.c_ulong),
            ("count", ctypes.c_ulong),
            ("pids", ctypes.c_size_t * 128),
        ]

    kernel = job.kernel
    kernel.QueryInformationJobObject.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_void_p,
    ]
    kernel.QueryInformationJobObject.restype = ctypes.c_int
    info = ProcessList()
    if not kernel.QueryInformationJobObject(
        job.handle, 3, ctypes.byref(info), ctypes.sizeof(info), None
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    if info.count > 128:
        raise RuntimeError("Unexpected verifier process count")
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.QueryFullProcessImageNameW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    result = {}
    for pid in info.pids[: info.count]:
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            continue  # Child exited between the job snapshot and this query.
        try:
            buffer = ctypes.create_unicode_buffer(32768)
            length = ctypes.c_ulong(len(buffer))
            if kernel.QueryFullProcessImageNameW(
                handle, 0, buffer, ctypes.byref(length)
            ):
                result[str(pid)] = buffer.value
        finally:
            kernel.CloseHandle(handle)
    return result


def ffmpeg_children(job: OwnedJob):
    return {
        pid: path
        for pid, path in job_processes(job).items()
        if Path(path).stem.lower() in {"ffmpeg", "ffprobe"}
    }


def close_owned_job(job: OwnedJob, processes: dict):
    """Wait on handles acquired before kill-on-close; do not inspect other PIDs."""
    kernel = job.kernel
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.WaitForSingleObject.restype = ctypes.c_ulong
    handles = []
    try:
        for pid in processes:
            handle = kernel.OpenProcess(0x100000, False, int(pid))
            if handle:
                handles.append(handle)
        job.close()
        return all(kernel.WaitForSingleObject(handle, 15000) == 0 for handle in handles)
    finally:
        job.close()
        for handle in handles:
            kernel.CloseHandle(handle)


def snapshot(path: Path, ffmpeg: Path, ffprobe: Path):
    result: dict = {"sha256": file_hash(path)}
    if path.suffix.lower() == ".png":
        # The port writes default PNGs with fpng: the same pixels in other bytes.
        result["semantic"] = decoded_png(path)
        return result
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        result["semantic"] = result["sha256"]
        return result
    probe = json.loads(
        run_owned(
            [
                str(ffprobe),
                "-v",
                "quiet",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    keys = {
        "codec_name",
        "codec_type",
        "profile",
        "pix_fmt",
        "width",
        "height",
        "r_frame_rate",
        "avg_frame_rate",
        "time_base",
        "start_time",
        "duration",
        "nb_frames",
        "sample_rate",
        "channels",
        "channel_layout",
        "bits_per_raw_sample",
    }
    streams = [
        {key: value for key, value in stream.items() if key in keys}
        for stream in probe["streams"]
    ]
    pixels = run_owned(
        [
            str(ffmpeg),
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-sws_flags",
            "lanczos+accurate_rnd+full_chroma_int+full_chroma_inp+bitexact",
            "pipe:1",
        ]
    )
    video = next(s for s in streams if s["codec_type"] == "video")
    frame_bytes = video["width"] * video["height"] * 3
    if len(pixels) % frame_bytes:
        raise AssertionError("Partial snapshot frame")
    semantics = {
        "streams": streams,
        "format": {
            k: v
            for k, v in probe["format"].items()
            if k in {"nb_streams", "format_name", "start_time", "duration"}
        },
        "decoded_frame_count": len(pixels) // frame_bytes,
        "decoded_frames_sha256": hashlib.sha256(pixels).hexdigest(),
    }
    if any(s["codec_type"] == "audio" for s in streams):
        audio = run_owned(
            [
                str(ffmpeg),
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "pipe:1",
            ]
        )
        semantics["decoded_audio_sha256"] = hashlib.sha256(audio).hexdigest()
        semantics["decoded_audio_bytes"] = len(audio)
    result["semantic"] = semantics
    return result


def provision_ffmpeg(storage: Path, ffmpeg: Path, ffprobe: Path):
    directory = storage / "ffmpeg"
    directory.mkdir()
    files = (
        set(ffmpeg.parent.glob("*.dll"))
        | set(ffprobe.parent.glob("*.dll"))
        | {ffmpeg, ffprobe}
    )
    hashes = {}
    for source in sorted(files):
        target = directory / source.name
        # New copies keep cleanup and ownership independent of original engines.
        shutil.copy2(source, target)
        hashes[source.name] = file_hash(target)
    return hashes


def run_backend(
    label: str,
    backend: Path,
    python: Path,
    root: Path,
    ffmpeg: Path,
    ffprobe: Path,
    timeout: float,
):
    directory = root / label
    directory.mkdir()
    for name in ("storage", "output", "profile", "appdata", "localappdata", "temp"):
        (directory / name).mkdir()
    binaries = provision_ffmpeg(directory / "storage", ffmpeg, ffprobe)
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "CUDA_VISIBLE_DEVICES": "-1",
            "NVIDIA_VISIBLE_DEVICES": "none",
            # ncnn's Vulkan, hidden on both sides (Consult 8 sweep item 6): no
            # instance, so no crash at the host's exit and no clone holding DLLs.
            "VK_LOADER_DRIVERS_DISABLE": "*",
            # A host's dependency installer must never write into the provisioned
            # runtime or the package (bench_backend.ISOLATION): a needed install
            # fails the start instead; pip list ignores it.
            "PIP_REQUIRE_VIRTUALENV": "1",
            "APPDATA": str(directory / "appdata"),
            "LOCALAPPDATA": str(directory / "localappdata"),
            "USERPROFILE": str(directory / "profile"),
            "HOME": str(directory / "profile"),
            "TMP": str(directory / "temp"),
            "TEMP": str(directory / "temp"),
            "PATH": str(ffmpeg.parent) + os.pathsep + env["PATH"],
        }
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        str(python),
        "-B",
        str(backend / "run.py"),
        str(port),
        "--storage-dir",
        str(directory / "storage"),
    ]
    job, process, events = OwnedJob(), None, None
    log = (directory / "backend.log").open("wb")
    result: dict = {
        "label": label,
        "command": command,
        "cpu_only": True,
        "ffmpeg_copies": binaries,
        "fixtures": [],
        "success": False,
    }
    try:
        process = subprocess.Popen(
            command,
            cwd=directory,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW
            | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        job.assign(process)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Backend exited: {process.returncode}")
            try:
                if request(port, "/status", timeout=3).get("ready"):
                    break
            except OSError, urllib.error.URLError, json.JSONDecodeError:
                pass
            time.sleep(0.2)
        else:
            raise TimeoutError("Backend startup deadline")
        metadata = {
            key: request(port, "/" + key, timeout=30) for key in METADATA_ENDPOINTS
        }
        for key, value in metadata.items():
            write_json(directory / (key + ".json"), value)
        result.update(metadata_record(metadata))
        schemas = {item["schemaId"]: item for item in metadata["nodes"]["nodes"]}
        if not VIDEO_IDS <= schemas.keys():
            raise AssertionError("Video registry missing")
        events = Events(port)
        if not events.ready.wait(10) or events.error:
            raise RuntimeError(f"SSE unavailable: {events.error}")
        for fixture in fixture_graphs(schemas, root / "fixtures", directory / "output"):
            name, nodes = fixture["name"], fixture["nodes"]
            output = Path(fixture["directory"])
            output.mkdir()
            payload = {"data": nodes, "options": OPTIONS, "sendBroadcastData": True}
            write_json(directory / (name + "-request.json"), payload)
            print(f"{label}: {name}", flush=True)
            attempts = []
            for repeat in range(1 if fixture["cancel"] else 2):
                start = len(events.events)
                if fixture["cancel"]:
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        pending = pool.submit(request, port, "/run", payload, 120)
                        deadline = time.monotonic() + 30
                        children = {}
                        while time.monotonic() < deadline:
                            children = ffmpeg_children(job)
                            if (
                                len(
                                    [
                                        p
                                        for p in children.values()
                                        if Path(p).stem.lower() == "ffmpeg"
                                    ]
                                )
                                >= 2
                            ):
                                break
                            if pending.done():
                                raise AssertionError(
                                    "Cancellation graph completed before reader/writer ownership could be observed"
                                )
                            time.sleep(0.02)
                        else:
                            raise TimeoutError(
                                "No active owned decoder and encoder for cancellation"
                            )
                        fixture["active_children_before_cancel"] = children
                        fixture["kill_response"] = request(
                            port, "/kill", {}, timeout=30
                        )
                        try:
                            response = pending.result(timeout=30)
                            status = 200
                        except urllib.error.HTTPError as error:
                            status, response = error.code, json.load(error)
                else:
                    try:
                        response = request(port, "/run", payload, timeout=120)
                        status = 200
                    except urllib.error.HTTPError as error:
                        if not fixture["expected_error"]:
                            raise
                        status, response = error.code, json.load(error)
                if fixture["expected_error"]:
                    if response.get("success") or response.get("type") == "success":
                        raise AssertionError(f"{name} unexpectedly succeeded")
                    deadline = time.monotonic() + 15
                    while not any(
                        e["event"] == "execution-error" for e in events.events[start:]
                    ):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Missing error SSE")
                        time.sleep(0.02)
                    # Installed error responses can precede queued broadcasts.
                    # Keep trailing events with their attempt, using a bounded
                    # completion wait rather than discarding asynchronous data.
                    settle_deadline = time.monotonic() + 10
                    quiet_since = time.monotonic()
                    count = len(events.events)
                    while time.monotonic() - quiet_since < 0.25:
                        if time.monotonic() >= settle_deadline or events.error:
                            raise RuntimeError(f"Error SSE did not settle: {name}")
                        time.sleep(0.02)
                        if count != len(events.events):
                            count = len(events.events)
                            quiet_since = time.monotonic()
                elif not fixture["cancel"]:
                    check_success(response)
                    events.wait_for(
                        {n["id"] for n in nodes},
                        start,
                        expected_broadcasts={
                            n["id"] for n in nodes if schemas[n["schemaId"]]["outputs"]
                        },
                    )
                # This checks owned FFmpeg children before job cleanup can hide leaks.
                deadline = time.monotonic() + 5
                remaining = ffmpeg_children(job)
                while remaining and time.monotonic() < deadline:
                    time.sleep(0.02)
                    remaining = ffmpeg_children(job)
                if label == "converted" and remaining:
                    raise AssertionError(
                        f"Native executor retained owned FFmpeg children: {remaining}"
                    )
                files = {}
                for path in sorted(output.rglob("*")):
                    if not path.is_file():
                        continue
                    relative = path.relative_to(output).as_posix()
                    files[relative] = (
                        {"sha256": file_hash(path)}
                        if (fixture["expected_error"] and name != "midstream-failure")
                        or fixture["cancel"]
                        else snapshot(path, ffmpeg, ffprobe)
                    )
                if name == "midstream-failure":
                    if set(files) != {"video.mkv"}:
                        raise AssertionError(
                            "Buffered frames were discarded after midstream failure"
                        )
                    if files["video.mkv"]["semantic"]["decoded_frame_count"] != 1:
                        raise AssertionError(
                            "Midstream failure must preserve exactly the one accepted frame"
                        )
                if not fixture["expected_error"] and not fixture["cancel"]:
                    if not any(
                        Path(x).suffix in VIDEO_EXTENSIONS for x in files
                    ) or not any(x.endswith(".png") for x in files):
                        raise AssertionError(
                            "No encoded video and consumed-frame PNG outputs"
                        )
                attempt = {
                    "http_status": status,
                    "response": canonical(response, directory),
                    **sse_record(events.events[start:], directory),
                    "files": files,
                    "remaining_ffmpeg_children": remaining,
                }
                attempts.append(attempt)
                write_json(directory / f"{name}-attempt-{repeat}.json", attempt)
            stable = [
                dict(
                    a,
                    files={
                        key: value.get("semantic") for key, value in a["files"].items()
                    },
                )
                for a in attempts
            ]
            coalesces = label == "converted"  # The port coalesces previews.
            if not fixture["cancel"] and (
                mismatches := repeat_mismatches(stable[0], stable[1], port=coalesces)
            ):
                raise AssertionError(
                    f"{name} repeat changed semantic outputs/events: {mismatches}"
                )
            fixture.update(
                attempts=attempts,
                generator_node_ids=generator_node_ids(nodes, schemas),
                repeat_semantic_rule="not compared (cancelled)"
                if fixture["cancel"]
                else REPEAT_RULES[coalesces],
                repeat_final_values_differ={}
                if fixture["cancel"]
                else repeat_final_value_differences(stable[0], stable[1]),
                encoded_bytes_repeat_equal=len(attempts) == 2
                and attempts[0]["files"] == attempts[1]["files"],
            )
            result["fixtures"].append(fixture)
            write_json(directory / "result.json", result)
        result["success"] = True
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        if events is not None:
            write_json(directory / "events.json", events.events)
            events.close()
        if process is not None and process.poll() is None:
            try:
                request(port, "/shutdown", {}, timeout=15)
            except (
                OSError,
                urllib.error.URLError,
                json.JSONDecodeError,
                http.client.HTTPException,
            ):
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
        result["owned_processes_before_job_close"] = job_processes(job)
        result["owned_children_stopped"] = close_owned_job(
            job, result["owned_processes_before_job_close"]
        )
        if process is not None:
            process.wait(timeout=15)
            result["owned_process_exited"] = process.poll() is not None
        result["owned_job_closed"] = True
        log.close()
        write_json(directory / "result.json", result)
    return result


def video_first(semantic: object) -> object:
    """A decoded video file's semantic with its video streams before the rest.

    Save Video muxes the source's audio after the video (owner-approved fix for
    upstream chaiNNer #3331; ARCHITECTURE section 7) where the installed oracle put
    it first. The port's file must match the oracle's stream for stream; only their
    order moves. Every other semantic passes unchanged.
    """
    if not isinstance(semantic, dict) or "streams" not in semantic:
        return semantic
    streams = semantic["streams"]
    return {
        **semantic,
        "streams": [s for s in streams if s["codec_type"] == "video"]
        + [s for s in streams if s["codec_type"] != "video"],
    }


def compare_runs(baseline: dict, converted: dict) -> dict:
    comparisons = {}
    for old, new in zip(baseline["fixtures"], converted["fixtures"], strict=True):
        if old["name"] != new["name"]:
            raise AssertionError("Fixture order changed")
        if old["cancel"]:
            comparisons[old["name"]] = {
                "pass": not new["attempts"][0]["remaining_ffmpeg_children"],
                "cancelled_iteration_count_not_compared": True,
            }
            continue
        a, b = old["attempts"][0], new["attempts"][0]
        semantics = {
            key: video_first(value.get("semantic")) for key, value in a["files"].items()
        } == {key: value.get("semantic") for key, value in b["files"].items()}
        oracle = {
            **a,
            "name": old["name"],
            "generator_node_ids": old["generator_node_ids"],
        }
        event_mismatches = [
            ("repeat: " if index else "") + mismatch
            for index, attempt in enumerate(new["attempts"])
            for mismatch in sse_event_mismatches(
                oracle,
                {
                    **attempt,
                    "name": new["name"],
                    "generator_node_ids": new["generator_node_ids"],
                },
            )
        ]
        events_equal = not event_mismatches and all(
            a[key] == b[key] for key in ("http_status", "response")
        )
        comparisons[old["name"]] = {
            "pass": semantics and events_equal,
            "semantic_outputs_equal": semantics,
            "http_sse_equal": events_equal,
            "sse_event_mismatches": event_mismatches,
            "broadcast_multisets": {
                "oracle": broadcast_multisets(old["attempts"]),
                "port": broadcast_multisets(new["attempts"]),
            },
            "encoded_file_bytes_equal": {
                name: entry["sha256"] == b["files"].get(name, {}).get("sha256")
                for name, entry in a["files"].items()
            },
        }
    metadata = compare_metadata(baseline, converted)
    return {
        "fixtures": comparisons,
        # Flagged, not failed (Consult D-35): the oracle's own attempts ended on
        # different final values of an iterated field.
        "oracle_repeat_final_values_differ": oracle_repeat_flags(baseline),
        **metadata,
        "success": all(metadata["metadata_equal"].values())
        and all(x["pass"] for x in comparisons.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=PROJECT / "out/chaiNNer-C")
    parser.add_argument("--include-port", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument(
        "--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg.exe")
    )
    parser.add_argument(
        "--ffprobe", type=Path, default=Path(shutil.which("ffprobe") or "ffprobe.exe")
    )
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    ffmpeg, ffprobe = (
        args.ffmpeg.resolve(strict=True),
        args.ffprobe.resolve(strict=True),
    )
    manifest = json.loads(
        (package / "chainner-c-package.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("state") != "complete"
        or manifest["identity"]["destination"] != str(package)
        or manifest["identity"]["project"] != str(PROJECT)
    ):
        raise ValueError("Completed matching package required")
    python = package / "python/python/python.exe"
    source, oracle_python, oracle = oracle_source(manifest, ORACLE)
    sources = {"baseline": source}
    pythons = {"baseline": oracle_python, "converted": python}
    if args.include_port:
        sources["converted"] = package / "resources/src"
        if (
            "video_loader_init"
            not in (sources["converted"] / "nodes/impl/video.py").read_text()
        ):
            raise ValueError("Stage-5 video package has not been released")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = PROJECT / "native/reports" / ("video-runtime-" + stamp)
    root.mkdir()
    protected = {label: source_hashes(path) for label, path in sources.items()}
    report: dict = {
        "utc": datetime.now(UTC).isoformat(),
        "oracle": oracle,
        "success": False,
        "mode": "oracle-versus-port"
        if args.include_port
        else "baseline-fixture-validation-only",
        "cpu_only": True,
        "verifier_sha256": file_hash(Path(__file__)),
        "performance_benchmark": False,
        "limitations": [
            "FFmpeg/ffprobe remain the existing codec engines; no GPU acceleration is requested.",
            "Container hashes are recorded but exact media parity is decoded pixels/audio plus selected stream metadata, not muxer timestamps/identifiers.",
            "PNG hashes are recorded but PNG parity is decoded pixels plus colour/orientation chunks (bench_oracle.decoded_png): the port encodes default PNGs with fpng.",
            SSE_EVENT_COMPARISON,
            METADATA_COMPARISON,
            "Cancellation iteration counts are deliberately not compared; owned children are observed before job cleanup.",
            "Only new fixture/storage/profile directories are written; original app and source are hash-checked unchanged.",
        ],
        "runs": [],
    }
    try:
        prepare_fixtures(root / "fixtures", ffmpeg, ffprobe)
        for label, source in sources.items():
            copied = root / (label + "-src")
            shutil.copytree(
                source, copied, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
            )
            report["runs"].append(
                run_backend(
                    label,
                    copied,
                    pythons[label],
                    root,
                    ffmpeg,
                    ffprobe,
                    args.startup_timeout,
                )
            )
        if args.include_port:
            report["comparison"] = compare_runs(*report["runs"])
            report["success"] = report["comparison"]["success"]
        else:
            report["success"] = report["runs"][0]["success"]
    except Exception as error:
        report["error"], report["traceback"] = str(error), traceback.format_exc()
        report["runs"] = [
            json.loads(p.read_text())
            for label in sources
            if (p := root / label / "result.json").is_file()
        ]
    finally:
        report["original_source_trees_unchanged"] = {
            label: protected[label] == source_hashes(path)
            for label, path in sources.items()
        }
        report["all_owned_processes_stopped"] = bool(report["runs"]) and all(
            run.get("owned_process_exited")
            and run.get("owned_job_closed")
            and run.get("owned_children_stopped")
            for run in report["runs"]
        )
        report["success"] = (
            report["success"]
            and all(report["original_source_trees_unchanged"].values())
            and report["all_owned_processes_stopped"]
        )
        # Flagged, not failed (Consult D-35): the oracle's own attempts ended on
        # different final values of an iterated field.
        report["oracle_repeat_final_values_differ"] = (
            oracle_repeat_flags(report["runs"][0]) if report["runs"] else {}
        )
        write_json(root / "report.json", report)
        print(
            json.dumps(
                {
                    "report": str(root / "report.json"),
                    "success": report["success"],
                    "error": report.get("error"),
                    "oracle_repeat_final_values_differ": sorted(
                        report["oracle_repeat_final_values_differ"]
                    ),
                }
            ),
            flush=True,
        )
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
