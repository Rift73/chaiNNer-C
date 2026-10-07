"""SP4c: the NumPy pool's growth and release checks, on a started worker.

The controller runs this tool at the gate (SP4c plan, Task 4 and P9). Each check starts
the repository's backend/src through the bench's own Backend (bench_backend.py: a
kill-on-close job, private directories, the packaged interpreter, a private FFmpeg copy,
the tree and interpreter identity checks) with CHAINNER_C_PROFILE=1: the counters the
checks read (requests, held_peak, evictions, threads, releases, released_bytes) are
kept under the profile switch only. They are the "numpy_pool" entry of the worker's
"native profile:" line, one line per /run; the entry is there only while the handler
is installed.

  video        video-resize-h264's chain (load, 50 % resize, H.264 save) on a
               synthesized video of --frames frames (default 3000): testsrc2 1280x720
               at 30 fps, FFV1 bgr0 like the bench fixture, made once with the pinned
               FFmpeg in growth/assets-<frames>/moving.mkv (about 220 MB at 3000 frames)
  video-cap16  the same video with CHAINNER_C_NUMPY_POOL=16, so the cap is reached and
               the oldest idle blocks are released
  images       resize-real-256's chain (the saved 50 % Hermite resize) with its load
               limit raised to --images (default 768) images of different sizes from
               --images-dir (default F:\\Download\\New folder\\DiverSeg-IP\\HR\\0),
               which is only read
  release      the release points (spec 4.3) on one worker with
               CHAINNER_C_NUMPY_POOL=256, in order:
               (0) a /run of Create Color 256x256 (below the threshold) -> Save Image,
                   the warm-up; then the baseline: the private bytes at rest
               (a) a /run of a 64 MiB Create Color -> Save Image: the baseline again,
                   and the run's entry has released_bytes >= 64 MiB
               (b) a /run/individual of the 64 MiB Create Color: baseline + 64 MiB (the
                   cached output); the same node id at 96 MiB: baseline + 96 MiB, not
                   + 160 MiB (the first output, popped at the request's start, idles
                   while the second runs and is released at its end)
               (c) /clear-cache/individual of that node id: the baseline again
               (d) during a /run of the video check's graph, one /run/individual of the
                   64 MiB Create Color, then one /clear-cache/individual of it, both
                   answered while the /run is in flight (else FAIL, "not mid-run"); the
                   run's entry then has releases == 1 and released_bytes >= 64 MiB
               (e) asserted: a /run of Create Color 64 MiB -> Crop (a border of the
                   image's height, which raises) -> Save Image. The run must fail in
                   its Crop. Reported: its entry's live_bytes and released_bytes
                   (logged after the run's release). Asserted: the private bytes are
                   back at the level at rest from before the run, both after the
                   error response ("after_response": the failed run's collection
                   frees what the error held, then releases it) and after the next
                   release point, the warm-up graph again ("after_next_release")
               (f) (e) again, from its own level at rest, with View Image, which
                   reads its image eagerly, in place of Save Image, which reads it
                   through a Lazy
               A level is the first private-bytes sample within 16 MiB of its expected
               value, sampled every 0.2 s for up to 5 s after the response (late
               collections). When no sample is, the line lists every sample and the
               check fails.

Asserted per growth check (video, video-cap16, images):
  - the handler is installed: the run's profile line has a "numpy_pool" entry, with at
    least one request at or above the threshold;
  - idle bytes never exceed the cap: the entry's held_peak over the /run, which bounds
    every sample, is at most cap_bytes;
  - the worker's private bytes plateau after the first third: the largest sample after
    the first third of the items is at most the largest within it plus
    max(64 MiB, 10 %). Samples are the worker process's PrivateUsage, taken each time
    the generator's progress passes a multiple of --every (video) or --every-images.
Reported, not asserted: the hit rate ((hits + in-place reallocs) / requests at or above
the threshold), the same without cold misses, evictions, the threads and the bytes
released at the run's end. A video-cap16 run that never evicted prints a WARNING.

Every output goes under native/build/numpy-pool-growth/growth/: run-<UTC>/ holds each
check's backend directory (its log and records), its outputs (deleted after the check)
and growth.json. Each check prints one "GROWTH <check> PASS|FAIL {...}" line; a check
that raises prints its traceback and fails. Then "GROWTH-RESULT pass|fail <run dir>";
the exit code is 1 when a check failed. The tool refuses to start when
CHAINNER_C_NUMPY_POOL or CHAINNER_C_PROFILE is set in its environment (each would
reach every worker). --check verifies the prerequisites, starts nothing and writes
nothing.

Run it with native\\.venv\\Scripts\\python.exe -B native\\tools\\numpy_pool_growth.py.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
import urllib.error
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import bench_cases
import psutil
from bench_backend import PROJECT, Backend, log_matches
from bench_oracle import FFMPEG, FFPROBE, require_pinned_tools
from verify_framework_runtime import OPTIONS
from verify_runtime import check_success, edge, make_node, request, run_owned

TREE = PROJECT / "backend" / "src"
GROWTH = PROJECT / "native" / "build" / "numpy-pool-growth" / "growth"
DATASET = Path(r"F:\Download\New folder\DiverSeg-IP\HR\0")
POOL_VARIABLE = "CHAINNER_C_NUMPY_POOL"  # nodes.impl.numpy_pool.VARIABLE
PROFILE_VARIABLE = "CHAINNER_C_PROFILE"  # nodes.impl.native_profile.VARIABLE
NATIVE_PROFILE = re.compile(r"native profile: (\{.*\})$")  # bench.NATIVE_PROFILE
CHECKS = ("video", "video-cap16", "images", "release")
CHECK_ENV = {
    "video": {PROFILE_VARIABLE: "1"},
    "video-cap16": {PROFILE_VARIABLE: "1", POOL_VARIABLE: "16"},
    "images": {PROFILE_VARIABLE: "1"},
    "release": {PROFILE_VARIABLE: "1", POOL_VARIABLE: "256"},
}
MIB = 1 << 20
PLATEAU_SLACK = 64 * MIB
PLATEAU_SHARE = 0.10
# The release check's levels.
NOISE = 16 * MIB  # a level is within this of its expected private bytes
REST = 2 * MIB  # at rest: two samples in a row within this of each other
SETTLE_STEP = 0.2  # seconds between samples
SETTLE_SAMPLES = 26  # every 0.2 s for 5 s
# Create Color sizes (width, height) of one gray f32 block: image_fill gives a
# one-value color the shape (h, w).
WARM = (256, 256)  # 256 KiB, below the threshold
BLOCK = (4096, 4096)  # 64 MiB
LARGER = (4096, 6144)  # 96 MiB
COLOR_NODE = "release-color"
CROP_ERROR = "Cropped area would result in an image with no height"


def parser() -> argparse.ArgumentParser:
    arguments = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    arguments.add_argument(
        "--frames", type=int, default=3000, help="video frames (default 3000)"
    )
    arguments.add_argument(
        "--images", type=int, default=768, help="images from --images-dir (default 768)"
    )
    arguments.add_argument(
        "--images-dir",
        type=Path,
        default=DATASET,
        help="the images check's folder, only read (default %(default)s)",
    )
    arguments.add_argument(
        "--every",
        type=int,
        default=100,
        help="video items between samples (default 100)",
    )
    arguments.add_argument(
        "--every-images",
        type=int,
        default=32,
        help="images between samples (default 32)",
    )
    arguments.add_argument("--checks", nargs="+", default=list(CHECKS), choices=CHECKS)
    arguments.add_argument(
        "--check",
        action="store_true",
        help="check the prerequisites and print the plan; start nothing, write nothing",
    )
    return arguments


def guard_output(path: Path) -> Path:
    """path resolved, when it is inside GROWTH; every output of the tool passes here."""
    resolved = path.resolve()
    if GROWTH.resolve() not in resolved.parents:
        raise RuntimeError(f"refusing an output outside {GROWTH}: {resolved}")
    return resolved


def count_frames(video: Path) -> int:
    output = run_owned(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(video),
        ],
        timeout=600,
    )
    return int(output.decode("ascii").strip())


def synthesize(frames: int) -> Path:
    """growth/assets-<frames>/ holding moving.mkv of exactly frames frames, made once."""
    assets = guard_output(GROWTH / f"assets-{frames}")
    video = assets / "moving.mkv"
    if video.is_file() and count_frames(video) == frames:
        print(f"video: reusing {video}", flush=True)
        return assets
    assets.mkdir(parents=True, exist_ok=True)
    partial = assets / "moving.partial.mkv"
    started = time.monotonic()
    run_owned(
        [
            str(FFMPEG),
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1280x720:rate=30",
            "-frames:v",
            str(frames),
            "-an",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "bgr0",
            "-threads",
            "8",
            str(partial),
        ],
        timeout=3600,
    )
    partial.replace(video)
    counted = count_frames(video)
    if counted != frames:
        raise RuntimeError(f"{video} has {counted} frames, not {frames}")
    print(
        f"video: synthesized {video} ({frames} frames, "
        f"{video.stat().st_size / MIB:.0f} MiB, {time.monotonic() - started:.0f} s)",
        flush=True,
    )
    return assets


def worker_process(backend: Backend) -> psutil.Process:
    """The worker (server.py) under the backend's host process."""
    if backend.process is None:
        raise RuntimeError(f"{backend.label} has no host process")
    for child in psutil.Process(backend.process.pid).children(recursive=False):
        try:
            if any(part.endswith("server.py") for part in child.cmdline()):
                return child
        except psutil.Error as error:
            print(f"{backend.label}: skipping child {child.pid}: {error!r}", flush=True)
    raise RuntimeError(f"{backend.label}: no worker process (server.py) under the host")


def run_payload(nodes: list[dict]) -> dict:
    return {"data": nodes, "options": OPTIONS, "sendBroadcastData": True}


def profile_entry(log: Path, offset: int) -> dict | None:
    """The "numpy_pool" entry of the one native profile line a /run logged after
    offset; None when the line has none (the handler is not installed)."""
    found = log_matches(log, offset, NATIVE_PROFILE, 10)
    if len(found) != 1:
        raise RuntimeError(
            f"{len(found)} native profile lines after offset {offset} in {log}; "
            "one /run logs one"
        )
    return json.loads(found[0][1]).get("numpy_pool")


def run_sampled(
    backend: Backend,
    nodes: list[dict],
    generator: str,
    every: int,
    during: Callable[[Callable[[], bool]], dict] | None = None,
) -> dict:
    """One /run of nodes, sampling the worker's private bytes every `every` items.

    during, when given, runs once at the first sample, while the /run is in flight,
    with a function that tells whether it still is; its result is "during".
    """
    if backend.events is None:
        raise RuntimeError(f"{backend.label} has no SSE stream")
    events = backend.events
    payload = run_payload(nodes)
    log = backend.directory / "backend.log"
    offset = log.stat().st_size
    worker = worker_process(backend)
    outcome: dict = {}

    def post() -> None:
        try:
            outcome["response"] = request(backend.port, "/run", payload, timeout=3600)
        except BaseException as error:  # re-raised on the sampling thread below
            outcome["error"] = error

    result: dict = {}
    cursor = len(events.events)
    index = 0
    mark = every
    samples: list[dict] = [{"index": 0, "private": worker.memory_info().private}]
    began = time.monotonic()
    thread = threading.Thread(target=post, name="growth /run", daemon=True)
    thread.start()
    while thread.is_alive():
        captured = events.events[cursor:]
        cursor += len(captured)
        for event in captured:
            data = event["data"]
            if event["event"] == "node-progress" and data.get("nodeId") == generator:
                index = max(index, int(data["index"]))
        if index >= mark:
            samples.append(
                {
                    "index": index,
                    "private": worker.memory_info().private,
                    "seconds": round(time.monotonic() - began, 3),
                }
            )
            mark = (index // every + 1) * every
            if during is not None and "during" not in result:
                result["during"] = {"index": index, **during(thread.is_alive)}
        thread.join(0.02)
    if "error" in outcome:
        raise outcome["error"]
    check_success(outcome["response"])
    samples.append(
        {
            "index": "end",
            "private": worker.memory_info().private,
            "seconds": round(time.monotonic() - began, 3),
        }
    )
    result["samples"] = samples
    result["stats"] = profile_entry(log, offset)
    return result


def judge(name: str, total: int, run: dict, expect_evictions: bool) -> dict:
    """The assertions and the reported figures of one growth check."""
    stats: dict | None = run["stats"]
    samples: list[dict] = run["samples"]
    failures: list[str] = []
    figures: dict = {}
    if stats is None:
        failures.append(
            "the run's native profile line has no numpy_pool entry: "
            "the handler is not installed"
        )
    else:
        if stats["requests"] < 1:
            failures.append("the handler saw no request at or above the threshold")
        cap = stats["cap_bytes"]
        if stats["held_peak"] > cap:
            failures.append(f"held_peak {stats['held_peak']} exceeds the cap {cap}")
        served = stats["hits"] + stats["realloc_inplace"]
        warm = stats["requests"] - stats["misses_cold"]
        figures = {
            "cap_mib": cap / MIB,
            "held_peak_mib": stats["held_peak"] / MIB,
            "live_peak_mib": stats["live_peak"] / MIB,
            "requests": stats["requests"],
            "hit_rate": served / stats["requests"] if stats["requests"] else None,
            "warm_hit_rate": served / warm if warm else None,
            "evictions": stats["evictions"],
            "dropped": stats["dropped"],
            "threads": stats["threads"],
            "releases": stats["releases"],
            "released_mib": stats["released_bytes"] / MIB,
        }
        if expect_evictions and stats["evictions"] == 0 and stats["dropped"] == 0:
            print(
                f"WARNING {name}: the cap was never reached (no eviction); "
                "it is untested here",
                flush=True,
            )
    numbered = [s for s in samples if isinstance(s["index"], int) and s["index"] > 0]
    first = [s["private"] for s in numbered if s["index"] <= total / 3]
    rest = [s["private"] for s in numbered if s["index"] > total / 3]
    rest.append(samples[-1]["private"])
    allowed = None
    if not first:
        failures.append("no private-bytes sample in the first third")
    else:
        allowed = max(first) + max(PLATEAU_SLACK, int(PLATEAU_SHARE * max(first)))
        if max(rest) > allowed:
            failures.append(
                f"private bytes grew after the first third: {max(rest) / MIB:.1f} MiB > "
                f"{max(first) / MIB:.1f} MiB + slack ({allowed / MIB:.1f} MiB)"
            )
    return {
        "check": name,
        "items": total,
        "passed": not failures,
        "failures": failures,
        **figures,
        "private_first_third_max_mib": max(first) / MIB if first else None,
        "private_after_max_mib": max(rest) / MIB,
        "private_allowed_mib": allowed / MIB if allowed is not None else None,
        "samples": len(samples),
    }


def settle(
    sample: Callable[[], int], expected: int, step: float = SETTLE_STEP
) -> tuple[int | None, list[int]]:
    """The first of up to SETTLE_SAMPLES samples, step seconds apart, within NOISE of
    expected (a collection can come late), or None; and every sample taken."""
    taken: list[int] = []
    for number in range(SETTLE_SAMPLES):
        if number:
            time.sleep(step)
        value = sample()
        taken.append(value)
        if abs(value - expected) <= NOISE:
            return value, taken
    return None, taken


def at_rest(
    sample: Callable[[], int], step: float = SETTLE_STEP
) -> tuple[int | None, list[int]]:
    """The private bytes at rest: the later of the first two samples in a row, step
    seconds apart, within REST of each other, of up to SETTLE_SAMPLES; or None. And
    every sample taken."""
    taken: list[int] = []
    for number in range(SETTLE_SAMPLES):
        if number:
            time.sleep(step)
        taken.append(sample())
        if len(taken) > 1 and abs(taken[-1] - taken[-2]) <= REST:
            return taken[-1], taken
    return None, taken


def mib(value: int) -> float:
    return round(value / MIB, 1)


def color(schemas: dict[str, dict], node_id: str, size: tuple[int, int]) -> dict:
    """A Create Color of one gray value: one f32 block of width x height x 4 bytes."""
    width, height = size
    gray = json.dumps({"kind": "grayscale", "values": [0.5]})
    return make_node(
        schemas, node_id, "chainner:image:create_color", {0: gray, 1: width, 2: height}
    )


def color_to_save(
    schemas: dict[str, dict],
    size: tuple[int, int],
    output: Path,
    name: str,
    crop: bool = False,
) -> list[dict]:
    """Create Color -> Save Image (PNG, 8 bits) as output/<name>.png; with crop, a Crop
    between them whose border is the image's height, which raises (CROP_ERROR)."""
    nodes = [color(schemas, "color", size)]
    source = "color"
    if crop:
        nodes.append(
            make_node(
                schemas,
                "crop",
                "chainner:image:crop",
                {0: edge("color"), 1: 0, 2: size[1]},
            )
        )
        source = "crop"
    save: dict[int, object] = {
        0: edge(source),
        1: str(output),
        2: None,
        3: name,
        4: "png",
        15: "u8",
    }
    nodes.append(make_node(schemas, "save", "chainner:image:save", save))
    return nodes


def individual_payload(schemas: dict[str, dict], size: tuple[int, int]) -> dict:
    node = color(schemas, COLOR_NODE, size)
    return {
        "id": node["id"],
        "schemaId": node["schemaId"],
        "inputs": [x["value"] for x in node["inputs"]],
        "options": OPTIONS,
    }


def release_steps(backend: Backend, output: Path, assets: Path, every: int) -> dict:
    """The release check's steps (0) to (f) on a started worker; its summary."""
    schemas = backend.schemas
    log = backend.directory / "backend.log"
    worker = worker_process(backend)
    failures: list[str] = []
    summary: dict = {"check": "release", "passed": False, "failures": failures}

    def private() -> int:
        return worker.memory_info().private

    def entry(log_offset: int) -> dict:
        stats = profile_entry(log, log_offset)
        if stats is None:
            raise RuntimeError(
                "the run's native profile line has no numpy_pool entry: "
                "the handler is not installed"
            )
        return stats

    def run(nodes: list[dict]) -> dict:
        """One /run of nodes, which must succeed; its "numpy_pool" entry."""
        offset = log.stat().st_size
        check_success(request(backend.port, "/run", run_payload(nodes), timeout=600))
        return entry(offset)

    def run_individual(size: tuple[int, int]) -> None:
        payload = individual_payload(schemas, size)
        check_success(request(backend.port, "/run/individual", payload, timeout=600))

    def clear_individual() -> None:
        payload = {"id": COLOR_NODE}
        check_success(
            request(backend.port, "/clear-cache/individual", payload, timeout=60)
        )

    def level(step: str, reference: int, expected: int) -> dict:
        """The level after a response, as MiB over reference (the last sample's when
        none is within NOISE of reference + expected; then every sample too, and a
        failure). The reference is the baseline in (a) to (d), and the level at rest
        before the error run in (e) and (f)."""
        value, taken = settle(private, reference + expected)
        figures: dict = {
            "expected_mib": mib(expected),
            "in_band": value is not None,
            "delta_mib": mib((taken[-1] if value is None else value) - reference),
            "samples": len(taken),
        }
        if value is None:
            figures["samples_mib"] = [mib(s - reference) for s in taken]
            failures.append(
                f"({step}) the private bytes never came within {NOISE // MIB} MiB "
                f"of the reference + {mib(expected)} MiB"
            )
        return figures

    # (0) The warm-up below the threshold, then the baseline at rest.
    run(color_to_save(schemas, WARM, output, "warmup"))
    baseline, rest = at_rest(private)
    if baseline is None:
        failures.append(
            f"(0) the private bytes never held within {REST // MIB} MiB "
            f"for {SETTLE_STEP} s"
        )
        summary["rest_mib"] = [mib(s) for s in rest]
        return summary
    summary["baseline_mib"] = mib(baseline)

    # (a) A /run releases the run's idle blocks at its end.
    stats = run(color_to_save(schemas, BLOCK, output, "block"))
    summary["a"] = level("a", baseline, 0) | {
        "released_mib": mib(stats["released_bytes"])
    }
    if stats["released_bytes"] < 64 * MIB:
        failures.append(f"(a) released_bytes {stats['released_bytes']} is under 64 MiB")

    # (b) The cached output stays live; a second request pops it and releases it.
    run_individual(BLOCK)
    summary["b64"] = level("b, 64 MiB", baseline, 64 * MIB)
    run_individual(LARGER)
    summary["b96"] = level("b, 96 MiB", baseline, 96 * MIB)

    # (c) A cache clear releases the dropped output.
    clear_individual()
    summary["c"] = level("c", baseline, 0)

    # (d) A preview and a cache clear during a /run release nothing until its end.
    nodes, _, _ = bench_cases.graph(schemas, "video-resize-h264", assets, output)
    loader = next(n for n in nodes if n["schemaId"] == "chainner:image:load_video")

    def mid_run(in_flight: Callable[[], bool]) -> dict:
        run_individual(BLOCK)
        answered = {"individual_mid_run": in_flight()}
        clear_individual()
        answered["clear_mid_run"] = in_flight()
        return answered

    video = run_sampled(backend, nodes, loader["id"], every, during=mid_run)
    during = video.get("during")
    video_stats = video["stats"]
    if video_stats is None:
        raise RuntimeError(
            "(d) the run's native profile line has no numpy_pool entry: "
            "the handler is not installed"
        )
    summary["d"] = {
        "mid_run": during,
        "releases": video_stats["releases"],
        "released_mib": mib(video_stats["released_bytes"]),
        "held_peak_mib": mib(video_stats["held_peak"]),
        "evictions": video_stats["evictions"],
    }
    if during is None:
        failures.append("(d) not mid-run: the /run ended before its first sample")
    elif not (during["individual_mid_run"] and during["clear_mid_run"]):
        failures.append(f"(d) not mid-run: {during}")
    if video_stats["releases"] != 1:
        failures.append(f"(d) the run's releases are {video_stats['releases']}, not 1")
    if video_stats["released_bytes"] < 64 * MIB:
        failures.append(
            f"(d) released_bytes {video_stats['released_bytes']} is under 64 MiB"
        )

    def error_run(step: str, view: bool) -> dict:
        """From its own level at rest, a /run of Create Color 64 MiB -> Crop (which
        raises) -> Save Image, or View Image with view, which must fail in its Crop;
        then the warm-up graph again. Its figures."""
        before, rest = at_rest(private)
        reference = rest[-1] if before is None else before
        offset = log.stat().st_size
        nodes = color_to_save(schemas, BLOCK, output, "error", crop=True)
        if view:
            # View Image reads its image eagerly; Save Image reads it through a Lazy.
            nodes[-1] = make_node(
                schemas, "view", "chainner:image:view", {0: edge("crop")}
            )
        status, body = 200, None
        try:
            body = request(backend.port, "/run", run_payload(nodes), timeout=600)
        except urllib.error.HTTPError as error:
            status, body = error.code, json.loads(error.read().decode("utf-8"))
        raised = (
            status == 500
            and isinstance(body, dict)
            and CROP_ERROR in str(body.get("exception"))
        )
        if not raised:
            failures.append(
                f"({step}) the run did not fail in its Crop: status {status}, {body}"
            )
        stats = entry(offset)
        figures = {
            "at_rest": before is not None,
            "status": status,
            "raised_in_crop": raised,
            "line_live_mib": mib(stats["live_bytes"]),
            "line_released_mib": mib(stats["released_bytes"]),
            "after_response": level(f"{step}, after the response", reference, 0),
        }
        run(color_to_save(schemas, WARM, output, "next-release"))
        figures["after_next_release"] = level(
            f"{step}, after the next release", reference, 0
        )
        return figures

    # (e) Asserted: an error run's arrays are freed after its response.
    summary["e"] = error_run("e", view=False)
    # (f) The same with View Image, which reads its image eagerly.
    summary["f"] = error_run("f", view=True)

    summary["passed"] = not failures
    return summary


def check_backend(backend: Backend) -> None:
    info = backend.info
    if not (
        info["owned_process_exited"]
        and info["owned_job_closed"]
        and info["backend_unchanged"]
        and info["interpreter_unchanged"]
    ):
        raise RuntimeError(
            f"Backend {backend.label} cleanup/identity check failed: {info}"
        )


def video_check(
    run_dir: Path, name: str, assets: Path, frames: int, every: int
) -> dict:
    output = guard_output(run_dir / f"{name}-out")
    output.mkdir()
    with Backend(name, TREE, run_dir, FFMPEG, FFPROBE, CHECK_ENV[name]) as backend:
        nodes, _, _ = bench_cases.graph(
            backend.schemas, "video-resize-h264", assets, output
        )
        loader = next(n for n in nodes if n["schemaId"] == "chainner:image:load_video")
        run = run_sampled(backend, nodes, loader["id"], every)
    check_backend(backend)
    shutil.rmtree(output)
    expect_evictions = name == "video-cap16"
    return {"run": run, "summary": judge(name, frames, run, expect_evictions)}


def images_graph(
    schemas: dict[str, dict], images_dir: Path, images: int, output: Path
) -> tuple[list[dict], str]:
    """resize-real-256's saved chain loading the first `images` images of images_dir
    and saving into output, which must be under GROWTH; the nodes and the loader's id."""
    output = guard_output(output)
    # resize-real-256 names its own folder; the assets argument is unused.
    nodes, _, _ = bench_cases.graph(schemas, "resize-real-256", GROWTH, output)
    loader = next(n for n in nodes if n["schemaId"] == "chainner:image:load_images")
    labels = [x["label"] for x in schemas["chainner:image:load_images"]["inputs"]]
    if labels[0] != "Directory" or labels[4:6] != ["Use limit", "Limit"]:
        raise RuntimeError(
            f"load_images inputs 0, 4 and 5 are {labels[0]!r}, {labels[4:6]!r}"
        )
    save = next(n for n in nodes if n["schemaId"] == "chainner:image:save")
    if Path(save["inputs"][1]["value"]) != output:
        raise RuntimeError(
            f"the save node writes to {save['inputs'][1]['value']}, not {output}"
        )
    loader["inputs"][0]["value"] = str(images_dir)
    loader["inputs"][4]["value"] = True
    loader["inputs"][5]["value"] = images
    return nodes, loader["id"]


def images_check(run_dir: Path, images_dir: Path, images: int, every: int) -> dict:
    output = guard_output(run_dir / "images-out")
    output.mkdir()
    with Backend(
        "images", TREE, run_dir, FFMPEG, FFPROBE, CHECK_ENV["images"]
    ) as backend:
        nodes, loader = images_graph(backend.schemas, images_dir, images, output)
        run = run_sampled(backend, nodes, loader, every)
    check_backend(backend)
    written = sum(1 for _ in output.rglob("*.png"))
    shutil.rmtree(output)
    summary = judge("images", images, run, expect_evictions=False)
    if written != images:
        summary["failures"].append(f"wrote {written} PNGs of {images}")
        summary["passed"] = False
    return {"run": run, "summary": summary}


def release_check(run_dir: Path, assets: Path, every: int) -> dict:
    output = guard_output(run_dir / "release-out")
    output.mkdir()
    with Backend(
        "release", TREE, run_dir, FFMPEG, FFPROBE, CHECK_ENV["release"]
    ) as backend:
        summary = release_steps(backend, output, assets, every)
    check_backend(backend)
    shutil.rmtree(output)
    return {"summary": summary}


def main(argv: list[str] | None = None) -> int:
    arguments = parser()
    args = arguments.parse_args(argv)
    for name in ("frames", "images", "every", "every_images"):
        if getattr(args, name) < 1:
            arguments.error(f"--{name.replace('_', '-')} must be at least 1")
    # The workers inherit this environment; each check sets its own variables.
    for variable in (POOL_VARIABLE, PROFILE_VARIABLE):
        if variable in os.environ:
            arguments.error(
                f"{variable} is set in this environment and would reach every worker"
            )
    require_pinned_tools()
    for required in (TREE / "run.py", TREE / "nodes" / "impl" / "_chainner_graph.pyd"):
        if not required.is_file():
            raise RuntimeError(f"the backend tree has no {required}")
    images_dir: Path = args.images_dir
    if "images" in args.checks:
        found = sum(1 for _ in itertools.islice(images_dir.glob("*.jpg"), args.images))
        if found < args.images:
            raise RuntimeError(
                f"{images_dir} holds {found} JPEGs at its top level, fewer than {args.images}"
            )
    assets = GROWTH / f"assets-{args.frames}"
    video = assets / "moving.mkv"
    if args.check:
        print(
            f"growth check ok: checks {' '.join(args.checks)}; {args.frames} frames "
            f"({'reusing ' + str(video) if video.is_file() else 'to synthesize'}), "
            f"sampled every {args.every}; {args.images} images from {images_dir}, "
            f"sampled every {args.every_images}; tree {TREE}; FFmpeg {FFMPEG}; "
            f"outputs under {GROWTH}",
            flush=True,
        )
        return 0
    run_dir = guard_output(
        GROWTH / f"run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_dir.mkdir(parents=True)
    print(f"growth: {run_dir}", flush=True)
    if set(args.checks) - {"images"}:
        synthesize(args.frames)
    results: dict[str, dict] = {}
    for name in args.checks:
        started = time.monotonic()
        try:
            if name == "images":
                result = images_check(
                    run_dir, images_dir, args.images, args.every_images
                )
            elif name == "release":
                result = release_check(run_dir, assets, args.every)
            else:
                result = video_check(run_dir, name, assets, args.frames, args.every)
        except Exception:
            trace = traceback.format_exc()
            print(f"{name} raised:\n{trace}", flush=True)
            last = trace.strip().splitlines()[-1]
            result = {
                "summary": {
                    "check": name,
                    "passed": False,
                    "failures": [f"raised {last}"],
                    "traceback": trace,
                }
            }
        results[name] = result
        summary = result["summary"]
        summary["seconds"] = round(time.monotonic() - started, 1)
        shown = {
            k: v
            for k, v in summary.items()
            if k not in {"check", "passed", "traceback"}
        }
        print(
            f"GROWTH {name} {'PASS' if summary['passed'] else 'FAIL'} {json.dumps(shown)}",
            flush=True,
        )
    (run_dir / "growth.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    passed = all(result["summary"]["passed"] for result in results.values())
    print(f"GROWTH-RESULT {'pass' if passed else 'fail'} {run_dir}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
