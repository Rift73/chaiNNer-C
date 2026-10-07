"""Exercise oracle and packaged ONNX/NCNN nodes with tiny synthetic CPU models.

The default runs the oracle baseline only (verify_runtime.oracle_source), on the
provisioned runtime the package was copied from. Pass --include-port only after
the packager has released a completed stage-3 package; it runs on the package's
runtime. Both backends run from private copies with the four
NCNN CPU-branch selectors explicitly disabled. No oracle source, portable source,
user profile, GPU, or downloaded model is modified/used.
This is correctness verification, not a performance or model-quality benchmark.
"""

from __future__ import annotations

import argparse
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
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    compare_images,
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
    sse_event_mismatches,
    sse_record,
    write_json,
)

CPU_SELECTORS = (
    "nodes/impl/ncnn/session.py",
    "nodes/impl/ncnn/auto_split.py",
    "packages/chaiNNer_ncnn/settings.py",
    "packages/chaiNNer_ncnn/ncnn/processing/upscale_image.py",
)
# Upstream selects Vulkan by importing ncnn_vulkan; compat.diff gives the oracle
# the port's form, which asks the PyPI ncnn wheel for a Vulkan GPU. Each selector
# file holds that form exactly once.
CPU_SELECTOR = b"use_gpu = ncnn.get_gpu_count() > 0"
OPTIONS = {
    "chaiNNer_pytorch": {"use_cpu": True, "use_fp16": False},
    "chaiNNer_onnx": {
        "execution_provider": "CPUExecutionProvider",
        "tensorrt_fp16_mode": False,
    },
    "chaiNNer_ncnn": {"threads": 1, "blocktime": 0, "budget_limit": 0},
}
FRAMEWORK_IDS = {
    "chainner:ncnn:" + name
    for name in (
        "load_model",
        "save_model",
        "load_models",
        "model_dim",
        "interpolate_models",
        "upscale_image",
    )
} | {
    "chainner:onnx:" + name
    for name in (
        "load_model",
        "save_model",
        "load_models",
        "model_info",
        "optimize_model",
        "convert_to_ncnn",
        "interpolate_models",
        "upscale_image",
        "rembg",
    )
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def cpu_copy(source: Path, destination: Path) -> list[dict]:
    shutil.copytree(
        source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    changes = []
    for relative in CPU_SELECTORS:
        path = destination / relative
        original = path.read_bytes()
        if original.count(CPU_SELECTOR) != 1:
            raise ValueError(f"Unreviewed NCNN CPU selector in {relative}")
        converted = original.replace(CPU_SELECTOR, b"use_gpu = False")
        path.write_bytes(converted)
        changes.append(
            {
                "path": relative,
                "source_sha256": hashlib.sha256(original).hexdigest(),
                "private_copy_sha256": hashlib.sha256(converted).hexdigest(),
                "edit": "One use_gpu = ncnn.get_gpu_count() > 0 assignment changed to False",
            }
        )
    # The custom installed host assumes its normal NCNN branch uses Vulkan.
    # Our private selectors change only that assumption, so prevent an unrelated
    # GPU lease and worker recycle for this explicitly CPU-only fixture run.
    lease = destination / "gpu_lease.py"
    if lease.is_file():
        original = lease.read_bytes()
        newline = b"\r\n" if b"\r\n" in original else b"\n"
        before = newline.join(
            (
                b'        elif schema.startswith("chainner:ncnn:"):',
                b"            # This installation uses ncnn_vulkan; no per-run CPU switch exists.",
                b"            return True",
            )
        )
        if original.count(before) != 1:
            raise ValueError("Unreviewed installed NCNN lease classification")
        converted = original.replace(
            before, before.replace(b"return True", b"return False")
        )
        lease.write_bytes(converted)
        changes.append(
            {
                "path": "gpu_lease.py",
                "source_sha256": hashlib.sha256(original).hexdigest(),
                "private_copy_sha256": hashlib.sha256(converted).hexdigest(),
                "edit": "Only NCNN needs_gpu branch returns False for this private CPU fixture",
            }
        )
    return changes


def prepare_fixtures(directory: Path) -> None:
    """Runs in the independent runtime; uses public model builders, no port code."""
    import cv2
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    directory.mkdir(parents=True, exist_ok=True)
    for name in ("onnx-batch", "ncnn-batch", "ncnn-mismatch"):
        (directory / name).mkdir()
    y, x = np.indices((39, 43))
    rgb = np.stack(
        ((x * 17 + y * 3) % 256, (x * 5 + y * 19) % 256, (x * 11 + y * 7) % 256), axis=2
    ).astype(np.uint8)
    gray = ((x * 13 + y * 23) % 256).astype(np.uint8)
    alpha = ((x + y) % 3 * 127).astype(np.uint8)
    alpha[0] = 255
    for name, image in (
        ("rgb", rgb),
        ("gray", gray),
        ("rgba", np.dstack((rgb, alpha))),
        ("opaque", np.dstack((rgb, np.full_like(alpha, 255)))),
    ):
        if not cv2.imwrite(str(directory / (name + ".png")), image):
            raise RuntimeError("Unable to write fixture PNG")

    def onnx_model(
        name: str,
        channels: int = 3,
        gain: float = 1.0,
        fixed: int | None = None,
        half: bool = False,
        rembg: bool = False,
        even: bool = False,
    ):
        shape = [1, channels, fixed or "height", fixed or "width"]
        output_shape = [1, 1 if rembg else channels, shape[2], shape[3]]
        dtype = np.float16 if half else np.float32
        weights = np.eye(channels, dtype=dtype).reshape(channels, channels, 1, 1) * gain
        if even:
            expanded = np.zeros((channels, channels, 2, 2), dtype=dtype)
            expanded[:, :, 0, 0] = weights[:, :, 0, 0]
            weights = expanded
        if rembg:
            weights = weights[:1]
        tensor_type = TensorProto.FLOAT16 if half else TensorProto.FLOAT
        graph = helper.make_graph(
            [
                helper.make_node(
                    "Conv",
                    ["input", "weights"],
                    ["output"],
                    name="conv",
                    kernel_shape=[2, 2] if even else [1, 1],
                    strides=[1, 1],
                    pads=[0, 0, 1, 1] if even else [0, 0, 0, 0],
                )
            ],
            name,
            [helper.make_tensor_value_info("input", tensor_type, shape)],
            [helper.make_tensor_value_info("output", tensor_type, output_shape)],
            [numpy_helper.from_array(weights.astype(dtype), "weights")],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
        )
        if rembg:
            # Exercise the installed byte-signature classifier; this is explicitly
            # a synthetic mask fixture, not a real U2Net or segmentation-quality test.
            helper.set_model_props(
                model, {"fixture": "1959 1960 1961 1962 1963 1964 1965"}
            )
        onnx.checker.check_model(model)
        onnx.save_model(model, directory / (name + ".onnx"))

    onnx_cases: tuple[tuple[str, dict[str, Any]], ...] = (
        ("identity", {}),
        ("gain", {"gain": 0.75}),
        ("gray", {"channels": 1}),
        ("half", {"half": True}),
        ("even", {"even": True}),
        ("fixed", {"fixed": 32}),
        ("rembg", {"fixed": 8, "rembg": True}),
    )
    for name, kwargs in onnx_cases:
        onnx_model(name, **kwargs)
    for name in ("identity", "gain"):
        shutil.copyfile(
            directory / (name + ".onnx"), directory / "onnx-batch" / (name + ".onnx")
        )

    def ncnn_model(name: str, channels: int = 3, gain: float = 1.0, half: bool = False):
        # Even fp16 element count avoids the installed legacy odd-weight padding
        # issue. A 2x2 one-channel kernel is a tested valid original CPU model.
        if half:
            params = "0=1 1=2 6=4"
            binary = b"\x47\x6b\x30\x01" + np.full(4, 0.25, np.float16).tobytes()
        else:
            params = f"0={channels} 1=1 6={channels * channels}"
            binary = b"\0\0\0\0" + (np.eye(channels, dtype=np.float32) * gain).tobytes()
        parameter = (
            "7767517\n2 2\nInput input 0 1 input\nConvolution conv 1 1 input output "
            + params
            + "\n"
        )
        (directory / (name + ".param")).write_text(parameter, encoding="utf-8")
        (directory / (name + ".bin")).write_bytes(binary)

    ncnn_cases: tuple[tuple[str, dict[str, Any]], ...] = (
        ("identity", {}),
        ("gain", {"gain": 0.75}),
        ("gray", {"channels": 1}),
        ("half", {"half": True}),
    )
    for name, kwargs in ncnn_cases:
        ncnn_model(name, **kwargs)
    for name in ("identity", "gain"):
        for suffix in (".param", ".bin"):
            shutil.copyfile(
                directory / (name + suffix), directory / "ncnn-batch" / (name + suffix)
            )
    shutil.copyfile(
        directory / "identity.param", directory / "ncnn-mismatch/identity.param"
    )
    (directory / "invalid.onnx").write_bytes(b"This is not an ONNX protobuf model")
    write_json(
        directory / "manifest.json",
        {
            "synthetic_models": True,
            "seed": "deterministic integer patterns; no RNG",
            "files": source_hashes(directory),
            "onnx_version": onnx.__version__,
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
        },
    )


def fixture_graphs(schemas: dict, assets: Path, output: Path):
    def node(name: str, schema: str, values: dict):
        return make_node(schemas, name, schema, values)

    def load(name: str, framework: str, model: str = "identity"):
        values = {
            0: str(assets / (model + (".onnx" if framework == "onnx" else ".param")))
        }
        if framework == "ncnn":
            values[1] = str(assets / (model + ".bin"))
        return node(name, f"chainner:{framework}:load_model", values)

    def save(
        name: str,
        framework: str,
        source: str,
        directory: Path,
        filename: str | dict | None = None,
    ):
        return node(
            name,
            f"chainner:{framework}:save_model",
            {0: edge(source), 1: str(directory), 2: filename or name},
        )

    def image_save(name: str, source: str, directory: Path, index: int = 0):
        return node(
            name,
            "chainner:image:save",
            {
                0: edge(source, index),
                1: str(directory),
                2: None,
                3: name,
                4: "png",
                15: "u8",
            },
        )

    def image_load(name: str, image: str = "rgb"):
        return node(name, "chainner:image:load", {0: str(assets / (image + ".png"))})

    def case(
        name: str, nodes: list[dict], files: list[str], error: bool = False
    ) -> dict:
        return {"name": name, "nodes": nodes, "files": files, "expected_error": error}

    for framework in ("onnx", "ncnn"):
        name = framework + "-load-info-save"
        directory = output / name
        info = "model_info" if framework == "onnx" else "model_dim"
        pattern = {
            0: "model-scale-{1}-purpose-{2}"
            if framework == "onnx"
            else "model-scale-{1}",
            1: edge("info"),
        }
        if framework == "onnx":
            pattern[2] = edge("info", 1)
        suffixes = [".onnx"] if framework == "onnx" else [".param", ".bin"]
        filename = (
            "model-scale-1-purpose-Generic" if framework == "onnx" else "model-scale-1"
        )
        yield case(
            name,
            [
                load("load", framework),
                node("info", f"chainner:{framework}:{info}", {0: edge("load")}),
                node("name", "chainner:utility:text_pattern", pattern),
                save("save", framework, "load", directory, edge("name")),
            ],
            [filename + suffix for suffix in suffixes],
        )

        for amount in (0, 37, 100):
            name = f"{framework}-interpolate-{amount}"
            directory = output / name
            yield case(
                name,
                [
                    load("a", framework),
                    load("b", framework, "gain"),
                    node(
                        "interpolate",
                        f"chainner:{framework}:interpolate_models",
                        {0: edge("a"), 1: edge("b"), 2: amount},
                    ),
                    node(
                        "name",
                        "chainner:utility:text_pattern",
                        {
                            0: "weights-{1}-{2}",
                            1: edge("interpolate", 1),
                            2: edge("interpolate", 2),
                        },
                    ),
                    save("save", framework, "interpolate", directory, edge("name")),
                    image_load("input"),
                    node(
                        "upscale",
                        f"chainner:{framework}:upscale_image",
                        {0: edge("interpolate"), 1: edge("input"), 2: -1},
                    ),
                    image_save("image", "upscale", directory),
                ],
                [f"weights-{100 - amount}-{amount}" + s for s in suffixes]
                + ["image.png"],
            )

        for model, image, tile, separate in (
            ("identity", "rgb", None, 0),
            ("identity", "gray", -1, 0),
            ("identity", "rgba", -1, 0),
            ("identity", "rgba", -1, 1),
            ("identity", "opaque", -1, 0),
            ("identity", "rgba", -3, 1),
            ("gray", "rgb", -1, 0),
            ("gray", "rgba", -1, 1),
            ("half", "gray", -1, 0),
        ):
            # NCNN's tested fp16 2x2 convolution changes spatial size and cannot
            # satisfy an integer upscale scale. Conversion+serialization tests
            # cover fp16 NCNN; the real inference tests cover this valid kernel.
            if framework == "ncnn" and model == "half":
                continue
            name = f"{framework}-upscale-{model}-{image}-{tile}-{separate}"
            directory = output / name
            values = {0: edge("model"), 1: edge("input"), 3: 32, 4: separate}
            if tile is not None:
                values[2] = tile
            yield case(
                name,
                [
                    load("model", framework, model),
                    image_load("input", image),
                    node("upscale", f"chainner:{framework}:upscale_image", values),
                    image_save("image", "upscale", directory),
                ],
                ["image.png"],
            )

        name = framework + "-directory-generator"
        directory = output / name
        yield case(
            name,
            [
                node(
                    "models",
                    f"chainner:{framework}:load_models",
                    {0: str(assets / (framework + "-batch"))},
                ),
                save("save", framework, "models", directory, edge("models", 3)),
            ],
            [model + suffix for model in ("gain", "identity") for suffix in suffixes],
        )

    name = "onnx-fixed-size-tiles"
    directory = output / name
    yield case(
        name,
        [
            load("model", "onnx", "fixed"),
            image_load("input", "rgba"),
            node(
                "upscale",
                "chainner:onnx:upscale_image",
                {0: edge("model"), 1: edge("input"), 4: 1},
            ),
            image_save("image", "upscale", directory),
        ],
        ["image.png"],
    )
    name = "onnx-optimize-save-upscale"
    directory = output / name
    yield case(
        name,
        [
            load("model", "onnx"),
            node("optimize", "chainner:onnx:optimize_model", {0: edge("model")}),
            save("optimized", "onnx", "optimize", directory),
            image_load("input"),
            node(
                "upscale",
                "chainner:onnx:upscale_image",
                {0: edge("optimize"), 1: edge("input")},
            ),
            image_save("image", "upscale", directory),
        ],
        ["optimized.onnx", "image.png"],
    )
    for half in (0, 1):
        name = f"onnx-convert-ncnn-fp{16 if half else 32}"
        directory = output / name
        # A 2x2 identity Conv has 36 weights and preserves spatial size through
        # asymmetric zero padding, avoiding the installed odd-fp16-padding bug.
        yield case(
            name,
            [
                load("model", "onnx", "even"),
                node(
                    "convert",
                    "chainner:onnx:convert_to_ncnn",
                    {0: edge("model"), 1: half},
                ),
                save("converted", "ncnn", "convert", directory, edge("convert", 1)),
                image_load("input"),
                node(
                    "upscale",
                    "chainner:ncnn:upscale_image",
                    {0: edge("convert"), 1: edge("input"), 2: -1},
                ),
                image_save("image", "upscale", directory),
            ],
            [f"fp{16 if half else 32}" + suffix for suffix in (".param", ".bin")]
            + ["image.png"],
        )
    for postprocess in (0, 1):
        name = f"onnx-rembg-postprocess-{postprocess}"
        directory = output / name
        yield case(
            name,
            [
                load("model", "onnx", "rembg"),
                image_load("input", "rgba"),
                node(
                    "remove",
                    "chainner:onnx:rembg",
                    {0: edge("input"), 1: edge("model"), 2: postprocess},
                ),
                image_save("image", "remove", directory),
                image_save("mask", "remove", directory, 1),
            ],
            ["image.png", "mask.png"],
        )
    yield case(
        "onnx-invalid-load", [load("invalid", "onnx", "invalid")], [], error=True
    )
    yield case(
        "ncnn-mismatched-directory",
        [
            node(
                "invalid",
                "chainner:ncnn:load_models",
                {0: str(assets / "ncnn-mismatch")},
            ),
            save(
                "save",
                "ncnn",
                "invalid",
                output / "ncnn-mismatched-directory",
                edge("invalid", 3),
            ),
        ],
        [],
        error=True,
    )


def run_backend(
    label: str,
    backend: Path,
    python: Path,
    root: Path,
    timeout: float,
    *,
    required_schema_ids: set[str] | None = None,
    graphs: Callable[[dict, Path, Path], Iterable[dict]] | None = None,
    synthetic_model_inference: bool = True,
):
    required_schema_ids = (
        FRAMEWORK_IDS if required_schema_ids is None else required_schema_ids
    )
    graphs = fixture_graphs if graphs is None else graphs
    directory = root / label
    directory.mkdir()
    for name in ("storage", "output", "profile", "appdata", "localappdata", "temp"):
        (directory / name).mkdir()
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
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
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
        "synthetic_model_inference": synthetic_model_inference,
        "real_trained_model_quality_test": False,
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
        result["owned_pid"] = process.pid
        limit = time.monotonic() + timeout
        while time.monotonic() < limit:
            if process.poll() is not None:
                raise RuntimeError(
                    f"{label} exited {process.returncode}; inspect backend.log"
                )
            try:
                if request(port, "/status", timeout=3).get("ready"):
                    break
            except OSError, urllib.error.URLError, json.JSONDecodeError:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError(f"{label} startup timeout; inspect backend.log")
        metadata = {
            key: request(port, "/" + key, timeout=30) for key in METADATA_ENDPOINTS
        }
        for key, value in metadata.items():
            write_json(directory / (key + ".json"), value)
        result.update(metadata_record(metadata))
        schemas = {node["schemaId"]: node for node in metadata["nodes"]["nodes"]}
        if not required_schema_ids <= schemas.keys():
            raise RuntimeError(
                f"Missing requested schemas: {required_schema_ids - schemas.keys()}"
            )
        events = Events(port)
        if not events.ready.wait(timeout=10) or events.error:
            raise RuntimeError(f"Cannot connect to SSE: {events.error}")
        for fixture in graphs(schemas, root / "fixtures", directory / "output"):
            name, nodes = fixture["name"], fixture["nodes"]
            expected_dead_nodes = set(fixture.get("expected_dead_nodes", ()))
            node_ids = {node["id"] for node in nodes}
            if not expected_dead_nodes <= node_ids:
                raise ValueError(f"Unknown expected dead node in {name}")
            print(f"{label}: {name}", flush=True)
            outputs = directory / "output" / name
            outputs.mkdir()
            payload = {"data": nodes, "options": OPTIONS, "sendBroadcastData": True}
            write_json(directory / (name + "-request.json"), payload)
            attempts = []
            for repeat in range(2):
                start = len(events.events)
                try:
                    response = request(port, "/run", payload, timeout=120)
                    status_code = 200
                except urllib.error.HTTPError as error:
                    if not fixture["expected_error"]:
                        raise
                    status_code = error.code
                    response = json.load(error)
                if fixture["expected_error"]:
                    if (
                        response.get("success") is True
                        or response.get("type") == "success"
                    ):
                        raise AssertionError(f"{name} unexpectedly succeeded")
                    deadline = time.monotonic() + 15
                    while not any(
                        e["event"] == "execution-error" for e in events.events[start:]
                    ):
                        if time.monotonic() >= deadline or events.error:
                            raise RuntimeError(
                                f"Missing error SSE for {name}: {events.error}"
                            )
                        time.sleep(0.05)
                    if fixture.get("settle_error_events", False):
                        # The installed executor does not await broadcasts after
                        # an error. Keep all trailing events in their own attempt
                        # before starting a repeat; this is a bounded UI/event
                        # completion wait, not a performance measurement.
                        settle_deadline = time.monotonic() + 10
                        quiet_since = time.monotonic()
                        count = len(events.events)
                        while time.monotonic() - quiet_since < 0.25:
                            if time.monotonic() >= settle_deadline or events.error:
                                raise RuntimeError(f"Error SSE did not settle: {name}")
                            time.sleep(0.05)
                            if count != len(events.events):
                                count = len(events.events)
                                quiet_since = time.monotonic()
                    received = events.events[start:]
                else:
                    check_success(response)
                    received = events.wait_for(
                        node_ids - expected_dead_nodes,
                        start,
                        expected_broadcasts={
                            node["id"]
                            for node in nodes
                            if schemas[node["schemaId"]]["outputs"]
                            and node["id"] not in expected_dead_nodes
                        },
                    )
                for event in received:
                    if event["event"] in {
                        "node-start",
                        "node-finish",
                        "node-broadcast",
                    }:
                        if event["data"].get("nodeId") in expected_dead_nodes:
                            raise AssertionError(
                                f"{name} executed an expected dead node"
                            )
                files = {
                    path.relative_to(outputs).as_posix(): file_hash(path)
                    for path in sorted(outputs.rglob("*"))
                    if path.is_file()
                }
                if set(files) != set(fixture["files"]):
                    raise AssertionError(
                        f"{name} files: expected {fixture['files']}, got {sorted(files)}"
                    )
                attempts.append(
                    {
                        "http_status": status_code,
                        "response": canonical(response, directory),
                        **sse_record(received, directory),
                        "files": files,
                    }
                )
                write_json(
                    directory / f"{name}-attempt-{repeat}.json",
                    {
                        "http_status": status_code,
                        "response": response,
                        "events": received,
                        "files": files,
                    },
                )
            coalesces = label == "converted"  # The port coalesces previews.
            if mismatches := repeat_mismatches(
                attempts[0], attempts[1], port=coalesces
            ):
                raise AssertionError(
                    f"{name} repeated request changed outputs or event contracts: "
                    f"{mismatches}"
                )
            result["fixtures"].append(
                {
                    "name": name,
                    "schema_ids": sorted({node["schemaId"] for node in nodes}),
                    "generator_node_ids": generator_node_ids(nodes, schemas),
                    "expected_error": fixture["expected_error"],
                    "expected_dead_nodes": sorted(expected_dead_nodes),
                    "repeat_rule": REPEAT_RULES[coalesces],
                    "repeat_final_values_differ": repeat_final_value_differences(
                        attempts[0], attempts[1]
                    ),
                    "contract": attempts[0],
                    "repeat_contract": attempts[1],
                    "output_directory": str(outputs),
                }
            )
            write_json(directory / "result.json", result)
        result["covered_schema_ids"] = sorted(
            {
                schema
                for fixture in result["fixtures"]
                for schema in fixture["schema_ids"]
            }
            & required_schema_ids
        )
        if set(result["covered_schema_ids"]) != required_schema_ids:
            raise AssertionError("Incomplete requested schema coverage")
        result["success"] = True
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        if events:
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
        job.close()
        if process is not None:
            process.wait(timeout=15)
            result["exit_code"] = process.returncode
            result["owned_process_exited"] = process.poll() is not None
        result["owned_job_closed"] = True
        log.close()
        write_json(directory / "result.json", result)
    return result


def compare_runs(a: dict, b: dict, python: Path) -> dict:
    contracts, event_mismatches, binary, pairs = {}, {}, {}, []
    broadcasts = {}
    for x, y in zip(a["fixtures"], b["fixtures"], strict=True):
        if x["name"] != y["name"]:
            raise AssertionError("Fixture ordering changed")
        name = x["name"]
        sides = {
            side: broadcast_multisets([run["contract"], run["repeat_contract"]])
            for side, run in (("oracle", x), ("port", y))
        }
        if any(any(attempt) for attempt in [*sides["oracle"], *sides["port"]]):
            broadcasts[name] = sides
        oracle = {
            **x["contract"],
            "name": name,
            "generator_node_ids": x["generator_node_ids"],
        }
        mismatches = [
            prefix + mismatch
            for prefix, key in (("", "contract"), ("repeat: ", "repeat_contract"))
            for mismatch in sse_event_mismatches(
                oracle,
                {
                    **y[key],
                    "name": name,
                    "generator_node_ids": y["generator_node_ids"],
                },
            )
        ]
        if mismatches:
            event_mismatches[name] = mismatches
        contracts[name] = not mismatches and all(
            x["contract"][key] == y["contract"][key]
            for key in ("http_status", "response")
        )
        for relative in x["contract"]["files"]:
            original, converted = (
                Path(x["output_directory"]) / relative,
                Path(y["output_directory"]) / relative,
            )
            if original.suffix == ".png":
                pairs.append(
                    {
                        "name": name + "/" + relative,
                        "baseline": str(original),
                        "converted": str(converted),
                    }
                )
            else:
                binary[name + "/" + relative] = (
                    original.read_bytes() == converted.read_bytes()
                )
    images = compare_images(python, pairs)
    metadata = compare_metadata(a, b)
    return {
        "contracts_equal": contracts,
        "sse_event_mismatches": event_mismatches,
        "broadcast_multisets": broadcasts,
        # Flagged, not failed (Consult D-35): the oracle's own attempts ended on
        # different final values of an iterated field.
        "oracle_repeat_final_values_differ": oracle_repeat_flags(a),
        "serialized_model_bytes_equal": binary,
        "images": images,
        **metadata,
        "success": all(contracts.values())
        and all(binary.values())
        and all(item["pass"] for item in images)
        and all(metadata["metadata_equal"].values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=PROJECT / "out/chaiNNer-C")
    parser.add_argument(
        "--include-port",
        action="store_true",
        help="Explicitly run the released stage-3 package after the baseline",
    )
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--prepare-fixtures", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prepare_fixtures:
        prepare_fixtures(args.prepare_fixtures)
        return 0
    package = args.package.resolve(strict=True)
    manifest = json.loads(
        (package / "chainner-c-package.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("state") != "complete"
        or manifest["identity"]["destination"] != str(package)
        or manifest["identity"]["project"] != str(PROJECT)
    ):
        raise ValueError(
            "A completed package manifest matching this project/location is required"
        )
    python = package / "python/python/python.exe"
    if not python.is_file() or not (package / "portable").is_file():
        raise ValueError("Independent portable Python runtime missing")
    source, oracle_python, oracle = oracle_source(manifest, ORACLE)
    sources = {"baseline": source}
    pythons = {"baseline": oracle_python, "converted": python}
    if args.include_port:
        sources["converted"] = package / "resources/src"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = PROJECT / "native/reports" / ("framework-runtime-" + stamp)
    root.mkdir(parents=True, exist_ok=False)
    protected = {label: source_hashes(path) for label, path in sources.items()}
    report: dict = {
        "utc": datetime.now(UTC).isoformat(),
        "package": str(package),
        "oracle": oracle,
        "success": False,
        "mode": "oracle-versus-port"
        if args.include_port
        else "baseline-fixture-validation-only",
        "cpu_only": True,
        "performance_benchmark": False,
        "trained_models": False,
        "ncNN_runtime_policy": "Private source copies select the installed CPU branch; default Vulkan policy is not exercised",
        "limitations": [
            "Tiny synthetic models validate application/runtime contracts, not trained-model image quality or GPU providers.",
            "The RemBg fixture is a tiny Conv carrying the installed classifier signature; no real U2Net model is claimed.",
            "RemBg alpha matting is covered by test_framework_images against the live PyMatting 1.1.16; these HTTP cases use default alpha-matting false.",
            "CPU execution retains the installed public ONNX Runtime and NCNN inference engines.",
            "PNG decoded components and serialized model bytes require exact equality; no numerical tolerance is used.",
            SSE_EVENT_COMPARISON,
            METADATA_COMPARISON,
        ],
        "runs": [],
        "private_cpu_selector_edits": {},
    }
    try:
        creation = subprocess.run(
            [
                str(python),
                "-B",
                str(Path(__file__).resolve()),
                "--prepare-fixtures",
                str(root / "fixtures"),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if creation.returncode:
            raise RuntimeError(f"Fixture preparation failed: {creation.stderr}")
        for label, source in sources.items():
            copied = root / (label + "-src")
            report["private_cpu_selector_edits"][label] = cpu_copy(source, copied)
            report["runs"].append(
                run_backend(label, copied, pythons[label], root, args.startup_timeout)
            )
        if args.include_port:
            report["comparison"] = compare_runs(
                report["runs"][0], report["runs"][1], python
            )
            report["success"] = report["comparison"]["success"]
        else:
            report["success"] = report["runs"][0]["success"]
    except Exception as error:
        report["error"], report["traceback"] = str(error), traceback.format_exc()
        report["runs"] = [
            json.loads(path.read_text(encoding="utf-8"))
            for label in sources
            if (path := root / label / "result.json").is_file()
        ]
    finally:
        report["original_source_trees_unchanged"] = {
            label: protected[label] == source_hashes(path)
            for label, path in sources.items()
        }
        report["all_owned_processes_stopped"] = all(
            run.get("owned_process_exited") and run.get("owned_job_closed")
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
