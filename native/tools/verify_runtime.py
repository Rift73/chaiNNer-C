"""Compare the oracle backend and portable C port using isolated CPU fixtures.

Run after package_port.py and make_oracle.py. The baseline is a per-run copy of
make_oracle.py's tree (the installed backend source plus compat.diff, with
chaiNNer-C's chainner_ext), run on the provisioned runtime the package was
copied from; the port runs on the package's runtime. An oracle whose
chainner_ext differs from the package's, or any other oracle interpreter, is
refused (oracle_source). The port's Dependency declarations may differ
from the oracle's only as DECLARATION_CHANGES and the lock say (compare_metadata). This
exercises real HTTP/SSE servers and PNG outputs, not the Electron UI or ML/GPU
inference. No performance timings are measured. Only private process trees
created by this helper are shut down.
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
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from ctypes import wintypes
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import package_files
import package_manifest
from packaging.utils import NormalizedName, canonicalize_name

PROJECT = Path(__file__).resolve().parents[2]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
# make_oracle.py's tree and record; every runtime verifier's baseline is its src.
ORACLE = PROJECT / "native/build/oracle"
ORACLE_RECORD = "oracle.json"
# chaiNNer-C's C chainner_ext in the oracle (Consult 6 P1): each backend/src file
# and its place under the oracle's src. chainner_native.dll goes beside the pyd,
# whose own directory Windows searches for its dependencies; __init__.py adds
# <src>/nodes/impl, which in the oracle is upstream's and holds no DLL.
CHAINNER_EXT = {
    "chainner_ext/__init__.py": "chainner_ext/__init__.py",
    "chainner_ext/__init__.pyi": "chainner_ext/__init__.pyi",
    "chainner_ext/chainner_ext.pyd": "chainner_ext/chainner_ext.pyd",
    "nodes/impl/chainner_native.dll": "chainner_ext/chainner_native.dll",
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def request(port: int, path: str, data: dict | None = None, timeout: float = 15):
    payload = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=payload,
        headers={"Content-Type": "application/json"} if payload else {},
    )
    with OPENER.open(req, timeout=timeout) as response:
        return json.load(response)


class OwnedJob:
    """Windows kill-on-close job containing only this verification server tree."""

    def __init__(self):
        if os.name != "nt":
            raise RuntimeError(
                "This packaged-runtime verifier currently targets Windows"
            )

        class IoCounters(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_ulonglong)
                for name in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            ]

        class BASIC(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class EXTENDED(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BASIC),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self.kernel.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
        ]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = EXTENDED()
        info.BasicLimitInformation.LimitFlags = (
            0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not self.kernel.SetInformationJobObject(
            self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)
        ):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, process: subprocess.Popen):
        # Popen still holds its own handle, so Windows cannot reuse the pid: the
        # handle opened here names the same process.
        handle = self.kernel.OpenProcess(
            0x0101,  # PROCESS_SET_QUOTA | PROCESS_TERMINATE
            False,
            process.pid,
        )
        assigned = bool(handle) and self.kernel.AssignProcessToJobObject(
            self.handle, handle
        )
        error = ctypes.get_last_error()
        if handle:
            self.kernel.CloseHandle(handle)
        if not assigned:
            process.terminate()
            process.wait(timeout=10)
            raise ctypes.WinError(error)

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def run_owned(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict | None = None,
    data: bytes | None = None,
    timeout: float = 120,
):
    job = OwnedJob()
    process = None
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE if data is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        job.assign(process)
        stdout, stderr = process.communicate(data, timeout=timeout)
        if process.returncode:
            raise RuntimeError(
                f"Owned command failed ({process.returncode}): {command!r}\n{stderr.decode('utf-8', errors='replace')}"
            )
        return stdout
    finally:
        job.close()
        if process is not None:
            process.wait(timeout=15)


class Events:
    def __init__(self, port: int):
        self.port = port
        self.events = []
        self.error = None
        self.ready = threading.Event()
        self.stopping = threading.Event()
        self.connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        self.thread = threading.Thread(target=self.collect, daemon=True)
        self.thread.start()

    def collect(self):
        try:
            self.connection.request("GET", "/sse")
            # The installed Sanic version flushes streaming response headers with
            # the first event. Signal request dispatch so the fixture can create
            # that event instead of waiting for headers before executing work.
            self.ready.set()
            response = self.connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f"SSE status {response.status}")
            if "text/event-stream" not in response.getheader("Content-Type", ""):
                raise RuntimeError("SSE has the wrong content type")
            event = None
            data = []
            while not self.stopping.is_set():
                raw = response.readline()
                if not raw:
                    break
                line = raw.decode("utf-8").rstrip("\r\n")
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].strip())
                elif not line and data:
                    self.events.append(
                        {
                            "event": event or "message",
                            "data": json.loads("\n".join(data)),
                        }
                    )
                    event, data = None, []
        except Exception as error:
            if not self.stopping.is_set():
                self.error = str(error)
            self.ready.set()

    def wait_for(
        self,
        expected: set[str],
        start: int,
        timeout: float = 15,
        expected_broadcasts: set[str] | None = None,
        broadcast_after_finish: set[str] | None = None,
    ):
        """Poll the captured events from start until the expected ones arrived.

        expected: nodes that need a node-finish. expected_broadcasts: nodes that
        need any node-broadcast. broadcast_after_finish: nodes that need a
        node-broadcast at a later index than their last node-finish, for a
        node that broadcasts again after it finishes (a generator restoring its
        static outputs), where the earlier broadcasts would otherwise satisfy
        expected_broadcasts before the last one arrives.
        """
        limit = time.monotonic() + timeout
        while time.monotonic() < limit:
            events = self.events[start:]
            if self.error:
                raise RuntimeError(self.error)
            errors = [x for x in events if x["event"] == "execution-error"]
            if errors:
                raise RuntimeError(f"SSE execution errors: {errors}")
            finish_index = {
                x["data"].get("nodeId"): i
                for i, x in enumerate(events)
                if x["event"] == "node-finish"
            }
            finished = set(finish_index)
            broadcast = {
                x["data"].get("nodeId")
                for x in events
                if x["event"] == "node-broadcast"
            }
            # A node that has not finished has no later broadcast: len(events) is
            # past every index.
            rebroadcast = {
                x["data"].get("nodeId")
                for i, x in enumerate(events)
                if x["event"] == "node-broadcast"
                and i > finish_index.get(x["data"].get("nodeId"), len(events))
            }
            if (
                expected <= finished
                and (expected_broadcasts or set()) <= broadcast
                and (broadcast_after_finish or set()) <= rebroadcast
            ):
                return events
            time.sleep(0.05)
        raise RuntimeError(
            f"Missing SSE events: finished={sorted(expected - finished)}, "
            f"broadcast={sorted((expected_broadcasts or set()) - broadcast)}, "
            "broadcast_after_finish="
            f"{sorted((broadcast_after_finish or set()) - rebroadcast)}"
        )

    def close(self):
        self.stopping.set()
        if self.connection.sock:
            try:
                self.connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.connection.close()
        self.thread.join(timeout=2)


def make_node(
    schemas: dict[str, dict], node_id: str, schema_id: str, overrides: dict[int, object]
) -> dict[str, Any]:
    schema = schemas[schema_id]
    inputs = []
    unknown = set(overrides) - {x["id"] for x in schema["inputs"]}
    if unknown:
        raise ValueError(f"Unknown input IDs {unknown} for {schema_id}")
    for item in schema["inputs"]:
        value = overrides.get(item["id"], item.get("def"))
        inputs.append(
            value
            if isinstance(value, dict) and value.get("type") == "edge"
            else {"type": "value", "value": value}
        )
    return {
        "id": node_id,
        "schemaId": schema_id,
        "inputs": inputs,
        "parent": None,
        "nodeType": schema["kind"],
    }


def edge(node_id: str, index: int = 0):
    return {"type": "edge", "id": node_id, "index": index}


def gradient(
    schemas: dict[str, dict], node_id: str, style: str = "Horizontal", rgba: bool = True
):
    colors = (
        ([0.11, 0.29, 0.57, 0.31], [0.89, 0.73, 0.41, 0.93])
        if rgba
        else ([0.4, 0.42, 0.81], [0.63, 0.69, 0.98])
    )
    color_kind = "rgba" if rgba else "rgb"
    return make_node(
        schemas,
        node_id,
        "chainner:image:create_gradient",
        {
            0: 32,
            1: 24,
            9: json.dumps({"kind": color_kind, "values": colors[0]}),
            10: json.dumps({"kind": color_kind, "values": colors[1]}),
            3: style,
            4: 37,
            5: 27,
            6: 12,
            7: 84,
            8: 29,
        },
    )


def fixture_graphs(schemas: dict[str, dict], output: Path):
    def save(nodes: list[dict], name: str):
        nodes.append(
            make_node(
                schemas,
                name + "-save",
                "chainner:image:save",
                {
                    0: edge(nodes[-1]["id"]),
                    1: str(output),
                    2: None,
                    3: name,
                    4: "png",
                    15: "u8",
                },
            )
        )
        return name, nodes

    def gray_gradient(node_id: str, style: str = "Horizontal"):
        return make_node(
            schemas,
            node_id,
            "chainner:image:create_gradient",
            {
                0: 32,
                1: 24,
                3: style,
                9: json.dumps({"kind": "grayscale", "values": [0.05]}),
                10: json.dumps({"kind": "grayscale", "values": [0.95]}),
            },
        )

    def checkerboard(node_id: str):
        return make_node(
            schemas,
            node_id,
            "chainner:image:create_checkerboard",
            {
                0: 32,
                1: 24,
                2: json.dumps({"kind": "rgb", "values": [0.12, 0.71, 0.39]}),
                3: json.dumps({"kind": "rgb", "values": [0.82, 0.24, 0.61]}),
                4: 5,
            },
        )

    def append_image(nodes: list[dict], node_id: str, schema: str, values: dict):
        nodes.append(
            make_node(schemas, node_id, schema, {0: edge(nodes[-1]["id"]), **values})
        )

    def scalar_strip(nodes: list[dict], name: str, scales: tuple[float, ...]):
        """Consume every scalar output and turn it into one visible gray swatch.

        Full numeric broadcast types are also compared, so PNG quantization does
        not replace the scalar parity check.
        """
        scalar_id = nodes[-1]["id"]
        strips = {}
        for index, scale in enumerate(scales):
            value = edge(scalar_id, index)
            if scale != 1:
                node_id = f"{name}-scale-{index}"
                nodes.append(
                    make_node(
                        schemas,
                        node_id,
                        "chainner:utility:math",
                        {0: value, 1: "mul", 2: scale},
                    )
                )
                value = edge(node_id)
            color_id, image_id = f"{name}-color-{index}", f"{name}-image-{index}"
            nodes.append(
                make_node(
                    schemas,
                    color_id,
                    "chainner:utility:color_from_channels",
                    {0: 0, 1: value},
                )
            )
            nodes.append(
                make_node(
                    schemas,
                    image_id,
                    "chainner:image:create_color",
                    {0: edge(color_id), 1: 8, 2: 8},
                )
            )
            strips[index] = edge(image_id)
        nodes.append(
            make_node(schemas, name + "-strip", "chainner:image:stack", strips)
        )
        return save(nodes, name)

    for style in ("Horizontal", "Vertical", "Diagonal", "Radial", "Conic"):
        name = "adjust-" + style.lower()
        nodes = [gradient(schemas, name + "-gradient", style)]
        for suffix, schema, values in (
            ("brightness", "chainner:image:brightness_and_contrast", {1: 13, 2: 24}),
            ("invert", "chainner:image:invert", {}),
            ("clamp", "chainner:image:clamp", {1: 0.07, 2: 0.91}),
            ("opacity", "chainner:image:opacity", {1: 68}),
        ):
            nodes.append(
                make_node(
                    schemas,
                    name + "-" + suffix,
                    schema,
                    {0: edge(nodes[-1]["id"]), **values},
                )
            )
        yield save(nodes, name)
    blend = schemas["chainner:image:blend"]
    mode = next(x for x in blend["inputs"] if x["id"] == 2)
    screen = next(
        x["value"] for x in mode["options"] if x["option"].lower() == "screen"
    )
    nodes = [
        gradient(schemas, "blend-base"),
        gradient(schemas, "blend-overlay", "Vertical"),
    ]
    nodes.append(
        make_node(
            schemas,
            "blend-screen",
            "chainner:image:blend",
            {0: edge("blend-base"), 1: edge("blend-overlay"), 2: screen},
        )
    )
    yield save(nodes, "blend")
    nodes = [gradient(schemas, "normals-input", "Diagonal", rgba=False)]
    nodes.append(
        make_node(
            schemas,
            "normals-normalize",
            "chainner:image:normalize_normal_map",
            {0: edge("normals-input")},
        )
    )
    nodes.append(
        make_node(
            schemas,
            "normals-scale",
            "chainner:image:strengthen_normals",
            {0: edge("normals-normalize"), 1: 142},
        )
    )
    yield save(nodes, "normals")

    # Every added fixture ends in a real Save Image node. Intermediate outputs
    # feed that sink, including all outputs of bounding boxes, materials,
    # statistics and metrics; these graphs cannot pass through unused nodes.
    nodes = [checkerboard("checker-source")]
    nodes.append(
        make_node(
            schemas,
            "checker-color",
            "chainner:image:create_color",
            {
                0: json.dumps({"kind": "rgba", "values": [0.24, 0.51, 0.83, 0.61]}),
                1: 32,
                2: 24,
            },
        )
    )
    nodes.append(
        make_node(
            schemas,
            "checker-blend",
            "chainner:image:blend",
            {0: edge("checker-source"), 1: edge("checker-color"), 2: screen},
        )
    )
    yield save(nodes, "checker-and-color")

    for number, method in enumerate(("Simplex", "Value Noise", "Smooth Value Noise")):
        name = "generated-noise-" + str(number)
        nodes = [
            make_node(
                schemas,
                name + "-source",
                "chainner:image:create_noise",
                {
                    0: 32,
                    1: 24,
                    2: 1729,
                    3: method,
                    4: 7,
                    5: 83,
                    6: "Pink noise",
                    7: 3,
                    8: 1.7,
                    9: 2.3,
                    10: int(number == 0),
                    11: int(number == 0),
                    12: int(number == 1),
                },
            )
        ]
        yield save(nodes, name)

    for number, mode_name in enumerate(
        ("gaussian", "uniform", "salt_and_pepper", "speckle", "poisson")
    ):
        name = "added-noise-" + mode_name
        nodes = [gradient(schemas, name + "-source", "Diagonal")]
        append_image(
            nodes,
            name + "-noise",
            "chainner:image:add_noise",
            {1: mode_name, 2: "rgb" if number % 2 == 0 else "gray", 3: 19, 4: 1729},
        )
        yield save(nodes, name)

    nodes = [gradient(schemas, "layout-source", "Diagonal")]
    append_image(nodes, "layout-flip", "chainner:image:flip", {1: -1})
    append_image(nodes, "layout-shift", "chainner:image:shift", {1: 3, 2: -2, 3: 1})
    append_image(nodes, "layout-content", "chainner:image:crop_content", {1: 0})
    append_image(nodes, "layout-split", "chainner:image:split_channels", {})
    nodes.append(
        make_node(
            schemas,
            "layout-rgba",
            "chainner:image:combine_rgba",
            {i: edge("layout-split", i) for i in range(4)},
        )
    )
    append_image(nodes, "layout-alpha", "chainner:image:split_transparency", {})
    nodes.append(
        make_node(
            schemas,
            "layout-merge",
            "chainner:image:merge_channels",
            {0: edge("layout-alpha"), 1: edge("layout-alpha", 1)},
        )
    )
    nodes.append(
        make_node(
            schemas,
            "layout-transparency",
            "chainner:image:merge_transparency",
            {0: edge("layout-merge"), 1: edge("layout-alpha", 1)},
        )
    )
    yield save(nodes, "channels-and-layout")

    nodes = [gray_gradient("bbox-source")]
    append_image(nodes, "bbox-shift", "chainner:image:shift", {1: 3, 2: 2, 3: 0})
    append_image(nodes, "bbox-measure", "chainner:image:get_bbox", {1: 50})
    nodes.append(
        make_node(
            schemas,
            "bbox-crop",
            "chainner:image:crop",
            {
                0: edge("bbox-shift"),
                1: 2,
                4: edge("bbox-measure", 0),
                3: edge("bbox-measure", 1),
                8: edge("bbox-measure", 2),
                7: edge("bbox-measure", 3),
            },
        )
    )
    append_image(nodes, "bbox-wrap", "chainner:image:shift", {1: -3, 2: 5, 3: 2})
    yield save(nodes, "bounding-box")

    nodes = [gradient(schemas, "hue-source", "Diagonal")]
    append_image(
        nodes,
        "hue-adjust",
        "chainner:image:hue_and_saturation",
        {1: 31.3, 2: 15.7, 3: -25.3},
    )
    yield save(nodes, "hue-and-saturation")

    for method in (0, 1):
        name = "generated-threshold-" + str(method)
        nodes = [gradient(schemas, name + "-source", "Diagonal", rgba=False)]
        append_image(
            nodes, name + "-measure", "chainner:image:generate_threshold", {1: method}
        )
        nodes.append(
            make_node(
                schemas,
                name + "-apply",
                "chainner:image:threshold",
                {
                    0: edge(name + "-source"),
                    1: edge(name + "-measure"),
                    2: 87,
                    3: method,
                    4: method,
                    5: 3 if method else 0,
                },
            )
        )
        yield save(nodes, name)
    nodes = [gradient(schemas, "threshold-truncate-source", rgba=False)]
    append_image(
        nodes,
        "threshold-truncate-apply",
        "chainner:image:threshold",
        {1: 53, 2: 87, 3: 2},
    )
    yield save(nodes, "threshold-truncate")
    for method in (0, 1):
        name = "adaptive-threshold-" + str(method)
        nodes = [gray_gradient(name + "-source", "Diagonal")]
        append_image(
            nodes,
            name + "-apply",
            "chainner:image:threshold_adaptive",
            {1: 91, 2: method, 3: method, 4: 2, 5: 3},
        )
        yield save(nodes, name)

    nodes = [
        gradient(schemas, "material-albedo", "Diagonal", rgba=False),
        gray_gradient("material-metal"),
        gray_gradient("material-rough", "Vertical"),
    ]
    nodes.append(
        make_node(
            schemas,
            "material-specular",
            "chainner:image:metal_to_specular",
            {
                0: edge("material-albedo"),
                1: edge("material-metal"),
                2: edge("material-rough"),
            },
        )
    )
    nodes.append(
        make_node(
            schemas,
            "material-roundtrip",
            "chainner:image:specular_to_metal",
            {i: edge("material-specular", i) for i in range(3)},
        )
    )
    nodes.append(
        make_node(
            schemas,
            "material-outputs",
            "chainner:image:stack",
            {i: edge("material-roundtrip", i) for i in range(3)},
        )
    )
    yield save(nodes, "material-conversion")

    nodes = [gradient(schemas, "normal-generator-source", "Radial")]
    append_image(
        nodes,
        "normal-generator-apply",
        "chainner:image:normal_generator",
        {1: 2, 2: 1.5, 3: 0.1, 4: 2, 5: "sobel", 7: "height", 16: 1, 17: 1},
    )
    yield save(nodes, "normal-generator")

    nodes = [gray_gradient("statistics-source", "Diagonal")]
    append_image(
        nodes, "statistics-measure", "chainner:image:image_statistics", {1: 37.5}
    )
    yield scalar_strip(nodes, "statistics", (1, 1, 1, 1))
    nodes = [gradient(schemas, "metrics-source", "Diagonal", rgba=False)]
    append_image(nodes, "metrics-shift", "chainner:image:shift", {1: 2, 2: 1, 3: 2})
    nodes.append(
        make_node(
            schemas,
            "metrics-measure",
            "chainner:image:image_metrics",
            {0: edge("metrics-source"), 1: edge("metrics-shift")},
        )
    )
    yield scalar_strip(nodes, "metrics", (255, 1, 255))

    nodes = [checkerboard("palette-source"), gray_gradient("palette-mask")]
    nodes.append(
        make_node(
            schemas,
            "palette-extract",
            "chainner:image:palette_from_image",
            {0: edge("palette-source"), 1: "all"},
        )
    )
    nodes.append(
        make_node(
            schemas,
            "palette-apply",
            "chainner:image:lut",
            {0: edge("palette-mask"), 1: edge("palette-extract")},
        )
    )
    yield save(nodes, "palette")

    nodes = [checkerboard("filter-source")]
    append_image(nodes, "filter-highpass", "chainner:image:high_pass", {1: 1.5, 2: 1.3})
    # Positive threshold and enabled contrast adaptation select the new C paths.
    append_image(
        nodes, "filter-unsharp", "chainner:image:sharpen", {1: 1.1, 2: 1.7, 3: 3}
    )
    append_image(
        nodes, "filter-adaptive", "chainner:image:sharpen_hbf", {2: 1.5, 3: 1, 4: 1}
    )
    yield save(nodes, "filter-arithmetic")

    for color_space in ("RGB", "L*a*b*"):
        name = "color-fix-" + ("rgb" if color_space == "RGB" else "lab")
        nodes = [
            gradient(schemas, name + "-source", "Diagonal", rgba=False),
            checkerboard(name + "-reference"),
        ]
        nodes.append(
            make_node(
                schemas,
                name + "-average",
                "chainner:image:average_color_fix",
                {0: edge(name + "-source"), 1: edge(name + "-reference"), 2: 37.5},
            )
        )
        nodes.append(
            make_node(
                schemas,
                name + "-transfer",
                "chainner:image:color_transfer",
                {
                    0: edge(name + "-average"),
                    1: edge(name + "-reference"),
                    2: color_space,
                    3: 1,
                    4: 1,
                    5: "mean_std",
                },
            )
        )
        yield save(nodes, name)

    # Broaden the original fixture set to include the remaining first-pass nodes.
    nodes = [gradient(schemas, "arithmetic-source", "Diagonal")]
    for suffix, schema, values in (
        ("add", "chainner:image:add", {1: 12}),
        ("multiply", "chainner:image:multiply", {1: 0.7}),
        ("divide", "chainner:image:divide", {1: 0.91}),
        (
            "levels",
            "chainner:image:color_levels",
            {5: 0.03, 6: 0.96, 7: 1.2, 8: 0.04, 9: 0.94},
        ),
        ("stretch", "chainner:image:stretch_contrast", {1: 2, 4: 12, 5: 230}),
        ("premultiply", "chainner:image:premultiplied_alpha", {}),
        ("linear", "chainner:image:log2lin", {}),
    ):
        append_image(nodes, "arithmetic-" + suffix, schema, values)
    yield save(nodes, "arithmetic-and-levels")

    nodes = [gradient(schemas, "normal-combine-source", "Diagonal", rgba=False)]
    append_image(
        nodes, "normal-combine-normalize", "chainner:image:normalize_normal_map", {}
    )
    append_image(nodes, "normal-combine-balance", "chainner:image:balance_normals", {})
    nodes.append(
        make_node(
            schemas,
            "normal-combine-add",
            "chainner:image:add_normals",
            {
                0: edge("normal-combine-normalize"),
                1: 67,
                2: edge("normal-combine-balance"),
                3: 113,
                4: 0,
            },
        )
    )
    append_image(
        nodes,
        "normal-combine-octahedral",
        "chainner:image:convert_normal_map",
        {1: "DirectX", 2: "Octahedral"},
    )
    yield save(nodes, "normal-combine")
    yield save(
        [
            make_node(
                schemas,
                "colorwheel-source",
                "chainner:image:create_colorwheel",
                {0: 32},
            )
        ],
        "colorwheel",
    )

    # Third-pass fixtures exercise the new arithmetic paths explicitly. Shared
    # image-buffer conversion is also covered by every Save Image above/below.
    for name, conversions in (
        ("color-model-cmyk", ((1000, 6), (6, 1))),
        ("color-model-hue", ((1000, 4), (1002, 5), (1003, 1))),
        ("color-model-lch", ((1000, 10), (1004, 12), (1005, 10), (1004, 1))),
    ):
        nodes = [gradient(schemas, name + "-source", "Diagonal")]
        for index, (source, target) in enumerate(conversions):
            append_image(
                nodes,
                f"{name}-convert-{index}",
                "chainner:image:change_colorspace",
                {1: source, 2: target, 3: int(name != "color-model-cmyk")},
            )
        yield save(nodes, name)

    key = json.dumps({"kind": "rgb", "values": [0.12, 0.43, 0.88]})

    def matting_source(name: str):
        # The endpoints match the real small solver fixtures: the left edge is
        # the key color, and the right edge is a clearly separated foreground.
        return make_node(
            schemas,
            name,
            "chainner:image:create_gradient",
            {
                0: 13,
                1: 12,
                9: key,
                10: json.dumps({"kind": "rgb", "values": [1, 1, 0]}),
                3: "Horizontal",
            },
        )

    for suffix, method, preview in (
        ("binary", 1, 0),
        ("trimap", 2, 1),
        ("matting", 2, 0),
    ):
        name = "chroma-" + suffix
        nodes = [matting_source(name + "-source")]
        # chroma-matting runs the full solvers. PyMatting 1.1.16's foreground is
        # deterministic (Consult 11 D-17.3), so the whole RGBA result is saved.
        append_image(
            nodes,
            name + "-key",
            "chainner:image:chroma_key",
            {1: key, 2: method, 3: 20, 4: 15, 5: 50, 6: 0, 7: 0, 8: preview},
        )
        yield save(nodes, name)

    nodes = [matting_source("alpha-matting-source")]
    strips = {}
    for index, (width, value) in enumerate(((3, 0), (7, 0.5), (3, 1))):
        name = "alpha-matting-strip-" + str(index)
        nodes.append(
            make_node(
                schemas,
                name,
                "chainner:image:create_color",
                {
                    0: json.dumps({"kind": "grayscale", "values": [value]}),
                    1: width,
                    2: 12,
                },
            )
        )
        strips[index] = edge(name)
    nodes.append(
        make_node(schemas, "alpha-matting-trimap", "chainner:image:stack", strips)
    )
    # Stack Images widens grayscale strips to RGB; the solver requires one plane.
    append_image(
        nodes,
        "alpha-matting-trimap-gray",
        "chainner:image:change_colorspace",
        {1: 1000, 2: 0, 3: 0},
    )
    nodes.append(
        make_node(
            schemas,
            "alpha-matting-apply",
            "chainner:image:alpha_matting",
            {
                0: edge("alpha-matting-source"),
                1: edge("alpha-matting-trimap-gray"),
                2: 240,
                3: 15,
            },
        )
    )
    yield save(nodes, "alpha-matting")

    nodes = [gradient(schemas, "pad-stack-source", "Diagonal")]
    append_image(
        nodes,
        "pad-stack-reflect",
        "chainner:image:pad",
        {1: 4, 3: 1, 5: 3, 6: 2, 7: 5, 8: 4},
    )
    nodes.append(
        make_node(
            schemas,
            "pad-stack-resize",
            "chainner:image:stack",
            {
                0: edge("pad-stack-source"),
                1: edge("pad-stack-reflect"),
                4: "horizontal",
            },
        )
    )
    append_image(nodes, "pad-stack-pixelate", "chainner:image:pixelate", {1: 3, 2: 5})
    append_image(
        nodes,
        "pad-stack-color",
        "chainner:image:pad",
        {
            1: 7,
            2: json.dumps({"kind": "rgba", "values": [0.2, 0.7, 0.4, 0.35]}),
            3: 0,
            4: 2,
        },
    )
    yield save(nodes, "pad-stack-pixelate")

    nodes = [gradient(schemas, "rotate-right-angle-source", "Diagonal")]
    append_image(
        nodes,
        "rotate-right-angle-apply",
        "chainner:image:rotate",
        {1: 90, 2: 0, 3: 1, 4: 1},
    )
    yield save(nodes, "rotate-right-angle")

    for operation in ("median", "mean", "minimum", "maximum"):
        name = "z-stack-" + operation
        nodes = [
            gradient(schemas, name + "-horizontal"),
            gradient(schemas, name + "-vertical", "Vertical"),
            gradient(schemas, name + "-diagonal", "Diagonal"),
        ]
        nodes.append(
            make_node(
                schemas,
                name + "-combine",
                "chainner:image:z_stack",
                {0: operation, **{i + 1: edge(n["id"]) for i, n in enumerate(nodes)}},
            )
        )
        yield save(nodes, name)

    # The real generator/collector pair rearranges all twelve tiles. Merely
    # calling a collector without its sequence would leave its C path untested.
    nodes = [gradient(schemas, "spritesheet-source", "Diagonal")]
    append_image(
        nodes,
        "spritesheet-split",
        "chainner:image:split_spritesheet",
        {1: 3, 2: 4, 3: 0},
    )
    append_image(
        nodes,
        "spritesheet-merge",
        "chainner:image:merge_spritesheet",
        {1: 4, 2: 3, 3: 1},
    )
    yield save(nodes, "spritesheet-rearrange")

    nodes = [gradient(schemas, "quantize-local-source", "Diagonal", rgba=False)]
    nodes.append(
        make_node(
            schemas,
            "quantize-local-reference",
            "chainner:image:create_checkerboard",
            {
                0: 16,
                1: 12,
                2: json.dumps({"kind": "rgb", "values": [0.12, 0.71, 0.39]}),
                3: json.dumps({"kind": "rgb", "values": [0.82, 0.24, 0.61]}),
                4: 3,
            },
        )
    )
    nodes.append(
        make_node(
            schemas,
            "quantize-local-apply",
            "chainner:image:quantize_to_referece",  # Preserve installed schema typo.
            {
                0: edge("quantize-local-source"),
                1: edge("quantize-local-reference"),
                2: 2,
                3: 37.5,
            },
        )
    )
    yield save(nodes, "quantize-local-reference")

    nodes = [checkerboard("lens-blur-source")]
    append_image(
        nodes, "lens-blur-apply", "chainner:image:lens_blur", {1: 3, 2: 1, 3: 1}
    )
    yield save(nodes, "lens-blur")

    for algorithm, suffix in ((6, "prewitt"), (8, "laplacian-denoise")):
        name = "edges-" + suffix
        nodes = [checkerboard(name + "-source")]
        append_image(
            nodes,
            name + "-apply",
            "chainner:image:edge_detection",
            {1: 1.3, 2: algorithm, 3: 1, 4: 1, 5: 2},
        )
        yield save(nodes, name)

    for shape, suffix in ((0, "square"), (1, "cross")):
        name = "morphology-" + suffix
        nodes = [checkerboard(name + "-source")]
        append_image(
            nodes, name + "-dilate", "chainner:image:dilate", {1: shape, 2: 2, 3: 2}
        )
        append_image(
            nodes, name + "-erode", "chainner:image:erode", {1: shape, 2: 1, 3: 1}
        )
        yield save(nodes, name)

    nodes = [gray_gradient("distance-binary-source", "Radial")]
    append_image(
        nodes,
        "distance-binary-threshold",
        "chainner:image:threshold",
        {1: 53, 2: 100, 3: 0, 4: 0},
    )
    append_image(
        nodes,
        "distance-binary-apply",
        "chainner:image:distance_transform",
        {1: 7, 2: 0},
    )
    yield save(nodes, "distance-binary")

    # Fourth-pass image fixtures are separate from the synthetic model-weight
    # tests. All dimensions stay small, and every new computation reaches PNG.
    nodes = [gradient(schemas, "palette-median-source", "Diagonal", rgba=False)]
    append_image(
        nodes,
        "palette-median-extract",
        "chainner:image:palette_from_image",
        {1: "median", 2: 7},
    )
    yield save(nodes, "palette-median-cut")

    for position in ("top", "bottom"):
        name = "caption-" + position
        nodes = [gradient(schemas, name + "-source")]
        append_image(
            nodes,
            name + "-apply",
            "chainner:image:caption",
            {1: "C port", 2: 20, 3: position},
        )
        yield save(nodes, name)

    nodes = [
        make_node(
            schemas,
            "text-color-create",
            "chainner:image:text_as_image",
            {
                0: "C port\nText",
                1: 1,
                2: 1,
                3: json.dumps({"kind": "rgb", "values": [0.19, 0.57, 0.83]}),
                4: "right",
                5: 96,
                6: 48,
                7: "centered",
            },
        )
    ]
    yield save(nodes, "text-colored-rgba")

    for interpolation, suffix in ((0, "nearest"), (2, "linear-alpha")):
        name = "resize-rgba-" + suffix
        nodes = [gradient(schemas, name + "-source", "Diagonal")]
        append_image(
            nodes,
            name + "-apply",
            "chainner:image:resize",
            {1: 1, 3: 47, 4: 17, 5: interpolation, 6: 0},
        )
        yield save(nodes, name)

    nodes = [gradient(schemas, "resize-side-source", "Radial")]
    append_image(
        nodes,
        "resize-side-apply",
        "chainner:image:resize_to_side",
        {1: 17, 2: "shorter side", 3: 0, 4: "both"},
    )
    yield save(nodes, "resize-side-nearest")

    for selection, suffix in ((1, "all"), (2, "center"), (3, "largest")):
        name = "crop-border-" + suffix
        nodes = [checkerboard(name + "-source")]
        append_image(
            nodes,
            name + "-pad",
            "chainner:image:pad",
            {1: 0, 3: 0, 4: 5},
        )
        append_image(
            nodes,
            name + "-apply",
            "chainner:image:remove_border",
            {1: 5, 2: selection, 3: 1},
        )
        yield save(nodes, name)

    for radius in (1, 3):
        name = "median-blur-" + str(radius)
        nodes = [
            gray_gradient(name + "-source")
            if radius == 1
            else checkerboard(name + "-source")
        ]
        append_image(
            nodes,
            name + "-noise",
            "chainner:image:add_noise",
            {
                1: "salt_and_pepper",
                2: "gray" if radius == 1 else "rgb",
                3: 19,
                4: 1729,
            },
        )
        append_image(
            nodes,
            name + "-apply",
            "chainner:image:median_blur",
            {1: radius},
        )
        yield save(nodes, name)

    for mode, diffusion, suffix in (
        ("None", "FS", "quantize"),
        ("Ordered", "FS", "ordered"),
        ("Diffusion", "FS", "floyd-steinberg"),
        ("Diffusion", "JJN", "jarvis"),
    ):
        name = "dither-uniform-" + suffix
        nodes = [gradient(schemas, name + "-source", "Diagonal")]
        append_image(
            nodes,
            name + "-apply",
            "chainner:image:dither",
            {1: 5, 2: mode, 3: "B8", 4: diffusion},
        )
        yield save(nodes, name)

    for mode, suffix in (("None", "quantize"), ("Diffusion", "floyd-steinberg")):
        name = "dither-palette-" + suffix
        nodes = [
            gradient(schemas, name + "-source", "Diagonal", rgba=False),
            checkerboard(name + "-colors"),
        ]
        append_image(
            nodes,
            name + "-palette",
            "chainner:image:palette_from_image",
            {1: "all"},
        )
        nodes.append(
            make_node(
                schemas,
                name + "-apply",
                "chainner:image:palette_dither",
                {
                    0: edge(name + "-source"),
                    1: edge(name + "-palette"),
                    2: mode,
                    3: "FS",
                },
            )
        )
        yield save(nodes, name)

    nodes = [
        make_node(
            schemas,
            "fourth-combined-text",
            "chainner:image:text_as_image",
            {
                0: "C port\nCPU pipeline",
                1: 1,
                2: 0,
                3: json.dumps({"kind": "rgb", "values": [0.19, 0.57, 0.83]}),
                4: "center",
                5: 128,
                6: 64,
                7: "centered",
            },
        )
    ]
    append_image(
        nodes,
        "fourth-combined-resize",
        "chainner:image:resize",
        {1: 1, 3: 151, 4: 73, 5: 0, 6: 0},
    )
    append_image(
        nodes,
        "fourth-combined-median",
        "chainner:image:median_blur",
        {1: 1},
    )
    append_image(
        nodes,
        "fourth-combined-dither",
        "chainner:image:dither",
        {1: 5, 2: "Ordered", 3: "B8"},
    )
    append_image(
        nodes,
        "fourth-combined-caption",
        "chainner:image:caption",
        {1: "C CPU workflow", 2: 20, 3: "bottom"},
    )
    yield save(nodes, "fourth-combined")

    # Fifth pass: numerical CPU algorithms, including a Torch-hosted image
    # operation without a learned model. These fixtures never run inference.
    for rgba, invert in ((False, False), (True, True)):
        name = "gamma-" + ("rgba-inverse" if rgba else "rgb")
        nodes = [gradient(schemas, name + "-source", "Diagonal", rgba=rgba)]
        append_image(
            nodes, name + "-apply", "chainner:image:gamma", {1: 2.2, 2: int(invert)}
        )
        yield save(nodes, name)

    for method, suffix in ((2, "color"), (1, "texture")):
        name = "fill-alpha-" + suffix
        nodes = [gradient(schemas, name + "-source", "Diagonal")]
        append_image(nodes, name + "-pad", "chainner:image:pad", {1: 5, 3: 0, 4: 7})
        append_image(nodes, name + "-fill", "chainner:image:fill_alpha", {1: method})
        yield save(nodes, name)

    for rx, ry, suffix in (
        (1, 1, "small"),
        (4, 2, "asymmetric"),
        (203.6, 201.5, "large"),
    ):
        name = "box-blur-" + suffix
        nodes = [checkerboard(name + "-source")]
        append_image(nodes, name + "-apply", "chainner:image:blur", {1: rx, 2: ry})
        yield save(nodes, name)

    for kernel, padding, suffix in (
        ("0 -1 0\n-1 5 -1\n0 -1 0", 0, "sharpen"),
        ("0.0625 0.125 0.0625\n0.125 0.25 0.125\n0.0625 0.125 0.0625", 3, "padded"),
    ):
        name = "convolve-" + suffix
        nodes = [gradient(schemas, name + "-source", "Diagonal")]
        append_image(
            nodes,
            name + "-apply",
            "chainner:image:image_convolve",
            {1: kernel, 2: padding},
        )
        yield save(nodes, name)

    for algorithm in ("linear_histogram", "principal_color"):
        name = "color-transfer-" + algorithm
        nodes = [gradient(schemas, name + "-source", "Diagonal", rgba=False)]
        append_image(
            nodes,
            name + "-noise",
            "chainner:image:add_noise",
            {1: "uniform", 2: "rgb", 3: 13, 4: 1739},
        )
        nodes.append(checkerboard(name + "-reference"))
        append_image(
            nodes,
            name + "-ref-noise",
            "chainner:image:add_noise",
            {1: "uniform", 2: "rgb", 3: 19, 4: 519},
        )
        nodes.append(
            make_node(
                schemas,
                name + "-apply",
                "chainner:image:color_transfer",
                {0: edge(name + "-noise"), 1: edge(name + "-ref-noise"), 5: algorithm},
            )
        )
        yield save(nodes, name)

    for levels in (2, 5):
        name = "wavelet-cpu-" + str(levels)
        nodes = [
            gradient(schemas, name + "-source", "Diagonal", rgba=False),
            checkerboard(name + "-reference"),
        ]
        nodes.append(
            make_node(
                schemas,
                name + "-apply",
                "chainner:pytorch:wavelet_color_fix",
                {0: edge(name + "-source"), 1: edge(name + "-reference"), 2: levels},
            )
        )
        yield save(nodes, name)

    for mode, keep, suffix in ((0, 1, "auto"), (1, 0, "percentile-channels")):
        name = "stretch-native-" + suffix
        nodes = [gradient(schemas, name + "-source", "Diagonal")]
        append_image(
            nodes,
            name + "-apply",
            "chainner:image:stretch_contrast",
            {1: mode, 2: keep, 3: 15.25},
        )
        yield save(nodes, name)

    nodes = [gradient(schemas, "fifth-combined-source", "Diagonal", rgba=False)]
    append_image(nodes, "fifth-combined-box", "chainner:image:blur", {1: 3, 2: 2})
    append_image(nodes, "fifth-combined-gamma", "chainner:image:gamma", {1: 1.8, 2: 0})
    append_image(
        nodes,
        "fifth-combined-convolve",
        "chainner:image:image_convolve",
        {1: "0 -0.25 0\n-0.25 2 -0.25\n0 -0.25 0", 2: 2},
    )
    append_image(
        nodes,
        "fifth-combined-caption",
        "chainner:image:caption",
        {1: "C filters", 2: 20, 3: "bottom"},
    )
    yield save(nodes, "fifth-combined")

    # Stage 2: every new node is consumed by the real image output sink.
    for method in (
        "adv_mame2x",
        "adv_mame3x",
        "adv_mame4x",
        "eagle2x",
        "eagle3x",
        "super_eagle2x",
        "sai2x",
        "super_sai2x",
        "hqx2x",
        "hqx3x",
        "hqx4x",
    ):
        name = "pixel-art-" + method
        nodes = [checkerboard(name + "-source")]
        append_image(
            nodes, name + "-resize", "chainner:image:resize_pixel_art", {1: method}
        )
        yield save(nodes, name)
    for number, radii in enumerate(((1, 1), (0, 3.1), (21, 33))):
        name = "gaussian-complete-" + str(number)
        nodes = [checkerboard(name + "-source")]
        append_image(
            nodes,
            name + "-blur",
            "chainner:image:gaussian_blur",
            {1: radii[0], 2: radii[1]},
        )
        yield save(nodes, name)
    for rgba in (False, True):
        name = "surface-" + ("rgba" if rgba else "rgb")
        nodes = [gradient(schemas, name + "-source", "Radial", rgba=rgba)]
        append_image(
            nodes, name + "-blur", "chainner:image:bilateral_blur", {1: 4, 2: 25, 3: 25}
        )
        yield save(nodes, name)
    for channels in (1, 3, 4):
        name = "denoise-" + str(channels)
        nodes = [
            gray_gradient(name + "-source")
            if channels == 1
            else gradient(schemas, name + "-source", "Conic", rgba=channels == 4)
        ]
        append_image(
            nodes,
            name + "-filter",
            "chainner:image:fast_nlmeans",
            {1: 3.0, 2: 3.0, 3: 3, 4: 10},
        )
        yield save(nodes, name)
    for lower, upper in ((100, 300), (0, 2**31)):
        name = "canny-" + str(upper)
        nodes = [checkerboard(name + "-source")]
        append_image(
            nodes,
            name + "-edges",
            "chainner:image:canny_edge_detection",
            {1: lower, 2: upper},
        )
        yield save(nodes, name)
    for method in (0, 1):
        name = "inpaint-" + str(method)
        nodes = [checkerboard(name + "-source"), gray_gradient(name + "-mask")]
        append_image(
            nodes, name + "-mask-threshold", "chainner:image:threshold", {1: 85}
        )
        nodes.append(
            make_node(
                schemas,
                name + "-repair",
                "chainner:image:inpaint",
                {0: edge(name + "-source"), 1: edge(nodes[-1]["id"]), 2: method, 3: 3},
            )
        )
        yield save(nodes, name)


def check_success(response: dict):
    if not (response.get("success") is True or response.get("type") == "success"):
        raise RuntimeError(f"Backend execution did not succeed: {response}")


def run_backend(label: str, backend: Path, python: Path, root: Path, timeout: float):
    directory = root / label
    directory.mkdir()
    for name in ("storage", "output", "profile", "appdata", "localappdata", "temp"):
        (directory / name).mkdir()
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
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
    log = (directory / "backend.log").open("wb")
    job = OwnedJob()
    process = None
    events = None
    result: dict[str, Any] = {
        "label": label,
        "command": command,
        "cpu_only": True,
        "models_executed": False,
    }
    try:
        print(f"Starting isolated {label} HTTP backend", flush=True)
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
        last_error = None
        while time.monotonic() < limit:
            if process.poll() is not None:
                raise RuntimeError(
                    f"{label} exited with {process.returncode}; inspect {directory / 'backend.log'}"
                )
            try:
                status = request(port, "/status", timeout=3)
                if status.get("ready"):
                    break
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
                last_error = str(error)
            time.sleep(0.5)
        else:
            raise RuntimeError(
                f"{label} did not become ready: {last_error}; inspect backend.log"
            )
        metadata = {}
        for endpoint in METADATA_ENDPOINTS:
            metadata[endpoint] = request(port, "/" + endpoint, timeout=30)
            write_json(directory / (endpoint + ".json"), metadata[endpoint])
        schemas = {x["schemaId"]: x for x in metadata["nodes"]["nodes"]}
        result.update(metadata_record(metadata))
        result["node_count"] = len(schemas)
        events = Events(port)
        if not events.ready.wait(timeout=10) or events.error:
            raise RuntimeError(f"Cannot connect to SSE: {events.error}")
        time.sleep(0.1)
        fixtures = []
        for name, nodes in fixture_graphs(schemas, directory / "output"):
            print(f"{label}: CPU chain {name}", flush=True)
            payload = {
                "data": nodes,
                "options": {"chaiNNer_pytorch": {"use_cpu": True, "use_fp16": False}},
                "sendBroadcastData": True,
            }
            write_json(directory / (name + "-request.json"), payload)
            index = len(events.events)
            response = request(port, "/run", payload, timeout=60)
            check_success(response)
            received = events.wait_for(
                {x["id"] for x in nodes},
                index,
                expected_broadcasts={
                    x["id"] for x in nodes if schemas[x["schemaId"]]["outputs"]
                },
            )
            path = directory / "output" / (name + ".png")
            if not path.is_file():
                raise RuntimeError(f"Expected fixture image missing: {path}")
            fixtures.append(
                {
                    "name": name,
                    "response": response,
                    "output": str(path),
                    **sse_record(received, directory),
                    "generator_node_ids": generator_node_ids(nodes, schemas),
                    "schema_ids": sorted({node["schemaId"] for node in nodes}),
                }
            )
        node = gradient(schemas, "individual-gradient")
        payload = {
            "id": node["id"],
            "schemaId": node["schemaId"],
            "inputs": [x["value"] for x in node["inputs"]],
            "options": {},
        }
        index = len(events.events)
        response = request(port, "/run/individual", payload, timeout=30)
        check_success(response)
        received = events.wait_for(
            {node["id"]}, index, expected_broadcasts={node["id"]}
        )
        result["individual"] = {"response": response, "events": received}
        # Distinct node IDs exercise overlapping requests without intentionally
        # cancelling one another through the backend's per-ID preview cache.
        concurrent = []
        for number, style in enumerate(
            ("Horizontal", "Vertical", "Diagonal", "Radial")
        ):
            item = gradient(schemas, f"concurrent-gradient-{number}", style)
            concurrent.append(
                {
                    "id": item["id"],
                    "schemaId": item["schemaId"],
                    "inputs": [x["value"] for x in item["inputs"]],
                    "options": {},
                }
            )
        index = len(events.events)
        with ThreadPoolExecutor(max_workers=len(concurrent)) as pool:
            responses = list(
                pool.map(
                    lambda item: request(port, "/run/individual", item, timeout=30),
                    concurrent,
                )
            )
        for response in responses:
            check_success(response)
        concurrent_events = events.wait_for(
            {item["id"] for item in concurrent},
            index,
            expected_broadcasts={item["id"] for item in concurrent},
        )

        def broadcasts(items: list[dict], node_id: str):
            return [
                x["data"]
                for x in items
                if x["event"] == "node-broadcast" and x["data"].get("nodeId") == node_id
            ]

        index = len(events.events)
        repeated = request(port, "/run/individual", concurrent[0], timeout=30)
        check_success(repeated)
        repeated_events = events.wait_for(
            {concurrent[0]["id"]}, index, expected_broadcasts={concurrent[0]["id"]}
        )
        first = broadcasts(concurrent_events, concurrent[0]["id"])
        again = broadcasts(repeated_events, concurrent[0]["id"])
        if not first or first != again:
            raise RuntimeError(
                "Repeated individual request changed its broadcast output contract"
            )
        result["individual_concurrency"] = {
            "overlapping_requests": len(concurrent),
            "all_successful": True,
            "repeated_request_broadcast_identical": True,
            "note": "Correctness only; no throughput or latency timing",
        }
        result["fixtures"] = fixtures
        result["fixture_schema_ids"] = sorted(
            {schema for fixture in fixtures for schema in fixture["schema_ids"]}
        )
        write_json(directory / "events.json", events.events)
        result["success"] = True
    except Exception as error:
        result["success"] = False
        result["error"] = str(error)
        raise
    finally:
        if events:
            events.close()
        if process is not None:
            if process.poll() is None:
                try:
                    request(port, "/shutdown", {}, timeout=15)
                except (
                    OSError,
                    urllib.error.URLError,
                    json.JSONDecodeError,
                    http.client.HTTPException,
                ):
                    pass  # Shutdown may close the connection before its response arrives.
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
            job.close()  # Terminates only remaining members of this owned job.
            process.wait(timeout=15)
            result["exit_code"] = process.returncode
        else:
            job.close()
        log.close()
        log_lines = (
            (directory / "backend.log")
            .read_text(encoding="utf-8", errors="replace")
            .splitlines()
        )
        diagnostic_lines = [
            line
            for line in log_lines
            if any(
                pattern in line.lower()
                for pattern in (
                    "access violation",
                    "fatal exception",
                    "failed to import",
                    "unable to import",
                    "error importing",
                    "error installing",
                    "traceback (most recent call last)",
                )
            )
        ]
        result["startup_and_runtime_diagnostics"] = {
            "matching_line_count": len(diagnostic_lines),
            "first_matching_lines": diagnostic_lines[:40],
            "full_log": str(directory / "backend.log"),
        }
        write_json(directory / "result.json", result)
    return result


def compare_images(python: Path, pairs: list[dict]):
    # Execute image decoding in the independent packaged Python environment.
    # differing_components counts channel values; differing_pixels counts the
    # positions where any channel differs (the same for a single-channel image).
    code = """import json,sys,cv2,numpy as np
result=[]
for pair in json.load(sys.stdin):
    a=cv2.imdecode(np.fromfile(pair['baseline'],dtype=np.uint8),cv2.IMREAD_UNCHANGED)
    b=cv2.imdecode(np.fromfile(pair['converted'],dtype=np.uint8),cv2.IMREAD_UNCHANGED)
    if a is None or b is None: raise ValueError('Unable to decode '+pair['name'])
    same_shape=a.shape==b.shape and a.dtype==b.dtype
    delta=np.abs(a.astype(np.int32)-b.astype(np.int32)) if same_shape else None
    result.append({'name':pair['name'],'shape':list(a.shape),'dtype':str(a.dtype),
       'same_shape_dtype':same_shape,'exact_pixels':bool(same_shape and np.array_equal(a,b)),
       'max_channel_difference':int(delta.max()) if same_shape else None,
       'differing_components':int(np.count_nonzero(delta)) if same_shape else None,
       'differing_pixels':int(np.count_nonzero(delta.reshape(a.shape[0],a.shape[1],-1).any(axis=2))) if same_shape else None,
       'pass':bool(same_shape and np.array_equal(a,b))})
print(json.dumps(result))
"""
    # -I ignores PYTHONDONTWRITEBYTECODE, so -B keeps the interpreter unwritten.
    result = subprocess.run(
        [str(python), "-I", "-B", "-c", code],
        input=json.dumps(pairs),
        capture_output=True,
        text=True,
        timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Image comparison failed: {result.stderr}")
    return json.loads(result.stdout)


def generator_node_ids(nodes: list[dict], schemas: dict) -> list[str]:
    """The sorted IDs of a graph's generator nodes, as sse_event_mismatches reads them."""
    return sorted(
        node["id"] for node in nodes if schemas[node["schemaId"]]["kind"] == "generator"
    )


def canonical(value: Any, directory: Path):
    if isinstance(value, dict):
        return {
            key: canonical(item, directory)
            for key, item in value.items()
            if key
            not in {"executionTime", "eta", "duration", "timestamp", "exceptionTrace"}
        }
    if isinstance(value, list):
        return [canonical(item, directory) for item in value]
    if isinstance(value, str):
        return value.replace(str(directory), "<run>").replace(
            directory.as_posix(), "<run>"
        )
    return value


def event_contract(events: list[dict], directory: Path):
    # Parallel graph scheduling can reorder independent events. Preserve event
    # multiplicities and output values, never compare elapsed time/ETA.
    relevant = [
        canonical(event, directory)
        for event in events
        if event["event"]
        in {
            "node-start",
            "node-finish",
            "node-broadcast",
            "chain-start",
            "execution-error",
        }
    ]
    return sorted(relevant, key=lambda item: json.dumps(item, sort_keys=True))


MAP_FIELDS = ("data", "types", "sequenceTypes")


def sse_contract(events: list[dict], directory: Path) -> dict:
    """Normalized control events plus the final state the UI shows for each node."""
    states: dict[tuple[str, str], dict] = {}
    controls = []
    for original in events:
        normalized = canonical(original, directory)
        if not isinstance(normalized, dict):
            raise TypeError(f"Malformed SSE event: {original!r}")
        event: dict[str, Any] = normalized
        kind = event["event"]
        if kind in {"chain-start", "execution-error"}:
            if kind == "chain-start":
                event["data"]["nodes"] = sorted(set(event["data"]["nodes"]))
            controls.append(event)
        elif kind in {"node-start", "node-finish", "node-progress", "node-broadcast"}:
            key = kind, event["data"]["nodeId"]
            if kind == "node-broadcast":
                previous = states.get(key, {}).get("data", {})
                for field in MAP_FIELDS:
                    event["data"][field] = {
                        **(previous.get(field) or {}),
                        **(event["data"].get(field) or {}),
                    }
            states[key] = event
    return {"controls": controls, "final_state": [states[k] for k in sorted(states)]}


def final_broadcasts(events: list[dict], directory: Path) -> dict[str, dict]:
    """Each node's final broadcast state, merged by sse_contract (newest wins per output id)."""
    return {
        event["data"]["nodeId"]: event["data"]
        for event in sse_contract(events, directory)["final_state"]
        if event["event"] == "node-broadcast"
    }


def sse_record(events: list[dict], directory: Path) -> dict:
    """What sse_event_mismatches compares of one request's SSE events.

    events are in the order received; the final state depends on it, the rest not.
    """
    return {
        "event_kinds": sorted({event["event"] for event in events}),
        "events": event_contract(events, directory),
        "final_state": final_broadcasts(events, directory),
    }


# What sse_event_mismatches compares, as each verifier that uses it records it.
SSE_EVENT_COMPARISON = (
    "SSE events per node: the port's broadcast payloads are a sub-multiset of the "
    "oracle backend's (previews coalesce; no order is compared), the final state "
    "merged newest-wins per output id is equal, the port starts each generator node "
    "once (upstream re-runs it per item), and every other event is exact in "
    "multiplicity and payload. An iterated item's broadcast field (a node's output "
    "type or value sent with more than one value: iterated_fields) has no defined "
    "last value, since upstream on 3.14 sends iterated items' broadcasts in "
    "timing-dependent order (CPython's _chain_future; Consult D-35): it is compared "
    "by its broadcast multiset, and the port's final value must be one the oracle "
    "broadcast. A request runs twice on each backend: the oracle's two attempts are "
    "equal in full but for such final values, the port's also but for how many "
    "previews it coalesced (repeat_mismatches), each attempt's final value of such "
    "a field being one the other attempt broadcast; both port attempts are compared "
    "with the oracle's first by this rule. Each fixture names the rule its attempts "
    "met: repeat_rule (framework) or repeat_semantic_rule (video). Each attempt's "
    "per-node broadcast multiset is recorded (broadcast_multisets), and a fixture "
    "whose two attempts ended on different final values of such a field records "
    "them (repeat_final_values_differ); the report flags the oracle's."
)
# What repeat_mismatches compares on a side, as the fixtures record it.
REPEAT_RULES = {
    False: "exact but iterated items' final values (broadcast multisets)",
    True: "exact but broadcast multiplicities and iterated items' final values",
}


def broadcast_field_values(attempt: dict) -> dict[str, dict[str, Counter[str]]]:
    """Per node, each broadcast field ("<map field>/<output id>", e.g. "types/0")
    and how many times each value (as JSON) was sent, in one attempt (an
    sse_record)."""
    found: defaultdict[str, defaultdict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for event in attempt["events"]:
        if event["event"] != "node-broadcast":
            continue
        node = found[event["data"]["nodeId"]]
        for field in MAP_FIELDS:
            for output, value in (event["data"].get(field) or {}).items():
                node[f"{field}/{output}"][json.dumps(value, sort_keys=True)] += 1
    return {node: dict(fields) for node, fields in found.items()}


def iterated_fields(*attempts: dict) -> dict[str, set[str]]:
    """Per node, the broadcast fields some attempt sent with more than one value:
    an iterated item's type or value, whose last value is timing-dependent
    upstream on 3.14 (Consult D-35)."""
    found: defaultdict[str, set[str]] = defaultdict(set)
    for attempt in attempts:
        for node, fields in broadcast_field_values(attempt).items():
            found[node] |= {
                field for field, values in fields.items() if len(values) > 1
            }
    return {node: fields for node, fields in found.items() if fields}


def final_field(final_state: dict, node: str, field: str) -> str | None:
    """A node's final value of a broadcast field, as JSON; None when absent."""
    kind, _, output = field.partition("/")
    values = (final_state.get(node) or {}).get(kind) or {}
    return json.dumps(values[output], sort_keys=True) if output in values else None


def without_fields(final_state: dict, fields: dict[str, set[str]]) -> dict:
    """final_state less the given broadcast fields of each node."""
    kept = {}
    for node, state in final_state.items():
        kept[node] = {
            kind: {
                output: value
                for output, value in values.items()
                if f"{kind}/{output}" not in fields.get(node, set())
            }
            if kind in MAP_FIELDS
            else values
            for kind, values in state.items()
        }
    return kept


def repeat_final_value_differences(first: dict, second: dict) -> dict[str, dict]:
    """Per node and iterated field, the two attempts' final values where they
    differ: upstream's own timing-dependent order, which the report flags."""
    differing: defaultdict[str, dict[str, list[str | None]]] = defaultdict(dict)
    for node, fields in sorted(iterated_fields(first, second).items()):
        for field in sorted(fields):
            values = [
                final_field(attempt["final_state"], node, field)
                for attempt in (first, second)
            ]
            if values[0] != values[1]:
                differing[node][field] = values
    return dict(differing)


def oracle_repeat_flags(run: dict) -> dict[str, dict]:
    """The oracle run's fixtures whose two attempts ended on different final values
    of an iterated field (repeat_final_values_differ), by fixture name."""
    return {
        fixture["name"]: fixture["repeat_final_values_differ"]
        for fixture in run.get("fixtures", [])
        if fixture.get("repeat_final_values_differ")
    }


def sse_event_mismatches(baseline: dict, converted: dict) -> list[str]:
    """Differences between the SSE events of one fixture on the two backends.

    Each side holds "name", "generator_node_ids" and an sse_record():
    "event_kinds", "events" (the event_contract) and "final_state". For every
    node:
    - the port's node-broadcast payloads are a sub-multiset of the oracle
      backend's: upstream sends each broadcast as its own task and the
      port keeps one latest pending preview per node (LatestBroadcasts), so the
      port may send fewer, never another; no order is compared;
    - its final state, merged newest-wins per output id, is equal, but for its
      iterated fields (iterated_fields, on either side): their last value is
      timing-dependent upstream, so the port's final value must be one the
      oracle broadcast for that field;
    - a generator node is started exactly once by the port. Upstream re-runs a
      generator per item, so the baseline's count is free.
    Every other event, and the remaining keys, must be exactly equal.
    """
    mismatches = [
        f"{key} differ"
        for key in ("name", "event_kinds", "generator_node_ids")
        if baseline[key] != converted[key]
    ]
    generators = set(converted["generator_node_ids"])

    def split(events: list[dict]):
        broadcasts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        generator_starts: Counter[str] = Counter()
        exact: Counter[str] = Counter()
        for event in events:
            key = json.dumps(event, sort_keys=True)
            node_id = event["data"].get("nodeId")
            if event["event"] == "node-broadcast":
                broadcasts[node_id][key] += 1
            elif event["event"] == "node-start" and node_id in generators:
                generator_starts[node_id] += 1
            else:
                exact[key] += 1
        return broadcasts, generator_starts, exact

    base_broadcasts, _, base_exact = split(baseline["events"])
    conv_broadcasts, conv_starts, conv_exact = split(converted["events"])
    for node_id in sorted(generators):
        if conv_starts[node_id] != 1:
            mismatches.append(
                f"{node_id}: converted generator started {conv_starts[node_id]} "
                "times, expected once"
            )
    for node_id, payloads in sorted(conv_broadcasts.items()):
        for key, count in sorted(payloads.items()):
            if count > base_broadcasts[node_id][key]:
                mismatches.append(
                    f"{node_id}: broadcast {key} sent {count} times by converted, "
                    f"{base_broadcasts[node_id][key]} by baseline"
                )
    iterated = iterated_fields(baseline, converted)
    base_final = without_fields(baseline["final_state"], iterated)
    conv_final = without_fields(converted["final_state"], iterated)
    for node_id in sorted(base_final.keys() | conv_final.keys()):
        if base_final.get(node_id) != conv_final.get(node_id):
            mismatches.append(
                f"{node_id}: final state differs "
                f"({json.dumps(base_final.get(node_id), sort_keys=True)} baseline, "
                f"{json.dumps(conv_final.get(node_id), sort_keys=True)} converted)"
            )
    base_values = broadcast_field_values(baseline)
    for node_id, fields in sorted(iterated.items()):
        for field in sorted(fields):
            value = final_field(converted["final_state"], node_id, field)
            sent = base_values.get(node_id, {}).get(field, Counter())
            if value not in sent:
                mismatches.append(
                    f"{node_id}: final {field} {value} converted is not among the "
                    f"baseline's broadcasts {sorted(sent)}"
                )
    mismatches.extend(
        f"event {key}: {base_exact[key]} baseline, {conv_exact[key]} converted"
        for key in sorted(base_exact.keys() | conv_exact.keys())
        if base_exact[key] != conv_exact[key]
    )
    return mismatches


def repeat_mismatches(first: dict, second: dict, *, port: bool) -> list[str]:
    """The keys in which two attempts of one request on one backend differ.

    Each attempt holds an sse_record() and the request's other results (status,
    response, files, ...), all exact. The port's node-broadcast multiplicities are
    left out: it coalesces previews (LatestBroadcasts keeps one latest pending
    preview per node), so how many it sends is timing; the cross-side rule checks
    each port attempt's broadcasts against the oracle's (sse_event_mismatches).
    The oracle sends every broadcast, so its broadcast multisets are equal. On both
    sides an iterated field's final value (iterated_fields) is left out of the final
    state, since upstream's order of iterated items' broadcasts is timing-dependent
    on 3.14 (Consult D-35); instead each attempt's final value of it must be one the
    other attempt broadcast. repeat_final_value_differences records where the two
    final values differ.
    """
    iterated = iterated_fields(first, second)

    def compared(attempt: dict) -> dict:
        events = attempt["events"]
        if port:
            events = [x for x in events if x["event"] != "node-broadcast"]
        final_state = without_fields(attempt["final_state"], iterated)
        return {**attempt, "events": events, "final_state": final_state}

    a, b = compared(first), compared(second)
    mismatches = [
        f"{key} differ"
        for key in sorted(a.keys() | b.keys())
        if a.get(key) != b.get(key)
    ]
    for attempt, other, name in ((first, second, "first"), (second, first, "second")):
        sent = broadcast_field_values(other)
        for node, fields in sorted(iterated.items()):
            for field in sorted(fields):
                value = final_field(attempt["final_state"], node, field)
                if value not in sent.get(node, {}).get(field, Counter()):
                    mismatches.append(
                        f"{node}: the {name} attempt's final {field} {value} is not "
                        "among the other attempt's broadcasts"
                    )
    return mismatches


def broadcast_multisets(attempts: list[dict]) -> list[dict[str, dict[str, int]]]:
    """Per attempt (an sse_record each), every node's node-broadcast payloads
    (the event's data but nodeId, as JSON) and how many times each was sent: the
    multiplicities the port's repeat rule leaves out, for the report."""
    found = []
    for attempt in attempts:
        nodes: defaultdict[str, Counter[str]] = defaultdict(Counter)
        for event in attempt["events"]:
            if event["event"] == "node-broadcast":
                data = {k: v for k, v in event["data"].items() if k != "nodeId"}
                nodes[event["data"]["nodeId"]][json.dumps(data, sort_keys=True)] += 1
        found.append(
            {
                node: dict(sorted(payloads.items()))
                for node, payloads in sorted(nodes.items())
            }
        )
    return found


# The metadata endpoints every runtime verifier reads from both backends, and the
# two that carry the Dependency declarations; each run records the latter whole.
METADATA_ENDPOINTS = ("nodes", "packages", "features", "installed-dependencies")
DEPENDENCY_ENDPOINTS = ("packages", "installed-dependencies")
# The /packages dependency fields the declarations change; every other is exact.
DECLARED_FIELDS = frozenset({"version", "pypiName", "findLink"})
PYTORCH_CU128 = "https://download.pytorch.org/whl/cu128"
PYTORCH_CU132 = "https://download.pytorch.org/whl/cu132"
# Our backend declares the tested stack where the oracle declares upstream's (the
# python-stack spec, section 1: every Dependency pin is the tested version,
# native/python-stack.lock.txt). The port's versions are therefore gated (the
# lock's, installed on its runtime); the oracle's are upstream's declarations for
# another stack, recorded in each run's dependency_metadata and not gated. These
# rows are the only other ways the port's /packages and /installed-dependencies
# may differ from the oracle's, each with its ruling (Consult 8 R-g ratified the
# rule). The pypiName/findLink table is frozen at its three ruled rows (ncnn;
# torch and torchvision; onnxruntime-gpu): a new row is a consult. compare_metadata
# applies them; any other difference fails.
DECLARATION_CHANGES: dict[str, dict[str, Any]] = {
    # /packages: by the oracle's pypiName, (the port's pypiName, why).
    "pypiName": {
        "ncnn-vulkan": (
            "ncnn",
            (
                "Consult 2 Q4: ncnn-vulkan has no cp314 wheel; ncnn 1.0.20260526 "
                "replaces it (e12d9164, U1 Task 2 pins; 8cd686a4, its import)"
            ),
        ),
    },
    # /packages: by the port's pypiName, (the oracle's findLink, the port's, why).
    "findLink": {
        "torch": (
            PYTORCH_CU128,
            PYTORCH_CU132,
            "design ruling section 1: torch 2.14.1 on cu132 (e12d9164, U1 Task 2 pins)",
        ),
        "torchvision": (
            PYTORCH_CU128,
            PYTORCH_CU132,
            (
                "design ruling section 1: torchvision 0.29.1 on cu132 (e12d9164, U1 "
                "Task 2 pins)"
            ),
        ),
        "onnxruntime-gpu": (
            (
                "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/"
                "onnxruntime-cuda-12/pypi/simple/"
            ),
            None,
            (
                "PyPI onnxruntime-gpu 1.30.0: the lock's wheel is PyPI's; upstream's "
                "1.17.1 came from the CUDA 12 feed (e12d9164, U1 Task 2 pins)"
            ),
        ),
    },
    # /installed-dependencies: a name only the oracle lists, why.
    "oracle_only": {
        "google-re2": "Consult 7 (7324446e): chaiNNer-C's backend no longer "
        "declares google-re2; the provisioned runtime keeps it for the oracle",
    },
    # /installed-dependencies: a name only the port lists, why.
    "port_only": {
        "ncnn": "the ncnn-vulkan -> ncnn row: both runtimes hold ncnn, so the "
        "oracle's ncnn-vulkan is not installed",
    },
}
# What compare_metadata compares, as each verifier that uses it records it.
METADATA_COMPARISON = (
    "Metadata: /nodes and /features are equal by hash. Our backend pins the tested "
    "stack where the oracle declares upstream's, so /packages is equal but for each "
    "dependency's version (the port's is the lock's and installed on its runtime; the "
    "oracle's is recorded, not gated), pypiName and findLink (the oracle's, or a row "
    "of verify_runtime.DECLARATION_CHANGES, a frozen table), and "
    "/installed-dependencies agrees on every name both list, differs in names only "
    "by DECLARATION_CHANGES rows, and holds the lock's versions on the port's side."
)


def metadata_record(metadata: dict[str, Any]) -> dict[str, Any]:
    """A run's METADATA_ENDPOINTS as compare_metadata reads them: each one's
    hash, and the DEPENDENCY_ENDPOINTS themselves."""
    return {
        "metadata_sha256": {key: digest(value) for key, value in metadata.items()},
        "dependency_metadata": {key: metadata[key] for key in DEPENDENCY_ENDPOINTS},
    }


def declaration_mismatches(
    oracle: dict[str, Any], port: dict[str, Any], pins: dict[NormalizedName, str]
) -> tuple[dict[str, list[str]], list[tuple[str, str, str]]]:
    """The port's DEPENDENCY_ENDPOINTS against the oracle's: the differences per
    endpoint beyond DECLARATION_CHANGES and the lock, and the rows that applied,
    each (table, key, why).

    /packages: the same packages and dependencies in the same order, each field
    equal but the DECLARED_FIELDS. A dependency's version is the lock's pin (pins,
    by canonical name) and the port's installed version; its pypiName and findLink
    are the oracle's or their row's.
    /installed-dependencies: equal versions for every name both list; the names
    only one side lists are exactly the oracle_only and port_only rows; every port
    version is the lock's.
    """
    rows = DECLARATION_CHANGES
    oracle_listed = oracle["installed-dependencies"]
    port_listed = port["installed-dependencies"]
    packages: list[str] = []
    applied: list[tuple[str, str, str]] = []
    old, new = oracle["packages"], port["packages"]
    ids = [x["id"] for x in old], [x["id"] for x in new]
    if ids[0] != ids[1]:
        packages.append(f"package ids {ids[0]} oracle, {ids[1]} port")
    for a, b in zip(old, new, strict=True) if ids[0] == ids[1] else ():
        fields = sorted(
            key
            for key in a.keys() | b.keys()
            if key != "dependencies" and a.get(key) != b.get(key)
        )
        if fields:
            packages.append(f"{a['id']}: {fields} differ")
        if len(a["dependencies"]) != len(b["dependencies"]):
            packages.append(
                f"{a['id']}: {len(a['dependencies'])} dependencies oracle, "
                f"{len(b['dependencies'])} port"
            )
            continue
        for x, y in zip(a["dependencies"], b["dependencies"], strict=True):
            where = f"{a['id']} {x['pypiName']}"
            fields = sorted(
                key
                for key in x.keys() | y.keys()
                if key not in DECLARED_FIELDS and x.get(key) != y.get(key)
            )
            if fields:
                packages.append(f"{where}: {fields} differ")
            rename = rows["pypiName"].get(x["pypiName"])
            name = rename[0] if rename else x["pypiName"]
            if y["pypiName"] != name:
                packages.append(f"{where}: pypiName {y['pypiName']!r}, not {name!r}")
            elif rename:
                applied.append(("pypiName", x["pypiName"], rename[1]))
            link = rows["findLink"].get(y["pypiName"])
            expected = link[:2] if link else (x["findLink"], x["findLink"])
            if (x["findLink"], y["findLink"]) != expected:
                packages.append(
                    f"{where}: findLink {x['findLink']!r} oracle, {y['findLink']!r} "
                    f"port, not {expected[0]!r} -> {expected[1]!r}"
                )
            elif link:
                applied.append(("findLink", y["pypiName"], link[2]))
            pin = pins.get(canonicalize_name(y["pypiName"]))
            if y["version"] != pin:
                packages.append(f"{where}: pin {y['version']!r}, the lock's {pin!r}")
            if port_listed.get(y["pypiName"]) != y["version"]:
                packages.append(
                    f"{where}: installed {port_listed.get(y['pypiName'])!r} on the "
                    f"port's runtime, pin {y['version']!r}"
                )
    names: list[str] = []
    for side, only, row in (
        ("oracle", oracle_listed.keys() - port_listed.keys(), "oracle_only"),
        ("port", port_listed.keys() - oracle_listed.keys(), "port_only"),
    ):
        if only != rows[row].keys():
            names.append(f"{side}-only names {sorted(only)}, not {sorted(rows[row])}")
        else:
            applied.extend((row, name, rows[row][name]) for name in sorted(only))
    names.extend(
        f"{name}: {oracle_listed[name]!r} oracle, {port_listed[name]!r} port"
        for name in sorted(oracle_listed.keys() & port_listed.keys())
        if oracle_listed[name] != port_listed[name]
    )
    names.extend(
        f"{name}: {version!r} on the port, the lock's "
        f"{pins.get(canonicalize_name(name))!r}"
        for name, version in sorted(port_listed.items())
        if pins.get(canonicalize_name(name)) != version
    )
    return {"packages": packages, "installed-dependencies": names}, applied


def compare_metadata(baseline: dict[str, Any], converted: dict[str, Any]) -> dict:
    """The oracle's (baseline) and the port's (converted) metadata_record compared,
    the one rule every runtime verifier uses (METADATA_COMPARISON).

    /nodes and /features are equal by hash; the DEPENDENCY_ENDPOINTS follow
    declaration_mismatches against the project's lock, which oracle_python has
    shown is the lock the package recorded. Returns metadata_equal per endpoint,
    the differences of each failing endpoint, and the DECLARATION_CHANGES rows
    that applied.
    """
    lock = package_manifest.read_lock(PROJECT / package_manifest.LOCK)
    pins = {canonicalize_name(name): pin for name, pin in lock["pins"].items()}
    mismatches, applied = declaration_mismatches(
        baseline["dependency_metadata"], converted["dependency_metadata"], pins
    )
    for key in METADATA_ENDPOINTS:
        if key not in DEPENDENCY_ENDPOINTS and (
            baseline["metadata_sha256"][key] != converted["metadata_sha256"][key]
        ):
            mismatches[key] = ["hash differs"]
    return {
        "metadata_equal": {key: not mismatches.get(key) for key in METADATA_ENDPOINTS},
        "metadata_mismatches": {
            key: value for key, value in mismatches.items() if value
        },
        "declaration_changes_applied": applied,
    }


def oracle_tree(
    oracle: Path, chainner_ext: dict[str, str], owner: str
) -> tuple[Path, dict]:
    """The oracle's src and record, once shown to hold this chainner_ext.

    make_oracle.py's record must list the four CHAINNER_EXT files with exactly
    the hashes given, owner's for the same backend files, and the oracle tree
    must hold those bytes, so both sides run one binary set. Any difference is a
    stale oracle and is refused. The record is returned without its per-file
    upstream hashes.
    """
    path = oracle / ORACLE_RECORD
    if not path.is_file():
        raise ValueError(f"Oracle record missing: {path}; run make_oracle.py")
    record = json.loads(path.read_text(encoding="utf-8"))
    recorded = record.get("chainner_ext") or {}
    differing = sorted(
        name
        for name in chainner_ext.keys() | recorded.keys()
        if recorded.get(name) != chainner_ext.get(name)
    )
    if differing:
        raise ValueError(
            f"Stale oracle refused: the chainner_ext hashes in {path} differ from "
            f"{owner}'s for {differing}; rerun make_oracle.py on the build it ships"
        )
    changed = sorted(
        dest
        for name, dest in CHAINNER_EXT.items()
        if not (oracle / "src" / dest).is_file()
        or package_files.hash_file(oracle / "src" / dest) != chainner_ext[name]
    )
    if changed:
        raise ValueError(
            f"Stale oracle refused: its tree's chainner_ext differs from {path}: "
            f"{changed}; rerun make_oracle.py"
        )
    return oracle / "src", {
        key: value for key, value in record.items() if key != "source_files"
    }


def oracle_python(manifest: dict[str, Any]) -> Path:
    """The oracle's interpreter: the provisioned runtime the package was copied from.

    Consult 7: the oracle runs on the provisioned runtime, which keeps what the
    package leaves out (google-re2, Sanic-Cors and pynvml, which upstream's ONNX
    loader and server import or declare). It is python_stack.runtime while the
    project's lock is still the one that runtime was recorded with; anything
    else is refused.
    """
    stack = manifest.get("python_stack")
    if not stack:
        raise ValueError(
            "Oracle interpreter refused: the package records no python_stack, so no "
            "provisioned runtime it was copied from; build it with package_port.py"
        )
    runtime = Path(stack["runtime"])
    lock = Path(manifest["identity"]["project"]) / package_manifest.LOCK
    if not lock.is_file() or package_manifest.lock_sha256(lock) != stack["lock_sha256"]:
        raise ValueError(
            f"Oracle interpreter refused: {lock} is not the lock the package recorded "
            f"for {runtime}, so that is not the provisioned runtime the package was "
            "copied from; rebuild the package"
        )
    python = runtime / "python.exe"
    if not python.is_file():
        raise ValueError(
            f"Oracle interpreter refused: the provisioned runtime the package was "
            f"copied from is missing: {python}"
        )
    return python


def oracle_source(manifest: dict[str, Any], oracle: Path) -> tuple[Path, Path, dict]:
    """The oracle's src, interpreter and record, once shown to match the package.

    The oracle holds the package manifest's chainner_ext (oracle_tree) and runs
    on the provisioned runtime the package was copied from (oracle_python). Any
    difference is refused before anything is written or started. The record
    gains the interpreter.
    """
    package = {
        name: manifest["backend"].get(package_manifest.BACKEND + name, {}).get("sha256")
        for name in CHAINNER_EXT
    }
    missing = sorted(name for name, sha256 in package.items() if sha256 is None)
    if missing:
        raise ValueError(
            f"Package manifest does not record chaiNNer-C's chainner_ext: {missing}"
        )
    source, record = oracle_tree(oracle, package, "the package manifest")
    python = oracle_python(manifest)
    return source, python, {**record, "python": str(python)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=PROJECT / "out/chaiNNer-C")
    parser.add_argument("--startup-timeout", type=float, default=180)
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    manifest = json.loads(
        (package / "chainner-c-package.json").read_text(encoding="utf-8")
    )
    if manifest.get("state") != "complete" or manifest["identity"][
        "destination"
    ] != str(package):
        raise ValueError(
            "A completed package manifest matching this location is required"
        )
    if manifest["identity"]["project"] != str(PROJECT):
        raise ValueError("Package was generated by a different project")
    python = package / "python/python/python.exe"
    if not python.is_file() or not (package / "portable").is_file():
        raise ValueError(
            "Independent Python runtime or portable isolation marker is missing"
        )
    source, oracle_python, oracle = oracle_source(manifest, ORACLE)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = PROJECT / "native/reports" / ("runtime-" + stamp)
    root.mkdir(parents=True, exist_ok=False)
    baseline = root / "baseline-src"
    shutil.copytree(
        source, baseline, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    report: dict = {
        "utc": datetime.now(UTC).isoformat(),
        "package": str(package),
        "baseline": "Copy of the oracle tree (make_oracle.py: the installed backend "
        "source plus compat.diff, with chaiNNer-C's chainner_ext) on the provisioned "
        "runtime the package was copied from",
        "oracle": oracle,
        "constraints": {
            "cpu_only_fixtures": True,
            "gpu_model_operations": False,
            "cuda_visible_devices": "-1",
            "performance_benchmark": False,
            "python_hash_seed": "0 (same deterministic metadata set ordering in both processes)",
            "user_gpu_limit_gib": 2,
            "note": "No model or GPU execution requested; this is correctness verification, not VRAM measurement",
        },
        "scope": "HTTP metadata, CPU chain execution, SSE output contracts including scalar values, and decoded 8-bit output pixels; no GUI interaction",
        "fixture_limitations": [
            "Representative parameter choices; exhaustive kernel mode/shape checks are in the differential unit suites",
            "Model interpolation arithmetic is tested separately with synthetic CPU tensors and serialized NCNN/ONNX weights. These image chains do not execute model interpolation nodes or inference.",
            "Is Grayscale is absent from the installed 161-node registry and is covered only by source-level tests",
            "The converted backend retains exactness-required public IPP DFT, distance and selected resize primitives, native font shaping/rasterization, and BLAS/LAPACK; these are disclosed dependencies, not pure-C library rewrites.",
            "Alpha Matting and Chroma Key execute the full solvers and save the whole RGBA result for the baseline comparison. Converted sparse and foreground solvers are C. PyMatting 1.1.16's foreground starts from the mean foreground and background colours, so it is deterministic and compared like every other pixel.",
            SSE_EVENT_COMPARISON,
            METADATA_COMPARISON,
        ],
        "pixel_tolerance": "Exact decoded pixel equality is required; maximum component error is reported for diagnosis",
        "success": False,
    }
    try:
        original = run_backend(
            "baseline", baseline, oracle_python, root, args.startup_timeout
        )
        converted = run_backend(
            "converted", package / "resources/src", python, root, args.startup_timeout
        )
        report["runs"] = [original, converted]
        report.update(compare_metadata(original, converted))
        pairs = [
            {"name": a["name"], "baseline": a["output"], "converted": b["output"]}
            for a, b in zip(original["fixtures"], converted["fixtures"], strict=True)
        ]
        report["images"] = compare_images(python, pairs)
        report["fixture_schema_ids"] = original["fixture_schema_ids"]
        report["sse_fixture_event_mismatches"] = [
            f"{a['name']}: {mismatch}"
            for a, b in zip(original["fixtures"], converted["fixtures"], strict=True)
            for mismatch in sse_event_mismatches(a, b)
        ]
        report["sse_fixture_events_equal"] = not report["sse_fixture_event_mismatches"]
        report["success"] = (
            all(report["metadata_equal"].values())
            and all(x["pass"] for x in report["images"])
            and report["sse_fixture_events_equal"]
        )
    except Exception as error:
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        report["runs"] = [
            json.loads(path.read_text(encoding="utf-8"))
            for label in ("baseline", "converted")
            if (path := root / label / "result.json").is_file()
        ]
    finally:
        write_json(root / "report.json", report)
        print(
            json.dumps(
                {
                    "report": str(root / "report.json"),
                    "success": report["success"],
                    "error": report.get("error"),
                }
            ),
            flush=True,
        )
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
