"""The native run engine through the server, as the UI drives it: /run/individual of a node (its output cached in
ctx.cache), then /run of a chain that reads that cache, on both executors (CHAINNER_C_EXECUTOR), compared step by
step: the responses, the cached outputs, the events (verify_runtime.sse_event_mismatches, the A6 order) and the files
written. Each path by which a native node's outputs leave the engine to Python has a scenario: an individual run's
result (a side-effect node's too: Load Image), the cache handed to /run (a lazy reader: Save Image), broadcasts
(View Image's LargeImage previews), a native generator's preview and its items into a collector, a native utility
node's output, and the inputs reported with an error."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from engine_differential import order_violations
from verify_runtime import sse_event_mismatches, sse_record

import api
from chain.chain import GeneratorNode
from chain.json import JsonNode, parse_json
from chain.optimize import optimize

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"

DRIVER = """
import asyncio, hashlib, importlib, json, os, shutil, sys
from pathlib import Path
from types import SimpleNamespace
import cv2, numpy as np
sys.path[:0] = [os.environ["TOOLS"]]
sys.modules["__main__"].__file__ = os.path.abspath("run.py")
import api
importlib.import_module("packages.chaiNNer_standard")
api.registry.load_nodes(os.path.abspath("server.py"))
import server
from engine_differential import every_broadcast
from process import RegularOutput

# The handlers run without a started Sanic app: its loop is the running one.
type(server.app).loop = property(lambda self: asyncio.get_running_loop())
from verify_runtime import edge, make_node

schemas = {s: {"inputs": [i.to_dict() for i in d.inputs], "kind": d.kind} for s, (d, _) in api.registry.nodes.items()}
work = Path(os.environ["WORK"])
files, output = work / "files", work / "out"
files.mkdir(exist_ok=True)
rng = np.random.default_rng(3)
image = cv2.resize((rng.random((7, 9, 3)) * 255).astype(np.uint8), (72, 64), interpolation=cv2.INTER_CUBIC)
cv2.imwrite(str(files / "picture.png"), image)
sheet = cv2.resize((rng.random((5, 5, 4)) * 255).astype(np.uint8), (64, 64), interpolation=cv2.INTER_CUBIC)
cv2.imwrite(str(files / "sheet.png"), sheet)
picture, missing = str(files / "picture.png"), str(files / "missing.png")


def node(node_id, schema, values):
    return make_node(schemas, node_id, schema, values)


def save(node_id, source, name):
    return node(node_id, "chainner:image:save",
                {0: edge(source), 1: str(output), 2: None, 3: name, 4: "png", 15: "u8"})


load = node("load", "chainner:image:load", {0: picture})
SCENARIOS = {
    # A side-effect native node's preview (its cached RegularOutput), read by a native node in /run.
    "load_preview_then_run": [
        ("individual", load),
        ("run", [load, node("blur", "chainner:image:gaussian_blur", {0: edge("load"), 1: 2, 2: 2}),
                 save("save", "blur", "blurred")]),
    ],
    # The cached output into a lazy native reader.
    "load_preview_into_lazy_save": [("individual", load), ("run", [load, save("save", "load", "saved")])],
    # View Image (native, side effects): its LargeImage previews broadcast from a cached input.
    "view_image": [("individual", load), ("run", [load, node("view", "chainner:image:view", {0: edge("load")})])],
    # A native generator's preview, then its items through a native node into a Python collector.
    "range_into_collector": [
        ("individual", node("range", "chainner:utility:range", {0: 0, 1: 1, 2: 5, 3: 1})),
        ("run", [node("range", "chainner:utility:range", {0: 0, 1: 1, 2: 5, 3: 1}),
                 node("math", "chainner:utility:math", {1: "mul", 0: edge("range"), 2: 3}),
                 node("total", "chainner:utility:accumulate", {0: edge("math"), 1: "sum"})]),
    ],
    # A native utility node's preview, its cached number read by /run.
    "utility_preview_then_run": [
        ("individual", node("math", "chainner:utility:math", {1: "add", 0: 24, 2: 24})),
        ("run", [node("math", "chainner:utility:math", {1: "add", 0: 24, 2: 24}),
                 node("board", "chainner:image:create_checkerboard", {0: edge("math"), 1: 40, 4: 8}),
                 save("save", "board", "board")]),
    ],
    # A native generator's items (Split Spritesheet) into a Python collector (Merge Spritesheet).
    "spritesheet_into_collector": [
        ("individual", node("sheet", "chainner:image:load", {0: str(files / "sheet.png")})),
        ("run", [node("sheet", "chainner:image:load", {0: str(files / "sheet.png")}),
                 node("split", "chainner:image:split_spritesheet", {0: edge("sheet"), 1: 2, 2: 2}),
                 node("merge", "chainner:image:merge_spritesheet", {0: edge("split"), 1: 2, 2: 2}),
                 save("save", "merge", "merged")]),
    ],
    # Errors: an individual run's (a missing file), and a native node's in /run with the inputs it reports.
    "errors": [
        ("individual", node("gone", "chainner:image:load", {0: missing})),
        ("individual", load),
        ("run", [load, node("crop", "chainner:image:crop", {0: edge("load"), 1: 1, 3: 40, 5: 40}),
                 save("save", "crop", "cropped")]),
    ],
}


class Recorder:
    def __init__(self):
        self.events = []

    def put(self, event):
        self.events.append(json.loads(json.dumps(event)))


def digest(value):
    if isinstance(value, np.ndarray):
        return ["array", list(value.shape), str(value.dtype), hashlib.sha256(np.ascontiguousarray(value)).hexdigest()]
    return [type(value).__name__, str(value)]


def response(answer):
    body = json.loads(answer.body)
    for key in ("exceptionTrace",):
        (body.get("error") if isinstance(body.get("error"), dict) else body).pop(key, None)
    return [answer.status, body]


def files_written():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())} if output.exists() else {}


async def scenario(steps):
    ctx = server.app.ctx
    ctx.__dict__["storage_dir"] = work / "storage"
    ctx.cache.clear()
    records = []
    for kind, payload in steps:
        recorder = Recorder()
        ctx.__dict__["queue"] = recorder
        shutil.rmtree(output, ignore_errors=True)
        output.mkdir()
        request = SimpleNamespace(app=server.app)
        if kind == "individual":
            request.json = {"id": payload["id"], "schemaId": payload["schemaId"], "options": {},
                            "inputs": [i["value"] for i in payload["inputs"]]}
            answer = await server.run_individual(request)
            await asyncio.gather(*list(server.individual_releases))
        else:
            request.json = {"data": payload, "options": {}, "sendBroadcastData": True}
            answer = await server.run(request)
        cached = {str(k): [digest(v) for v in o.output] if isinstance(o, RegularOutput) else type(o).__name__
                  for k, o in sorted(ctx.cache.items())}
        records.append({"step": kind, "response": response(answer), "cache": cached, "events": recorder.events,
                        "files": files_written()})
    return records


results = {}
for name, steps in SCENARIOS.items():
    results[name] = {}
    for kind in ("python", "native"):
        os.environ["CHAINNER_C_EXECUTOR"] = kind
        # The oracle sends every broadcast (its LatestBroadcasts would drop some, by timing), the engine without its
        # loop preview rate limit.
        with every_broadcast():
            results[name][kind] = asyncio.run(scenario(steps))
print(json.dumps({"work": str(work), "scenarios": {n: [s for s in steps] for n, steps in SCENARIOS.items()},
                  "results": results}))
"""


@pytest.fixture(scope="module")
def server_runs(tmp_path_factory) -> dict[str, Any]:
    work = tmp_path_factory.mktemp("server")
    environment = dict(
        os.environ,
        WORK=str(work),
        TOOLS=str(BACKEND.parents[1] / "native" / "tools"),
        CUDA_VISIBLE_DEVICES="-1",
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", textwrap.dedent(DRIVER)],
        cwd=BACKEND,
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, (
        result.returncode,
        result.stdout[-3000:],
        result.stderr[-6000:],
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


SCENARIO_NAMES = [
    "load_preview_then_run",
    "load_preview_into_lazy_save",
    "view_image",
    "range_into_collector",
    "utility_preview_then_run",
    "spritesheet_into_collector",
    "errors",
]


@pytest.fixture(scope="module")
def registry_loaded():
    main = sys.modules["__main__"]
    previous = getattr(main, "__file__", None)
    main.__file__ = str(BACKEND / "run.py")
    importlib.import_module("packages.chaiNNer_standard")
    # As the worker logs them: a package whose optional dependency is missing (e.g. tensorrt, registered once
    # another test imported it) leaves its nodes out.
    errors = [
        e
        for e in api.registry.load_nodes(str(BACKEND / "server.py"))
        if not isinstance(e.error, ModuleNotFoundError)
    ]
    assert not errors, [e.module for e in errors]
    yield
    if previous is None:
        del main.__file__
    else:
        main.__file__ = previous


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_server_runs_match(server_runs, registry_loaded, name):
    work = Path(server_runs["work"])
    steps = server_runs["scenarios"][name]
    oracle, engine = (
        server_runs["results"][name]["python"],
        server_runs["results"][name]["native"],
    )
    assert len(oracle) == len(engine) == len(steps)
    for index, ((kind, payload), expected, found) in enumerate(
        zip(steps, oracle, engine, strict=True)
    ):
        where = f"{name} step {index} ({kind})"
        assert found["response"] == expected["response"], where
        assert found["cache"] == expected["cache"], where
        assert found["files"] == expected["files"], where
        nodes: list[JsonNode] = payload if kind == "run" else [payload]
        chain = parse_json(nodes)
        optimize(chain)
        generators = sorted(
            n.id for n in chain.nodes.values() if isinstance(n, GeneratorNode)
        )

        records = [
            {
                "name": where,
                "generator_node_ids": generators,
                **sse_record(events, work),
            }
            for events in (expected["events"], found["events"])
        ]
        assert sse_event_mismatches(*records) == [], where
        if kind == "run":
            assert order_violations(found["events"], chain) == [], where
    # The steps did what the UI expects: every request succeeded (the errors scenario's as listed), and a regular
    # node's preview left its output in the cache.
    outcomes = [
        (
            step["response"][0],
            step["response"][1].get("type", step["response"][1].get("success")),
        )
        for step in engine
    ]
    if name == "errors":
        assert outcomes == [(200, False), (200, True), (500, "error")], outcomes
        source = engine[2]["response"][1]["source"]
        assert source is not None and source["nodeId"] == "crop" and source["inputs"], (
            source
        )
    else:
        assert all(
            outcome in ((200, True), (200, "success")) for outcome in outcomes
        ), outcomes
    for (kind, payload), step in zip(steps, engine, strict=True):
        if (
            kind == "individual"
            and payload["nodeType"] == "regularNode"
            and step["response"][1].get("success")
        ):
            data, _ = api.registry.nodes[payload["schemaId"]]
            assert len(step["cache"][payload["id"]]) == len(data.outputs), (
                name,
                step["cache"],
            )
