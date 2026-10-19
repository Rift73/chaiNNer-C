from __future__ import annotations

import asyncio
import gc
import importlib
import itertools
import logging
import os
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import Final, TypedDict

import psutil
from sanic import Sanic
from sanic.log import access_logger, logger
from sanic.request import Request
from sanic.response import HTTPResponse, json

import api
from api import (
    ExecutionOptions,
    Group,
    JsonExecutionOptions,
    NodeId,
)
from chain.cache import OutputCache
from chain.chain import Chain, FunctionNode, GeneratorNode
from chain.json import JsonNode, parse_json
from chain.optimize import optimize
from dependencies.store import installed_packages
from events import EventConsumer, EventQueue, ExecutionErrorData, event_text
from native_executor import NativeExecutor, executor_kind
from nodes.impl import native, native_profile, numpy_pool
from nodes.impl.cors import add_cors
from nodes.impl.tensorrt import cache as tensorrt_cache
from process import ExecutionId, Executor, NodeExecutionError, NodeOutput
from process_temp import use_process_temp_folder
from progress_controller import Aborted
from response import (
    RESTART_WORKER_HEADER,
    already_running_response,
    error_response,
    no_executor_response,
    success_response,
)
from server_config import LOG_CONFIG, ServerConfig
from system import load_system_cpp_runtime

POOL_SIZE = 8  # executor pool workers, each Executor's pool_size; item window P = max(1, POOL_SIZE - 2)

# The packages in the order they register, which orders their categories.
PACKAGES: Final = (
    "chaiNNer_standard",
    "chaiNNer_pytorch",
    "chaiNNer_ncnn",
    "chaiNNer_onnx",
    "chaiNNer_tensorrt",
    "chaiNNer_modelopt",
    "chaiNNer_vapoursynth",
    "chaiNNer_external",
)
# The packages whose node modules load first, so their nodes run while the frameworks
# load: none of their modules imports a framework or another heavy library, except
# HEAVY_MODULES (test_light_nodes.py checks both).
LIGHT_PACKAGES: Final = ("chaiNNer_standard", "chaiNNer_external")
HEAVY_MODULES: Final = frozenset(
    {
        "packages.chaiNNer_standard.image.create_images.create_noise",  # SciPy
        "packages.chaiNNer_standard.image_channel.misc.alpha_matting",  # PyMatting
    }
)
# The order the other node modules load in, a package at a time: PyTorch, the most
# used framework (Load Model, Upscale Image), first; the standard package's last, as
# PyMatting compiles its kernels (about 25 s) on a first start.
HEAVY_ORDER: Final = (
    "chaiNNer_pytorch",
    "chaiNNer_ncnn",
    "chaiNNer_onnx",
    "chaiNNer_tensorrt",
    "chaiNNer_modelopt",
    "chaiNNer_vapoursynth",
    "chaiNNer_external",
    "chaiNNer_standard",
)


class RunTracker:
    """The runs the worker serves (/run, /run/individual), for NodeLoading's one
    gc.freeze, which is skipped while a run is in flight or once a run of a node outside
    the light set has answered (its model may sit in the output cache, and frozen
    objects are never collected). Updated by the request middleware on the loop thread;
    the lock keeps a run from starting between the check and the freeze."""

    ROUTES: Final = frozenset({"/run", "/run/individual"})

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.in_flight = 0
        self.ran_heavy_node = False

    def started(self) -> None:
        with self.lock:
            self.in_flight += 1

    def answered(self, heavy: bool) -> None:
        with self.lock:
            self.in_flight -= 1
            self.ran_heavy_node = self.ran_heavy_node or heavy

    def freeze_blocker(self) -> str | None:
        """Why the freeze must be skipped, or None. Call it holding the lock."""
        if self.in_flight > 0:
            return f"{self.in_flight} run(s) in flight"
        if self.ran_heavy_node:
            return "a run of a node outside the light set has completed"
        return None


class NodeLoading:
    """How far the worker's node loading is. It runs beside serving: the packages and
    their light node modules first, then the other node modules a package at a time.
    Set by the loading thread, read by requests."""

    def __init__(self) -> None:
        # The packages are registered (their dependencies are known).
        self.packages_registered = threading.Event()
        # The packages are registered and the light node modules loaded.
        self.light_loaded = threading.Event()
        # The schema ids of the light nodes, once they have loaded.
        self.light_schemata: frozenset[str] = frozenset()
        # By package id: every node module of the package has loaded or failed.
        self.package_loaded: dict[str, threading.Event] = {
            name: threading.Event() for name in PACKAGES
        }
        # By package id: the node modules that failed to load.
        self.errors: dict[str, list[api.LoadErrorInfo]] = {}
        # Every package has loaded, or the loading failed.
        self.done = threading.Event()

    def package_of(self, schema_id: str) -> api.Package | None:
        """The package a node most likely belongs to, by its schema id, which names one
        of the package's categories ("chainner:pytorch:upscale_image")."""
        if not self.light_loaded.is_set():
            return None
        parts = schema_id.split(":")
        if len(parts) != 3:
            return None
        for package in api.registry.packages.values():
            if any(category.id == parts[1] for category in package.categories):
                return package
        return None

    def load_error(self, schema_id: str) -> str | None:
        """Why the package of a node that is not registered failed to load, once it has;
        None while it loads, or if its modules loaded."""
        package = self.package_of(schema_id)
        if package is None or not self.package_loaded[package.id].is_set():
            return None
        errors = self.errors.get(package.id)
        if not errors:
            return None
        reasons = "; ".join(sorted({str(e.error) for e in errors})[:3])
        return f"{schema_id} is not available: the {package.name} nodes failed to load ({reasons})."

    def settled(self, schema_id: str) -> bool:
        """Whether a node that is not registered is known not to come anymore."""
        return self.done.is_set() or self.load_error(schema_id) is not None

    def complete(self) -> bool:
        """Whether every node module loaded, but those whose dependency is not
        installed. A module that failed otherwise may load at the next start (torch
        under memory pressure: WinError 1455), so the app does not cache this list."""
        return not any(
            not isinstance(error.error, ModuleNotFoundError)
            for errors in self.errors.values()
            for error in errors
        )

    def load(self, config: ServerConfig) -> None:
        """Loads every package's nodes, setting the events as it goes, failures included.
        Raises if config.error_on_failed_node and a node module failed, or if a node
        failed its checks."""
        try:
            load_errors = self._load()
            log_load_errors(load_errors)
            if len(load_errors) > 0 and config.error_on_failed_node:
                raise ValueError("Error importing nodes")
        finally:
            for loaded in self.package_loaded.values():
                loaded.set()
            self.packages_registered.set()
            self.light_loaded.set()
            self.done.set()

    def _load(self) -> list[api.LoadErrorInfo]:
        # Before any module that may load a C++ runtime (Pillow's).
        try:
            load_system_cpp_runtime()
        except OSError as error:
            logger.error(
                f"Could not load the system C++ runtime (msvcp140.dll): {error}."
                " PyTorch and ONNX Runtime may fail to load."
            )
        light, rest = register_packages()
        self.packages_registered.set()
        load_errors: list[api.LoadErrorInfo] = []

        def load(name: str, batch: list[api.NodeModule]) -> None:
            errors = api.registry.load_modules(batch)
            self.errors.setdefault(name, []).extend(errors)
            load_errors.extend(errors)

        start = time.perf_counter()
        logger.info("Loading Nodes...")
        for name, batch in light.items():
            load(name, batch)
        self.light_schemata = frozenset(api.registry.nodes)
        self.light_loaded.set()
        for name in PACKAGES:
            if not rest[name]:
                self.package_loaded[name].set()
        logger.info(
            f"Loaded {len(api.registry.nodes)} nodes in"
            f" {time.perf_counter() - start:.2f} s; loading the other nodes"
        )

        for name in HEAVY_ORDER:
            if rest[name]:
                try:
                    load(name, rest[name])
                finally:
                    self.package_loaded[name].set()
                logger.info(
                    f"Loaded the {name} nodes after {time.perf_counter() - start:.2f} s"
                )
        self.freeze_objects()
        return load_errors

    def freeze_objects(self) -> None:
        """Moves every object the start made (the modules, the node registry, the
        frameworks' state) out of the collector's reach, once: each later full
        collection (collect_failed_run, return_device_memory) then took about 0 ms
        instead of 136-164 ms. Gate ruling R1 (native/reports/fable-decisions.md,
        "instant-exec gate R1"): skipped, and never retried, while a run is in flight
        or once a run of a node outside the light set has completed (RunTracker), as
        their objects must stay collectable."""
        with run_tracker.lock:
            reason = run_tracker.freeze_blocker()
            if reason is None:
                gc.collect()
                gc.freeze()
        if reason is None:
            logger.info(f"gc frozen: {gc.get_freeze_count()} objects")
        else:
            logger.info(f"freeze skipped: {reason}")


def register_packages() -> tuple[
    dict[str, list[api.NodeModule]], dict[str, list[api.NodeModule]]
]:
    """Registers the packages and splits each one's node modules (by package id) into
    the light ones, which load first, and the others."""
    for name in PACKAGES:
        importlib.import_module(f"packages.{name}")
    packages = {p.id: p for p in api.registry.packages.values()}
    modules = {
        name: api.registry.node_modules(__file__, packages[name]) for name in PACKAGES
    }
    light = {
        name: [m for m in modules[name] if m.name not in HEAVY_MODULES]
        for name in LIGHT_PACKAGES
    }
    rest = {
        name: [m for m in modules[name] if m not in light.get(name, [])]
        for name in PACKAGES
    }
    return light, rest


class AppContext:
    def __init__(self):
        self.config: Final[ServerConfig] = ServerConfig.parse_argv()
        self.executor: Executor | NativeExecutor | None = None
        # A /run is in its handler: set before its first await (it waits for previews
        # before its executor exists), so a second /run is refused meanwhile.
        self.run_starting = False
        # Stop arrived while that /run waited for previews: it ends without running.
        self.run_stopped_early = False
        self.individual_executors: dict[ExecutionId, Executor | NativeExecutor] = {}
        self.cache: dict[NodeId, NodeOutput] = {}
        # Increased from 4 to 8 workers for better parallelism in batch processing
        self.pool: Final[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=POOL_SIZE, initializer=numpy_pool.install
        )

    @cached_property
    def queue(self) -> EventQueue:
        return EventQueue()

    @cached_property
    def storage_dir(self) -> Path:
        if self.config.storage_dir is not None:
            logger.info(f"Using given storage directory: {self.config.storage_dir}")
            return Path(self.config.storage_dir)

        default_sub_dir = "chaiNNer/backend-storage"

        # try using the chaiNNer's app dir
        try:
            # appdirs is likely only installed on dev machines
            from appdirs import user_data_dir

            app_data_dir = Path(user_data_dir(roaming=True)) / default_sub_dir
            logger.info(f"Using app data as storage directory: {app_data_dir}")
            return app_data_dir
        except:  # noqa: E722
            # ignore errors
            pass

        # last resort: use the system's temporary directory
        temp = Path(tempfile.gettempdir()) / default_sub_dir
        logger.info(f"No storage directory given. Using temporary directory: {temp}")
        return Path(temp)

    @staticmethod
    def get(app_instance: Sanic) -> AppContext:
        assert isinstance(app_instance.ctx, AppContext)
        return app_instance.ctx


app = Sanic("chaiNNer_executor", ctx=AppContext(), log_config=LOG_CONFIG)
app.config.REQUEST_TIMEOUT = sys.maxsize
app.config.RESPONSE_TIMEOUT = sys.maxsize
add_cors(app)


class SSEFilter(logging.Filter):
    def filter(self, record):  # noqa: ANN001
        request = record.request  # type: ignore
        return not (
            (request.endswith("/sse")) and record.status == 200  # type: ignore
        )


class ZeroCounter:
    def __init__(self) -> None:
        self.count = 0

    async def wait_zero(self) -> None:
        while self.count != 0:
            await asyncio.sleep(0.01)

    def __enter__(self):
        self.count += 1

    def __exit__(self, _exc_type: object, _exc_value: object, _exc_traceback: object):
        self.count -= 1


run_individual_counter = ZeroCounter()

run_tracker = RunTracker()
node_loading = NodeLoading()


def run_schema_ids(request: Request) -> list[str] | None:
    """The schema ids a /run or /run/individual request runs, None if unreadable."""
    try:
        body = request.json
        if request.path == "/run":
            return [node["schemaId"] for node in body["data"]]
        return [body["schemaId"]]
    except Exception:
        return None


@app.on_request
async def track_run_start(request: Request):
    if request.path in RunTracker.ROUTES:
        run_tracker.started()


@app.on_response
async def track_run_end(request: Request, _response: HTTPResponse):
    if request.path in RunTracker.ROUTES:
        schema_ids = run_schema_ids(request)
        run_tracker.answered(
            schema_ids is None
            or any(s not in node_loading.light_schemata for s in schema_ids)
        )


setup_task = None
close_task = None


async def nodes_available():
    """Waits until every node has loaded (or failed to): the node and package lists are
    complete then."""
    if setup_task is not None:
        await setup_task


async def wait_for_nodes(
    schema_ids: Iterable[str], stopped: Callable[[], bool] = lambda: False
) -> None:
    """Waits until the nodes of `schema_ids` are registered, or known not to come: a run
    issued while the worker starts waits for its nodes to load. A node whose package
    failed to load raises ValueError with the package's error; a node no package has is
    left to the caller. Returns early once `stopped()`."""
    while True:
        pending = [
            schema_id
            for schema_id in schema_ids
            if schema_id not in api.registry.nodes
            and not node_loading.settled(schema_id)
        ]
        if not pending or stopped():
            break
        await asyncio.sleep(0.01)
    if stopped():
        return
    for schema_id in schema_ids:
        if schema_id not in api.registry.nodes:
            error = node_loading.load_error(schema_id)
            if error is not None:
                raise ValueError(error)


access_logger.addFilter(SSEFilter())


@app.route("/health")
async def health(_request: Request):
    """Simple health check endpoint that doesn't require full setup"""
    return json({"status": "ok"})


@app.route("/nodes/registered")
async def registered_nodes(_request: Request):
    """The schema ids of the nodes registered so far, and whether every node has loaded
    (or failed to). The host reads it while the nodes load (announce_registered_nodes)."""
    # Done first: once done, the list read after it is complete.
    done = node_loading.done.is_set()
    return json({"schemaIds": list(api.registry.nodes), "done": done})


@app.route("/nodes")
async def nodes(_request: Request):
    """Gets a list of all nodes as well as the node information"""
    await nodes_available()
    logger.debug(api.registry.categories)

    node_list = []
    for node, sub in api.registry.nodes.values():
        node_dict = {
            "schemaId": node.schema_id,
            "name": node.name,
            "category": sub.category.id,
            "nodeGroup": sub.id,
            "inputs": [x.to_dict() for x in node.inputs],
            "outputs": [x.to_dict() for x in node.outputs],
            "groupLayout": [
                g.to_dict() if isinstance(g, Group) else g for g in node.group_layout
            ],
            "iteratorInputs": [x.to_dict() for x in node.iterable_inputs],
            "iteratorOutputs": [x.to_dict() for x in node.iterable_outputs],
            "keyInfo": node.key_info.to_dict() if node.key_info else None,
            "suggestions": [x.to_dict() for x in node.suggestions],
            "keywords": node.keywords,
            "description": node.description,
            "seeAlso": node.see_also,
            "icon": node.icon,
            "kind": node.kind,
            "hasSideEffects": node.side_effects,
            "deprecated": node.deprecated,
            "features": node.features,
        }
        node_list.append(node_dict)

    return json(
        {
            "nodes": node_list,
            "categories": [x.to_dict() for x in api.registry.categories],
            "categoriesMissingNodes": [],
            "complete": node_loading.complete(),
        }
    )


class RunRequest(TypedDict):
    data: list[JsonNode]
    options: JsonExecutionOptions
    sendBroadcastData: bool


@app.route("/run", methods=["POST"])
async def run(request: Request):
    """Runs the provided nodes"""
    ctx = AppContext.get(request.app)

    if ctx.executor or ctx.run_starting:
        message = "Cannot run another executor while the first one is still running."
        logger.warning(message)
        return json(already_running_response(message), status=500)
    # No await between the check and this: both /runs could otherwise pass it while
    # the first waits for previews.
    ctx.run_starting = True
    ctx.run_stopped_early = False

    executor_id = ExecutionId("main-executor " + uuid.uuid4().hex)

    tracer = None
    try:
        if ctx.config.trace:
            logger.info("Starting VizTracer...")
            from viztracer import VizTracer

            tracer = VizTracer()
            tracer.log_async = True
            tracer.start()

        full_data: RunRequest = dict(request.json)  # type: ignore
        # wait until the chain's nodes have loaded and all previews are done
        await wait_for_nodes(
            {node["schemaId"] for node in full_data["data"]},
            lambda: ctx.run_stopped_early,
        )
        await run_individual_counter.wait_zero()
        if ctx.run_stopped_early:
            logger.info("The run was stopped before it started.")
            return json(success_response(), status=200)

        logger.debug(full_data)
        chain = parse_json(full_data["data"])
        optimize(chain)

        # Remove all Generator values from the cache for each new run
        # Otherwise, their state will cause them to resume from where they left off
        schema_data = api.registry.nodes
        for node in chain.nodes.values():
            node_schema = schema_data.get(node.schema_id)
            if node_schema:
                node_data, _ = node_schema
                if node_data.kind == "generator" and ctx.cache.get(node.id):
                    ctx.cache.pop(node.id)

        logger.info("Running new executor...")
        # CHAINNER_C_EXECUTOR: the native run engine or process.Executor (native_executor.py).
        executor_class = NativeExecutor if executor_kind() == "native" else Executor
        executor = executor_class(
            id=executor_id,
            chain=chain,
            send_broadcast_data=full_data["sendBroadcastData"],
            options=ExecutionOptions.parse(full_data["options"]),
            loop=app.loop,
            queue=ctx.queue,
            pool=ctx.pool,
            storage_dir=ctx.storage_dir,
            parent_cache=OutputCache(static_data=ctx.cache.copy()),
            pool_size=POOL_SIZE,
        )
        try:
            native_profile.reset()
            if native_profile.enabled():
                native_profile.start_loop_thread_cpu()
            ctx.executor = executor
            await executor.run()
        except Aborted:
            asyncio.get_running_loop().call_soon(collect_failed_run, ctx)
        except asyncio.CancelledError:
            # The app's Stop drops its /run request before /kill, so a stopped run
            # usually ends here, cancelled, not with Aborted.
            asyncio.get_running_loop().call_soon(collect_failed_run, ctx)
            raise
        finally:
            ctx.executor = None
            # No gc.collect() here: a successful run never collects; collect_failed_run
            # runs one after a failed or stopped run.
            # Before the line, whose "numpy_pool" entry reports the release.
            numpy_pool.release()
            line = native_profile.log_line()
            if line is not None:
                logger.info(line)

        return json(success_response(), status=200)
    except Exception as exception:
        logger.error(exception)
        if (
            isinstance(exception, NodeExecutionError)
            and exception.__cause__ is not None
        ):
            trace = "".join(
                traceback.format_exception(
                    type(exception.__cause__),
                    exception.__cause__,
                    exception.__cause__.__traceback__,
                )
            )
        else:
            trace = traceback.format_exc()
        logger.error(trace)

        error: ExecutionErrorData = {
            "message": "Error running nodes!",
            "source": None,
            "exception": str(exception),
            "exceptionTrace": trace,
        }
        if isinstance(exception, NodeExecutionError):
            error["source"] = {
                "nodeId": exception.node_id,
                "schemaId": exception.node_data.schema_id,
                "inputs": exception.inputs,
            }

        ctx.queue.put({"event": "execution-error", "data": error})
        # After this handler returns, so the error is unreferenced by then.
        asyncio.get_running_loop().call_soon(collect_failed_run, ctx)
        return json(
            error_response("Error running nodes!", exception),
            status=500,
            headers=restart_headers(),
        )
    finally:
        ctx.run_starting = False
        if ctx.config.trace and tracer is not None:
            logger.info("Stopping VizTracer...")
            tracer.stop()
            tracer.save(f"../traces/trace_{executor_id}.json")


def collect_failed_run(ctx: AppContext) -> None:
    """Frees what a failed or stopped /run's reference cycles hold, then releases the
    NumPy pool if no /run executes.

    The error's traceback holds the run's frames, and the futures and Lazy results
    that hold the error are reachable from those frames, so the run's inputs (its
    arrays) live until a collection reaches the cycle's oldest generation. A stopped
    /run is collected too: its arrays were measured still held after the next /runs.
    run schedules this with call_soon, so it runs once the handler has returned and
    dropped the exception. A successful run never collects: gc's only other use is
    return_device_memory, outside runs. The tensors the collection frees stay
    reserved by PyTorch, so they go back to the driver too.
    """
    gc.collect()
    # Not while a /run executes: it releases at its own end.
    if ctx.executor is None:
        numpy_pool.release()
        try:
            return_device_memory(collect=False)
        except Exception as error:
            # A broken CUDA context; the host restarts the worker (restart_headers).
            logger.error("Could not return the GPU's memory: %s", error)


def cuda_context_broken() -> bool:
    """Whether a GPU's CUDA context in this process has a sticky error (an illegal
    access, or an out-of-memory error inside a kernel). Every later CUDA call fails in
    it, freeing memory included, so its memory stays taken until the process ends."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_initialized():
        return False
    for device in range(torch.cuda.device_count()):
        # Only devices this process used: synchronizing another would create a context.
        if torch.cuda.memory_reserved(device) == 0:
            continue
        try:
            torch.cuda.synchronize(device)
        except Exception as error:
            logger.error("The CUDA context of GPU %d is broken: %s", device, error)
            return True
    return False


def restart_headers() -> dict[str, str]:
    """The headers that ask the host to restart this worker when its CUDA context is
    broken; none otherwise."""
    return {RESTART_WORKER_HEADER: "1"} if cuda_context_broken() else {}


class RunIndividualRequest(TypedDict):
    id: NodeId
    inputs: list[object]
    schemaId: str
    options: JsonExecutionOptions


@app.route("/run/individual", methods=["POST"])
async def run_individual(request: Request):
    """Runs a single node"""
    ctx = AppContext.get(request.app)
    start = time.perf_counter()
    try:
        full_data: RunIndividualRequest = dict(request.json)  # type: ignore
        logger.debug(full_data)

        node_id = full_data["id"]
        ctx.cache.pop(node_id, None)
        arrival = next(individual_arrivals)
        latest_individual_arrival[node_id] = arrival
        # Counted from before the wait for its node: a /run waits for every preview,
        # also one still waiting for its node to load.
        with run_individual_counter:
            await wait_for_nodes([full_data["schemaId"]])
            if latest_individual_arrival[node_id] != arrival:
                # A newer preview of this node arrived while this one waited for its node
                # to load (an input changed): the waiters resume in no particular order,
                # and only the newest may run, else it could be killed by this one and
                # this one's output cached.
                return json({"success": True, "data": None})

            schema_data = api.registry.nodes.get(full_data["schemaId"])

            if schema_data is None:
                raise ValueError(
                    f"Invalid node {full_data['schemaId']} attempted to run individually"
                )

            node_data, _ = schema_data
            if node_data.kind == "generator":
                node = GeneratorNode(node_id, full_data["schemaId"])
            elif node_data.kind == "regularNode":
                node = FunctionNode(node_id, full_data["schemaId"])
            else:
                raise ValueError(
                    f"Invalid node kind {node_data.kind} attempted to run individually"
                )
            chain = Chain()
            chain.add_node(node)

            for index, i in enumerate(full_data["inputs"]):
                chain.inputs.set(node_id, node.data.inputs[index].id, i)

            # only yield certain types of events
            queue = EventConsumer.filter(
                ctx.queue, {"node-finish", "node-broadcast", "execution-error"}
            )

            execution_id = ExecutionId("individual-executor " + node_id)
            executor_class = NativeExecutor if executor_kind() == "native" else Executor
            executor = executor_class(
                id=execution_id,
                chain=chain,
                send_broadcast_data=True,
                options=ExecutionOptions.parse(full_data["options"]),
                loop=app.loop,
                queue=queue,
                storage_dir=ctx.storage_dir,
                pool=ctx.pool,
                pool_size=POOL_SIZE,
            )

            try:
                if execution_id in ctx.individual_executors:
                    # kill the previous executor (if any)
                    old_executor = ctx.individual_executors[execution_id]
                    old_executor.kill()

                ctx.individual_executors[execution_id] = executor
                if node_data.kind == "generator":
                    assert isinstance(node, GeneratorNode)
                    output = await executor.process_generator_node(node)
                elif node_data.kind == "regularNode":
                    assert isinstance(node, FunctionNode)
                    output = await executor.process_regular_node(node)
                else:
                    raise ValueError(
                        f"Invalid node kind {node_data.kind} attempted to run individually"
                    )
                if not isinstance(node, GeneratorNode):
                    ctx.cache[node_id] = output
            except Aborted:
                pass
            finally:
                if ctx.individual_executors.get(execution_id, None) == executor:
                    ctx.individual_executors.pop(execution_id, None)
                # The response does not wait for the broadcasts, as upstream's;
                # the NumPy pool is released once they are sent.
                release = asyncio.create_task(release_after_broadcasts(ctx, executor))
                individual_releases.add(release)
                release.add_done_callback(individual_releases.discard)
                # gc.collect() removed - called after every individual node, too frequent

        # From the request's arrival, a wait for the node to load included: the time
        # from a launch to a startup chain's first preview reads from the log.
        logger.info(
            f"Individual run of {full_data['schemaId']} finished in"
            f" {(time.perf_counter() - start) * 1000:.0f} ms"
        )
        return json({"success": True, "data": None})
    except Exception as exception:
        logger.error(exception, exc_info=True)
        return json({"success": False, "error": str(exception)})


# The deferred NumPy pool releases of /run/individual, held until they finish.
individual_releases: set[asyncio.Task[None]] = set()
# /run/individual's arrival numbers, and by node id the newest arrival's.
individual_arrivals = itertools.count()
latest_individual_arrival: dict[NodeId, int] = {}


async def release_after_broadcasts(
    ctx: AppContext, executor: Executor | NativeExecutor
) -> None:
    """Releases the NumPy pool once an individual run's broadcasts are sent.

    The broadcasts free their temporaries when they finish; a release before that
    would leave them idle until the next release point.
    """
    try:
        try:
            await executor.flush_broadcasts()
        finally:
            # Not while a /run executes: it releases at its own end.
            if ctx.executor is None:
                numpy_pool.release()
    except Exception:
        logger.exception(
            "Error sending an individual run's broadcasts or releasing the NumPy pool"
        )


def return_device_memory(collect: bool = True) -> None:
    """Hand GPU memory that released node resources freed back to the driver.

    PyTorch keeps freed blocks reserved for reuse, so the memory stays occupied in the OS
    until emptied. Never imports torch: without a loaded, initialized CUDA there is
    nothing to return. collect=False for a caller that has just collected.
    """
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        if collect:
            gc.collect()
        torch.cuda.empty_cache()


# Seconds after the last /clear-cache/individual of a burst before the one return.
DEVICE_MEMORY_RETURN_DELAY = 0.05
# The pending return of a burst, None when none is pending.
device_memory_return: asyncio.TimerHandle | None = None


def return_device_memory_after_burst(ctx: AppContext) -> None:
    """Runs return_device_memory once, DEVICE_MEMORY_RETURN_DELAY seconds after the
    last of a burst of node clears.

    Deleting N nodes sends N /clear-cache/individual requests, and each return is a
    full collection (tens of ms with a loaded chain) on this loop; a new request
    replaces the pending return, so the burst ends with one.
    """
    global device_memory_return
    if device_memory_return is not None:
        device_memory_return.cancel()

    def run() -> None:
        global device_memory_return
        device_memory_return = None
        # Not while a /run executes: it releases at its own end.
        if ctx.executor is None:
            return_device_memory()

    device_memory_return = asyncio.get_running_loop().call_later(
        DEVICE_MEMORY_RETURN_DELAY, run
    )


@app.route("/clear-cache/individual", methods=["POST"])
async def clear_cache_individual(request: Request):
    ctx = AppContext.get(request.app)
    try:
        full_data = dict(request.json)  # type: ignore
        # TensorRT and VOSR engines, sessions and models outlive runs (every
        # SharedCache); a session in use is freed when its run returns it.
        tensorrt_cache.release_node(full_data["id"])
        return_device_memory_after_burst(ctx)
        if ctx.cache.get(full_data["id"], None) is not None:
            del ctx.cache[full_data["id"]]
            if ctx.executor is None:
                numpy_pool.release()
        return json({"success": True, "data": None})
    except Exception as exception:
        logger.error(exception, exc_info=True)
        return json({"success": False, "error": str(exception)})


@app.route("/pause", methods=["POST"])
async def pause(request: Request):
    """Pauses the current execution"""
    ctx = AppContext.get(request.app)

    logger.info("Attempting to pause executor...")

    if not ctx.executor:
        logger.warning("No executor to pause.")
        return json(no_executor_response(), status=400)

    try:
        ctx.executor.pause()
        logger.info("Paused executor.")
        return json(success_response(), status=200)
    except Exception as exception:
        logger.log(2, exception, exc_info=True)
        return json(error_response("Error pausing execution!", exception), status=500)


@app.route("/resume", methods=["POST"])
async def resume(request: Request):
    """Pauses the current execution"""
    ctx = AppContext.get(request.app)

    logger.info("Attempting to resume executor...")

    if not ctx.executor:
        logger.warning("No executor to resume.")
        return json(no_executor_response(), status=400)

    try:
        ctx.executor.resume()
        logger.info("Resumed executor.")
        return json(success_response(), status=200)
    except Exception as exception:
        logger.log(2, exception, exc_info=True)
        return json(error_response("Error resuming execution!", exception), status=500)


@app.route("/kill", methods=["POST"])
async def kill(request: Request):
    """Kills the current execution"""
    ctx = AppContext.get(request.app)

    logger.info("Attempting to kill executor...")

    if not ctx.executor:
        if ctx.run_starting:
            # A /run waiting for previews: it ends there instead of starting.
            ctx.run_stopped_early = True
            logger.info("Stopped a run that waited for previews.")
            return json(success_response(), status=200)
        logger.warning("No executor to kill.")
        return json(no_executor_response(), status=400)

    try:
        ctx.executor.kill()
        while ctx.executor:
            await asyncio.sleep(0.0001)
        logger.info("Killed executor.")
        return json(success_response(), status=200, headers=restart_headers())
    except Exception as exception:
        logger.log(2, exception, exc_info=True)
        return json(error_response("Error killing execution!", exception), status=500)


@app.route("/status")
async def status(_request: Request):
    ctx = AppContext.get(app)

    executor_status: str
    if ctx.executor:
        e = ctx.executor
        if e.progress.aborted:
            executor_status = "killing"
        elif e.progress.paused:
            executor_status = "paused"
        else:
            executor_status = "running"
    else:
        executor_status = "ready"

    return json({"executor": executor_status})


@app.route("/packages", methods=["GET"])
async def get_packages(request: Request):
    # The host reads only the dependencies, which the packages register before their
    # nodes load; some node modules add settings.
    if request.args.get("waitForNodes", "true") == "true":
        await nodes_available()
    else:
        while not node_loading.packages_registered.is_set():
            await asyncio.sleep(0.01)

    hide_internal = request.args.get("hideInternal", "true") == "true"

    packages = []
    for package in api.registry.packages.values():
        if package.name == "chaiNNer_standard" and hide_internal:
            continue

        pkg_dependencies = []
        for pkg_dep in package.dependencies:
            installed_version = installed_packages.get(pkg_dep.pypi_name, None)
            pkg_dep_item = {
                **pkg_dep.to_dict(),
            }
            if installed_version is None:
                pkg_dep_item["installed"] = None
            else:
                pkg_dep_item["installed"] = installed_version
            pkg_dependencies.append(pkg_dep_item)

        packages.append(package.to_dict())

    return json(packages)


@app.route("/features")
async def get_features(_request: Request):
    await nodes_available()

    features: list[tuple[api.Feature, api.Package]] = []
    for package in api.registry.packages.values():
        for feature in package.features:
            features.append((feature, package))

    # check all features in parallel
    async def check(feature: api.Feature) -> api.FeatureState | None:
        if feature.behavior is None:
            # no behavior assigned
            return None

        try:
            return await feature.behavior.check()
        except Exception as e:
            return api.FeatureState.disabled(str(e))

    # because good API design just isn't pythonic, asyncio.gather will return List[Any].
    results: list[api.FeatureState | None] = await asyncio.gather(
        *[check(f) for f, _ in features]
    )

    features_json = []
    for (feature, package), state in zip(features, results):
        if state is None:
            continue

        features_json.append(
            {
                "packageId": package.id,
                "featureId": feature.id,
                "enabled": state.is_enabled,
                "details": state.details,
            }
        )

    return json(features_json)


@app.get("/sse")
async def sse(request: Request):
    ctx = AppContext.get(request.app)
    headers = {"Cache-Control": "no-cache"}
    response = await request.respond(headers=headers, content_type="text/event-stream")
    if response is None:
        return

    while True:
        message = await ctx.queue.get()
        await response.send(event_text(message))


def log_load_errors(load_errors: list[api.LoadErrorInfo]) -> None:
    if len(load_errors) > 0:
        import_errors: list[api.LoadErrorInfo] = []
        for e in load_errors:
            if not isinstance(e.error, ModuleNotFoundError):
                logger.warning(
                    f"Failed to load {e.module} ({e.file}):", exc_info=e.error
                )
            else:
                import_errors.append(e)

        if len(import_errors) > 0:
            logger.warning(f"Failed to import {len(import_errors)} modules:")

            by_error: dict[str, list[api.LoadErrorInfo]] = {}
            for e in import_errors:
                key = str(e.error)
                if key not in by_error:
                    by_error[key] = []
                by_error[key].append(e)

            for error in sorted(by_error.keys()):
                modules = [e.module for e in by_error[error]]
                if len(modules) == 1:
                    logger.warning(f"{error}  ->  {modules[0]}")
                else:
                    count = len(modules)
                    if count > 3:
                        modules = [*modules[:2], f"and {count - 2} more ..."]
                    l = "\n".join("  ->  " + m for m in modules)
                    logger.warning(f"{error}  ->  {count} modules ...\n{l}")


async def import_packages(config: ServerConfig):
    """Loads every package's nodes in a thread of its own, so the worker serves from its
    start: runs of the nodes that have loaded go ahead, the others wait for theirs
    (wait_for_nodes)."""
    await asyncio.to_thread(node_loading.load, config)


async def setup(sanic_app: Sanic):
    await import_packages(AppContext.get(sanic_app).config)
    logger.info(native.isa_line())


exit_code = 0


async def close_server(sanic_app: Sanic):
    # pylint: disable=global-statement
    global exit_code

    try:
        await nodes_available()
    except Exception as ex:
        logger.error(f"Error waiting for server to start: {ex}")
        exit_code = 1

    # Sanic 25 serves only after the after_server_start listeners return, and loses
    # a stop requested before that. Setup can finish first.
    while not sanic_app.state.is_running:
        await asyncio.sleep(0.01)

    # now we can close the server
    logger.info("Closing server...")
    sanic_app.stop()


@app.after_server_start
async def after_server_start(sanic_app: Sanic):
    # pylint: disable=global-statement
    global setup_task, close_task
    ctx = AppContext.get(sanic_app)
    loop = asyncio.get_running_loop()

    # start the setup task
    setup_task = loop.create_task(setup(sanic_app))

    # start task to close the server
    if ctx.config.close_after_start:
        close_task = loop.create_task(close_server(sanic_app))


def exit_with_host(pid: int) -> None:
    """Ends this worker once its host process has ended.

    A host that is killed or crashes stops nothing (the app kills the host when its
    /shutdown takes over 10 s), so the worker kept running with its GPU memory, DLLs
    and port.
    """
    try:
        psutil.Process(pid).wait()
    except psutil.NoSuchProcess:
        pass
    logger.warning(f"The host process {pid} ended; the worker exits.")
    os._exit(1)


def main():
    ctx = AppContext.get(app)
    config = ctx.config
    if config.host_pid is not None:
        threading.Thread(
            target=exit_with_host,
            args=(config.host_pid,),
            daemon=True,
            name="host watch",
        ).start()
    # numba (PyMatting's cache=True kernels) otherwise writes its cache beside the
    # runtime's site-packages. numba reads the variable when it is first imported, and
    # the node modules that import it load in setup's thread, after this.
    os.environ["NUMBA_CACHE_DIR"] = str(ctx.storage_dir / "numba-cache")
    # The VapourSynth package keeps its function catalog, user functions and overlays
    # there; it reads the variable when its node modules load, in setup.
    os.environ["CHAINNER_VS_DIR"] = str(ctx.storage_dir / "vapoursynth")
    # Temporary files go to a folder of this worker's own; the folders of workers that
    # ended without removing theirs are removed.
    use_process_temp_folder(ctx.storage_dir)
    # Before the event loop exists: its tasks copy this thread's context.
    numpy_pool.install()
    app.run(port=config.port, single_process=True)
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
