"""The worker's node loading: the light node modules load first and import no heavy
library, and a run waits for nodes that are still loading."""

from __future__ import annotations

import ast
import asyncio
import importlib
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

BACKEND = Path(__file__).resolve().parents[1] / "src"

# The top-level modules the light node modules must not import: each takes from a tenth
# of a second (networkx) to seconds (torch) to import.
HEAVY = [
    "torch",
    "torchvision",
    "spandrel",
    "sympy",
    "networkx",
    "scipy",
    "numba",
    "pymatting",
    "onnxruntime",
    "tensorrt",
    "ncnn",
    "vapoursynth",
]

# Loads the light node modules as the worker does first, then prints what they imported.
LIGHT = """
import json, os, sys

# The worker runs server.py: modules find the backend's files by its path.
sys.modules["__main__"].__file__ = os.path.abspath("server.py")
import server

light, _ = server.register_packages()
errors = server.api.registry.load_modules(m for batch in light.values() for m in batch)
print(json.dumps({
    "nodes": len(server.api.registry.nodes),
    "errors": [f"{e.module}: {e.error!r}" for e in errors],
    "loaded": sorted({name.split(".")[0] for name in sys.modules}),
}))
"""


def test_the_light_node_modules_import_no_heavy_library():
    done = subprocess.run(
        [sys.executable, "-B", "-c", LIGHT],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    light = json.loads(done.stdout.splitlines()[-1])
    assert light["errors"] == []
    assert light["nodes"] > 100
    assert sorted(set(HEAVY) & set(light["loaded"])) == []


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # The module parses its command line when imported.
    monkeypatch.setattr(sys, "argv", ["server.py"])
    return importlib.import_module("server")


def test_every_heavy_module_exists(server: ModuleType):
    for module in server.HEAVY_MODULES:
        path = BACKEND.joinpath(*module.split(".")).with_suffix(".py")
        assert path.is_file(), module


@pytest.mark.parametrize("kind", ["inputs", "outputs"])
def test_every_framework_input_and_output_loads_on_first_use(kind: str):
    package = importlib.import_module(f"nodes.properties.{kind}")
    lazy: dict[str, str] = getattr(package, f"FRAMEWORK_{kind.upper()}")
    for module in sorted(set(lazy.values())):
        tree = ast.parse(
            (BACKEND / "nodes" / "properties" / kind / f"{module}.py").read_text(
                encoding="utf-8"
            )
        )
        defined = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        } | {
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        public = {name for name in defined if not name.startswith("_")}
        assert public == {name for name, m in lazy.items() if m == module}, module


def loading_with(server: ModuleType, monkeypatch: pytest.MonkeyPatch):
    """A fresh loading state with the PyTorch package registered and no node, and the
    package."""
    import api

    pytorch = api.Package(
        where="packages/chaiNNer_pytorch/__init__.py",
        id="chaiNNer_pytorch",
        name="PyTorch",
        description="",
        icon="PyTorch",
        color="#DD6B20",
    )
    pytorch.add_category("PyTorch", "", "PyTorch", "#DD6B20")
    monkeypatch.setattr(api.registry, "packages", {pytorch.where: pytorch})
    monkeypatch.setattr(api.registry, "nodes", {})
    monkeypatch.setattr(api.registry, "categories", [])
    loading = server.NodeLoading()
    monkeypatch.setattr(server, "node_loading", loading)
    return loading, pytorch


def upscale_image_node() -> None:
    pass


def test_clearing_active_preview_stops_side_effect_and_allows_undo(
    server: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    import api

    loading, pytorch = loading_with(server, monkeypatch)
    loading.light_loaded.set()
    loading.package_loaded["chaiNNer_pytorch"].set()
    group = pytorch.categories[0].add_node_group("Processing")
    group.register(
        UPSCALE, name="Preview", description="", inputs=[], outputs=[], generated=True
    )(upscale_image_node)
    api.registry.load_modules([])
    markers: list[str] = []

    async def scenario():
        entered, resume = asyncio.Event(), asyncio.Event()

        class Executor:
            def __init__(self, **_kwargs: object):
                self.stopped = False

            def kill(self):
                self.stopped = True

            async def process_regular_node(self, _node: object):
                entered.set()
                await resume.wait()
                if self.stopped:
                    raise server.Aborted()
                markers.append("written")
                return "output"

            async def flush_broadcasts(self):
                pass

        monkeypatch.setattr(server, "Executor", Executor)
        monkeypatch.setattr(server, "executor_kind", lambda: "python")
        monkeypatch.setattr(
            server, "return_device_memory_after_burst", lambda _ctx: None
        )
        monkeypatch.setattr(server.tensorrt_cache, "release_node", lambda _id: None)
        monkeypatch.setattr(server.app.ctx, "cache", {})
        monkeypatch.setattr(server.app.ctx, "individual_executors", {})
        app = SimpleNamespace(ctx=server.app.ctx, loop=asyncio.get_running_loop())
        monkeypatch.setattr(server, "app", app)
        request = SimpleNamespace(
            app=app,
            json={
                "id": "cleared-node",
                "schemaId": UPSCALE,
                "inputs": [],
                "options": {},
            },
        )
        preview = asyncio.create_task(server.run_individual(request))
        await asyncio.wait_for(entered.wait(), 2)
        cleared = await server.clear_cache_individual(
            SimpleNamespace(app=app, json={"id": "cleared-node"})
        )
        assert json.loads(cleared.body)["success"]
        resume.set()
        await asyncio.wait_for(preview, 2)
        assert markers == []
        assert "cleared-node" not in app.ctx.cache
        # Undo restores the same ID: it gets a new executor, not a permanent tombstone.
        restored = await server.run_individual(request)
        assert json.loads(restored.body)["success"]
        assert markers == ["written"]
        assert app.ctx.cache["cleared-node"] == "output"
        if server.individual_releases:
            await asyncio.gather(*server.individual_releases)

    asyncio.run(scenario())


UPSCALE = "chainner:pytorch:upscale_image"


@pytest.mark.parametrize("undo_first", [False, True])
@pytest.mark.parametrize("background", [False, True])
def test_cleared_preview_late_cache_use_preserves_undo(
    server: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    undo_first: bool,
    background: bool,
):
    import api
    from chain.chain import FunctionNode
    from nodes.impl.tensorrt.cache import SharedCache

    loading, pytorch = loading_with(server, monkeypatch)
    loading.light_loaded.set()
    loading.package_loaded["chaiNNer_pytorch"].set()
    entered, proceed = threading.Event(), threading.Event()
    closed: list[object] = []
    shared: SharedCache[object] = SharedCache(closed.append)
    old_value, new_value = object(), object()
    node_id = api.NodeId("late-cache-preview")

    def preview_body(context: SimpleNamespace):
        if context.first:
            entered.set()
            assert proceed.wait(5)
        value = old_value if context.first else new_value
        with shared.use(value, node_id, lambda: value):
            return value

    group = pytorch.categories[0].add_node_group("Processing")
    group.register(
        UPSCALE,
        name="Preview",
        description="",
        inputs=[],
        outputs=[],
        generated=True,
        node_context=True,
    )(preview_body)
    api.registry.load_modules([])

    async def scenario():
        bodies: list[asyncio.Task[object]] = []
        executors: list[Executor] = []

        class Executor:
            def __init__(self, **_kwargs: object):
                self.first = not executors
                self.stopped = asyncio.Event()
                executors.append(self)

            def kill(self):
                self.stopped.set()

            async def process_regular_node(self, node: FunctionNode):
                body = asyncio.create_task(
                    asyncio.to_thread(node.data.run, SimpleNamespace(first=self.first))
                )
                bodies.append(body)
                if self.first and background:
                    await self.stopped.wait()
                    # Native previews can return Aborted while their node still runs.
                    raise server.Aborted()
                value = await body
                if self.stopped.is_set():
                    raise server.Aborted()
                return value

            async def flush_broadcasts(self):
                pass

        monkeypatch.setattr(server, "Executor", Executor)
        monkeypatch.setattr(server, "executor_kind", lambda: "python")
        monkeypatch.setattr(
            server, "return_device_memory_after_burst", lambda _ctx: None
        )
        monkeypatch.setattr(server.app.ctx, "cache", {})
        monkeypatch.setattr(server.app.ctx, "individual_executors", {})
        app = SimpleNamespace(ctx=server.app.ctx, loop=asyncio.get_running_loop())
        monkeypatch.setattr(server, "app", app)
        request = SimpleNamespace(
            app=app,
            json={"id": node_id, "schemaId": UPSCALE, "inputs": [], "options": {}},
        )
        preview = asyncio.create_task(server.run_individual(request))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            cleared = await server.clear_cache_individual(request)
            assert json.loads(cleared.body)["success"]
            if background:
                await asyncio.wait_for(preview, 2)
            if undo_first:
                restored = await server.run_individual(request)
                assert json.loads(restored.body)["success"]
                assert app.ctx.cache[node_id] is new_value
                assert len(shared) == 1 and not closed
            proceed.set()
            await asyncio.wait_for(preview, 2)
            await asyncio.wait_for(asyncio.gather(*bodies), 2)
            assert closed == [old_value]
            if undo_first:
                assert app.ctx.cache[node_id] is new_value
                assert len(shared) == 1
            else:
                assert node_id not in app.ctx.cache and len(shared) == 0
                restored = await server.run_individual(request)
                assert json.loads(restored.body)["success"]
                assert app.ctx.cache[node_id] is new_value
            cleared = await server.clear_cache_individual(request)
            assert json.loads(cleared.body)["success"]
            assert closed == [old_value, new_value] and len(shared) == 0
        finally:
            proceed.set()
            await asyncio.gather(preview, *bodies, return_exceptions=True)
            if server.individual_releases:
                await asyncio.gather(*server.individual_releases)
            shared.release_node(node_id)

    asyncio.run(scenario())


def test_a_run_waits_for_its_node_to_load(
    server: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    import api

    loading, pytorch = loading_with(server, monkeypatch)

    async def run_at_startup() -> bool:
        waiting = asyncio.create_task(server.wait_for_nodes([UPSCALE]))
        await asyncio.sleep(0.05)
        assert not waiting.done()

        # the light nodes load: the PyTorch nodes are still loading
        loading.light_loaded.set()
        await asyncio.sleep(0.05)
        assert not waiting.done()

        # loaded in the loading thread
        def load() -> None:
            group = pytorch.categories[0].add_node_group("Processing")
            group.register(
                UPSCALE,
                name="Upscale Image",
                description="Upscales an image.",
                inputs=[],
                outputs=[],
                generated=True,
            )(upscale_image_node)
            api.registry.load_modules([])
            loading.package_loaded["chaiNNer_pytorch"].set()

        threading.Thread(target=load).start()
        await asyncio.wait_for(waiting, 1)
        return True

    assert asyncio.run(run_at_startup())


def test_of_two_previews_waiting_for_their_node_only_the_newer_runs(
    server: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    # The user picks another model while PyTorch still loads: the newer preview runs
    # and its output is cached, whichever waiter resumes first.
    import api

    loading, pytorch = loading_with(server, monkeypatch)
    loading.light_loaded.set()
    ran: list[object] = []
    killed: list[object] = []

    class Executor:
        def __init__(self, *, options: api.ExecutionOptions, **_kwargs: object) -> None:
            self.n = options.get_package_settings_json("request")["n"]

        async def process_regular_node(self, _node: object) -> str:
            ran.append(self.n)
            return f"output {self.n}"

        def kill(self) -> None:
            killed.append(self.n)

        async def flush_broadcasts(self) -> None:
            pass

    monkeypatch.setattr(server, "executor_kind", lambda: "python")
    monkeypatch.setattr(server, "Executor", Executor)
    cache: dict[str, object] = {}
    monkeypatch.setattr(server.app.ctx, "cache", cache)
    # the app's loop exists only while it serves
    app = SimpleNamespace(ctx=server.app.ctx, loop=None)
    monkeypatch.setattr(server, "app", app)

    def preview(n: int):
        return server.run_individual(
            SimpleNamespace(
                app=app,
                json={
                    "id": "node",
                    "schemaId": UPSCALE,
                    "inputs": [],
                    "options": {"request": {"n": n}},
                },
            )
        )

    async def startup() -> list[object]:
        older = asyncio.create_task(preview(1))
        await asyncio.sleep(0.05)
        newer = asyncio.create_task(preview(2))
        await asyncio.sleep(0.05)
        assert not older.done() and not newer.done()
        # a /run waits for both (run_individual_counter.wait_zero)
        assert server.run_individual_counter.count == 2
        group = pytorch.categories[0].add_node_group("Processing")
        group.register(
            UPSCALE,
            name="Upscale Image",
            description="Upscales an image.",
            inputs=[],
            outputs=[],
            generated=True,
        )(upscale_image_node)
        api.registry.load_modules([])
        loading.package_loaded["chaiNNer_pytorch"].set()
        replies = await asyncio.wait_for(asyncio.gather(older, newer), 2)
        return [json.loads(reply.body) for reply in replies]

    for _ in range(10):
        ran.clear()
        assert asyncio.run(startup()) == [{"success": True, "data": None}] * 2
        assert ran == [2]
        assert killed == []
        assert cache == {"node": "output 2"}
        # no entry stays once no preview of the node waits
        assert "node" not in server.latest_individual_arrival
        monkeypatch.setattr(api.registry, "nodes", {})
        pytorch.categories[0].node_groups.clear()
        loading.package_loaded["chaiNNer_pytorch"].clear()


def test_a_run_of_a_package_that_failed_to_load_fails_with_its_error(
    server: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    import api

    loading, _ = loading_with(server, monkeypatch)
    loading.light_loaded.set()
    loading.errors["chaiNNer_pytorch"] = [
        api.LoadErrorInfo(
            "packages.chaiNNer_pytorch.pytorch.processing.upscale_image",
            "upscale_image.py",
            ModuleNotFoundError("No module named 'torch'"),
        )
    ]
    loading.package_loaded["chaiNNer_pytorch"].set()

    with pytest.raises(ValueError, match=r"PyTorch nodes failed to load \(No module"):
        asyncio.run(server.wait_for_nodes([UPSCALE]))


@pytest.mark.parametrize(
    ("error", "complete"),
    [
        # a dependency that is not installed: the list is as it will stay
        (ModuleNotFoundError("No module named 'tensorrt'", name="tensorrt"), True),
        # torch under memory pressure: it may load at the next start
        (OSError("[WinError 1455] The paging file is too small"), False),
    ],
)
def test_the_node_list_says_whether_a_package_failed_to_load(
    server: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    complete: bool,
):
    import api

    loading, _ = loading_with(server, monkeypatch)
    loading.errors["chaiNNer_pytorch"] = [
        api.LoadErrorInfo("packages.chaiNNer_pytorch.settings", "settings.py", error)
    ]
    monkeypatch.setattr(server, "setup_task", None)
    reply = asyncio.run(server.nodes(None))
    assert json.loads(reply.body)["complete"] is complete


def test_a_node_no_package_has_is_left_to_the_run_once_all_loaded(
    server: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    loading, _ = loading_with(server, monkeypatch)
    loading.light_loaded.set()
    # the PyTorch nodes loaded without it: another package may still have it
    loading.package_loaded["chaiNNer_pytorch"].set()

    async def wait() -> None:
        waiting = asyncio.create_task(server.wait_for_nodes(["chainner:pytorch:gone"]))
        await asyncio.sleep(0.05)
        assert not waiting.done()
        loading.done.set()
        await asyncio.wait_for(waiting, 1)

    asyncio.run(wait())


def test_a_stopped_run_stops_waiting(
    server: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    loading_with(server, monkeypatch)
    stopped = threading.Event()

    async def wait() -> None:
        waiting = asyncio.create_task(server.wait_for_nodes([UPSCALE], stopped.is_set))
        await asyncio.sleep(0.05)
        assert not waiting.done()
        stopped.set()
        await asyncio.wait_for(waiting, 1)

    asyncio.run(wait())


def test_a_node_list_request_given_up_while_the_nodes_load_leaves_the_loading_alone(
    server: ModuleType, monkeypatch: pytest.MonkeyPatch
):
    """A client that gives up on /nodes (a timeout, a closed window) while the nodes
    still load cancels its own request only: the loading goes on, and the next
    request gets the list. Awaiting the setup task directly cancelled it with the
    request, so every later /nodes failed at once."""

    async def scenario() -> None:
        loaded = asyncio.Event()

        async def setup() -> None:
            await loaded.wait()

        monkeypatch.setattr(server, "setup_task", asyncio.create_task(setup()))
        request = asyncio.create_task(server.nodes_available())
        await asyncio.sleep(0.01)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert not server.setup_task.cancelled()
        loaded.set()
        await asyncio.wait_for(server.nodes_available(), 1)

    asyncio.run(scenario())
