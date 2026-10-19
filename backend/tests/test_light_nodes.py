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


UPSCALE = "chainner:pytorch:upscale_image"


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
