"""real_use_pass's pieces that need no running app.

The chain conversion and the Save rewrite, the launch's tree diff against the
ruled allow-list, the log checks, the dependency rule, the comparisons and the
summary's gates.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import independent_ui
import psutil
import pytest
import real_use_pass as r
from verify_runtime import OwnedJob

LOAD = r.LOAD_IMAGE
SAVE = r.SAVE
RESIZE = "chainner:image:resize"
TEXT = "chainner:utility:text_append"
MODEL = "chainner:pytorch:load_model"


def uid(tag: str) -> str:
    """A 36-character node id, as the UI's handles need."""
    return f"{tag:0>8}-0000-4000-8000-000000000000"


def schema(inputs: list[dict], outputs: list[int], side_effects: bool) -> dict:
    return {
        "inputs": inputs,
        "outputs": [{"id": output} for output in outputs],
        "hasSideEffects": side_effects,
    }


SCHEMAS = {
    LOAD: schema([{"id": 0, "label": "Image File", "kind": "file"}], [0, 1, 2], True),
    SAVE: schema(
        [
            {"id": 0, "label": "Image", "kind": "generic"},
            {"id": 1, "label": "Directory", "kind": "directory"},
            {
                "id": 2,
                "label": "Subdirectory Path",
                "kind": "text",
                "optional": True,
                "def": None,
            },
            {"id": 3, "label": "Image Name", "kind": "text", "def": None},
            {"id": 4, "label": "Image Format", "kind": "dropdown", "def": "png"},
        ],
        [],
        True,
    ),
    RESIZE: schema(
        [
            {"id": 0, "label": "Image", "kind": "generic"},
            {"id": 2, "label": "Percentage", "kind": "number", "def": 100.0},
            {"id": 1, "label": "Mode", "kind": "dropdown", "def": 0},
        ],
        [0],
        False,
    ),
    TEXT: schema(
        [
            {"id": 0, "label": "Separator", "kind": "text", "def": "-"},
            {"id": 1, "label": "Text A", "kind": "text", "def": None},
            {"id": 2, "label": "Text B", "kind": "text", "def": None},
        ],
        [0],
        False,
    ),
    # Upstream's output order differs from the ids: Name (id 1) is index 2.
    MODEL: schema([{"id": 0, "label": "Model", "kind": "file"}], [0, 2, 1], True),
}


def node(tag: str, schema_id: str, values: dict | None = None, **data: object) -> dict:
    return {
        "id": uid(tag),
        "type": "regularNode",
        "data": {
            "schemaId": schema_id,
            "inputData": values or {},
            "id": uid(tag),
            **data,
        },
    }


def edge(source: str, output: int, target: str, input_id: int) -> dict:
    return {
        "id": f"{source}-{output}-{target}-{input_id}",
        "source": uid(source),
        "target": uid(target),
        "sourceHandle": f"{uid(source)}-{output}",
        "targetHandle": f"{uid(target)}-{input_id}",
        "type": "main",
        "data": {},
    }


def resize_chain(path: str | None = None) -> dict:
    """resize.chn's shape: Save's directory and name come from Load Image."""
    return {
        "nodes": [
            node("save", SAVE, {"4": "png", "1000": 0}),
            node("text", TEXT, {"0": "-", "2": "resize"}),
            node("resize", RESIZE, {"2": 25}),
            node("load", LOAD, {} if path is None else {"0": path}),
        ],
        "edges": [
            edge("text", 0, "save", 3),
            edge("load", 1, "save", 1),
            edge("load", 0, "resize", 0),
            edge("load", 2, "text", 1),
            edge("resize", 0, "save", 0),
        ],
    }


def write_save(
    path: Path, content: dict, migration: object = r.CURRENT_MIGRATION
) -> Path:
    save: dict[str, object] = {
        "version": "0.24.2-nightly.2025-03-12",
        "content": content,
    }
    if migration is not None:
        save["migration"] = migration
    path.write_text(json.dumps(save), encoding="utf-8")
    return path


def bench_images(tmp_path: Path, count: int) -> list[dict[str, str]]:
    images = []
    for index in range(count):
        image = tmp_path / f"bench-{index}.jpg"
        image.write_bytes(f"image {index}".encode())
        images.append(
            {
                "path": str(image),
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            }
        )
    return images


# The conversion and the Save rewrite.


def test_backend_request_matches_the_ui_for_a_resize_chain(tmp_path):
    chain = resize_chain(str(tmp_path / "in.png"))
    data = r.backend_request(chain, SCHEMAS, tmp_path / "out")
    by_id = {item["id"]: item for item in data}
    assert [item["id"] for item in data] == [
        uid(t) for t in ("save", "text", "resize", "load")
    ]
    assert all(item["nodeType"] == "regularNode" for item in data)
    assert by_id[uid("save")]["inputs"] == [
        {"type": "edge", "id": uid("resize"), "index": 0},
        {"type": "value", "value": str(tmp_path / "out")},
        {"type": "value", "value": None},
        {"type": "edge", "id": uid("text"), "index": 0},
        {"type": "value", "value": "png"},
    ]
    assert by_id[uid("text")]["inputs"] == [
        {"type": "value", "value": "-"},
        {"type": "edge", "id": uid("load"), "index": 2},
        {"type": "value", "value": "resize"},
    ]
    # Schema order (ids 0, 2, 1); a saved value over the default.
    assert by_id[uid("resize")]["inputs"] == [
        {"type": "edge", "id": uid("load"), "index": 0},
        {"type": "value", "value": 25},
        {"type": "value", "value": 0},
    ]
    assert chain == resize_chain(str(tmp_path / "in.png")), "the content is not edited"


def test_an_edge_index_is_the_output_position_not_its_id(tmp_path):
    chain = {
        "nodes": [
            node("model", MODEL, {"0": "m.pth"}),
            node("text", TEXT),
            node("save", SAVE, {"3": "x"}),
            node("load", LOAD, {"0": "a.png"}),
        ],
        "edges": [
            edge("model", 1, "text", 1),
            edge("text", 0, "save", 3),
            edge("load", 0, "save", 0),
        ],
    }
    data = r.backend_request(chain, SCHEMAS, tmp_path)
    text = next(item for item in data if item["id"] == uid("text"))
    assert text["inputs"][1] == {"type": "edge", "id": uid("model"), "index": 2}


def test_a_saved_null_takes_the_default():
    saved = node("resize", RESIZE, {"2": None, "1": 3, "99": "extra"})
    data = r.with_defaults(saved, SCHEMAS)["data"]["inputData"]
    assert data == {"0": None, "2": 100.0, "1": 3, "99": "extra"}


@pytest.mark.parametrize(
    ("subdirectory", "refused"),
    [
        (None, False),
        ("", False),
        ("sub/dir", False),
        (r"sub\dir", False),
        ("..", True),
        (r"a\..\..\b", True),
        (r"C:\elsewhere", True),
        (r"\rooted", True),
        ("D:relative", True),
    ],
)
def test_a_save_subdirectory_must_stay_in_the_run(tmp_path, subdirectory, refused):
    chain = resize_chain("in.png")
    chain["nodes"][0]["data"]["inputData"]["2"] = subdirectory
    if refused:
        with pytest.raises(ValueError, match="escapes"):
            r.rewrite_saves(chain, SCHEMAS, tmp_path)
    else:
        nodes, edges = r.rewrite_saves(chain, SCHEMAS, tmp_path)
        assert nodes[0]["data"]["inputData"]["1"] == str(tmp_path)
        assert len(edges) == len(chain["edges"]) - 1
        assert all(e["targetHandle"] != f"{uid('save')}-1" for e in edges)


def test_a_connected_save_subdirectory_is_refused(tmp_path):
    chain = resize_chain("in.png")
    chain["edges"].append(edge("text", 0, "save", 2))
    with pytest.raises(ValueError, match="subdirectory is connected"):
        r.rewrite_saves(chain, SCHEMAS, tmp_path)


def test_an_unreviewed_save_schema_is_refused(tmp_path):
    schemas = copy.deepcopy(SCHEMAS)
    schemas[SAVE]["inputs"][1]["kind"] = "text"
    with pytest.raises(ValueError, match="Unreviewed"):
        r.rewrite_saves(resize_chain("in.png"), schemas, tmp_path)


def test_disabled_nodes_and_their_dependents_are_left_out(tmp_path):
    chain = resize_chain("in.png")
    chain["nodes"][2]["data"]["isDisabled"] = True  # resize feeds the Save
    nodes, edges = r.rewrite_saves(chain, SCHEMAS, tmp_path)
    kept, kept_edges = r.optimize_chain(nodes, edges, SCHEMAS)
    # The disabled resize takes the Save it feeds. The Text Append then reaches no
    # Save, and Load Image, left without edges, has a required input with no
    # default, so the UI drops it as an unused side-effect node.
    assert kept == []
    assert kept_edges == []
    with pytest.raises(ValueError, match="No node"):
        r.backend_request(chain, SCHEMAS, tmp_path)


def test_nodes_without_an_effect_are_left_out(tmp_path):
    beep = "chainner:test:beep"
    schemas = {
        **SCHEMAS,
        beep: schema(
            [{"id": 0, "label": "Tone", "kind": "number", "def": 440}], [], True
        ),
    }
    chain = resize_chain("in.png")
    chain["nodes"].append(node("dangle", RESIZE))
    chain["edges"].append(edge("load", 0, "dangle", 0))
    # Unconnected nodes with side effects: a Load Image needs an edge (its path
    # has no default) and a Save needs its image, so both go; a node whose every
    # input has a default stays.
    chain["nodes"].append(node("lonely", LOAD, {"0": "b.png"}))
    chain["nodes"].append(node("orphan", SAVE))
    chain["nodes"].append(node("beep", beep))
    data = r.backend_request(chain, schemas, tmp_path)
    assert [item["id"] for item in data] == [
        uid(tag) for tag in ("save", "text", "resize", "load", "beep")
    ]


def test_an_edge_to_a_missing_output_is_refused(tmp_path):
    chain = resize_chain("in.png")
    chain["edges"].append(edge("load", 7, "text", 2))
    with pytest.raises(ValueError, match="no output 7"):
        r.backend_request(chain, SCHEMAS, tmp_path)


def test_read_chain_accepts_a_current_save(tmp_path):
    content = resize_chain()
    path = write_save(tmp_path / "c.chn", content)
    assert r.read_chain(path, SCHEMAS) == {
        "nodes": content["nodes"],
        "edges": content["edges"],
    }


def test_read_chain_accepts_pending_migrations_that_change_nothing(tmp_path):
    path = write_save(tmp_path / "c.chn", resize_chain(), migration=43)
    assert r.read_chain(path, SCHEMAS)["nodes"]


@pytest.mark.parametrize(
    ("change", "migration", "message"),
    [
        (
            lambda c: c["nodes"][0].update(type="newIterator"),
            43,
            "newIteratorToGenerator",
        ),
        (
            lambda c: c["nodes"][3]["data"].update(
                schemaId="chainner:image:load_image_pairs"
            ),
            44,
            "splitLoadImagePairs",
        ),
        (lambda c: None, 42, "needs upstream migration 42"),
        (lambda c: None, 46, "not one this driver reads"),
        (lambda c: None, None, "not one this driver reads"),
        (
            lambda c: c["nodes"][1]["data"].update(schemaId="chainner:x:unknown"),
            45,
            "unknown",
        ),
        (lambda c: c["nodes"][1]["data"].update(isPassthrough=True), 45, "passthrough"),
    ],
)
def test_read_chain_refuses_what_it_cannot_run_as_saved(
    tmp_path, change, migration, message
):
    content = resize_chain()
    change(content)
    path = write_save(tmp_path / "c.chn", content, migration=migration)
    with pytest.raises(ValueError, match=message):
        r.read_chain(path, SCHEMAS)


def test_read_chain_refuses_an_unreviewed_node(tmp_path):
    schemas = {**SCHEMAS, "chainner:image:view": schema([], [], True)}
    content = resize_chain()
    content["nodes"].append(node("view", "chainner:image:view"))
    with pytest.raises(ValueError, match="not reviewed"):
        r.read_chain(write_save(tmp_path / "c.chn", content), schemas)


def test_an_unset_image_takes_a_bench_image(tmp_path):
    images = bench_images(tmp_path, 2)
    chain = resize_chain()
    chain["nodes"].append(node("load2", LOAD, {"0": ""}))
    resolved, substitutions, missing = r.resolve_inputs(chain, SCHEMAS, images, False)
    assert missing == []
    assert [s["used"] for s in substitutions] == [images[0]["path"], images[1]["path"]]
    assert resolved["nodes"][3]["data"]["inputData"]["0"] == images[0]["path"]
    assert resolved["nodes"][4]["data"]["inputData"]["0"] == images[1]["path"]
    assert "0" not in chain["nodes"][3]["data"]["inputData"], (
        "the content is not edited"
    )


def test_a_saved_image_present_here_is_read_in_place(tmp_path):
    present = tmp_path / "owner.png"
    present.write_bytes(b"png")
    resolved, substitutions, missing = r.resolve_inputs(
        resize_chain(str(present)), SCHEMAS, bench_images(tmp_path, 1), True
    )
    assert (substitutions, missing) == ([], [])
    assert resolved["nodes"][3]["data"]["inputData"]["0"] == str(present)


def test_a_missing_saved_image_skips_the_chain_without_fallback(tmp_path):
    absent = str(tmp_path / "absent.png")
    images = bench_images(tmp_path, 1)
    _, substitutions, missing = r.resolve_inputs(
        resize_chain(absent), SCHEMAS, images, False
    )
    assert substitutions == []
    assert missing == [{"node": uid("load"), "input": "Image File", "saved": absent}]
    _, substitutions, missing = r.resolve_inputs(
        resize_chain(absent), SCHEMAS, images, True
    )
    assert missing == []
    assert substitutions[0]["used"] == images[0]["path"]


def test_a_missing_model_is_never_replaced(tmp_path):
    chain = resize_chain(str(tmp_path / "absent.png"))
    chain["nodes"].append(node("model", MODEL, {"0": str(tmp_path / "absent.pth")}))
    _, _, missing = r.resolve_inputs(chain, SCHEMAS, bench_images(tmp_path, 2), True)
    assert [entry["input"] for entry in missing] == ["Model"]


def test_a_connected_file_input_is_left_alone(tmp_path):
    chain = resize_chain()
    chain["nodes"].append(node("text2", TEXT))
    chain["edges"].append(edge("text2", 0, "load", 0))
    _, substitutions, missing = r.resolve_inputs(chain, SCHEMAS, [], False)
    assert (substitutions, missing) == ([], [])


def test_bench_images_run_out_or_go_missing(tmp_path):
    chain = resize_chain()
    _, _, missing = r.resolve_inputs(chain, SCHEMAS, [], False)
    assert missing[0]["reason"] == "no bench image left"
    gone = {"path": str(tmp_path / "gone.jpg"), "sha256": "0" * 64}
    _, _, missing = r.resolve_inputs(chain, SCHEMAS, [gone], False)
    assert "missing" in missing[0]["reason"]


def test_a_changed_bench_image_fails(tmp_path):
    images = bench_images(tmp_path, 1)
    images[0]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="changed"):
        r.resolve_inputs(resize_chain(), SCHEMAS, images, False)


def test_prepare_skips_and_refuses_with_reasons(tmp_path):
    images = bench_images(tmp_path, 1)
    absent = r.Chain("absent", tmp_path / "absent.chn")
    assert r.prepare(absent, SCHEMAS, images)["status"] == "skipped"
    missing = r.Chain("missing", write_save(tmp_path / "m.chn", resize_chain("x.png")))
    record = r.prepare(missing, SCHEMAS, images)
    assert record["status"] == "skipped"
    assert record["missing"][0]["saved"] == "x.png"
    refused = r.Chain(
        "old", write_save(tmp_path / "o.chn", resize_chain(), migration=1)
    )
    record = r.prepare(refused, SCHEMAS, images)
    assert record["status"] == "refused"
    assert "migration" in record["error"]
    ready = r.Chain(
        "ready", write_save(tmp_path / "r.chn", resize_chain()), torch_cpu=True
    )
    record = r.prepare(ready, SCHEMAS, images)
    assert record["status"] == "ready"
    assert record["torch_cpu"] is True
    assert record["substitutions"][0]["used"] == images[0]["path"]


def test_the_owner_chains_are_the_ruled_four():
    assert [(c.name, c.path.name, c.torch_cpu, c.fallback) for c in r.CHAINS] == [
        ("resize", "resize.chn", False, False),
        ("Blend", "Blend.chn", False, False),
        ("Getnative", "Getnative.chn", False, False),
        ("default", "default.chn", True, True),
    ]
    assert all(c.path.is_relative_to(r.MODELS) for c in r.CHAINS)


# The launch's tree diff.


def test_tree_state_hashes_files_and_lists_empty_directories(tmp_path):
    (tmp_path / "a" / "empty").mkdir(parents=True)
    (tmp_path / "a" / "f.txt").write_bytes(b"x")
    assert r.tree_state(tmp_path) == {
        "a/": r.DIRECTORY,
        "a/empty/": r.DIRECTORY,
        "a/f.txt": hashlib.sha256(b"x").hexdigest(),
    }
    assert r.tree_state(tmp_path / "absent") == {}


def test_tree_diff_names_added_removed_and_modified():
    before = {"a": "1", "b": "2", "d/": r.DIRECTORY}
    after = {"a": "1", "b": "3", "c": "4"}
    assert r.tree_diff(before, after) == {
        "added": ["c"],
        "removed": ["d/"],
        "modified": ["b"],
    }


SITE = "python/python/Lib/site-packages/"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # Consult 6 P2 and P3.
        ("logs/", "allowed"),
        ("logs/main.log", "allowed"),
        ("backend-storage/", "allowed"),
        ("backend-storage/ffmpeg/x.dll", "allowed"),
        ("settings.json", "allowed"),
        ("Settings.JSON", "allowed"),
        # Chromium's profile in the root (Consult 8 R-i).
        ("Cache/", "allowed"),
        ("Cache/Cache_Data/data_0", "allowed"),
        ("Code Cache/js/index", "allowed"),
        ("DawnCache/data_0", "allowed"),
        ("GPUCache/", "allowed"),
        ("Local Storage/leveldb/LOG", "allowed"),
        ("Session Storage/000003.log", "allowed"),
        ("Network/Cookies", "allowed"),
        ("blob_storage/", "allowed"),
        ("Local State", "allowed"),
        ("local state", "allowed"),
        ("Preferences", "allowed"),
        ("electron-log-preload.js", "allowed"),
        # At the root only, and a file entry is not a directory (or the reverse).
        ("resources/Cache/data_0", "disallowed"),
        ("resources/logs/main.log", "disallowed"),
        ("python/Local State", "disallowed"),
        ("resources/src/settings.json", "disallowed"),
        (SITE + "GPUCache/x", "disallowed"),
        ("Cache", "disallowed"),
        ("Preferences/", "disallowed"),
        ("Preferences/x", "disallowed"),
        ("settings.json.tmp", "disallowed"),
        ("logs.txt", "disallowed"),
        ("Crashpad/", "disallowed"),
        # Bytecode alone under the backend (Consult 8 R-i). The runtime's Lib ships
        # compiled (Consult 10 D-9), so bytecode there is a module the build missed.
        ("resources/src/nodes/__pycache__/", "bytecode"),
        ("resources/src/nodes/__pycache__/x.cpython-314.pyc", "bytecode"),
        (SITE + "torch/__pycache__/x.cpython-314.pyc", "disallowed"),
        ("python/python/lib/SITE-PACKAGES/numpy/__PYCACHE__/", "disallowed"),
        ("resources/src/run.py", "disallowed"),
        ("resources/src/nodes/__pycache__/x.py", "disallowed"),
        ("resources/src/nodes/__pycache__/x.pyc.1234", "disallowed"),
        ("resources/src/nodes/__pycache__/sub/", "disallowed"),
        ("resources/src/nodes/__pycache__/sub/x.pyc", "disallowed"),
        (SITE, "disallowed"),
        (SITE + "x.pyc", "disallowed"),
        (SITE + "x.pyd", "disallowed"),
        ("python/python/lib/SITE-PACKAGES/new-1.0.dist-info/RECORD", "disallowed"),
        # Nor is bytecode in the stdlib or anywhere else.
        ("python/python/Lib/__pycache__/os.cpython-314.pyc", "disallowed"),
        ("python/python/Lib/asyncio/__pycache__/", "disallowed"),
        ("python/python/Lib/encodings/__pycache__/", "disallowed"),
        ("python/python/python.exe", "disallowed"),
    ],
)
def test_launch_change_follows_the_ruled_allow_list(path, expected):
    assert r.launch_change(path) == expected


def test_bytecode_is_admitted_under_the_backend_only():
    assert r.BYTECODE_ROOTS == ("resources/src/",)


@pytest.mark.parametrize(
    ("path", "module"),
    [
        (SITE + "sanic/__pycache__/app.cpython-314.pyc", "sanic.app"),
        (SITE + "numpy/__pycache__/__init__.cpython-314.pyc", "numpy"),
        (SITE + "__pycache__/typing_extensions.cpython-314.pyc", "typing_extensions"),
        (
            "python/python/Lib/asyncio/__pycache__/events.cpython-314.pyc",
            "asyncio.events",
        ),
        ("python/python/Lib/__pycache__/argparse.cpython-314.pyc", "argparse"),
        ("python/python/Lib/asyncio/__pycache__/", None),
        ("python/python/Lib/asyncio/events.py", None),
        ("resources/src/nodes/__pycache__/x.cpython-314.pyc", None),
        (SITE + "x.pyc", None),
    ],
)
def test_bytecode_module_names_a_runtime_pyc(path, module):
    assert r.bytecode_module(path) == module


def test_a_site_packages_bytecode_write_fails_and_names_its_module():
    # Consult 10 D-9: the build compiles every Lib module, so a launch writing one
    # means the build missed it; the gate fails and says which.
    written = SITE + "sanic/__pycache__/app.cpython-314.pyc"
    backend = "resources/src/nodes/__pycache__/x.cpython-314.pyc"
    tree = r.launch_tree(
        {
            "added": [SITE + "sanic/__pycache__/", written, backend],
            "removed": [],
            "modified": [],
        }
    )
    assert tree["pass"] is False
    assert tree["bytecode"] == [f"added: {backend}"]
    assert tree["disallowed"] == [
        f"added: {SITE}sanic/__pycache__/",
        f"added: {written} (module sanic.app, not compiled by the build)",
    ]


def test_the_root_allow_list_is_frozen_with_its_evidence():
    # An addition is a consult with its evidence (Consult 8 R-i): the 11 entries of
    # Chromium's profile out\chaiNNer-C-py311's root held after its launches, and
    # Consult 6's three.
    assert r.CHROMIUM_PROFILE == (
        "Cache/",
        "Code Cache/",
        "DawnCache/",
        "GPUCache/",
        "Local Storage/",
        "Session Storage/",
        "Network/",
        "blob_storage/",
        "Local State",
        "Preferences",
        "electron-log-preload.js",
    )
    assert r.RULED_ROOT == ("logs/", "backend-storage/", "settings.json")
    assert len(r.ROOT_ENTRIES) == 14


def test_launch_tree_fails_on_any_disallowed_change():
    allowed = {
        "added": ["logs/", "logs/main.log", "settings.json", "GPUCache/"],
        "removed": [],
        "modified": [],
    }
    assert r.launch_tree(allowed)["pass"] is True
    cache = "resources/src/a/__pycache__/"
    bytecode = {**allowed, "added": [*allowed["added"], cache]}
    assert r.launch_tree(bytecode)["pass"] is True
    assert r.launch_tree({**allowed, "modified": ["resources/src/run.py"]}) == {
        "allowed": [
            "added: logs/",
            "added: logs/main.log",
            "added: settings.json",
            "added: GPUCache/",
        ],
        "bytecode": [],
        "disallowed": ["modified: resources/src/run.py"],
        "pass": False,
    }
    removed = {**allowed, "removed": [SITE + "x.py"]}
    assert r.launch_tree(removed)["disallowed"] == [f"removed: {SITE}x.py"]
    assert r.launch_tree(removed)["pass"] is False


def test_the_app_inherits_the_cpu_settings_but_writes_bytecode(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("VK_LOADER_DRIVERS_DISABLE", "*")
    monkeypatch.setenv("TEMP", "elsewhere")
    env = r.launch_environment(tmp_path)
    assert "PYTHONDONTWRITEBYTECODE" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "-1"
    assert env["VK_LOADER_DRIVERS_DISABLE"] == "*"
    assert env["TEMP"] == env["TMP"] == str(tmp_path)
    assert env["PIP_REQUIRE_VIRTUALENV"] == "1"
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"  # The tool's own is kept.
    # The summary records what the app ran with.
    assert {name: env.get(name) for name in r.LAUNCH_ENVIRONMENT} == {
        "TEMP": str(tmp_path),
        "TMP": str(tmp_path),
        "PIP_REQUIRE_VIRTUALENV": "1",
        "PYTHONDONTWRITEBYTECODE": None,
        "CUDA_VISIBLE_DEVICES": "-1",
        "VK_LOADER_DRIVERS_DISABLE": "*",
    }


# The launch's logs.

PYTHON = Path(r"C:\pkg\python\python\python.exe")
STARTED = [
    f"[2026-10-06 21:00:00.000] [info]  {r.VERSION_LINE}",
    "[2026-10-06 21:00:00.001] [info]  Attempting to check for a port...",
    "[2026-10-06 21:00:00.002] [info]  Attempting to check integrated Python env...",
    rf"[2026-10-06 21:00:00.003] [info]  Final Python binary: {PYTHON}",
    "[2026-10-06 21:00:00.004] [info]  {",
    r"  python: 'C:\\pkg\\python\\python\\python.exe',",
    "  version: '3.14.8'",
    "}",
    "[2026-10-06 21:00:00.005] [info]  Attempting to spawn backend...",
    "[2026-10-06 21:00:05 +0800] [100] [INFO] Starting setup...",
    "[2026-10-06 21:00:05 +0800] [100] [INFO] Checking dependencies...",
    "[2026-10-06 21:00:05 +0800] [100] [INFO] Checking dependencies for PyTorch...",
    "[2026-10-06 21:00:05 +0800] [100] [INFO] No dependencies to install. Skipping worker restart.",
    "[2026-10-06 21:00:05 +0800] [100] [INFO] Done checking dependencies...",
    "[2026-10-06 21:00:05 +0800] [100] [INFO] Backend almost ready...",
    "[2026-10-06 21:00:06 +0800] [100] [INFO] Done.",
]
CLOSED = [
    "[2026-10-06 21:00:20.000] [info]  Attempting to kill backend...",
    "[2026-10-06 21:00:20.001] [info]  Cleaning up temp folders...",
    "[2026-10-06 21:00:20.002] [error] Python subprocess exited with code null and signal SIGTERM",
]


def checks(started: list[str], closed: list[str] = CLOSED) -> dict:
    before = "\n".join(started)
    return r.log_checks("\n".join(started + closed), before, PYTHON, "3.14.8")


def test_a_clean_start_passes_every_log_check():
    result = checks(STARTED)
    assert {key: result[key]["pass"] for key in r.LOG_CHECKS} == dict.fromkeys(
        r.LOG_CHECKS, True
    )
    assert result["integrated_python"]["found"] == [str(PYTHON)]


def test_the_version_line_must_be_chainner_c_s():
    started = [
        line.replace(r.VERSION_LINE, "chaiNNer Version: 0.25.1") for line in STARTED
    ]
    assert checks(started)["version_line"]["pass"] is False


@pytest.mark.parametrize(
    "line",
    [
        f"[2026-10-06 21:00:00.006] [error] Error: {independent_ui.PYTHON_MISSING}",
        "[2026-10-06 21:00:00.006] [info]  Downloading integrated Python...",
        "[2026-10-06 21:00:00.006] [info]  Extracting integrated Python...",
    ],
)
def test_the_redownload_branch_or_a_download_fails_the_python_check(line):
    assert checks([*STARTED, line])["integrated_python"]["pass"] is False


def test_another_python_or_version_fails_the_python_check():
    other = [line.replace(r"C:\pkg", r"C:\elsewhere") for line in STARTED]
    assert checks(other)["integrated_python"]["pass"] is False
    older = [line.replace("3.14.8", "3.11.5") for line in STARTED]
    assert checks(older)["integrated_python"]["pass"] is False


def test_the_host_must_start_and_not_exit_before_the_close():
    assert checks(STARTED[:-1])["host_start"]["pass"] is False, "no Done."
    exited = [*STARTED, CLOSED[2]]
    result = checks(exited, [])
    assert result["host_start"]["pass"] is False
    assert result["host_start"]["exited_before_close"] == [CLOSED[2]]


@pytest.mark.parametrize(
    "line",
    [
        "[2026-10-06 21:00:05 +0800] [100] [INFO] Collecting chainner-pip",
        "Backend: Successfully installed chainner-pip-23.2.0",
        "[2026-10-06 21:00:05 +0800] [100] [INFO] Progress: Collecting torch... 0.1 None",
        "[2026-10-06 21:00:05 +0800] [100] [ERROR] Error installing dependencies: x",
    ],
)
def test_any_install_fails_the_dependency_check(line):
    result = checks([*STARTED, line])
    assert result["dependency_check"]["pass"] is False
    assert result["dependency_check"]["install_lines"] == [line]


def test_a_dependency_check_that_did_not_finish_fails():
    started = [line for line in STARTED if "No dependencies to install" not in line]
    result = checks(started)["dependency_check"]
    assert result["missing_lines"] == [r.DEPENDENCY_CHECK[1]]
    assert result["pass"] is False


def test_new_log_text_reads_what_each_log_gained(tmp_path):
    assert r.log_offsets(tmp_path / "logs") == {}
    assert r.new_log_text(tmp_path / "logs", {}) == ""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "main.log").write_bytes(b"old\n")
    (logs / "notes.txt").write_bytes(b"not a log")
    offsets = r.log_offsets(logs)
    assert offsets == {"main.log": 4}
    with (logs / "main.log").open("ab") as stream:
        stream.write(b"new\n")
    (logs / "renderer.log").write_bytes(b"fresh\n")
    assert r.new_log_text(logs, offsets) == "new\n\nfresh\n"
    # A rotated (shrunk) log is read whole.
    (logs / "main.log").write_bytes(b"x\n")
    assert r.new_log_text(logs, offsets).startswith("x\n")


# The dependency manager.


@pytest.mark.parametrize(
    ("pin", "installed", "satisfied"),
    [
        ("2.14.1+cu132", "2.14.1+cu132", True),
        ("2.14.1", "2.14.1+cu132", True),
        ("1.0.20260526", "1.0.20260526", True),
        ("0.4.2", "0.5.0", True),
        ("12.3.0", "12.2.9", False),
        ("1.30.0", None, False),
    ],
)
def test_a_pin_is_satisfied_when_installed_and_not_older(pin, installed, satisfied):
    packages = [{"id": "p", "dependencies": [{"pypiName": "x", "version": pin}]}]
    report = r.dependency_report(
        packages, {} if installed is None else {"x": installed}
    )
    assert report["pins"][0]["satisfied"] is satisfied
    assert report["pass"] is satisfied


def test_no_pins_is_not_a_pass():
    assert r.dependency_report([{"id": "p", "dependencies": []}], {})["pass"] is False


def test_coerce_version_reads_the_first_numbers():
    assert r.coerce_version("2.14.1+cu132") == (2, 14, 1)
    assert r.coerce_version("v3") == (3, 0, 0)
    assert r.coerce_version("1.2.3.4") == (1, 2, 3)
    assert r.coerce_version("none") is None


# The comparisons and the summary.


def run(**overrides: object) -> dict:
    record = {
        "directory": "d",
        "seconds": 1.0,
        "response": {"type": "success"},
        "request": [{"id": "n"}],
        "saves": 2,
        "files": ["a.png", "b.png"],
        "sse_contract": {"controls": [], "final_state": []},
    }
    return {**record, **overrides}


def image(name: str, exact: bool = True, shape: bool = True) -> dict:
    return {
        "name": name,
        "same_shape_dtype": shape,
        "pass": exact,
        "max_channel_difference": 0 if exact else 3,
        "differing_components": 0 if exact else 17,
    }


def test_equal_runs_pass_every_comparison():
    result = r.run_comparison(run(), run(), [image("a.png"), image("b.png")], False)
    assert (result["run_pass"], result["save_pass"], result["sse_contract_equal"]) == (
        True,
        True,
        True,
    )
    assert result["pixel_differences"] == []


def test_pixel_differences_fail_save_unless_torch_cpu():
    images = [image("a.png"), image("b.png", exact=False)]
    plain = r.run_comparison(run(), run(), images, False)
    assert plain["save_pass"] is False
    assert plain["pixel_differences"] == [
        {"name": "b.png", "max_channel_difference": 3, "differing_components": 17}
    ]
    torch = r.run_comparison(run(), run(), images, True)
    assert torch["save_pass"] is True
    assert torch["pixel_differences"] == plain["pixel_differences"]
    shapes = r.run_comparison(
        run(), run(), [image("a.png"), image("b.png", False, False)], True
    )
    assert shapes["save_pass"] is False, "a shape difference is never PyTorch numerics"


@pytest.mark.parametrize(
    ("port", "oracle", "failed"),
    [
        (run(files=["a.png"]), run(), "save_pass"),
        (run(saves=3, files=["a.png", "b.png"]), run(saves=3), "save_pass"),
        (run(saves=0, files=[]), run(saves=0, files=[]), "save_pass"),
        (run(response={"type": "error"}), run(), "run_pass"),
        (run(request=[{"id": "m"}]), run(), "run_pass"),
        (run(sse_contract={"controls": [1]}), run(), "sse_contract_equal"),
    ],
)
def test_a_differing_run_fails_its_comparison(port, oracle, failed):
    images = [image(name) for name in sorted(set(port["files"]) & set(oracle["files"]))]
    assert r.run_comparison(port, oracle, images, False)[failed] is False


def test_a_failed_run_is_not_compared():
    result = r.run_comparison({"error": "boom"}, run(), [], False)
    assert result == {"errors": ["boom"]}
    assert r.compared(result) is False
    assert r.compared({"error": "compare failed"}) is False
    assert r.compared(
        r.run_comparison(run(), run(), [image("a.png"), image("b.png")], False)
    )


def test_the_ui_state_after_a_stop():
    late = [
        {"event": "node-progress", "data": {}},
        {"event": "node-finish", "data": {}},
    ]
    assert r.ui_state(late, "ready", None)["pass"] is True
    assert r.ui_state(late, "ready", None)["late_event_kinds"] == [
        "node-finish",
        "node-progress",
    ]
    started = [{"event": "node-start", "data": {"nodeId": "n"}}]
    assert r.ui_state(started, "ready", None)["pass"] is False
    assert (
        r.ui_state([{"event": "chain-start", "data": {}}], "ready", None)["pass"]
        is False
    )
    assert r.ui_state([], "running", None)["pass"] is False
    assert r.ui_state([], "ready", "reset")["pass"] is False


def stopped(**overrides: object) -> dict:
    record = {
        "kill_response": {"type": "success"},
        "run_response": {"type": "success"},
        "interrupted": True,
        "ui_state": {"pass": True},
    }
    return {**record, **overrides}


def test_the_stop_comparison():
    rerun = r.run_comparison(run(), run(), [image("a.png"), image("b.png")], True)
    assert r.stop_comparison(stopped(), stopped(), rerun)["pass"] is True
    for change in (
        {"kill_response": {"type": "no-executor"}},
        {"interrupted": False},
        {"ui_state": {"pass": False}},
    ):
        assert r.stop_comparison(stopped(**change), stopped(), rerun)["pass"] is False
    assert r.stop_comparison(stopped(), stopped(), {"errors": ["x"]})["pass"] is False
    failed = r.stop_comparison({"error": "kill missed"}, stopped(), rerun)
    assert failed == {"errors": ["kill missed"], "pass": False}


def test_the_longest_chain_is_the_slowest_completed_run():
    runs = {"a": {"seconds": 2.0}, "b": {"seconds": 9.0}, "c": {"error": "x"}}
    assert r.longest_chain(runs) == "b"
    assert r.longest_chain({"c": {"error": "x"}}) is None


def passing_summary() -> dict:
    return {
        "launch": {
            "start": {"started": True},
            "appdata": {"unchanged": True},
            "package_tree": {"pass": True},
            "log": {key: {"pass": True} for key in r.LOG_CHECKS},
            "close": {"main_window": True, "no_survivors": True},
        },
        "dependencies": {"pass": True},
        "chains": {
            "a": {
                "status": "compared",
                "comparison": {
                    "run_pass": True,
                    "save_pass": True,
                    "sse_contract_equal": True,
                },
            },
            "b": {"status": "skipped"},
        },
        "stop": {"comparison": {"pass": True}},
        "hosts": {
            "port": dict.fromkeys(r.HOST_CLEAN, True),
            "oracle": dict.fromkeys(r.HOST_CLEAN, True),
            "package_tree": {"pass": True},
        },
    }


def test_every_gate_holds_on_a_passing_summary():
    gates = r.gates(passing_summary())
    assert all(gates.values())
    assert set(gates) == {
        *LAUNCH_GATES,
        "dependencies",
        "chains_run",
        "save_outputs",
        "sse_contract",
        "stop",
        "hosts_clean",
    }


LAUNCH_GATES = {
    "launch_started",
    "launch_appdata_unchanged",
    "launch_package_tree",
    *(f"launch_log_{key}" for key in r.LOG_CHECKS),
    "launch_closed",
}


@pytest.mark.parametrize(
    ("change", "failing"),
    [
        (lambda s: s.pop("launch"), LAUNCH_GATES),
        (
            lambda s: s["launch"]["appdata"].update(unchanged=False),
            {"launch_appdata_unchanged"},
        ),
        (lambda s: s["launch"]["log"].pop("host_start"), {"launch_log_host_start"}),
        (lambda s: s["launch"]["close"].update(no_survivors=False), {"launch_closed"}),
        (lambda s: s["launch"]["close"].update(main_window=False), {"launch_closed"}),
        (lambda s: s.update(dependencies={"error": "x"}), {"dependencies"}),
        (lambda s: s["chains"]["b"].update(status="refused"), {"chains_run"}),
        (lambda s: s["chains"]["b"].update(status="failed"), {"chains_run"}),
        (
            lambda s: s["chains"]["a"].update(status="skipped"),
            {"chains_run", "save_outputs", "sse_contract"},
        ),
        (
            lambda s: s["chains"]["a"]["comparison"].update(run_pass=False),
            {"chains_run"},
        ),
        (
            lambda s: s["chains"]["a"]["comparison"].update(save_pass=False),
            {"save_outputs"},
        ),
        (
            lambda s: s["chains"]["a"]["comparison"].update(sse_contract_equal=False),
            {"sse_contract"},
        ),
        (lambda s: s["stop"].pop("comparison"), {"stop"}),
        (
            lambda s: s["hosts"]["oracle"].update(interpreter_unchanged=False),
            {"hosts_clean"},
        ),
        (lambda s: s["hosts"].update(package_tree={"pass": False}), {"hosts_clean"}),
        (lambda s: s["hosts"].pop("package_tree"), {"hosts_clean"}),
    ],
)
def test_each_gate_fails_on_its_own_record(change, failing):
    summary = passing_summary()
    change(summary)
    gates = r.gates(summary)
    assert {name for name, held in gates.items() if not held} == failing


def test_an_empty_summary_fails_every_gate():
    assert not any(r.gates({}).values())


# The window and process reads the launch makes (no app is started).


def test_a_process_without_a_window_has_no_main_window():
    # The test process is a console program: its console window is the host's.
    assert r.main_window(os.getpid()) is None


def test_running_from_finds_a_process_by_its_executable_directory(tmp_path):
    executable = Path(psutil.Process().exe())
    running, _ = r.running_from([executable.parent])
    assert f"{os.getpid()} {executable}" in running
    assert r.running_from([tmp_path]) == ([], [])


INSTALLED = Path(os.environ["APPDATA"], "chaiNNer")
PYTHON_EXE = str(INSTALLED / "python/python/python.exe")
CREATED = 1791309527.25  # 2026-10-06T17:58:47.25Z


class FakeProcess:
    """A process_iter entry: info as psutil fills it, None where access is denied
    (process_iter's ad_value)."""

    def __init__(
        self,
        pid: int,
        exe: str,
        num_threads: int | None,
        create_time: float | None = CREATED,
    ):
        self.info = {
            "pid": pid,
            "exe": exe,
            "num_threads": num_threads,
            "create_time": create_time,
        }


def fake_processes(monkeypatch, *processes):
    def process_iter(attrs):
        assert set(attrs) == {"pid", "exe", "num_threads", "create_time"}
        return iter(processes)

    monkeypatch.setattr(r.psutil, "process_iter", process_iter)


def test_an_exited_process_object_is_skipped_and_recorded(monkeypatch):
    # An exited process's object survives while another process holds a handle
    # to it: 0 threads, so it can neither execute nor write.
    fake_processes(
        monkeypatch,
        FakeProcess(35532, PYTHON_EXE, 0),
        FakeProcess(35533, PYTHON_EXE, 0, create_time=None),
        FakeProcess(7, r"C:\Windows\explorer.exe", 40),
    )
    assert r.running_from([INSTALLED]) == (
        [],
        [
            {
                "pid": 35532,
                "exe": PYTHON_EXE,
                "created": "2026-10-06T17:58:47.250000+00:00",
            },
            {"pid": 35533, "exe": PYTHON_EXE, "created": None},
        ],
    )


@pytest.mark.parametrize(
    "num_threads",
    [1, 12, None],
    ids=["suspended-one-thread", "running", "thread-count-access-denied"],
)
def test_a_process_with_threads_or_an_unreadable_count_is_refused(
    monkeypatch, num_threads
):
    # A suspended process keeps its threads; a count that cannot be read is
    # taken as running (conservative).
    fake_processes(monkeypatch, FakeProcess(35532, PYTHON_EXE, num_threads))
    assert r.running_from([INSTALLED]) == ([f"35532 {PYTHON_EXE}"], [])


def test_an_empty_job_has_no_active_process():
    job = OwnedJob()
    try:
        assert r.active_processes(job) == 0
    finally:
        job.close()
    with pytest.raises(RuntimeError, match="already closed"):
        r.active_processes(job)
