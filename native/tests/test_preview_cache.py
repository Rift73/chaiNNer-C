"""LargeImage previews of unchanged images (hunt-misc 8): the engine encodes an image's previews once and reuses them
(keyed by the image's storage, held weakly), and /sse splices a preview's base64 data URL into the event's text instead
of escaping it, byte-identical to json.dumps."""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytest
from engine_differential import execute, mismatches
from verify_runtime import make_node

import api
from chain.json import JsonNode, parse_json
from chain.optimize import optimize
from events import PreviewUrl, event_text
from nodes.impl.native_graph import graph

BACKEND = Path(__file__).resolve().parents[2] / "backend" / "src"


@pytest.fixture(scope="module")
def schemas():
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
    yield {
        schema: {"inputs": [i.to_dict() for i in data.inputs], "kind": data.kind}
        for schema, (data, _) in api.registry.nodes.items()
    }
    if previous is None:
        del main.__file__
    else:
        main.__file__ = previous


def write_image(path: Path, seed: int) -> None:
    small = (np.random.default_rng(seed).random((12, 16, 3)) * 255).astype(np.uint8)
    assert cv2.imwrite(
        str(path), cv2.resize(small, (320, 240), interpolation=cv2.INTER_CUBIC)
    )


def load_previews(kind: str, schemas, image: Path, tmp_path: Path):
    """A /run of Load Image alone: its result and its broadcast's data."""
    chain = parse_json(
        cast(
            "list[JsonNode]",
            [make_node(schemas, "load", "chainner:image:load", {0: str(image)})],
        )
    )
    optimize(chain)
    result = execute(kind, chain, storage=tmp_path / "storage")
    assert result.error is None
    broadcasts = [
        e["data"]
        for e in result.events
        if e["event"] == "node-broadcast" and e["data"]["nodeId"] == "load"
    ]
    assert len(broadcasts) == 1
    return chain, result, broadcasts[0]


def test_an_unchanged_image_reuses_its_previews(schemas, tmp_path):
    # The second run's Load Image is a memo hit, the same image: its previews come from the cache, the same data.
    image = tmp_path / "picture.png"
    write_image(image, 1)
    _, _, first = load_previews("native", schemas, image, tmp_path)
    before = graph().preview_cache_stats()
    chain, engine, second = load_previews("native", schemas, image, tmp_path)
    after = graph().preview_cache_stats()
    assert after["hits"] == before["hits"] + 1
    assert second == first
    _, oracle, expected = load_previews("python", schemas, image, tmp_path)
    assert second == expected
    assert mismatches("cached preview", chain, oracle, engine, tmp_path) == []


def test_an_equal_shape_different_image_is_encoded(schemas, tmp_path):
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    write_image(first, 2)
    write_image(second, 3)
    _, _, a = load_previews("native", schemas, first, tmp_path)
    before = graph().preview_cache_stats()
    _, _, b = load_previews("native", schemas, second, tmp_path)
    after = graph().preview_cache_stats()
    assert after["hits"] == before["hits"]
    assert after["misses"] == before["misses"] + 1
    assert a["data"]["0"]["width"] == b["data"]["0"]["width"]
    assert a["data"]["0"]["previews"] != b["data"]["0"]["previews"]
    # Replaced on disk by other content of the same shape: a new image, encoded again.
    shutil.copyfile(second, first)
    _, _, replaced = load_previews("native", schemas, first, tmp_path)
    assert graph().preview_cache_stats()["misses"] == after["misses"] + 1
    assert replaced["data"]["0"]["previews"] == b["data"]["0"]["previews"]


def broadcast(previews: list[dict[str, Any]], name: str = "View") -> dict[str, Any]:
    return {
        "event": "node-broadcast",
        "data": {
            "nodeId": name,
            "data": {
                "0": {"previews": previews, "height": 240, "width": 320, "channels": 3},
                "1": None,
            },
            "types": {"0": {"type": "named", "name": "Image"}},
            "sequenceTypes": {},
        },
    }


def test_event_text_splices_preview_urls_as_json_dumps_writes_them(schemas):
    from nodes.properties.outputs.numpy_outputs import LargeImageOutput

    image = (np.random.default_rng(4).random((240, 320, 3))).astype(np.float32)
    data = LargeImageOutput().get_broadcast_data(image)
    previews = cast("list[dict[str, Any]]", data["previews"])
    assert all(isinstance(p["url"], PreviewUrl) for p in previews)
    for message in (
        broadcast(previews),
        broadcast(
            previews, name="\x00preview\x00"
        ),  # a string equal to the placeholder: json.dumps alone
        broadcast(
            [{"width": 1, "height": 1, "url": "data:image/png;base64,AAAA"}]
        ),  # a plain str: escaped as usual
        {"event": "node-finish", "data": {"nodeId": "a", "executionTime": 0.5}},
    ):
        expected = f"event: {message['event']}\ndata: {json.dumps(message['data'])}\n\n"
        assert event_text(cast("Any", message)) == expected


def test_an_entry_goes_with_its_image(schemas, tmp_path):
    # An entry holds its image weakly: once the image is freed (here the memo's copy dropped), the next entry kept
    # removes it.
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    write_image(first, 6)
    write_image(second, 7)
    graph().memo_clear()
    load_previews("native", schemas, first, tmp_path)
    kept = graph().preview_cache_stats()
    graph().memo_clear()  # the first image's last reference
    load_previews("native", schemas, second, tmp_path)
    after = graph().preview_cache_stats()
    assert after["misses"] == kept["misses"] + 1
    assert after["entries"] <= kept["entries"]
