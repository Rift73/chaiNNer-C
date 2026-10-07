"""Object adapters for the native NCNN parser, optimizer and serializer.

The compiled extension owns graph traversal and weight computation. Python
retains class identities, file objects and registration metadata.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any

from .native import lib


@lru_cache(maxsize=1)
def _api():
    # Load the explicitly located dependency before Windows resolves the .pyd.
    lib()
    return importlib.import_module("nodes.impl._chainner_graph")


def _types():
    return importlib.import_module("nodes.impl.ncnn.model")


def param_get(collection: Any, pid: int):
    return _api().ncnn_param_get(collection, pid, _types())


def param_set(collection: Any, pid: int, value: Any) -> None:
    _api().ncnn_param_set(collection, pid, value, _types())


def param_string(collection: Any) -> str:
    return _api().ncnn_param_string(collection, _types())


def parse_layer(line: str):
    return _api().ncnn_parse_layer(line, _types())


def add_weight(layer: Any, name: str, data: Any, tag: bytes) -> int:
    return _api().ncnn_add_weight(layer, name, data, tag, _types())


def load_weights(stream: Any, op: str, layer: Any):
    return _api().ncnn_load_weights(stream, op, layer, _types())


def read_model(parameters: Any, weights: Any):
    return _api().ncnn_read_model(parameters, weights, _types())


def serialize_weights(model: Any) -> bytes:
    return _api().ncnn_serialize_weights(model)


def write_param(model: Any) -> str:
    return _api().ncnn_write_param(model, _types())


def broadcast_data(model: Any):
    return _api().ncnn_broadcast_data(model, _types())


def get_nf(layer: Any):
    return _api().ncnn_get_nf(layer)


def optimize(model: Any, name: str | None = None) -> None:
    if name is None:
        _api().ncnn_optimize(model, _types())
    else:
        _api().ncnn_optimizer_pass(model, _types(), name)


def interp_layers(a: Any, b: Any, alpha: float):
    return _api().ncnn_interp_layers(a, b, alpha, _types())


def interpolate(a: Any, b: Any, alpha: float):
    return _api().ncnn_interpolate(a, b, alpha, _types())
