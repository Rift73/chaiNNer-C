"""An RDUAL model's TensorRT engines for one input size as one container: the bytes of a
`TensorRTEngine` whose info has `temporal` set, and so of an Engine Cache entry, a file
Save Engine writes and Load Engine reads (compressed or not, compressed_file.py).

Layout: the magic line, the header's length (8 bytes, little-endian), the header (JSON:
the format version, the engine's info, and each part's name, size and SHA-256), then the
parts' bytes in the header's order. `unpack` checks all of it before anything reaches
TensorRT; a container of another format version is refused with a message saying so.
`pack_container` and `unpack_container` are that layout for any magic and header, which
RDUAL's ONNX bundle shares.

This module must not import TensorRT (the tests have none)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

from .model import TemporalInfo, TensorRTEngine, TensorRTEngineInfo

MAGIC = b"chaiNNer-C RDUAL engine set\n"
# The layout `pack_container` writes: a new one takes the next version.
FORMAT_VERSION = 1
_LENGTH = 8
_WHAT = "RDUAL engine set"
_REMEDY = "Build the engine again."


class EngineSetError(ValueError):
    """Bytes that are not a container this version reads, or a damaged one."""


def is_engine_set(data: bytes) -> bool:
    return data.startswith(MAGIC)


def pack_container(
    magic: bytes, header: dict[str, Any], parts: dict[str, bytes]
) -> bytes:
    """`parts` (name -> bytes, in order) behind `magic` and `header`, to which the
    format version and the parts' names, sizes and SHA-256 are added."""
    encoded = json.dumps(
        {
            "format": FORMAT_VERSION,
            **header,
            "parts": [
                {
                    "name": name,
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
                for name, data in parts.items()
            ],
        }
    ).encode()
    body = b"".join(parts.values())
    return magic + len(encoded).to_bytes(_LENGTH, "little") + encoded + body


def unpack_container(
    magic: bytes, data: bytes, what: str, part: str, remedy: str
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """The header and parts of `pack_container`'s bytes, each part checked by size and
    SHA-256. Messages name the container `what` (after "an": "RDUAL engine set"), a
    part as "<what>'s <name> <part>", and end a version mismatch with `remedy`."""
    if not data.startswith(magic):
        raise EngineSetError(f"This is not an {what}")
    damaged = f"This {what} is damaged"
    start = len(magic) + _LENGTH
    end = start + int.from_bytes(data[len(magic) : start], "little")
    try:
        header = json.loads(data[start:end])
    except ValueError as error:
        raise EngineSetError(damaged) from error
    if not isinstance(header, dict) or header.get("format") != FORMAT_VERSION:
        version = header.get("format") if isinstance(header, dict) else None
        raise EngineSetError(
            f"This {what} has format version {version}; this chaiNNer-C"
            f" reads version {FORMAT_VERSION}. {remedy}"
        )
    try:
        entries = [(p["name"], int(p["size"]), p["sha256"]) for p in header["parts"]]
    except (KeyError, TypeError, ValueError) as error:
        raise EngineSetError(damaged) from error
    offset = end
    if offset + sum(size for _, size, _ in entries) != len(data):
        raise EngineSetError(f"{damaged} (its size is wrong)")
    parts: dict[str, bytes] = {}
    for name, size, digest in entries:
        data_part = data[offset : offset + size]
        if hashlib.sha256(data_part).hexdigest() != digest:
            raise EngineSetError(f"This {what}'s {name} {part} is damaged")
        parts[name] = data_part
        offset += size
    return header, parts


def pack(parts: dict[str, bytes], info: TensorRTEngineInfo) -> TensorRTEngine:
    """The engine set of `parts` (name -> serialized engine, in `info.temporal.parts`'
    order)."""
    temporal = info.temporal
    if temporal is None or list(parts) != list(temporal.parts):
        raise ValueError("An engine set's parts are its temporal info's, in order")
    data = pack_container(MAGIC, {"info": dataclasses.asdict(info)}, parts)
    return TensorRTEngine(data, info)


def info_from_dict(info: dict[str, Any]) -> TensorRTEngineInfo:
    """A TensorRTEngineInfo from its `dataclasses.asdict` (JSON) form."""
    values = dict(info)
    for name in ("min_shape", "opt_shape", "max_shape"):
        if values.get(name) is not None:
            values[name] = tuple(values[name])
    temporal = values.get("temporal")
    if temporal is not None:
        values["temporal"] = TemporalInfo(
            architecture=str(temporal["architecture"]),
            height=int(temporal["height"]),
            width=int(temporal["width"]),
            plugin_key=str(temporal["plugin_key"]),
            parts=tuple(str(part) for part in temporal["parts"]),
            flow_precision=str(temporal.get("flow_precision", "mixed")),
        )
    return TensorRTEngineInfo(**values)


def unpack(data: bytes) -> tuple[TensorRTEngineInfo, dict[str, bytes]]:
    """The info and parts of an engine set, each part checked by size and SHA-256."""
    header, parts = unpack_container(MAGIC, data, _WHAT, "engine", _REMEDY)
    try:
        info = info_from_dict(header["info"])
    except (KeyError, TypeError, ValueError) as error:
        raise EngineSetError(f"This {_WHAT} is damaged") from error
    if info.temporal is None or list(parts) != list(info.temporal.parts):
        raise EngineSetError(f"This {_WHAT} is damaged")
    return info, parts


def engine_set(data: bytes) -> TensorRTEngine:
    """The engine of an engine set's container (read from a file, for one), checked."""
    info, _ = unpack(data)
    return TensorRTEngine(data, info)


__all__ = [
    "FORMAT_VERSION",
    "MAGIC",
    "EngineSetError",
    "engine_set",
    "info_from_dict",
    "is_engine_set",
    "pack",
    "pack_container",
    "unpack",
    "unpack_container",
]
