"""RDUAL engine sets (nodes/impl/tensorrt/rdual_engine_set.py) and their Engine Cache
entries; stand-in engine bytes, no TensorRT."""

from __future__ import annotations

import dataclasses
import json
from collections import OrderedDict
from pathlib import Path

import pytest

from nodes.impl.compressed_file import ZSTD_HEADER
from nodes.impl.tensorrt import build_cache, rdual_engine_set
from nodes.impl.tensorrt.build_cache import SUFFIX, cached_engine
from nodes.impl.tensorrt.model import TemporalInfo, TensorRTEngine, TensorRTEngineInfo
from nodes.impl.tensorrt.rdual_engine_set import EngineSetError, pack, unpack

PARTS = ("prefix", "revcore", "fwdcore", "flow")
INFO = TensorRTEngineInfo(
    precision="mixed",
    input_channels=3,
    output_channels=3,
    scale=2,
    gpu_architecture="sm_120",
    tensorrt_version="11.4.0",
    has_dynamic_shapes=False,
    min_shape=(1, 3, 384, 384),
    opt_shape=(1, 3, 384, 384),
    max_shape=(1, 3, 384, 384),
    temporal=TemporalInfo("RDUAL", 384, 384, "c128_h192_w192", PARTS),
)


def parts() -> dict[str, bytes]:
    return {name: bytes([i]) * (1000 + 37 * i) for i, name in enumerate(PARTS)}


def test_a_set_unpacks_to_its_info_and_parts():
    engine = pack(parts(), INFO)

    assert rdual_engine_set.is_engine_set(engine.bytes)
    assert engine.info == INFO
    info, read = unpack(engine.bytes)
    assert info == INFO and read == parts()
    assert list(read) == list(PARTS)
    assert rdual_engine_set.engine_set(engine.bytes).info == INFO


def test_parts_must_be_the_infos_in_order():
    with pytest.raises(ValueError, match="in order"):
        pack(dict(reversed(parts().items())), INFO)
    with pytest.raises(ValueError, match="in order"):
        pack(parts(), dataclasses.replace(INFO, temporal=None))


def test_another_format_version_is_refused_by_name():
    data = bytearray(pack(parts(), INFO).bytes)
    start = len(rdual_engine_set.MAGIC) + 8
    end = start + int.from_bytes(data[len(rdual_engine_set.MAGIC) : start], "little")
    header = json.loads(data[start:end])
    header["format"] = rdual_engine_set.FORMAT_VERSION + 1
    encoded = json.dumps(header).encode()
    other = (
        rdual_engine_set.MAGIC
        + len(encoded).to_bytes(8, "little")
        + encoded
        + bytes(data[end:])
    )

    with pytest.raises(
        EngineSetError, match=r"format version 2.*Build the engine again"
    ):
        unpack(other)


@pytest.mark.parametrize("damage", ["truncated", "longer", "flipped", "magic"])
def test_a_damaged_set_is_refused_before_tensorrt_sees_it(damage: str):
    data = bytearray(pack(parts(), INFO).bytes)
    if damage == "truncated":
        del data[-1]
    elif damage == "longer":
        data += b"\0"
    elif damage == "flipped":
        data[-500] ^= 0xFF
    else:
        data[0] ^= 0xFF

    with pytest.raises(EngineSetError):
        unpack(bytes(data))


def test_an_image_engine_is_not_a_set():
    assert not rdual_engine_set.is_engine_set(b"\x00" * 64)
    with pytest.raises(EngineSetError, match="not an RDUAL engine set"):
        unpack(b"\x00" * 64)


@pytest.fixture
def memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_cache, "_memory", OrderedDict())
    monkeypatch.setattr(build_cache, "_building", {})


def test_a_set_is_one_cache_entry_kept_in_zstandard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, memory: None
):
    built: list[TensorRTEngine] = []

    def build() -> TensorRTEngine:
        built.append(pack(parts(), INFO))
        return built[-1]

    cached_engine("set", build, str(tmp_path))
    (path,) = tmp_path.iterdir()
    assert path.name == f"set{SUFFIX}"
    record = path.read_bytes()
    assert ZSTD_HEADER in record
    assert len(record) < len(built[0].bytes)
    monkeypatch.setattr(build_cache, "_memory", OrderedDict())  # a restart

    read = cached_engine("set", build, str(tmp_path))

    assert len(built) == 1
    assert read.bytes == built[0].bytes and read.info == INFO
    assert unpack(read.bytes)[1] == parts()


def test_an_image_engines_record_is_as_before(tmp_path: Path, memory: None):
    image = dataclasses.replace(INFO, temporal=None)
    cached_engine("image", lambda: TensorRTEngine(b"engine", image), str(tmp_path))
    record = (tmp_path / f"image{SUFFIX}").read_bytes()
    # after the magic line and the header's length
    start = record.index(b"\n") + 1 + 8
    end = start + int.from_bytes(record[start - 8 : start], "little")
    header = json.loads(record[start:end])

    assert "temporal" not in header["info"] and "compression" not in header
    assert record[end:] == b"engine"


def test_a_set_s_info_names_its_flow_recipe():
    temporal = INFO.temporal
    assert temporal is not None
    full = dataclasses.replace(
        INFO, temporal=dataclasses.replace(temporal, flow_precision="full")
    )

    read = unpack(pack(parts(), full).bytes)[0].temporal
    assert read is not None and read.flow_precision == "full"
    # a set written before the field reads as the mixed recipe it was built with
    before = dataclasses.asdict(temporal)
    del before["flow_precision"]
    old = rdual_engine_set.info_from_dict(
        {**dataclasses.asdict(INFO), "temporal": before}
    )
    assert old.temporal is not None and old.temporal.flow_precision == "mixed"
