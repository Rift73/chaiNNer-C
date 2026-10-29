"""The DiT's BF16 copy: a BF16 load reads the same weights it converted before, and the
copy is made once, checked before each use and made again on any difference."""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from safetensors.torch import load_file, save_file

from api import NodeContext
from nodes.impl.pytorch import vosr2_cache, vosr2_download, vosr2_engine_cache
from nodes.impl.pytorch.vosr2 import BundlePaths, load_weights
from vendor.vosr2.lightningdit import LightningDiT


def dit() -> LightningDiT:
    """The DiT's layout (the checkpoint's tensor names) at a small size."""
    with torch.device("meta"):
        return LightningDiT(
            input_size=8,
            patch_size=2,
            in_channels=32,
            out_channels=16,
            hidden_size=64,
            depth=2,
            num_heads=2,
            mlp_ratio=4,
            use_qknorm=True,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
            z_dims=32,
            encdim_ratio=3,
        )


@pytest.fixture
def paths(tmp_path) -> BundlePaths:
    root = tmp_path / "vosr2-model" / "revision"
    generator = torch.Generator().manual_seed(7)
    state = {
        name: torch.randn(tensor.shape, generator=generator)
        for name, tensor in dit().state_dict().items()
    }
    # Values a BF16 rounding sends up, down and to even
    state["x_embedder.proj.bias"][:3] = torch.tensor([1 + 2**-8, 1 + 3 * 2**-9, -0.0])
    official = BundlePaths(root)
    official.dit.parent.mkdir(parents=True)
    save_file(state, official.dit)
    return official


def context() -> NodeContext:
    return cast(NodeContext, SimpleNamespace(suspend=lambda: None))


def loaded(path: Path) -> dict[str, torch.Tensor]:
    model = dit()
    load_weights(model, path, torch.bfloat16)
    return model.state_dict()


def test_the_copy_loads_exactly_what_the_conversion_loaded(paths):
    copied = vosr2_download.bf16_dit(context(), paths)
    copy = paths.root / "checkpoints" / "ema_model.bf16.safetensors"
    assert copied == BundlePaths(paths.root, dit_copy=copy) and copied.dit == copy
    assert {tensor.dtype for tensor in load_file(copy).values()} == {torch.bfloat16}
    expected, actual = loaded(paths.dit), loaded(copied.dit)
    assert expected.keys() == actual.keys()
    for name, tensor in expected.items():
        assert actual[name].dtype == torch.bfloat16
        assert torch.equal(actual[name].view(torch.int16), tensor.view(torch.int16))
    record = json.loads(copy.with_name("ema_model.bf16.json").read_text())
    assert record["format"] == 1 and record["dtype"] == "bfloat16"
    assert record["size"] == copy.stat().st_size
    assert record["sha256"] == vosr2_engine_cache.file_hash(copy)
    assert record["source_sha256"] == vosr2_engine_cache.file_hash(paths.dit)


def conversions(monkeypatch) -> list[Path]:
    made = []
    convert = vosr2_download.convert_checkpoint

    def counted(source: Path, target: Path, dtype: torch.dtype) -> None:
        made.append(target)
        convert(source, target, dtype)

    monkeypatch.setattr(vosr2_download, "convert_checkpoint", counted)
    return made


def test_the_copy_is_made_once(monkeypatch, paths):
    made = conversions(monkeypatch)
    first = vosr2_download.bf16_dit(context(), paths)
    assert vosr2_download.bf16_dit(context(), paths) == first
    assert len(made) == 1


def test_the_copy_hash_is_remembered_until_the_file_changes(monkeypatch, paths):
    made = conversions(monkeypatch)
    copy = vosr2_download.bf16_dit(context(), paths).dit_copy
    assert copy is not None
    hashed = []
    digest = hashlib.file_digest

    def counted(stream, name):
        hashed.append(Path(stream.name).name)
        return digest(stream, name)

    monkeypatch.setattr(hashlib, "file_digest", counted)
    # Another process: nothing remembered in memory, only beside the model
    monkeypatch.setattr(vosr2_engine_cache, "_HASHES", {})
    assert vosr2_download.bf16_dit(context(), paths).dit_copy == copy
    assert hashed == []
    times = copy.stat()
    os.utime(copy, ns=(times.st_atime_ns, times.st_mtime_ns + 1_000_000_000))
    assert vosr2_download.bf16_dit(context(), paths).dit_copy == copy
    assert hashed == [copy.name] and len(made) == 1
    with copy.open("ab") as stream:
        stream.write(b"\0")
    assert vosr2_download.bf16_dit(context(), paths).dit_copy == copy
    assert len(made) == 2


def damage_copy(copy: Path) -> None:
    data = bytearray(copy.read_bytes())
    data[-1] ^= 1
    copy.write_bytes(bytes(data))
    # An edit made later than the copy: Linux's coarse file times can give a rewrite
    # within milliseconds the copy's own modification time, and the remembered hash is
    # trusted by the file's size, times and index.
    times = copy.stat()
    os.utime(copy, ns=(times.st_atime_ns, times.st_mtime_ns + 1_000_000_000))


def shorten_copy(copy: Path) -> None:
    copy.write_bytes(copy.read_bytes()[:-2])


def remove_copy(copy: Path) -> None:
    copy.unlink()


def damage_sidecar(copy: Path) -> None:
    copy.with_name("ema_model.bf16.json").write_text("{")


def remove_sidecar(copy: Path) -> None:
    copy.with_name("ema_model.bf16.json").unlink()


@pytest.mark.parametrize(
    "change",
    [damage_copy, shorten_copy, remove_copy, damage_sidecar, remove_sidecar],
)
def test_a_changed_or_missing_copy_is_made_again(monkeypatch, paths, change):
    made = conversions(monkeypatch)
    copied = vosr2_download.bf16_dit(context(), paths)
    assert copied.dit_copy is not None
    change(copied.dit_copy)
    assert vosr2_download.bf16_dit(context(), paths) == copied
    assert len(made) == 2
    expected, actual = loaded(paths.dit), loaded(copied.dit)
    assert all(torch.equal(actual[name], tensor) for name, tensor in expected.items())


def test_a_new_format_or_a_rewritten_source_makes_the_copy_again(monkeypatch, paths):
    made = conversions(monkeypatch)
    vosr2_download.bf16_dit(context(), paths)
    monkeypatch.setattr(vosr2_download, "BF16_FORMAT", 2)
    vosr2_download.bf16_dit(context(), paths)
    assert len(made) == 2
    paths.dit.write_bytes(paths.dit.read_bytes())
    vosr2_download.bf16_dit(context(), paths)
    assert len(made) == 3
    vosr2_download.bf16_dit(context(), paths)
    assert len(made) == 3


def test_without_room_for_the_copy_the_load_converts_as_before(
    monkeypatch, paths, caplog
):
    made = conversions(monkeypatch)
    needed = paths.dit.stat().st_size // 2 + 1_000_000_000
    monkeypatch.setattr(
        vosr2_download.shutil, "disk_usage", lambda _: SimpleNamespace(free=needed - 1)
    )
    with caplog.at_level(logging.INFO, logger="sanic.root"):
        assert vosr2_download.bf16_dit(context(), paths) == paths
    assert not made
    assert "Not keeping a BF16 copy of the VOSR 2.0 DiT" in caplog.text
    assert not list(paths.dit.parent.glob("ema_model.bf16*"))
    monkeypatch.setattr(
        vosr2_download.shutil, "disk_usage", lambda _: SimpleNamespace(free=needed)
    )
    assert vosr2_download.bf16_dit(context(), paths) != paths
    assert len(made) == 1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_only_a_bf16_bundle_reads_the_copy(monkeypatch, paths, dtype):
    read = []

    class Bundle:
        def __init__(self, weights, dtype, progress, *, keep_cpu_weights):
            read.append(weights.dit)

        def offload(self) -> None:
            pass

    monkeypatch.setattr(vosr2_cache, "VosrBundle", Bundle)
    owner = cast(NodeContext, SimpleNamespace(suspend=lambda: None, node_id="vosr"))
    with vosr2_cache.lease_bundle(owner, paths, dtype, reuse=False):
        pass
    copy = paths.dit.with_name("ema_model.bf16.safetensors")
    assert read == [copy if dtype == torch.bfloat16 else paths.dit]


def test_a_failed_write_loads_the_fp32_weights_and_leaves_no_partial_copy(
    monkeypatch, paths, caplog
):
    def full_disk(state: dict, path: Path) -> None:
        path.write_bytes(b"partial")
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(vosr2_download, "save_file", full_disk)
    with caplog.at_level(logging.WARNING, logger="sanic.root"):
        assert vosr2_download.bf16_dit(context(), paths) == paths
    assert "Could not keep a BF16 copy of the VOSR 2.0 DiT" in caplog.text
    assert "No space left on device" in caplog.text
    assert not list(paths.dit.parent.glob("ema_model.bf16*"))
    monkeypatch.undo()
    assert vosr2_download.bf16_dit(context(), paths).dit_copy is not None


def test_a_stale_copy_goes_even_without_room_for_a_new_one(monkeypatch, paths):
    copy = vosr2_download.bf16_dit(context(), paths).dit_copy
    assert copy is not None
    # A killed conversion's part, then a conversion change
    vosr2_download.partial(copy).write_bytes(b"partial")
    monkeypatch.setattr(vosr2_download, "BF16_FORMAT", 2)
    monkeypatch.setattr(
        vosr2_download.shutil, "disk_usage", lambda _: SimpleNamespace(free=0)
    )
    assert vosr2_download.bf16_dit(context(), paths) == paths
    assert not list(paths.dit.parent.glob("ema_model.bf16*"))
