"""VOSR weights survive chain completion; the normal node-cache clear releases them."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from api import NodeContext, NodeId
from nodes.impl.tensorrt.cache import SharedCache

from .vosr2 import BundlePaths, VosrBundle
from .vosr2_download import bf16_dit

_BUNDLES: SharedCache[VosrBundle] = SharedCache(VosrBundle.offload)


def release_bundle(node_id: NodeId) -> None:
    _BUNDLES.release_node(node_id)


@contextmanager
def lease_bundle(
    context: NodeContext,
    paths: BundlePaths,
    dtype: torch.dtype,
    *,
    reuse: bool,
) -> Iterator[VosrBundle]:
    key = (paths.identity(), str(dtype))

    def create() -> VosrBundle:
        # A BF16 load reads the DiT's BF16 copy instead of converting the FP32 weights.
        weights = bf16_dit(context, paths) if dtype == torch.bfloat16 else paths
        bundle = VosrBundle(weights, dtype, context, keep_cpu_weights=True)
        if (paths.identity(), str(dtype)) != key:
            raise RuntimeError("VOSR bundle changed while loading; run again.")
        return bundle

    if not reuse:
        _BUNDLES.release_node(context.node_id)
        bundle = create()
        try:
            yield bundle
        finally:
            bundle.offload()
        return

    with _BUNDLES.use(key, context.node_id, create) as bundle:
        try:
            yield bundle
        finally:
            bundle.offload()
