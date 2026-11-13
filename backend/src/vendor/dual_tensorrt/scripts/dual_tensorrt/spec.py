"""Deployment contracts, deliberately independent of torch/CUDA/TensorRT."""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "traiNNer/archs/dual_arch.py"
FACTORIES = ("dual_light", "dual_xs", "dual_s", "dual_m", "dual_l", "dual_xl")
SOURCE_SHA256 = "4d1e082ee80296520b7e12779a3284b4d13eaeb75a84d0f8a0acb59f4b7a3881"
SCHEMA = 1
# The context sites' operator the exports lower (the manifest's "context"): the source's
# neighborhood attention, where a trunk that fits the window keeps whole-window
# attention. Manifests without the key are earlier exports (non-overlapping windows).
CONTEXT = "na-v1"
# The local and retrieval sites' operators (the manifest's "local" and "retrieval"): the
# source's local_na and retrieval_na (both on by default; no parameter or key changes),
# where a map of at most 16 positions per axis (local) or a coarse grid of at most 48 per
# axis (retrieval) keeps the window or block path. Manifests without the keys are
# earlier exports (local windows, 96 px retrieval blocks).
LOCAL = "na-v1"
RETRIEVAL = "na-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Specialization:
    factory: str
    scale: int
    height: int
    width: int
    channels: int
    unshuffle: bool
    depths: tuple[int, ...]
    indices: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        if self.factory not in FACTORIES or self.scale not in (1, 2, 4):
            raise ValueError("Choose a canonical DUAL factory and scale 1, 2 or 4")
        if any(type(n) is not int or n < 4 or n % 4 for n in (self.height, self.width)):
            raise ValueError("Static B1 export requires H/W >=4 and multiples of 4")
        if self.channels not in (128, 160, 192, 224, 256):
            raise ValueError("Unsupported canonical factory width")
        h, w = self.trunk
        if 3 * h * w * self.channels >= 2**31:
            raise ValueError(
                "Specialization exceeds the kernels' 32-bit indexing bound"
            )

    @property
    def trunk(self):
        divisor = 2 if self.unshuffle else 1
        return self.height // divisor, self.width // divisor

    @property
    def plugin_key(self):
        h, w = self.trunk
        return f"c{self.channels}_h{h}_w{w}"

    @property
    def internal_scale(self):
        return self.scale * (2 if self.unshuffle else 1)

    def to_dict(self):
        return {
            **asdict(self),
            "batch": 1,
            "trunk": self.trunk,
            "internal_scale": self.internal_scale,
            "plugin_key": self.plugin_key,
        }


def from_model(model, factory, scale, height, width):
    """Read geometry from the actual canonical model, not a copied preset table."""
    return Specialization(
        factory,
        scale,
        height,
        width,
        model.embed_dim,
        model.unshuffle,
        tuple(model.depths),
        tuple(tuple(v) for v in model.selected_original_indices),
    )
