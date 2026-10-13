from __future__ import annotations

from pathlib import Path

from nodes.impl.pytorch.vosr2 import BundlePaths, check_weights, read_configs
from nodes.properties.inputs import DirectoryInput
from nodes.properties.outputs import VosrModelOutput

from .. import io_group


@io_group.register(
    schema_id="chainner:pytorch:load_vosr_model",
    name="Load VOSR Model",
    description=[
        "Load a VOSR 2.0 model folder for Upscale Image (VOSR).",
        "The folder holds the one-step checkpoint with its VAE and DINOv2 weights. The files are checked here; the weights are loaded when the first image is upscaled.",
    ],
    icon="PyTorch",
    inputs=[DirectoryInput("Model Folder")],
    outputs=[VosrModelOutput().suggest()],
    see_also=["chainner:pytorch:upscale_image_vosr"],
    side_effects=True,
)
def load_vosr_model_node(directory: Path) -> BundlePaths:
    paths = BundlePaths(directory)
    read_configs(paths)
    check_weights(paths)
    return paths
