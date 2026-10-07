from __future__ import annotations

import os
from pathlib import Path

from sanic.log import logger

from api import NodeContext
from nodes.impl.pytorch.resource_cache import file_identity
from nodes.impl.tensorrt.cache import engines
from nodes.impl.tensorrt.engine_info import deserialize_engine, read_engine_info
from nodes.impl.tensorrt.model import TensorRTEngine
from nodes.properties.inputs import TensorRTFileInput
from nodes.properties.outputs import (
    DirectoryOutput,
    FileNameOutput,
    TensorRTEngineOutput,
)
from nodes.utils.utils import split_file_path

from ...settings import get_settings
from .. import io_group

if io_group is not None:

    @io_group.register(
        schema_id="chainner:tensorrt:load_engine",
        name="Load Engine",
        description=(
            "Load a TensorRT engine file (.engine, .trt, .plan). "
            "TensorRT engines are built for a specific GPU architecture and may not work "
            "on different GPUs. The node will warn you if there's a potential compatibility issue."
        ),
        icon="BsNvidia",
        inputs=[TensorRTFileInput(primary_input=True)],
        outputs=[
            TensorRTEngineOutput(kind="tagged").suggest(),
            DirectoryOutput("Directory", of_input=0).with_id(2),
            FileNameOutput("Name", of_input=0).with_id(1),
        ],
        side_effects=True,
        node_context=True,
    )
    def load_engine_node(
        context: NodeContext, path: Path
    ) -> tuple[TensorRTEngine, Path, str]:
        assert os.path.exists(path), f"Engine file at location {path} does not exist"
        assert os.path.isfile(path), f"Path {path} is not a file"

        settings = get_settings(context)
        gpu_index = settings.gpu_index

        def load() -> TensorRTEngine:
            logger.debug("Reading TensorRT engine from path: %s", path)
            with open(path, "rb") as f:
                engine_bytes = f.read()
            loaded = deserialize_engine(engine_bytes, gpu_index)
            try:
                return TensorRTEngine(engine_bytes, read_engine_info(loaded), loaded)
            except BaseException:
                loaded.release()
                raise

        # Deserialized once and kept across runs until no node or session uses
        # it; a changed file reloads.
        with engines.use(
            (file_identity(path), gpu_index), context.node_id, load
        ) as engine:
            dirname, basename, _ = split_file_path(path)
            return engine, dirname, basename
