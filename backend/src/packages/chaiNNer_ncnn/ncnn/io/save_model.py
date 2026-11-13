import os
from pathlib import Path

from sanic.log import logger

from nodes.impl.native_graph import graph
from nodes.impl.ncnn.model import NcnnModelWrapper
from nodes.properties.inputs import DirectoryInput, NcnnModelInput, RelativePathInput

from .. import io_group


@io_group.register(
    schema_id="chainner:ncnn:save_model",
    name="Save Model",
    description="Save an NCNN model (.param and .bin) to the specified directory, in the precision it has.",
    icon="MdSave",
    inputs=[
        NcnnModelInput(),
        DirectoryInput(must_exist=False),
        RelativePathInput("Param/Bin Name"),
    ],
    outputs=[],
    side_effects=True,
)
def save_model_node(model: NcnnModelWrapper, directory: Path, name: str) -> None:
    full_bin_path = (directory / f"{name}.bin").resolve()
    full_param_path = (directory / f"{name}.param").resolve()

    logger.debug(f"Writing NCNN model to paths: {full_bin_path} {full_param_path}")

    full_bin_path.parent.mkdir(parents=True, exist_ok=True)
    # The pair is replaced together: both temporary files are written and both renames
    # checked before either file changes, and the .param goes back as it was when the
    # .bin then cannot be replaced. A mixed pair loads as a wrong model or not at all.
    # The bytes are write_param's and write_bin's (open(path, "w") text for the .param).
    param_bytes = model.model.write_param().replace("\n", os.linesep).encode("utf-8")
    old_param = full_param_path.read_bytes() if full_param_path.is_file() else None
    native = graph()
    param_target = native.AtomicTarget(full_param_path)
    with param_target as param, native.AtomicTarget(full_bin_path) as weights:
        Path(param.temp).write_bytes(param_bytes)
        Path(weights.temp).write_bytes(model.model.serialize_weights())
        param.check()
        weights.check()
        param.commit()
        try:
            weights.commit()
        except BaseException:
            if old_param is None:
                full_param_path.unlink(missing_ok=True)
            else:
                native.atomic_write(full_param_path, old_param)
            raise
