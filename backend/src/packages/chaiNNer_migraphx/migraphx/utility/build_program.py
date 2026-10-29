from __future__ import annotations

from enum import Enum

from sanic.log import logger

from api import NodeContext
from gpu import amd
from nodes.groups import Condition, if_group
from nodes.impl.migraphx import drft, dual
from nodes.impl.migraphx.build_cache import cached_program, program_key
from nodes.impl.migraphx.model import MIGraphXProgram
from nodes.impl.migraphx.program import compile_in_child
from nodes.impl.migraphx.program_info import read_program_info
from nodes.impl.migraphx.runner import runner_identity
from nodes.impl.onnx.load import model_facts
from nodes.impl.onnx.model import OnnxModel
from nodes.properties.inputs import (
    BoolInput,
    EnumInput,
    NumberInput,
    OnnxModelInput,
)
from nodes.properties.outputs import MIGraphXProgramOutput
from system import is_windows

from ...settings import MIGraphXSettings, get_settings
from .. import utility_group


class ProgramPrecision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    FP8 = "fp8"


PRECISION_LABELS = {
    ProgramPrecision.FP32: "FP32 (Higher Precision)",
    ProgramPrecision.FP16: "FP16 (Faster)",
    ProgramPrecision.FP8: "FP8 (Fastest, Lowest Precision)",
}

# Shown until the ONNX is known to be DUAL's, which keeps its own precision
NOT_DUAL = Condition.type(
    0, 'OnnxModel { arch: invStrSet("DUAL") }', if_not_connected=True
)

Size = tuple[int, int]


if utility_group is not None:

    @utility_group.register(
        schema_id="chainner:migraphx:build_program",
        name="Build Program",
        description=[
            "Compile an ONNX model into a MIGraphX program for an AMD GPU.",
            "Compiling can take several minutes depending on the model size and settings; it runs in a process of its own, which Stop ends.",
            "A program takes one input size, Height x Width; Upscale Image splits images of any size into overlapping tiles of that size. It is compiled for your GPU's target (gfx) and MIGraphX version: save it for reuse, as compiling is slow.",
            "A DUAL ONNX from Convert To ONNX compiles in its own mixed precision with chaiNNer-C's DUAL operators; a dynamic one takes this node's size (multiples of 4 px), a fixed one keeps its own. Precision does not apply and is hidden. Linux only: DUAL's operators need Triton, which has no AMD build for Windows.",
            "A legacy DRFT ONNX from Convert To ONNX also compiles in its own mixed precision, with chaiNNer-C's DRFT operators. Its size must be a multiple of the ONNX's alignment and at least its minimum size on each side (Build Program says which if not). Precision does not apply.",
        ],
        icon="BsAmd",
        inputs=[
            OnnxModelInput("ONNX Model"),
            if_group(NOT_DUAL)(
                EnumInput(
                    ProgramPrecision,
                    label="Precision",
                    default=ProgramPrecision.FP16,
                    option_labels=PRECISION_LABELS,
                    conditions={
                        ProgramPrecision.FP8: Condition.const(
                            any(d.supports_fp8 for d in amd.devices)
                        )
                    },
                ).with_docs(
                    "FP16: lower precision but faster and uses less memory. FP16 does not"
                    " work with certain models.",
                    "FP32: higher precision but slower. Use especially if FP16 fails.",
                    "FP8: MIGraphX's FP8 quantization, on GPUs that support it (shown only"
                    " then). Fastest, but the least precise; check its output against FP16.",
                    "An FP16 ONNX stays FP16 in FP32 (MIGraphX does not widen a model), and a"
                    " mixed-precision ONNX (Convert To ONNX's Mixed FP16 or Mixed BF16) keeps"
                    " its own precision. Get Program Info shows what a program runs in.",
                ),
            ),
            NumberInput(
                "Height",
                default=256,
                min=64,
                max=8192,
                unit="px",
            ).with_docs("The program's input height: the tile height."),
            NumberInput(
                "Width",
                default=256,
                min=64,
                max=8192,
                unit="px",
            ).with_docs("The program's input width: the tile width."),
            BoolInput("Exhaustive Tuning", default=False).with_docs(
                "MIGraphX tries more kernels for each layer: the compile takes much longer"
                " and the program may run faster.",
            ),
        ],
        outputs=[
            MIGraphXProgramOutput(),
        ],
        node_context=True,
    )
    def build_program_node(
        context: NodeContext,
        onnx_model: OnnxModel,
        precision: ProgramPrecision,
        height: int,
        width: int,
        exhaustive_tune: bool,
    ) -> MIGraphXProgram:
        settings = get_settings(context)
        size = (height, width)

        facts = model_facts(onnx_model)
        if facts.drft_manifest is not None:
            return build_drft_program(
                context,
                onnx_model,
                facts.drft_manifest,
                settings,
                size,
                exhaustive_tune,
            )
        if facts.dual_manifest is not None:
            return build_dual_program(
                context,
                onnx_model,
                facts.dual_manifest,
                settings,
                size,
                exhaustive_tune,
            )
        if precision == ProgramPrecision.FP8 and not settings.supports_fp8:
            raise ValueError(
                f"FP8 needs a GPU that supports it; {settings.gpu_name}"
                f" ({settings.gfx}) does not. Pick FP16 or FP32."
            )

        # A mixed-precision ONNX (ModelOpt AutoCast) keeps its own types.
        mixed = onnx_model.info.dtype.startswith("mixed-")
        fp16 = not mixed and precision == ProgramPrecision.FP16
        fp8 = not mixed and precision == ProgramPrecision.FP8
        channels = onnx_model.info.input_channels or 3

        def build() -> MIGraphXProgram:
            logger.info(
                "Compiling a MIGraphX program: %s, %dx%d, %d channels, exhaustive=%s,"
                " GPU %d (%s)",
                "mixed" if mixed else precision.value,
                width,
                height,
                channels,
                exhaustive_tune,
                settings.gpu_index,
                settings.gfx,
            )
            program = compile_in_child(
                onnx_model.bytes,
                settings.gpu_index,
                size,
                fp16=fp16,
                fp8=fp8,
                exhaustive_tune=exhaustive_tune,
                channels=channels,
                context=context,
            )
            return MIGraphXProgram(
                program, read_program_info(program, settings.gpu_index)
            )

        key = program_key(
            onnx_model,
            ("onnx", fp16, fp8, size, channels, exhaustive_tune),
            settings.gpu_name,
            settings.gfx,
            runner_identity(),
            None,
        )
        return cached_program(key, build, settings.program_cache_path)


def build_dual_program(
    context: NodeContext,
    onnx_model: OnnxModel,
    manifest: dict,
    settings: MIGraphXSettings,
    size: Size,
    exhaustive_tune: bool,
) -> MIGraphXProgram:
    """DUAL's program with its custom operators, in the ONNX's mixed precision. A dynamic
    DUAL ONNX takes the node's size; a fixed one has its own. The node's precision input
    does not apply (hidden). Not on Windows, where Triton cannot compile its kernels."""
    if is_windows:
        raise ValueError(
            "DUAL's MIGraphX route needs Triton, which has no AMD build for Windows;"
            " DUAL runs through PyTorch."
        )
    spec = manifest["specialization"]
    build_size: Size | None = None
    if spec.get("dynamic", False):
        dual.check_profile(manifest, size, size, size)
        build_size = size

    def build() -> MIGraphXProgram:
        logger.info(
            "Compiling a DUAL program (%s x%d, %s, exhaustive=%s, GPU %d (%s))",
            spec["factory"],
            spec["scale"],
            "its own size" if build_size is None else f"{size[1]}x{size[0]}",
            exhaustive_tune,
            settings.gpu_index,
            settings.gfx,
        )
        program = dual.build_program(
            onnx_model.bytes,
            manifest,
            settings.gpu_index,
            build_size,
            exhaustive_tune=exhaustive_tune,
            context=context,
        )
        return MIGraphXProgram(program, read_program_info(program, settings.gpu_index))

    key = program_key(
        onnx_model,
        ("dual", build_size, exhaustive_tune),
        settings.gpu_name,
        settings.gfx,
        runner_identity(),
        dual.build_identity(),
    )
    return cached_program(key, build, settings.program_cache_path)


def build_drft_program(
    context: NodeContext,
    onnx_model: OnnxModel,
    manifest: dict,
    settings: MIGraphXSettings,
    size: Size,
    exhaustive_tune: bool,
) -> MIGraphXProgram:
    """Legacy DRFT's program with its custom operators, in the ONNX's mixed precision,
    for the node's size; its precision input does not apply."""
    drft.check_profile(manifest, size, size, size)

    def build() -> MIGraphXProgram:
        logger.info(
            "Compiling a DRFT program (x%d, window %d, %dx%d, exhaustive=%s, GPU %d (%s))",
            manifest["scale"],
            manifest["window_size"],
            size[1],
            size[0],
            exhaustive_tune,
            settings.gpu_index,
            settings.gfx,
        )
        program = drft.build_program(
            onnx_model.bytes,
            manifest,
            settings.gpu_index,
            size,
            exhaustive_tune=exhaustive_tune,
            context=context,
        )
        return MIGraphXProgram(program, read_program_info(program, settings.gpu_index))

    key = program_key(
        onnx_model,
        ("drft", size, exhaustive_tune),
        settings.gpu_name,
        settings.gfx,
        runner_identity(),
        drft.build_identity(),
    )
    return cached_program(key, build, settings.program_cache_path)
