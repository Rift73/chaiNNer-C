"""Names the native mirrors read by string from a backend module's namespace.

Backend modules hand their globals() to the C graph (ncnn.model hands over the
module object), and the C side looks names up by string: name(g, "mkdtemp") in
image_io.cpp, global(g, "ffmpeg") in video_io.cpp, item(globals, py::str("onph"))
in onnx_converter_passes.hpp, option(globals, value, "Operation", ...) in
execution_ops.cpp, types.attr("logger") in ncnn_graph.cpp. Linters cannot see those
reads, so the imports behind them use the explicit-export form
(`import shutil as shutil`). A missing name fails only when its C path runs.

Each table lists the names the module's C entries read, taken from the C sources:
the entry's function, the helpers and classes it reaches, and nothing read from
another module's dict. The accessors in utility_scalar.cpp, file_sequence.cpp and
execution_ops.cpp fall back to builtins, as a Python global lookup does, so those
modules may name a builtin. upscale/tiler.py hands over its globals but its C entries
read no name from them, so it has no table.

The converse holds too: every `x as x` import under backend/src names something a
mirror reads, so the form cannot hide a dead import.
"""

import ast
import builtins
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend" / "src"

IMAGE_IO = ("native/src/image_io.cpp",)
VIDEO_IO = ("native/src/video_io.cpp",)
SCALAR = ("native/src/utility_scalar.cpp",)
EXECUTION = ("native/src/execution_ops.cpp",)
TILING = ("native/include/tiling_control.hpp",)

# module: (C sources, names it reads from that module's namespace, builtins allowed)
CONTRACT = {
    "nodes.impl.dds.texconv": (
        IMAGE_IO,
        (
            "Path SRGB_FORMATS __TEXCONV_EXE __decode __run_texconv cv_save_image"
            " logger mkdtemp os platform shutil split_file_path subprocess uuid"
        ),
        False,
    ),
    "nodes.impl.native_image_io": (IMAGE_IO, "cv2 split_file_path", False),
    "nodes.impl.ncnn.model": (
        ("native/src/ncnn_graph.cpp", "native/include/ncnn_optimizer_passes.hpp"),
        (
            "BinaryOpTypes DTYPE_DICT DTYPE_FP16 DTYPE_FP32 EltwiseOpTypes NcnnLayer"
            " NcnnModel NcnnParam NcnnParamCollection NcnnWeight logger param_schema"
        ),
        False,
    ),
    "nodes.impl.onnx.onnx_to_ncnn": (
        ("native/include/onnx_converter_passes.hpp",),
        (
            "APT AttributeProto BOT DTYPE_FP16 DTYPE_FP32 EOT FLOAT32_MAX GRU IRT NEM"
            " NcnnLayer NcnnModel NcnnOptimizer NodeProto PAM PAT POT ROT TensorProto"
            " UOT get_node_attr_af get_node_attr_ai get_node_attr_f"
            " get_node_attr_from_input_af get_node_attr_from_input_ai"
            " get_node_attr_from_input_f get_node_attr_i get_node_attr_s"
            " get_node_attr_tensor get_tensor_proto_data_size logger np onph"
            " set_node_attr_ai"
        ),
        False,
    ),
    "nodes.impl.upscale.auto_split": (
        TILING,
        (
            "BlendDirection Region Split TileBlender TileOverlap _SplitEx"
            " _exact_split _max_split exact_split get_h_w_c half_sin_blend_fn logger"
            " math"
        ),
        False,
    ),
    "nodes.impl.upscale.exact_split": (
        TILING,
        (
            "BlendDirection BorderType Padding Region TileBlender TileOverlap _Segment"
            " _exact_split_into_regions _exact_split_into_segments"
            " _exact_split_without_padding _pad_image create_border get_h_w_c"
            " half_sin_blend_fn logger math"
        ),
        False,
    ),
    "nodes.impl.video": (
        VIDEO_IO,
        "BufferedIOBase VideoMetadata ffmpeg logger np subprocess",
        False,
    ),
    "packages.chaiNNer_standard.image.batch_processing.load_images": (
        ("native/src/file_sequence.cpp",),
        "Generator Path get_available_image_formats glob list_glob load_image_node os",
        True,
    ),
    "packages.chaiNNer_standard.image.io.load_image": (
        IMAGE_IO,
        (
            "Image Path _decoders _read_cv cv2 dds_to_png_texconv get_ext get_h_w_c"
            " get_opencv_formats get_pil_formats logger np os platform"
            " remove_unnecessary_alpha split_file_path"
        ),
        False,
    ),
    "packages.chaiNNer_standard.image.io.save_image": (
        IMAGE_IO,
        (
            "BC7Compression DDSErrorMetric Image ImageFormat LEGACY_TO_DXGI PREFER_DX9"
            " PngColorDepth TiffColorDepth cv2 cv_save_image get_full_path get_h_w_c"
            " logger save_as_dds to_dxgi to_uint16 to_uint8"
        ),
        False,
    ),
    "packages.chaiNNer_standard.image.io.view_image_external": (
        IMAGE_IO,
        "cv2 logger mkdtemp os platform subprocess time to_uint8",
        False,
    ),
    "packages.chaiNNer_standard.image.video_frames.load_video": (
        VIDEO_IO,
        "FFMpegEnv Generator VideoLoader split_file_path",
        False,
    ),
    "packages.chaiNNer_standard.image.video_frames.save_video": (
        VIDEO_IO,
        (
            "AudioSettings FFMpegEnv PARAMETERS SimpleVideoFormat Simplicity"
            " VideoCollector VideoEncoder VideoFormat VideoPreset Writer ffmpeg"
            " get_h_w_c get_simple_format logger np os to_uint8"
        ),
        False,
    ),
    "packages.chaiNNer_standard.utility.directory.directory_go_up": (
        SCALAR,
        "range",
        True,
    ),
    "packages.chaiNNer_standard.utility.math.accumulate": (
        EXECUTION,
        "Collector Operation float max min",
        True,
    ),
    "packages.chaiNNer_standard.utility.math.logic_operation": (
        EXECUTION,
        "LogicOperation",
        True,
    ),
    "packages.chaiNNer_standard.utility.math.math": (
        SCALAR,
        (
            "Exception MathOperation ValueError _special_mod_numbers float int"
            " isinstance math max min pow"
        ),
        True,
    ),
    "packages.chaiNNer_standard.utility.math.round": (
        SCALAR,
        "RoundOperation RoundScale math",
        True,
    ),
    "packages.chaiNNer_standard.utility.value.parse_number": (SCALAR, "int", True),
    "packages.chaiNNer_standard.utility.value.range": (EXECUTION, "Generator", True),
}

# server.import_packages() order, then the node modules; prints each namespace.
# texconv locates its executable beside the entry script, so __main__ gets run.py's
# path. os._exit skips the interpreter shutdown that crashes after every package
# loaded (native/STATUS.md).
NAMESPACES = """
import importlib, json, os, sys
__file__ = os.path.abspath("run.py")
import api

for package in ("standard", "pytorch", "ncnn", "onnx", "external"):
    importlib.import_module("packages.chaiNNer_" + package)
errors = api.registry.load_nodes(os.path.abspath("server.py"))
names = {m: sorted(vars(importlib.import_module(m))) for m in sys.argv[1:]}
print(json.dumps({"errors": [repr(e.error) for e in errors], "names": names}))
sys.stdout.flush()
os._exit(0)
"""


def explicit_exports(source: str) -> list[str]:
    """Names a module imports in the redundant-alias form `x as x`."""
    names = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                plain = isinstance(node, ast.ImportFrom) or "." not in alias.name
                if plain and alias.asname == alias.name:
                    names.append(alias.name)
    return names


def stray_exports(module: str, source: str) -> list[str]:
    """`x as x` imports in a module that no mirror reads from it."""
    read = CONTRACT[module][1].split() if module in CONTRACT else []
    return [name for name in explicit_exports(source) if name not in read]


def test_every_name_is_a_string_the_c_sources_read():
    for module, (sources, names, _) in CONTRACT.items():
        text = "".join((ROOT / s).read_text(encoding="utf-8") for s in sources)
        for name in names.split():
            assert f'"{name}"' in text, (module, sources, name)


def test_every_name_the_c_mirror_reads_exists_after_import():
    done = subprocess.run(
        [sys.executable, "-B", "-c", NAMESPACES, *CONTRACT],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout.splitlines()[-1])
    assert result["errors"] == []
    missing = {}
    for module, (_, names, builtin_fallback) in CONTRACT.items():
        namespace = set(result["names"][module])
        absent = [
            name
            for name in names.split()
            if name not in namespace
            and not (builtin_fallback and hasattr(builtins, name))
        ]
        if absent:
            missing[module] = absent
    assert missing == {}


def test_every_explicit_export_is_a_name_a_mirror_reads():
    stray = {}
    for path in sorted(BACKEND.rglob("*.py")):
        module = ".".join(path.relative_to(BACKEND).with_suffix("").parts)
        names = stray_exports(module, path.read_text(encoding="utf-8"))
        if names:
            stray[module] = names
    assert stray == {}


def test_a_stray_explicit_export_is_reported():
    module = "nodes.impl.upscale.auto_split"
    source = (BACKEND / "nodes/impl/upscale/auto_split.py").read_text(encoding="utf-8")
    assert stray_exports(module, source) == []
    assert stray_exports(module, "import os as os\n" + source) == ["os"]
    assert stray_exports("nodes.impl.blend", "import os as os\n") == ["os"]
