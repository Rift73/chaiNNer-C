"""The ONNX and tiling generators reproduce the committed C++ and bridges exactly.

generate_onnx_converter.py and generate_tiling_cpp.py write two headers and four
Python bridges, LF on every platform (backend/src is -text, so the bridges' bytes are
what every checkout gets). Regenerating them into a temporary directory must give the
committed files byte for byte, line endings aside: a checkout made before
.gitattributes read eol=lf may still hold the headers with CRLF. No generated file may
carry a lint directive: the bridges keep the globals native code reads as explicit
exports instead.
"""

import re
import shutil
from pathlib import Path

import generate_onnx_converter
import generate_tiling_cpp

ROOT = Path(__file__).resolve().parents[2]
UPSCALE = ROOT / "backend/src/nodes/impl/upscale"
DIRECTIVE = re.compile(r"#\s*(?:noqa\b|ruff:|pyright:|isort:|type:\s*ignore)")


def text(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def regenerate(tmp_path: Path, monkeypatch) -> dict[Path, Path]:
    """Run both generators into tmp_path; map each output to its committed file."""
    bridge = tmp_path / "onnx_to_ncnn.py"
    shutil.copyfile(generate_onnx_converter.BRIDGE_HOME, bridge)
    monkeypatch.setattr(generate_onnx_converter, "BRIDGE", bridge)
    monkeypatch.setattr(
        generate_onnx_converter, "OUTPUT", tmp_path / "onnx_converter_passes.hpp"
    )
    generate_onnx_converter.main()
    bridges = tmp_path / "upscale"
    bridges.mkdir()
    monkeypatch.setattr(generate_tiling_cpp, "BRIDGES", bridges)
    monkeypatch.setattr(generate_tiling_cpp, "OUTPUT", tmp_path / "tiling_control.hpp")
    generate_tiling_cpp.main()
    outputs = {
        bridge: generate_onnx_converter.BRIDGE_HOME,
        tmp_path / "onnx_converter_passes.hpp": ROOT
        / "native/include/onnx_converter_passes.hpp",
        tmp_path / "tiling_control.hpp": ROOT / "native/include/tiling_control.hpp",
    }
    for name in ("auto_split.py", "exact_split.py", "tiler.py"):
        outputs[bridges / name] = UPSCALE / name
    assert sorted(p.name for p in bridges.iterdir()) == sorted(
        p.name for p in outputs if p.parent == bridges
    )
    return outputs


def test_generators_reproduce_the_committed_files(tmp_path, monkeypatch):
    outputs = regenerate(tmp_path, monkeypatch)
    differ = [
        str(committed.relative_to(ROOT))
        for generated, committed in outputs.items()
        if text(generated) != text(committed)
    ]
    assert differ == []


def test_generated_files_carry_no_lint_directive(tmp_path, monkeypatch):
    outputs = regenerate(tmp_path, monkeypatch)
    found = [
        f"{committed.name}:{number}: {line.strip()}"
        for generated, committed in outputs.items()
        for number, line in enumerate(
            generated.read_text(encoding="utf-8").splitlines(), 1
        )
        if DIRECTIVE.search(line)
    ]
    assert found == []
