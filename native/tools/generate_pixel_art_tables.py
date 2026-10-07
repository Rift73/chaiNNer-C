"""Translate the reviewed, pinned pixel decision tables into readable C++.

This deliberately handles only the expressions present in chaiNNer-rs's seven
small pixel kernels and three exhaustive HQ match tables. It is not a general
Rust compiler. The generated code performs all image work in the native DLL;
this script is a reproducible development step, never a runtime dependency.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "native/tests/reference_pixel_art/rust"
OUTPUT = ROOT / "native/include/pixel_art_tables.inc"


def block(text: str, opening: int) -> tuple[str, int]:
    assert text[opening] == "{"
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[opening + 1 : end - 1], end


def function(text: str, name: str) -> str:
    start = re.search(r"\bfn " + name + r"<", text)
    assert start is not None, name
    return block(text, text.index("{", start.end()))[0]


def source(name: str) -> str:
    data = (REFERENCE / name).read_bytes()
    expected = json.loads((REFERENCE.parent / "sources.json").read_text())["files"]
    assert hashlib.sha256(data).hexdigest() == expected[name]
    return re.sub(r"//[^\n]*", "", data.decode())


def split_top(text: str, operator: str) -> list[str]:
    """text's operands of operator outside parentheses."""
    parts, depth, start = [], 0, 0
    for index, char in enumerate(text):
        depth += (char == "(") - (char == ")")
        if depth == 0 and index >= start and text.startswith(operator, index):
            parts.append(text[start:index])
            start = index + len(operator)
    return [*parts, text[start:]]


def grouped(condition: str) -> str:
    """Parenthesize the && operands of a top-level ||. Rust and C++ both bind &&
    tighter, so the grouping is unchanged; C++ compilers warn on the bare form."""
    operands = split_top(condition, "||")
    if len(operands) == 1:
        return condition
    return "||".join(
        re.sub(r"^(\s*)(.*?)(\s*)$", r"\1(\2)\3", operand, flags=re.S)
        if len(split_top(operand, "&&")) > 1
        else operand
        for operand in operands
    )


def statements(text: str) -> str:
    text = re.sub(r"#\[allow\(clippy::needless_late_init\)\]", "", text)
    # Rust shadows the preceding local here; C++ can reuse the same pixel slot.
    text = text.replace("let i = avg2(color5, color3);", "i = avg2(color5, color3);")
    # The only value-producing conditional in the pinned SaI pixel blocks.
    text, count = re.subn(
        r"product2 = if r > 0 \{\s*color_a\s*\} else if r < 0 \{\s*color_b\s*\} else \{\s*avg4\(color_a, color_b, color_c, color_d\)\s*\};",
        "product2 = r > 0 ? color_a : (r < 0 ? color_b : avg4(color_a, color_b, color_c, color_d));",
        text,
    )
    assert count <= 1
    assert "= if " not in text
    text = re.sub(r"\blet (?:mut )?(\w+): u8 =", r"unsigned \1 =", text)
    text = re.sub(r"\blet (?:mut )?(\w+): T;", r"T \1{};", text)
    text = text.replace(
        "let mut r: [T; 16] = Default::default();", "std::array<T, 16> r{};"
    )
    text = re.sub(r"\blet (?:mut )?(\w+) =", r"auto \1 =", text)
    text = re.sub(r"\blet (\w+);", r"T \1{};", text)
    text = text.replace("for k in 1..=9 {", "for (size_t k = 1; k <= 9; ++k) {")
    text = re.sub(
        r"\bif\s+([^{}]+?)\s*\{", lambda m: "if (" + grouped(m[1].strip()) + ") {", text
    )
    assert not re.search(r"\blet\b|\bmatch\b|\bas\b|=>", text)
    return text


def simple(name: str, filename: str, scale: int) -> str:
    text = function(source(filename), name)
    # Extract the per-pixel decision body; traversal/border loads are shared C++.
    start = re.search(r"let (?:\w+) = src\[", text)
    assert start
    end = re.search(r"write_[23]x\(dest, w, x, y, \[([^\]]+)\]\);", text)
    assert end
    body = text[start.start() : end.start()]
    offsets = {"": 1, "_m1": 0, "_p1": 2, "_p2": 3}
    body = re.sub(
        r"src\[y(_m1|_p1|_p2)? \* w \+ x(_m1|_p1|_p2)?\]",
        lambda m: f"v[{4 * offsets[m[1] or ''] + offsets[m[2] or '']}]",
        body,
    )
    assert "src[" not in body
    return (
        f"template<class T> std::array<T, {scale * scale}> {name}(const std::array<T, 16>& v) {{\n"
        + statements(body)
        + "\nreturn {"
        + end[1]
        + "};\n}\n"
    )


def hqx(scale: int) -> str:
    text = function(source(f"hqx/hq{scale}x.rs"), f"hq{scale}x_pixel")
    start = text.index("match pattern {")
    cases, end = block(text, text.index("{", start))
    cursor = 0
    patterns = []
    converted = []
    while cursor < len(cases):
        match = re.match(r"\s*([\d\s|]+)\s*=>\s*\{", cases[cursor:])
        if not match:
            assert not cases[cursor:].strip()
            break
        values = [int(v.strip()) for v in match[1].split("|")]
        patterns.extend(values)
        body, cursor = block(cases, cursor + match.end() - 1)
        converted.append(
            " ".join(f"case {v}:" for v in values)
            + " {"
            + statements(body)
            + "break;\n}"
        )
    assert sorted(patterns) == list(range(256)), (
        "HQ pattern coverage must be exhaustive and unique"
    )
    tail = text[end:].strip()
    if scale == 4:
        assert tail == "r"
        tail = "return r;"
    else:
        assert re.fullmatch(r"\[r1(?:, r\d+)+\]", tail)
        tail = "return {" + tail[1:-1] + "};"
    return (
        f"template<class T> std::array<T, {scale * scale}> hq{scale}x_pixel(const std::array<T, 10>& w) {{\n"
        + statements(text[:start])
        + "\nswitch (pattern) {\n"
        + "\n".join(converted)
        + "\n}\n"
        + tail
        + "\n}\n"
    )


def main() -> None:
    parts = [
        (
            "/* Generated by native/tools/generate_pixel_art_tables.py.\n"
            " * Reviewed chaiNNer-rs commit 6f6ead6064f81b4049d3deb803c9279c7950736b.\n"
            " * AdvMAME/Eagle: chaiNNer-rs MIT. SaI: Derek Liauw Kie Fa, GPL.\n"
            " * HQ decision tables: original hqx authors, GNU Lesser GPL.\n"
            " * See reference_pixel_art/sources.json and distributed notices. */\n"
        )
    ]
    for name, filename, scale in (
        ("adv_mame_2x", "adv_mame.rs", 2),
        ("adv_mame_3x", "adv_mame.rs", 3),
        ("eagle_2x", "eagle.rs", 2),
        ("eagle_3x", "eagle.rs", 3),
        ("sai_2x", "sai.rs", 2),
        ("super_eagle_2x", "sai.rs", 2),
        ("super_sai_2x", "sai.rs", 2),
    ):
        parts.append(simple(name, filename, scale))
    parts.extend(hqx(scale) for scale in (2, 3, 4))
    result = "\n".join(parts)
    OUTPUT.write_text(result, encoding="utf-8", newline="\n")
    print(f"Generated {OUTPUT.relative_to(ROOT)} ({len(result)} characters)")


if __name__ == "__main__":
    main()
