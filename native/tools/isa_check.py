"""The .text comparison of two native binaries (SP4 spec 4.1).

Usage:
  python native/tools/isa_check.py text A B

text compares the .text sections of two PE files; headers, timestamps and debug
directories are ignored. It exits 1 on any difference.

The instruction-set checks this tool held (objects: each object's VEX/EVEX use against
its level, the baseline VEX-free; dll: every VEX/EVEX instruction placed through the
linker map) retired with the Ice Lake-SP build (SP5a-c); the ThinLTO objects are bitcode,
not instructions.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


def text_section(path: Path) -> tuple[int, bytes]:
    """File offset and raw bytes (VirtualSize long) of the PE file's .text."""
    data = path.read_bytes()
    if data[:2] != b"MZ":
        raise ValueError(f"{path}: no DOS header")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe : pe + 4] != b"PE\0\0":
        raise ValueError(f"{path}: no PE signature at {pe:#x}")
    count, _, _, _, optional = struct.unpack_from("<HIIIH", data, pe + 6)
    table = pe + 24 + optional
    for index in range(count):
        name, size, _, _, offset = struct.unpack_from(
            "<8sIIII", data, table + 40 * index
        )
        if name.rstrip(b"\0") == b".text":
            code = data[offset : offset + size]
            if len(code) != size:
                raise ValueError(f"{path}: .text extends past the end of the file")
            return offset, code
    raise ValueError(f"{path}: no .text section")


def compare_text(a: Path, b: Path) -> int:
    _, code_a = text_section(a)
    _, code_b = text_section(b)
    if code_a == code_b:
        print(f"identical .text ({len(code_a)} bytes)")
        return 0
    first = next(
        (i for i, (x, y) in enumerate(zip(code_a, code_b, strict=False)) if x != y),
        min(len(code_a), len(code_b)),
    )
    print(
        f"different .text ({len(code_a)} and {len(code_b)} bytes), "
        f"first difference at .text offset {first}"
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    text = commands.add_parser("text", help="compare the .text of two PE files")
    text.add_argument("a", type=Path)
    text.add_argument("b", type=Path)
    args = parser.parse_args(argv)
    return compare_text(args.a, args.b)


if __name__ == "__main__":
    sys.exit(main())
