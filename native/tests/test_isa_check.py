"""isa_check's .text comparison of two PE files."""

from __future__ import annotations

import struct
from pathlib import Path

import isa_check

DLL = Path(__file__).resolve().parents[2] / "backend/src/nodes/impl/chainner_native.dll"


def test_text_section_ignores_headers_and_sees_code(tmp_path, capsys):
    data = DLL.read_bytes()
    a = tmp_path / "a.dll"
    a.write_bytes(data)
    offset, code = isa_check.text_section(a)
    assert offset > 0
    assert len(code) > 0
    assert data[offset : offset + len(code)] == code

    stamped = bytearray(data)
    stamped[struct.unpack_from("<I", data, 0x3C)[0] + 8] ^= 0xFF
    b = tmp_path / "b.dll"
    b.write_bytes(stamped)
    assert isa_check.text_section(b) == (offset, code)

    patched = bytearray(data)
    patched[offset + 16] ^= 0xFF
    c = tmp_path / "c.dll"
    c.write_bytes(patched)
    assert isa_check.text_section(c)[1] != code

    assert isa_check.main(["text", str(a), str(b)]) == 0
    assert capsys.readouterr().out == f"identical .text ({len(code)} bytes)\n"
    assert isa_check.main(["text", str(a), str(c)]) == 1
    assert "first difference at .text offset 16" in capsys.readouterr().out
