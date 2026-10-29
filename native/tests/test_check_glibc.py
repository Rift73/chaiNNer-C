"""check_glibc's reading of ELF dependencies and its glibc 2.28 floor."""

from __future__ import annotations

import struct

import check_glibc

BASE = 0x1000


def shared_object(needed: list[str], versions: dict[str, list[str]]) -> bytes:
    """A minimal ELF64 shared object: one PT_LOAD at BASE over the whole file, and a
    PT_DYNAMIC with DT_NEEDED, DT_STRTAB and DT_VERNEED entries."""
    strings = bytearray(b"\0")

    def add(text: str) -> int:
        offset = len(strings)
        strings.extend(text.encode() + b"\0")
        return offset

    names = [add(library) for library in needed]
    verneed = bytearray()
    for index, (library, required) in enumerate(versions.items()):
        following = 0 if index == len(versions) - 1 else 16 + 16 * len(required)
        verneed += struct.pack("<HHIII", 1, len(required), add(library), 16, following)
        for position, version in enumerate(required):
            last = position == len(required) - 1
            verneed += struct.pack("<IHHII", 0, 0, 0, add(version), 0 if last else 16)
    strtab = 64 + 2 * 56
    verneed_offset = strtab + len(strings)
    dynamic_offset = verneed_offset + len(verneed)
    entries = [(1, name) for name in names] + [
        (5, BASE + strtab),
        (0x6FFFFFFE, BASE + verneed_offset),
        (0x6FFFFFFF, len(versions)),
        (0, 0),
    ]
    size = dynamic_offset + 16 * len(entries)
    header = struct.pack(
        "<16sHHIQQQIHHHHHH",
        b"\x7fELF\x02\x01\x01".ljust(16, b"\0"),
        3,
        62,
        1,
        0,
        64,
        0,
        0,
        64,
        56,
        2,
        0,
        0,
        0,
    )
    load = struct.pack("<IIQQQQQQ", 1, 4, 0, BASE, BASE, size, size, 0x1000)
    dynamic = struct.pack(
        "<IIQQQQQQ",
        2,
        4,
        dynamic_offset,
        BASE + dynamic_offset,
        BASE + dynamic_offset,
        16 * len(entries),
        16 * len(entries),
        8,
    )
    body = b"".join(struct.pack("<qQ", tag, value) for tag, value in entries)
    return header + load + dynamic + bytes(strings) + bytes(verneed) + body


GLIBC_OK = {
    "libm.so.6": ["GLIBC_2.2.5", "GLIBC_2.27"],
    "libc.so.6": ["GLIBC_2.14", "GLIBC_2.28", "GLIBC_2.3.4"],
}


def test_reads_needed_libraries_and_versions(tmp_path):
    module = tmp_path / "module.so"
    module.write_bytes(shared_object(["libm.so.6", "libc.so.6"], GLIBC_OK))
    dependencies = check_glibc.read_dependencies(module)
    assert dependencies.needed == ["libm.so.6", "libc.so.6"]
    assert dependencies.versions == GLIBC_OK


def test_floor_and_static_runtimes(tmp_path, capsys):
    good = tmp_path / "ours" / "good.so"
    good.parent.mkdir()
    good.write_bytes(shared_object(["libc.so.6"], GLIBC_OK))
    (tmp_path / "ours" / "notes.txt").write_text("not an ELF file")
    assert check_glibc.main([str(tmp_path / "ours")]) == 0
    out = capsys.readouterr().out
    assert f"{good}: GLIBC_2.28; needs libc.so.6" in out
    assert "notes.txt" not in out

    cases = {
        "newer.so": (
            ["libm.so.6"],
            {"libm.so.6": ["GLIBC_2.29"]},
            "libm.so.6 GLIBC_2.29",
        ),
        "relr.so": (
            ["libc.so.6"],
            {"libc.so.6": ["GLIBC_2.2.5", "GLIBC_ABI_DT_RELR"]},
            "libc.so.6 GLIBC_ABI_DT_RELR",
        ),
        "cxx.so": (
            ["libstdc++.so.6", "libc.so.6"],
            {"libstdc++.so.6": ["GLIBCXX_3.4"]},
            "needs libstdc++.so.6",
        ),
        "python.so": (["libpython3.14.so.1.0"], {}, "needs libpython3.14.so.1.0"),
    }
    for name, (needed, versions, problem) in cases.items():
        bad = tmp_path / name
        bad.write_bytes(shared_object(needed, versions))
        assert check_glibc.main([str(good), str(bad)]) == 1
        assert f"  FAIL: {problem}\n" in capsys.readouterr().out
        # Third-party files are reported, never failed, and for glibc alone: every
        # desktop has the C++ runtime that Electron's files need.
        assert check_glibc.main([str(good), "--third-party", str(bad)]) == 0
        out = capsys.readouterr().out
        assert (f"  note: {problem}\n" in out) == (not problem.startswith("needs"))


def test_shared_runtime_module(tmp_path, capsys):
    """The MIGraphX runner may need libstdc++ and libgcc_s up to GCC 8's versions;
    any other module still may not."""
    runner = tmp_path / "ok" / "_chainner_migraphx.so"
    runner.parent.mkdir()
    runtime = {
        "libstdc++.so.6": ["GLIBCXX_3.4", "GLIBCXX_3.4.25", "CXXABI_1.3.11"],
        "libgcc_s.so.1": ["GCC_3.0", "GCC_7.0.0"],
        **GLIBC_OK,
    }
    needed = ["libstdc++.so.6", "libgcc_s.so.1", "libm.so.6", "libc.so.6"]
    runner.write_bytes(shared_object(needed, runtime))
    assert check_glibc.main([str(runner)]) == 0
    capsys.readouterr()

    cases = {
        "libstdc++.so.6": ("GLIBCXX_3.4.26", "CXXABI_1.3.12", "CXXABI_TM_1"),
        "libgcc_s.so.1": ("GCC_12.0.0",),
    }
    for library, versions in cases.items():
        for version in versions:
            newer = tmp_path / version / "_chainner_migraphx.so"
            newer.parent.mkdir()
            newer.write_bytes(shared_object(needed, {**runtime, library: [version]}))
            assert check_glibc.main([str(newer)]) == 1
            assert f"  FAIL: {library} {version}\n" in capsys.readouterr().out

    other = tmp_path / "other" / "chainner_native.so"
    other.parent.mkdir()
    other.write_bytes(shared_object(needed, runtime))
    assert check_glibc.main([str(other)]) == 1
    out = capsys.readouterr().out
    assert "  FAIL: needs libstdc++.so.6\n" in out
    assert "  FAIL: needs libgcc_s.so.1\n" in out


def test_third_party_leaves_out_our_files(tmp_path, capsys):
    ours = tmp_path / "app" / "resources" / "module.so"
    ours.parent.mkdir(parents=True)
    ours.write_bytes(shared_object(["libc.so.6"], {"libc.so.6": ["GLIBC_2.29"]}))
    electron = tmp_path / "app" / "electron"
    electron.write_bytes(shared_object(["libc.so.6"], {"libc.so.6": ["GLIBC_2.34"]}))
    app = tmp_path / "app"
    assert check_glibc.main([str(ours.parent), "--third-party", str(app)]) == 1
    out = capsys.readouterr().out
    assert out.count("module.so") == 1
    assert (
        f"{electron}: GLIBC_2.34; needs libc.so.6\n  note: libc.so.6 GLIBC_2.34\n"
        in out
    )
