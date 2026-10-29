"""The glibc floor of the Linux release's ELF files.

Usage:
  python native/tools/check_glibc.py [--floor 2.28] PATH... [--third-party PATH...]

Each PATH is an ELF file or a directory searched for them (symbolic links are
skipped). The loader's view decides: the dynamic section's needed libraries and
version requirements, read through the program headers. Every file is listed
with its highest GLIBC_ version and its needed libraries. A file of PATH fails
when it needs
- a GLIBC_ version above the floor, or an unnumbered one (GLIBC_PRIVATE, or
  GLIBC_ABI_DT_RELR, which glibc 2.36 introduced);
- libstdc++ or libgcc_s: the modules link both statically, so that they need
  glibc alone;
- libpython: the interpreter provides its symbols as a module loads.
The --third-party files (the Electron runtime's; files also under a PATH are
left out) are listed with their glibc findings alone, which never fail the
check. Exit status 1 when a file of PATH fails.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

ELF_MAGIC = b"\x7fELF"
PT_LOAD = 1
PT_DYNAMIC = 2
DT_NULL = 0
DT_NEEDED = 1
DT_STRTAB = 5
DT_VERNEED = 0x6FFFFFFE
DT_VERNEEDNUM = 0x6FFFFFFF
GLIBC_VERSION = re.compile(r"GLIBC_(\d+)\.(\d+)(?:\.(\d+))?")
# The C++ and compiler runtimes the modules link statically.
STATIC_RUNTIMES = re.compile(r"(libstdc\+\+|libgcc_s)\.so")
LIBPYTHON = re.compile(r"libpython\d")


@dataclass
class Dependencies:
    """The needed libraries and, per library, the versions a file requires."""

    needed: list[str] = field(default_factory=list)
    versions: dict[str, list[str]] = field(default_factory=dict)


def read_dependencies(path: Path) -> Dependencies:
    """The dynamic section's DT_NEEDED entries and DT_VERNEED requirements."""
    data = path.read_bytes()
    if data[4:6] != b"\x02\x01":
        raise ValueError(f"{path}: not a little-endian ELF64 file")
    (phoff,) = struct.unpack_from("<Q", data, 32)
    phentsize, phnum = struct.unpack_from("<HH", data, 54)
    loads: list[tuple[int, int, int]] = []
    dynamic: tuple[int, int] | None = None
    for index in range(phnum):
        kind, _, offset, vaddr, _, filesz, _, _ = struct.unpack_from(
            "<IIQQQQQQ", data, phoff + index * phentsize
        )
        if kind == PT_LOAD:
            loads.append((vaddr, filesz, offset))
        elif kind == PT_DYNAMIC:
            dynamic = (offset, filesz)
    result = Dependencies()
    if dynamic is None:
        return result

    def file_offset(address: int) -> int:
        for vaddr, filesz, offset in loads:
            if vaddr <= address < vaddr + filesz:
                return address - vaddr + offset
        raise ValueError(f"{path}: address {address:#x} lies in no loaded segment")

    entries: dict[int, list[int]] = {}
    start, size = dynamic
    for position in range(start, start + size, 16):
        tag, value = struct.unpack_from("<qQ", data, position)
        if tag == DT_NULL:
            break
        entries.setdefault(tag, []).append(value)
    strtab = file_offset(entries[DT_STRTAB][0])

    def string(offset: int) -> str:
        begin = strtab + offset
        return data[begin : data.index(b"\0", begin)].decode()

    result.needed = [string(offset) for offset in entries.get(DT_NEEDED, [])]
    if DT_VERNEED in entries:
        record = file_offset(entries[DT_VERNEED][0])
        for _ in range(entries[DT_VERNEEDNUM][0]):
            _, count, name, aux, following = struct.unpack_from("<HHIII", data, record)
            versions = result.versions.setdefault(string(name), [])
            entry = record + aux
            for _ in range(count):
                _, _, _, version, entry_next = struct.unpack_from("<IHHII", data, entry)
                versions.append(string(version))
                entry += entry_next
            record += following
    return result


def glibc_key(version: str) -> tuple[int, ...]:
    match = GLIBC_VERSION.fullmatch(version)
    if match is None:
        raise ValueError(f"not a GLIBC_ version: {version}")
    return tuple(int(part) for part in match.groups() if part is not None)


def problems(
    dependencies: Dependencies, floor: tuple[int, ...], ours: bool
) -> list[str]:
    """What keeps a file from loading on every glibc from the floor on (and, for
    our own files, with glibc alone)."""
    found: list[str] = []
    for library, versions in dependencies.versions.items():
        for version in versions:
            if version.startswith("GLIBC_") and (
                not GLIBC_VERSION.fullmatch(version) or glibc_key(version) > floor
            ):
                found.append(f"{library} {version}")
    for library in dependencies.needed:
        if ours and (STATIC_RUNTIMES.match(library) or LIBPYTHON.match(library)):
            found.append(f"needs {library}")
    return found


def elf_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = []
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and not candidate.is_symlink():
            with candidate.open("rb") as stream:
                if stream.read(4) == ELF_MAGIC:
                    files.append(candidate)
    return files


def check(files: list[Path], floor: tuple[int, ...], ours: bool) -> int:
    """Lists files; the number of them that fail, when they are ours."""
    failed = 0
    for file in files:
        dependencies = read_dependencies(file)
        glibc = [
            version
            for versions in dependencies.versions.values()
            for version in versions
            if GLIBC_VERSION.fullmatch(version)
        ]
        highest = max(glibc, key=glibc_key) if glibc else "no GLIBC_ version"
        needed = " ".join(dependencies.needed) or "nothing"
        print(f"{file}: {highest}; needs {needed}")
        found = problems(dependencies, floor, ours)
        for problem in found:
            print(f"  {'FAIL' if ours else 'note'}: {problem}")
        if ours and found:
            failed += 1
    return failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--floor", default="2.28", help="the oldest glibc (2.28)")
    parser.add_argument("--third-party", nargs="+", type=Path, default=[])
    args = parser.parse_args(argv)
    floor = glibc_key(f"GLIBC_{args.floor}")
    ours = [file for path in args.paths for file in elf_files(path)]
    resolved = {file.resolve() for file in ours}
    third_party = [
        file
        for path in args.third_party
        for file in elf_files(path)
        if file.resolve() not in resolved
    ]
    failed = check(ours, floor, ours=True)
    if third_party:
        print("third-party files (reported only):")
        check(third_party, floor, ours=False)
    if failed:
        print(f"{failed} file(s) need more than glibc {args.floor}")
        return 1
    print(f"every checked file needs glibc {args.floor} or older alone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
