"""Builds the ROCm meta package the PyTorch package installs for AMD GPUs.

Usage:
  python native/tools/build_rocm_meta_wheel.py SDIST [--out DIR]

PyTorch's ROCm build requires rocm[device-all,libraries]: AMD's meta package
`rocm`, published as an sdist only, whose device-all extra requires the GPU
kernel wheel (rocm-sdk-device-gfx<arch>) of every architecture ROCm supports,
about 5.8 GB on Linux. A GPU needs only its architecture's wheel. This writes
AMD's package as the wheel rocm-VERSION+chainner, which differs from what pip
builds out of AMD's sdist in its metadata alone:
- the device-all extra requires nothing, so that pip installs PyTorch, ROCm's
  runtime and libraries, and no kernel wheel; the PyTorch package lists the
  kernel wheels of the GPUs it finds (backend/src/packages/chaiNNer_pytorch);
- the device extra is left out: AMD's sdist chooses its wheel when it is built
  on the user's machine, and its PKG-INFO holds the choice of AMD's build
  machine (the default architecture);
- the Dynamic fields are left out: a wheel's metadata is final.
The rocm_sdk module and the console script are AMD's, byte for byte. The
wheel is deterministic. SDIST is AMD's rocm-VERSION.tar.gz
(https://repo.amd.com/rocm/whl-multi-arch/), checked against its SHA-256
below. DIR defaults to backend/src/dependencies/whls/rocm.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import sys
import tarfile
import zipfile
from pathlib import Path

VERSION = "7.14.1"
SDIST_SHA256 = "ffea97f6a9da9987a9260f9e635995b27a44dd6985d1b2b1663431356aaf669b"
LOCAL_LABEL = "chainner"
# The architectures with a kernel wheel in VERSION: the device-all extra's lines.
DEVICE_ALL_COUNT = 25
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPOSITORY / "backend/src/dependencies/whls/rocm"


def wheel_metadata(pkg_info: str) -> str:
    """AMD's PKG-INFO as the wheel's METADATA (see the module's docstring)."""
    kept: list[str] = []
    dropped_device_all = 0
    dropped_device = 0
    for line in pkg_info.splitlines():
        if line == f"Version: {VERSION}":
            kept.append(f"Version: {VERSION}+{LOCAL_LABEL}")
        elif line.startswith("Requires-Dist: ") and line.endswith(
            'extra == "device-all"'
        ):
            dropped_device_all += 1
        elif line == "Provides-Extra: device" or (
            line.startswith("Requires-Dist: ") and line.endswith('extra == "device"')
        ):
            dropped_device += 1
        elif not line.startswith("Dynamic: "):
            kept.append(line)
    # Another ROCm's metadata, read with this one's rules, fails here instead of
    # building a wheel that requires every kernel wheel or names the wrong version.
    if f"Version: {VERSION}+{LOCAL_LABEL}" not in kept:
        raise ValueError(f"the sdist's PKG-INFO is not version {VERSION}")
    if dropped_device_all != DEVICE_ALL_COUNT or dropped_device != 2:
        raise ValueError(
            f"the sdist's PKG-INFO has {dropped_device_all} device-all requirements"
            f" (expected {DEVICE_ALL_COUNT}) and {dropped_device} device-extra lines"
            " (expected 2)"
        )
    if "Provides-Extra: device-all" not in kept:
        raise ValueError("the sdist's PKG-INFO provides no device-all extra")
    return "\n".join(kept) + "\n"


def record_line(name: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"{name},sha256={digest.decode()},{len(data)}"


def build(sdist: Path, out: Path) -> Path:
    data = sdist.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    if sha256 != SDIST_SHA256:
        raise ValueError(f"{sdist}: SHA-256 {sha256}, not rocm {VERSION}'s")

    root = f"rocm-{VERSION}"
    module_prefix = f"{root}/src/rocm_sdk/"
    files: dict[str, bytes] = {}
    with tarfile.open(sdist, "r:gz") as archive:

        def read(name: str) -> bytes:
            member = archive.extractfile(name)
            if member is None:
                raise ValueError(f"{sdist}: {name} is not a file")
            return member.read()

        for member in archive.getmembers():
            if member.isfile() and member.name.startswith(module_prefix):
                files["rocm_sdk/" + member.name[len(module_prefix) :]] = read(
                    member.name
                )
        pkg_info = read(f"{root}/PKG-INFO").decode("utf-8")
        entry_points = read(f"{root}/src/rocm.egg-info/entry_points.txt")
        top_level = read(f"{root}/src/rocm.egg-info/top_level.txt")

    dist_info = f"rocm-{VERSION}+{LOCAL_LABEL}.dist-info"
    files[f"{dist_info}/METADATA"] = wheel_metadata(pkg_info).encode("utf-8")
    files[f"{dist_info}/WHEEL"] = (
        b"Wheel-Version: 1.0\n"
        b"Generator: build_rocm_meta_wheel\n"
        b"Root-Is-Purelib: true\n"
        b"Tag: py3-none-any\n"
    )
    files[f"{dist_info}/entry_points.txt"] = entry_points
    files[f"{dist_info}/top_level.txt"] = top_level
    record = f"{dist_info}/RECORD"
    lines = [record_line(name, files[name]) for name in sorted(files)]
    files[record] = ("\n".join([*lines, f"{record},,"]) + "\n").encode("utf-8")

    out.mkdir(parents=True, exist_ok=True)
    wheel = out / f"rocm-{VERSION}+{LOCAL_LABEL}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
        # RECORD last, as the wheel format recommends.
        for name in [*sorted(n for n in files if n != record), record]:
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, files[name])
    return wheel


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sdist", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    wheel = build(args.sdist, args.out)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    print(f"{wheel} {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
