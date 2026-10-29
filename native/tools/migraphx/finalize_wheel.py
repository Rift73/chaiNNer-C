"""Turn the auditwheel-repaired onnxruntime_migraphx 1.30.0 wheel into the shipped one.

1. Local version: 1.30.0 -> 1.30.0+rocm7.14 (METADATA and .dist-info name; onnxruntime.__version__ stays 1.30.0).
2. MIGraphX is not part of AMD's ROCm pip packages, so its libraries are added to onnxruntime/capi/ under their
   sonames: everything the MIGraphX provider needs (transitively), the gpu and ref targets MIGraphX dlopens by
   versioned soname, and migraphx-hiprtc-driver (MIGraphX looks for it next to libmigraphx_gpu and compiles kernels
   in that child process). They are stripped of symbols and debug info.
3. Run path of the provider, the MIGraphX libraries and the driver: $ORIGIN, then the ROCm pip packages installed in
   the same site-packages (rocm-sdk-core, its rocm_sysdeps, rocm-sdk-libraries), the same entries torch's ROCm wheel
   uses (DT_RPATH, as torch). ROCm itself is never copied into the wheel.
4. onnxruntime/MIGraphX_ThirdPartyNotices.txt: the licenses of MIGraphX and of what is compiled into it (rocMLIR and
   its LLVM, composable_kernel, protobuf, abseil, msgpack, nlohmann/json, Eigen (MPL2-only parts), SQLite).
5. Check every ELF file: each NEEDED library is in the wheel, a manylinux_2_28 system library, or a library of the
   ROCm 7.14.1 pip packages; no GLIBC symbol version above 2.28.

Usage: python -I finalize_wheel.py <repaired wheel> <dist_out> <migraphx prefix> <rocm venv site-packages> <build root>
"""

import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

LOCAL = "+rocm7.14"
DIST = "onnxruntime_migraphx"
RPATH = ":".join(
    [
        "$ORIGIN",
        "$ORIGIN/../../_rocm_sdk_core/lib",
        "$ORIGIN/../../_rocm_sdk_core/lib/rocm_sysdeps/lib",
        "$ORIGIN/../../_rocm_sdk_libraries/lib",
    ]
)
SYSTEM = {
    "libc.so.6",
    "libm.so.6",
    "libdl.so.2",
    "librt.so.1",
    "libpthread.so.0",
    "libgcc_s.so.1",
    "libstdc++.so.6",
    "ld-linux-x86-64.so.2",
}
DLOPENED_TARGETS = ("gpu", "ref")
NOTICES = [
    ("AMD MIGraphX 2.16.1", "src/AMDMIGraphX/LICENSE"),
    ("rocMLIR", "deps-src/rocmlir/rocMLIR-*/LICENSE"),
    (
        "LLVM (in rocMLIR)",
        "deps-src/rocmlir/rocMLIR-*/external/llvm-project/llvm/LICENSE.TXT",
    ),
    ("composable_kernel", "deps-src/ck/composable_kernel-*/LICENSE"),
    ("Protocol Buffers v30.0", "deps-src/protobuf/protobuf-*/LICENSE"),
    ("Abseil 20250512.0", "deps-src/abseil/abseil-cpp-*/LICENSE"),
    ("msgpack-c 3.3.0", "deps-src/msgpack/msgpack-c-*/LICENSE_1_0.txt"),
    ("nlohmann/json 3.8.0", "deps-src/json/json-*/LICENSE.MIT"),
    ("Eigen 5.0.1 (MPL2-only)", "deps-src/eigen/eigen-*/COPYING.MPL2"),
]
SQLITE_NOTICE = "SQLite 3.50.4\n\nThe SQLite source code is in the public domain (https://sqlite.org/copyright.html).\n"


def run(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def needed(path: str) -> list[str]:
    return re.findall(
        r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", run("readelf", "-d", path)
    )


def is_elf(path: str) -> bool:
    with open(path, "rb") as f:
        return f.read(4) == b"\x7fELF"


def glibc_max(path: str) -> tuple[int, ...]:
    versions = re.findall(r"GLIBC_([0-9.]+)", run("objdump", "-T", path))
    return max((tuple(int(x) for x in v.split(".")) for v in versions), default=(0,))


def rocm_sonames(site: str) -> set[str]:
    names = set()
    for d in (
        "_rocm_sdk_core/lib",
        "_rocm_sdk_core/lib/rocm_sysdeps/lib",
        "_rocm_sdk_libraries/lib",
    ):
        names.update(
            os.path.basename(p) for p in glob.glob(os.path.join(site, d, "*.so*"))
        )
    return names


def main() -> None:
    repaired, dist_out, migx, site, build_root = sys.argv[1:6]
    # MIGraphX 2.16 installs libmigraphx_c in lib/ and the other libraries in lib/migraphx/lib/.
    migx_lib_dirs = [
        os.path.join(migx, "lib"),
        os.path.join(migx, "lib", "migraphx", "lib"),
    ]

    def migx_lib(name: str) -> str:
        (path,) = [
            p
            for p in (os.path.join(d, name) for d in migx_lib_dirs)
            if os.path.exists(p)
        ]
        return path

    rocm = rocm_sonames(site)
    os.makedirs(dist_out, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dist_out) as tmp:
        run(sys.executable, "-m", "wheel", "unpack", "-d", tmp, repaired)
        (root,) = glob.glob(os.path.join(tmp, f"{DIST}-*"))
        (info,) = glob.glob(os.path.join(root, f"{DIST}-*.dist-info"))
        version = os.path.basename(info)[len(DIST) + 1 : -len(".dist-info")]
        new_version = version + LOCAL
        meta = os.path.join(info, "METADATA")
        with open(meta, encoding="utf-8") as f:
            lines = f.read().split("\n")
        lines = [
            f"Version: {new_version}" if ln.startswith("Version: ") else ln
            for ln in lines
        ]
        with open(meta, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        os.rename(info, os.path.join(root, f"{DIST}-{new_version}.dist-info"))
        new_root = os.path.join(tmp, f"{DIST}-{new_version}")
        os.rename(root, new_root)
        capi = os.path.join(new_root, "onnxruntime", "capi")
        provider = os.path.join(capi, "libonnxruntime_providers_migraphx.so")

        soname = re.search(
            r"libmigraphx\.so\.(\d+)", " ".join(os.listdir(migx_lib_dirs[1]))
        )
        if soname is None:
            raise SystemExit(
                f"{migx_lib_dirs[1]} holds no libmigraphx.so.<version>: "
                "not a MIGraphX 2.16 install prefix"
            )
        so_major = soname.group(1)
        queue = [provider, os.path.join(migx, "bin", "migraphx-hiprtc-driver")]
        queue += [migx_lib(f"libmigraphx_{t}.so.{so_major}") for t in DLOPENED_TARGETS]
        added: list[str] = []
        while queue:
            src = queue.pop()
            name = os.path.basename(src)
            dst = os.path.join(capi, name)
            if src != provider and not os.path.exists(dst):
                shutil.copyfile(os.path.realpath(src), dst)
                os.chmod(dst, 0o755)
                run("strip", "--strip-unneeded", dst)
                added.append(name)
            for lib in needed(dst):
                if lib.startswith("libmigraphx") and not os.path.exists(
                    os.path.join(capi, lib)
                ):
                    queue.append(migx_lib(lib))
        for name in ["libonnxruntime_providers_migraphx.so", *added]:
            run(
                "patchelf",
                "--force-rpath",
                "--set-rpath",
                RPATH,
                os.path.join(capi, name),
            )
        print("added to onnxruntime/capi:", ", ".join(sorted(added)))

        parts = []
        for title, pattern in NOTICES:
            (path,) = glob.glob(os.path.join(build_root, pattern))
            with open(path, encoding="utf-8") as f:
                parts.append(f"{title}\n\n{f.read().strip()}\n")
        parts.append(SQLITE_NOTICE)
        rule = "\n" + "-" * 78 + "\n\n"
        with open(
            os.path.join(new_root, "onnxruntime", "MIGraphX_ThirdPartyNotices.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(rule.join(parts))

        problems = []
        in_wheel = set()
        elves = []
        for dirpath, _, files in os.walk(new_root):
            for fn in files:
                p = os.path.join(dirpath, fn)
                if is_elf(p):
                    elves.append(p)
                    in_wheel.add(fn)
        for p in sorted(elves):
            rel = os.path.relpath(p, new_root)
            libs = needed(p)
            external = [n for n in libs if n not in in_wheel and n not in SYSTEM]
            missing = [n for n in external if n not in rocm]
            g = glibc_max(p)
            rp = re.findall(
                r"R(?:UN)?PATH\)\s+Library r(?:un)?path: \[([^\]]*)\]",
                run("readelf", "-d", p),
            )
            print(
                f"{rel}: GLIBC max {'.'.join(map(str, g))}, ROCm pip libs {external or '-'}, rpath {rp or '-'}"
            )
            if missing:
                problems.append(
                    f"{rel} needs {missing}, which is neither in the wheel nor in the ROCm pip packages"
                )
            if g > (2, 28):
                problems.append(f"{rel} needs GLIBC {'.'.join(map(str, g))}")
        if problems:
            raise SystemExit("\n".join(problems))
        run(sys.executable, "-m", "wheel", "pack", "-d", dist_out, new_root)
    print(glob.glob(os.path.join(dist_out, f"{DIST}-*{LOCAL}-*.whl")))


if __name__ == "__main__":
    main()
