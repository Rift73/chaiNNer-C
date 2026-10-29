"""Turn the built Windows onnxruntime_migraphx 1.30.0 wheel (DirectML + MIGraphX + CPU) into the shipped one.

1. Local version: 1.30.0 -> 1.30.0+rocm7.14.dml (METADATA and .dist-info name; onnxruntime.__version__ stays 1.30.0).
2. MIGraphX's DLLs in onnxruntime/capi: setup.py copies the ones ORT's Windows packaging names (migraphx, _c,
   _device, _gpu, _onnx, _tf, migraphx-hiprtc-driver.exe); migraphx_ref.dll (the CPU reference target MIGraphX
   loads by name, shipped on Linux too) is added from the MIGraphX install. ROCm is never copied in.
3. The Visual C++ runtime DLLs the wheel's binaries import (msvcp140*, vcruntime140*) are added to onnxruntime/capi
   from the MSVC 14.44 redistributable folder of the VS 2022 Build Tools (the toolset they were built with), as
   Microsoft's own ONNX Runtime Windows wheels carry them, so no Visual C++ Redistributable install is needed.
4. onnxruntime/MIGraphX_ThirdPartyNotices.txt: the licenses of MIGraphX and of what is compiled into it (rocMLIR and
   its LLVM, protobuf, abseil, msgpack, nlohmann/json, Eigen (MPL2-only parts), SQLite).
5. Check every DLL/EXE/PYD: each imported DLL is in the wheel, a Windows/DirectX system DLL, Python's own, or a DLL
   of the ROCm 7.14.1 pip packages (rocm-sdk-core, rocm-sdk-libraries).

Usage: python -I finalize_wheel_win.py [--vs-root <VS 2022 Build Tools>] [--readobj <llvm-readobj.exe>]
           <dist_in> <dist_out> <migraphx install> <build root> <rocm site-packages>
The build root is the work directory of the other scripts (settings_win.cmd), whose src/ and deps-src/ folders hold
the license files. The rocm site-packages folder holds _rocm_sdk_core, _rocm_sdk_libraries and _rocm_sdk_devel;
llvm-readobj defaults to the one of that SDK.
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

LOCAL = "+rocm7.14.dml"
DIST = "onnxruntime_migraphx"
DEFAULT_VS_ROOT = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
# The toolset the wheel's binaries are built with: its redistributable folder holds the runtime DLLs they import.
VC_REDIST_VERSION = "14.44.35112"
MIGRAPHX_DLLS = [
    "migraphx.dll",
    "migraphx_c.dll",
    "migraphx_device.dll",
    "migraphx_gpu.dll",
    "migraphx_onnx.dll",
    "migraphx_ref.dll",
    "migraphx_tf.dll",
    "migraphx-hiprtc-driver.exe",
]
VC_RUNTIME = re.compile(
    r"^(msvcp140(_\w+)?|vcruntime140(_\w+)?|concrt140)\.dll$", re.IGNORECASE
)
SYSTEM = re.compile(
    r"^(api-ms-win-.*|ext-ms-.*|kernel32|user32|advapi32|shell32|ole32|oleaut32|ws2_32|setupapi|dbghelp|ntdll|"
    r"bcrypt|crypt32|shlwapi|version|psapi|rpcrt4|secur32|winmm|gdi32|comdlg32|d3d12|dxgi|dxcore|"
    r"python3|python314)\.dll$",
    re.IGNORECASE,
)
IMPORT_NAME = re.compile(r"Name: (\S+\.dll)", re.IGNORECASE)
WHEEL_INFO = re.compile(rf"{re.escape(DIST)}-(.+)\.dist-info")
NOTICES = [
    ("AMD MIGraphX 2.16.1", "src/AMDMIGraphX/LICENSE"),
    ("rocMLIR", "deps-src/rocmlir/rocMLIR-*/LICENSE"),
    (
        "LLVM (in rocMLIR)",
        "deps-src/rocmlir/rocMLIR-*/external/llvm-project/llvm/LICENSE.TXT",
    ),
    ("Protocol Buffers v30.0", "deps-src/protobuf/protobuf-*/LICENSE"),
    ("Abseil 20250512.0", "deps-src/abseil/abseil-cpp-*/LICENSE"),
    ("msgpack-c 3.3.0", "deps-src/msgpack/msgpack-c-*/LICENSE_1_0.txt"),
    ("nlohmann/json 3.8.0", "deps-src/json/json-*/LICENSE.MIT"),
    ("Eigen 5.0.1 (MPL2-only)", "deps-src/eigen/eigen-*/COPYING.MPL2"),
]
SQLITE_NOTICE = "SQLite 3.50.4\n\nThe SQLite source code is in the public domain (https://sqlite.org/copyright.html).\n"


def single(pattern: str, what: str) -> str:
    found = glob.glob(pattern)
    if len(found) != 1:
        raise SystemExit(
            f"{what}: {len(found)} matches of {pattern}, expected exactly one"
        )
    return found[0]


def imports(readobj: str, path: str) -> list[str]:
    out = subprocess.run(
        [readobj, "--coff-imports", path], check=True, capture_output=True, text=True
    ).stdout
    names = IMPORT_NAME.findall(out)
    if not names:
        raise SystemExit(
            f"{readobj} --coff-imports printed no 'Name: <library>.dll' line for {path}"
        )
    return sorted(set(names))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Turn the built Windows onnxruntime_migraphx wheel into the shipped one."
    )
    parser.add_argument(
        "--vs-root",
        default=DEFAULT_VS_ROOT,
        help="VS 2022 Build Tools (default %(default)s)",
    )
    parser.add_argument("--readobj", help="llvm-readobj.exe (default: the ROCm SDK's)")
    parser.add_argument("dist_in", help="folder of the wheel build_ort_win.cmd built")
    parser.add_argument("dist_out", help="folder for the shipped wheel")
    parser.add_argument("migraphx", help="MIGraphX install prefix")
    parser.add_argument("build_root", help="work directory holding src/ and deps-src/")
    parser.add_argument("site", help="site-packages folder of the ROCm 7.14.1 SDK")
    args = parser.parse_args()

    readobj = args.readobj or os.path.join(
        args.site, "_rocm_sdk_devel", "lib", "llvm", "bin", "llvm-readobj.exe"
    )
    if not os.path.isfile(readobj):
        raise SystemExit(f"llvm-readobj not found: {readobj} (--readobj names another)")
    vc_redist = os.path.join(
        args.vs_root,
        "VC",
        "Redist",
        "MSVC",
        VC_REDIST_VERSION,
        "x64",
        "Microsoft.VC143.CRT",
    )
    if not os.path.isdir(vc_redist):
        raise SystemExit(
            f"Visual C++ redistributable folder not found: {vc_redist} (--vs-root names the Build Tools)"
        )
    wheel = single(os.path.join(args.dist_in, f"{DIST}-*.whl"), "built wheel")
    rocm = {
        os.path.basename(p).lower()
        for d in ("_rocm_sdk_core/bin", "_rocm_sdk_libraries/bin")
        for p in glob.glob(os.path.join(args.site, d, "*.dll"))
    }
    if not rocm:
        raise SystemExit(
            f"no DLLs in {args.site}\\_rocm_sdk_core\\bin or _rocm_sdk_libraries\\bin: not the ROCm SDK's site-packages"
        )
    os.makedirs(args.dist_out, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.dist_out) as tmp:
        subprocess.run(
            [sys.executable, "-m", "wheel", "unpack", "-d", tmp, wheel], check=True
        )
        root = single(os.path.join(tmp, f"{DIST}-*"), "unpacked wheel")
        info = single(os.path.join(root, f"{DIST}-*.dist-info"), "unpacked .dist-info")
        info_match = WHEEL_INFO.fullmatch(os.path.basename(info))
        if info_match is None:
            raise SystemExit(f"{info} is not named {DIST}-<version>.dist-info")
        new_version = info_match.group(1) + LOCAL
        meta = os.path.join(info, "METADATA")
        with open(meta, encoding="utf-8") as f:
            lines = f.read().split("\n")
        if sum(ln.startswith("Version: ") for ln in lines) != 1:
            raise SystemExit(f"{meta} does not hold exactly one 'Version: ' line")
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

        added = []
        for name in MIGRAPHX_DLLS:
            dst = os.path.join(capi, name)
            src = os.path.join(args.migraphx, "bin", name)
            if os.path.exists(dst):
                with open(dst, "rb") as a, open(src, "rb") as b:
                    if a.read() != b.read():
                        raise SystemExit(
                            f"{name} in the wheel differs from the MIGraphX install"
                        )
            else:
                shutil.copyfile(src, dst)
                added.append(name)
        print("added to onnxruntime/capi:", ", ".join(added) or "-")

        # The Visual C++ runtime DLLs any binary in capi imports, and theirs in turn.
        runtime = []
        queue = [
            os.path.join(capi, fn)
            for fn in os.listdir(capi)
            if fn.lower().endswith((".dll", ".pyd", ".exe"))
        ]
        while queue:
            for name in imports(readobj, queue.pop()):
                dst = os.path.join(capi, name.lower())
                if VC_RUNTIME.match(name) and not os.path.exists(dst):
                    shutil.copyfile(os.path.join(vc_redist, name.lower()), dst)
                    runtime.append(name.lower())
                    queue.append(dst)
        print(
            f"added the Visual C++ runtime from {vc_redist}:",
            ", ".join(sorted(runtime)) or "-",
        )

        parts = []
        for title, pattern in NOTICES:
            path = single(os.path.join(args.build_root, pattern), f"license of {title}")
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

        in_wheel = set()
        binaries = []
        for dirpath, _, files in os.walk(new_root):
            for fn in files:
                if fn.lower().endswith((".dll", ".pyd", ".exe")):
                    binaries.append(os.path.join(dirpath, fn))
                    in_wheel.add(fn.lower())
        problems = []
        for p in sorted(binaries):
            rel = os.path.relpath(p, new_root)
            libs = imports(readobj, p)
            external = [
                n for n in libs if n.lower() not in in_wheel and not SYSTEM.match(n)
            ]
            missing = [n for n in external if n.lower() not in rocm]
            print(f"{rel}: ROCm pip DLLs {external or '-'}")
            if missing:
                problems.append(
                    f"{rel} imports {missing}, neither in the wheel nor in the ROCm pip packages"
                )
        if problems:
            raise SystemExit("\n".join(problems))
        subprocess.run(
            [sys.executable, "-m", "wheel", "pack", "-d", args.dist_out, new_root],
            check=True,
        )
    print(glob.glob(os.path.join(args.dist_out, f"{DIST}-*{LOCAL}-*.whl")))


if __name__ == "__main__":
    main()
