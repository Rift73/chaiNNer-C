#!/usr/bin/env bash
# Configures and builds chaiNNer-C's native binaries on Linux, the counterpart of Build.ps1: Ninja with the toolchain
# that toolchain-linux.cmake pins (clang 23.1.2, ld.lld, llvm-ar).
#
# native/build_linux.sh [-c Release|Debug] [-B <build directory>] [-D NAME=VALUE]... [-j <jobs>] [--ninja <ninja>]
#   -c       the configuration (default Release).
#   -B       the CMake build directory (default native/build-linux).
#   -D       a CMake cache entry, as Build.ps1's -Define: CHAINNER_C_PYTHON_ROOT=<runtime> (bin/python3,
#            include/python3.14 and NumPy), CHAINNER_C_NASM=<NASM 3.01> (else nasm on PATH, which must be 3.01),
#            CHAINNER_C_OUTPUT_DIR=<dir>, CHAINNER_C_EXT_OUTPUT_DIR=<dir>, CHAINNER_C_PDB=ON, CHAINNER_C_REGEX_ONLY=ON
#            or, for local builds only, CHAINNER_C_MARCH=<cpu> (never for releases); on a directory's first configure,
#            toolchain-linux.cmake's chainner_c_llvm=<LLVM 23.1.2's bin>; CHAINNER_C_TENSORRT_ROOT with
#            CHAINNER_C_CUDA_ROOT also builds the TensorRT parts (without them they are skipped); CHAINNER_C_ROCM_ROOT
#            (ROCm 7.14) builds the AMD GPU parts (the HIP custom-op libraries), and CHAINNER_C_MIGRAPHX_ROOT
#            (MIGraphX 2.16) with it the MIGraphX runner (cmake/rocm.cmake).
#   -j       parallel build jobs (default: Ninja's).
#   --ninja  ninja (default: ninja on PATH); it must exist.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
configuration=Release
build_directory=$here/build-linux
defines=()
jobs=()
ninja=$(command -v ninja || true)
while [ $# -gt 0 ]; do
    case $1 in
        -c) configuration=$2 ;;
        -B) build_directory=$2 ;;
        -D) defines+=("-D$2") ;;
        -j) jobs=(--parallel "$2") ;;
        --ninja) ninja=$2 ;;
        *) echo "build_linux.sh: unknown argument '$1'" >&2; exit 2 ;;
    esac
    shift 2
done
case $configuration in
    Release | Debug) ;;
    *) echo "build_linux.sh: the configuration is Release or Debug, not '$configuration'" >&2; exit 2 ;;
esac
if [ -z "$ninja" ] || [ ! -x "$ninja" ]; then
    echo "build_linux.sh: Ninja not found: '$ninja'" >&2
    exit 1
fi
arguments=(-S "$here" -B "$build_directory" -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" "-DCMAKE_BUILD_TYPE=$configuration"
    "${defines[@]}")
# CMake reads the toolchain at a directory's first configure only (and warns if passed again).
if [ ! -f "$build_directory/CMakeCache.txt" ]; then
    arguments+=("-DCMAKE_TOOLCHAIN_FILE=$here/toolchain-linux.cmake")
fi
cmake "${arguments[@]}"
cmake --build "$build_directory" "${jobs[@]}"
