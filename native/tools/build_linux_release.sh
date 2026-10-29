#!/usr/bin/env bash
# Builds chaiNNer-C's Linux release on a Linux machine with Docker, the counterpart of the Windows release's Build.ps1
# and `npm run make`: the cnl-build image (native/linux/Dockerfile), and in a container of it the native binaries with
# the CUDA and TensorRT parts against the image's glibc 2.28 sysroot (native/build_linux.sh), their glibc check
# (native/tools/check_glibc.py), and the portable zip (npm ci, npm run make-linux-zip) with the same check over the
# packaged app. The build reads one commit of a repository (git archive), never a working tree's changes or ignored
# files, so a Windows checkout's binaries cannot reach the zip.
#
# native/tools/build_linux_release.sh [--repo <git repository>] [--ref <commit>] [--work <directory>]
#     [--rocm <ROCm SDK> --migraphx <MIGraphX>] [--no-package]
#   --repo        the repository to read (default: the one holding this script); a Windows worktree read from WSL is
#                 given by its main .git directory, which Linux git can open (a worktree's .git file names a Windows
#                 path).
#   --ref         the commit, branch or tag to build (default HEAD).
#   --work        the work directory (default ~/chainner-linux/build/release): src/ (the exported tree, where the
#                 binaries and the zip land), build/ (CMake's) and cache/ (npm's and Electron's downloads, kept).
#   --rocm        with --migraphx, also builds the AMD GPU parts (native/cmake/rocm.cmake): the MIGraphX runner and the
#                 HIP custom-op libraries in nodes/impl/migraphx, with device code for every architecture ROCm 7.14
#                 ships. --rocm is the ROCm 7.14 SDK root (include/hip, lib/libamdhip64.so, lib/llvm/bin/clang++), such
#                 as the _rocm_sdk_devel directory that `rocm-sdk init` makes from AMD's rocm-sdk-core, -libraries and
#                 -devel wheels; --migraphx is MIGraphX 2.16's install prefix. Both are mounted read-only, at /opt/rocm
#                 and /opt/migraphx, and the zip carries neither: the modules load MIGraphX and HIP from the installed
#                 onnxruntime-migraphx and rocm-sdk packages (nodes/impl/migraphx/runner.py).
#   --no-package  stop after the native binaries and their check.
# The zip: <work>/src/out/make/zip/linux/x64/. Containers: --memory=16g, named cnl-build-release-<pid>.
set -euo pipefail

if [ "${1:-}" = --in-container ]; then
    # The container's part, in the exported tree (/work/src), as the host's user.
    package=$2
    amd=()
    if [ "$3" = yes ]; then
        amd=(-D CHAINNER_C_ROCM_ROOT=/opt/rocm -D CHAINNER_C_MIGRAPHX_ROOT=/opt/migraphx)
    fi
    bash native/build_linux.sh -B /work/build -j 12 -D chainner_c_llvm=/opt/llvm/bin -D CHAINNER_C_PYTHON_ROOT=/opt/python \
        -D CHAINNER_C_NASM=/opt/nasm/bin/nasm -D CHAINNER_C_TENSORRT_ROOT=/opt/tensorrt -D CHAINNER_C_CUDA_ROOT=/opt/cuda \
        -D CHAINNER_C_NVCC_HOST=/opt/gcc/bin/g++ -D CHAINNER_C_VAPOURSYNTH_INCLUDE=/opt/vapoursynth/include "${amd[@]}"
    if [ "$3" = yes ] && [ ! -f backend/src/nodes/impl/migraphx/_chainner_migraphx.so ]; then
        echo "build_linux_release.sh: no MIGraphX runner was built: this commit has no AMD GPU parts" >&2
        exit 1
    fi
    python3 -I native/tools/check_glibc.py backend/src
    if [ "$package" = yes ]; then
        npm ci --no-audit --no-fund
        npm run make-linux-zip
        app=$(find out -mindepth 1 -maxdepth 1 -type d -name '*-linux-x64')
        python3 -I native/tools/check_glibc.py "$app/resources/src" --third-party "$app"
        sha256sum out/make/zip/linux/x64/*.zip
    fi
    find backend/src -name '*.so' -exec sha256sum {} +
    exit 0
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$here
ref=HEAD
work=$HOME/chainner-linux/build/release
package=yes
rocm=
migraphx=
while [ $# -gt 0 ]; do
    case $1 in
        --repo) repo=$2; shift 2 ;;
        --ref) ref=$2; shift 2 ;;
        --work) work=$2; shift 2 ;;
        --rocm) rocm=$2; shift 2 ;;
        --migraphx) migraphx=$2; shift 2 ;;
        --no-package) package=no; shift ;;
        *) echo "build_linux_release.sh: unknown argument '$1'" >&2; exit 2 ;;
    esac
done
amd=no
mounts=()
if [ -n "$rocm" ] || [ -n "$migraphx" ]; then
    if [ -z "$rocm" ] || [ -z "$migraphx" ]; then
        echo "build_linux_release.sh: the AMD GPU parts need --rocm and --migraphx together" >&2
        exit 2
    fi
    for directory in "$rocm" "$migraphx"; do
        if [ ! -d "$directory" ]; then
            echo "build_linux_release.sh: no directory '$directory'" >&2
            exit 2
        fi
    done
    amd=yes
    mounts=(-v "$(readlink -f "$rocm"):/opt/rocm:ro" -v "$(readlink -f "$migraphx"):/opt/migraphx:ro")
fi
commit=$(git -C "$repo" rev-parse --verify "$ref^{commit}")
echo "building $commit ($ref of $repo) in $work"
rm -rf "$work/src" "$work/build"
mkdir -p "$work/src" "$work/build" "$work/cache/home"
git -C "$repo" archive --format=tar "$commit" | tar -x -C "$work/src"
docker build -t cnl-build "$work/src/native/linux"
docker run --rm --name "cnl-build-release-$$" --memory=16g --user "$(id -u):$(id -g)" -v "$work:/work" "${mounts[@]}" \
    -w /work/src -e HOME=/work/cache/home -e CI=1 cnl-build \
    bash native/tools/build_linux_release.sh --in-container "$package" "$amd"
