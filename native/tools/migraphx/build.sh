#!/usr/bin/env bash
# Builds MIGraphX 2.16.1 for ROCm 7.14.1 and chaiNNer-C's ONNX Runtime 1.30.0 MIGraphX wheel (onnxruntime-migraphx
# 1.30.0+rocm7.14, CPython 3.14, manylinux_2_28 x86-64) on a Linux machine with Docker, in containers of the image
# cnm-build (this directory's Dockerfile). AMD ships no MIGraphX for ROCm 7.14, so it is built from source; the wheel
# carries its libraries in onnxruntime/capi/, and the native release build takes its install prefix as --migraphx
# (native/BUILDING.md, Linux).
#
# native/tools/migraphx/build.sh --rocm-wheels <directory> [--work <directory>] [--jobs <n>] [step...]
#   --rocm-wheels  AMD's ROCm 7.14.1 packages from https://repo.amd.com/rocm/whl-multi-arch/: rocm-7.14.1.tar.gz and
#                  rocm_sdk_core-, rocm_sdk_libraries- and rocm_sdk_devel-7.14.1-py3-none-linux_x86_64.whl
#                  (setup_sdk.sh checks their SHA-256). Mounted read-only at /rocm-wheels.
#   --work         the work directory (default ~/chainner-migraphx), mounted at /work: src/ (ONNX Runtime's and
#                  MIGraphX's sources), rocm-venv/ (the SDK), deps/, migraphx-7.14/ (MIGraphX's install prefix),
#                  ort-build/, dist/ (the wheel), the build directories and logs. Steps that are done are kept, so a
#                  run continues where the last one stopped.
#   --jobs         parallel build jobs (default 12).
#   step           image, sources, sdk, deps, migraphx, ort or wheel; default all of them, in that order.
# Containers run as the caller's user, named cnm-build-<step>-<pid>. Peak memory is about 21 GB (rocMLIR's links).
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
wheels=
work=$HOME/chainner-migraphx
jobs=12
steps=()
while [ $# -gt 0 ]; do
    case $1 in
        --rocm-wheels) wheels=$2; shift 2 ;;
        --work) work=$2; shift 2 ;;
        --jobs) jobs=$2; shift 2 ;;
        image | sources | sdk | deps | migraphx | ort | wheel) steps+=("$1"); shift ;;
        *) echo "build.sh: unknown argument '$1'" >&2; exit 2 ;;
    esac
done
if [ -z "$wheels" ] || [ ! -d "$wheels" ]; then
    echo "build.sh: --rocm-wheels names the directory of AMD's ROCm 7.14.1 packages" >&2
    exit 2
fi
if [ ${#steps[@]} -eq 0 ]; then
    steps=(image sources sdk deps migraphx ort wheel)
fi
mkdir -p "$work/home"
for step in "${steps[@]}"; do
    echo "[$(date -Is)] $step"
    case $step in
        image) docker build -t cnm-build "$here"; continue ;;
        sources) command=(bash /recipe/fetch_sources.sh) ;;
        sdk) command=(bash /recipe/setup_sdk.sh) ;;
        deps) command=(bash /recipe/build_deps.sh) ;;
        migraphx) command=(bash /recipe/build_migraphx.sh) ;;
        ort) command=(bash /recipe/build_ort.sh) ;;
        wheel)
            # The auditwheel-repaired wheel made final: the +rocm7.14 version, MIGraphX's libraries with a run path
            # to the ROCm pip packages, their notices, and a check of every ELF file (finalize_wheel.py).
            command=(bash -c 'rm -rf /work/dist && python3.14 -I /recipe/finalize_wheel.py \
                /work/ort-build/Release/dist/onnxruntime_migraphx-1.30.0-cp314-cp314-manylinux_2_28_x86_64.whl \
                /work/dist /work/migraphx-7.14 "$(dirname "$(cat /work/rocm-root.txt)")" /work \
                && sha256sum /work/dist/*.whl') ;;
    esac
    docker run --rm --name "cnm-build-$step-$$" --user "$(id -u):$(id -g)" -e HOME=/work/home -e "JOBS=$jobs" \
        -v "$work:/work" -v "$here:/recipe:ro" -v "$(readlink -f "$wheels"):/rocm-wheels:ro" -w /work cnm-build \
        "${command[@]}" 2>&1 | tee "$work/$step.log"
done
