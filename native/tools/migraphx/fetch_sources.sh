#!/bin/bash
# In the cnm-build image (build.sh's sources step): the sources into /work/src, each checked by its commit:
#   onnxruntime  ONNX Runtime's v1.30.0 tag (f2c39fe2) with its submodules;
#   AMDMIGraphX  release/rocm-rel-7.14 at 2487b688, "MIGraphX 2.16.1 for ROCm 7.14.1" (the rocm-7.14 tag is 2.16.0).
set -euo pipefail
ORT_COMMIT=f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7
MIGRAPHX_COMMIT=2487b688c453b9007ac69ff1d79ad6cf9b509901
S=/work/src
mkdir -p "$S"
cd "$S"
if [ ! -d onnxruntime/.git ]; then
    git clone --branch v1.30.0 --depth 1 https://github.com/microsoft/onnxruntime.git onnxruntime
fi
test "$(git -C onnxruntime rev-parse HEAD)" = "$ORT_COMMIT"
git -C onnxruntime submodule update --init --recursive --depth 1
if [ ! -d AMDMIGraphX/.git ]; then
    git init -q AMDMIGraphX
    git -C AMDMIGraphX fetch -q --depth 1 https://github.com/ROCm/AMDMIGraphX.git "$MIGRAPHX_COMMIT"
    git -C AMDMIGraphX -c advice.detachedHead=false checkout -q FETCH_HEAD
fi
test "$(git -C AMDMIGraphX rev-parse HEAD)" = "$MIGRAPHX_COMMIT"
echo "onnxruntime $ORT_COMMIT, AMDMIGraphX $MIGRAPHX_COMMIT"
