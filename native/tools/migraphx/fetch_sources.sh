#!/bin/bash
# In the cnm-build image (build.sh's sources step): the sources into /work/src, each checked by its commit:
#   onnxruntime  ONNX Runtime's v1.31.1 tag (6af1a932) with its submodules;
#   AMDMIGraphX  release/rocm-rel-10.0 at becdb3da, MIGraphX 2.17.0 (the rocm-10.0 tag).
set -euo pipefail
ORT_COMMIT=6af1a932e2286d5c58166085d24c68783033edee
MIGRAPHX_COMMIT=becdb3da862f2297041b746b90bc6130e2b1d1f7
S=/work/src
mkdir -p "$S"
cd "$S"
if [ ! -d onnxruntime/.git ]; then
    git clone --branch v1.31.1 --depth 1 https://github.com/microsoft/onnxruntime.git onnxruntime
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
