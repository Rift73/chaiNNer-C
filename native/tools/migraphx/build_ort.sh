#!/bin/bash
# In the cnm-build image (build.sh's ort step): the ONNX Runtime 1.30.0 Python wheel (CPU and MIGraphX execution
# providers) from the v1.30.0 tag plus ort-migraphx-cache-name.patch and ort-migraphx-shape-programs.patch, in that
# order and re-applied from the tag on every run, against MIGraphX 2.16.1 (/work/migraphx-7.14) and the ROCm 7.14.1 pip
# SDK. Flags: those of the 1.30.0+trt11.4 Linux wheel with the CUDA/TensorRT options replaced by --use_migraphx
# (--enable_lto --use_binskim_compliant_compile_flags, GCC Toolset 14, unit tests not built); a link pool of 1, so the
# large links run one at a time. setup.py names the distribution onnxruntime-migraphx.
# The ROCm root is NOT on CMAKE_PREFIX_PATH: it ships flatbuffers 25 (for hipDNN), which would replace ONNX Runtime's
# own flatbuffers 23 dependency; the MIGraphX provider's CMake finds HIP through $HIP_PATH instead.
set -euo pipefail
ROCM=$(cat /work/rocm-root.txt)
I=/work/migraphx-7.14
S=/work/src/onnxruntime
PY=/opt/python/cp314-cp314/bin/python3.14
export ONNX_ML=1
export CMAKE_ARGS="-DONNX_GEN_PB_TYPE_STUBS=ON -DONNX_WERROR=OFF"
export ROCM_PATH=$ROCM HIP_PATH=$ROCM
export CC=/opt/rh/gcc-toolset-14/root/usr/bin/gcc CXX=/opt/rh/gcc-toolset-14/root/usr/bin/g++

cd "$S"
test "$(git rev-parse HEAD)" = "f2c39fe2f838cf35ce7da92824f5a5e3ee6e88a7"
git checkout HEAD -- onnxruntime/core/providers/migraphx
git apply /recipe/ort-migraphx-cache-name.patch
git apply /recipe/ort-migraphx-shape-programs.patch
git status --porcelain
gcc --version | head -1
cmake --version | head -1
"$PY" --version

echo "[$(date -Is)] build start"
"$PY" tools/ci_build/build.py \
    --build_dir /work/ort-build --config Release --update --build \
    --skip_submodule_sync --skip_tests --parallel "${JOBS:-12}" \
    --cmake_generator Ninja \
    --enable_pybind --build_wheel --enable_lto --use_binskim_compliant_compile_flags \
    --use_migraphx --migraphx_home "$I" \
    --compile_no_warning_as_error \
    --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF "CMAKE_PREFIX_PATH=$I" \
        CMAKE_JOB_POOLS=link_pool=1 CMAKE_JOB_POOL_LINK=link_pool
ls -la /work/ort-build/Release/dist
echo "[$(date -Is)] ONNX Runtime done"
