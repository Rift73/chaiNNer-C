#!/bin/bash
# In the cnm-build image (build.sh's migraphx step): MIGraphX 2.16.1 (AMDMIGraphX release/rocm-rel-7.14 at 2487b688)
# against the ROCm 7.14.1 pip SDK with ROCm's clang (GCC Toolset 14's libstdc++, glibc 2.28): the GPU target with
# rocMLIR, MIOpen, rocBLAS, hipBLASLt and composable_kernel (AMD's Linux defaults), the Python module for CPython 3.14,
# no tests, device code for every gfx target ROCm 7.14.1 ships on Linux. The image's python-build-standalone 3.14.8
# provides the headers and, for MIGraphX's separate embedding library libmigraphx_py_3.14.so (which only MIGraphX's
# own driver opens), libpython; the extension module links no libpython.
# Install prefix: /work/migraphx-7.14, the --migraphx of native/tools/build_linux_release.sh.
set -euo pipefail
ROCM=$(cat /work/rocm-root.txt)
P=/work/deps
SRC=/work/src/AMDMIGraphX
B=/work/migraphx-build
I=/work/migraphx-7.14
JOBS=${JOBS:-12}
TARGETS="gfx908;gfx90a;gfx942;gfx950;gfx1010;gfx1011;gfx1012;gfx1030;gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036;"
TARGETS+="gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201;gfx1250"
GCCTC=--gcc-toolchain=/opt/rh/gcc-toolset-14/root/usr
# Libraries and the Python module bind their own function references (-Bsymbolic-functions) and keep the statically
# linked archives (SQLite, protobuf/abseil, rocMLIR, CK) local (--exclude-libs,ALL), so a process that already loaded
# other copies of msgpack/protobuf/LLVM (torch, Triton) neither redirects MIGraphX's calls nor gets MIGraphX's
# copies. Data symbols stay shared between MIGraphX's own libraries (op/compiler registries), as in AMD's builds.
LIBLINK="$GCCTC -Wl,-Bsymbolic-functions -Wl,--exclude-libs,ALL"
export ROCM_PATH=$ROCM HIP_PATH=$ROCM
test "$(git -C "$SRC" rev-parse HEAD)" = "2487b688c453b9007ac69ff1d79ad6cf9b509901"
echo "[$(date -Is)] MIGraphX $(git -C "$SRC" rev-parse HEAD), targets $TARGETS"
cmake -S "$SRC" -B "$B" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$I" -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_PREFIX_PATH="$P;$ROCM" \
    -DCMAKE_C_COMPILER="$ROCM/lib/llvm/bin/clang" -DCMAKE_CXX_COMPILER="$ROCM/lib/llvm/bin/clang++" \
    -DCMAKE_C_FLAGS="$GCCTC" -DCMAKE_CXX_FLAGS="$GCCTC" -DCMAKE_EXE_LINKER_FLAGS="$GCCTC" \
    -DCMAKE_SHARED_LINKER_FLAGS="$LIBLINK" -DCMAKE_MODULE_LINKER_FLAGS="$LIBLINK" \
    -DGPU_TARGETS="$TARGETS" -DBUILD_TESTING=OFF \
    -DMIGRAPHX_ENABLE_GPU=ON -DMIGRAPHX_ENABLE_CPU=OFF -DMIGRAPHX_ENABLE_FPGA=OFF \
    -DMIGRAPHX_ENABLE_PYTHON=ON -DPYTHON_DISABLE_VERSIONS="3.6;3.7;3.8;3.9;3.10;3.11;3.12;3.13" \
    -DPYTHON_CONFIG_3.14=/opt/pbs-python/bin/python3.14-config \
    -DPYTHON_3.14_EXECUTABLE=/opt/pbs-python/bin/python3.14 \
    -DMIGRAPHX_USE_MIOPEN=ON -DMIGRAPHX_USE_ROCBLAS=ON -DMIGRAPHX_USE_HIPBLASLT=ON \
    -DMIGRAPHX_USE_COMPOSABLEKERNEL=ON -DMIGRAPHX_ENABLE_MLIR=ON \
    -DCMAKE_JOB_POOLS="link_pool=2" -DCMAKE_JOB_POOL_LINK=link_pool \
    > "$B.configure.log" 2>&1 || { tail -60 "$B.configure.log"; exit 1; }
grep -E "Enable gpu|Build with rocMLIR|Python .* found|GPU_TARGETS|composable|MIOpen|hipBLASLt|rocBLAS|Eigen" \
    "$B.configure.log" | head -30
echo "[$(date -Is)] build"
cmake --build "$B" -j "$JOBS" > "$B.build.log" 2>&1 \
    || { grep -n -E "error|FAILED" "$B.build.log" | head -40; tail -40 "$B.build.log"; exit 1; }
cmake --install "$B" > "$B.install.log" 2>&1 || { tail -40 "$B.install.log"; exit 1; }
echo "[$(date -Is)] installed"
for f in "$I"/lib/libmigraphx*.so.* "$I"/lib/migraphx/lib/libmigraphx*.so.* "$I"/bin/migraphx-hiprtc-driver; do
    [ -L "$f" ] && continue
    echo "== $f"
    readelf -d "$f" | grep -E "NEEDED|RPATH|RUNPATH|SONAME"
    objdump -T "$f" | grep -o "GLIBC_[0-9.]*" | sort -uV | tail -1
    objdump -T "$f" | grep -o "GLIBCXX_[0-9.]*" | sort -uV | tail -1
done
echo "[$(date -Is)] MIGraphX done"
