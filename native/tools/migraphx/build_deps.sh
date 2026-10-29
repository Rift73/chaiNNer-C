#!/bin/bash
# In the cnm-build image (build.sh's deps step): MIGraphX 2.16.1's third-party dependencies at the versions its
# requirements.txt pins (the recipe AMD's rbuild/cget uses), built with ROCm 7.14.1's clang on GCC Toolset 14's
# libstdc++ into /work/deps. Static and position-independent where a library is linked, so libmigraphx*.so needs no
# extra .so. rocMLIR is built as the fat static rockCompiler library (BUILD_FAT_LIBROCKCOMPILER), as for AMD's own
# packages; LLVM's optional zlib/zstd/terminfo/libxml2 links are off (not manylinux libraries, and MIGraphX does not
# use them). Every archive is checked against its SHA-256.
set -euo pipefail
ROCM=$(cat /work/rocm-root.txt)
P=/work/deps
SRC=/work/deps-src
BLD=/work/deps-build
JOBS=${JOBS:-12}
mkdir -p "$P" "$SRC" "$BLD"
export CC=$ROCM/lib/llvm/bin/clang CXX=$ROCM/lib/llvm/bin/clang++
GCCTC=--gcc-toolchain=/opt/rh/gcc-toolset-14/root/usr
COMMON=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$P" -DCMAKE_PREFIX_PATH="$P;$ROCM"
        -DCMAKE_C_FLAGS="$GCCTC" -DCMAKE_CXX_FLAGS="$GCCTC" -DCMAKE_EXE_LINKER_FLAGS="$GCCTC"
        -DCMAKE_SHARED_LINKER_FLAGS="$GCCTC" -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_LIBDIR=lib)

fetch() {  # fetch <name> <url> <sha256>: download and check once, unpack to $SRC/<name>, print the source directory
    local name=$1 url=$2 sha256=$3
    if [ ! -d "$SRC/$name" ]; then
        curl -fsSL --retry 3 -o "$SRC/$name.archive" "$url"
        echo "$sha256  $SRC/$name.archive" | sha256sum -c - >&2
        mkdir -p "$SRC/$name"
        case "$url" in
            *.zip) (cd "$SRC/$name" && unzip -q "../$name.archive") ;;
            *) tar -xzf "$SRC/$name.archive" -C "$SRC/$name" ;;
        esac
    fi
    find "$SRC/$name" -mindepth 1 -maxdepth 1 -type d | head -1
}

cmbuild() {  # cmbuild <name> <source dir> [cmake args...]
    local name=$1 dir=$2
    shift 2
    if [ -f "$BLD/$name.done" ]; then echo "[$(date -Is)] $name: already built"; return; fi
    echo "[$(date -Is)] $name: configure"
    cmake -S "$dir" -B "$BLD/$name" "${COMMON[@]}" "$@" > "$BLD/$name.configure.log" 2>&1 \
        || { tail -40 "$BLD/$name.configure.log"; return 1; }
    echo "[$(date -Is)] $name: build"
    cmake --build "$BLD/$name" -j "$JOBS" > "$BLD/$name.build.log" 2>&1 \
        || { tail -60 "$BLD/$name.build.log"; return 1; }
    cmake --install "$BLD/$name" > "$BLD/$name.install.log" 2>&1 || { tail -40 "$BLD/$name.install.log"; return 1; }
    touch "$BLD/$name.done"
    echo "[$(date -Is)] $name: done"
}

"$CXX" --version | head -1
D=$(fetch rocm-cmake https://github.com/ROCm/rocm-cmake/archive/1d4652ae2ec0e44a67a7c415dd7e51c88a6aa68d.tar.gz \
    77e583f4154e521be102798b77ee958fd04b2b07ab6fdbd545e3e364024be154)
cmbuild rocm-cmake "$D"
D=$(fetch abseil https://github.com/abseil/abseil-cpp/archive/20250512.0.tar.gz \
    7262daa7c1711406248c10f41026d685e88223bc92817d16fb93c19adb57f669)
cmbuild abseil "$D" -DABSL_ENABLE_INSTALL=ON -DABSL_PROPAGATE_CXX_STD=ON -DCMAKE_CXX_STANDARD=17
D=$(fetch protobuf https://github.com/protocolbuffers/protobuf/archive/v30.0.tar.gz \
    9df0e9e8ebe39f4fbbb9cf7db3d811287fe3616b2f191eb2bf5eaa12539c881f)
cmbuild protobuf "$D" -Dprotobuf_BUILD_TESTS=Off -Dprotobuf_BUILD_SHARED_LIBS=Off -Dprotobuf_BUILD_LIBPROTOC=On \
    -Dprotobuf_BUILD_PROTOC_BINARIES=On -Dprotobuf_BUILD_PROTOBUF_BINARIES=On -Dprotobuf_ABSL_PROVIDER=package \
    -Dprotobuf_WITH_ZLIB=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5
D=$(fetch json https://github.com/nlohmann/json/archive/v3.8.0.tar.gz \
    7d0edf65f2ac7390af5e5a0b323b31202a6c11d744a74b588dc30f5a8c9865ba)
cmbuild json "$D" -DJSON_BuildTests=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5
D=$(fetch pybind11 https://github.com/pybind/pybind11/archive/3e9dfa2866941655c56877882565e7577de6fc7b.tar.gz \
    9a7d245f405f470798b9d2a48912cc97230658024775299eac203f7c9c9ae37c)
cmbuild pybind11 "$D" -DPYBIND11_TEST=OFF -DPython_EXECUTABLE=/opt/python/cp314-cp314/bin/python3.14
D=$(fetch msgpack https://github.com/msgpack/msgpack-c/archive/cpp-3.3.0.tar.gz \
    754c3ace499a63e45b77ef4bcab4ee602c2c414f58403bce826b76ffc2f77d0b)
cmbuild msgpack "$D" -DMSGPACK_BUILD_TESTS=Off -DMSGPACK_BUILD_EXAMPLES=Off -DMSGPACK_ENABLE_SHARED=OFF \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
D=$(fetch eigen https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz \
    e9c326dc8c05cd1e044c71f30f1b2e34a6161a3b6ecf445d56b53ff1669e3dec)
cmbuild eigen "$D" -DBUILD_TESTING=Off -DEIGEN_BUILD_DOC=Off -DEIGEN_BUILD_LAPACK=Off -DEIGEN_BUILD_BLAS=Off

# SQLite 3.50.4 amalgamation as a static PIC library (cget's sqlite3 recipe builds the same amalgamation).
if [ ! -f "$BLD/sqlite.done" ]; then
    D=$(fetch sqlite https://www.sqlite.org/2025/sqlite-amalgamation-3500400.zip \
        1d3049dd0f830a025a53105fc79fd2ab9431aea99e137809d064d8ee8356b032)
    mkdir -p "$BLD/sqlite" "$P/include" "$P/lib"
    "$CC" $GCCTC -O2 -fPIC -DSQLITE_THREADSAFE=1 -DSQLITE_OMIT_LOAD_EXTENSION -c "$D/sqlite3.c" \
        -o "$BLD/sqlite/sqlite3.o"
    ar rcs "$P/lib/libsqlite3.a" "$BLD/sqlite/sqlite3.o"
    cp "$D/sqlite3.h" "$D/sqlite3ext.h" "$P/include/"
    touch "$BLD/sqlite.done"
    echo "[$(date -Is)] sqlite: done"
fi

# composable_kernel's host-side JIT library (codegen subdirectory), as MIGraphX's requirements pin it.
D=$(fetch ck https://github.com/ROCm/composable_kernel/archive/ad0db05b040bacda751c65c705261b8a0a7ed25d.tar.gz \
    cc710fe25418996f1975b3c59033e0472829592b370667124dd687d910a6ef7c)
cmbuild ck "$D/codegen" -DBUILD_TESTING=Off -DCMAKE_POLICY_VERSION_MINIMUM=3.5

# rocMLIR always configures its Python bindings on Linux (not part of librockCompiler, not shipped); they need a
# CPython with libpython, the image's python-build-standalone 3.14.8.
D=$(fetch rocmlir https://github.com/ROCm/rocMLIR/archive/518955cab3ec2a53cbb03216cb55ac6ab47ee1da.tar.gz \
    e6a8e0d7499db384e0daa7225f5563bf09d6eabb6f2007c345ebd4e423b8dd57)
cmbuild rocmlir "$D" -DBUILD_FAT_LIBROCKCOMPILER=On -DLLVM_INCLUDE_TESTS=Off -DROCMLIR_PARALLEL_LINK_JOBS=2 \
    -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_LIBXML2=OFF \
    -DLLVM_ENABLE_LIBEDIT=OFF -DLLVM_ENABLE_LIBPFM=OFF \
    -DPython3_EXECUTABLE=/opt/pbs-python/bin/python3.14 -DPython3_LIBRARY=/opt/pbs-python/lib/libpython3.14.so \
    -DPython3_INCLUDE_DIR=/opt/pbs-python/include/python3.14 -Dpybind11_DIR="$P/share/cmake/pybind11"
echo "[$(date -Is)] deps done"
