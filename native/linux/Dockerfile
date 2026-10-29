# cnl-build: the image that builds chaiNNer-C's Linux release (native/tools/build_linux_release.sh builds it as
# `docker build -t cnl-build native/linux`). The binaries it links need glibc 2.28 or newer and nothing else from the
# system: Rocky/Alma/RHEL 8+, Debian 10+, Ubuntu 20.04+ and every newer distribution (native/tools/check_glibc.py
# checks it).
#
# LLVM 23.1.2's official Linux binaries need glibc 2.34 and ICU 70 to run, so the image is Ubuntu 22.04, and clang
# compiles and links against a glibc 2.28 sysroot instead of the image's own glibc: Rocky Linux 8.10's glibc, kernel
# headers and GCC Toolset 14 (libstdc++ 14 and libgcc, which the modules link statically). clang.cfg and clang++.cfg
# beside the compilers pass --sysroot and --gcc-install-dir to every clang call. nvcc's host compiler is the same
# toolset's g++ (/opt/gcc/bin/g++, against the same sysroot), since nvcc 13 refuses clang 23.
#
# Every download is pinned by version and SHA-256 (fetch.sh): the base image by digest, Ubuntu's packages through the
# snapshot service at a fixed time (apt checks the signed indexes' SHA-256), the rest by their publishers' hashes.
FROM ubuntu:22.04@sha256:e0f80c829509ef4df9e76bd8944eb7b8ec25cbb379b112a7f9448b2c77a77c9b

ARG UBUNTU_SNAPSHOT=20261028T000000Z
# ca-certificates comes from jammy's release pocket, which never changes; the snapshot service needs HTTPS.
RUN set -eu; \
    echo 'deb http://archive.ubuntu.com/ubuntu jammy main' > /etc/apt/sources.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates; \
    for pocket in jammy jammy-updates jammy-security; do \
        echo "deb https://snapshot.ubuntu.com/ubuntu/$UBUNTU_SNAPSHOT $pocket main universe"; \
    done > /etc/apt/sources.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends cpio curl git libicu70 make rpm2cpio unzip xz-utils zip zstd; \
    rm -rf /var/lib/apt/lists/*
COPY fetch.sh /usr/local/bin/fetch


# LLVM 23.1.2: the compilers, lld, the archivers and binary tools, llvm-ifs (the TensorRT stubs below) and clang's
# resource directory.
RUN set -eu; \
    fetch https://github.com/llvm/llvm-project/releases/download/llvmorg-23.1.2/LLVM-23.1.2-Linux-X64.tar.zst \
        6382de1c1a210ce5a5cc49d18bc8444d137742e7cbf9b19f4ae602bb1ab52534 /tmp/llvm.tar.zst; \
    mkdir /opt/llvm; \
    tar -I 'zstd --long=31' -xf /tmp/llvm.tar.zst -C /opt/llvm --strip-components=1 --wildcards \
        '*/bin/clang' '*/bin/clang++' '*/bin/clang-23' '*/bin/ld.lld' '*/bin/lld' '*/bin/llvm-ar' \
        '*/bin/llvm-ranlib' '*/bin/llvm-nm' '*/bin/llvm-objcopy' '*/bin/llvm-strip' '*/bin/llvm-objdump' \
        '*/bin/llvm-readobj' '*/bin/llvm-readelf' '*/bin/llvm-ifs' '*/lib/clang/23/*'; \
    rm /tmp/llvm.tar.zst

# CUDA 13.4.92 (the Windows build's nvcc): the redistributable archives of CUDA 13.4.2 that nvcc and the static
# runtime need, merged into one toolkit root.
ARG CUDA_REDIST=https://developer.download.nvidia.com/compute/cuda/redist
RUN set -eu; \
    mkdir /opt/cuda; \
    for item in \
        cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-13.4.92-archive.tar.xz:60998f40cc9df5826b2a846e2dcc328133fbc5ef539398ce0a21cf5dcac7193d \
        cuda_crt/linux-x86_64/cuda_crt-linux-x86_64-13.4.92-archive.tar.xz:c969e61ded12dbf0cf20ace72867f1622ec0e8ea1e1d305e27761012025f7aaa \
        libnvvm/linux-x86_64/libnvvm-linux-x86_64-13.4.92-archive.tar.xz:0e619cf3d5b27f7c812665599ca9761fc296e56129b24a85d822a5d8a1a3bd75 \
        cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-13.4.92-archive.tar.xz:0ac5dbc538d04e9983bc493b410cce4b459e1ee9f5f6654b6464ef7b3e14a8b5 \
        cccl/linux-x86_64/cccl-linux-x86_64-13.3.4.3.1-archive.tar.xz:6b7516074f42f80dd3feb9c11318c8e1c5ab56cdd9f18fff4c5ace2e7e0a8209 \
    ; do \
        fetch "$CUDA_REDIST/${item%%:*}" "${item##*:}" /tmp/cuda.tar.xz; \
        tar -xJf /tmp/cuda.tar.xz -C /opt/cuda --strip-components=1; \
    done; \
    ln -s lib /opt/cuda/lib64; \
    rm /tmp/cuda.tar.xz

# TensorRT 11.4.0.106 for CUDA 13.4 from NVIDIA's public Ubuntu 22.04 repository: the headers (NvInferPlugin.h
# included) and a link stub of libnvinfer (llvm-ifs: its dynamic symbols and soname only), the one TensorRT library the
# binaries link. They never carry TensorRT; the application loads the libraries of the tensorrt package it installs.
ARG CUDA_REPO=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64
RUN set -eu; \
    mkdir -p /tmp/trt /opt/tensorrt/lib; \
    for item in \
        libnvinfer-headers-dev:8e7d07c8a62139842f321e2cb26d123b52378597bd9062890d2fbdc9beef1c84 \
        libnvinfer-headers-plugin-dev:d0afddcbd89703f9c6bd42e111478b52d4b20dd60c8c4f75041e7f4a188cb926 \
        libnvinfer11:bd24db760728779e980c8852ba87e148a0bf9fb4e69d69bc14f3bdabc0423797 \
    ; do \
        fetch "$CUDA_REPO/${item%%:*}_11.4.0.106-1+cuda13.4_amd64.deb" "${item##*:}" /tmp/trt.deb; \
        dpkg-deb -x /tmp/trt.deb /tmp/trt; \
        rm /tmp/trt.deb; \
    done; \
    mv /tmp/trt/usr/include/x86_64-linux-gnu /opt/tensorrt/include; \
    /opt/llvm/bin/llvm-ifs --input-format=ELF --output-elf=/opt/tensorrt/lib/libnvinfer.so.11.4.0 \
        /tmp/trt/usr/lib/x86_64-linux-gnu/libnvinfer.so.11.4.0; \
    ln -s libnvinfer.so.11.4.0 /opt/tensorrt/lib/libnvinfer.so.11; \
    ln -s libnvinfer.so.11 /opt/tensorrt/lib/libnvinfer.so; \
    rm -rf /tmp/trt

# VapourSynth R81's API 4 header (the VapourSynth package's version), for the TensorRT runner's VapourSynth plugin.
RUN fetch https://raw.githubusercontent.com/vapoursynth/vapoursynth/R81/include/VapourSynth4.h \
        d079c0e1602718204c36fe743079cf6667f832bc98f28a2bd54ff19b44cfc1db /tmp/VapourSynth4.h \
    && install -D -m 644 /tmp/VapourSynth4.h /opt/vapoursynth/include/VapourSynth4.h && rm /tmp/VapourSynth4.h

# CMake 3.31.6 and Ninja 1.12.1, the Windows build's versions.
RUN set -eu; \
    fetch https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz \
        5a1133ff103c71eb5120e2cc3de922733e7d8a26a98ae716397e8676adb367bf /tmp/cmake.tar.gz; \
    mkdir /opt/cmake; \
    tar -xzf /tmp/cmake.tar.gz -C /opt/cmake --strip-components=1 --exclude='*/doc' --exclude='*/man'; \
    fetch https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-linux.zip \
        6f98805688d19672bd699fbbfa2c2cf0fc054ac3df1f0e6a47664d963d530255 /tmp/ninja.zip; \
    unzip -q /tmp/ninja.zip -d /opt/cmake/bin; \
    rm /tmp/cmake.tar.gz /tmp/ninja.zip

# CPython 3.14.8 (python-build-standalone 20261003, the lock's interpreter) with the lock's NumPy: the modules'
# headers, and the interpreter of the build's own tools.
RUN set -eu; \
    fetch https://github.com/astral-sh/python-build-standalone/releases/download/20261003/cpython-3.14.8%2B20261003-x86_64-unknown-linux-gnu-install_only.tar.gz \
        371b6c281bbb09b29279e9e3a2996bab4ae2ea03cca52bf869f8bd89286b0ae8 /tmp/python.tar.gz; \
    mkdir /opt/python; \
    tar -xzf /tmp/python.tar.gz -C /opt/python --strip-components=1; \
    rm /tmp/python.tar.gz; \
    echo 'numpy==2.5.3 --hash=sha256:b0521d0f4aebb6e06189451025fa17a913287b13c03d5fe05c017333b654ea5b' > /tmp/numpy.txt; \
    /opt/python/bin/python3 -I -m pip install --disable-pip-version-check --no-input --no-cache-dir \
        --only-binary=:all: --require-hashes -r /tmp/numpy.txt; \
    rm /tmp/numpy.txt

# Node.js 24.22.1 (.nvmrc's major, the Windows release's version) for npm ci and Electron Forge.
RUN set -eu; \
    fetch https://nodejs.org/dist/v24.22.1/node-v24.22.1-linux-x64.tar.xz \
        6c172a9ab6718f7b801fb5a7bd486af2ee5e32758086a04b84fcce2aa77095e6 /tmp/node.tar.xz; \
    mkdir /opt/node; \
    tar -xJf /tmp/node.tar.xz -C /opt/node --strip-components=1; \
    rm /tmp/node.tar.xz

# The glibc 2.28 sysroot: Rocky Linux 8.10's packages, unpacked under /opt/sysroot with Rocky's merged /usr layout,
# and the GMP, MPFR and MPC libraries that the toolset's compiler runs with (/opt/gcc/lib, read by g++ below).
# ROCKY_MIRROR may name the vault (https://dl.rockylinux.org/vault/rocky/8.10) once 8.10 leaves the main mirror.
ARG ROCKY_MIRROR=https://dl.rockylinux.org/pub/rocky/8.10
RUN set -eu; \
    mkdir -p /opt/sysroot/usr/lib64 /opt/sysroot/usr/lib /opt/sysroot/usr/bin /opt/sysroot/usr/sbin; \
    for dir in lib64 lib bin sbin; do ln -s "usr/$dir" "/opt/sysroot/$dir"; done; \
    cd /opt/sysroot; \
    for item in \
        BaseOS/x86_64/os/Packages/g/glibc-2.28-251.el8_10.48.x86_64.rpm:ac6b703086afead9a4cfeb5cf50d0974056e8c810f059d17dba0e050cf2ed477 \
        BaseOS/x86_64/os/Packages/g/glibc-devel-2.28-251.el8_10.48.x86_64.rpm:ba67a4c21c184e5a5fd31bfe6cdd879196b98b8745d24e7d951d4f23281f3f31 \
        BaseOS/x86_64/os/Packages/g/glibc-headers-2.28-251.el8_10.48.x86_64.rpm:8a2496f1385a372db6a66c48b7f71678f97b3815ce0553711e81118d5b9a990c \
        BaseOS/x86_64/os/Packages/k/kernel-headers-4.18.0-553.179.1.el8_10.x86_64.rpm:905de4a4c2b39ccdd8f59188c953010b3666a36b731730edb6cee1e607357b66 \
        BaseOS/x86_64/os/Packages/l/libgcc-8.5.0-30.el8_10.x86_64.rpm:8e8b889b7f2c7c1812b50758b9199857fc7e9631776ca0605d81b69bcb9e1ca7 \
        BaseOS/x86_64/os/Packages/l/libstdc++-8.5.0-30.el8_10.x86_64.rpm:ca9bdd0b2fc20d073d6d22167c4e71d4010fdf0ba0d9d8612cd96a4fd90766a5 \
        AppStream/x86_64/os/Packages/g/gcc-toolset-14-gcc-14.2.1-11.el8_10.x86_64.rpm:f0e96cad0302503088b5d7731a44c1921539e4abea64eb4a7760c1f0871f3625 \
        AppStream/x86_64/os/Packages/g/gcc-toolset-14-libstdc++-devel-14.2.1-11.el8_10.x86_64.rpm:18f42df5bb962d2f0d98ef3eaebb859270ca2eece65d0dc35b3c47bf2056e573 \
        AppStream/x86_64/os/Packages/g/gcc-toolset-14-gcc-c++-14.2.1-11.el8_10.x86_64.rpm:7554b0e8511498aeb130f3e1e4efa2103b173a7fec2b7c928046e0ef06cffc36 \
        AppStream/x86_64/os/Packages/g/gcc-toolset-14-binutils-2.41-4.el8_10.1.x86_64.rpm:07dc612539d7bfa2ff002df3441b2ecc1ebee1b80fd4340a84f9a11cb520b9a3 \
    ; do \
        fetch "$ROCKY_MIRROR/${item%%:*}" "${item##*:}" /tmp/package.rpm; \
        rpm2cpio /tmp/package.rpm | cpio -idm --quiet; \
    done; \
    rm -rf usr/share opt/rh/gcc-toolset-14/root/usr/share opt/rh/gcc-toolset-14/root/usr/lib/gcc/x86_64-redhat-linux/14/32; \
    mkdir -p /tmp/gcc-libraries /opt/gcc/lib; \
    cd /tmp/gcc-libraries; \
    for item in \
        BaseOS/x86_64/os/Packages/g/gmp-6.1.2-12.el8.x86_64.rpm:df8ebe29fce2c3e431bcb5fb169ff8d72cb952aabb945a8ddd0cfb1a7900078f \
        BaseOS/x86_64/os/Packages/m/mpfr-3.1.6-1.el8.x86_64.rpm:64294d87f40c5e7ac7db7d3d7d4b80b7de6a8b9a8ebd4efca104fbb558fbcb94 \
        AppStream/x86_64/os/Packages/l/libmpc-1.1.0-9.1.el8.x86_64.rpm:4fbf751702394211896af9f253ad2f38c80f603e007a7c9ed2b9c8e5d97f2f95 \
    ; do \
        fetch "$ROCKY_MIRROR/${item%%:*}" "${item##*:}" /tmp/package.rpm; \
        rpm2cpio /tmp/package.rpm | cpio -idm --quiet; \
    done; \
    cp -a usr/lib64/libgmp.so.* usr/lib64/libmpfr.so.* usr/lib64/libmpc.so.* /opt/gcc/lib/; \
    cd /; \
    rm -rf /tmp/package.rpm /tmp/gcc-libraries
RUN printf '%s\n' --sysroot=/opt/sysroot \
        --gcc-install-dir=/opt/sysroot/opt/rh/gcc-toolset-14/root/usr/lib/gcc/x86_64-redhat-linux/14 \
        | tee /opt/llvm/bin/clang.cfg > /opt/llvm/bin/clang++.cfg
COPY g++ /opt/gcc/bin/g++
RUN chmod 755 /opt/gcc/bin/g++

# NASM 3.01 (libjpeg-turbo's SIMD; Ubuntu's is older), built with the sysroot's clang.
RUN set -eu; \
    fetch https://www.nasm.us/pub/nasm/releasebuilds/3.01/nasm-3.01.tar.xz \
        b7324cbe86e767b65f26f467ed8b12ad80e124e3ccb89076855c98e43a9eddd4 /tmp/nasm.tar.xz; \
    mkdir /tmp/nasm; \
    tar -xJf /tmp/nasm.tar.xz -C /tmp/nasm --strip-components=1; \
    cd /tmp/nasm; \
    CC=/opt/llvm/bin/clang LDFLAGS=-fuse-ld=lld AR=/opt/llvm/bin/llvm-ar RANLIB=/opt/llvm/bin/llvm-ranlib \
        ./configure --prefix=/opt/nasm > /dev/null; \
    make -j8 nasm > /dev/null; \
    install -D -m 755 nasm /opt/nasm/bin/nasm; \
    cd /; \
    rm -rf /tmp/nasm.tar.xz /tmp/nasm

ENV PATH=/opt/llvm/bin:/opt/cmake/bin:/opt/nasm/bin:/opt/cuda/bin:/opt/node/bin:/opt/python/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
