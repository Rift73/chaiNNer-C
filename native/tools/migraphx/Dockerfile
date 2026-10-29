# cnm-build: the image that builds MIGraphX 2.16.1 for ROCm 7.14.1 and the ONNX Runtime 1.30.0 MIGraphX wheel
# (CPython 3.14, manylinux_2_28 x86-64); native/tools/migraphx/build.sh builds it as `docker build -t cnm-build
# native/tools/migraphx`. Base: PyPA manylinux_2_28 (AlmaLinux 8.10, glibc 2.28, GCC Toolset 14, CPython 3.14,
# auditwheel, patchelf), the image of the 1.30.0+trt11.4 Linux wheel. The ROCm 7.14.1 SDK (AMD's rocm-sdk-* pip wheels,
# built for glibc 2.28) is installed into a venv of the mounted work directory by setup_sdk.sh, not baked in.
ARG MANYLINUX_IMAGE=quay.io/pypa/manylinux_2_28_x86_64@sha256:e0adf18521ad35192df733a6c62d35df7801957d24e0470238aacecfab2188a3

FROM ${MANYLINUX_IMAGE}

COPY build-requirements.txt /tmp/req/
# Build-time Python packages (versions of the 1.30.0+trt11.4 builds: cmake 3.31.6, ninja, numpy 2.5.3).
RUN /opt/python/cp314-cp314/bin/python3.14 -m pip install --no-cache-dir --no-deps --require-hashes \
        -r /tmp/req/build-requirements.txt \
 && rm -rf /tmp/req

# CPython 3.14.8 (python-build-standalone 20261003, chaiNNer-C's Linux CPython; native/linux/Dockerfile pins the same
# archive): rocMLIR configures its Python bindings on Linux (never shipped) and MIGraphX builds its embedding library
# libmigraphx_py_3.14.so, and both need a CPython with libpython, which manylinux's builds do not have.
RUN curl -fsSL --retry 3 -o /tmp/python.tar.gz \
        https://github.com/astral-sh/python-build-standalone/releases/download/20261003/cpython-3.14.8%2B20261003-x86_64-unknown-linux-gnu-install_only.tar.gz \
 && echo "371b6c281bbb09b29279e9e3a2996bab4ae2ea03cca52bf869f8bd89286b0ae8  /tmp/python.tar.gz" | sha256sum -c - \
 && mkdir /opt/pbs-python \
 && tar -xzf /tmp/python.tar.gz -C /opt/pbs-python --strip-components=1 \
 && rm /tmp/python.tar.gz

# ONNX Runtime 1.30.0's setup.py excludes only the ROCm 5/6 sonames from `auditwheel repair`; this build links ROCm 7
# (AMD's pip packages provide them at run time), so a wrapper adds the ROCm 7 sonames and the MIGraphX 2.16 ones
# (finalize_wheel.py adds MIGraphX to the wheel, with a run path to the ROCm pip packages).
COPY auditwheel-rocm7 /usr/local/ort-shim/auditwheel
RUN chmod 755 /usr/local/ort-shim/auditwheel

ENV PATH=/usr/local/ort-shim:/opt/python/cp314-cp314/bin:/opt/rh/gcc-toolset-14/root/usr/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    AUDITWHEEL_PLAT=manylinux_2_28_x86_64
