# The AMD kit's settings and helpers, sourced by its scripts (native/AMD.md). The lists and URLs a release changes are
# at the top.

# The test files with amd_gpu tests. The kit runs every amd_gpu test it collects from backend/tests and native/tests;
# a file listed here that gives none (renamed, or it failed to import) is named in the report.
AMD_GPU_TEST_FILES=(
    backend/tests/test_amd_gpu_devices.py
    backend/tests/test_attention_export_migraphx.py
    backend/tests/test_migraphx_drft.py
    backend/tests/test_migraphx_dual_gpu.py
    backend/tests/test_migraphx_dual_ops.py
    backend/tests/test_migraphx_gpu.py
    backend/tests/test_migraphx_runner.py
    backend/tests/test_onnx_migraphx_gpu.py
    backend/tests/test_pytorch_attention_amd.py
    backend/tests/test_pytorch_drft_rocm.py
    backend/tests/test_pytorch_dual_rocm.py
    backend/tests/test_vapoursynth_upscale.py
    native/tests/test_vosr2_migraphx_gpu.py
)
# MIGraphX's example custom-op library (native/migraphx_runner/tests/example_ops.hip), which the runner's GPU custom-op
# test loads (CHAINNER_C_MIGRAPHX_TEST_OPS); the app's zip does not carry it.
EXAMPLE_OPS=test-ops/chainner_example_migx_ops.so
# Wheels the kit installs before the dependency manager runs, from a copy in --wheels when there is one (a file name
# matching the pattern), else from the URL (empty: the dependency manager installs its own pin). An installed wheel of
# the version a package lists is one the dependency manager keeps.
ORT_MIGRAPHX_WHEEL_PATTERN="onnxruntime_migraphx-*.whl"
ORT_MIGRAPHX_WHEEL_URL="https://github.com/Rift73/onnxruntime/releases/download/v1.30.0-rocm7.14/onnxruntime_migraphx-1.30.0%2Brocm7.14-cp314-cp314-manylinux_2_28_x86_64.whl"
ORT_MIGRAPHX_WHEEL_SHA256="d9dc6fbe7acb8d7b4a40cf33bb7345bae92b154bd9da7023e67c66c1d11cf153"
SPANDREL_WHEEL_PATTERN="spandrel-*.whl"
SPANDREL_WHEEL_URL="https://github.com/Rift73/spandrel/releases/download/v0.4.2-c11/spandrel-0.4.2%2Bc11-py3-none-any.whl"
SPANDREL_WHEEL_SHA256="77467a3fbd2e915ee83b52ec10546714c6820304c7983d4a55d5185952b08cb2"
# The ESRGAN the kit downloads when --models has none: Real-ESRGAN's 4x anime model (RRDB, 6 blocks, 17.9 MB).
ESRGAN_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
ESRGAN_SHA256="f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da"
# The GPUs ROCm 7.14.1 has kernels for on Linux (AMD's whl-multi-arch index).
ROCM_TARGETS="gfx908 gfx90a gfx942 gfx950 gfx1010 gfx1011 gfx1012 gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1153 gfx1200 gfx1201 gfx1250"
# The packages the kit installs through the dependency manager (VapourSynth for its AI Upscale tests on AMD GPUs)
KIT_PACKAGES=(chaiNNer_pytorch chaiNNer_onnx chaiNNer_ncnn chaiNNer_migraphx chaiNNer_vapoursynth)
# Free disk (GiB) the kit needs in its work folder: the app, PyTorch with ROCm, ROCm's libraries and kernels, ONNX
# Runtime with MIGraphX, VOSR's model (7 GB) and FP8 graphs (3.5 GB), and compiled programs.
KIT_MIN_DISK_GB=40

KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$KIT_DIR/../../.." && pwd)
WORK=${KIT_WORK:-$HOME/amd-kit}
OUT=$WORK/out
LOGS=$OUT/logs
APP=$WORK/app/chaiNNer-linux-x64
APP_PYTHON=$APP/python/python/bin/python3.14
TEST_PYTHON=$WORK/test-env/bin/python
# The MIGraphX programs the chains and the speed table compile, in one cache
PROGRAMS=$APP/backend-storage/kit-migraphx-programs
DRY_RUN=${KIT_DRY_RUN:-0}
mkdir -p "$LOGS"

say() {
    printf '%s %s\n' "$(date +%H:%M:%S)" "$*"
}

die() {
    say "error: $*" >&2
    exit 1
}

# Runs a command, or in a dry run prints it.
run() {
    if [ "$DRY_RUN" = 1 ]; then
        say "[dry run] $*"
        return 0
    fi
    "$@"
}

# The report fragment of a step: $OUT/<name>.md, written by the step and assembled by report.sh.
fragment() {
    printf '%s/%s.md' "$OUT" "$1"
}

# A file's SHA-256, for the report.
sha256_of() {
    sha256sum "$1" | cut -d ' ' -f 1
}

# The app's backend, started as the app starts it (the dependency manager's requests go to it): backend_start LOG
# returns once it is ready (status 1 if it never is), `backend GET|POST PATH [JSON]` prints a response, backend_stop
# asks it to shut down and waits.
BACKEND_PORT=8765
backend_pid=""
backend_start() {
    PYTHONNOUSERSITE=1 "$APP_PYTHON" "$APP/resources/src/run.py" "$BACKEND_PORT" --storage-dir "$APP/backend-storage" \
        >"$1" 2>&1 </dev/null &
    backend_pid=$!
    local _
    for _ in $(seq 600); do
        backend GET /status 2>/dev/null | grep -q '"ready": *true' && return 0
        kill -0 "$backend_pid" 2>/dev/null || return 1
        sleep 1
    done
    return 1
}

backend() {
    if [ "$1" = POST ]; then
        local data=${3:-}
        [ -n "$data" ] || data='{}'
        curl -sS --max-time 5400 -X POST -H 'Content-Type: application/json' -d "$data" \
            "http://127.0.0.1:$BACKEND_PORT$2" 2>&1
    else
        curl -sS --max-time 600 "http://127.0.0.1:$BACKEND_PORT$2" 2>&1
    fi
}

backend_stop() {
    [ -n "$backend_pid" ] || return 0
    backend POST /shutdown >/dev/null
    local _
    for _ in $(seq 60); do
        kill -0 "$backend_pid" 2>/dev/null || break
        sleep 1
    done
    kill "$backend_pid" 2>/dev/null
    wait "$backend_pid" 2>/dev/null
    backend_pid=""
}

# A field of /etc/os-release (ID, ID_LIKE, PRETTY_NAME), without its quotes.
os_release() {
    sed -n "s/^$1=//p" /etc/os-release 2>/dev/null | tr -d '"'
}

# The AMD GPUs of the KFD topology, one "node gfx vendor:device" line each (gfx_target_version is major * 10000 +
# minor * 100 + stepping; LLVM names it with the major in decimal and a hex digit each for the others).
kfd_gpus() {
    local node properties target
    for node in /sys/class/kfd/kfd/topology/nodes/*; do
        properties=$node/properties
        [ -r "$properties" ] || continue
        target=$(awk '$1 == "gfx_target_version" { print $2 }' "$properties")
        if [ -z "$target" ] || [ "$target" = 0 ]; then
            continue
        fi
        printf '%s gfx%d%x%x %04x:%04x\n' "${node##*/}" $((target / 10000)) $((target / 100 % 100)) $((target % 100)) \
            "$(awk '$1 == "vendor_id" { print $2 }' "$properties")" "$(awk '$1 == "device_id" { print $2 }' "$properties")"
    done
}
