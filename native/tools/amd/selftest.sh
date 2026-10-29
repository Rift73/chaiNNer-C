#!/usr/bin/env bash
# The kit's self-test on a machine without an AMD GPU, in Docker containers of a bare image (native/AMD.md, Updating
# the kit): the host check must stop with exit status 2; a dry run must finish; and with --zip the kit's real steps run
# on the CPU (KIT_SELFTEST=1 lets the host check go on, VOSR is skipped, 2 timed calls): setup through the dependency
# manager, which installs a CPU build of PyTorch, the tests, the chains, the speed table and the uninstall check. The
# PyTorch and NCNN chains must pass there: this run is chaiNNer-C's only check of a CPU build of PyTorch (the test suite
# runs a CUDA build with CUDA hidden). The checkout is copied without its build outputs; results go to
# $KIT_SELFTEST_DIR (default ~/amd-kit-selftest).
# Usage: native/tools/amd/selftest.sh [--zip FILE] [--image ubuntu:24.04]
set -uo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
dir=${KIT_SELFTEST_DIR:-$HOME/amd-kit-selftest}
zip=""
image=ubuntu:24.04
while [ $# -gt 0 ]; do
    case "$1" in
        --zip) zip=${2:?--zip needs a file}; shift ;;
        --image) image=${2:?--image needs a Docker image}; shift ;;
        *) echo "error: unknown argument $1" >&2; exit 1 ;;
    esac
    shift
done
command -v docker >/dev/null 2>&1 || { echo "error: the self-test needs Docker" >&2; exit 1; }
[ -z "$zip" ] || [ -f "$zip" ] || { echo "error: no zip at $zip" >&2; exit 1; }

rm -rf -- "$dir"
mkdir -p "$dir/checkout" "$dir/z" "$dir/results"
tar -C "$repo" --exclude=./.git --exclude=./node_modules --exclude=./out --exclude=./native/build \
    --exclude=./native/runtime --exclude=./native/.venv --exclude=./native/local -cf - . | tar -C "$dir/checkout" -xf -
[ -z "$zip" ] || cp "$zip" "$dir/z/app.zip"
cat >"$dir/in_container.sh" <<'EOF'
set -u
cp -r /kit /root/chaiNNer-C
kit=/root/chaiNNer-C/native/tools/amd
case $1 in
    stop) bash $kit/kit.sh --zip /z/app.zip ;;
    dry) bash $kit/kit.sh --dry-run ;;
    cpu)
        # PyTorch in FP16 on a CPU takes many minutes per image: the CPU's routes are FP32 and NCNN
        KIT_SELFTEST=1 KIT_SKIP_VOSR=1 KIT_ONLY='^esrgan-(pytorch-fp32|ncnn-vulkan)$' \
            bash $kit/kit.sh --zip /z/app.zip --iterations 2
        ;;
esac
status=$?
cp /root/amd-kit/out/summary.md "/results/summary-$1.md"
cp /root/amd-kit/amd-kit-report-*.tar.gz "/results/report-$1.tar.gz"
exit $status
EOF

failed=0
check() {
    if [ "$2" = 0 ]; then
        echo "PASS $1"
    else
        echo "FAIL $1"
        failed=1
    fi
}
container() {
    timeout 3600 docker run --rm --name "amd-kit-selftest-$1" -v "$dir/checkout:/kit:ro" -v "$dir/z:/z:ro" \
        -v "$dir/results:/results" -v "$dir/in_container.sh:/in_container.sh:ro" --entrypoint bash "$image" \
        /in_container.sh "$1" >"$dir/results/$1.log" 2>&1
}

container stop
status=$?
check "the host check stops without an AMD GPU (exit $status, expected 2)" "$([ "$status" = 2 ]; echo $?)"
container dry
check "a dry run finishes (exit $?)" "$?"
if [ -n "$zip" ]; then
    container cpu
    summary=$dir/results/summary-cpu.md
    for chain in esrgan-pytorch-fp32 esrgan-ncnn-vulkan; do
        grep -q "^| $chain | esrgan | passed |" "$summary" 2>/dev/null
        check "the chain $chain passes on the CPU" "$?"
    done
    grep -q '^| PyTorch | enabled |' "$summary" 2>/dev/null
    check "the PyTorch package is enabled with a CPU build of PyTorch" "$?"
fi
echo "results: $dir/results"
exit "$failed"
