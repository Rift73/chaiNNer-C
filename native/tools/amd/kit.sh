#!/usr/bin/env bash
# The AMD kit in one command (native/AMD.md): host check, setup, GPU proof and amd_gpu tests, chains, every spandrel
# architecture, spandrel's HIP kernels on and off, speed table, the dependency manager's uninstall check and the report,
# $WORK/amd-kit-report-*.tar.gz. A step that fails is reported and the next ones still run; a machine without a usable
# AMD GPU stops after the host check (exit status 2) with the reason in the report.
# --dry-run walks every step without a GPU and without installing anything, printing what each would run, and writes a
# sample report. KIT_WORK sets the work folder (default ~/amd-kit); KIT_ONLY (a regular expression) runs only the
# chains and speed rows whose names it matches (esrgan-pytorch-fp32, dual-migraphx, ...); KIT_ARCHS (a regular
# expression) only the spandrel architectures whose IDs it matches (ESRGAN, DAT, SST, ...); KIT_SKIP_VOSR=1 leaves out
# VOSR; KIT_SELFTEST=1 runs the steps without an AMD GPU (selftest.sh).
# Usage: native/tools/amd/kit.sh --zip FILE [--models DIR] [--wheels DIR] [--iterations N] [--dry-run]
set -uo pipefail

zip=""
models=""
wheels=""
iterations=50
for argument in "$@"; do
    [ "$argument" = --dry-run ] && export KIT_DRY_RUN=1
done
# shellcheck source=common.sh
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"
while [ $# -gt 0 ]; do
    case "$1" in
        --zip) zip=${2:?--zip needs a file}; shift ;;
        --models) models=${2:?--models needs a folder}; shift ;;
        --wheels) wheels=${2:?--wheels needs a folder}; shift ;;
        --iterations) iterations=${2:?--iterations needs a number}; shift ;;
        --dry-run) ;;
        *) die "unknown argument $1" ;;
    esac
    shift
done
[ -n "$zip" ] || [ "$DRY_RUN" = 1 ] || [ -x "$APP/chaiNNer.sh" ] || die "--zip FILE (the Linux zip) is needed"

rm -rf -- "$OUT"
mkdir -p "$LOGS"
date +%s >"$OUT/started"
exec > >(tee -a "$LOGS/kit.log") 2>&1
say "chaiNNer-C AMD kit: work folder $WORK$([ "$DRY_RUN" = 1 ] && echo ', dry run')"

steps=()
step() {
    local name=$1
    shift
    local start=$SECONDS
    say "== $name"
    "$@"
    local status=$?
    steps+=("- $name: exit $status, $((SECONDS - start)) s")
    say "== $name: exit $status"
    return "$status"
}
finish() {
    {
        echo "## Steps"
        echo
        printf '%s\n' "${steps[@]}"
    } >"$(fragment 05-steps)"
    bash "$KIT_DIR/report.sh"
}

step "host check" bash "$KIT_DIR/host_check.sh"
if [ $? = 2 ]; then
    finish
    exit 2
fi
setup_args=()
[ -n "$zip" ] && setup_args+=(--zip "$zip")
[ -n "$models" ] && setup_args+=(--models "$models")
[ -n "$wheels" ] && setup_args+=(--wheels "$wheels")
if ! step setup bash "$KIT_DIR/setup.sh" "${setup_args[@]}"; then
    finish
    exit 1
fi
step "GPU proof and amd_gpu tests" bash "$KIT_DIR/test.sh"
step chains bash "$KIT_DIR/chains.sh"
# The self-test has no AMD GPU: both checks run their code on the CPU there (selftest.sh names three architectures in
# KIT_ARCHS). Both write their rows as they finish and their Result line last; finished FRAGMENT STATUS gives a check
# the kit stopped (no Result line) one that says so, keeping its rows, and returns STATUS.
finished() {
    grep -q '^Result: ' "$1" ||
        printf '\nResult: STOPPED (exit %s) before it finished; the rows above are those that did.\n' "$2" >>"$1"
    return "$2"
}
archs() {
    local args=()
    [ -n "${KIT_ARCHS:-}" ] && args+=(--only "$KIT_ARCHS")
    [ "${KIT_SELFTEST:-0}" = 1 ] && args+=(--device cpu)
    if [ "$DRY_RUN" = 1 ]; then
        say "[dry run] PYTHONPATH=$REPO/backend/src $TEST_PYTHON $KIT_DIR/arch_check.py ${args[*]}"
        printf '## Every architecture\n\nResult: not run (dry run).\n' >"$(fragment 44-archs)"
        return 0
    fi
    # The check ends by its time cap (900 s, its first process included): no architecture starts after it, and the
    # last one started is stopped at it. 1500 s leaves room for the stopped process to end.
    (cd "$REPO" && PYTHONPATH=$REPO/backend/src timeout -k 60 1500 "$TEST_PYTHON" -B "$KIT_DIR/arch_check.py" \
        "${args[@]}") >"$(fragment 44-archs)" 2>"$LOGS/arch-check.log"
    finished "$(fragment 44-archs)" "$?"
}
hip_kernels() {
    local args=(--models "$OUT/models.txt" --image "$KIT_DIR/chains/input.png")
    [ "${KIT_SELFTEST:-0}" = 1 ] && args+=(--device cpu --size 64 --iterations 1)
    if [ "$DRY_RUN" = 1 ]; then
        say "[dry run] PYTHONPATH=$REPO/backend/src $TEST_PYTHON $KIT_DIR/hip_kernels.py ${args[*]}"
        printf "## spandrel's HIP kernels on and off\n\nResult: not run (dry run).\n" >"$(fragment 46-hip-kernels)"
        return 0
    fi
    # At most 8 processes of 300 s each (2 models, FP32 and FP16, kernels off and on): 3000 s covers them.
    (cd "$REPO" && PYTHONPATH=$REPO/backend/src timeout -k 60 3000 "$TEST_PYTHON" -B "$KIT_DIR/hip_kernels.py" \
        "${args[@]}") >"$(fragment 46-hip-kernels)" 2>"$LOGS/hip-kernels.log"
    finished "$(fragment 46-hip-kernels)" "$?"
}
step "every architecture" archs
step "HIP kernels on and off" hip_kernels
speed() {
    if [ "$DRY_RUN" = 1 ]; then
        say "[dry run] bench.py --models $OUT/models.txt --iterations $iterations"
        printf '## Speed\n\nNot run (dry run): the routes of native/tools/amd/bench.py at 256x256 and 512x512.\n' \
            >"$(fragment 50-speed)"
        return 0
    fi
    (cd "$REPO" && PYTHONPATH=$REPO/backend/src timeout 5400 "$TEST_PYTHON" -B "$KIT_DIR/bench.py" \
        --models "$OUT/models.txt" --image "$KIT_DIR/chains/input.png" --storage "$APP/backend-storage" \
        --programs "$PROGRAMS" --iterations "$iterations" --only "${KIT_ONLY:-}") >"$(fragment 50-speed)" \
        2>"$LOGS/bench.log"
}
step "speed table" speed
step "uninstall check" bash "$KIT_DIR/uninstall_check.sh"
finish
