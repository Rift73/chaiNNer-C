#!/usr/bin/env bash
# The AMD kit's third step: the GPU proof (gpu_proof.py), then every amd_gpu test of backend/tests and native/tests in
# the test environment, with the environment native/tools/run_tests.py gives the suite except that it hides no GPU
# (its CUDA_VISIBLE_DEVICES=-1 would hide AMD GPUs from HIP too). On a machine with two or more GPUs the GPU list's
# test runs again with HIP_VISIBLE_DEVICES=1,0, which must give the same GPUs in that order.
# Usage: native/tools/amd/test.sh
set -uo pipefail
# shellcheck source=common.sh
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

section=$(fragment 30-tests)
export PYTHONPATH=$REPO/native/tools
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR=$REPO/native/build/numba-cache
export NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"
if [ "${CUDA_VISIBLE_DEVICES-unset}" = -1 ] || [ "${CUDA_VISIBLE_DEVICES-unset}" = "" ]; then
    say "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} hides every GPU from HIP; the tests run without it"
    unset CUDA_VISIBLE_DEVICES
fi
# The small models some tests read, which setup.sh found, and MIGraphX's example custom ops
test_models=$(awk '$1 == "tests" { $1 = ""; sub(/^ /, ""); print; exit }' "$OUT/models.txt" 2>/dev/null)
[ -z "$test_models" ] || export CHAINNER_C_TEST_MODELS=$test_models
export CHAINNER_C_MIGRAPHX_TEST_OPS=$KIT_DIR/$EXAMPLE_OPS

if [ "$DRY_RUN" = 1 ]; then
    say "[dry run] PYTHONPATH=$REPO/backend/src $TEST_PYTHON $KIT_DIR/gpu_proof.py"
    printf '## GPU proof\n\nNot run (dry run).\n' >"$(fragment 25-gpu)"
else
    (cd "$REPO" && PYTHONPATH=$REPO/backend/src "$TEST_PYTHON" -B "$KIT_DIR/gpu_proof.py") \
        >"$(fragment 25-gpu)" 2>"$LOGS/gpu-proof.log"
    say "GPU proof: exit $? ($(fragment 25-gpu))"
fi

# Every test file that marks amd_gpu tests, and those common.sh expects
printf '%s\n' "${AMD_GPU_TEST_FILES[@]}" >"$OUT/expected-tests.txt"
mapfile -t files < <(cd "$REPO" && {
    grep -l 'amd_gpu' backend/tests/test_*.py native/tests/test_*.py
    for file in "${AMD_GPU_TEST_FILES[@]}"; do
        [ -f "$file" ] && echo "$file"
    done
} | sort -u)
say "amd_gpu tests from ${#files[@]} files"
reports=("$LOGS/amd-gpu-tests.xml")
# VapourSynth's vsengine registers a pytest plugin that the tests do not use
pytest_args=(-B -m pytest -m amd_gpu -p no:cacheprovider -p no:vsengine --continue-on-collection-errors -rA)
# A run that takes too long is interrupted as with Ctrl+C, which pytest ends by writing the report of the tests it ran
# (a plain kill leaves none); killed if it is still running 5 minutes later.
limit=(timeout -s INT -k 300)
start=$SECONDS
(cd "$REPO" && run "${limit[@]}" 3600 "$TEST_PYTHON" "${pytest_args[@]}" --junitxml="$LOGS/amd-gpu-tests.xml" \
    "${files[@]}") >"$LOGS/amd-gpu-tests.log" 2>&1
status=$?
say "amd_gpu tests: exit $status in $((SECONDS - start)) s ($LOGS/amd-gpu-tests.log)"
gpu_count=$(kfd_gpus | wc -l)
if [ "$gpu_count" -ge 2 ]; then
    reports+=("$LOGS/amd-gpu-tests-reordered.xml")
    (cd "$REPO" && run env HIP_VISIBLE_DEVICES=1,0 "${limit[@]}" 900 "$TEST_PYTHON" "${pytest_args[@]}" \
        --junitxml="$LOGS/amd-gpu-tests-reordered.xml" backend/tests/test_amd_gpu_devices.py) \
        >"$LOGS/amd-gpu-tests-reordered.log" 2>&1
    say "GPU list with HIP_VISIBLE_DEVICES=1,0: exit $?"
fi

{
    echo "## amd_gpu tests"
    echo
    echo "pytest -m amd_gpu over ${#files[@]} files, exit $status, $((SECONDS - start)) s$([ "$gpu_count" -ge 2 ] && echo "; the GPU list's test again with HIP_VISIBLE_DEVICES=1,0")$([ "$status" = 124 ] && echo "; stopped at the time limit (3600 s): the counts are of the tests that finished before it")."
    echo
    if [ "$DRY_RUN" = 1 ]; then
        echo "Not run (dry run). Files:"
        echo
        printf -- '- %s\n' "${files[@]}"
    else
        "$TEST_PYTHON" -I "$KIT_DIR/describe.py" tests "$OUT/expected-tests.txt" "${reports[@]}"
    fi
} >"$section" 2>>"$LOGS/amd-gpu-tests.log"
say "tests done ($section)"
