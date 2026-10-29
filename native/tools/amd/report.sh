#!/usr/bin/env bash
# The AMD kit's last step: summary.md from the steps' sections ($WORK/out/*.md) under a summary of the run and the
# versions, then one archive with it, the logs and the chains' outputs: $WORK/amd-kit-report-<host>-<time>.tar.gz,
# the file to send back.
# Usage: native/tools/amd/report.sh
set -uo pipefail
# shellcheck source=common.sh
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

summary=$OUT/summary.md
started=$(cat "$OUT/started" 2>/dev/null || date +%s)
count() {
    grep -c "$1" "$2" 2>/dev/null || true
}
{
    echo "# chaiNNer-C AMD kit report"
    echo
    echo "- Machine: $(hostname), $(date -u '+%Y-%m-%d %H:%M UTC'), kit run $(( ($(date +%s) - started) / 60 )) min$([ "$DRY_RUN" = 1 ] && echo ', DRY RUN')"
    echo "- Checkout: $(git -C "$REPO" log -1 --format='%h %s' 2>/dev/null) ($(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null))"
    gpus=""
    [ -f "$OUT/gpus.txt" ] && gpus=$(tr '\n' ' ' <"$OUT/gpus.txt")
    echo "- GPUs ROCm 7.14.1 supports: ${gpus:-none}"
    if [ -f "$(fragment 30-tests)" ]; then
        echo "- amd_gpu tests: $(sed -n '5p' "$(fragment 30-tests)")"
    fi
    if [ -f "$(fragment 40-chains)" ]; then
        echo "- Chains: $(count '| passed |' "$(fragment 40-chains)") passed, $(count '| FAILED' "$(fragment 40-chains)") failed, $(count '| skipped' "$(fragment 40-chains)") skipped"
    fi
    if [ -f "$(fragment 50-speed)" ]; then
        echo "- Speed rows: $(grep -cE '^\| [a-z0-9]+ \| .* \| [0-9.]+ \| [0-9.]+ \|' "$(fragment 50-speed)") timed, $(count 'FAILED' "$(fragment 50-speed)") failed"
    fi
    echo
    for section in "$OUT"/[0-9]*.md; do
        [ -f "$section" ] || continue
        cat "$section"
        echo
    done
    echo "## Versions"
    echo
    echo "- Electron $(cat "$APP/version" 2>/dev/null || echo unknown)"
    if [ -x "$APP_PYTHON" ]; then
        echo "- $("$APP_PYTHON" -s -c 'import sys; print("Python", sys.version.split()[0])')"
        "$APP_PYTHON" -s -m pip list --format=freeze --disable-pip-version-check 2>/dev/null |
            grep -iE '^(torch|torchvision|triton|rocm|onnx|onnxruntime|spandrel|ncnn|numpy|opencv)' | sed 's/^/- /'
    fi
    echo
    echo "Logs: logs/ in this archive."
} >"$summary"

archive=$WORK/amd-kit-report-$(hostname)-$(date +%Y%m%d-%H%M).tar.gz
contents=(summary.md logs)
[ -d "$OUT/chains" ] && contents+=(chains)
tar -czf "$archive" -C "$OUT" "${contents[@]}" || die "writing $archive failed"
say "report: $archive ($(du -h "$archive" | cut -f 1)); summary: $summary"
