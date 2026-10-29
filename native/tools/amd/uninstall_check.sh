#!/usr/bin/env bash
# The AMD kit's last check before the report: the dependency manager uninstalls the ONNX package, which shares ROCm's
# packages with PyTorch; PyTorch's ROCm packages must stay and PyTorch must still see the GPU. Then it installs ONNX
# again, which must give back MIGraphX as ONNX Runtime's provider, so the app is whole for the remote desktop.
# Usage: native/tools/amd/uninstall_check.sh
set -uo pipefail
# shellcheck source=common.sh
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

section=$(fragment 60-uninstall)
lines=()
note() {
    lines+=("- $1")
    say "$1"
}
# PyTorch's packages and ROCm's, as pip lists them
shared() {
    "$APP_PYTHON" -s -m pip list --format=freeze --disable-pip-version-check 2>/dev/null |
        grep -E '^(torch|torchvision|triton-rocm|rocm|rocm-sdk-[a-z0-9-]+)==' | sort
}
gpu_check() {
    "$APP_PYTHON" -s -c 'import torch; print(torch.version.hip, torch.cuda.is_available())' 2>&1 | tail -n 1
}
providers() {
    "$APP_PYTHON" -s -c 'import onnxruntime; print(onnxruntime.get_available_providers()[0])' 2>&1 | tail -n 1
}

if [ "$DRY_RUN" = 1 ]; then
    say "[dry run] uninstall chaiNNer_onnx, check PyTorch's ROCm packages, install chaiNNer_onnx again"
    printf '## Dependency manager: uninstalling ONNX\n\nNot run (dry run).\n' >"$section"
    exit 0
fi
before=$(shared)
failed=0
if backend_start "$LOGS/uninstall-check.log"; then
    note "Uninstall chaiNNer_onnx: $(backend POST /packages/uninstall '{"packages": ["chaiNNer_onnx"]}')"
    after=$(shared)
    if [ "$before" = "$after" ]; then
        note "PyTorch's and ROCm's packages: all kept ($(grep -c . <<<"$after") packages)"
    else
        failed=1
        note "PyTorch's and ROCm's packages: CHANGED: $(diff <(echo "$before") <(echo "$after") | grep '^[<>]' | tr '\n' ' ')"
    fi
    torch_after=$(gpu_check)
    note "PyTorch afterwards (HIP version, GPU available): $torch_after"
    [[ $torch_after == *" True" ]] || failed=1
    note "Install chaiNNer_onnx again: $(backend POST /packages/install '{"packages": ["chaiNNer_onnx"]}')"
    provider=$(providers)
    note "ONNX Runtime's first provider afterwards: $provider"
    [ "$provider" = MIGraphXExecutionProvider ] || failed=1
else
    failed=1
    note "The backend did not get ready ($LOGS/uninstall-check.log)"
fi
backend_stop
{
    echo "## Dependency manager: uninstalling ONNX"
    echo
    printf '%s\n' "${lines[@]}"
} >"$section"
exit "$failed"
