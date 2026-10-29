#!/usr/bin/env bash
# The AMD kit's fourth step: real chains through the app's command line (chaiNNer.sh run, headless on a virtual
# display), one per route, each with its package settings in the app's settings.json as the window would save them.
# Every model of models.txt runs PyTorch FP32 (the reference its other routes are compared with), PyTorch FP16 and
# MIGraphX; the ESRGAN also ONNX Runtime with the MIGraphX provider and NCNN on Vulkan; VOSR (which downloads its own
# model unless KIT_SKIP_VOSR=1) runs its PyTorch backend (also in GPU Memory Low) and MIGraphX, and MIGraphX FP8 on a
# GPU with FP8. KIT_ONLY (a regular expression) runs only the chains whose names it matches.
# Usage: native/tools/amd/chains.sh
set -uo pipefail
# shellcheck source=common.sh
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

section=$(fragment 40-chains)
results=$OUT/chains
mkdir -p "$results"
fp32='{"chaiNNer_pytorch":{"use_fp16":false}}'
fp16='{"chaiNNer_pytorch":{"use_fp16":true}}'
# Build Program keeps its programs where the speed table looks for them (bench.py), so neither compiles one twice
migraphx='{"chaiNNer_migraphx":{"program_cache":"@PROGRAMS@"}}'
# name, chain (chains/<chain>.chn), model role (models.txt), package settings
ROUTES=(
    "esrgan-pytorch-fp32 pytorch esrgan $fp32"
    "esrgan-pytorch-fp16 pytorch esrgan $fp16"
    'esrgan-onnx-migraphx-fp32 onnx esrgan {"chaiNNer_onnx":{"execution_provider":"MIGraphXExecutionProvider","migraphx_fp16_mode":false}}'
    'esrgan-onnx-migraphx-fp16 onnx esrgan {"chaiNNer_onnx":{"execution_provider":"MIGraphXExecutionProvider","migraphx_fp16_mode":true}}'
    "esrgan-migraphx-fp32 migraphx-fp32 esrgan $migraphx"
    "esrgan-migraphx-fp16 migraphx-fp16 esrgan $migraphx"
    "esrgan-ncnn-vulkan ncnn esrgan {}"
    "dual-pytorch-fp32 pytorch dual $fp32"
    "dual-pytorch-fp16 pytorch dual $fp16"
    "dual-migraphx migraphx-fp16 dual $migraphx"
    "drft-pytorch-fp32 pytorch drft $fp32"
    "drft-pytorch-fp16 pytorch drft $fp16"
    "drft-migraphx migraphx-fp16 drft $migraphx"
    "hat-pytorch-fp32 pytorch hat $fp32"
    "hat-pytorch-fp16 pytorch hat $fp16"
    "hat-migraphx-fp16 migraphx-fp16 hat $migraphx"
    "dat2-pytorch-fp32 pytorch dat2 $fp32"
    "dat2-pytorch-fp16 pytorch dat2 $fp16"
    "dat2-migraphx-fp16 migraphx-fp16 dat2 $migraphx"
    "vosr-pytorch vosr-pytorch vosr {}"
    "vosr-pytorch-low vosr-pytorch-low vosr {}"
    "vosr-migraphx vosr-migraphx vosr {}"
    "vosr-migraphx-fp8 vosr-migraphx-fp8 vosr {}"
)
# FP8: CDNA3 (gfx94x), CDNA4 (gfx950) and gfx12 (as gpu.AmdDevice.supports_fp8)
fp8=0
grep -qE '^gfx(94[0-9a-f]|950|12[0-9a-f]{2})$' "$OUT/gpus.txt" 2>/dev/null && fp8=1

model_of() {
    awk -v role="$1" '$1 == role { $1 = ""; sub(/^ /, ""); print; exit }' "$OUT/models.txt" 2>/dev/null
}

rows=$OUT/chains.tsv
: >"$rows"
for route in "${ROUTES[@]}"; do
    read -r name chain role settings <<<"$route"
    log=$LOGS/chain-$name.log
    if [ -n "${KIT_ONLY:-}" ] && ! [[ $name =~ $KIT_ONLY ]]; then
        printf '%s\t%s\tskipped (not in KIT_ONLY)\t\t\t\n' "$name" "$role" >>"$rows"
        continue
    fi
    model=$(model_of "$role")
    if [ -z "$model" ]; then
        printf '%s\t%s\tskipped (no %s model)\t\t\t\n' "$name" "$role" "$role" >>"$rows"
        continue
    fi
    if [ "$name" = vosr-migraphx-fp8 ] && [ "$fp8" = 0 ]; then
        printf '%s\t%s\tskipped (this GPU has no FP8)\t\t\t\n' "$name" "$role" >>"$rows"
        continue
    fi
    reference=$results/$role-pytorch-fp32.png
    [ "$role" = vosr ] && reference=$results/vosr-pytorch.png
    override=$WORK/override-$name.json
    sed -e "s|@IMAGE@|$KIT_DIR/chains/input.png|" -e "s|@MODEL@|$model|" -e "s|@OUT@|$results|" \
        -e "s|@NAME@|$name|" -e 's|"@SIZE@"|256|' "$KIT_DIR/chains/$chain.json" >"$override"
    [ "$DRY_RUN" = 1 ] || printf '{"packageSettings": %s}\n' "${settings//@PROGRAMS@/$PROGRAMS}" >"$APP/settings.json"
    rm -f -- "$results/$name.png"
    say "chain $name ($chain.chn, $role)"
    start=$SECONDS
    run timeout 2400 "$APP/chaiNNer.sh" run "$KIT_DIR/chains/$chain.chn" --override "$override" >"$log" 2>&1
    status=$?
    if [ "$DRY_RUN" = 1 ]; then
        result="not run (dry run)"
    elif [ "$status" = 0 ] && [ -f "$results/$name.png" ]; then
        result=passed
    else
        result="FAILED (exit $status)"
    fi
    say "chain $name: $result in $((SECONDS - start)) s"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$role" "$result" "$((SECONDS - start))" "$results/$name.png" \
        "$reference" >>"$rows"
done

{
    echo "## Chains"
    echo
    echo "Headless runs of native/tools/amd/chains/ on chains/input.png (256x256); MIGraphX programs of 256x256."
    echo
    if [ "$DRY_RUN" = 1 ]; then
        echo "| Chain | Model | Result |"
        echo "|---|---|---|"
        awk -F '\t' '{ print "| " $1 " | " $2 " | " $3 " |" }' "$rows"
    else
        "$TEST_PYTHON" -I "$KIT_DIR/describe.py" chains "$rows" "$LOGS"
    fi
} >"$section" 2>>"$LOGS/chains-summary.log"
say "chains done ($section)"
