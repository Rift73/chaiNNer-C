#!/usr/bin/env bash
# The AMD kit's first step: what the machine is and whether it has an AMD GPU ROCm 7.14.1 can use. Writes the report's
# host section and $WORK/out/gpus.txt (the usable GPUs' gfx targets), and stops with exit status 2 and the reason when
# there is no usable GPU (a dry run notes it and goes on). Needs no ROCm: the GPUs come from the kernel driver's KFD
# topology, as chaiNNer's own detection reads them.
# Usage: native/tools/amd/host_check.sh
set -uo pipefail
# shellcheck source=common.sh
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

section=$(fragment 10-host)
problems=()
facts=()

fact() {
    facts+=("- $1: $2")
    say "$1: $2"
}

fact "System" "$(os_release PRETTY_NAME) ($(getconf GNU_LIBC_VERSION 2>/dev/null || echo 'glibc unknown'))"
fact "Kernel" "$(uname -r)"
if [ -r /sys/module/amdgpu/version ]; then
    driver="amdgpu $(cat /sys/module/amdgpu/version)"
elif [ -d /sys/module/amdgpu ]; then
    driver="amdgpu (the kernel's own, no version file)"
else
    driver="no amdgpu module loaded"
    problems+=("The amdgpu kernel driver is not loaded on the host.")
fi
fact "Driver" "$driver"
fact "CPU" "$(awk -F': ' '/^model name/ { print $2; exit }' /proc/cpuinfo) x $(nproc)"
fact "RAM" "$(awk '/^MemTotal/ { total = $2 } /^MemAvailable/ { free = $2 } END { printf "%.1f GiB, %.1f GiB available", total / 1048576, free / 1048576 }' /proc/meminfo)"
mkdir -p "$WORK"
free_gb=$(df -P -BG "$WORK" | awk 'NR == 2 { sub("G", "", $4); print $4 }')
fact "Disk" "$free_gb GiB free in $WORK"
if [ "$free_gb" -lt "$KIT_MIN_DISK_GB" ]; then
    problems+=("$WORK has $free_gb GiB free; the kit needs $KIT_MIN_DISK_GB GiB (rent a machine with more disk, or set KIT_WORK).")
fi
fact "User" "$(id)"
# Chromium (the app's window and its headless runs) keeps shared memory in /dev/shm, which a container may give 64 MB
fact "/dev/shm" "$(df -P -BM /dev/shm 2>/dev/null | awk 'NR == 2 { print $2 }')"
visible=$(env | grep -E '^(HIP_VISIBLE_DEVICES|ROCR_VISIBLE_DEVICES|CUDA_VISIBLE_DEVICES|GPU_DEVICE_ORDINAL|HSA_OVERRIDE_GFX_VERSION[_0-9]*)=' | tr '\n' ' ')
fact "GPU variables" "${visible:-none}"

# /dev/kfd is ROCm's compute interface, /dev/dri/renderD* each GPU's render node; a container gets them with
# --device /dev/kfd --device /dev/dri.
if [ -e /dev/kfd ]; then
    if [ -r /dev/kfd ] && [ -w /dev/kfd ]; then
        fact "/dev/kfd" "$(stat -c '%U:%G %a' /dev/kfd), readable and writable"
    else
        fact "/dev/kfd" "$(stat -c '%U:%G %a' /dev/kfd), NOT readable and writable by this user"
        problems+=("This user cannot open /dev/kfd: add it to the group $(stat -c %G /dev/kfd), or run as root.")
    fi
else
    fact "/dev/kfd" "missing"
    problems+=("/dev/kfd is missing: the machine has no AMD GPU, or the container was started without --device /dev/kfd (on vast.ai: an AMD machine and a ROCm template).")
fi
render_nodes=()
for render in /dev/dri/renderD*; do
    [ -e "$render" ] || continue
    access="no access"
    [ -r "$render" ] && [ -w "$render" ] && access="readable and writable"
    render_nodes+=("$render $(stat -c '%U:%G %a' "$render") $access")
done
if [ "${#render_nodes[@]}" = 0 ]; then
    fact "/dev/dri" "no render nodes"
    problems+=("No /dev/dri/renderD* node: the container was started without --device /dev/dri.")
else
    fact "/dev/dri" "$(printf '%s; ' "${render_nodes[@]}")"
fi

# The GPUs, and whether ROCm 7.14.1 has kernels for each
usable=()
gpu_lines=()
while read -r node gfx ids; do
    [ -n "$node" ] || continue
    if [[ " $ROCM_TARGETS " == *" $gfx "* ]]; then
        usable+=("$gfx")
        gpu_lines+=("KFD node $node: $gfx ($ids), supported")
    else
        gpu_lines+=("KFD node $node: $gfx ($ids), NOT among ROCm 7.14.1's targets")
    fi
done < <(kfd_gpus)
if [ "${#gpu_lines[@]}" = 0 ]; then
    fact "GPUs" "none in the KFD topology"
    [ -e /dev/kfd ] && problems+=("The KFD topology lists no GPU.")
else
    fact "GPUs" "$(printf '%s; ' "${gpu_lines[@]}")"
    if [ "${#usable[@]}" = 0 ]; then
        problems+=("No GPU here is among ROCm 7.14.1's targets ($ROCM_TARGETS).")
    fi
fi
printf '%s\n' "${usable[@]}" | sort -u | grep . >"$OUT/gpus.txt"
if command -v lspci >/dev/null 2>&1; then
    fact "PCI display devices" "$(lspci -nn | grep -Ei 'vga|display|3d|processing accelerators' | tr '\n' ';')"
fi
# A system ROCm (AMD's ROCm images) is not what chaiNNer-C uses (its ROCm comes with PyTorch's wheels), but its tools
# tell what the host sees.
for tool in rocminfo amd-smi; do
    path=$(command -v "$tool" || ls /opt/rocm/bin/"$tool" 2>/dev/null)
    [ -n "$path" ] || continue
    case $tool in
        rocminfo) fact "System rocminfo" "$("$path" 2>&1 | grep -oE 'gfx[0-9a-f]+' | sort -u | tr '\n' ' ')" ;;
        amd-smi) fact "System amd-smi" "$("$path" version 2>&1 | tr '\n' ' ')" ;;
    esac
done
if ldconfig -p 2>/dev/null | grep -q 'libatomic\.so\.1 '; then
    fact "libatomic.so.1" "present"
else
    fact "libatomic.so.1" "missing (setup installs it; ROCm's profiler libraries need it)"
fi
# The MIGraphX runner links the system's libstdc++, which must provide GLIBCXX_3.4.22 or newer
libstdcxx=$(ldconfig -p 2>/dev/null | awk '/libstdc\+\+\.so\.6 .*x86-64/ { print $NF; exit }')
if [ -n "$libstdcxx" ]; then
    fact "libstdc++" "$libstdcxx, up to $(grep -aoE 'GLIBCXX_3\.4\.[0-9]+' "$libstdcxx" | sort -uV | tail -n 1)"
else
    fact "libstdc++" "missing (setup installs it)"
fi

{
    echo "## Host"
    echo
    printf '%s\n' "${facts[@]}"
    if [ "${#problems[@]}" -gt 0 ]; then
        echo
        echo "Host check: **stopped**"
        printf -- '- %s\n' "${problems[@]}"
    else
        echo
        echo "Host check: passed ($(tr '\n' ' ' <"$OUT/gpus.txt"))"
    fi
} >"$section"

if [ "${#problems[@]}" -gt 0 ]; then
    for problem in "${problems[@]}"; do
        say "STOP: $problem" >&2
    done
    if [ "$DRY_RUN" = 1 ]; then
        say "[dry run] no usable AMD GPU; the dry run goes on"
        exit 0
    fi
    if [ "${KIT_SELFTEST:-0}" = 1 ]; then
        say "[self-test] no usable AMD GPU; the steps run on the CPU"
        exit 0
    fi
    say "The kit stops here: this machine cannot run chaiNNer-C on an AMD GPU (reasons above)." >&2
    exit 2
fi
say "host check passed: $(tr '\n' ' ' <"$OUT/gpus.txt")"
