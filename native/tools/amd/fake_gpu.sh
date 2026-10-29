#!/usr/bin/env bash
# A fake AMD GPU for the kit's self-test (selftest.sh): inside a container started as root with --cap-add SYS_ADMIN,
# the kernel driver's view of an AMD GPU without the GPU. Its KFD topology goes over /sys/class, its PCI folders (IDs,
# memory counters) over /sys/bus/pci/devices and an amdgpu module over /sys/module, as read-only overlays on the real
# sysfs; /dev/kfd and the render nodes are /dev/null's device, so ROCm opens them, fails its first request and finds
# no GPU. The host check, chaiNNer's GPU list and the dependency manager then see the GPU, and everything up to the
# GPU's own work runs as on a rented machine. Profiles: gfx1100 (one Radeon RX 7900 XTX: 96 compute units, 24 GiB,
# the kernel's own amdgpu) and gfx942 (two Instinct MI300X: 304 compute units, 192 GiB each, AMD's amdgpu with a
# version). It runs only in a container (/.dockerenv or /run/.containerenv) and when told so with --container, as
# selftest.sh runs it: on a machine it would replace the real /dev/kfd and render nodes and hide its sysfs.
# Usage: native/tools/amd/fake_gpu.sh --container gfx1100|gfx942
set -euo pipefail

if [ "${1:-}" != --container ] || ! { [ -f /.dockerenv ] || [ -f /run/.containerenv ]; }; then
    echo "fake_gpu.sh replaces /dev/kfd and the render nodes and mounts over /sys, so it runs only inside a container," \
        "as selftest.sh runs it: fake_gpu.sh --container gfx1100|gfx942" >&2
    exit 2
fi
shift

layer=/run/fake-amd-gpu
topology=$layer/class/kfd/kfd/topology
rm -rf -- "$layer"
mkdir -p "$topology/nodes" "$layer/devices" "$layer/module/amdgpu" /dev/dri
printf '1\n' >"$topology/generation_id"
printf 'platform_oem 0\nplatform_id 0\nplatform_rev 0\n' >"$topology/system_properties"

# A KFD node: its properties ("name value" lines from the arguments' name=value pairs) and the files beside them that
# ROCm's libraries read (rocprofiler-sdk ends the process without gpu_id).
node() {
    local dir=$topology/nodes/$1 gpu_id=$2 name=$3 memory=$4 heap=$5
    shift 5
    mkdir -p "$dir/caches" "$dir/p2p_links" "$dir/perf" "$dir/mem_banks/0" "$dir/io_links/0"
    printf '%s\n' "${@/=/ }" >"$dir/properties"
    printf '%s\n' "$gpu_id" >"$dir/gpu_id"
    printf '%s\n' "$name" >"$dir/name"
    printf 'heap_type %d\nsize_in_bytes %d\nflags 0\nwidth 64\nmem_clk_max 2400\n' "$heap" "$memory" \
        >"$dir/mem_banks/0/properties"
    printf 'type 2\nversion_major 0\nversion_minor 0\nnode_from %d\nnode_to 0\nweight 20\nmin_latency 0\n' "${dir##*/}" \
        >"$dir/io_links/0/properties"
    printf 'max_latency 0\nmin_bandwidth 312\nmax_bandwidth 64000\nrecommended_transfer_size 0\nflags 1\n' \
        >>"$dir/io_links/0/properties"
}

# A GPU: KFD node, PCI folder (bus 0000:BUS:00.0), render node renderD<128 + index>.
gpu() {
    local index=$1 bus=$2 gfx_version=$3 simds=$4 simds_per_cu=$5 device=$6 revision=$7 class=$8 vram=$9 wave=${10}
    local name=${11}
    node "$((index + 1))" "$((40263 + index * 7919))" "$name" "$vram" 1 cpu_cores_count=0 "simd_count=$simds" \
        mem_banks_count=1 caches_count=0 io_links_count=1 p2p_links_count=0 max_waves_per_simd=16 lds_size_in_kb=64 \
        "wave_front_size=$wave" "simd_per_cu=$simds_per_cu" "gfx_target_version=$gfx_version" vendor_id=4098 \
        "device_id=$((device))" "location_id=$((bus << 8))" domain=0 "drm_render_minor=$((128 + index))" hive_id=0 \
        num_sdma_engines=2 num_cp_queues=8 max_engine_clk_fcompute=2500 local_mem_size=0 \
        "unique_id=$((0x1234567890abcd00 + index))" num_xcc=1 max_engine_clk_ccompute=5000
    local pci
    pci=$layer/devices/$(printf '0000:%02x:00.0' "$bus")
    mkdir -p "$pci"
    printf '0x1002\n' >"$pci/vendor"
    printf '0x%04x\n' "$device" >"$pci/device"
    printf '0x%02x\n' "$revision" >"$pci/revision"
    printf '0x%06x\n' "$class" >"$pci/class"
    printf '%d\n' "$vram" >"$pci/mem_info_vram_total"
    printf '%d\n' "$((vram / 50))" >"$pci/mem_info_vram_used"
    printf '0\n' >"$pci/irq"
    # The configuration space's header, which lspci reads: vendor and device, command and status, revision, class
    local byte
    for byte in 2 16 "$((device & 255))" "$((device >> 8))" 7 4 16 0 "$revision" 0 "$((class >> 8 & 255))" \
        "$((class >> 16))"; do
        printf '%b' "\\x$(printf %02x "$byte")"
    done >"$pci/config"
    head -c 244 /dev/zero >>"$pci/config"
    rm -f -- "/dev/dri/renderD$((128 + index))" "/dev/dri/card$index"
    mknod -m 666 "/dev/dri/renderD$((128 + index))" c 1 3
    mknod -m 666 "/dev/dri/card$index" c 1 3
}

# The CPU's node
node 0 0 "" 33520001024 0 "cpu_cores_count=$(nproc)" simd_count=0 mem_banks_count=1 caches_count=0 io_links_count=1 \
    p2p_links_count=0 gfx_target_version=0 vendor_id=0 device_id=0 location_id=0 domain=0 drm_render_minor=0
case ${1:-} in
    gfx1100)
        gpu 0 0x03 110000 192 2 0x744c 0xc8 0x030000 25769803776 32 gfx1100
        ;;
    gfx942)
        printf '6.12.12\n' >"$layer/module/amdgpu/version"
        gpu 0 0x05 90402 1216 4 0x74a1 0x00 0x120000 206158430208 64 gfx942
        gpu 1 0x15 90402 1216 4 0x74a1 0x00 0x120000 206158430208 64 gfx942
        ;;
    *)
        echo "usage: $0 --container gfx1100|gfx942" >&2
        exit 1
        ;;
esac
rm -f -- /dev/kfd
mknod -m 666 /dev/kfd c 1 3
for target in class bus/pci/devices module; do
    mount -t overlay overlay -o "lowerdir=$layer/${target##*/}:/sys/$target" "/sys/$target"
done
echo "fake AMD GPU: $1"
