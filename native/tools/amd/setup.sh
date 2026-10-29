#!/usr/bin/env bash
# The AMD kit's second step: installs what chaiNNer-C needs on this machine and sets it up as a user would. The system
# packages of the launcher's table (resources/src/linux_libraries.txt) and the kit's own (libatomic for ROCm, Mesa's
# Vulkan driver for NCNN, Xvfb for chains); the app from its zip; a first headless run, which downloads the app's
# Python; the wheels given as copies or release URLs (common.sh); then the dependency manager installs the packages
# of common.sh (KIT_PACKAGES) through the backend, as its window would, with the ROCm builds it picks for this
# GPU. Last, the test environment (the app's Python with pytest), the zip's native binaries in this checkout's
# backend/src (the tests run from the checkout), and the models (models.txt).
# Usage: native/tools/amd/setup.sh --zip FILE [--wheels DIR] [--models DIR]
set -uo pipefail
# shellcheck source=common.sh
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

zip=""
wheels=""
models=""
while [ $# -gt 0 ]; do
    case "$1" in
        --zip) zip=${2:?--zip needs a file}; shift ;;
        --wheels) wheels=${2:?--wheels needs a folder}; shift ;;
        --models) models=${2:?--models needs a folder}; shift ;;
        *) die "unknown argument $1" ;;
    esac
    shift
done
section=$(fragment 20-setup)
lines=()
note() {
    if [ "$DRY_RUN" = 1 ]; then
        lines+=("- (dry run, not done) $1")
    else
        lines+=("- $1")
    fi
    say "$1"
}
# The report's section, also when a step stops the setup
write_section() {
    {
        echo "## Setup"
        echo
        printf '%s\n' "${lines[@]}"
        if [ -f "$section.packages" ]; then
            echo
            cat "$section.packages"
            rm -f -- "$section.packages"
        fi
    } >"$section"
}
trap write_section EXIT
fail() {
    note "STOPPED: $*"
    die "$*"
}

# The package manager, and the system packages: the launcher's table (every group) and the kit's own.
manager=""
case " $(os_release ID) $(os_release ID_LIKE) " in
    *" debian "* | *" ubuntu "*) manager=apt ;;
    *" rhel "* | *" fedora "* | *" centos "* | *" rocky "* | *" almalinux "*) manager=dnf ;;
esac
[ -n "$manager" ] || fail "the kit installs system packages with apt or dnf; $(os_release PRETTY_NAME) has neither"
case $manager in
    apt)
        column=3
        kit_packages=(ca-certificates curl unzip git xz-utils procps pciutils libatomic1 libnuma1 libvulkan1
            mesa-vulkan-drivers vulkan-tools)
        ;;
    dnf)
        column=4
        kit_packages=(ca-certificates curl unzip git xz procps-ng pciutils libatomic numactl-libs vulkan-loader
            mesa-vulkan-drivers vulkan-tools)
        ;;
esac
# Debian 13 and Ubuntu 24.04 (glibc 2.38 and newer) name some packages with a t64 suffix (the table's "(t64)").
suffix=""
glibc_minor=$(getconf GNU_LIBC_VERSION | sed -n 's/^glibc 2\.\([0-9]*\).*/\1/p')
[ "${glibc_minor:-0}" -ge 38 ] && suffix=t64
table=$REPO/backend/src/linux_libraries.txt
mapfile -t table_packages < <(awk -v column="$column" -v suffix="$suffix" \
    '!/^#/ && NF >= 6 { name = $column; sub(/\(t64\)$/, suffix, name); print name }' "$table" | sort -u)
packages=("${kit_packages[@]}" "${table_packages[@]}")
say "system packages ($manager): ${packages[*]}"
case $manager in
    apt)
        export DEBIAN_FRONTEND=noninteractive
        run apt-get update -qq >"$LOGS/system-packages.log" 2>&1 || fail "apt-get update failed ($LOGS/system-packages.log)"
        run apt-get install -y -qq --no-install-recommends "${packages[@]}" >>"$LOGS/system-packages.log" 2>&1 ||
            fail "installing the system packages failed ($LOGS/system-packages.log)"
        ;;
    dnf)
        run dnf -y -q install "${packages[@]}" >"$LOGS/system-packages.log" 2>&1 ||
            fail "installing the system packages failed ($LOGS/system-packages.log)"
        ;;
esac
note "System packages: ${#packages[@]} through $manager"

# The app, unpacked once per zip
if [ -n "$zip" ]; then
    [ -f "$zip" ] || fail "no zip at $zip"
    zip_sha=$(sha256_of "$zip")
    note "App zip: $(basename "$zip"), SHA-256 $zip_sha"
    if [ "$(cat "$WORK/app/zip-sha256" 2>/dev/null)" != "$zip_sha" ]; then
        run rm -rf -- "$WORK/app"
        run mkdir -p "$WORK/app"
        run unzip -q "$zip" -d "$WORK/app" || fail "unzipping $zip failed"
        [ "$DRY_RUN" = 1 ] || printf '%s\n' "$zip_sha" >"$WORK/app/zip-sha256"
    fi
elif [ "$DRY_RUN" = 0 ] && [ ! -x "$APP/chaiNNer.sh" ]; then
    fail "no app: pass --zip with the Linux zip (chaiNNer-linux-x64)"
fi
# A zip from before the fresh-install fix of 2026-10-29 cannot start its backend on a fresh install: its
# dependencies/store.py imports packaging, which a fresh Python lacks, before the server's packages are installed.
if [ "$DRY_RUN" = 0 ] && ! grep -q '_version_module' "$APP/resources/src/dependencies/store.py"; then
    fail "this zip is older than the fix of 2026-10-29 that lets a fresh install start its backend: build the zip from the current chaiNNer-C branch"
fi

# The first headless run: the app downloads its Python and the backend's own packages, then runs a chain that needs
# no package (Load Image, Save Image).
smoke_out=$WORK/smoke
mkdir -p "$smoke_out"
sed "s|@IMAGE@|$KIT_DIR/chains/input.png|; s|@OUT@|$smoke_out|; s|@NAME@|smoke|" "$KIT_DIR/chains/smoke.json" >"$WORK/smoke-override.json"
start=$SECONDS
if run "$APP/chaiNNer.sh" run "$KIT_DIR/chains/smoke.chn" --override "$WORK/smoke-override.json" >"$LOGS/first-run.log" 2>&1 &&
    { [ "$DRY_RUN" = 1 ] || [ -f "$smoke_out/smoke.png" ]; }; then
    note "First headless run (Python download, Load Image, Save Image): passed in $((SECONDS - start)) s"
else
    note "First headless run: FAILED ($LOGS/first-run.log)"
    tail -20 "$LOGS/first-run.log" >&2
    fail "the app did not run its first chain"
fi
[ "$DRY_RUN" = 1 ] || [ -x "$APP_PYTHON" ] || fail "the first run left no Python at $APP_PYTHON"
pip=("$APP_PYTHON" -s -m pip --disable-pip-version-check --no-input)

# The wheels the kit installs before the dependency manager: a copy, else the URL. spandrel goes without its
# dependencies, which are the PyTorch package's (the ROCm build of torch).
preinstall() {
    local name=$1 pattern=$2 url=$3 sha=$4 deps=$5 copy=""
    if [ -n "$wheels" ]; then
        copy=$(find "$wheels" -maxdepth 1 -name "$pattern" | sort | tail -n 1)
    fi
    local source=$copy
    if [ -z "$source" ]; then
        [ -n "$url" ] || return 0
        source=$url${sha:+#sha256=$sha}
    fi
    local args=()
    [ "$deps" = 1 ] || args+=(--no-deps)
    if run "${pip[@]}" install "${args[@]}" "$source" >>"$LOGS/wheels.log" 2>&1; then
        note "$name: installed from ${copy:-$url}${copy:+ (copy, SHA-256 $(sha256_of "$copy"))}"
    else
        note "$name: FAILED to install from ${copy:-$url} ($LOGS/wheels.log)"
    fi
}
preinstall "ONNX Runtime MIGraphX wheel" "$ORT_MIGRAPHX_WHEEL_PATTERN" "$ORT_MIGRAPHX_WHEEL_URL" "$ORT_MIGRAPHX_WHEEL_SHA256" 1
preinstall "spandrel wheel" "$SPANDREL_WHEEL_PATTERN" "$SPANDREL_WHEEL_URL" "$SPANDREL_WHEEL_SHA256" 0

# The dependency manager: the backend the app starts, asked to install each package as its window asks (one at a
# time, so that one that fails leaves the others installed).
if [ "$DRY_RUN" = 1 ]; then
    say "[dry run] start the backend on port $BACKEND_PORT and POST /packages/install for ${KIT_PACKAGES[*]}"
    note "Dependency manager: not run (dry run)"
else
    start=$SECONDS
    if backend_start "$LOGS/dependency-manager.log"; then
        listed=$(backend GET /packages)
        for package in "${KIT_PACKAGES[@]}"; do
            # A build without MIGraphX lists no chaiNNer_migraphx
            if ! grep -q "\"id\": *\"$package\"" <<<"$listed"; then
                note "Dependency manager: this build has no package $package"
                continue
            fi
            step=$SECONDS
            response=$(backend POST /packages/install "{\"packages\": [\"$package\"]}")
            note "Dependency manager: $package in $((SECONDS - step)) s: $response"
        done
        backend GET /packages >"$LOGS/packages.json"
        backend GET /nodes >"$LOGS/nodes.json"
        backend GET /installed-dependencies >"$LOGS/installed.json"
        note "Dependency manager: done in $((SECONDS - start)) s"
    else
        note "Dependency manager: the backend did not get ready ($LOGS/dependency-manager.log)"
    fi
    backend_stop
    if [ -s "$LOGS/packages.json" ]; then
        "$APP_PYTHON" -I "$KIT_DIR/describe.py" packages "$LOGS/packages.json" "$LOGS/nodes.json" "$LOGS/installed.json" >>"$section.packages" ||
            note "Reading the package list failed ($LOGS/packages.json)"
    fi
fi

# The test environment: the app's Python with its packages, plus pytest at the lock's versions.
mapfile -t test_pins < <(grep -E '^(pytest|pytest-asyncio)==' "$REPO/native/python-stack.lock.linux.txt")
if [ ! -x "$TEST_PYTHON" ]; then
    run "$APP_PYTHON" -s -m venv --system-site-packages "$WORK/test-env" || fail "making the test environment failed"
fi
run "$TEST_PYTHON" -m pip install --disable-pip-version-check -q "${test_pins[@]}" >"$LOGS/test-env.log" 2>&1 ||
    fail "installing ${test_pins[*]} failed ($LOGS/test-env.log)"
note "Test environment: $WORK/test-env (${test_pins[*]})"

# The zip's native binaries (git-ignored build outputs) into this checkout's backend/src, and how the zip's backend
# differs from the checkout's (a zip from another commit can fail tests written for this one).
if [ -d "$APP/resources/src" ]; then
    copied=0
    while IFS= read -r -d '' file; do
        relative=${file#"$APP/resources/src/"}
        if [ ! -e "$REPO/backend/src/$relative" ]; then
            run mkdir -p "$(dirname "$REPO/backend/src/$relative")"
            run cp -p "$file" "$REPO/backend/src/$relative"
            copied=$((copied + 1))
        fi
    done < <(find "$APP/resources/src" -type f ! -name '*.pyc' ! -path '*/__pycache__/*' -print0)
    differing=$(cd "$APP/resources/src" && find . -name '*.py' -type f -print0 |
        while IFS= read -r -d '' file; do
            cmp -s "$file" "$REPO/backend/src/$file" || echo "${file#./}"
        done)
    note "Zip binaries copied into the checkout: $copied files"
    if [ -n "$differing" ]; then
        note "Zip and checkout differ in $(wc -l <<<"$differing") backend Python files (the zip is from another commit): $(head -n 10 <<<"$differing" | tr '\n' ' ')"
    else
        note "Zip and checkout: the same backend Python files"
    fi
fi

# The models: role files in --models (esrgan, dual, drft, hat, dat2; .pth or .safetensors), and a public ESRGAN when
# there is none.
: >"$OUT/models.txt"
for role in esrgan dual drft hat dat2; do
    found=""
    if [ -n "$models" ]; then
        found=$(find "$models" -maxdepth 1 \( -name "$role.pth" -o -name "$role.safetensors" \) | head -n 1)
    fi
    if [ -z "$found" ] && [ "$role" = esrgan ]; then
        found=$WORK/models/esrgan.pth
        if [ ! -f "$found" ]; then
            run mkdir -p "$WORK/models"
            run curl -fsSL -o "$found.part" "$ESRGAN_URL" || fail "downloading $ESRGAN_URL failed"
            if [ "$DRY_RUN" = 0 ]; then
                [ "$(sha256_of "$found.part")" = "$ESRGAN_SHA256" ] || fail "$ESRGAN_URL does not match its SHA-256"
                mv "$found.part" "$found"
            fi
        fi
    fi
    if [ -n "$found" ]; then
        printf '%s %s\n' "$role" "$found" >>"$OUT/models.txt"
        note "Model $role: $found"
    else
        note "Model $role: none (its chains and speed rows are skipped)"
    fi
done
# The small models some amd_gpu tests read (CHAINNER_C_TEST_MODELS): the folder tests in --models.
if [ -n "$models" ] && [ -d "$models/tests" ]; then
    echo "tests $models/tests" >>"$OUT/models.txt"
    note "Test models: $models/tests ($(find "$models/tests" -maxdepth 1 -type f -printf '%f '))"
else
    note "Test models: none (the tests that read them skip)"
fi
# MIGraphX's example custom-op library, which neither the zip nor the repository carries: from --models, where the
# runner's GPU custom-op test finds it (EXAMPLE_OPS).
if [ -n "$models" ] && [ -f "$models/chainner_example_migx_ops.so" ]; then
    run mkdir -p "$KIT_DIR/$(dirname "$EXAMPLE_OPS")"
    run cp "$models/chainner_example_migx_ops.so" "$KIT_DIR/$EXAMPLE_OPS"
    note "MIGraphX example ops: $models/chainner_example_migx_ops.so"
else
    note "MIGraphX example ops: none (the runner's GPU custom-op test skips)"
fi
# VOSR's model has no file here: the app downloads it (7 GB) on first use.
if [ "${KIT_SKIP_VOSR:-0}" = 1 ]; then
    note "Model vosr: skipped (KIT_SKIP_VOSR=1)"
else
    echo "vosr download" >>"$OUT/models.txt"
    note "Model vosr: downloaded by the app on first use"
fi
say "setup done"
