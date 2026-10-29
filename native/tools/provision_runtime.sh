#!/usr/bin/env bash
# Provisions the Python runtime chaiNNer-C builds, tests and packages with on Linux: python-build-standalone CPython
# into native/runtime/cpython-<version> (git-ignored), then the stack the tracked lock native/python-stack.lock.linux.txt
# records. It is provision_runtime.ps1's counterpart.
# By default the lock is the input. Its header names the interpreter, the build tag, the archive's SHA-256 and pip's
# version; its pins are installed as they stand, without dependency resolution, each from one source (torch and
# torchvision from the PyTorch CUDA index, chainner-pip from the backend's bundled wheel, spandrel from its fork's release
# wheel as the lock's direct-URL pin, the rest from PyPI). The run fails unless pip check is clean, pip's version is the
# header's and pip freeze equals the pins line for line. The lock is not written.
# --relock is the deliberate upgrade step, followed by the full acceptance: the archive checked against the release's
# SHA256SUMS, the backend's dependencies at their latest versions, then the freeze and a new lock (the freeze under a
# header with the interpreter, the archive's SHA-256 and pip's version).
# --build-only provisions what the native build needs and nothing else: the lock's CPython, downloaded and checked as
# above, with only the lock's numpy and pybind11 pins (no torch, no pip check or freeze comparison). The full run on the
# same runtime later installs the rest.
# --with-extras is for development and test runtimes: after the lock's pins it installs the packages the app's dependency
# manager installs on demand (TensorRT, CUDA Python, VapourSynth with its plugins) at the backend's versions, and the
# test tools (ruff, pyright, viztracer, appdirs), with the lock's pins as constraints. The freeze then has to contain
# every pin of the lock instead of equalling them.
# Usage: native/tools/provision_runtime.sh [--relock | --build-only] [--with-extras] [--version 3.14.8] [--tag 20261003]
#                                          [--runtime-dir DIR] [--lock FILE]
# --version and --tag default to the lock's; without --relock any other value is refused. --relock needs no readable
# lock header (it writes a new one), only --version and --tag when the lock cannot supply them. --runtime-dir (default
# native/runtime) is where cpython-<version> is extracted, for a checkout on a filesystem that cannot hold it.
# The archive's SHA-256 is recorded beside the runtime it was extracted into (cpython-<version>.archive-sha256), and an
# existing runtime extracted from another archive, or with no record, is refused: remove it to extract this one.
set -euo pipefail

relock=0
build_only=0
with_extras=0
version=""
tag=""
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runtime_dir="$root/native/runtime"
lock_path="$root/native/python-stack.lock.linux.txt"
torch_index="https://download.pytorch.org/whl/cu132"
jet_index="https://jaded-encoding-thaumaturgy.github.io/vs-wheels/simple"
spandrel_pin="spandrel @ https://github.com/Rift73/spandrel/releases/download/v0.4.2-c10/spandrel-0.4.2%2Bc10-py3-none-any.whl#sha256=2496741447f096efa803d01798262ab065d0c4e79b4266f30a32cab4e254b50f"

die() {
    echo "error: $*" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --relock) relock=1 ;;
        --build-only) build_only=1 ;;
        --with-extras) with_extras=1 ;;
        --version) version=${2:?--version needs a value}; shift ;;
        --tag) tag=${2:?--tag needs a value}; shift ;;
        --runtime-dir) runtime_dir=${2:?--runtime-dir needs a value}; shift ;;
        --lock) lock_path=${2:?--lock needs a value}; shift ;;
        -h | --help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument $1" ;;
    esac
    shift
done
[ "$relock" = 1 ] && [ "$build_only" = 1 ] && die "--build-only provisions from the lock; it cannot be combined with --relock"
[ "$with_extras" = 1 ] && [ "$build_only$relock" != 00 ] && die "--with-extras cannot be combined with --build-only or --relock"

# Isolated from user state: no user site-packages, no pip user configuration and no PIP_* variables, so every
# requirement is installed into (and frozen from) the runtime itself.
export PYTHONNOUSERSITE=1

# The lock's header fields, matched as package_manifest.LOCK_FIELDS matches them, and its pins without their notes.
header_field() {
    [ -f "$lock_path" ] && sed -n -E "s/$1/\\1/p" "$lock_path" | head -n 1 || true
}
header_interpreter=$(header_field '^# Interpreter: CPython ([0-9]+\.[0-9]+\.[0-9]+) .*$')
header_build_tag=$(header_field '^# Build tag: python-build-standalone ([0-9]+)$')
header_archive_sha256=$(header_field '^# Archive SHA-256: ([0-9a-f]{64})$')
header_pip=$(header_field '^# pip: ([^ ]+) .*$')
pins=()
if [ -f "$lock_path" ]; then
    mapfile -t pins < <(grep -vE '^(#|$)' "$lock_path" | sed 's/  # .*//')
fi

# Without --relock the lock is the input and must be whole; --relock writes a new one, so a missing lock or header field
# fails it only when --version or --tag is absent too.
if [ "$relock" = 0 ]; then
    missing=""
    [ -n "$header_interpreter" ] || missing="$missing interpreter"
    [ -n "$header_build_tag" ] || missing="$missing build_tag"
    [ -n "$header_archive_sha256" ] || missing="$missing archive_sha256"
    [ -n "$header_pip" ] || missing="$missing pip"
    [ -z "$missing" ] || die "the lock $lock_path lacks the header fields:$missing"
    [ -z "$version" ] || [ "$version" = "$header_interpreter" ] ||
        die "--version $version is not the lock's CPython $header_interpreter; another interpreter is an upgrade (--relock)"
    [ -z "$tag" ] || [ "$tag" = "$header_build_tag" ] ||
        die "--tag $tag is not the lock's build tag $header_build_tag; another build is an upgrade (--relock)"
fi
version=${version:-$header_interpreter}
tag=${tag:-$header_build_tag}
[ -n "$version" ] && [ -n "$tag" ] || die "the lock $lock_path names no interpreter or build tag; --relock needs --version and --tag"

target="$runtime_dir/cpython-$version"
download="$runtime_dir/download"
name="cpython-$version+$tag-x86_64-unknown-linux-gnu-install_only.tar.gz"
base="https://github.com/astral-sh/python-build-standalone/releases/download/$tag"
python="$target/bin/python3"
pip=("$python" -s -m pip --isolated)

mkdir -p "$download"
# The archive is kept and checked on every run: against the lock's SHA-256, or with --relock against the release's
# SHA256SUMS (the new lock then records it).
archive="$download/$name"
[ -f "$archive" ] || curl -fsSL -o "$archive" "$base/${name//+/%2B}"
if [ "$relock" = 1 ]; then
    sums_file="$download/SHA256SUMS-$tag"
    [ -f "$sums_file" ] || curl -fsSL -o "$sums_file" "$base/SHA256SUMS"
    expected=$(awk -v name="$name" '$2 == name { print $1 }' "$sums_file")
    [ -n "$expected" ] || die "no SHA-256 for $name in the release's SHA256SUMS"
else
    expected=$header_archive_sha256
fi
archive_sha=$(sha256sum "$archive" | cut -d ' ' -f 1)
if [ "$archive_sha" != "$expected" ]; then
    # A partial or corrupt download would fail every later run the same way.
    rm -f -- "$archive"
    die "SHA-256 mismatch for $name: expected '$expected', got '$archive_sha'; the download was removed"
fi
echo "verified $name $archive_sha ($(stat -c %s "$archive") bytes)"

# The runtime's directory names its version, not its build: the archive it was extracted from is recorded beside it
# (outside it, so the packaged runtime is unchanged), and an existing runtime is used only if that is this archive.
extracted_from="$runtime_dir/cpython-$version.archive-sha256"
if [ -x "$python" ]; then
    recorded=$(cat "$extracted_from" 2>/dev/null || true)
    [ -n "$recorded" ] || die "the runtime $target has no record of its archive ($extracted_from); remove it to extract $name"
    [ "$recorded" = "$expected" ] ||
        die "the runtime $target was extracted from the archive $recorded, not $name ($expected); remove it to extract this one"
    echo "runtime $target exists, extracted from $name; skipping its extraction (pip still installs into it)"
else
    stage="$runtime_dir/stage-$version"
    # A failed run's stage is never extracted over.
    rm -rf -- "$stage"
    mkdir -p "$stage"
    tar -xzf "$archive" -C "$stage"
    # Recorded before the move: a move that fails leaves no runtime, and the next run extracts again.
    printf '%s\n' "$archive_sha" >"$extracted_from"
    mv "$stage/python" "$target"
    rmdir "$stage"
fi
minor_version=${version%.*}
[ -f "$target/include/python$minor_version/Python.h" ] || die "the runtime has no headers (include/python$minor_version/Python.h)"
"$python" -c "import sys; print('runtime', sys.version)"

if [ "$build_only" = 1 ]; then
    # The build's inputs only: NumPy's headers and pybind11, at the lock's pins, from PyPI.
    mapfile -t build_pins < <(printf '%s\n' "${pins[@]}" | grep -E '^(numpy|pybind11)==' || true)
    [ "${#build_pins[@]}" = 2 ] || die "the lock $lock_path does not pin numpy and pybind11 once each"
    "${pip[@]}" install --no-cache-dir --no-deps "${build_pins[@]}"
    echo "provisioned $target for the build only (${build_pins[*]})"
    exit 0
fi

# chainner-pip from the backend's bundled wheel, which the host would otherwise install into the package at first
# launch. Resolved by name from that directory only, so the freeze pins its version, not a file path.
wheels="$root/backend/src/dependencies/whls/chainner-pip"
if [ "$relock" = 1 ]; then
    "${pip[@]}" install --no-cache-dir --upgrade pip
    # The backend's dependencies at their latest versions, CUDA torch from the PyTorch index (which also serves the CUDA
    # libraries and Triton it depends on).
    "${pip[@]}" install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url "$torch_index"
    packages=(
        'numpy==2.5.3' 'opencv-python==5.0.0.93' 'Pillow==12.3.0' 'scipy==1.18.1' 'numba==0.68.0' 'llvmlite==0.50.0'
        'PyMatting==1.1.16' 'pillow-avif-plugin' 'ffmpeg-python' 'requests' 'wcmatch'
        'google-re2'  # oracle/reference only (upstream's onnx/load.py imports re2)
        'Sanic-Cors==2.2.0'  # oracle/reference only (upstream's server imports sanic_cors)
        "$spandrel_pin"
        'spandrel_extra_arches' 'facexlib' 'einops' 'safetensors'
        'onnxruntime-gpu @ https://github.com/Rift73/onnxruntime/releases/download/v1.30.0-trt11/onnxruntime_gpu-1.30.0%2Btrt11.4-cp314-cp314-manylinux_2_28_x86_64.whl#sha256=b67a9a5b03df5fd4687d33d41dc035542d4ec2c4da8da57741749ee21b783778'
        'onnx' 'onnxoptimizer' 'protobuf' 'ncnn==1.0.20260526'
        'sanic==25.12.1' 'aiofiles' 'html5tagger' 'sanic-routing' 'tracerite' 'websockets' 'typing_extensions'
        'nvidia-ml-py' 'psutil' 'aiohttp'
        'pynvml'  # oracle/reference only (upstream's server declares it; nvidia-ml-py provides the module)
        'pybind11==3.1.0' 'pytest' 'pytest-asyncio' 'pytest-cov'
    )
    "${pip[@]}" install --no-cache-dir "${packages[@]}"
    "${pip[@]}" install --no-cache-dir --no-deps --no-index --find-links "$wheels" chainner-pip
else
    # The lock is a whole freeze, so every pin is installed as it stands (--no-deps), each from its one source.
    "${pip[@]}" install --no-cache-dir "pip==$header_pip"
    mapfile -t torch_pins < <(printf '%s\n' "${pins[@]}" | grep -E '^(torch|torchvision)==')
    "${pip[@]}" install --no-cache-dir --no-deps --index-url "$torch_index" "${torch_pins[@]}"
    # The rest from PyPI: the lock less those lines (pip reads the notes after a pin as comments).
    pypi_pins="$runtime_dir/pypi-$version.txt"
    grep -vE '^(torch|torchvision|chainner-pip)==' "$lock_path" >"$pypi_pins"
    "${pip[@]}" install --no-cache-dir --no-deps -r "$pypi_pins"
    "${pip[@]}" install --no-cache-dir --no-deps --no-index --find-links "$wheels" "$(printf '%s\n' "${pins[@]}" | grep -E '^chainner-pip==')"
    if [ "$with_extras" = 1 ]; then
        # What the dependency manager installs on demand and what the tests use, at the backend's versions, resolved with
        # their dependencies under the lock's pins.
        constraints="$runtime_dir/constraints-$version.txt"
        printf '%s\n' "${pins[@]}" | grep -E '^[A-Za-z0-9._-]+==' >"$constraints"
        extras=(
            'tensorrt==11.4.0.106' 'cuda-python==13.4.1'
            'VapourSynth==81' 'vsjetpack[deinterlace]==2.2.6' 'vapoursynth-mvutensils==9' 'vapoursynth-cas==3.0'
            'vapoursynth-vszipcu==1.2.0' 'dfttest2[cpu,cufft,nvrtc]==11.0.0' 'vsview==0.13.0'
            'ruff==0.16.10' 'pyright==1.1.414' 'appdirs==1.4.4' 'viztracer==1.1.1'
        )
        "${pip[@]}" install --no-cache-dir -c "$constraints" --extra-index-url "$jet_index" "${extras[@]}"
    fi
    "${pip[@]}" check
fi
"$python" -s -m chainner_pip --version || die "chainner-pip does not run"
pip_version=$("${pip[@]}" --version | cut -d ' ' -f 2)
freeze_path="$runtime_dir/freeze-$version.txt"
"${pip[@]}" freeze >"$freeze_path"
if [ "$relock" = 1 ]; then
    # The lock: the freeze under a header package_port.py reads (python_stack); a pin's note follows it on its line.
    declare -A notes=(
        [google-re2]="oracle/reference only (upstream's onnx/load.py imports re2)"
        [Sanic-Cors]="oracle/reference only (upstream's server imports sanic_cors)"
        [pynvml]="oracle/reference only (upstream's server declares it; nvidia-ml-py provides the module)"
    )
    {
        echo "# The tested set, not a ceiling: the stack chaiNNer-C is tested on. The backend's Dependency"
        echo "# pins are set at these versions and act as floors; newer versions are not excluded."
        echo "# Interpreter: CPython $version (standard build, not free-threaded)."
        echo "# Build tag: python-build-standalone $tag"
        echo "# Archive: $name"
        echo "# Archive SHA-256: $archive_sha"
        echo "# pip: $pip_version (installed; pip freeze does not list it)"
        echo "# Packages: pip freeze of that runtime, provisioned by native/tools/provision_runtime.sh (torch and"
        echo "# torchvision from $torch_index, chainner-pip from the bundled wheel in"
        echo "# backend/src/dependencies/whls/chainner-pip/, spandrel and onnxruntime-gpu from their forks' release"
        echo "# wheels (github.com/Rift73/spandrel, github.com/Rift73/onnxruntime), the rest from PyPI). It is the Windows lock's set with Linux's own"
        echo "# additions: torch's CUDA libraries and Triton, CUDA bindings, and Sanic's uvloop and ujson."
        while IFS= read -r pin; do
            note=${notes[${pin%%==*}]:-}
            if [ -n "$note" ]; then echo "$pin  # $note"; else echo "$pin"; fi
        done <"$freeze_path"
    } >"$lock_path"
elif [ "$with_extras" = 1 ]; then
    # Development runtime: the lock's pins are all present (the extras add packages beyond them).
    missing_pins=$(comm -23 <(printf '%s\n' "${pins[@]}" | sort) <(sort "$freeze_path"))
    [ -z "$missing_pins" ] || die "the runtime lacks pins of the lock ($freeze_path):"$'\n'"$missing_pins"
elif [ "$pip_version" != "$header_pip" ] || ! diff -q <(printf '%s\n' "${pins[@]}") "$freeze_path" >/dev/null; then
    # The runtime is the lock only if pip's version and the freeze, line for line, are the lock's.
    differences=""
    [ "$pip_version" = "$header_pip" ] || differences="pip $pip_version, the lock's $header_pip"$'\n'
    differences="$differences$({ diff <(printf '%s\n' "${pins[@]}") "$freeze_path" || true; } | sed -n 's/^< /lock only: /p; s/^> /freeze only: /p')"
    die "the runtime is not the lock ($freeze_path):"$'\n'"${differences:-the freeze lists the same pins in another order}"
fi
echo "provisioned $target"
