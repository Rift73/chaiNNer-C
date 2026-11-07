#!/bin/bash
# In the cnm-build image (build.sh's sdk step): AMD's ROCm 10.0.0 pip SDK (rocm-sdk-core, -libraries and -devel from
# /rocm-wheels) in a venv of the work directory, its devel tree expanded by `rocm-sdk init`. AMD publishes no hashes for
# these files; the SHA-256 below are those of the files the released wheel was built from. The `rocm` meta package is
# an sdist; its pure-Python rocm_sdk module is copied in, since building the sdist would need an index. The ROCm root
# goes to /work/rocm-root.txt.
set -euo pipefail
V=/work/rocm-venv
PY=/opt/python/cp314-cp314/bin/python3.14
cd /rocm-wheels
sha256sum -c - <<'EOF'
65b5982249612a310135f5e061b9f28909e8ad718a1ddb3f1abba92298b7216b  rocm-10.0.0.tar.gz
3bf4a72d11aa2a4ee1e90572c73630f937d38c1b7af4dda45b483b54655138a7  rocm_sdk_core-10.0.0-py3-none-linux_x86_64.whl
bbe83a550cb8ceeb306abf6d57df6617857a1b19384549526db2ec81a4374e57  rocm_sdk_libraries-10.0.0-py3-none-linux_x86_64.whl
867f06bfe06fc26cfbcd5c30cc9ba88d8b4de86345c3295a3885bfd963346086  rocm_sdk_devel-10.0.0-py3-none-linux_x86_64.whl
EOF
if [ ! -x "$V/bin/python" ]; then
    "$PY" -m venv "$V"
fi
"$V/bin/python" -m pip install --disable-pip-version-check --no-index --no-deps -q \
    rocm_sdk_core-10.0.0-py3-none-linux_x86_64.whl rocm_sdk_libraries-10.0.0-py3-none-linux_x86_64.whl \
    rocm_sdk_devel-10.0.0-py3-none-linux_x86_64.whl
SP=$("$V/bin/python" -I -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
TMP=$(mktemp -d /work/sdist.XXXX)
tar -xzf rocm-10.0.0.tar.gz -C "$TMP"
rm -rf "$SP/rocm_sdk"
cp -r "$TMP/rocm-10.0.0/src/rocm_sdk" "$SP/rocm_sdk"
rm -rf "$TMP"
cd /work
"$V/bin/python" -I -m rocm_sdk init
ROOT=$("$V/bin/python" -I -m rocm_sdk path --root)
echo "$ROOT" > /work/rocm-root.txt
echo "ROCm root: $ROOT"
"$V/bin/python" -I -m rocm_sdk version
"$ROOT/lib/llvm/bin/clang++" --version | head -2
