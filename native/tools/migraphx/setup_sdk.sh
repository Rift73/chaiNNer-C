#!/bin/bash
# In the cnm-build image (build.sh's sdk step): AMD's ROCm 7.14.1 pip SDK (rocm-sdk-core, -libraries and -devel from
# /rocm-wheels) in a venv of the work directory, its devel tree expanded by `rocm-sdk init`. AMD publishes no hashes for
# these files; the SHA-256 below are those of the files the released wheel was built from. The `rocm` meta package is
# an sdist; its pure-Python rocm_sdk module is copied in, since building the sdist would need an index. The ROCm root
# goes to /work/rocm-root.txt.
set -euo pipefail
V=/work/rocm-venv
PY=/opt/python/cp314-cp314/bin/python3.14
cd /rocm-wheels
sha256sum -c - <<'EOF'
ffea97f6a9da9987a9260f9e635995b27a44dd6985d1b2b1663431356aaf669b  rocm-7.14.1.tar.gz
da1f9c960a55a64c3134004ba1346d2424090cc3e2336ccfec94e9f62169cf85  rocm_sdk_core-7.14.1-py3-none-linux_x86_64.whl
dfe9ed137a53d4725b2e280662f3a240fe961dce4f3ad0b977e44a74b5ba6f68  rocm_sdk_libraries-7.14.1-py3-none-linux_x86_64.whl
858088ac949d45be6621557e5d7b7f020235e12068c62b6be2bfd0b4234be3dd  rocm_sdk_devel-7.14.1-py3-none-linux_x86_64.whl
EOF
if [ ! -x "$V/bin/python" ]; then
    "$PY" -m venv "$V"
fi
"$V/bin/python" -m pip install --disable-pip-version-check --no-index --no-deps -q \
    rocm_sdk_core-7.14.1-py3-none-linux_x86_64.whl rocm_sdk_libraries-7.14.1-py3-none-linux_x86_64.whl \
    rocm_sdk_devel-7.14.1-py3-none-linux_x86_64.whl
SP=$("$V/bin/python" -I -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
TMP=$(mktemp -d /work/sdist.XXXX)
tar -xzf rocm-7.14.1.tar.gz -C "$TMP"
rm -rf "$SP/rocm_sdk"
cp -r "$TMP/rocm-7.14.1/src/rocm_sdk" "$SP/rocm_sdk"
rm -rf "$TMP"
cd /work
"$V/bin/python" -I -m rocm_sdk init
ROOT=$("$V/bin/python" -I -m rocm_sdk path --root)
echo "$ROOT" > /work/rocm-root.txt
echo "ROCm root: $ROOT"
"$V/bin/python" -I -m rocm_sdk version
"$ROOT/lib/llvm/bin/clang++" --version | head -2
