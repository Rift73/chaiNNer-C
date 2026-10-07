# Maintaining chaiNNer-C

Checklists for the changes that touch the exactness contract, the build or the package. Why each rule exists:
`DESIGN-DECISIONS.md`. Commands in full: `README.md`. Run everything in PowerShell from the repository root;
`$py` is `.\native\.venv\Scripts\python.exe`, always run with `-B` (README Test sets the environment).

## 1. Upgrade a Python dependency or the whole stack

1. Edit the versions in the `-Relock` install list (`$Packages`, and the torch line) in
   `native/tools/provision_runtime.ps1`; an unpinned name takes the latest version.
2. `powershell -NoProfile -File native\tools\provision_runtime.ps1 -Relock` writes a new
   `native/python-stack.lock.txt`. Review its diff.
3. Set each changed `Dependency` pin in `backend/src/packages/*/__init__.py` to the lock's version. A changed
   `pypi_name` or index needs a row in `verify_runtime.DECLARATION_CHANGES` (`native/tools/verify_runtime.py`).
4. NumPy, OpenCV or ONNX Runtime: re-validate the mirrors against the new version's code (the dispatch mirrors
   `native_opencv_simd.py`, `native_numpy_simd.py`, `native_numpy_reduce.py` in `backend/src/nodes/impl/`), then
   change `CV2`, `NUMPY` or `ORT` in `native_versions.py`. Until then the old constants keep the library's own path.
   A NumPy change also updates `ORACLE` in `native/tools/golden_kernels.py` and its pin in `test_golden_kernels.py`.
5. Rebuild (`_chainner_graph.pyd` compiles against NumPy's headers): `& '.\native\Build.ps1' -Configuration Release`.
6. Run the full suite (README Test), ruff and pyright; the goldens (`test_golden_kernels.py`, `test_forced_paths.py`)
   must pass unchanged.
7. A changed lock needs a new package (a refresh refuses another stack): `package_port.py --destination out\<child>`.
8. `make_oracle.py`, then the oracle verifiers and the parity bench (section 10).
9. Update the stack lines in the root `README.md` and the versions in `ARCHITECTURE.md` section 7.

## 2. Upgrade CPython

1. `powershell -NoProfile -File native\tools\provision_runtime.ps1 -Relock -Version <x.y.z> -Tag <build tag>`
   (python-build-standalone) creates `native\runtime\cpython-<x.y.z>` and the new lock.
2. Recreate the venv: `native\runtime\cpython-<x.y.z>\python.exe -m venv --system-site-packages native\.venv`, then
   `& $py -m pip install -r requirements.txt`.
3. Replace `cpython-3.14.8` everywhere it is named: `native/CMakeLists.txt` (`CHAINNER_C_PYTHON_ROOT`), `package.json`
   (`type-check:runtime`, `test:py`), `pyrightconfig.json` (`venv`, and `pythonVersion` for a new minor),
   `.github/actions/backend-env/action.yml`, `.github/workflows/lint-backend.yml`, and the docs. A new minor also
   changes `target-version` in `native/pyproject.toml`.
4. The source UI's download table: `src/main/python/integratedPython.ts` (URLs and version, all platforms).
5. Re-check the mirrors pinned to CPython against the new source: `native/src/utility_random.cpp` (`random.py`) and
   `native/src/video_io.cpp` (`gen_close`); `ORACLE` in `native/tools/golden_kernels.py`.
6. The package's integrated-Python patch accepts 3.14.0 or newer (`independent_ui.py`; `verify_independent_ui.cjs`
   tests it). A version below 3.14 needs that patch changed. Python 3.16 removes the event-loop policy API Sanic
   uses, so 3.16 also needs a Sanic release without it.
7. Then section 1, steps 5 to 9 (new build, suite, new package, oracle, verifiers, docs).

## 3. Add or change a native kernel

1. Scalar code in `native/src/<name>.c`, declared in `native/include/chainner.h`; the C entry reads `cn_isa_current()`
   once and passes the level down.
2. SIMD code only where bit-identical, in an export-free unit `native/src/<family>_avx2.c` (or `_avx512.c`, kept only
   for a measured win). Add it to the source list and to the matching `set_source_files_properties` list in
   `native/CMakeLists.txt`; per-element code shared with the scalar unit goes in a static helper in
   `native/include/<family>_shared.h`.
3. The bridge in `backend/src/nodes/impl/native*.py` validates dtype, contiguity and alignment; a mirror of a library
   path checks `native_versions.py`.
4. Save the old binaries to `native\build\held\`, then build (`Build.ps1 -Configuration Release`).
5. Run the suite once per level: `$env:CHAINNER_C_ISA = 'scalar'`, `'avx2'`, `'avx512'` (then remove it).
   `test_golden_kernels.py` (manifests in `native/tests/golden/`; commands in the `golden_kernels.py` docstring),
   `test_forced_paths.py` and `test_isa_dispatch.py` must pass at each.
6. A change meant to leave code unchanged: `& $py -B native\tools\isa_check.py text <old> <new>`.
7. Record the three new SHA-256 values (`Get-FileHash`) in the hash table of `native/README.md` (Build).
8. An adapted algorithm gets its notice in `backend/src/nodes/impl/chainner_native.LICENSE.txt`.
9. Package refresh, `make_oracle.py`, verifiers and parity bench (section 10).

## 4. Change compiler flags or the toolchain

1. Versions are pinned once each: clang-cl in `native/CMakeLists.txt`, the MSVC toolset and Windows SDK in
   `native/toolchain.cmake`; `package_port.py` reads them into the manifest's `native_toolchain`.
2. Any codegen change (an LLVM bump, a new flag, PGO, `-mprefer-vector-width=512`): run
   `native/tests/test_spectral_filter.py::test_packed_spectrum_all_parities`, `test_golden_kernels.py` (`-k "tail or
   dither"` at least) and `test_forced_paths.py` under each `CHAINNER_C_ISA` level, then the full suite.
3. A NaN-bit failure is fixed in source, in the integer shape of `normal_nan_outputs`
   (`native/include/color_shared.h`), never by a flag.
4. Keep `/fp:strict`, dynamic denormals, `/Brepro` and the prefix map; `CHAINNER_C_MARCH` never in a release.
5. Check reproducibility (two checkouts give the same hashes), then update the `native/README.md` hash table and the
   prerequisites in both READMEs.

## 5. Any UI change

1. Change BOTH paths: `src/` (the source UI) and a reviewed patch for the package: `native/tools/independent_ui.py` or
   a patch JSON beside it (`input-drop.json`, `tensorrt-types.json`, `tensorrt-clear.json`), each pinned to the
   baseline bundles' SHA-256, with every anchor occurring exactly once. `independent_ui.py` loads each JSON (a new one
   is added there, as `TENSORRT_PATCHES` are).
2. Tests: `native/tests/test_independent_ui.py`, then build the package (`package_port.py --refresh`).
3. Check the patched bundles: `node native\tools\verify_independent_ui.cjs --package <dir> --installed-app <dir>`, and
   `verify_input_drop.cjs`, `verify_edge_drop.cjs`, `verify_directory_drop.cjs` the same way.
4. `src/`: `npm run lint:js`, `npm run type-check:js`, `npm run test:js`.
5. A version bump changes `src/common/version.ts`, `PRODUCT_VERSION` in `independent_ui.py` and in
   `verify_independent_ui.cjs`.

## 6. Move to a new upstream version or nightly

1. Upstream base: `src/`, `tests/` and `backend/tests` are upstream `d56e507f`. Rebase onto the new commit, keep the
   native wiring in `backend/src`, take the new `backend/tests`, and update `UPSTREAM` in
   `native/tests/test_gpu_lease_removed.py`.
2. Install the new nightly (read-only), then point the tools at it: `INSTALLED_APP` in `native/tools/make_oracle.py`
   and the `--installed-app` path in the README commands.
3. Re-anchor the patches: new `BASELINE_HASHES` in `independent_ui.py`, each patch JSON's baseline hashes and
   anchors, `INSTALLED_SHA256` in `native/tools/bundle_verify_common.cjs`; review the minified names the
   `verify_*.cjs` tools use. Unknown bundles fail closed until this is done.
4. `& $py -B native\tools\make_oracle.py --check`: `native/tests/oracle/compat.diff` must still apply.
5. Port upstream's UI changes into `src/` (section 5); build a new package (`--destination out\<child>`).
6. The full acceptance (section 10). Frozen `native/tests/reference_*` oracles are never edited to make a test pass.

## 7. TensorRT and cuda-python

1. Pins live in `backend/src/packages/chaiNNer_tensorrt/__init__.py` (`tensorrt` 11.2.1.2 with `auto_update=False`,
   `cuda-python` 13.4.1); they are GPU-only and not in the lock.
2. A serialized engine loads only on the TensorRT version and GPU architecture that built it
   (`nodes/impl/tensorrt/engine_info.py`). `auto_update=False` keeps a pin bump from replacing an installed TensorRT
   at startup; after a bump, users rebuild engines with Build Engine.
3. API changes: update the stubs `stubs/tensorrt/__init__.pyi` and `stubs/cuda/bindings/runtime.pyi` (pyright).
4. CPU tests: `backend/tests/test_tensorrt_tiling.py` (tiling and seams), `native/tests/test_tensorrt_cache.py`
   (sharing and release). GPU runs are manual.

## 8. Package and release

1. Build the native layer; package: `package_port.py --validate-only`, then build or `--refresh`, then a repeat run
   that must report `verified_no_op` (README Package).
2. Bundle checks (section 5, step 3), `make_oracle.py`, the gates of section 10.
3. Release zip from source: `npm ci`, the native build (Forge copies `backend/src` with the binaries), then
   `npm run make`; `forge.config.js` adds the `portable` marker to the zip and deletes `.pyc` files first. Its top
   maps cross-zip's `fs.rmdir(..., { recursive })`, which Node 25 removed, to `fs.rm`: keep it while maker-zip uses
   cross-zip.
4. The `native/README.md` hash table must match the shipped binaries.

## 9. Publish

1. Nothing private: the git-ignored `native/reports/`, `docs/superpowers/`, `native/local/`, `native/runtime/` and
   `out/` stay local (`git status --ignored`); grep the tree for personal paths and names before publishing.
2. One commit on upstream `d56e507f`, for example `git commit-tree "HEAD^{tree}" -p d56e507f -m "<message>"`, and a
   branch at the printed hash.
3. `git diff --check`, LF only (`.gitattributes`), hosted CI green (`.github/workflows/lint-backend.yml`).
4. Optionally attach the release zip (section 8, step 3).

## 10. Local acceptance gates

1. Suite, ruff and pyright (README Test).
2. `& $py -B native\tools\make_oracle.py` after every native build.
3. Runtime verifiers: `verify_runtime.py`, then `verify_io_execution_runtime.py`, `verify_utility_runtime.py` and
   `verify_video_runtime.py`, each with `--include-port` (without it, the oracle alone). `verify_framework_runtime.py`
   loads models: not in a no-model CPU run.
4. Parity bench against the oracle: `bench.py --a native\build\oracle\src` (13 cases; README Bench).
5. Real-use pass: `& $py -B native\tools\real_use_pass.py [--package DIR]`; exit 0 only if every gate holds.
6. Machine-specific paths, edited on another machine: the bench media root `F:\chaiNNer-C-diagnostics`
   (`DIAGNOSTICS`, `bench.py`), the real-use chains under `C:\Executables\models\ESRGAN` (`MODELS`,
   `real_use_pass.py`), the growth dataset (`DATASET`, `numpy_pool_growth.py`), FFmpeg 8.1 and Node (README
   Prerequisites).
