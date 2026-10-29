# Maintaining chaiNNer-C

Checklists for the changes that touch the exactness contract, the build or the package. Why each rule exists:
`DESIGN-DECISIONS.md`. Commands in full: `README.md`. Run everything in PowerShell from the repository root;
`$py` is `.\native\.venv\Scripts\python.exe`, always run with `-B` (README Test sets the environment). What Linux
adds to these procedures is in section 11.

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

1. Versions are pinned once each: clang-cl and the TensorRT and CUDA majors in `native/CMakeLists.txt`, the MSVC
   toolset and Windows SDK in `native/toolchain.cmake`; `package_port.py` reads them into the manifest's
   `native_toolchain`.
2. Any codegen change (an LLVM bump, a new flag, PGO, `-mprefer-vector-width=512`): run
   `native/tests/test_spectral_filter.py::test_packed_spectrum_all_parities`, `test_golden_kernels.py` (`-k "tail or
   dither"` at least) and `test_forced_paths.py` under each `CHAINNER_C_ISA` level, then the full suite.
3. A NaN-bit failure is fixed in source, in the integer shape of `normal_nan_outputs`
   (`native/include/color_shared.h`), never by a flag.
4. Keep `/fp:strict`, dynamic denormals, `/Brepro` and the prefix map; `CHAINNER_C_MARCH` never in a release.
5. Check reproducibility (two checkouts give the same hashes), then update the `native/README.md` hash table and the
   prerequisites in both READMEs.
6. VOSR's TensorRT plugin `vosr2_qkv.dll` is the one exception: nvcc compiles `src/vosr2/qkv.cu` with the pinned
   toolset's `cl.exe` as its host (approved); the TensorRT (11) and CUDA (13) majors are pinned in
   `CMakeLists.txt`. After a change there (an SDK bump, a GPU architecture, an nvcc flag), build it twice from clean
   directories (same hash), load it (`import tensorrt`, then `ctypes.CDLL`), record its hash and SDK versions in the
   README table and run `native/tests/test_vosr2_trt_rewrite.py`. A new DLL rebuilds the VOSR TensorRT engines once.
   The same exception covers legacy DRFT's plugin library `drft_trt_plugins.dll`: nvcc compiles the four `.cu` files
   of `src/drft` (the kernels and plugin classes) the same way, with nvcc's default device math (the
   numerics) and host warnings not fatal (README); the same checks apply, with `native/tests/test_drft_plugins.py`.
   Engines reference its plugins without carrying the DLL (an engine carrying it registers it at every load, and
   TensorRT refuses a second registration in a process); `drft.load_plugins` registers it once per process.

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
2. Extract the new nightly's official Windows zip into `native/local/upstream/` (git-ignored, read-only), then point
   the tools at it: `UPSTREAM_APP` in `native/tools/independent_ui.py` (the packager, `make_oracle.py` and the tests
   read it), and the zip's URL, SHA-256 and folder in README's Prerequisites.
3. Re-anchor the patches: new `BASELINE_HASHES` in `independent_ui.py`, each patch JSON's baseline hashes and
   anchors, `INSTALLED_SHA256` in `native/tools/bundle_verify_common.cjs`; review the minified names the
   `verify_*.cjs` tools use. Unknown bundles fail closed until this is done.
4. `& $py -B native\tools\make_oracle.py --check`: `native/tests/oracle/compat.diff` must still apply.
5. Port upstream's UI changes into `src/` (section 5); build a new package (`--destination out\<child>`).
6. The full acceptance (section 10). Frozen `native/tests/reference_*` oracles are never edited to make a test pass.

## 7. TensorRT and cuda-python

VOSR's TensorRT backends need `vosr2_qkv.dll`: build the native layer with the TensorRT and CUDA roots before
packaging (section 4, item 6; README Build). VOSR's modules, engine caches, hosted FP8 set and tests:
[`VOSR2-MAINTAINING.md`](VOSR2-MAINTAINING.md).

1. Pins live in `backend/src/packages/chaiNNer_tensorrt/__init__.py` (`tensorrt` 11.4.0.106, `cuda-python` 13.4.1);
   they are GPU-only and not in the lock.
2. The native binaries (`vosr2_qkv.dll`, `drft_trt_plugins.dll`, the runner's `_chainner_trt.pyd` and
   `chainner_trt_vs.dll`) are built against the pinned SDK, and TensorRT runs a binary only on the version it was
   built against or a later one, so an installed TensorRT older than the pin is replaced at startup (auto update, no
   fallback). A serialized engine loads
   only on the TensorRT version and GPU architecture that built it (`nodes/impl/tensorrt/engine_info.py`): after a
   bump, users rebuild engines with Build Engine (VOSR's engines rebuild themselves once).
3. API changes: update the stubs `stubs/tensorrt/__init__.pyi` and `stubs/cuda/bindings/runtime.pyi` (pyright).
4. CPU tests: `backend/tests/test_tensorrt_tiling.py` (tiling and seams), `native/tests/test_tensorrt_cache.py`
   (sharing and release). GPU runs are manual.

## 8. Package and release

1. Build the native layer, with the TensorRT and CUDA roots so that `vosr2_qkv.dll` and `drft_trt_plugins.dll`
   ship;
   package: `package_port.py --validate-only`, then build or `--refresh`, then a repeat run that must report
   `verified_no_op` (README Package).
2. Bundle checks (section 5, step 3), `make_oracle.py`, the gates of section 10.
3. Release zip from source: `npm ci`, the native build (Forge copies `backend/src` with the binaries), then
   `npm run make`, both under Node 24 LTS (`.nvmrc`, `engines.node`); `forge.config.js` adds the `portable` marker to the zip and deletes `.pyc` files first. Its top
   maps cross-zip's `fs.rmdir(..., { recursive })`, which Node 25 removed, to `fs.rm`: keep it while maker-zip uses
   cross-zip.
4. The `native/README.md` hash table must match the shipped binaries.
5. VOSR's TensorRT plugin `backend/src/nodes/impl/pytorch/vosr2_qkv.dll` must be built (README Build): without it
   `package_port.py` (all runs, `--validate-only` too) and `npm run make` on Windows (Forge's `prePackage`) refuse to
   package, since such a package has no VOSR TensorRT backends. Only for a package meant to lack them, set
   `CHAINNER_C_PACKAGE_WITHOUT_VOSR_PLUGIN=1` (exactly `1`) for that run. Legacy DRFT's plugin library
   `drft_trt_plugins.dll` builds from the same roots in the same configure (a failure of its target fails the
   build), so this check covers it; a package without it has no DRFT TensorRT route.
6. A Windows release with the AMD GPU parts builds with `Build.ps1 -RocmRoot <ROCm 7.14 SDK> -MigraphxRoot <MIGraphX
   2.16 install>` as well (`BUILDING.md`, AMD GPUs; the MIGraphX whose DLLs the published Windows `onnxruntime-migraphx`
   wheel bundles): `nodes\impl\migraphx\_chainner_migraphx.pyd`, `drft_migx_ops.dll` and `vosr2_migx_ops.dll` ship
   when built and are never required (`package_files.py`), and the other binaries keep the SHA-256 of a build without
   the roots at the same path (`drft_trt_plugins.dll` depends on the checkout's path).

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

## 11. Linux

Linux adds these steps to the procedures above. Run them in bash from the repository root; the Linux binaries are `.so`
files with the Windows names (`BUILDING.md`, Linux). `package_port.py` and `independent_ui.py` build the local Windows
package from upstream's Windows nightly (sections 5, 6 and 8) and have no Linux counterpart: the Linux zip is Forge's
(`npm run make-linux-zip`).

1. **Both platforms.** A file that both compile keeps its Windows statements as they are, and its Linux code goes in
   the `#else` branch (`if(NOT WIN32)` in CMake). Build on Windows as well as on Linux for such a change, and run the
   Windows tests that name the file.
2. **Python stack (sections 1 and 2).** `native/python-stack.lock.linux.txt` is the Windows set without `colorama`, plus
   PyTorch's CUDA libraries and Triton, `cuda-bindings`, `uvloop` and `ujson`. Change the install list (`packages`) and
   the torch line in `native/tools/provision_runtime.sh` together with `provision_runtime.ps1`'s, so that both locks
   stay one stack, then `native/tools/provision_runtime.sh --relock` (a CPython upgrade adds
   `--version <x.y.z> --tag <build tag>`) and review the diff. The `Dependency` pins in
   `backend/src/packages/*/__init__.py` serve both platforms; `chaiNNer_tensorrt` alone differs, with `triton` where
   Windows has `triton-windows`. `package_manifest.LOCK`, `verify_runtime.py` and the oracle read the Windows lock
   only. A CPython upgrade also changes the Linux URL in `src/main/python/integratedPython.ts` and the CPython step of
   `native/linux/Dockerfile` (URL and SHA-256). NumPy's Linux wheels dispatch AVX-512 loops its Windows wheels lack,
   and the C mirrors follow the Windows ones, so Linux workers set
   `NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"` unless the user set it (`backend/src/worker_spawn.py`):
   after a NumPy upgrade, check that these names still switch the new wheel's AVX-512 loops off.
3. **Golden twins (section 3).** The manifests of the kernels that call the C library's math (`LIBM_KERNELS` in
   `native/tools/golden_kernels.py`: `lens_power` and `resample`) were recorded from `ucrtbase.dll`. On Linux those
   kernels follow the process's libm, as NumPy and CPython do, so `native/tests/golden/linux/<kernel>.json`, the
   kernel's Linux twin, holds the cases whose outputs differ, and the golden tests read its outputs in place of the
   manifest's (`load_expected`). Windows reads the manifest alone, and a manifest is never edited for Linux. After a
   change to such a kernel or to the math functions it calls, build on Linux, check that the kernel's live comparisons
   with its Python reference pass (the twin records what that reference gives there), then:
   `python native/tools/golden_kernels.py twin --dll backend/src/nodes/impl/chainner_native.so --kernel <kernel>`. It
   refuses when the ISA levels disagree and records the glibc version in the twin. Another glibc can round otherwise:
   a twin that fails on another distribution is a finding to look into, not a file to rewrite.
   Other comparisons against NumPy or OpenCV differ on Linux where their GCC builds do (OpenCV's AVX2 code contracts
   multiply-adds into FMA, NumPy converts NaN and out-of-range floats to integers otherwise);
   `native/tests/linux_differences.json` lists the tests that Linux skips for that reason. To refresh it, record each
   system's comparisons (with `native/tools` on `PYTHONPATH`, as `run_tests.py` sets it), then list:

   ```bash
   LINUX_DIFFERENCES_RECORD=windows.json python -m pytest -p linux_differences @ids.txt   # on Windows
   LINUX_DIFFERENCES_RECORD=linux.json python -m pytest -p linux_differences @ids.txt     # on Linux
   python native/tools/linux_differences.py list windows.json linux.json
   ```

4. **Compiler and flags (section 4).** The flags are in `native/toolchain-linux.cmake` and `native/cmake/linux.cmake`,
   which writes every clang-cl option of the CMake files in clang's spelling at the end of configuration and stops the
   configure on one it does not know: a new Windows option needs a mapping there or an `if(WIN32)` block. LLVM 23.1.2
   is pinned in `native/CMakeLists.txt` and `native/toolchain-linux.cmake`. Keep `-ffp-model=strict`,
   `-fdenormal-fp-math=dynamic`, the `-ffile-prefix-map` and math errno (clang-cl's `/fp:strict` keeps it). `sqrt` and
   `sqrtf` reach `native/src/ucrt_sqrt.c`, compiled outside ThinLTO, which gives the static UCRT's results and flags
   because glibc's differ under DAZ. Check reproducibility by building twice in clean directories and comparing the
   modules' SHA-256. The TensorRT parts follow section 4, item 6 with `.so` names; nvcc's host compiler is `g++`
   (`CHAINNER_C_NVCC_HOST`), since nvcc 13 refuses clang 23.
5. **Image pins (sections 7 and 8).** `native/linux/Dockerfile` pins every download by version and SHA-256
   (`native/linux/fetch.sh <url> <sha256> <file>` fails and removes the file on a mismatch): the base image by digest,
   Ubuntu's packages by the `UBUNTU_SNAPSHOT` date, and LLVM, CMake, Ninja, NASM, CPython with NumPy's hash, Node, the
   CUDA redistributables, the TensorRT packages, VapourSynth's header and the Rocky Linux 8.10 packages of the glibc
   2.28 sysroot (their file names carry their versions). To refresh a pin, change its version and hash together, run
   `native/tools/build_linux_release.sh`, and read `check_glibc.py`'s result. Keep the image equal to the Windows build
   where Windows pins the same tool: LLVM 23.1.2, the CUDA 13 and TensorRT 11 majors (`CMakeLists.txt`) and TensorRT's
   version (`chaiNNer_tensorrt`), CMake, Ninja, NASM 3.01, Node 24 (`.nvmrc`), and the lock's CPython and NumPy. Once
   Rocky 8.10 leaves the main mirror, set the `ROCKY_MIRROR` default to `https://dl.rockylinux.org/vault/rocky/8.10`.
   texconv's sources are the one download outside the image: the configure fetches them into the build directory,
   pinned the same way in `native/texconv/CMakeLists.txt` (item 8).
6. **Release (section 8).** `native/tools/build_linux_release.sh --ref <commit>` builds that commit in the image,
   checks `backend/src` and the packaged app with `check_glibc.py`, makes the zip and prints the SHA-256 of the zip and
   of every module (`native/README.md`'s hash table lists the Windows binaries). A release with the AMD GPU parts adds
   `--rocm <ROCm 7.14 SDK root> --migraphx <MIGraphX 2.16 prefix>` (`BUILDING.md`, Linux); the modules the build had
   without them keep their SHA-256, which a build of the same commit without the two options shows. DRFT's MIGraphX ops,
   `nodes/impl/migraphx/drft_migx_ops.so`, are built whenever the ROCm root is given and ship with them. Forge names the
   zip from `package.json`'s version (`chaiNNer-linux-x64-<version>.zip`): give it the release's name before attaching
   it (section 9, step 4).
7. **Tests (section 10).** On Linux, from a checkout on a Linux file system with the natives built
   (`native/build_linux.sh`), with `PYTHONNOUSERSITE=1`, scratch `APPDATA` and `TMPDIR`, and the pinned Linux FFmpeg
   first on `PATH`: `<runtime>/bin/python3 -B native/tools/run_tests.py --workers 3 --deselect
   native/tests/test_matting_complete.py` (`run_tests.py` sets `NPY_DISABLE_CPU_FEATURES` there). Cases whose Linux
   reference library rounds differently from the Windows build are listed with their reason in
   `native/tests/linux_differences.json`; both conftests skip them on Linux.
8. **texconv (DDS).** Linux runs `native/texconv` where Windows runs `backend/src/texconv/texconv.exe` (DirectXTex's
   release binary): texconv's steps for the options chaiNNer passes, on the same DirectXTex release, with texconv.exe's
   WIC steps and MSVC `/fp:fast` arithmetic rebuilt from measurements (`wic.cpp`, `bc4_unorm_gradient.h`, the
   DirectXTex compile options). `native/tests/test_texconv.py` compares both with `native/tests/golden/texconv.json`,
   texconv.exe's results for chaiNNer's command lines. A texconv.exe update moves the DirectXTex tag and SHA-256 in
   `native/texconv/CMakeLists.txt` to that release (the configure stops when BC4BC5.cpp's patched line moved), then
   re-records the golden file on Windows (`native\.venv\Scripts\python.exe -B native/tools/texconv_golden.py`), runs
   the test on both systems, and lists the cases WIC decides that the rebuild misses in `linux_differences.json`
   (groups `texconv-wic-*`). A new texconv option in `native/src/image_io.cpp` needs its counterpart in `texconv.cpp`,
   which refuses options it does not know.
