# Building, testing and packaging chaiNNer-C

The developer reference that used to live in the root README. The full command reference is
[`README.md`](README.md); design and exactness rules are in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## What is converted

- Image, filter, color, resize, blend, convolution, pixel-art, tiling and utility nodes run in a native layer
  (`native/`): C17 kernels in `chainner_native.dll`, a C++20/pybind11 module `_chainner_graph.pyd`, and
  `chainner_ext.pyd`, a C replacement for upstream's Rust `chainner_ext`.
- The Python backend (`backend/src`) is upstream's with the native paths wired in; nodes without one run upstream's
  Python unchanged. The GPU nodes keep their engines (PyTorch, ONNX Runtime, NCNN, TensorRT); the ONNX CPU session
  adapter, the ONNX-to-NCNN converter, NCNN param handling and tiling control are native. chaiNNer-C adds two PyTorch
  nodes for VOSR 2.0 generative upscaling ([`VOSR2.md`](VOSR2.md)).
- `src/` and `tests/` are upstream's frontend at `d56e507f` with chaiNNer-C's UI changes (no updates, `v0.3.8`, the
  drop repairs, the TensorRT types and Clear item, the VOSR model type and the iterated input condition), its
  integrated Python moved to CPython 3.14.8 and its own `%APPDATA%\chaiNNer-C`. The local portable package uses an
  installed upstream nightly's UI with reviewed patches, which do not include the VOSR changes.

**Output contract.** The reference is upstream chaiNNer's own backend, run on the same Python stack with the same
inputs, and chaiNNer-C's outputs must be bit-exact against it. The documented deviations and test tolerances are in
[ARCHITECTURE section 7](ARCHITECTURE.md#7-retained-engines-and-exactness-exceptions).

## The tested Python stack

[`python-stack.lock.txt`](python-stack.lock.txt) records the stack chaiNNer-C is built and tested on: the interpreter,
its archive's SHA-256, pip's version and a full `pip freeze`. Main entries: CPython 3.14.8 (python-build-standalone
`20261003`), NumPy 2.5.3, OpenCV 5.0.0.93, Pillow 12.3.0, PyTorch 2.14.1+cu132, ONNX Runtime (`onnxruntime-gpu`)
1.30.0 and ncnn 1.0.20260526.

The pins are floors, not ceilings: the dependency manager installs a package only when it is missing or older than its
pin. Native kernels that mirror a library's numerics check its exact version
(`backend/src/nodes/impl/native_versions.py`); on any other version they use the library's own code, so results stay
correct and only the speedup is lost.

## Build and run from source

Prerequisites: LLVM 23.1.2 (`clang-cl`, `lld-link`, `llvm-lib`, `llvm-rc`); Visual Studio 2022 Build Tools with the
MSVC toolset 14.44.35207 and the Windows SDK 10.0.26100.0 (no Developer environment needed); CMake 3.20 or newer on
`PATH`; Ninja (by default the Build Tools' copy); NASM 3.01 on `PATH` (libjpeg-turbo's SIMD; or
`-Define CHAINNER_C_NASM=<nasm.exe>`); Node.js with npm. Optional: a TensorRT 11 SDK and a CUDA 13 toolkit,
which build VOSR's TensorRT plugin `vosr2_qkv.dll` when step 3 adds
`-Define CHAINNER_C_TENSORRT_ROOT=<SDK>,CHAINNER_C_CUDA_ROOT=<toolkit>` (README Build); without it VOSR runs on
PyTorch only. Run every command from the repository root in PowerShell.

1. Clone the `chaiNNer-C` branch.
2. Provision the build's Python: `-BuildOnly` downloads the lock's CPython 3.14.8 into `native\runtime\cpython-3.14.8`,
   checked against the lock's SHA-256, and installs only the lock's NumPy and pybind11 (the build's headers):
   `powershell -NoProfile -File native\tools\provision_runtime.ps1 -BuildOnly`
3. Build the native layer with clang-cl (the binaries land in `backend/src`): `& '.\native\Build.ps1' -Configuration Release`
4. `npm ci`, then `npm start`. On first start the app downloads its integrated Python and the dependency manager
   installs the packages; everything it keeps lives in `%APPDATA%\chaiNNer-C`.

`npm run dev` is upstream's developer mode: it runs the backend with the `python` on `PATH` (with `debugpy`) as a
remote backend, so it neither downloads the integrated Python nor uses the dependency manager.

A release zip is built with `npm run make` after step 3 (see [`MAINTAINING.md`](MAINTAINING.md)).

## Develop and test

1. Provision the whole tested stack from the lock (torch and torchvision from the PyTorch CUDA 13.2 index,
   `chainner-pip` from the backend's bundled wheel, spandrel and `onnxruntime-gpu` from their forks' release wheels,
   the rest from PyPI); it fails unless `pip check` is clean and
   `pip freeze` equals the lock, and never writes the lock (`-Relock` is the deliberate upgrade step):
   `powershell -NoProfile -File native\tools\provision_runtime.ps1`
2. Create the test and tool environment (`requirements.txt` adds the lint and type-check tools):

   ```powershell
   native\runtime\cpython-3.14.8\python.exe -m venv --system-site-packages native\.venv
   & '.\native\.venv\Scripts\python.exe' -m pip install -r requirements.txt
   ```

3. Build: `& '.\native\Build.ps1' -Configuration Release`
4. Test:

   ```powershell
   $env:PYTHONDONTWRITEBYTECODE = '1'; $env:CUDA_VISIBLE_DEVICES = '-1'
   $env:NUMBA_CACHE_DIR = "$env:TEMP\chainner-c-numba"
   & '.\native\.venv\Scripts\python.exe' -B -m pytest native/tests backend/tests -q -p no:cacheprovider
   ```

**The toolchain** is pinned in `native/toolchain.cmake` and `native/CMakeLists.txt`: clang-cl 23.1.2 with lld-link and
llvm-lib, baseline x86-64 code with per-file AVX2 and AVX-512 targets for the ISA units, ThinLTO, no PGO,
`/fp:strict`, and `/W4 /WX` for first-party code. Configuration refuses any other clang-cl, MSVC toolset or Windows
SDK version. The tool locations are CMake cache entries set on a build directory's first configure (for example
`-Define 'chainner_c_llvm=<LLVM 23.1.2>/bin'`; also `chainner_c_vc_tools`, `chainner_c_sdk`,
`chainner_c_sdk_version`), and `-Ninja <path to ninja.exe>` selects another Ninja. The image codecs
(`native/cmake/codecs.cmake`: libjpeg-turbo, zlib, libpng, libwebp, libtiff at today's Pillow and OpenCV versions)
build from vendored sources in `native/third_party` with the same options; NASM 3.01 assembles libjpeg-turbo's SIMD.

`-Define CHAINNER_C_MARCH=<cpu>` (for example `native` for the building machine's own CPU, or `icelake-server`) adds
`-march` to every target, for builds that stay on that machine, never for releases. The hot kernels already pick AVX2
or AVX-512 at runtime, so the gain is modest; run the tests once after such a build.

Builds are reproducible: `/Brepro` (no timestamps, no PDB in Release) and a source-path map make the binaries
byte-identical wherever the repository is checked out, and `.gitattributes` gives every checkout LF line endings.

**AMD GPUs** (optional; `native/cmake/rocm.cmake`): `-RocmRoot <ROCm 7.14 SDK root>` (the `rocm-sdk-devel` wheel's
expanded `_rocm_sdk_devel`, with `lib\amdhip64.lib` and `lib\llvm\bin\clang++.exe`) builds `drft_migx_ops.dll` and
`vosr2_migx_ops.dll` in `nodes\impl\migraphx`, device code for ROCm 7.14's 22 Windows targets by the SDK's `clang++`
against the pinned MSVC toolset and Windows SDK; with `-MigraphxRoot <MIGraphX 2.16 install>` (the very MIGraphX whose
DLLs chaiNNer-C's Windows `onnxruntime-migraphx` wheel bundles; recipe in `native\tools\migraphx\windows`, below) also
the runner `_chainner_migraphx.pyd`, which links MIGraphX's import libraries and, unlike the other modules, the DLL C
runtime (`/MD`), MIGraphX's own. DUAL's op library is Linux only (its kernels come from Triton). Without either root
the configuration has no AMD cache entry and every other binary is the same. The runner's tests on Windows
(`backend/tests/test_migraphx_runner.py`, `test_migraphx_windows.py`) need `onnxruntime-migraphx` and `rocm-sdk-core`,
`rocm-sdk-libraries` installed, with the build directory's `migraphx_runner\chainner_example_migx_ops.dll` in
`CHAINNER_C_MIGRAPHX_TEST_OPS`.

## Local portable package

`native/tools/package_port.py` builds a portable package in `out\chaiNNer-C` from the full runtime above and the
upstream nightly
[`0.25.1-nightly.2025-10-21`](https://github.com/chaiNNer-org/chaiNNer-nightly/releases/tag/2025-10-21), whose
official Windows zip is extracted to `native\local\upstream\app-0.25.1-nightly2025-10-21` (git-ignored; the commands
are in [`README.md`](README.md), Prerequisites):

```powershell
$py = '.\native\.venv\Scripts\python.exe'; $pkg = @('-B', '.\native\tools\package_port.py',
  '--installed-python', '.\native\runtime\cpython-3.14.8')
& $py @pkg --validate-only   # check the inputs; writes nothing
& $py @pkg                   # build, or refresh an existing package
```

- It copies the installed app's shell and the provisioned runtime, writes `backend/src` byte for byte, and records
  everything in a manifest, `chainner-c-package.json`.
- It accepts only that nightly, whose UI bundles are pinned by SHA-256 and get reviewed patches (updates removed, the
  `v0.3.8` label, CPython 3.14 accepted, drop repairs, the TensorRT types and Clear item).
  `node native\tools\verify_independent_ui.cjs --package <dir> --installed-app <dir>` checks them.
- The runtime's `Lib` is precompiled with hash-based bytecode (`unchecked-hash`), so two builds are byte-identical.
- A `portable` marker keeps the profile inside the package; run `chaiNNer.exe` from its folder.

## Linux

x86-64 only. The same sources build into `.so` modules with the Windows modules' names and directories under
`backend/src`: `chainner_native.so`, `_chainner_graph.so`, `chainner_ext/chainner_ext.so` and, with the TensorRT parts,
`nodes/impl/pytorch/vosr2_qkv.so`, `nodes/impl/tensorrt/_chainner_trt.so`, `chainner_trt_vs.so` and
`drft_trt_plugins.so`, and, for AMD GPUs, `nodes/impl/migraphx/_chainner_migraphx.so` with the ROCm SDK and
MIGraphX and the custom-op libraries beside it with the ROCm SDK alone: `dual_migx_ops.so`, `vosr2_migx_ops.so` and
`drft_migx_ops.so`, which is built whenever the ROCm root is given and exports `chainner_migx_ops`, the `drftHip*`
kernel launchers its tests call and the compiler's `__hip_cuid_*` symbol. Linux code sits in `#else` branches and
`if(NOT WIN32)` blocks, so the Windows build is the one above, unchanged. Run every command from the repository root in
bash.

**Develop.** Prerequisites: LLVM 23.1.2's official Linux release (`clang`, `clang++`, `ld.lld`, `llvm-ar`,
`llvm-ranlib`); its binaries need glibc 2.34 or newer, and `ld.lld` needs ICU 70 (`libicuuc.so.70`, `libicui18n.so.70`,
`libicudata.so.70`), so on a distribution with another ICU put Ubuntu 22.04's `libicu70` libraries in the release's
`lib` folder, which `ld.lld`'s run path covers; NASM 3.01 on `PATH` (or `-D CHAINNER_C_NASM=<nasm>`; distributions ship
older ones, so build it from nasm.us's 3.01 source release); CMake 3.20 or newer; Ninja; Node.js with npm. Optional, for
the TensorRT parts: CUDA 13.4.92 (`nvcc`, whose host compiler is GCC's `g++` 16 or older, since nvcc 13 refuses clang
23; `-D CHAINNER_C_NVCC_HOST=<g++>` if it is not the `g++` on `PATH`), TensorRT 11.4.0.106 (its headers and
`libnvinfer.so.11`, the one library the modules link; the app loads the `tensorrt` package's) and, for the runner's
VapourSynth plugin, VapourSynth R81's `VapourSynth4.h` (`-D CHAINNER_C_VAPOURSYNTH_INCLUDE=<its folder>`).

1. Provision the build's Python: `native/tools/provision_runtime.sh --build-only` downloads the Linux lock's CPython
   3.14.8 (python-build-standalone `20261003`) into `native/runtime/cpython-3.14.8`, checked against the lock's SHA-256,
   and installs only the lock's NumPy and pybind11.
2. Build with `native/build_linux.sh` (`Build.ps1`'s counterpart; the binaries land in `backend/src`). Options: `-c
   Release|Debug`, `-B <build directory>` (default `native/build-linux`), `-D NAME=VALUE`, `-j <jobs>`, `--ninja
   <ninja>`. LLVM is the cache entry `chainner_c_llvm` (default `~/chainner-linux/toolchains/llvm-23.1.2/bin`), set
   with `-D` on a build directory's first configure. The TensorRT parts are built when `CHAINNER_C_TENSORRT_ROOT=<SDK>`
   and `CHAINNER_C_CUDA_ROOT=<toolkit>` are given together, and skipped when neither is:

   ```bash
   native/tools/provision_runtime.sh --build-only
   native/build_linux.sh -j 12 -D chainner_c_llvm=<LLVM 23.1.2>/bin \
     -D CHAINNER_C_TENSORRT_ROOT=<TensorRT 11.4.0.106> -D CHAINNER_C_CUDA_ROOT=<CUDA 13.4.92>
   ```

   The AMD GPU parts (`native/cmake/rocm.cmake`; Windows: Build.ps1's `-RocmRoot`, above) are built when `CHAINNER_C_ROCM_ROOT=<ROCm 7.14 SDK root>`
   is given (the `rocm-sdk-devel` wheel's expanded `_rocm_sdk_devel`): the HIP custom-op libraries of `src/<model>_hip`
   (`chainner_c_hip_module`, device code for all 25 ROCm 7.14 targets by the SDK's `clang++`); with
   `CHAINNER_C_MIGRAPHX_ROOT=<MIGraphX 2.16 install>` also the MIGraphX runner `nodes/impl/migraphx/_chainner_migraphx.so`
   (`native/migraphx_runner`), which links MIGraphX's and HIP's libraries by soname and, unlike the other modules,
   libstdc++ and libgcc shared, as MIGraphX's exceptions unwind through it. Its tests:
   `backend/tests/test_migraphx_runner.py`, with the build's `migraphx_runner/chainner_example_migx_ops.so` in
   `CHAINNER_C_MIGRAPHX_TEST_OPS`.

   Every Linux build also makes `backend/src/texconv/texconv`, the DDS nodes' converter there (`native/texconv`), and
   its notices `texconv.LICENSE.txt`. Its sources, DirectXTex may2026, DirectXMath apr2025, DirectX-Headers v1.619.5
   and the .NET runtime 9.0.2's `sal.h`, are fetched at the first configure into `<build>/texconv/sources`, each
   checked against its SHA-256 (`native/texconv/CMakeLists.txt`), so that configure needs the network once.

3. The tested stack: `native/tools/provision_runtime.sh` (no option) provisions it from
   [`python-stack.lock.linux.txt`](python-stack.lock.linux.txt), the Windows stack plus what Linux adds (PyTorch's CUDA
   libraries and Triton, `cuda-bindings`, Sanic's `uvloop` and `ujson`). As on Windows it installs the pins as
   they stand (PyTorch and torchvision from the PyTorch CUDA 13.2 index, `chainner-pip` from the backend's bundled
   wheel, spandrel from its fork's release wheel, the rest from PyPI), fails unless `pip check` is clean and `pip
   freeze` equals the lock, and never writes the lock (`--relock` is the upgrade step). `--with-extras` also installs
   what the dependency manager installs on demand (TensorRT, CUDA Python, VapourSynth with its plugins) and the test and
   lint tools; `--runtime-dir <directory>` extracts the runtime elsewhere, for a checkout on a file system that cannot
   hold it. Tests: [`MAINTAINING.md`](MAINTAINING.md), section 11.

**The toolchain** is pinned in `native/toolchain-linux.cmake` and `native/cmake/linux.cmake`: clang 23.1.2 with `ld.lld`
and `llvm-ar` (configuration refuses any other clang), baseline x86-64 code with per-file AVX2 and AVX-512 targets for
the ISA units, ThinLTO, `-ffp-model=strict`, dynamic denormals, and `-Wall -Wextra -Werror` for first-party code. These
are the Windows options in clang's spelling: `native/cmake/linux.cmake` rewrites them at the end of configuration and
stops on one it does not know. `libstdc++` and `libgcc` link statically, so each module needs only glibc (the MIGraphX
runner aside, below); run paths are `$ORIGIN`-relative. The image codecs build from the same vendored sources
(libjpeg-turbo's SIMD assembles as ELF64, libtiff uses `tif_unix.c`). ELF objects and `ld.lld` carry no timestamps and
the source-path map keeps build paths out of the binaries, so builds are reproducible. `-D CHAINNER_C_MARCH=<cpu>` works
as on Windows: never for releases.

**Package from a development build.** After the native build, `npm ci` and `npm run make-linux-zip` (Node 24, Info-ZIP's
`zip` on `PATH`, on a Linux file system: a Windows drive mounted into WSL gives every file mode 777). A zip is made on
Linux, because the `portable` marker follows the build host. Forge's `prePackage` refuses a package without
`chainner_native.so`, `_chainner_graph.so`, `chainner_ext.so` or VOSR's `vosr2_qkv.so` (set
`CHAINNER_C_PACKAGE_WITHOUT_VOSR_PLUGIN=1` to waive the last, as on Windows), `postPackage` removes `texconv.exe` (a
Windows program; the Linux `texconv` beside it stays), and `postMake` adds the `portable` marker and the launcher `chaiNNer.sh` (`src/public/linux/chaiNNer.sh`).
The result is `out/make/zip/linux/x64/chaiNNer-linux-x64-<version>.zip`, holding `chaiNNer-linux-x64/` with `chainner`
(Electron), `chaiNNer.sh`, `portable` and `resources/src`.

**Release build (glibc 2.28 or newer).** `native/tools/build_linux_release.sh [--repo <git repository>] [--ref <commit>]
[--work <directory>] [--rocm <ROCm SDK> --migraphx <MIGraphX>] [--no-package]` on a Linux machine with Docker builds the
image `cnl-build` (`native/linux/Dockerfile`), then, in a container of it (`--memory=16g`), the native binaries with the
TensorRT and CUDA parts, the zip, and the glibc check. It reads one commit of the repository (`git archive`), never a
working tree's changes or ignored files, so no Windows `.dll` or `.pyd` can reach the zip. Binaries and zip are in
`<work>/src` (default work directory `~/chainner-linux/build/release`; the zip in `out/make/zip/linux/x64/`), and the
script prints the SHA-256 of the zip and of every module. The first run downloads about 3.3 GB for the image.

- LLVM 23.1.2's binaries need glibc 2.34, so the image is Ubuntu 22.04, and clang compiles and links against a glibc
  2.28 sysroot instead of the image's own glibc: Rocky Linux 8.10's glibc, kernel headers, and GCC Toolset 14's
  `libstdc++` and `libgcc` (`clang.cfg` and `clang++.cfg` pass `--sysroot` and `--gcc-install-dir` on every call).
  nvcc's host compiler is the same toolset's `g++`.
- The image also holds CMake 3.31.6, Ninja 1.12.1, NASM 3.01, CUDA 13.4.92 (`nvcc` and the static runtime), the
  TensorRT 11.4.0.106 headers with a link stub of `libnvinfer.so.11`, VapourSynth R81's header, CPython 3.14.8 with
  NumPy 2.5.3, and Node 24.22.1. Every download is pinned by SHA-256 (`native/linux/fetch.sh`), the base image by
  digest, and Ubuntu's packages by a snapshot date.
- `native/tools/check_glibc.py [--floor 2.28] PATH... [--third-party PATH...]` reads the ELF files' dynamic sections
  and fails a file of PATH that needs a `GLIBC_` version above the floor or an unnumbered one (`GLIBC_PRIVATE`,
  `GLIBC_ABI_DT_RELR`), or that needs `libstdc++`, `libgcc_s` or `libpython`; the MIGraphX runner
  (`_chainner_migraphx.so`), which links `libstdc++` and `libgcc` shared as MIGraphX does so that MIGraphX's exceptions
  unwind through it, may need both, with versions no newer than the sysroot's GCC 8 runtime defines (`GLIBCXX_3.4.25`,
  `CXXABI_1.3.11`, `GCC_7.0.0`). The release script runs it over `backend/src` and over the packaged app; the Electron
  runtime's files (`--third-party`) are listed, never failed.
- `--rocm` and `--migraphx`, given together, add the AMD GPU parts (`native/cmake/rocm.cmake`): `--rocm` is the ROCm
  7.14 SDK root (`include/hip`, `lib/libamdhip64.so`, `lib/llvm/bin/clang++`; the `_rocm_sdk_devel` directory that
  `rocm-sdk init` makes from AMD's `rocm-sdk-core`, `-libraries` and `-devel` wheels), `--migraphx` MIGraphX 2.16's
  install prefix (AMD ships none for ROCm 7.14; `native/tools/migraphx` builds it, below). The runner uses MIGraphX's
  C++ API, so the prefix must be the very MIGraphX build whose libraries chaiNNer-C's `onnxruntime-migraphx` wheel
  bundles. The
  container mounts both read-only at `/opt/rocm` and `/opt/migraphx`, and the SDK's `clang++` compiles the HIP code for
  every architecture ROCm 7.14 ships with the image's `clang++.cfg`, so its host code uses the same sysroot. The zip
  carries neither tree: the modules need `libamdhip64.so.7` and MIGraphX's libraries by their sonames and have no run
  path, and the MIGraphX package loads those libraries from the installed `onnxruntime-migraphx` and `rocm-sdk`
  packages before the modules (`nodes/impl/migraphx/runner.py`), as the TensorRT parts' `libnvinfer.so.11` comes from
  the `tensorrt` package. The AMD kit's example custom-op library (`native/AMD.md`), which the zip does not carry
  either, is built with them and stays in `<work>/build/migraphx_runner/chainner_example_migx_ops.so`, checked by
  `check_glibc.py` and hashed with the modules.

**MIGraphX and its ONNX Runtime wheel (AMD GPUs).** AMD ships no MIGraphX for ROCm 7.14, so
`native/tools/migraphx/build.sh --rocm-wheels <directory> [--work <directory>] [--jobs <n>] [step...]` builds it from
source on a Linux machine with Docker, in containers of the image `cnm-build` (`native/tools/migraphx/Dockerfile`:
PyPA's manylinux_2_28, glibc 2.28, GCC Toolset 14), together with the ONNX Runtime 1.30.0 wheel with the MIGraphX
provider that the dependency manager installs (`onnxruntime-migraphx` `1.30.0+rocm7.14`, CPython 3.14). Its steps, each
a script beside it, run in this order, and a finished one is kept for the next run:

- `image`, then `sources`: ONNX Runtime's v1.30.0 tag and AMDMIGraphX 2.16.1 (`release/rocm-rel-7.14` at `2487b688`),
  each checked by its commit.
- `sdk`: AMD's ROCm 7.14.1 `rocm`, `rocm-sdk-core`, `-libraries` and `-devel` packages from `--rocm-wheels`
  (repo.amd.com/rocm/whl-multi-arch), checked against the SHA-256 in `setup_sdk.sh`, since AMD publishes none.
- `deps`: MIGraphX's pinned dependencies (rocMLIR, composable_kernel, protobuf and the rest), each archive checked by
  its SHA-256; `migraphx`: device code for the 25 Linux targets, every library linked with `-Bsymbolic-functions` and
  `--exclude-libs,ALL`, installed in `<work>/migraphx-7.14`.
- `ort`: ONNX Runtime with `ort-migraphx-cache-name.patch`, then `ort-migraphx-shape-programs.patch`; `wheel`:
  `finalize_wheel.py` gives it the `+rocm7.14` version and MIGraphX's libraries in `onnxruntime/capi/` with a run path
  to the ROCm pip packages and their notices, and checks every ELF file against glibc 2.28 (the wheel: `<work>/dist`).

A run needs about 21 GB of memory at its peak. **The release build's `--migraphx` is that run's `<work>/migraphx-7.14`,
the run that made the published wheel**: the runner links MIGraphX's C++ API, and a new run is a new MIGraphX build,
which needs a new wheel as well.

**The same on Windows.** AMD publishes no MIGraphX for Windows either. `native\tools\migraphx\windows` builds MIGraphX
2.16.1 (device code for ROCm 7.14.1's 22 Windows targets, gfx908 to gfx1201; no composable_kernel, as in AMD's own
Windows build) and the ONNX Runtime 1.30.0 wheel with the DirectML and MIGraphX providers that the dependency manager
installs on Windows (`onnxruntime-migraphx` `1.30.0+rocm7.14.dml`, CPython 3.14, win_amd64), with the Visual Studio 2022
Build Tools' MSVC 14.44 and the ROCm SDK's AMD clang; no Docker. `settings_win.cmd` holds the settings, each an
environment variable that is set before a script runs to replace its default: `MIGX_WORK` (the work directory, default
`%USERPROFILE%\chainner-migraphx`, no parentheses in the path), `MIGX_JOBS` (12), `MIGX_VS` (the Build Tools),
`MIGX_ROCM` (required: the ROCm 7.14.1 SDK root, the `_rocm_sdk_devel` folder of the expanded `rocm-sdk-devel` wheel) and
`MIGX_PYTHON` (a CPython 3.14 with `wheel`, default `%MIGX_WORK%\venv-build\Scripts\python.exe`, a venv made with
`python -m venv`; ONNX Runtime's build installs the rest of its requirements into it).

What the Linux `sources` and `sdk` steps do is prepared by hand here, with Git on `PATH`: the SDK is AMD's `rocm`,
`rocm-sdk-core`, `-libraries` and `-devel` 7.14.1 packages installed with `pip install --target <sdk>\site` and expanded
by `rocm-sdk init` (no device wheels); the work directory holds `src\AMDMIGraphX` (the commit `fetch_sources.sh`
names), `ort-src` (ONNX Runtime's v1.30.0 tag with its submodules) and, unpacked as `deps-src\<name>\`, the archives
`build_deps.sh` downloads (same URLs, same SHA-256) for rocm-cmake, abseil, protobuf, json, msgpack, eigen, sqlite and
rocmlir. The steps, each started with `windows\run_low.cmd <script> <log>` (below-normal priority, output to the log,
whose folder must exist), run in this order, and a finished dependency or rocMLIR is kept for the next run:

- `build_deps_win.cmd`: MIGraphX's pinned dependencies as AMD's Windows CI builds them (clang-cl, `/std:c++17`, DLL C
  runtime, static): rocm-cmake, abseil, protobuf, nlohmann/json, msgpack, Eigen and the SQLite amalgamation, into
  `<work>\deps`; `build_rocmlir_win.cmd`: rocMLIR 518955ca as the static fat rockCompiler library, into the same prefix.
- `build_migraphx_win.cmd`: MIGraphX with the SDK's `clang++` against rocMLIR, MIOpen, rocBLAS and hipBLASLt (hipRTC
  compiles the kernels at run time, as on Linux), installed in `<work>\install` (`bin`, `lib`, `include`,
  `lib\cmake\migraphx`), the release build's `-MigraphxRoot`.
- `build_ort_win.cmd`: ONNX Runtime from the v1.30.0 tag with `ort-migraphx-cache-name.patch`, then
  `ort-migraphx-shape-programs.patch` (the two Linux patches) and `ort-migraphx-windows-build.patch` (build system only:
  the MIGraphX provider compiles as C++17 on MSVC, since MIGraphX's C++ header does not compile as C++20, and the HIP
  SDK DLL copy is skipped when ROCm 7.14's DLL names match none), re-applied on every run; `build.py --use_dml
  --use_migraphx --enable_lto` in Release, the wheel in `<work>\ort-build\Release\dist`.
- `finalize_wheel_win.cmd` (`finalize_wheel_win.py`): the `+rocm7.14.dml` version, MIGraphX's DLLs (with
  `migraphx_ref.dll`) and the Visual C++ 14.44 runtime in `onnxruntime\capi`, the third-party notices, and a check that
  every DLL import is in the wheel, in Windows or in the ROCm pip packages (the wheel: `<work>\dist`, then its SHA-256).

ROCm itself is not in the wheel. The published wheel is
`onnxruntime_migraphx-1.30.0+rocm7.14.dml-cp314-cp314-win_amd64.whl` (77,744,295 bytes), SHA-256
`9a8c2d1fddfd9a00e429d21fe8f50a32496d0dfe797db1675d3ef299782b870c`. A new run is a new build with a new hash (even
finalizing the same build twice differs, in the zip timestamps only). **The release build's `-MigraphxRoot` is the
`<work>\install` of the run that made the published wheel**, for the reason given for Linux.

## Checks and CI

Hosted CI (`.github/workflows/lint-backend.yml`) runs `ruff format --check` and `ruff check` on `ubuntu-latest`, and
pyright and `backend/tests` on `windows-latest` against CPU builds of the lock. The native build, `native/tests`, the
runtime verifiers, the parity checks and the benchmark are local gates, documented in [`README.md`](README.md).
