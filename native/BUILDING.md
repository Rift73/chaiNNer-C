# Building, testing and packaging chaiNNer-C

The developer reference that used to live in the root README. The full command reference is
[`README.md`](README.md); design and exactness rules are in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## What is converted

- Image, filter, color, resize, blend, convolution, pixel-art, tiling and utility nodes run in a native layer
  (`native/`): C17 kernels in `chainner_native.dll`, a C++20/pybind11 module `_chainner_graph.pyd`, and
  `chainner_ext.pyd`, a C replacement for upstream's Rust `chainner_ext`.
- The Python backend (`backend/src`) is upstream's with the native paths wired in; nodes without one run upstream's
  Python unchanged. The GPU nodes keep their engines (PyTorch, ONNX Runtime, NCNN, TensorRT); the ONNX CPU session
  adapter, the ONNX-to-NCNN converter, NCNN param handling and tiling control are native.
- `src/` and `tests/` are upstream's frontend at `d56e507f` with chaiNNer-C's UI changes (no updates, `v0.3.2`, the
  drop repairs, the TensorRT types and Clear item), its integrated Python moved to CPython 3.14.8 and its own
  `%APPDATA%\chaiNNer-C`. The local portable package uses an installed upstream nightly's UI with reviewed patches.

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
`PATH`; Ninja (by default the Build Tools' copy); Node.js with npm. Run every command from the repository root in
PowerShell.

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
   `chainner-pip` from the backend's bundled wheel, the rest from PyPI); it fails unless `pip check` is clean and
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
`chainner_c_sdk_version`), and `-Ninja <path to ninja.exe>` selects another Ninja.

`-Define CHAINNER_C_MARCH=<cpu>` (for example `native` for the building machine's own CPU, or `icelake-server`) adds
`-march` to every target, for builds that stay on that machine, never for releases. The hot kernels already pick AVX2
or AVX-512 at runtime, so the gain is modest; run the tests once after such a build.

Builds are reproducible: `/Brepro` (no timestamps, no PDB in Release) and a source-path map make the binaries
byte-identical wherever the repository is checked out, and `.gitattributes` gives every checkout LF line endings.

## Local portable package

`native/tools/package_port.py` builds a portable package in `out\chaiNNer-C` from the full runtime above and an
installed upstream nightly
[`0.25.1-nightly.2025-10-21`](https://github.com/chaiNNer-org/chaiNNer-nightly/releases/tag/2025-10-21):

```powershell
$py = '.\native\.venv\Scripts\python.exe'; $pkg = @('-B', '.\native\tools\package_port.py',
  '--installed-app', "$env:LOCALAPPDATA\chaiNNer\app-0.25.1-nightly2025-10-21",
  '--installed-python', '.\native\runtime\cpython-3.14.8')
& $py @pkg --validate-only   # check the inputs; writes nothing
& $py @pkg                   # build, or refresh an existing package
```

- It copies the installed app's shell and the provisioned runtime, writes `backend/src` byte for byte, and records
  everything in a manifest, `chainner-c-package.json`.
- It accepts only that nightly, whose UI bundles are pinned by SHA-256 and get reviewed patches (updates removed, the
  `v0.3.2` label, CPython 3.14 accepted, drop repairs, the TensorRT types and Clear item).
  `node native\tools\verify_independent_ui.cjs --package <dir> --installed-app <dir>` checks them.
- The runtime's `Lib` is precompiled with hash-based bytecode (`unchecked-hash`), so two builds are byte-identical.
- A `portable` marker keeps the profile inside the package; run `chaiNNer.exe` from its folder.

## Checks and CI

Hosted CI (`.github/workflows/lint-backend.yml`) runs `ruff format --check` and `ruff check` on `ubuntu-latest`, and
pyright and `backend/tests` on `windows-latest` against CPU builds of the lock. The native build, `native/tests`, the
runtime verifiers, the parity checks and the benchmark are local gates, documented in [`README.md`](README.md).
