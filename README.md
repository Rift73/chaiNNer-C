# chaiNNer-C

chaiNNer-C is [chaiNNer](https://github.com/chaiNNer-org/chaiNNer), the node-based image processing GUI by
[chaiNNer-org](https://github.com/chaiNNer-org) and its contributors, with its CPU image nodes converted to C and C++
for speed. The application, the node graph, the nodes and the UI are upstream's work; chaiNNer-C changes only how the
CPU nodes compute, and their outputs match upstream's. It is an independent fork, not affiliated with or endorsed by
chaiNNer-org. For chaiNNer itself, its documentation and community, see the
[upstream repository](https://github.com/chaiNNer-org/chaiNNer).

chaiNNer-C ships as a portable Windows package built on the upstream nightly `0.25.1-nightly.2025-10-21`, with no
auto-update and the header `chaiNNer v0.3.0`.

## What is converted

- Image, filter, color, resize, blend, convolution, pixel-art, tiling and utility nodes run in a native layer
  (`native/`): C17 kernels in `chainner_native.dll`, a C++20/pybind11 module `_chainner_graph.pyd`, and
  `chainner_ext.pyd`, a C replacement for upstream's Rust `chainner_ext`.
- The Python backend (`backend/src`) is upstream's with the native paths wired in; nodes without one run upstream's
  Python unchanged. The GPU nodes keep their engines (PyTorch, ONNX Runtime, NCNN); the ONNX CPU session adapter, the
  ONNX-to-NCNN converter, NCNN param handling and tiling control are native.
- `src/` and `tests/` are upstream's frontend at `d56e507f` with the package's UI changes ported (no updates, `v0.3.0`,
  the drop repairs), its integrated Python moved to CPython 3.14.8 and its own `%APPDATA%\chaiNNer-C`: the
  UI when running from source ([Build and run](#build-and-run)). The portable package uses the installed nightly's UI.

**Output contract.** The reference is upstream chaiNNer's own backend, run on the same Python stack with the same
inputs, and chaiNNer-C's outputs must be bit-exact against it. The few documented deviations and test tolerances are
in [ARCHITECTURE section 7](native/ARCHITECTURE.md#7-retained-engines-and-exactness-exceptions).

## Requirements

- **Windows x64**, the only platform validated and packaged.
- **Any x86-64 CPU with x86-64-v2 (SSE4.2, POPCNT)**, the baseline NumPy 2.x requires. The native binaries are
  baseline x86-64 code; their AVX2 and AVX-512 kernels are used at runtime when the CPU has them (`native/src/isa.c`).

- **GPU (optional):** PyTorch (CUDA 13.2 build) and ONNX Runtime GPU need an NVIDIA GPU; NCNN runs on Vulkan.

## The tested Python stack

[`native/python-stack.lock.txt`](native/python-stack.lock.txt) records the stack chaiNNer-C is built and tested on: the
interpreter, its archive's SHA-256, pip's version and a full `pip freeze`. Main entries: CPython 3.14.8
(python-build-standalone `20261003`), NumPy 2.5.3, OpenCV 5.0.0.93, Pillow 12.3.0, PyTorch 2.14.1+cu132, ONNX Runtime
(`onnxruntime-gpu`) 1.30.0 and ncnn 1.0.20260526.

The pins are floors, not ceilings: the dependency manager installs a package only when it is missing or older than its
pin. Native kernels that mirror a library's numerics check its exact version
(`backend/src/nodes/impl/native_versions.py`); on any other version they use the library's own code, so results stay
correct and only the speedup is lost.

## Build and run

Prerequisites: LLVM 23.1.2 (`clang-cl`, `lld-link`, `llvm-lib`, `llvm-rc`); Visual Studio 2022 Build Tools with the
MSVC toolset 14.44.35207 and the Windows SDK 10.0.26100.0 for headers and libraries (no Developer environment needed);
CMake 3.20 or newer on `PATH`; Ninja (by default the Build Tools' copy); Node.js with npm. Run every command from the
repository root in PowerShell.

1. **Clone** the `chaiNNer-C` branch: `git clone --branch chaiNNer-C <chaiNNer-C repository URL>`.
2. **Provision the build's Python.** `-BuildOnly` downloads the lock's python-build-standalone CPython 3.14.8 into
   `native\runtime\cpython-3.14.8`, checked against the lock's SHA-256, and installs only the lock's NumPy and
   pybind11 (the build's headers; no PyTorch):

   ```powershell
   powershell -NoProfile -File native\tools\provision_runtime.ps1 -BuildOnly
   ```

3. **Build the native layer** with clang-cl (the binaries land in `backend/src`):

   ```powershell
   & '.\native\Build.ps1' -Configuration Release
   ```

4. **Install the UI's packages and start chaiNNer-C:**

   ```powershell
   npm ci
   npm start
   ```

5. **First start:** the app downloads its integrated Python, python-build-standalone CPython 3.14.8 (the lock's), and
   chaiNNer's dependency manager installs the Python packages. Everything it keeps (the integrated Python, settings,
   logs and backend storage) lives in its own folder, `%APPDATA%\chaiNNer-C`, never in an installed chaiNNer's
   `%APPDATA%\chaiNNer`.

`npm start` runs the app with its own backend. `npm run dev` is upstream's developer mode: it runs the backend with
the `python` on `PATH` (with `debugpy`) as a remote backend, so it neither downloads the integrated Python nor uses
the dependency manager.

## Develop, test and package

The developer and test path provisions the whole tested stack and needs, for packaging, the upstream nightly
[`0.25.1-nightly.2025-10-21`](https://github.com/chaiNNer-org/chaiNNer-nightly/releases/tag/2025-10-21) installed,
whose UI is the shell the package patches.

1. **Provision the Python runtime from the lock.** It downloads the lock's python-build-standalone CPython 3.14.8 into
   `native\runtime\cpython-3.14.8`, checked against the lock's SHA-256 (never extracted over an existing runtime,
   which is used only if it was extracted from that archive),
   installs the lock's pip and exactly its pins (torch and torchvision from the PyTorch CUDA 13.2 index, `chainner-pip`
   from the backend's bundled wheel, the rest from PyPI), and fails unless `pip check` is clean and `pip freeze` equals
   the lock. It never writes the lock; `-Relock` is the deliberate upgrade step (the latest versions, then a new lock
   and the full acceptance).

   ```powershell
   powershell -NoProfile -File native\tools\provision_runtime.ps1
   ```

2. **Create the test and tool environment** (`requirements.txt` adds the lint and type-check tools):

   ```powershell
   native\runtime\cpython-3.14.8\python.exe -m venv --system-site-packages native\.venv
   & '.\native\.venv\Scripts\python.exe' -m pip install -r requirements.txt
   ```

3. **Build the native layer** (`chainner_native.dll` and `_chainner_graph.pyd` into `backend/src/nodes/impl`,
   `chainner_ext.pyd` into `backend/src/chainner_ext`):

   ```powershell
   & '.\native\Build.ps1' -Configuration Release
   ```

4. **Test:**

   ```powershell
   $env:PYTHONDONTWRITEBYTECODE = '1'; $env:CUDA_VISIBLE_DEVICES = '-1'
   $env:NUMBA_CACHE_DIR = "$env:TEMP\chainner-c-numba"
   & '.\native\.venv\Scripts\python.exe' -B -m pytest native/tests backend/tests -q -p no:cacheprovider
   ```

**The toolchain** is the only one, pinned in `native/toolchain.cmake` and `native/CMakeLists.txt`: clang-cl 23.1.2
with lld-link and llvm-lib, baseline x86-64 code with per-file AVX2 and AVX-512 targets for the ISA units, ThinLTO,
no PGO, `/fp:strict`, and `/W4 /WX` for first-party code. `-Define CHAINNER_C_MARCH=<cpu>` (for example `native`
for the building machine's own CPU, or `icelake-server`) adds `-march` to every target, for builds that stay on that
machine, never for releases. The hot kernels already pick AVX2 or AVX-512 at runtime, so the gain is modest; after
such a build, run the tests (step 4) once, since a different code generation is checked there.
Configuration refuses any other clang-cl, MSVC toolset or Windows SDK version. The tool locations are CMake cache
entries, set on a build directory's first configure (for example `-Define 'chainner_c_llvm=<LLVM 23.1.2>/bin'`; also
`chainner_c_vc_tools`, `chainner_c_sdk`, `chainner_c_sdk_version`; defaults in `native/toolchain.cmake`), and
`-Ninja <path to ninja.exe>` selects another Ninja.

Builds are reproducible: `/Brepro` (no timestamps, no PDB in Release) and a source-path map make the binaries
byte-identical wherever the repository is checked out, and `.gitattributes` gives every checkout LF line endings.

## Package

`native/tools/package_port.py` is the local portable-package path: from the full runtime of
[Develop, test and package](#develop-test-and-package) and the installed nightly, it builds the portable package in
`out\chaiNNer-C`:

```powershell
$py = '.\native\.venv\Scripts\python.exe'; $pkg = @('-B', '.\native\tools\package_port.py',
  '--installed-app', "$env:LOCALAPPDATA\chaiNNer\app-0.25.1-nightly2025-10-21",
  '--installed-python', '.\native\runtime\cpython-3.14.8')
& $py @pkg --validate-only   # check the inputs; writes nothing
& $py @pkg                   # build, or refresh an existing package
```

- It copies the installed app's shell and the provisioned runtime, writes `backend/src` byte for byte, and records
  everything in a manifest, `chainner-c-package.json`.
- It accepts only the `0.25.1-nightly.2025-10-21` app, whose two UI bundles are pinned by SHA-256 and get two reviewed
  patches: updates removed, chaiNNer-C's version `v0.3.0` shown, the integrated-Python check accepting CPython 3.14, and
  file and folder drops repaired.
- The runtime's `Lib` is precompiled with hash-based bytecode (`unchecked-hash`), so two builds are byte-identical.
- A `portable` marker keeps the profile inside the package; run `chaiNNer.exe` from its folder.
- A repeat run with unchanged inputs reports `verified_no_op`. `node native\tools\verify_independent_ui.cjs` and the
  other `verify_*.cjs` scripts check the patched bundles.

## Checks and CI

Hosted CI (`.github/workflows/lint-backend.yml`) runs `ruff format --check` and `ruff check` on `ubuntu-latest`, and
pyright and `backend/tests` on `windows-latest` against CPU builds of the lock. The native build, `native/tests`, the
runtime verifiers, the parity checks and the benchmark are local gates, documented in
[`native/README.md`](native/README.md).

## Documentation

- [`native/README.md`](native/README.md): the detailed reference to build, test, package, verify and benchmark.
- [`native/ARCHITECTURE.md`](native/ARCHITECTURE.md): design, exactness rules, deviations and packaging.
- [`native/DESIGN-DECISIONS.md`](native/DESIGN-DECISIONS.md): the key decisions, why they were taken and where they
  live.
- [`native/MAINTAINING.md`](native/MAINTAINING.md): checklists for upgrades, kernel and UI changes, packaging,
  publishing and the local acceptance gates.
- [`native/STATUS.md`](native/STATUS.md): current state, pending work and the decision log.

## License

chaiNNer-C is free software under the GNU General Public License v3.0 ([`LICENSE`](LICENSE)), chaiNNer's license.
chaiNNer is by chaiNNer-org and its contributors.

- The native kernels adapt algorithms from other projects (chaiNNer-rs, OpenCV, Pillow, NumPy, PyMatting and others);
  `backend/src/nodes/impl/chainner_native.LICENSE.txt` lists each adaptation and its license, file by file.
- Vendored and ported third-party code (fpng, pybind11, the ONNX Runtime C header and the sources ported into
  `chainner_ext.pyd`) keeps its license texts under `native/third_party`.
- The runtime is python-build-standalone CPython, and each installed wheel keeps its own license. The UI shell is
  upstream's installed build, unmodified apart from the reviewed patches above.
