# chaiNNer-C: build, test, package, bench

chaiNNer-C v0.3.2, based on chaiNNer 0.25.1-nightly.2025-10-21, is an independent portable edition of chaiNNer: the
installed nightly `0.25.1-nightly.2025-10-21` UI with pinned bundle patches, the single backend in `backend/src`, and a
native C17/C++20 layer. The UI shows v0.3.2 (header, About, startup log, system information with `app.upstream`), while
`app.getVersion()` stays upstream's, so saved chains keep the upstream version stamp and upstream's migrations read
them correctly. Design: `ARCHITECTURE.md`. State and
decisions: `STATUS.md`. Parked GPU work: `GPU-DEFERRED.md`. Run every command in PowerShell from
`<repository root>`.

## Layout

| Path | Role |
| --- | --- |
| `backend/src/` | The only backend; ships byte for byte (stored verbatim as LF, upstream's line ending: `backend/src/** -text`); its `nodes/impl/chainner_native.dll` and `_chainner_graph.pyd` are git-ignored binaries written by the build |
| `native/src/`, `native/include/`, `native/third_party/` | C17 kernels, C++20 modules; vendored pybind11, fpng 1.0.6, ONNX Runtime C header |
| `native/tests/` | pytest suites and frozen `reference_*` oracles (stored verbatim: `native/tests/reference_*/** -text`) |
| `native/tools/` | Packager, UI patches, runtime verifiers, benchmark harness, code generators, `isa_check.py text` (compares two binaries' code sections), `golden_kernels.py` (golden manifests of the SP4 kernels; usage in each docstring) |
| `native/reports/`, `out/chaiNNer-C/` | Ignored: local evidence (never needed to build or package), and the portable package |
| `out/chaiNNer-C-py311/` | Ignored: the last CPython 3.11 package, kept read-only as the old-stack bench side and the undo; deleting it is owner-only |
| `src/`, `tests/` | Upstream frontend at `d56e507f` with the package's two UI patches ported, CPython 3.14.8 as its integrated Python and `%APPDATA%\chaiNNer-C` as its root; the run-from-source UI (`npm start`), never shipped |

## Prerequisites

- CPU: any x86-64 CPU with x86-64-v2 (SSE4.2, POPCNT), the baseline NumPy 2.x requires; the native binaries are
  baseline x86-64 code and use their AVX2 and AVX-512 kernels when the CPU has them (`src/isa.c`).
- LLVM 23.1.2 (`clang-cl`, `lld-link`, `llvm-lib`, `llvm-rc`) at
  `C:\Executables\clang+llvm-23.1.2-x86_64-pc-windows-msvc\bin`; Visual Studio 2022 Build Tools at
  `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools` (MSVC 14.44.35207's headers and libraries,
  and its bundled Ninja), Windows SDK 10.0.26100.0, and CMake 3.20 or newer.
- CPython 3.14.8 at `native\runtime\cpython-3.14.8`, provisioned by `native\tools\provision_runtime.ps1`
  from the tracked lock `native\python-stack.lock.txt` (interpreter, build tag, archive SHA-256, pip's version
  and every pin, installed without dependency resolution); the run fails unless `pip check` is clean and the
  freeze (`native\runtime\freeze-3.14.8.txt`) equals the lock line for line, and never writes the lock.
  The archive's SHA-256 is recorded beside the runtime it was extracted into
  (`native\runtime\cpython-3.14.8.archive-sha256`); an existing runtime with no record, or another archive's, is
  refused (remove it to extract the lock's). `-Relock` is the deliberate upgrade (the latest versions, a new lock,
  then the full acceptance). The runtime is CMake's `CHAINNER_C_PYTHON_ROOT`, the source of the packaged
  runtime and the oracle's interpreter. Its interpreter reports the headers, the import library
  (`libs\python314.lib`) and NumPy's headers (`np.get_include()`). It holds `chainner-pip` (from the
  backend's bundled wheel), `google-re2` and `Sanic-Cors` (upstream's ONNX loader and server import them;
  chaiNNer-C does not) and `pynvml` (upstream's server declares it; chaiNNer-C declares `nvidia-ml-py`,
  which provides the `pynvml` module).
- Installed app `$env:LOCALAPPDATA\chaiNNer\app-0.25.1-nightly2025-10-21` (read-only).
- `native\.venv`: the test and tool interpreter, made with
  `native\runtime\cpython-3.14.8\python.exe -m venv --system-site-packages native\.venv`; the runtime
  carries NumPy 2.5.3, OpenCV 5.0.0, pytest, pytest-asyncio and pytest-cov. `native\.venv-py311` keeps
  the old stack (Python 3.11.5, NumPy 1.24.4, OpenCV 4.8.0). Recreate a venv only if it is missing,
  and ask before installing anything.
- Node `C:\nvm4w\nodejs\node.exe` for the bundle verifiers. FFmpeg 8.1 shared build in
  `C:\Executables\ffmpeg-8.1-full_build-shared\bin` (hashes pinned in `native/tools/bench_oracle.py`).

## Build

```powershell
& '.\native\Build.ps1' -Configuration Release   # or Debug
```

`Build.ps1` configures `native/build` (`-BuildDirectory` for another) with Ninja (`-Ninja <ninja.exe>`; default the
Build Tools' copy, which must exist) and `native/toolchain.cmake`, which pins the one toolchain (owner directive
2026-10-06): clang-cl 23.1.2 with lld-link and llvm-lib, MSVC toolset 14.44.35207 and Windows SDK 10.0.26100.0, no
Developer environment. Configuration refuses any other version of the three (`CMakeLists.txt` for clang-cl,
`toolchain.cmake` for the toolset directory's version and the SDK version), and the package manifest records them
(`native_toolchain`). It builds `chainner_native.dll` (C17, C ABI version 2), `_chainner_graph.pyd` (C++20/pybind11, linked
against the DLL, fpng and the runtime's `python314.lib`) and `chainner_ext.pyd`. Every target: baseline x86-64 code
(the ISA units for their level only, exactly what `src/isa.c` checks: `*_avx2.c` AVX2, FMA, BMI1 and BMI2,
`*_avx512.c` those plus AVX-512 F/DQ/CD/BW/VL; fpng `-msse4.1 -mpclmul` behind its own CPU check), ThinLTO, `/fp:strict`, dynamic denormals, `/Brepro` (the same sources give the same binaries; no PDB in Release) and
the static runtime; first-party targets `/W4 /WX` (the standing strict check; no suppressions), `/EHsc` for C++;
fpng keeps its vendor `/W3`. `-Define NAME=VALUE` passes CMake entries: the runtime
(`CHAINNER_C_PYTHON_ROOT=...`, on a directory's first configure), the toolchain's four roots for other install
locations (`chainner_c_llvm`, `chainner_c_vc_tools`, `chainner_c_sdk`, `chainner_c_sdk_version`; defaults are the
paths above, in `toolchain.cmake`; also on a directory's first configure), `CHAINNER_C_PDB=ON` (PDBs and linker maps; it must
set `CHAINNER_C_OUTPUT_DIR` outside `backend/src`, as in `native\build\pdb-out`) or `CHAINNER_C_REGEX_ONLY=ON` (the
regex library and `regex_probe.exe` alone, in `native\build-regex`: `src/regex/README.md`); for builds that stay
on one machine, never for releases, `CHAINNER_C_MARCH=<cpu>` adds `-march=<cpu>` to every target (for example
`native` or `icelake-server`).

**Every default configuration writes the three binaries into `backend/src`** (`chainner_native.dll` and
`_chainner_graph.pyd` into `CHAINNER_C_OUTPUT_DIR`, default `backend/src/nodes/impl`; `chainner_ext.pyd` into
`CHAINNER_C_EXT_OUTPUT_DIR`, default `backend/src/chainner_ext`), replacing the only known-good copies. Copy them to
`native\build\held\` first and check `Get-FileHash` against those copies. The build is reproducible: the same sources
and toolchain give the same three binaries whatever the checkout's path and line endings (`/Brepro`, the source-path
map, `.gitattributes`). This source's binaries (the portable build of 2026-10-09; `chainner_native.dll` and `chainner_ext.pyd` are
unchanged since v0.3.1 and v0.3.2), for reference:

| Binary | SHA-256 |
| --- | --- |
| `chainner_native.dll` | `6da1217311bb59a67d1a7ad50e20007ef733b5dbce763ed6bfcf708271cfecf5` |
| `_chainner_graph.pyd` | `9ede6c6d3f91b7c01bb22a2b6ab7a38feb775761b6c2c40683632d6ae2738499` |
| `chainner_ext.pyd` | `e59941177c46cc8a5f97c812750d8ea69d8e3f1b3ccb37dc1162a451a3232276` |

`generate_onnx_converter.py`, `generate_pixel_art_tables.py` and `generate_tiling_cpp.py` in `native/tools` turn pinned
sources into C++ and never run in the application, as `native/tests/reference_ncnn/generate_optimizer_cpp.py` does for
the NCNN optimizer with its recorded corrections. Rerun one only when its pinned input changes, and review the diff.

## Test

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'; $env:CUDA_VISIBLE_DEVICES = '-1'
$env:NUMBA_CACHE_DIR = "$env:TEMP\chainner-c-numba"
& '.\native\.venv\Scripts\python.exe' -B -m pytest native/tests backend/tests -q -p no:cacheprovider
```

- `CUDA_VISIBLE_DEVICES` hides the GPU from CUDA only. `native/tests/conftest.py` sets
  `VK_LOADER_DRIVERS_DISABLE=*`, which hides it from ncnn's Vulkan (the port's ncnn modules query it when
  they load), as the runtime verifiers do for their hosts; never the package, the bench or a real launch.
- One process. `native/tests/conftest.py` loads the system `msvcp140.dll` first: Pillow 9.2 bundles
  version 14.29, and Torch and ONNX Runtime fail to initialize (WinError 1114) once that copy is the
  process runtime. Windows logs each such failure as a crash and keeps a `python.exe` clone that
  locks DLLs. Never skip tests or change the framework stack instead.
- Root `pyproject.toml` sets pytest `pythonpath = ["backend/src", "native/tools"]`, so tests import
  backend and tool modules by plain name. `backend/tests` is the upstream `d56e507f` set.
  `native/tests/reference_*` are frozen oracles and fixtures; never edit them to make a test pass.
- `test_bench_baseline.py` and `test_bench_cli.py` need the git checkout. `test_video_runtime_tool.py`
  reads the tracked `native/tests/video_runtime/nodes.json` (regeneration: its module docstring).
- Bundle-patch checks (read-only, stdout): `& 'C:\nvm4w\nodejs\node.exe' native\tools\verify_<name>.cjs
  [--manifest out\chaiNNer-C\chainner-c-package.json]` for `independent_ui`, `input_drop`, `edge_drop`
  and `directory_drop`. Non-zero exit on failure; offline, not proof of real Explorer/OLE drag delivery.
- Lint from `native\` with the discovered config, never `--config` (target py314; the root `ignore` holds
  `PLC0415`, since tests defer imports to sequence native DLL loads): `ruff format`, `ruff check --select I
  --fix`, `ruff check` (`native\.venv\Scripts\ruff.exe`, the `requirements.txt` pin). No suppression comments;
  the tests are white-box, so `native/pyproject.toml` turns SLF001 off for them alone. Gates run with
  `--no-cache`: ruff's cache keys on each file, not on files that change import categories.
  `npm run lint:py` runs the backend's two gates with that binary (`lint:fix` its fix and format).
- Type check: plain `pyright` from the repo root, no flags (`native\.venv\Scripts\python.exe -B -m pyright`,
  which `npm run type-check:py` runs).
  `pyrightconfig.json` pins Python 3.14, `native/runtime/cpython-3.14.8` as the environment and
  `native/.venv`'s own site-packages (the dev packages), so the PATH interpreter does not matter. It checks
  `backend`, `native/tools` and `native/tests` (the frozen `reference_*` oracles excluded). A test that
  deliberately breaks a declared contract carries `# pyright: ignore[<rules>] -- <reason>`;
  `test_type_directives.py` pins every such directive by file, rules and reason and rejects any other
  suppression comment.
- `npm run test:py` runs `backend/tests` and `npm run type-check:runtime` the backend's startup type check
  on the provisioned runtime (`python.exe -B`; the latter also with `PYTHONDONTWRITEBYTECODE=1` for the
  worker it spawns, and `PIP_REQUIRE_VIRTUALENV=1`, so a missing dependency fails instead of installing into
  the runtime). `npm run dev:py` stays upstream's and runs PATH `python`.
- Hosted CI (`.github/workflows/lint-backend.yml`, Windows jobs on `.github/actions/backend-env`'s venv of
  the lock's CPU builds) runs ruff, pyright and `backend/tests` only. The native build, `native/tests`, the
  runtime verifiers, parity and the bench are the local gates above and below.

## Package

```powershell
$py = '.\native\.venv\Scripts\python.exe'; $pkg = @('-B', '.\native\tools\package_port.py',
  '--installed-app', "$env:LOCALAPPDATA\chaiNNer\app-0.25.1-nightly2025-10-21",
  '--installed-python', '.\native\runtime\cpython-3.14.8')
& $py @pkg --validate-only; if ($LASTEXITCODE -ne 0) { throw 'preflight failed' }
& $py @pkg --refresh;       if ($LASTEXITCODE -ne 0) { throw 'refresh failed' }
& $py @pkg --refresh;       if ($LASTEXITCODE -ne 0) { throw 'repeat failed' }
```

- `--validate-only` checks the inputs (installed app, UI pins, backend inventory, git state, the
  runtime and its exclusions), prints counts, digests and the `python_stack` and `native_toolchain` records, writes nothing
  and does not inspect the destination. The repeat run must report result `verified_no_op` and leave
  the manifest bytes unchanged.
- `--refresh` requires an existing package. For a new package omit it and pass
  `--destination out\<child>` (a child of `out\`; default `out\chaiNNer-C`). `--copy-workers N` (1-32).
- Output: `out\chaiNNer-C\chaiNNer.exe`, the `portable` marker beside it (keeps the profile inside
  the package) and the manifest `chainner-c-package.json` (schema `chaiNNer-C/package/v2`).
- The runtime `python\python` is copied once and never re-verified (portable dependency installs
  change it). The copy skips every `__pycache__` of the source runtime (its timestamp `.pyc`, and numba's `.nbi`/`.nbc`
  caches for the build CPU); the build then compiles the copy's whole `Lib`, stdlib and site-packages, with the
  packaged interpreter (`compileall --invalidation-mode unchecked-hash`, `-s` the destination, so the bytes do not
  depend on the build directory), records each `.pyc` in the manifest's `runtime` and verifies them (Consult 10 D-9).
  A file that does not compile fails the build. `resources\src` is not compiled. `python_stack.bytecode` names the
  mode, so a package copied before it refuses `--refresh`. The portable profile is never touched. Never rebuild the real package from scratch,
  with one exception: the Python-stack upgrade required a fresh package (the runtime is not refreshable); the
  previous 3.11 package was kept as `out\chaiNNer-C-py311`.
- At run time the worker sets `NUMBA_CACHE_DIR` to `<storage_dir>\numba-cache` (`backend-storage\numba-cache` in the
  package) before any node module imports numba, so PyMatting's cached kernels never land in `python\` (ARCHITECTURE
  section 9; upstream keeps numba's cache in-tree).
- The runtime source is the provisioned `native\runtime\cpython-3.14.8`, and it must be exactly what
  the lock describes: CPython 3.14 or newer with its `python3XX.dll`, the lock's interpreter, every
  pin and pip's version, plus `chainner-pip` (else the host installs it into the package at first
  launch). The copy leaves out eleven distributions (`package_files.EXCLUDED`): dev-only `pytest`,
  `pytest-asyncio`, `pytest-cov`, `coverage`, `pluggy`, `iniconfig`, `Pygments`, `pybind11`, and
  oracle-only `google-re2`, `Sanic-Cors` and `pynvml`; every file their RECORDs list goes (import directories,
  `*.dist-info`, `a1_coverage.pth`, `_pynvml_redirector.pth`, `Scripts\` entry points). Two checks fail the run, at `--validate-only` and at
  build: an excluded name missing from the runtime's metadata, and a shipped distribution's active
  `Requires-Dist` naming an excluded one. The manifest's `python_stack` records the interpreter
  version, build tag, archive and lock SHA-256, the runtime's path, the exclusions with their
  reasons and chainner-pip's version; a refresh refuses a package copied from another stack.
  `native_toolchain` records the clang-cl, MSVC toolset and Windows SDK versions the build pins (read from
  `CMakeLists.txt` and `toolchain.cmake`, each pinned exactly once, or the run fails); a refresh rewrites it with
  the backend's binaries, and a package from before the record gains it at its next refresh. A package copied
  before 2026-10-07 also carries `python_stack.overlay`, a retired key a refresh ignores. Releasing is owner-only.
- A locked DLL/PYD (a running package) fails the run before any write and names the paths. Close
  that app yourself; never kill processes. A run interrupted mid-write recovers on the next run.
- A file changed outside the tool, an untracked or missing file under `backend/src`, or an
  unknown installed UI bundle fails the run. Investigate; never disable a guard.
- After a refresh, run the runtime verifiers (owned backends and profiles, CPU-only, HTTP + SSE,
  oracle versus package, decoded-output equality; evidence to `native/reports/<kind>-<UTC>/`). The oracle is
  `make_oracle.py`'s tree with chaiNNer-C's `chainner_ext`, run on the provisioned runtime the package was copied from
  (`python_stack.runtime`, which keeps `google-re2`, `Sanic-Cors` and `pynvml` for upstream's code); the port
  runs on the package's runtime; no host may install into its interpreter (`PIP_REQUIRE_VIRTUALENV=1`). `/nodes`
  and `/features` compare by hash; our Dependency pins (the lock's tested stack) may differ from upstream's only as
  `verify_runtime.DECLARATION_CHANGES` says, each row with its ruling (`compare_metadata`). Each request runs twice
  per backend; a fixture's `repeat_rule` (framework verifiers) or `repeat_semantic_rule` (video) names what the two
  attempts had to match (`verify_runtime.REPEAT_RULES`: exact but iterated items' final values on the oracle, and also
  but broadcast multiplicities on the port), and `broadcast_multisets` records each attempt's previews per node. An
  iterated item's broadcast field (a node output sent with several values) is compared by its broadcast multiset, not
  last-wins: upstream on 3.14 sends such broadcasts in timing-dependent order (Consult D-35). The port's final value
  must be one the oracle broadcast, and `report.json` and the printed summary flag a run whose oracle attempts ended on
  different final values (`oracle_repeat_final_values_differ`). Rerun
  `make_oracle.py` after each native build: the verifiers refuse an oracle whose `chainner_ext` hashes
  differ from the package manifest's, and any other oracle interpreter (no `python_stack`, a changed lock, a missing
  runtime). `& $py -B native\tools\verify_runtime.py`, then `verify_<kind>.py --include-port` (without it, oracle only) for
  `io_execution_runtime`, `utility_runtime` and `video_runtime`. `verify_framework_runtime.py` loads models by default:
  never in a no-model CPU run. `verify_resize_pipeline.py prepare|run|compare --root <dir>` checks every resize filter.
- The real-use CPU pass (U3-3, Consult 6 P3): `& $py -B native\tools\real_use_pass.py [--package DIR]`. It launches the
  package's `chaiNNer.exe` (an owned job, no input; WM_CLOSE, then termination by PID) and checks `%APPDATA%\chaiNNer`
  unchanged, the package diff against the ruled allow-list (`launch_change`, Consult 8 R-i: a frozen list at the root,
  bytecode only under the backend, since the runtime ships compiled, Consult 10 D-9, and a write there names the
  module the build missed; the app does not inherit `PYTHONDONTWRITEBYTECODE`) and the
  startup log; then the dependency manager and the owner chains (`CHAINS`, `default.chn` with PyTorch on the CPU: allow
  time) on the package's host against the oracle, with a mid-run `/kill`. It refuses to start while the installed
  chaiNNer or the package runs (an exited process object, 0 threads, is skipped and listed in the summary's
  `exited_processes_skipped`). Evidence and `summary.json` go to
  `native/reports/real-use-<UTC>/`; exit 0 only if every gate holds.

## Bench

```powershell
.\native\.venv\Scripts\python.exe -B native\tools\bench.py [--a TREE] [--b TREE] [--cases CASE ...] [--repeats N] [--record-noise] [--env-a NAME=VALUE] [--env-b NAME=VALUE] [--profile] [--python-a PATH] [--python-b PATH] [--png-compare sha256|decoded]
# against the installed chaiNNer (its app and Python are read-only; B is the package; PNG bytes differ by encoder, pixels do not):
.\native\.venv\Scripts\python.exe -B native\tools\bench.py --a "$env:LOCALAPPDATA\chaiNNer\app-0.25.1-nightly2025-10-21\resources\src" --python-a "$env:APPDATA\chaiNNer\python\python\python.exe" --png-compare decoded
```

- A defaults to the frozen B0 copy `native/reports/baselines/B0-20261001/src` (local, read-only), B to the packaged backend
  `out/chaiNNer-C/resources/src`; each side runs on the packaged Python with CUDA hidden, or on another existing `python.exe`
  (`--python-a`/`--python-b`, recorded as `pythons`; installs into a foreign interpreter are blocked and checked: `bench.py` docstring).
  Parity against the oracle: `--a native\build\oracle\src`. The verifiers' guard applies (the four `chainner_ext` hashes
  equal the package manifest's, or B's own files for a plain tree), and A runs on the provisioned runtime, `--python-a`'s
  default; another `--python-a`, or the oracle as B, is refused. Cases: `resize-real-256`, `gaussian`, `lens-spectral`, `morphology`,
  `normal-map`, `caption`, `split-channels`, `palette-median`, `palette-dither`, `parallel-branches`, `video-ffv1`,
  `video-h264`, `video-resize-h264`. All 13 at 6 pairs take roughly 10-20 minutes.
- Per case: one warm-up pair, then N measured pairs in alternating order (N even; verdicts need `MIN_PAIRS = 6`). Outputs and final
  UI state must be identical in every trial or the run fails. Oracles: PNG files by SHA-256 (default) or, with `--png-compare
  decoded`, by pixels and colour chunks (another encoder's files); ffprobe plus streamed FFmpeg raw BGR24 SHA-256; SSE tail validation.
- Each trial records B's item-window decision and counts (`item_windows`); `CHAINNER_C_ITEM_WINDOW=k` in the environment forces K (`1` = sequential); `CHAINNER_C_SERIAL_PRODUCER=1` forces the serial producer (any other value is ignored); `CHAINNER_C_ISA=scalar|avx2|avx512` (set per side with `--env-a`/`--env-b`) caps our ISA units, while the C runtime's and the compiler's own dispatched code follow the CPU (logged as `native isa=...`, recorded as `native_isa`); `avx2` also caps the AVX-512 units, for a part that downclocks under them (spec 4.3 residual).
- `CHAINNER_C_PROFILE=1` only through `--profile`: a timer table per trial in `native_profile`, plus a `pool` entry (this run's deltas of the native pool's dispatches, wakes and empty wakes); never a gate.
- `CHAINNER_C_NUMPY_POOL` (the backend's NumPy memory pool: freed blocks of 512 KiB or more stay warm; read once per process; also the A/B switch, set per side with `--env-a`/`--env-b`): unset or empty = on, idle bytes capped at min(512 MiB, 10 % of RAM); `0` = the pool is not installed (NumPy's default handler, as before SP4c); digits = the cap in MiB; an invalid value = one warning, and the pool is not installed. It needs NumPy 1.23 or newer: an older or mismatched NumPy gets one warning and the default handler. `0` does not undo D8: Load Video's frames are read into fresh NumPy buffers either way, D8 has no runtime switch, and its undo is `git revert 14176509 4cceb734`. With `CHAINNER_C_PROFILE=1` its counters are a `numpy_pool` entry of the same profile line.
- Verdicts compare the median paired ratio with two floors from `native/tools/bench-data/noise-floor.json`, the noise floor and the
  current floor (this K setting's A/A launches): `win` (beyond both, most pairs agree), `regression` (beyond both; blocks),
  `regression (within current A/A)` (beyond the noise floor only; reported, not blocking), `neutral`, `unknown` (no usable floor or
  too few pairs). CPU per item is information only: never confirmed, never a gate.
- Auto-confirmation (not with `--record-noise` or `--profile`): flagged cases re-run on fresh backends. A win must repeat; a regression
  is final when 2 of 3 runs flag it; anything else is `unconfirmed`, which does not clear a regression. Quote the confirmation run's
  change, never the first run's. A floor needs `MIN_LAUNCHES = 3` A/A launches on identical trees (`unknown` until then);
  `--record-noise` writes only `cpu_floors.<auto|k<n>>`, never the shared floors. A WARNING marks runs whose background load exceeds the floors'.
- Records: `native/reports/bench-<UTC>/` (`result.json`, `summary.md`, `confirmation.*`); media: `F:\chaiNNer-C-diagnostics\bench-<UTC>\`
  (each trial's JSON after a success, everything after a failure; a backend deletes its private FFmpeg copy on exit). Rules: `bench.py`.

## Safety

The `F:\` paths in this section and in the bench records above are the local gates' paths on the owner's machine
(Consult 14 D-30 (d)).

- Protected, read-only: the installed app, `$env:APPDATA\chaiNNer` (profile and
  Python), the dataset `F:\Download\New folder\DiverSeg-IP\HR\0`. Never write to or run diagnostics
  against `F:\Download\New folder\test` (the user's output). Owned media: `F:\chaiNNer-C-diagnostics`.
- Always run `native\.venv\Scripts\python.exe` with `-B` and `NUMBA_CACHE_DIR` set: the venv sits on
  the provisioned runtime and must not write bytecode or numba caches into it. Never use PATH `python`.
- CPU only: `CUDA_VISIBLE_DEVICES=-1`, no AI models in benchmarks; log GPU needs in `GPU-DEFERRED.md`.
- No GUI or desktop automation. Never kill processes by name; reap only processes you started.
- Never run `native/reports/folder-drop/directory-drop.chn` as saved (its Save Image targets the
  input dataset); `native/reports/gui-smoke/resize-test.chn` targets the user's output: inspect only.
- License: GPLv3 (root `LICENSE`). Third-party notices ship beside the DLL in
  `backend/src/nodes/impl/chainner_native.LICENSE.txt`; see `ARCHITECTURE.md` section 10.
