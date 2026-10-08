# chaiNNer-C design decisions

The key decisions and why they were taken. The published history is one commit, so this file is the record of the
reasoning. Each entry: the decision, why, then where it lives. Mechanics: `ARCHITECTURE.md`. Commands: `README.md`.
Checklists for changing any of this: `MAINTAINING.md`.

## 1. Scope and the exactness contract

- **Upstream chaiNNer, with its CPU image nodes in C/C++.** The app, node graph, schemas, saved chains and UI stay
  upstream's; only how the CPU nodes compute changes, so chains and behavior carry over. `native/` (kernels and
  modules), `backend/src/nodes/impl/native*.py` (bridges).
- **Outputs match upstream.** The reference (the oracle) is upstream's own backend code, run on the same Python stack
  with the same inputs, not a frozen copy. Where upstream is undefined or defective, the port does the deterministic,
  correct thing and records it. `native/tools/make_oracle.py`, `native/tests/oracle/compat.diff`, the runtime
  verifiers; `ARCHITECTURE.md` section 7 lists every accepted difference and test tolerance.
- **Retained engines stay engines.** OpenCV/IPP, NumPy's BLAS/LAPACK, FFmpeg, FreeType and the inference frameworks
  are called, never wrapped or renamed and counted as a conversion. `ARCHITECTURE.md` sections 7 and 8.

## 2. Numerics

- **`/fp:strict`: no contraction, reassociation or fast-math.** The bits must follow the mirrored library's operation
  order, and the kernels' FP-flag windows mirror `np.seterr`, so no FP operation may move across them.
  `native/CMakeLists.txt` (`add_compile_options`).
- **Dynamic denormals** (`-fdenormal-fp-math=dynamic`). Kernels may run under DAZ/FTZ, so LLVM must not fold denormals
  as IEEE. `native/CMakeLists.txt`.
- **NaN bits are fixed in source, never by a flag.** LLVM leaves a NaN's sign and payload unspecified, so where
  upstream's NaN bits are observable they are derived from the operands as integers. `normal_nan_outputs` in
  `native/include/color_shared.h`; the palette key (`native/include/palette_shared.h`).
- **SIMD only where bit-identical** (elementwise, per-output lane), and no reordered reductions. `ARCHITECTURE.md`
  section 5.
- **Exact-version gating.** A native path that mirrors a library's numerics runs only on the exact tested version;
  any other version uses the library's own code (logged once), so results stay correct and only the speedup is lost.
  `backend/src/nodes/impl/native_versions.py` (`CV2`, `NUMPY`, `ORT`); engine-dispatch mirrors
  `native_opencv_simd.py`, `native_numpy_simd.py`, `native_numpy_reduce.py` in the same directory.
- **CRT math as the library calls it.** Mirrors of NumPy's, SciPy's and CPython's scalar math call the process's
  `ucrtbase.dll`, as those libraries do, so `pow` and friends round alike. `native/include/cn_crt_math.h`,
  `native/src/crt_math.c`.

## 3. Instruction sets

- **One portable x86-64 build with an x86-64-v2 floor.** NumPy 2.x already requires x86-64-v2, so the native layer
  adds no CPU requirement and one binary serves every user. `native/CMakeLists.txt`.
- **Per-file AVX2/AVX-512 units chosen at run time.** Each `*_avx2.c`/`*_avx512.c` unit is compiled for its level only
  and holds no exports; `cn_isa_current()` probes CPUID and XCR0 once, and a C entry reads it once per call, so one
  call never mixes levels. An AVX-512 unit is kept only for a measured win. `native/src/isa.c`,
  `native/include/isa.h`, the unit lists in `native/CMakeLists.txt`.
- **`CHAINNER_C_ISA=scalar|avx2|avx512`** caps the level, for tests and for parts that downclock under AVX-512.
  `native/src/isa.c`.
- **`CHAINNER_C_MARCH` is local-only, never in releases.** `-march` code runs only on CPUs like the build machine's;
  the hot kernels already dispatch, so the gain is modest. `native/CMakeLists.txt`.

## 4. Toolchain

- **clang-cl 23.1.2 only**, with lld-link, llvm-lib and Ninja; configuration refuses any other compiler or version.
  One compiler means one set of bits to verify and reproduce. `native/CMakeLists.txt`, `native/toolchain.cmake`,
  `native/Build.ps1`.
- **ThinLTO, no PGO in shipped builds.** Link-time optimization across units; the shipped code is fixed by the sources
  and flags alone, with no training profile as an extra input. PGO stays an option for local builds, like
  `CHAINNER_C_MARCH`. `native/CMakeLists.txt`.
- **Pinned MSVC toolset 14.44.35207 and Windows SDK 10.0.26100.0** (headers and libraries; no Developer environment).
  The package manifest records them (`native_toolchain`). `native/toolchain.cmake`.
- **Reproducible binaries.** `/Brepro` (no timestamps, no PDB in Release) and the source-path map
  `-fmacro-prefix-map=<native>=.` (clang's MSVC mangling hashes the main file's path into anonymous-namespace names),
  plus LF checkouts (`.gitattributes`), so anyone can rebuild and compare with the hash table in `README.md`.
- **`/W4 /WX`, no suppressions** on first-party targets. Vendored code keeps its vendor warning level and is patched
  only for an output or safety defect, recorded with its new hash in its `PROVENANCE.md`.

## 5. Python stack

- **CPython 3.14.8 with current dependencies**, recorded as a full freeze in `native/python-stack.lock.txt`
  (interpreter, build tag, archive SHA-256, pip). The tests, the oracle and the package run on that one stack.
  `native/tools/provision_runtime.ps1` installs exactly the lock and writes it only under `-Relock`.
- **Users get packages from chaiNNer's dependency manager; pins are floors.** The backend's `Dependency` pins are the
  lock's versions, installed only when missing or older, so newer versions are not blocked and the version gate
  (section 2) keeps them correct. `backend/src/packages/*/__init__.py`.
- **Declarations differ from upstream's only as listed:** ncnn for ncnn-vulkan (no cp314 wheel), torch and torchvision
  from the CUDA 13.2 index, onnxruntime-gpu from PyPI. `verify_runtime.DECLARATION_CHANGES` in
  `native/tools/verify_runtime.py`.
- **The same interpreter everywhere.** The source UI downloads the lock's CPython
  (`src/main/python/integratedPython.ts`); the package ships the provisioned runtime, precompiled with `unchecked-hash`
  bytecode so builds are byte-identical.

## 6. chainner_ext in C

- **`chainner_ext.pyd` is a C module (limited API)**, a drop-in for the Rust chainner_ext 0.3.10 with every name of it;
  its image functions bind `chainner_native.dll`'s kernels. It was remade in C during the CPython 3.14 upgrade, for
  speed: no Rust toolchain in the build, and the image functions share the nodes' C kernels.
  `native/src/chainner_ext/`, `backend/src/chainner_ext/`.
- **The regex engine is ported from the Rust `regex` 1.8.4 and `regex-syntax` 0.7.2 crates**, and also serves
  `_chainner_graph.pyd` through the `_C_API` capsule. `native/src/regex/` (file map in its `README.md`).
- **Parity with the Rust module is held by a conformance corpus.** `native/tests/chainner_ext/`,
  `native/tests/test_chainner_ext_manifest.py`, `native/tools/chainner_ext_conformance.py`; licences under
  `native/third_party/chainner_ext`, `regex`, `regex-syntax` and `unicode`.

## 7. The CPU scheduler

- **One native CPU pool, budgeted by the process affinity mask.** `cn_parallel_for(count, grain)` partitions by
  `(count, grain)` alone and shares helpers between concurrent calls, so results never depend on the CPU count or on
  other callers. `native/src/parallel.c`, `native/include/parallel.h`; guard `native/tests/test_parallel_affinity.py`.
  OpenCV's and BLAS's own pools keep their defaults, since their thread counts can change outputs.
- **Independent pure-CPU inputs are awaited together.** `owned_gather`, `NodeFlights` and `pure_cpu_dependency` in
  `backend/src/nodes/impl/execution_scheduler.py`; 8 pool workers (`POOL_SIZE`, `backend/src/server.py`).
- **Cross-item pipelining with in-order commit (the item window).** Later items' pure-CPU nodes run ahead
  speculatively; one sequential commit path replays their results in item order, so side effects, events, cache and
  errors keep upstream's order. Read-ahead only when no output can alias an input; speculation is capped by memory.
  `backend/src/nodes/impl/item_window.py`, `backend/src/process.py`; `CHAINNER_C_ITEM_WINDOW` forces K.
- **Decode off the producer.** Load Images items come in two phases, a serial `describe()` and a parallel
  `materialize(token)` run as an engine job, because the serial producer had become the bound. `backend/src/process.py`,
  `item_window.py`, `backend/src/api/lazy.py`; `CHAINNER_C_SERIAL_PRODUCER=1` restores the serial producer.
- **The NumPy memory pool.** A NumPy data-memory handler keeps freed blocks of 512 KiB or more warm, because first
  touch of fresh output buffers was the measured cost; idle blocks are released after each run.
  `native/src/numpy_pool.cpp`, `backend/src/nodes/impl/numpy_pool.py`; `CHAINNER_C_NUMPY_POOL`.
- **A GC after failed runs.** One `gc.collect()` after a failed or stopped `/run` (`collect_failed_run`,
  `backend/src/server.py`), since a failed run's arrays stayed alive in reference cycles; successful runs never collect.

## 8. GPU engines

- **PyTorch, ONNX Runtime and NCNN are kept as upstream.** Inference stays theirs; only the ONNX CPU session adapter,
  the ONNX-to-NCNN converter, NCNN param handling and tiling control are native.
  `backend/src/packages/chaiNNer_pytorch`, `chaiNNer_onnx`, `chaiNNer_ncnn`.
- **TensorRT nodes follow upstream's node design** (upstream's `chaiNNer_tensorrt` package, schema ids
  `chainner:tensorrt:*`) on the TensorRT 11 API with cuda-python. `backend/src/packages/chaiNNer_tensorrt/`,
  `backend/src/nodes/impl/tensorrt/`.
- **Automatic overlapped tiling.** No tile-size input: tiles are as large as the engine's optimization profile allows
  and always overlap. At each shared edge a tile's outer margin is discarded, then the shared band is feather-blended,
  so no seam reaches the output; image borders are never feathered. `nodes/impl/tensorrt/tiling.py`
  (`TILE_OVERLAP`); test `backend/tests/test_tensorrt_tiling.py`.
- **Engines and sessions outlive runs.** Deserialized engines and sessions are shared by key and reference-counted per
  node, so a repeated run does not deserialize again; they are freed by the node's Clear item or by node removal
  (both call `/clear-cache/individual`). `nodes/impl/tensorrt/cache.py`; test `native/tests/test_tensorrt_cache.py`.

## 9. UI

- **Two paths.** The portable package ships the installed upstream nightly's built UI with reviewed, hash-pinned
  patches, so it keeps the upstream build's bytes except reviewed edits; unknown bundles fail closed.
  `native/tools/package_port.py` applies `independent_ui.py`, `input-drop.json`, `tensorrt-types.json` and
  `tensorrt-clear.json`. `src/` is upstream's frontend with the same changes ported: the run-from-source UI and the
  source of `npm run make`.
- **Its own app-data folder.** From source, `%APPDATA%\chaiNNer-C` (`src/main/platform.ts`); the package keeps its
  profile inside its folder (`portable` marker). An installed chaiNNer's `%APPDATA%\chaiNNer` is never touched.
- **No update check.** An independent edition never checks upstream releases, and old settings cannot re-enable it.
  `src/renderer/components/Header/AppInfo.tsx`; both bundle patches force `checkForUpdatesOnStartup` off.
- **Version label.** The UI shows `v0.3.2` (header, About, startup log, system information) while `app.getVersion()`
  stays upstream's, since it stamps saved chains that upstream's migrations read. `src/common/version.ts`,
  `independent_ui.PRODUCT_VERSION`.

## 10. Publishing

- **Source plus an optional release zip.** The source tree is the product; `npm run make` builds a release from `src/`
  with `backend/src` (binaries included) as an extra resource, and its zip gets a `portable` marker
  (`forge.config.js`). The local portable package (`package_port.py`) is the artifact the acceptance gates test.
- **A single commit on the upstream base** (`d56e507f`), so the diff against upstream is the whole change; the working
  history stays local.
- **Local-only files never ship.** Git-ignored: `native/reports/` (evidence), `docs/superpowers/` (planning records),
  `native/local/` (local overlays), `native/runtime/`, `out/`.
