# chaiNNer-C

**A faster [chaiNNer](https://github.com/chaiNNer-org/chaiNNer).** chaiNNer-C is the node-based image processing app
you already know, with its CPU image nodes rewritten in C and C++ for speed. Same nodes, same chains, same results;
it just gets there sooner.

chaiNNer-C is an independent fork of chaiNNer by [chaiNNer-org](https://github.com/chaiNNer-org) and its contributors.
The app, the node graph, the nodes and the interface are their work. chaiNNer-C is not affiliated with or endorsed by
chaiNNer-org. For chaiNNer itself, its documentation and its community, see the
[upstream repository](https://github.com/chaiNNer-org/chaiNNer).

## Why chaiNNer-C?

- **Much faster on the CPU.** Resizing, blurring, color work, blending, dithering, palettes and more run as optimized
  native code that uses all your cores and your CPU's AVX2 or AVX-512 when it has them. On our benchmark graphs,
  CPU-heavy chains ran 11 to 86 times faster than chaiNNer on the same PC; how much you gain depends on your CPU and
  your chains.
- **The same results.** Every converted node is tested to produce the same output as chaiNNer's own code.
  The rare, documented exceptions are listed in [the architecture notes](native/ARCHITECTURE.md#7-retained-engines-and-exactness-exceptions).
- **TensorRT upscaling.** New TensorRT nodes run your `.engine` files on NVIDIA GPUs. Large images are split into
  overlapping tiles automatically, with no visible seams, and engines stay loaded between runs (right-click the node
  and choose **Clear** to free the memory).
- **An up-to-date Python stack:** Python 3.14, NumPy 2, OpenCV 5, current PyTorch and ONNX Runtime.
- **Lives next to chaiNNer.** chaiNNer-C keeps its settings and its Python in its own folder
  (`%APPDATA%\chaiNNer-C`), so an existing chaiNNer install is never touched.

## Download and run

1. Download the latest `chaiNNer-C-…-win-x64.zip` from the [Releases page](https://github.com/Rift73/chaiNNer-C/releases).
2. Unzip it anywhere and run **`chaiNNer.exe`**.
3. On first start, chaiNNer-C downloads its own Python. Then open the **dependency manager** (as in chaiNNer) to
   install the packages for the nodes you want, for example PyTorch, ONNX Runtime, NCNN or TensorRT.

chaiNNer-C doesn't update itself; check the Releases page for new versions.

## System requirements

- **Windows 10 or 11, 64-bit.**
- **Any reasonably modern 64-bit Intel or AMD CPU:** Intel from about 2009 on, AMD from about 2011 on. Every PC that
  runs Windows 11 qualifies. (Technically: x86-64-v2, the baseline NumPy 2 requires.) Newer CPUs with AVX2 or AVX-512
  are used automatically for extra speed.
- **A GPU is optional.** PyTorch, ONNX Runtime and TensorRT use NVIDIA GPUs; NCNN runs on any Vulkan GPU.

Not supported: 32-bit Windows, ARM PCs (such as Snapdragon laptops), macOS and Linux.

## FAQ

**Do my chaiNNer chains work in chaiNNer-C?** Yes. The nodes are the same, so chains open and run as in chaiNNer, and
chains saved in chaiNNer-C open in chaiNNer too, except chains using the TensorRT nodes, which need a chaiNNer version
with TensorRT support.

**Will it affect my existing chaiNNer?** No. chaiNNer-C uses its own folder for its settings and Python.

**Do I need an NVIDIA GPU?** No. The CPU nodes, which are what chaiNNer-C speeds up, need no GPU at all. The GPU is
only for AI upscaling with PyTorch, ONNX Runtime or TensorRT models.

**Where do I report a problem?** Open an issue on this repository, not on upstream chaiNNer.

## Building from source

You need Windows x64, [LLVM 23.1.2](https://github.com/llvm/llvm-project/releases), Visual Studio 2022 Build Tools
(MSVC toolset 14.44.35207 and Windows SDK 10.0.26100.0), CMake, and Node.js. In PowerShell, from the repository folder:

```powershell
powershell -NoProfile -File native\tools\provision_runtime.ps1 -BuildOnly   # Python 3.14 headers for the build
& '.\native\Build.ps1' -Configuration Release                               # compile the native code
npm ci                                                                      # install the app's packages
npm start                                                                   # run chaiNNer-C
```

Details (development setup, tests, packaging, toolchain options) are in [`native/BUILDING.md`](native/BUILDING.md).

## Documentation

- [`native/BUILDING.md`](native/BUILDING.md): build, test and package.
- [`native/README.md`](native/README.md): the full developer command reference (tests, verifiers, benchmarks).
- [`native/ARCHITECTURE.md`](native/ARCHITECTURE.md): how it's built and the exactness rules.
- [`native/DESIGN-DECISIONS.md`](native/DESIGN-DECISIONS.md): the key decisions and why they were made.
- [`native/MAINTAINING.md`](native/MAINTAINING.md): checklists for upgrades, changes and releases.
- [`native/STATUS.md`](native/STATUS.md): current state and pending work.

## License and credits

chaiNNer-C is free software under the GNU General Public License v3.0 ([`LICENSE`](LICENSE)), chaiNNer's license.
chaiNNer is by chaiNNer-org and its contributors.

- The native code adapts algorithms from other projects (chaiNNer-rs, OpenCV, Pillow, NumPy, PyMatting and others);
  [`chainner_native.LICENSE.txt`](backend/src/nodes/impl/chainner_native.LICENSE.txt) lists each one with its license.
- Vendored and ported third-party code (fpng, pybind11, the ONNX Runtime C header and the sources ported into
  `chainner_ext`) keeps its license texts under [`native/third_party`](native/third_party).
- Python and each installed package keep their own licenses.
