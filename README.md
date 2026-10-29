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
- **VOSR 2.0 generative upscaling.** A PyTorch node runs [VOSR 2.0](https://github.com/cswry/VOSR), a one-step
  generative super-resolution model, on PyTorch or TensorRT (BF16 or FP8), and stabilize video frames from Load Video
  over time automatically. See the [VOSR guide](native/VOSR2.md).
- **An up-to-date Python stack:** Python 3.14, NumPy 2, OpenCV 5, current PyTorch and ONNX Runtime.
- **Lives next to chaiNNer.** chaiNNer-C keeps its settings and its Python in its own folder (the folder you
  unzip it to; `%APPDATA%\chaiNNer-C` when run from source), so an existing chaiNNer install is never touched.

## Download and run

1. Download the latest `chaiNNer-C-…-win-x64.zip` from the [Releases page](https://github.com/Rift73/chaiNNer-C/releases).
2. Unzip it anywhere and run **`chaiNNer.exe`**.
3. On first start, chaiNNer-C downloads its own Python. Then open the **dependency manager** (as in chaiNNer) to
   install the packages for the nodes you want, for example PyTorch, ONNX Runtime, NCNN or TensorRT.

chaiNNer-C doesn't update itself; check the Releases page for new versions.

On Linux, the Linux zip unzips to `chaiNNer-linux-x64/`, where you run `./chaiNNer.sh` ([On Linux](#on-linux)). Linux
zips arrive with the next release; until then, build from source (below).

## System requirements

- **Windows 10 or 11, 64-bit, or Linux x86-64 with glibc 2.28 or newer:** Ubuntu 20.04, Debian 10, RHEL, Rocky and
  AlmaLinux 8 and every later release of them, and current Fedora, Arch and openSUSE.
- **Any reasonably modern 64-bit Intel or AMD CPU:** Intel from about 2009 on, AMD from about 2011 on. Every PC that
  runs Windows 11 qualifies. (Technically: x86-64-v2, the baseline NumPy 2 requires.) Newer CPUs with AVX2 or AVX-512
  are used automatically for extra speed.
- **A GPU is optional.** PyTorch, ONNX Runtime (also through TensorRT) and TensorRT use NVIDIA GPUs, on Linux as on
  Windows; NCNN runs on any Vulkan GPU. Without a supported GPU the CPU does the work.

Not supported: 32-bit systems, ARM PCs (such as Snapdragon laptops) and ARM Linux, Linux without glibc (such as
Alpine), and macOS.

### On Linux

- Unzip to a folder you can write to: like the Windows zip, chaiNNer-C keeps its settings, logs and Python in the folder
  it runs from. Start `./chaiNNer.sh` from the unzipped `chaiNNer-linux-x64/`. An archive tool that drops file modes
  (such as Python's `zipfile`) leaves `chaiNNer.sh` and `chainner` not executable: `chmod +x chaiNNer.sh chainner
  resources/src/texconv/texconv` (chaiNNer-C also restores the last, its DDS converter, itself).
- A minimal install can lack system libraries that Electron, OpenCV or VapourSynth link against. At every start,
  `chaiNNer.sh` checks for them and, if some are missing, lists them and prints the command that installs them for your
  distribution (`apt-get`, `dnf`, `pacman` or `zypper`). `CHAINNER_SKIP_LIBRARY_CHECK=1` skips the check.
- Chromium's sandbox needs user namespaces, which some systems do not allow (Docker, Ubuntu 23.10 and later, hardened
  kernels). There `chaiNNer.sh` starts without the sandbox and says so. `CHAINNER_SANDBOX=0` always starts without it
  and `CHAINNER_SANDBOX=1` always with it.
- `./chaiNNer.sh run chain.chn` (the command line) also needs a display, because Electron cannot start without one. On a
  server, install `xvfb` and `xauth`; the script then runs the chain on a virtual display.
- Copy To Clipboard needs `wl-clipboard` (Wayland) or `xclip` (X11).
- DDS files are read and written by chaiNNer-C's own texconv, built on the DirectXTex release Windows' texconv.exe
  comes from, with the same results except some steps Windows runs through its imaging component (WIC): the Cubic
  mip filter on 8-bit formats, a colour image saved to a one-channel 8- or 16-bit format, and the
  mip levels of a few formats (SNORM, 4-bit), which can differ by a step. BC6H and BC7 compress on the CPU.
- Limits: every package works from glibc 2.28; below glibc 2.34 (RHEL 8, Ubuntu 20.04, Debian 10 and 11) VapourSynth's
  Preview Clips cannot open vsview, whose Qt needs a newer one. A result that goes through the C math library (for
  example the exponentials and trigonometry of Resize's filters) follows the system's library on Linux, as NumPy does,
  so it can differ from Windows in the last bit, and between glibc versions.

## FAQ

**Do my chaiNNer chains work in chaiNNer-C?** Yes. The nodes are the same, so chains open and run as in chaiNNer, and
chains saved in chaiNNer-C open in chaiNNer too, except chains using the TensorRT nodes, which need a chaiNNer version
with TensorRT support, or the VOSR node, which only chaiNNer-C has.

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

On Linux (x86-64) you need LLVM 23.1.2, NASM 3.01, CMake, Ninja and Node.js, and run
`native/tools/provision_runtime.sh --build-only` and `native/build_linux.sh` in place of the two PowerShell scripts;
`native/tools/build_linux_release.sh` builds the zip in Docker.

Details (development setup, tests, packaging, toolchain options, Linux) are in
[`native/BUILDING.md`](native/BUILDING.md).

## Documentation

- [`native/VOSR2.md`](native/VOSR2.md): the VOSR 2.0 node (model download, settings, video, speed).
- [`native/VAPOURSYNTH.md`](native/VAPOURSYNTH.md): the VapourSynth nodes (install, preview in vsview, family nodes, AI Upscale, your own functions).
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
- Vendored and ported third-party code (the image codec libraries, vapoursynth-descale, pybind11, the ONNX Runtime C
  header and the sources ported into `chainner_ext`) keeps its license texts under
  [`native/third_party`](native/third_party).
- VOSR 2.0 is by Rongyuan Wu, Lingchen Sun, Zhengqiang Zhang, Xiangtao Kong, Jixin Zhao, Shihao Wang and Lei Zhang
  (Apache-2.0); its model code in `backend/src/vendor/vosr2` is adapted from ComfyUI-VOSR2 and VOSR and keeps their
  notices (`NOTICE.md`). The weights are downloaded separately.
- Python and each installed package keep their own licenses.
