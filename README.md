# chaiNNer-C

[chaiNNer](https://github.com/chaiNNer-org/chaiNNer) with its CPU image nodes rewritten in C and C++. Same nodes, same
chains; CPU-heavy chains ran 11 to 86 times faster on our benchmarks.

An independent fork, not affiliated with or endorsed by chaiNNer-org. The app, the node graph and the interface are
their work.

## What it adds

- **Native CPU nodes** that use every core, and AVX2 or AVX-512 when the CPU has them.
- **TensorRT nodes** for `.engine` files on NVIDIA GPUs, with automatic seamless tiling.
- **[VOSR 2.0](native/VOSR2.md)** generative upscaling for images and video.
- **[VapourSynth](native/VAPOURSYNTH.md)** nodes.
- **AMD GPUs**: PyTorch on ROCm, ONNX models through MIGraphX.
- **A current stack**: Python 3.14, NumPy 2, OpenCV 5.

## Install

Download the zip from [Releases](https://github.com/Rift73/chaiNNer-C/releases), unzip it and run `chaiNNer.exe`
(Linux: `./chaiNNer.sh`, see the [Linux notes](native/LINUX.md)). The first start downloads Python; the dependency
manager installs PyTorch, ONNX Runtime, NCNN or TensorRT.

Settings and Python stay in the unzipped folder, so an installed chaiNNer is never touched. New releases are offered at
start.

## Requirements

- Windows 10 or 11 (64-bit), or Linux x86-64 with glibc 2.28 or newer.
- An x86-64-v2 CPU: Intel since about 2009, AMD since about 2011.
- A GPU is optional: NVIDIA, AMD (ROCm 10.0), or any Vulkan GPU for NCNN.
- Not supported: macOS, ARM, 32-bit, Linux without glibc.

## Build and documentation

[Building](native/BUILDING.md) · [Developer reference](native/README.md) · [Architecture](native/ARCHITECTURE.md) ·
[Design decisions](native/DESIGN-DECISIONS.md) · [Maintaining](native/MAINTAINING.md) · [Status](native/STATUS.md)

## License

GPL-3.0 ([`LICENSE`](LICENSE)), as chaiNNer. Third-party code keeps its licenses:
[adapted algorithms](backend/src/nodes/impl/chainner_native.LICENSE.txt), [vendored libraries](native/third_party),
[VOSR 2.0](backend/src/vendor/vosr2/NOTICE.md).
