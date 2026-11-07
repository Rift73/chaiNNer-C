# chaiNNer-C on an AMD GPU: the test kit

`native/tools/amd/kit.sh` checks chaiNNer-C on a Linux machine with an AMD GPU (reached over SSH) in one command and
writes one report file to send back. It installs the app from its Linux zip and its packages through the dependency
manager (as a user would), proves that PyTorch runs on the GPU, runs every test marked `amd_gpu`, runs real chains
through the app's command line, runs every architecture spandrel knows on the GPU, compares spandrel's own HIP kernels
with PyTorch's, and measures a speed table. Allow about an hour and a half.

## The machine

- **GPU**: any AMD GPU that ROCm 10.0.0 supports: Instinct MI100 (gfx908), MI210/MI250 (gfx90a), MI300X/MI325X (gfx942),
  MI350/MI355 (gfx950); Radeon RX 6800/6900 (gfx1030) and other RX 6000; RX 7900 (gfx1100), RX 7800/7700 (gfx1101),
  RX 7600 (gfx1102); RX 9070 (gfx1201), RX 9060 (gfx1200); Ryzen AI Max (gfx1151). Most useful, in this order: one
  Radeon RX 7000 or 9000 (the GPUs users have; the RX 9000 also runs FP8), one Instinct MI300X (wave size 64 and its
  own FP8 format), a machine with two or more GPUs (checks the GPU order), and a machine with an integrated plus a
  discrete AMD GPU if one is offered (memory reporting of integrated GPUs).
- **Image**: plain `ubuntu:24.04` or `ubuntu:22.04`, or one of AMD's ROCm images
  (`rocm/dev-ubuntu-24.04`, `rocm/pytorch`); the kit brings its own ROCm (AMD's pip packages), so the image's ROCm
  does not matter. The machine must give the container the GPU (`/dev/kfd` and `/dev/dri`; with Docker,
  `--device /dev/kfd --device /dev/dri`); the kit says so if not.
- **Launch mode**: SSH. **Disk**: 80 GB (the kit needs 40 GB free; VOSR alone downloads 7 GB). Root is fine.

## What to copy from this PC

| Name in the commands | What | Where it is |
|---|---|---|
| `<LINUX_ZIP>` | the Linux app zip, built from the commit the machine clones (the tests run from the clone; the report names any difference) | `native/tools/build_linux_release.sh` writes it to `<work>/src/out/make/zip/linux/x64/chaiNNer-linux-x64-<version>.zip` |
| `esrgan.pth` | an ESRGAN (without one the kit downloads Real-ESRGAN's 4x anime model) | your models |
| `dual.safetensors` | a DUAL model | your models |
| `drft.safetensors` | a legacy DRFT-L x4 | your models |
| `hat.pth` | a HAT | your models |
| `dat2.pth` | a DAT2 | your models |
| `tests\` | the two small models the VapourSynth tests read, names unchanged | `2x_Waifaux-NL3-SuperLite_latest_G.pth` and `4xLSDIRCompactN-fp32-opset17.onnx` |
| `chainner_example_migx_ops.so` | MIGraphX's example custom-op library, for the runner's GPU custom-op test (it skips without it) | built with the zip's AMD modules (`native/migraphx_runner/tests/example_ops.hip`, glibc 2.28): `native/tools/build_linux_release.sh` with `--rocm` and `--migraphx` leaves it in `<work>/build/migraphx_runner/` |

The model files are renamed on the way (the kit finds them by these names); leave out any to skip its chains and
speed rows. VOSR needs no file: the app downloads its model on first use. The ONNX Runtime and spandrel wheels need no
copy either: their releases are public, and the dependency manager installs them (a wheel in `--wheels` replaces a
download, for testing a newer build).

## Commands

On this PC (PowerShell; `<PORT>` and `<HOST>` are the machine's SSH port and address):

```
ssh -p <PORT> root@<HOST> "mkdir -p /root/kit-in/models/tests"
scp -P <PORT> <LINUX_ZIP> root@<HOST>:/root/kit-in/app.zip
scp -P <PORT> <ESRGAN> root@<HOST>:/root/kit-in/models/esrgan.pth
scp -P <PORT> <DRFT> root@<HOST>:/root/kit-in/models/drft.safetensors
scp -P <PORT> <HAT> root@<HOST>:/root/kit-in/models/hat.pth
scp -P <PORT> <DAT2> root@<HOST>:/root/kit-in/models/dat2.pth
scp -P <PORT> <DUAL> root@<HOST>:/root/kit-in/models/dual.safetensors
scp -P <PORT> <EXAMPLE_OPS> root@<HOST>:/root/kit-in/models/
scp -P <PORT> <TEST_MODEL_1> <TEST_MODEL_2> root@<HOST>:/root/kit-in/models/tests/
ssh -p <PORT> root@<HOST>
```

On the AMD machine:

```
apt-get update && apt-get install -y git tmux   # skip if both are there
git clone --depth 1 -b chaiNNer-C https://github.com/Rift73/chaiNNer-C /root/chaiNNer-C
bash /root/chaiNNer-C/native/tools/amd/kit.sh --zip /root/kit-in/app.zip --models /root/kit-in/models
```

Back on this PC, the report (the last line of the kit names it):

```
scp -P <PORT> "root@<HOST>:/root/amd-kit/amd-kit-report-*.tar.gz" .
```

Send that file back. A connection that drops does not stop the kit if it runs in `tmux` (`tmux new -s kit`, then the
command; `tmux attach -t kit` after reconnecting). `--dry-run` walks every step without a GPU or installing anything and
writes a sample report; `--iterations N` changes the speed table's 50 timed calls; `KIT_SKIP_VOSR=1` (before the
command) leaves out VOSR's 7 GB download, chains and speed rows; `KIT_ONLY='<regular expression>'` runs only the chains
and speed rows whose names match (for example `KIT_ONLY='^dual-'` to repeat DUAL's); `KIT_ARCHS='<regular expression>'`
runs only the spandrel architectures whose IDs match in the architecture check (for example `KIT_ARCHS='^(SST|DAT)$'`);
`KIT_WORK=<folder>` moves the work folder from `/root/amd-kit`.

## What it does and how long it takes

| Step | Script | About |
|---|---|---|
| Host check: kernel, amdgpu driver, `/dev/kfd` and `/dev/dri` access, groups, the GPUs (gfx), libatomic, libstdc++, `/dev/shm`, disk, RAM. Stops with the reason if no GPU is usable | `host_check.sh` | seconds |
| Setup: system packages (the launcher's list, libatomic, Mesa's Vulkan driver for NCNN, a C compiler for the test that launches Triton's own kernels), the app (a zip older than the 2026-10-29 fresh-install fix is refused), a first headless run (it downloads its Python), the copied wheels, then the dependency manager installs PyTorch, ONNX, NCNN, MIGraphX and VapourSynth with the ROCm builds it picks for this GPU | `setup.sh` | 10-15 min |
| GPU proof (PyTorch on each GPU against the CPU, chaiNNer's GPU list and VRAM gauge against HIP and amd-smi, only this GPU's kernel wheels installed, MIGraphX first in ONNX Runtime, rocminfo, Vulkan devices), then every `amd_gpu` test; with two GPUs the GPU-list test again with the order swapped | `test.sh`, `gpu_proof.py` | 5-15 min |
| Chains, headless through `chaiNNer.sh run`: PyTorch FP32 and FP16, ONNX Runtime with MIGraphX, MIGraphX Build Program + Upscale Image, NCNN on Vulkan, DUAL, DRFT, HAT, DAT2 and VOSR (also in GPU Memory Low), each output compared with PyTorch FP32 (PSNR) | `chains.sh`, `chains/` | 15-20 min |
| Every architecture: each architecture of the installed spandrel and spandrel_extra_arches (about 60) as a small network with random weights (seed 0; its default settings, or small ones the script names), loaded through its spandrel loader, one 64x64 input on the GPU in FP32, and in FP16 and BF16 where the model allows them, against the CPU in FP32: error, NaN or infinity (a failed row), largest difference (marked large above a threshold per precision), ms per call. Each architecture runs in a process of its own with a time limit; architectures not reached, or stopped, by the 15-minute cap are listed as not run, not as failures. SST and SSTReal are expected to fail on RX 6000 (gfx103x) and MI100 (gfx908), whose PyTorch has no flash attention; the FFT-based FIGSR, GFISRv2, FFTformer and LaMa get their first FP16 check on AMD here | `arch_check.py` | 10-15 min |
| spandrel's HIP kernels (spandrel 0.4.2+c12 and newer; an older spandrel is a line in the report): the ESRGAN and the DAT2 at 256x256 in FP32 and FP16, each with `SPANDREL_HIP_KERNELS=0` and with the default, in processes of their own: largest difference (ESRGAN's RRDB kernels must give the same bits), ms per image before and after, and whether the kernels ran (spandrel's log lines; a warning from them means a kernel fell back, which fails) | `hip_kernels.py` | 2-5 min |
| Speed: each route at 256x256 and 512x512 (HAT, DAT2, VOSR at 256x256), ms per call over 50 calls after a warm-up, PSNR against PyTorch FP32 | `bench.py` | 15-20 min |
| The dependency manager uninstalls ONNX (PyTorch's ROCm packages must stay) and installs it again | `uninstall_check.sh` | 2 min |
| Report: `summary.md` (host, steps, test counts and failing tests, chains, speed and PSNR tables, versions) with the logs and chain outputs | `report.sh` | seconds |

A step that fails is reported and the next ones still run. The steps also run alone (each script's header says how),
after `setup.sh` once.

## The GUI in a browser (optional)

```
bash /root/chaiNNer-C/native/tools/amd/desktop.sh            # on the AMD machine
ssh -p <PORT> -L 6080:127.0.0.1:6080 root@<HOST>              # on this PC, then open http://127.0.0.1:6080/vnc.html
bash /root/chaiNNer-C/native/tools/amd/desktop.sh stop
```

It starts a virtual display with a light window manager, a VNC server and its web client, all reachable only through
that SSH tunnel, and chaiNNer-C in it.

## Updating the kit

The test files expected to carry `amd_gpu` tests, the wheels' URLs and the fallback ESRGAN are at the top of
`native/tools/amd/common.sh`. Before a session on an AMD machine, `native/tools/amd/selftest.sh --zip <LINUX_ZIP>`
checks the kit with the zip on any Linux machine with Docker, without an AMD GPU: the host check must stop, a dry run
must finish, and the real steps run on the CPU, where the PyTorch FP32 and NCNN chains must pass (it is also the only
check of chaiNNer-C with a CPU build of PyTorch) and the two spandrel checks run their code on the CPU against itself
(the architecture check on Compact, SPAN and DRFT), then beside a fake AMD GPU (`fake_gpu.sh`; `--gpu gfx1100` or
`gfx942`): the host check, the dependency manager's ROCm builds with that GPU's kernels alone, MIGraphX's nodes, ONNX
Runtime's MIGraphX provider and ROCm's tools must all work as on a real AMD machine, short of the GPU's own work. `setup.sh` copies MIGraphX's example
custom-op library (see the table above) from `--models` to `native/tools/amd/test-ops/`, where one runner test loads it;
the app does not ship it. The architecture check builds an architecture from the network class its spandrel loader
names, with that class's defaults; one without defaults, with large ones, or with more than one kind of network gets
small settings in `NETWORKS` at the top of `arch_check.py` (or a reason in `SKIPPED`), and a test
(`native/tests/test_amd_kit.py`) fails until a new spandrel architecture has one or the other. The kit runs every
`amd_gpu` test it finds in `backend/tests` and `native/tests`; the list only names files that must contribute. The
chains were made for the node versions of this commit; a chain whose node changed is rebuilt by saving it from the
editor with the same nodes (keep the node IDs the override files name).
