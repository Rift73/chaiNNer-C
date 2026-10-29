# VOSR 2.0 in chaiNNer-C

VOSR 2.0 is a one-step generative super-resolution model: a diffusion transformer (DiT) restores the enlarged image in
a single step, guided by DINOv2 features of the input. Model and paper: Rongyuan Wu, Lingchen Sun, Zhengqiang
Zhang, Xiangtao Kong, Jixin Zhao, Shihao Wang and Lei Zhang, "VOSR: A Vision-Only Generative Model for Image
Super-Resolution" (arXiv:2604.03225, [github.com/cswry/VOSR](https://github.com/cswry/VOSR)); weights `CSWRY/VOSR` on
Hugging Face, Apache-2.0. Maintainer notes: [`VOSR2-MAINTAINING.md`](VOSR2-MAINTAINING.md).

## Get the model

chaiNNer-C uses one model: the official VOSR 2.0 upload on Hugging Face (`CSWRY/VOSR` at revision `f24b3061`,
Apache-2.0). The first upscale downloads its five files, about 7 GB (6.97 GB), into the backend's storage folder at
`vosr2-model/f24b3061b7f350b81e1907bdcfc27f71fb4ff3f3/` (`args.json`, `checkpoints/`, `Qwen-Image-vae-2d/`,
`dinov2_vitl14_pretrain.pth`), with progress on the node. A stopped or broken download resumes on the next run; each
file must match its pinned SHA-256 (a mismatch is deleted and reported). Files already there are hashed once and never
downloaded again. On a GPU, the PyTorch backend's first load keeps a BF16 copy of the DiT beside it
(`checkpoints/ema_model.bf16.safetensors`, 2.8 GB) so later loads skip converting its 5.6 GB of FP32 weights. The first upscale also downloads OpenCV's YuNet face detector (230 KB, MIT licence, `opencv_zoo` at
`26cc381e`, `face_detection_yunet_2026may.onnx`) into `vosr2-faces/`, the same way.

## The nodes

**Upscale Image (VOSR)** (PyTorch > Processing) takes an RGB Image (no alpha) and:

| Input | Values | Effect |
| --- | --- | --- |
| Scale | 1 to 8, default 4 | Exact output size (the input is enlarged with 8-bit bicubic first) |
| Seed | | Chooses the generated detail; same seed, settings, backend: same output |
| Color Correction | Wavelet (default), AdaIN, None | Wavelet: the input's coarse colours (Gaussian, sigma 5); AdaIN: channel mean and spread |
| GPU Memory, Backend | Default, Low (under 7 GB); PyTorch, TensorRT, TensorRT FP8, MIGraphX, MIGraphX FP8 | MIGraphX runs Low on PyTorch |
| Temporal Method, Strength | Three Frames, Persistent Detail; 0 to 100 %, default 65 | Only when Load Video feeds the image; 0 %: each frame alone |
| Texture Fix | On (default), off | The final grain pass below; off keeps the model's own output in flat areas |

The model works in overlapping 512-pixel windows, each also predicted with a 64-pixel strip of its surroundings; the
input's detail decides the mix, so blurred areas stay soft (scene focus). Neighbouring windows invent different
amounts of detail, so their seams can show: windows are placed so that the seams run through the least detailed parts
of the picture (as many windows as a regular grid), and a face that a seam would still split gets a window of its own
that alone gives the face's middle (up to 400 px across) and fades into its neighbours over 48 px (about 50 ms each
at 4K output, at most a fifth of the windows). In a video the layout is planned at a shot's first
frame and kept until a cut, so seams stay put. A final pass (Texture Fix) gives flat grainy areas one fine
grain across windows, as strong as the input's and in its colours. In a video, areas whose fine texture changed since
the previous frame get new grain, and still areas keep one grain, so still backgrounds do not shimmer. For alpha: Split
Transparency, Resize (Lanczos) the alpha, Merge Transparency.

## Video

In a **Load Video** loop the node stabilizes frames automatically: each frame is restored once, the neighbours'
restorations are aligned to it by optical flow and blended in only where motion and colours agree, and a cut starts
afresh. **Three Frames** uses the previous and next frame. **Persistent Detail** also carries stable fine texture from
earlier frames and starts each scene from up to five frames (a slower first output). Keep the seed and settings for a
whole video (a change is an error); frames need one size, 16 to 32,766 pixels per side. Load Images loops run per frame.

Frames ahead come from chaiNNer-C's read-ahead (5 to 7 by default; Persistent Detail needs 5). When a node between
Load Video and this one is not read ahead (CPU image and utility nodes and TensorRT Upscale Image are; a PyTorch upscale
is not), only past frames are used: Three Frames takes the previous frame alone, Persistent Detail starts each scene
from its first frame, and the log says `using past frames only`.

## Backends and memory

Every backend computes in BF16, with FP32 where precision matters (VAE, normalization, residuals); CPU mode uses FP32
and is very slow. The GPU must support BF16. **PyTorch** needs nothing more; on AMD GPUs it runs on the ROCm build of
PyTorch (Linux and Windows).

- **TensorRT** needs an NVIDIA GPU, the TensorRT package (dependency manager) and VOSR's plugin `vosr2_qkv.dll`
  (included when the build made it; otherwise the node says so). The plugin has code for RTX 30, 40 and 50 GPUs; only
  an RTX 5090 is tested. The first run builds ten engines: about 3 minutes, 5.5 GB of free GPU memory, 3.5 GB on disk.
  A TensorRT, driver, GPU or chaiNNer-C update can rebuild them once; the replaced engines are deleted once unused for
  two weeks (another GPU's are kept).
- **TensorRT FP8** needs an RTX 40 series or newer and the official, unchanged model files. FP8 (8-bit floating point)
  runs the DiT's projections and feed-forward layers and most of DINOv2; DiT attention and the VAE stay BF16/FP32. First
  use downloads the calibrated graphs (3.45 GB from `steropes.feralhosting.com`, SHA-256 checked, resumed after Stop)
  and builds engines; 5.2 GB on disk. Faster, slightly lossy.
- **MIGraphX** is TensorRT's counterpart for AMD GPUs on Linux and Windows: the ROCm PyTorch, chaiNNer-C's MIGraphX
  runner and VOSR's op library `vosr2_migx_ops.so` (Windows: `.dll`). DINO and the DiT run as MIGraphX programs
  built from the same graphs; the VAE runs in PyTorch (its shapes follow the image, and MIGraphX compiles fixed
  shapes). The first run builds four programs (the DiT twice, once per window width, each with its own weights);
  every AMD GPU ROCm 7.14 supports, none tested yet.
- **MIGraphX FP8** has MIGraphX quantize the same graphs where TensorRT FP8 is FP8 (the DiT's projections and
  feed-forward layers, DINOv2's products) on GPUs with FP8 math (Instinct MI300 and MI350, Radeon RX 9000 series),
  calibrated on built-in synthetic images (the TensorRT FP8 set's format is one MIGraphX 2.16.1 cannot read). On
  other GPUs it is MIGraphX, or FP16 where BF16 is slow (Radeon RX 5000 and 6000 series, Instinct MI100). Faster,
  lossy; nothing to download. Both MIGraphX options show only on Linux and Windows with an AMD GPU and no NVIDIA
  GPU.

**Default** may use the whole GPU except other programs' use and 800 MB. **Low** keeps the GPU total, other programs
included, under 7 GB by streaming the DiT weights: slower; TensorRT streams its engines' weights, MIGraphX cannot, so
MIGraphX runs Low on PyTorch. Weights and engines stay loaded until the node
is deleted, unless Force CUDA Cache Wipe or a Memory Budget Limit is set (PyTorch settings).

## Measured speed

RTX 5090, Ryzen 9 9950X3D, 640×360 → 2560×1440 (4×), GPU Memory Default, a 30-frame clip through Load Video
(2026-10-13). GPU memory is the peak above the desktop's own use.

| Case | Each later frame | First output | GPU memory |
| --- | ---: | ---: | ---: |
| PyTorch, Three Frames | 4.60 s | 11.2 s | +10.1 GB |
| PyTorch, Persistent Detail | 4.95 s | 30.9 s | +10.6 GB |
| TensorRT, Three Frames | 1.40 s | 12.2 s (198 s with the engine build) | +13.7 GB |

About 1.1 s per frame is CPU stabilization, overlapped on TensorRT. One image in a new backend, with loading:
6.3 s (PyTorch), 9.4 s (TensorRT). FP8's first use: 191 s (build 177 s; download about 28 s at 1 Gbit/s). FP8 earlier
that day, same graphs: 50 warm frames 1920×1080 → 3840×2160 in 122.6 s, TensorRT 154.1 s, no file I/O or stabilization.

## Limits

- Generative: detail is invented, not recovered; text, faces and stair-stepped outlines can change. Scene focus can
  take low-contrast detail for blur; the grain pass turns photographic noise into fine grain of the same strength. CPU
  RAM grows with the output size.
- Stabilization is post-processing, not a video model: it reduces flicker without removing it; fast motion,
  reflections, transparency and changing light defeat the optical flow.
- TensorRT restores up to 3840 × 2176 pixels in one piece, larger outputs in tiles with their own scene context.
  FP8 against BF16 TensorRT: 50.0 and 42.5 dB PSNR on two 2560×1440 images.
