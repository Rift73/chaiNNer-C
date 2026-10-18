# VapourSynth in chaiNNer-C

VapourSynth is a video-processing engine driven by a short Python program, a script (`.vpy`): it opens a video as a
clip, passes it through filters that come from plugins (compiled add-ons) and names the clips it wants out. vs-jetpack
is a large collection of ready-made Python functions for it, from the Jaded Encoding Thaumaturgy (JET) group:
deinterlacing, denoising, scaling, masks, video loading and more. These nodes build such a script by wiring nodes
instead of writing code: each node writes one line. vsview, a separate viewer, shows the result (it succeeds the
archived vspreview).

## Install

Install the **VapourSynth** package in the dependency manager: VapourSynth 81, vs-jetpack 2.2.6 with its source,
deinterlacing, denoising and scaling plugins, MVTools 9, CAS, vszip CUDA (BM3D, NL-Means, DFTTest on NVIDIA GPUs),
DFTTest2 and vsview 0.13.0, about 1.75 GB. It goes into chaiNNer-C's own Python; a VapourSynth you installed yourself
is not used, changed or loaded. VapourSynth runs in its own processes (vspipe, its command-line runner, and vsview),
never in chaiNNer's, so a plugin crash costs only that process; the setting **VapourSynth cache (MB)** (default 4096)
caps each one's frame cache. Below, `<storage>` is the `backend-storage` folder of chaiNNer-C's own folder (where you
unzipped it; `%APPDATA%\chaiNNer-C` from source).

## The nodes

22 nodes in the groups Clips, Filters and Functions; 23 once you add functions of your own. Method counts are about.

| Node | What it does | Methods |
| --- | --- | ---: |
| Load Clip | Opens a video file: BestSource (default), another indexer or a plugin loader | 10 |
| Clip Frames, Frames to Clip | Clip to images and back, for chaiNNer's image nodes | |
| Split Planes | Splits a clip into Y, U, V (R, G, B for RGB) clips | |
| Preview Clips, Save Script | Opens up to 16 clips in vsview; writes the chain as a `.vpy` file | |
| Clip Properties, Convert Format | Colour info, aspect ratio, frame rate, frame statistics; format, bit depth, colour space | 15, 12 |
| Resize, Cut and Splice, Planes | Scale with any kernel, descale, crop, pad, turn; trim, join, replace and shift frames; split, join, merge planes | 39, 16, 21 |
| Deinterlace | Field matching (VFM), decimation, NNEDI3, Bwdif, fields | 25 |
| Denoise | BM3D, NL-Means, Motion-compensated degrain, DFTTest, prefilters | 29 |
| Sharpen and Blur, Anti-alias, Dehalo | Sharpening and blurring; jagged edges; halos | 26, 21, 9 |
| Deband and Grain, Masks, Colour, Expression | Banding and grain; masks; levels, tone mapping, CLAHE; per-pixel formulas | 16, 71, 4, 14 |
| AI Upscale | Runs an upscaling model on every frame (below) | |
| User Functions | Your own functions (below); shown only when there are some | |
| Any Function | Every other function, grouped by module (vs-jetpack's internal helper classes left out) | 192 |

## One chain, one script

Each node adds one statement. The script lists them in dependency order, each once even if several outputs use it and
tagged `# node:` so an error maps back to its node, then the named outputs. Preview Clips, Clip Frames and Save Script
compile the chain the same way, so what you preview is what Clip Frames renders and Save Script exports. A failing
statement is reported on its own node with the end of VapourSynth's message, and chaiNNer keeps running.

| Node | Writes |
| --- | --- |
| Preview Clips | `<storage>\vapoursynth\chains\<node id>\preview.vpy`, with `vsview.log` beside it |
| Clip Frames | `<storage>\vapoursynth\chains\<node id>\frames.vpy` |
| Frames to Clip | `frames-<id>.mkv` (16-bit) or `.nut` (32-bit float), a temporary video, in the same kind of folder |
| Save Script | `<Directory>\<Script Name>.vpy` |

`<node id>` is the random name chaiNNer gives each node; sort `chains` by date. When chaiNNer-C starts it deletes the
`frames-*` files there that are more than a day old (not the scripts or logs), so a saved script that reads one stops
working then. A script carries a path to chaiNNer-C's own code (AI Upscale needs it), so a saved script is for this
computer. Run it with chaiNNer-C's `vspipe.exe` (in its Python, `site-packages\vapoursynth`): a bare `vspipe` may find
another VapourSynth.

## Preview Clips and vsview

Preview Clips takes Clip 1 and, with the plus button, up to 15 more, each with an optional name. A run writes the
script, checks it with one trial evaluation, and opens vsview with each clip as an output under its name (a tab). If the
check fails, the error shows on its node and no window opens. Each Preview Clips node has one window. While it is open
a run only rewrites the script: press **Reload Script** (Ctrl+R) in vsview. After you close it the next run opens a new
one (so does a restart of chaiNNer-C). vsview keeps its settings in `%LOCALAPPDATA%\vsview`.

## Clip Frames and Frames to Clip

**Clip Frames** (a loop, like Load Video) outputs Frame, Index and FPS (0 if the rate varies). Gray clips give gray
images; any other clip becomes RGB (YUV by its matrix, else by its size as Load Video; add Convert Format before it to choose).
**Mode** (as Load Video's) Planes gives a YUV clip's Y, U and V planes at their stored sizes instead (Y 0 to 1 from black
to white, U and V centred on 0.5, the clip's range and bit depth undone; float clips as they are); Both gives the RGB
frames and the planes from one render of the chain. **Colour Info** comes from the first frame's `_Matrix`, `_Transfer`,
`_Primaries`, `_Range` and `_ChromaLocation`; one that is not set is guessed as Load Video guesses it. Reading them
renders the first frame once more when the run starts.
Every frame needs one size and format. **Frames to Clip** ends a loop of RGB or gray images, with an FPS, and saves
them as a file that BestSource reads as a clip; the next run replaces it. **Precision** 16-bit (default) rounds the
values to 16 bits, clipped to 0-1, in a lossless FFV1 video (`.mkv`, an RGB48 or Gray16 clip); 32-bit float keeps every
value, also outside 0-1, uncompressed (`.nut`, an RGBS or GrayS clip): 12 bytes per pixel, gray frames too (about
25 MB per 1080p frame; the log says the size). **Input** Planes takes Y, U and V planes at their own sizes (as Load
Video's Planes mode gives them; 4:4:4, 4:2:2 or 4:2:0, the Y plane a multiple of the subsampling) and an optional
Colour Info, and makes a 16-bit YUV clip tagged with it (the clip's `_Matrix`, `_Transfer`, `_Primaries`, `_Range` and
`_ChromaLocation`); each colour dropdown other than Auto replaces the Colour Info's value, and without Colour Info Auto
guesses from the size as Load Video does for untagged video.

## Family nodes

A family node is one task, such as Denoise. Its **Method** dropdown lists the family's functions and the settings below
it are the picked method's. Functions with a plain name come first under "Common"; a dropdown with more than 12 entries
is searchable. Node search finds a family by its methods: typing `bm3d`, `scale` or `mc_degrain` offers "Denoise: BM3D"
and the like, with that method picked. The **Inputs** tabs give **Simple** (the main settings, plainly named) or
**Advanced** (everything). A setting left at its function's default is not written. A function that exists in many
variants (every scaling kernel, edge mask or prefilter) is one method with a dropdown ("Kernel").

**Use as filter** gives the method itself instead of its result, for a setting of another node that wants a function:
the Clip input goes away and the output becomes "Clip or filter". Example: Deinterlace > Field match (VFM) has a
"Post-process" setting; add a second Deinterlace node with Method NNEDI3 deinterlace and Use as filter on, and wire it
in. That is `vfm(postprocess=NNEDI3(...).deinterlace)`: two nodes, one wire. Denoise > Prefilter does the same for
Motion-compensated degrain's Prefilter. **Any Function** has up to 12 rows `name = value` (a Python value: a number,
'text', True, a list; never code) and two sockets for a clip or filter, each with the argument name it fills.

A function that returns several clips (Shift Clip Multi, SplitPlanes) gives them as one output: wire it into a setting that takes several clips (Interleave, Splice, Stack, Expression's clips); a setting that takes one clip says so on its node.

## AI Upscale

AI Upscale runs a model on every frame inside the script with chaiNNer's own code: PyTorch models, ONNX models and
TensorRT engines, with Backend, Precision (FP32 or FP16, PyTorch only) and Tile Size as in Upscale Image. A frame comes
out as Upscale Image would make it (tested identical on the CPU). PyTorch and ONNX use the first NVIDIA GPU if there is
one, else the CPU.

- The clip must be 32-bit float RGB or gray (RGBS or GRAYS) of one size: use Convert Format before it (Convert (Point)
  starts at RGBS) and usually after it. Each frame is clipped to 0-1 before the model, as chaiNNer's images always are.
- A model loads once per file when the script is evaluated (the output size needs its scale), and nodes share it. Some
  functions read frame properties while the script is built (a linear-light Scale, say), which computes the first
  frame then. So the check, vsview's start and each reload wait for it (over a minute on a CPU with several AI Upscale
  nodes), and Reload Script loads the models again (about 2 s for a small one). One model call runs at a time.

## Adding a function you found

Put a `.py` file in `<storage>\vapoursynth\user` (create the folder). Each public function in it (no leading
underscore) that returns a clip, or has no return annotation, appears in **User Functions** as "Name (file)"; for
example `def brighten(clip: vs.VideoNode, gain: float = 1.1) -> vs.VideoNode` in `mine.py`. Settings are typed from the
annotations; one that cannot be typed is a text box taking a Python value. Without annotations, the first parameter
without a default is the clip: annotate a function whose first parameter is not a clip. Or list pip packages in
`<storage>\vapoursynth\user_modules.toml`, as `modules = ["vs-colorfix"]`: missing ones are installed when chaiNNer-C
starts (this one brings about 300 MB of plugins; an install that fails or takes over 15 minutes is skipped) and their
functions appear the same way; its colour match lands under Colour. pip installs a listed module's own dependencies into
chaiNNer-C's Python as well, which may change packages that chaiNNer-C uses.

An optional **overlay** gives a function a plain name and a node. Put a `.toml` in `<storage>\vapoursynth\overlays` (yours
win over the shipped ones); a function's id is `<file or module>.<function>`, such as `vsdenoise.bm3d`:

```toml
[method."mine.brighten"]
label = "Brighten"
family = "Colour"                  # one of the node names above
labels = { gain = "Gain" }         # plain names for settings
defaults = { gain = 1.2 }          # what the input starts at, and what the node writes
```

Other keys: `help`, `simple` (the settings in Simple), `hidden`, `annotations`, `class_default`. A function with an
overlay is listed under "Common". A file may not be named `sys`, `vs`, `core` or `set_output` (names chaiNNer's scripts
use), like a clip variable (`n_<word>_<number>`), like an installed module or like one of chaiNNer's own (`util`,
`system`, `api`...); it is skipped with a line in the log. The
function list is read when chaiNNer-C starts (Settings > Restart chaiNNer) and after a dependency install: it scans
again (a few seconds) if your files, `user_modules.toml`, the plugins or the VapourSynth or vs-jetpack versions changed,
and re-reads the overlays.

## Limits and known gaps

- vs-jetpack's automatic backend choice takes a GPU plugin whenever one is installed, and the function fails if CUDA is
  not usable (the error mentions `cuInit`). Choose a CPU backend where the function has the setting: BM3D OLD (VapourSynth
  81 refuses the API 3 plugin bm3dcpu that the installer pulls in), NL-Means ISPC, DFTTest CPU. Prefilter has no such
  setting: for a CPU prefilter use DFTTest with Backend CPU and Use as filter, which differs from vs-jetpack's own
  prefilter (that one also leaves dark areas out).
- AI Upscale: TensorRT has not been run on a GPU yet (2026-10-15). It ignores chaiNNer's PyTorch and ONNX settings (GPU
  choice, memory limit), offers no BF16, and takes gray through the nodes' BT.601 luma.
- Any Function calls a class's method without constructor settings. An error in an import line shows on the node that
  runs the script, a bad filter setting on the node it is wired into.
- Do not upgrade MVTools: the dependency manager accepts a newer version, but 10 breaks Motion-compensated degrain.
