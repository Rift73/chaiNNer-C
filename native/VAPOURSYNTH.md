# VapourSynth in chaiNNer-C

VapourSynth is a video-processing engine driven by a short Python program, a script (`.vpy`): it opens a video as a
clip, passes it through filters that come from plugins (compiled add-ons) and names the clips it wants out. vs-jetpack
is a large collection of ready-made Python functions for it, from the Jaded Encoding Thaumaturgy (JET) group:
deinterlacing, denoising, scaling, masks, video loading and more. These nodes build such a script by wiring nodes
instead of writing code: each node writes one line. vsview, a separate viewer, shows the result (it succeeds the
archived vspreview).

## Install

Install the **VapourSynth** package in the dependency manager: VapourSynth 81, vs-jetpack 2.2.6 with its source,
deinterlacing, denoising and scaling plugins, MVTools 9, CAS, the noise plugin (Deband and Grain's Grainer), vszip CUDA
(BM3D, NL-Means, DFTTest on NVIDIA GPUs),
DFTTest2 and vsview 0.13.0, about 1.75 GB. It goes into chaiNNer-C's own Python; a VapourSynth you installed yourself
is not used, changed or loaded. Uninstall removes it with what pip pulled in for it that nothing else installed needs
(vsview's PySide6, the plugins); NumPy, SciPy and what chaiNNer-C or your user modules need stay. VapourSynth runs in its own processes (vspipe, its command-line runner, and vsview),
never in chaiNNer's, so a plugin crash costs only that process; the setting **VapourSynth cache (MB)** (default 4096)
caps each one's frame cache. Below, `<storage>` is the `backend-storage` folder of chaiNNer-C's own folder (where you
unzipped it; `%APPDATA%\chaiNNer-C` from source).

## Start here

Every chain starts with **Load Clip** (Open a video (BestSource)), not Load Video: the filter nodes work on a whole
clip, and most cannot take single frames. Then the filter nodes, and at the end one of:

- **Preview Clips**: a Run renders nothing; it opens vsview, which computes each frame when you scrub to it. The first
  Run on a new video waits while BestSource indexes the file once. After a change, Run again and the open vsview
  reloads by itself; if it does not, press Reload Script (Ctrl+R).
- **Clip Frames**: the frames as images, for Save Image or Save Video (use its own Index and FPS). Its seek bar fills
  after the first Run; Run this frame renders that one frame through the whole chain, correct after field matching.
- **Save Script**: the chain as a `.vpy` file.

Routes, as node > Method:

- Detelecine (film telecined to 29.97 fps): Deinterlace > Detelecine: film back to 24 fps (detelecine). Its two
  steps also exist apart: Detelecine step 1: field match (VFM), then Detelecine step 2: drop repeats (VDecimate).
- Wrong levels (too dark and contrasty, or washed out): Convert Format > Range fix: full or limited (range_fix).
- Deinterlace: Deinterlace > Deinterlace, fast (Bwdif), or Deinterlace, best and slow (QTempGaussMC).
- Denoise: Denoise > Denoise, gentle (NL-Means), or Denoise, strong (BM3D).
- Descale (undo an upscale): Resize > Descale: undo an upscale (descale). Half or double size: Resize > Scale by a
  factor (0.5 = half size).
- AI upscale: Load Clip > Convert Format, Convert format and colours (Point) to RGBS > AI Upscale > Convert Format >
  Clip Frames.
- Before and after: wire the source and the result into Preview Clips (Clip 1 and Clip 2); vsview shows each as a tab.

To look at one frame of a filtered clip, use Clip Frames' seek bar and Run this frame: it renders that frame through
the whole chain, neighbouring frames included where a filter needs them. Images turned into a clip by Frames to Clip
under Run this frame make a clip of one frame instead, which field matching, decimation and temporal denoising cannot
work on (Frames to Clip warns of it).

## The nodes

22 nodes in the groups Clips, Filters and Functions; 23 once you add functions of your own. Method counts are about.

| Node | What it does | Methods |
| --- | --- | ---: |
| Load Clip | Opens a video file: BestSource (default), another indexer or a plugin loader | 10 |
| Clip Frames, Frames to Clip | Clip to images and back, for chaiNNer's image nodes | |
| Split Planes | Splits a clip into Y, U, V (R, G, B for RGB) clips | |
| Preview Clips, Save Script | Opens up to 16 clips in vsview; writes the chain as a `.vpy` file | |
| Clip Properties, Convert Format | Colour info, aspect ratio, frame rate, frame statistics; format, bit depth, colour space, range fix | 15, 13 |
| Resize, Cut and Splice, Planes | Scale with any kernel or by a factor, descale, crop, pad, turn; trim, join, replace and shift frames; split, join, merge planes | 40, 16, 21 |
| Deinterlace | Detelecine in one step, field matching (VFM), decimation, NNEDI3, Bwdif, fields | 26 |
| Denoise | BM3D, NL-Means, Motion-compensated degrain, DFTTest, prefilters | 29 |
| Sharpen and Blur, Anti-alias, Dehalo | Sharpening and blurring; jagged edges; halos | 26, 21, 9 |
| Deband and Grain, Masks, Colour, Expression | Banding and grain; masks; levels, tone mapping, CLAHE; per-pixel formulas | 16, 71, 4, 14 |
| AI Upscale | Runs an upscaling model on every frame (below) | |
| User Functions | Your own functions (below); shown only when there are some | |
| Any Function | Every other function, grouped by module (vs-jetpack's internal helper classes left out) | 192 |

chaiNNer's own methods: Convert Format's Range fix (Compress (Full to Limited) or Expand (Limited to Full) rescales the
values and leaves the range flag as it is; a float clip comes back unchanged), Deinterlace's Detelecine (VFM with NNEDI3
on leftover combing, then VDecimate: 30000/1001 to 24000/1001) and Resize's Scale by a factor (sizes rounded to whole
chroma samples).

## One chain, one script

Each node adds one statement. The script lists them in dependency order, each once even if several outputs use it and
tagged `# node:` so an error maps back to its node, then the named outputs. Preview Clips, Clip Frames and Save Script
compile the chain the same way, so what you preview is what Clip Frames renders and Save Script exports. A failing
statement is reported on its own node with the end of VapourSynth's message, and chaiNNer keeps running.

| Node | Writes |
| --- | --- |
| Preview Clips | `<storage>\vapoursynth\chains\<node id>\preview.vpy`, with `vsview.log` and `vsview.pid` beside it |
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
script, checks it with one trial evaluation, and opens vsview with each clip as an output under its name (a tab); a
clip without a name takes the name of its node's Method entry (Denoise, strong (BM3D), say), or Clip 2 and so on when
two clips share it. The window title lists the tab names. If the check fails, the error shows on its node, no window
opens, and an open window keeps the last script that worked. Each Preview Clips node has one window. While it is open, also after chaiNNer-C restarts, a run
rewrites the script and the window reloads it within a second, as **Reload Script** (Ctrl+R) does: frame, tab and zoom
stay. After you close it the next run opens a new one. If vsview ends within three seconds of its start, the node
reports it with the end of `vsview.log`. vsview keeps its settings in `%LOCALAPPDATA%\vsview`.

vsview has no interface for reloading from outside, so `backend/src/nodes/impl/vapoursynth/viewer.py` starts it and
uses a few of its internals, listed in that file. If one is missing after a vsview update, vsview runs as before (press
Ctrl+R) and `vsview.log` says that auto-reload is off and why.

## Clip Frames and Frames to Clip

**Clip Frames** (a loop, like Load Video) outputs Frame, Index and FPS (0 if the rate varies). Gray clips give gray
images; any other clip becomes RGB (YUV by its matrix, else by its size as Load Video; add Convert Format before it to choose).
**Mode** (as Load Video's) Planes gives a YUV clip's Y, U and V planes at their stored sizes instead (Y 0 to 1 from black
to white, U and V centred on 0.5, the clip's range and bit depth undone; float clips as they are); Both gives the RGB
frames and the planes from one render of the chain. **Colour Info** comes from the first frame's `_Matrix`, `_Transfer`,
`_Primaries`, `_Range` and `_ChromaLocation`; one that is not set is guessed as Load Video guesses it. Reading them
renders the first frame once more when the run starts.
**Seek**, **Frames** (All or Range, Start to End) and **Run this frame** work as Load Video's. VapourSynth computes only
the frames asked for, each through the whole chain, so a Range or a single frame after field matching and decimation
equals the same frame of a full run, and **Index** is the frame's number in the clip. The seek bar spans **Frame Count**
from the last run, so it is empty until the first; it stays off for a variable frame rate (FPS 0). The **Preview** is
the Seek frame in RGB (a YUV matrix RGB cannot convert shows the Y plane in gray), rendered in the run's first
evaluation of the chain; Run this frame in Mode RGB uses that render as its frame, one vspipe start in all (a Range or
All: two).
Every frame needs one size and format. **Frames to Clip** ends a loop of RGB or gray images, with an FPS, and saves
them as a file that BestSource reads as a clip; the next run replaces it. **Precision** 16-bit (default) rounds the
values to 16 bits, clipped to 0-1, in a lossless FFV1 video (`.mkv`, an RGB48 or Gray16 clip); 32-bit float keeps every
value, also outside 0-1, uncompressed (`.nut`, an RGBS or GrayS clip): 12 bytes per pixel, gray frames too (about
25 MB per 1080p frame; the log says the size). **Input** Planes takes Y, U and V planes at their own sizes (as Load
Video's Planes mode gives them; 4:4:4, 4:2:2 or 4:2:0, the Y plane a multiple of the subsampling) and an optional
Colour Info, and makes a 16-bit YUV clip tagged with it (the clip's `_Matrix`, `_Transfer`, `_Primaries`, `_Range` and
`_ChromaLocation`); each colour dropdown other than Auto replaces the Colour Info's value, and without Colour Info Auto
guesses from the size as Load Video does for untagged video.

## Images in the nodes

A method that looks only at the current frame also takes an image at its **Clip** input (Load Image, or per item under
Load Images) and then gives an image, ready for Save Image or any image node; each other clip input of that method then
takes an image too. Methods that use other frames (Motion-compensated degrain, Field match, Temporal Median, Cut and
Splice...) take clips only: while one is picked in Method the Clip input does not connect to an image, and picking one
with an image connected shows an error on the node. Split Planes (R, G, B of an RGB image, as gray images) and AI Upscale
take images too (Upscale Image is the faster way for that); Clip Frames, Frames to Clip, Preview Clips and Save Script
take clips.

- **Which methods.** Measured, not guessed: on a clip of different frames, frame n's output (pixels and frame properties)
  must stay the same when every other frame changes, the clip must keep its length, and the function must run on a clip of
  one frame whose result makes an image and is the result of the same frame among others (vs-jetpack's SangNom antialias on
  one frame does what it does on a clip's last frame, so it takes clips only; a function that changes with the frame number
  or the length, such as grain, stays). The result is `frame_methods.json` beside the catalog code, for vs-jetpack 2.2.6 and
  VapourSynth 81; under other versions the log says so and Clip takes clips only, as it does for functions the file does not
  list. `native/tools/vs_frame_probe.py` measures it again. A method with a Class dropdown takes images only if none of its
  classes uses other frames, so Denoise > Prefilter takes clips only (one class, FluxSmooth, is temporal). A method that did
  not run on such a clip with no GPU and its default settings (it wants a field order, a model file you download, integer
  samples, or a GPU it cannot do without) takes clips only too: Based AA, Field match, Vine Dehalo. Some plugins crash on tiny
  images: Median Blur, Remove Grain, Repair and zsmooth's Median, Interquartile Mean and Smart Median refuse an image under
  4x4 by name. Your own function takes images when an overlay says `images = true`, which is your statement that it looks at
  one frame.
- **A setting that switches other frames on.** A method that uses other frames only through a temporal radius (`tr` of
  BM3D, NL-Means and DFTTest, `temporal_radius` of Soothe) takes an image, and on an image that setting is 0, whatever it
  says: one image has no other frames, and repeating the frame would change the result for nothing.
- **The image as a clip.** The values go in as they are, 32-bit float: an RGB image is an RGBS clip and a gray image a
  GRAYS clip, of one frame, tagged sRGB, BT.709 primaries, full range, progressive. An unfiltered round trip returns every
  value unchanged. vs-jetpack's functions often filter only the first plane, meaning luma, which on RGB would be red: those
  (measured: the result stays the same whatever green and blue are, or they refuse RGB) run on the image as YUV444PS (BT.709)
  and back to RGB. A mask method that would mask each YUV plane runs on the image's luma alone and gives a gray mask. A plane
  number in the Planes node means what it says: R, G or B. A result in another format comes back as RGB by its `_Matrix`, else
  BT.709; a gray result stays gray. A result centred on 0 (Get U, Get V, Make Diff and the like) would lose everything below 0
  in an image, so those methods take clips only. A result with NaN or infinity among its values is an error on the node, not
  an image (NL-Means on the CPU makes them on float samples).
- **Transparency.** A clip has no alpha. The method runs on the colour and the alpha stays exactly as it is. If the method
  moves pixels (the Resize node's methods, flips and turns too) or changes the image's size, it also runs on the alpha, as
  a gray image. A gray result has no alpha. A second image must be RGB or gray.
- **Speed.** One process per chain run evaluates the images' scripts one after another, so VapourSynth, its plugins, the
  vs-jetpack modules and AI Upscale's models load once per run, not per image. It ends with the run or at Stop. A plugin's
  crash costs that process and the image's error, as a vspipe's would; the next image starts a new one.

## Family nodes

A family node is one task, such as Denoise. Its **Method** dropdown lists the family's functions and the settings below
it are the picked method's. Functions with a plain name come first under "Common"; a dropdown with more than 12 entries
is searchable. Node search finds a family by its methods: typing `bm3d`, `scale` or `mc_degrain` offers "Denoise: BM3D"
and the like, with that method picked. The **Inputs** tabs give **Simple** (the main settings, plainly named) or
**Advanced** (everything). A setting left at its function's default is not written. A function that exists in many
variants (every scaling kernel, edge mask or prefilter) is one method with a dropdown ("Kernel"). A setting that takes
several kinds of value shows the main kind; Advanced adds "<Setting> (other values)", a Python value, for the others (`(2, 2)` for a
yes/no prefilter that also takes a radius pair, `[1, 2, 2]` for per-plane modes), which wins when filled. An expression
setting takes one expression for every plane or a list with one per plane (`['x 2 *', '']`).

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
one, else the CPU. On AMD GPUs, PyTorch runs on ROCm and ONNX on MIGraphX, which compiles the model for each tile
size it meets (the edge tiles' sizes too) without the ONNX settings' program cache; TensorRT engines need NVIDIA.

- The clip must be 32-bit float RGB or gray (RGBS or GRAYS) of one size: use Convert Format before it (Convert (Point)
  starts at RGBS) and usually after it. Each frame is clipped to 0-1 before the model, as chaiNNer's images always are.
- Frame properties carry over. A model that scales width and height differently (some ONNX models) changes the sample
  aspect ratio (`_SARNum`/`_SARDen`) with them, so the picture keeps its shape.
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
`<storage>\vapoursynth\user_modules.toml`, as `modules = ["vs-colorfix"]`, each with a version spec if you want one
(`"vs-colorfix>=2"`, passed to pip as written): missing ones, and ones whose installed version the spec does not allow,
are installed when chaiNNer-C starts and after an install in the dependency manager (VapourSynth's first), with the
progress naming each (this one brings about 300 MB of plugins; an install that fails or takes over 15 minutes is
skipped and tried again at the next start), and their functions appear the same way; its colour match lands under
Colour. pip installs a listed module's own dependencies into chaiNNer-C's Python as well, which may change packages
that chaiNNer-C uses.

An optional **overlay** gives a function a plain name and a node. Put a `.toml` in `<storage>\vapoursynth\overlays` (yours
win over the shipped ones); a function's id is `<file or module>.<function>`, such as `vsdenoise.bm3d`:

```toml
[method."mine.brighten"]
label = "Brighten"
family = "Colour"                  # one of the node names above
labels = { gain = "Gain" }         # plain names for settings
defaults = { gain = 1.2 }          # what the input starts at, and what the node writes
```

Other keys: `help`, `simple` (the settings in Simple), `hidden`, `annotations`, `class_default`, `images` (true: it looks
only at the current frame, so it takes images; false: it does not), `hide = true` (the function is in no node),
`rank` (a whole number). A function with a `rank`
is listed under "Common", ranked ones first by rank, and a new node starts on the first; any other key except the
texts (`help`, `docs`, `init_docs`, `choices`) lists it there too, after them, when it comes from an overlay of
yours (the shipped overlays list functions by `rank` only). So an overlay of yours with only `help` documents a
function without listing it there.

Texts are Markdown, shown as pop-ups when the pointer rests on the setting or the Method entry: `help` (the method),
`docs = { gain = "..." }` (settings), `init_docs` (a class's constructor settings, else `docs`), `choices =
{ Catrom = "..." }` (the entries of a Kernel or other class dropdown). A setting without one shows the shipped glossary's
text of its name (`[param."planes"]` tables with `doc` and an optional `annotation` the setting's type must contain)
when the entry has `override = true`, else the function's docstring, else the glossary's text; an entry with
`hidden = true` leaves the setting out (vs-jetpack's error-message `func`). A file may not be named `sys`, `vs`, `core` or `set_output` (names chaiNNer's scripts
use), like a clip variable (`n_<word>_<number>`), like an installed module or like one of chaiNNer's own (`util`,
`system`, `api`...); it is skipped with a line in the log. The
function list is read when chaiNNer-C starts (Settings > Restart chaiNNer) and after a dependency install: it scans
again (a few seconds) if your files, `user_modules.toml`, the plugins or the VapourSynth or vs-jetpack versions changed,
and re-reads the overlays. A scan that takes over two minutes is stopped, and the log names the file or module it was
importing: a file's import must not wait (move slow work into its functions).

## Linux

The same package works on Linux x86-64; paths read `<storage>/vapoursynth` and the like, and `vspipe` is the file
`site-packages/vapoursynth/vspipe` of chaiNNer-C's Python (its plugins are `.so` files in `plugins/`).

- **Installing.** The pip wheels of CPython 3.14 exist for everything the package lists, but PyPI's wheels of three
  of them need a newer C library than the rest of chaiNNer-C: vs-jetpack's `vapoursynth-akarin` wants glibc 2.35
  (Ubuntu 22.04, Debian 12, Fedora and newer), its `vapoursynth-nlm-ispc` and vsview's PySide6 glibc 2.34. Below
  those, the dependency manager installs the same releases built for glibc 2.28 (manylinux_2_28) in place of PyPI's:
  akarin 1.5.0 below glibc 2.35 (release `v1.5.0-glibc2.28` of the fork `Rift73/vapoursynth-akarin`, LLVM 22.1.8 as in
  PyPI's wheel) and NL-Means ISPC 4.1 below 2.34 (release `v4.1-glibc2.28` of `Rift73/vapoursynth-nlm-ispc`, ISPC
  1.32.0); their filters give the same frames as PyPI's builds. vsview is left out below glibc 2.34 (Rocky/Alma/RHEL 8,
  Ubuntu 20.04, Debian 10 and 11): PySide6 6.11 and newer, which it needs, have no wheels there, and the older ones
  declare Python 3.14 unsupported. VapourSynth's own programs also need the system's `libstdc++.so.6`, which every
  desktop distribution has.
- **Configuration.** vspipe needs a line naming the Python it runs scripts in; `vapoursynth config` cannot write it for
  chaiNNer-C's Python (its sysconfig names the machine that built it), so chaiNNer-C writes it itself, in
  `vsconfig/vapoursynth/vapoursynth.toml` inside its Python's folder, and again when the Python is moved or replaced.
  Nothing goes to `~/.config/vapoursynth`.
- **CUDA plugins.** vszip CUDA and DFTTest2's CUDA and NVRTC need an NVIDIA driver (`libcuda.so.1`); without one they
  do not load.
- **Preview Clips** needs a desktop session (X11 or Wayland), Qt's `libxcb-cursor0` and glibc 2.34 or newer (vsview).
  Below glibc 2.34 or without a session the node says so; Save Script writes the script for another previewer. If
  vsview ends within three seconds of its start, the node reports it with the end of `vsview.log`, which has Qt's
  reason. vsview keeps its settings in `~/.config/vsview`.

## Limits and known gaps

- vs-jetpack's automatic backend choice takes a GPU plugin whenever one is installed, and the function fails if CUDA is
  not usable (the error mentions `cuInit`). Choose a CPU backend where the function has the setting: BM3D OLD (VapourSynth
  81 refuses the API 3 plugin bm3dcpu that the installer pulls in), NL-Means ISPC, DFTTest CPU. Prefilter has no such
  setting: for a CPU prefilter use DFTTest with Backend CPU and Use as filter, which differs from vs-jetpack's own
  prefilter (that one also leaves dark areas out).
- AI Upscale ignores chaiNNer's PyTorch and ONNX settings (GPU
  choice, memory limit), offers no BF16, and takes gray through the nodes' BT.601 luma.
- Any Function calls a class's method without constructor settings. An error in an import line shows on the node that
  runs the script, a bad filter setting on the node it is wired into.
- Do not upgrade MVTools: the dependency manager accepts a newer version, but 10 breaks Motion-compensated degrain.
- Grainer's Strength starts at 0, which adds no grain. Its noise plugin (vsnoise 1.3, GPL-3.0) has two faults of its own.
  Poisson has no float path, so Grainer's Poisson runs a float clip or an image through 16 bits (values outside 0..1
  clip there); for Poisson a higher Strength means less grain. Perlin, Simplex and FBM Simplex on RGB take the green and
  blue planes' seed from past the end of the plugin's seed list, so a `seed` does not repeat them there. Any Function's
  `noise.Add` is the plugin as it is and takes clips only: its type 4 on float samples gives what the frame's memory held.
  Placebo does not use this plugin: libplacebo needs a Vulkan device, and without one the VapourSynth process ends.
  An install without the plugin (one that has not been updated) gets the plugin's name in the error of these five types.
- BM3D's backends other than Old take only 32-bit float, so Denoise's BM3D (`chainner_vs.bm3d`) runs another clip
  through 32-bit float and back to its own format, as Convert Format's Bit depth (depth) does both ways, and gives
  a Search or Reference clip 32-bit float too. A 32-bit float clip and the Old backend go to vs-jetpack's BM3D as they are.
- What always fails with vs-jetpack 2.2.6 is left out: base classes in Class and kernel dropdowns (Kernel, Scaler,
  Descaler...), MVTools' compensate, degrain, flow and mask methods (they need an analyze on the same object first; use
  Motion-compensated degrain), functions that need an object no input gives, and methods that give no clip. Another
  vs-jetpack version shows them all.
