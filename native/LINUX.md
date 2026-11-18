# Running chaiNNer-C on Linux

Download `chaiNNer-C-…-linux-x64.zip` from the [Releases page](https://github.com/Rift73/chaiNNer-C/releases); it
unzips to `chaiNNer-linux-x64/`. Building on Linux is in [`BUILDING.md`](BUILDING.md#linux).

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
