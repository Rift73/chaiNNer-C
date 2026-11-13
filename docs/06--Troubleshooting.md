# Troubleshooting

chaiNNer-C keeps its settings, logs and integrated Python in its own folder: the folder you unzipped it to, or
`%APPDATA%\chaiNNer-C` when run from source. Its Python is the `python` folder there.

## After installing PyTorch, the nodes still do not show up

- Ensure you have at least 3GB of free RAM before starting chaiNNer-C.

- Uninstall PyTorch, restart chaiNNer-C, and reinstall it.

## PyTorch is failing to install from the dependency manager

- Make sure you have nothing blocking pip requests: the dependency manager downloads PyTorch from the PyTorch index
  (NVIDIA, CUDA 13.2) or from AMD's index (ROCm 10.0).

- Uninstall PyTorch in the dependency manager, restart chaiNNer-C, and install it again.

## Has Nvidia GPU, PyTorch using CPU

- Check the dependency manager and ensure it installed the `+cu132` version. If it did not, and it says CUDA supported at the top, then uninstall and reinstall PyTorch.

- Check to make sure you have the `nvidia-smi` command available to you. Open a terminal (command prompt, powershell, windows terminal) and type `nvidia-smi`. If it errors, reinstall your drivers and restart your computer.

- If chaiNNer-C or a terminal still isn't able to use `nvidia-smi`, try [adding your System32 folder to your PATH environment variable](https://www.computerhope.com/issues/ch000549.htm). Make sure to restart your computer after.

## Has AMD/Intel GPU, PyTorch using CPU

- PyTorch uses NVIDIA GPUs and, through ROCm 10.0, the AMD GPUs ROCm supports. On another GPU, use NCNN as the processing framework instead. You can convert most supported PyTorch models to NCNN through chaiNNer-C (just be sure to also install ONNX as it is needed in the conversion process).

## Checked logs and see `CUDA_INITIALIZE` errors

- Ensure your drivers are up to date. Outdated drivers might not support the CUDA version that PyTorch and ONNX rely on.

- Ensure your GPU isn't too old: PyTorch's CUDA 13 build needs an NVIDIA GPU from the Turing generation (RTX 20 and GTX 16 series) or newer. chaiNNer-C runs only on its integrated Python, so it cannot use an older PyTorch installed in a system Python.

## vkQueueSubmit error with NCNN

- If you are using an auto-tiling mode, try setting a manual tile size. The lower the number the more likely it will work.

- If you are on Windows, you can try using [this](https://github.com/chaiNNer-org/chaiNNer/issues/913#issuecomment-1247849063) fix suggested by one of the NCNN devs.

## "ChaiNNer was unable to install its integrated Python environment" message or simply fails to start up

- Ensure you have access to the internet and that you did not restrict chaiNNer-C from accessing the internet. It needs internet access in order to download its Python environment and the required dependencies.

- If you do have internet access, delete the `python` folder in chaiNNer-C's folder (above). This forces chaiNNer-C to download Python again if the previous download was corrupt.

- chaiNNer-C cannot use a system Python instead: its native modules need its integrated CPython 3.14. A "use system Python" setting saved by upstream chaiNNer is switched off.

## Integrated Python downloaded, but failed to start up after installing dependencies

- Windows 7 and macOS are not supported; use Windows 10 or 11, or Linux x86-64.
