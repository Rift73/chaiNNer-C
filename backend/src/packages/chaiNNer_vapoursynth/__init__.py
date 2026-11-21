from __future__ import annotations

import sys

from sanic.log import logger

from api import KB, MB, Dependency, add_package
from system import glibc_version

package_description = (
    "VapourSynth filters video with scripts. These nodes build a VapourSynth script"
    " from vs-jetpack's functions, the installed plugins and your own Python"
    " functions, preview it in vsview, save it, or turn its frames into images."
    " VapourSynth runs in separate processes, never inside chaiNNer's own."
)

install_hint = (
    "VapourSynth video filtering with vs-jetpack, its plugins and the vsview previewer."
)

# PyPI's Linux wheels of vapoursynth-akarin (vs-jetpack needs 1.4.1 or newer) need glibc
# 2.35; those of vapoursynth-nlm-ispc (vs-jetpack's deinterlace extra) and of PySide6
# 6.11 and newer (vsview's floor) need 2.34.
AKARIN_PYPI_GLIBC = (2, 35)
NLM_ISPC_PYPI_GLIBC = (2, 34)
VSVIEW_GLIBC = (2, 34)

# ffms2 and fmtconv (pulled in by vs-jetpack's deinterlace extra) have their Windows
# wheels only on JET's index.
JET_INDEX = "https://jaded-encoding-thaumaturgy.github.io/vs-wheels/simple"

# The same akarin release built for glibc 2.28 (manylinux_2_28, LLVM 22.1.8 linked
# statically as in PyPI's wheel), for the systems PyPI's wheel excludes.
AKARIN_GLIBC_2_28 = Dependency(
    display_name="akarin",
    pypi_name="vapoursynth-akarin",
    version="1.5.0",
    size_estimate=71 * MB,
    url=(
        "https://github.com/Rift73/vapoursynth-akarin/releases/download/"
        "v1.5.0-glibc2.28/vapoursynth_akarin-1.5.0-py3-none-"
        "manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
        "#sha256=a4f70733405a2ebdc10c7050d7716fbe6a71fdbf0b1482ef28687dae6474f22c"
    ),
)

# The same NL-Means ISPC release built for glibc 2.28 (manylinux_2_28, ISPC 1.32.0 as in
# PyPI's wheel), for the systems PyPI's wheel excludes.
NLM_ISPC_GLIBC_2_28 = Dependency(
    display_name="NL-Means ISPC",
    pypi_name="vapoursynth-nlm-ispc",
    version="4.1",
    size_estimate=1.1 * MB,
    url=(
        "https://github.com/Rift73/vapoursynth-nlm-ispc/releases/download/"
        "v4.1-glibc2.28/vapoursynth_nlm_ispc-4.1-py3-none-"
        "manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
        "#sha256=9ae1dc5cb56b9feb261ef07871a2cd7a544b1020ce4c9f8df4344b908133f707"
    ),
)

# Deband and Grain's Grainer needs the noise plugin (github.com/wwww-wwww/vs-noise, GPL-3.0)
# for all its types but Placebo, and vs-jetpack does not require it: the plugin's own wheels
# on PyPI, by hash.
NOISE_WHEELS = {
    "win32": (
        "https://files.pythonhosted.org/packages/e9/b2/"
        "74caf326050f8ae03d8a6e24f810d5cca36f31f18dba5a77eb0c26a8eb02/"
        "vsnoise-1.3-py3-none-win_amd64.whl"
        "#sha256=39201f4c0ed11d4d3f44e4106a039831ae1e72b57a1fa45a313a13369267d779"
    ),
    "linux": (
        "https://files.pythonhosted.org/packages/e3/86/"
        "497a531cc75061b609790542de3816cf61e8a902201aa0df909b75f583f7/"
        "vsnoise-1.3-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
        "#sha256=35cc606ef085cd140d62108023f30c16e9dddcea91f23c074cf798db520d66db"
    ),
}

# Denoise's rainbow and dot crawl removers, which vs-jetpack does not wrap: each plugin's
# own wheels on PyPI, by hash. Bifrost (github.com/dubhatervapoursynth/vapoursynth-bifrost,
# WTFPL) and DotKill (github.com/myrsloik/DotKill, MIT), both 8-16 bit integer YUV.
BIFROST_WHEELS = {
    "win32": (
        "https://files.pythonhosted.org/packages/d5/01/"
        "863a55724fd35b175e571b20946736717852e7c8ed6ed7a10462c9f76a5b/"
        "vapoursynth_bifrost-4.0-py3-none-win_amd64.whl"
        "#sha256=8d30dd3c99b09b71c470486440a2143d02a01f05eb5d9760126289692b9dc7fa"
    ),
    "linux": (
        "https://files.pythonhosted.org/packages/8c/e1/"
        "a40673957ecfb50254a34ce4f172d3f3e602785bdc5a461a683479279f00/"
        "vapoursynth_bifrost-4.0-py3-none-"
        "manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
        "#sha256=f0144ea7596aa0bfcd9780da0a8979a3a150b4552bd8364d93111be090ba52b9"
    ),
}
DOTKILL_WHEELS = {
    "win32": (
        "https://files.pythonhosted.org/packages/8b/6c/"
        "61785800126669c46c067f27d001e7315e72550f80401b5bdbde0df0faef/"
        "vapoursynth_dotkill-4.0-py3-none-win_amd64.whl"
        "#sha256=1d367651a252d0c70c5a539f8775d351e8e3e3d33a2cd1f4c1ce140a35edc3d3"
    ),
    "linux": (
        "https://files.pythonhosted.org/packages/34/be/"
        "acae6b4886127dfad619b90da880d1246666f4199e553f72eacccdffbef0/"
        "vapoursynth_dotkill-4.0-py3-none-"
        "manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
        "#sha256=cb23808a9a5b7bb82fe593f09e23830a0c4f5c142d1789b801394cdfd5bd3a5b"
    ),
}
# DeDot (github.com/dubhatervapoursynth/vapoursynth-dedot, GPL-2.0-only), 8-16 bit integer
# YUV or gray. VapourSynth loads it in its own processes; chaiNNer-C neither links nor ships
# it.
DEDOT_WHEELS = {
    "win32": (
        "https://files.pythonhosted.org/packages/cd/1f/"
        "9dd3d68a2f7ac76b4f3f349a04e49e596b7dd1565dc29e77381c972b810e/"
        "vapoursynth_dedot-4.0-py3-none-win_amd64.whl"
        "#sha256=a85330f71d47cde7b3381f0e141c2d764b195d9c9f8990c9feae9f068c5fb03b"
    ),
    "linux": (
        "https://files.pythonhosted.org/packages/06/f7/"
        "6764a8974f1d5b65f92bb81201aec5e6cb5005f01c4640408cf6508b449a/"
        "vapoursynth_dedot-4.0-py3-none-manylinux2014_x86_64."
        "manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl"
        "#sha256=0a386e6ede263dfa55eb58c622d13fcd50c7e8c0f51d3a81342715deb1b0ed8e"
    ),
}


def vsview_unavailable_reason(glibc: tuple[int, int] | None) -> str | None:
    """The reason the package leaves vsview out on a system with this glibc, or None
    (also where there is no glibc): PySide6 6.11 and newer publish no Linux wheels for a
    glibc older than 2.34, and its older releases declare Python 3.14 unsupported."""
    if glibc is None or glibc >= VSVIEW_GLIBC:
        return None
    return (
        "vsview is not available on this system: its Qt (PySide6 6.11 or newer) needs"
        f" glibc {VSVIEW_GLIBC[0]}.{VSVIEW_GLIBC[1]} or newer, and this system has glibc"
        f" {glibc[0]}.{glibc[1]}. The other VapourSynth nodes work, and Save Script writes"
        " a script that another VapourSynth previewer can show."
    )


def vapoursynth_dependencies(glibc: tuple[int, int] | None) -> list[Dependency]:
    """The package's dependencies on a system with this glibc (None where there is
    none): below PyPI's floors for akarin and NL-Means ISPC, the glibc 2.28 builds of
    the releases that vs-jetpack would take from PyPI; vsview only where its Qt
    installs."""
    dependencies = [
        Dependency(
            display_name="VapourSynth",
            pypi_name="VapourSynth",
            version="81",
            size_estimate=99 * MB,
        ),
        # The extra brings the source, deinterlacing, denoising and scaling plugins
        # (and vapoursynth-bm3dcpu, which R81 refuses as an API 3 plugin; harmless).
        Dependency(
            display_name="vs-jetpack",
            pypi_name="vsjetpack[deinterlace]",
            version="2.2.6",
            size_estimate=580 * MB,
            extra_index_url=JET_INDEX,
        ),
        # 10 dropped the fields/tff arguments that vs-jetpack 2.2.6's mc_degrain passes.
        Dependency(
            display_name="MVTools",
            pypi_name="vapoursynth-mvutensils",
            version="9",
            size_estimate=45 * MB,
        ),
        Dependency(
            display_name="CAS",
            pypi_name="vapoursynth-cas",
            version="3.0",
            size_estimate=400 * KB,
        ),
        Dependency(
            display_name="Noise",
            pypi_name="vsnoise",
            version="1.3",
            size_estimate=500 * KB,
            url=NOISE_WHEELS.get(sys.platform),
        ),
        Dependency(
            display_name="Bifrost",
            pypi_name="vapoursynth-bifrost",
            version="4.0",
            size_estimate=700 * KB,
            url=BIFROST_WHEELS.get(sys.platform),
        ),
        Dependency(
            display_name="DotKill",
            pypi_name="vapoursynth-dotkill",
            version="4.0",
            size_estimate=700 * KB,
            url=DOTKILL_WHEELS.get(sys.platform),
        ),
        Dependency(
            display_name="DeDot",
            pypi_name="vapoursynth-dedot",
            version="4.0",
            size_estimate=30 * KB,
            url=DEDOT_WHEELS.get(sys.platform),
        ),
        # The CUDA BM3D, NLMeans and DFTTest that vs-jetpack prefers on NVIDIA GPUs.
        Dependency(
            display_name="vszip CUDA",
            pypi_name="vapoursynth-vszipcu",
            version="1.2.0",
            size_estimate=15 * MB,
        ),
        Dependency(
            display_name="DFTTest2",
            pypi_name="dfttest2[cpu,cufft,nvrtc]",
            version="11.0.0",
            size_estimate=478 * MB,
        ),
    ]
    if glibc is not None and glibc < AKARIN_PYPI_GLIBC:
        dependencies.append(AKARIN_GLIBC_2_28)
    if glibc is not None and glibc < NLM_ISPC_PYPI_GLIBC:
        dependencies.append(NLM_ISPC_GLIBC_2_28)
    if vsview_unavailable_reason(glibc) is None:
        dependencies.append(
            Dependency(
                display_name="vsview",
                pypi_name="vsview",
                version="0.13.0",
                size_estimate=707 * MB,
            )
        )
    return dependencies


package = add_package(
    __file__,
    id="chaiNNer_vapoursynth",
    name="VapourSynth",
    description=package_description,
    dependencies=vapoursynth_dependencies(glibc_version()),
    icon="BsFilm",
    color="#D53F8C",
)


vapoursynth_category = package.add_category(
    name="VapourSynth",
    description="Nodes that build, preview and run VapourSynth video scripts.",
    icon="BsFilm",
    color="#D53F8C",
    install_hint=install_hint,
)

logger.debug("Loaded package %s", package.name)
