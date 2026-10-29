from __future__ import annotations

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

# vapoursynth-akarin and PySide6 (vsview's) publish no Linux wheels for an older glibc.
MIN_GLIBC = (2, 35)

# ffms2 and fmtconv (pulled in by vs-jetpack's deinterlace extra) have their Windows
# wheels only on JET's index.
JET_INDEX = "https://jaded-encoding-thaumaturgy.github.io/vs-wheels/simple"

package = add_package(
    __file__,
    id="chaiNNer_vapoursynth",
    name="VapourSynth",
    description=package_description,
    dependencies=[
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
        Dependency(
            display_name="vsview",
            pypi_name="vsview",
            version="0.13.0",
            size_estimate=707 * MB,
        ),
    ],
    icon="BsFilm",
    color="#D53F8C",
)


def _unsupported_reason(glibc: tuple[int, int] | None) -> str | None:
    if glibc is None or glibc >= MIN_GLIBC:
        return None
    return (
        f"VapourSynth needs glibc {MIN_GLIBC[0]}.{MIN_GLIBC[1]} or newer on Linux"
        " (vapoursynth-akarin and PySide6 publish no wheels for an older one);"
        f" this system has glibc {glibc[0]}.{glibc[1]}. Every other package works."
    )


_reason = _unsupported_reason(glibc_version())
if _reason is not None:
    package.disabled = True
    package.disabled_reason = _reason

vapoursynth_category = package.add_category(
    name="VapourSynth",
    description="Nodes that build, preview and run VapourSynth video scripts.",
    icon="BsFilm",
    color="#D53F8C",
    install_hint=install_hint,
)

logger.debug("Loaded package %s", package.name)
