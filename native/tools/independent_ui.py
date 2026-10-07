"""Remove upstream update behavior from the explicitly reviewed installed bundles.

They also show chaiNNer-C's version and keep its integrated Python (never
deleted or downloaded). The installed bundles are patched in place; src/ is the
run-from-source UI (upstream's, with its integrated Python moved to CPython
3.14.8) and never enters the package, so these narrow, hash-pinned transforms
are the package's only implementation of the independent edition. Unknown
baselines fail closed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

RENDERER = "resources/app/.vite/renderer/main_window/index.js"
MAIN = "resources/app/.vite/build/main.js"
CSS = "resources/app/.vite/renderer/main_window/index.css"
# The product version shown in the header. The package manifest's app_version
# stays the installed UI baseline (upstream package.json); this is display only.
PRODUCT_VERSION = "0.3.0"
BASELINE_HASHES = {
    RENDERER: "50f22a101c16c97fd1db413e3e0becf647c360b817f09b8a3089fa649fd16877",
    MAIN: "df979a26746d814f4dd8e4ce555f6114304a0b7f72cef429de783c646ad26f10",
    CSS: "2b33220c0429e43d9a11cea32cc32b5df6fb919d3cc36be7d074f8e5609c9f81",
}
# The installed main bundle keeps the integrated Python only while its version
# equals the table's 3.11.5 (QP win32); any other version deletes <root>/python
# and downloads 3.11.5. chaiNNer-C ships CPython 3.14, so any 3.14.0+ is kept.
PYTHON_CHECK_UPSTREAM = "if(gR.eq(s.version,I))return s}"
PYTHON_CHECK_PORT = 'if(gR.gte(s.version,"3.14.0"))return s}'
# The upstream base the version display names beside chaiNNer-C's (Consult 2 Q5):
# the pinned app's package.json version, checked equal to it at every patch, so
# the reviewed output never follows an unpinned file.
UPSTREAM_VERSION = "0.25.1-nightly.2025-10-21"
PACKAGE_JSON = "resources/app/package.json"
# What the patched main bundle logs at startup, and the error its integrated-Python
# check throws instead of upstream's delete and download (Q6); the real-use pass
# reads both in the package's log.
VERSION_LOG_LINE = f"chaiNNer-C {PRODUCT_VERSION} (upstream {UPSTREAM_VERSION})"
PYTHON_MISSING = (
    "chaiNNer-C's integrated Python is missing or older than 3.14; reinstall chaiNNer-C"
)
# The main bundle's further reviewed edits, (before, after), each anchor exactly
# once. Consult 2 Q5: every shown version is chaiNNer-C's, while app.getVersion()
# (xf) stays upstream's, since it stamps saved chains and their migrations
# compare it; the Release Notes item, which opens upstream's tag, goes. Q6: the
# fall-through after the integrated-Python check (delete <root>/python, download
# and extract 3.11.5) throws instead; upstream's catch in setup logs it and
# offers Retry, system Python or exit. Nothing is deleted or downloaded.
MAIN_EDITS = (
    (
        "message:`chaiNNer ${Wg.app.getVersion()}`",
        f'message:"chaiNNer v{PRODUCT_VERSION}"',
    ),
    (
        "_g.info(`chaiNNer Version: ${xf}`)",
        f'_g.info("{VERSION_LOG_LINE}")',
    ),
    (
        "app:{version:Wg.app.getVersion(),packaged:",
        f'app:{{version:"{PRODUCT_VERSION}",upstream:"{UPSTREAM_VERSION}",packaged:',
    ),
    (
        (
            '...xs?[]:[{label:"About chaiNNer",click:C}],{label:"Release Notes",'
            "click:async()=>{await Wg.shell.openExternal(`https://github.com/chaiNNer-org"
            "/chaiNNer/releases/tag/v${Wg.app.getVersion()}`)}},"
        ),
        '...xs?[]:[{label:"About chaiNNer",click:C}],',
    ),
    (
        (
            'const i=eB.resolve(eB.join(A,"/python"));await pe.rm(i,{recursive:!0,'
            "force:!0}),_g.info(`Integrated Python not found at ${C}`);"
            'const o="python.tar.gz",n=eB.join(A,o);if(_g.info("Downloading integrated '
            'Python..."),g(0,"download"),await new YrA({url:Q,directory:A,fileName:o,'
            'cloneFiles:!1,onProgress:s=>g(Number(s),"download")}).download(),'
            '_g.info("Extracting integrated Python..."),g(0,"extract"),await JrA(A,n,'
            's=>g(s,"extract")),_g.info("Removing downloaded files..."),await pe.rm(n),'
            'B==="linux"||B==="darwin"){_g.info("Granting permissions for integrated '
            'python...");try{await pe.chmod(C,4095)}catch(s){_g.warn(s)}}return qU([C])}'
        ),
        f'throw new Error("{PYTHON_MISSING}")}}',
    ),
)
# Upstream release and download behavior that must not survive in main.
MAIN_FORBIDDEN = (
    "Release Notes",
    "releases/tag",
    "chaiNNer Version: ",
    "Downloading integrated Python",
    "new YrA(",
)


def replace_once(data: str, before: str, after: str) -> str:
    if data.count(before) != 1:
        raise ValueError(f"Expected exactly one reviewed UI anchor: {before[:80]}")
    return data.replace(before, after, 1)


# The reviewed TensorRT patches, applied in this order: the types (third UI
# patch), then the node menu's Clear item and the removed nodes' cache clears
# (fourth).
TENSORRT_PATCHES = ("tensorrt-types.json", "tensorrt-clear.json")


def tensorrt_patch(name: str) -> dict:
    """A reviewed TensorRT patch, pinned to BASELINE_HASHES (build_ui_overrides)."""
    return json.loads(Path(__file__).with_name(name).read_text(encoding="utf-8"))


def tensorrt_types(path: str, data: str) -> str:
    for name in TENSORRT_PATCHES:
        for edit in tensorrt_patch(name)["edits"]:
            if edit["bundle"] == path:
                data = replace_once(data, edit["before"], edit["after"])
    return data


def independent_bundle(path: str, original: bytes) -> bytes:
    if hashlib.sha256(original).hexdigest() != BASELINE_HASHES[path]:
        raise ValueError(f"Unreviewed installed UI bundle; refusing to patch: {path}")
    data = original.decode("utf-8")
    if path == CSS:
        return tensorrt_types(path, data).encode("utf-8")
    data = replace_once(
        data, "checkForUpdatesOnStartup:!0", "checkForUpdatesOnStartup:!1"
    )
    if path == RENDERER:
        start = ',ln0="chaiNNer-org/chaiNNer"'
        end = ",clr=W.memo("
        if data.count(start) != 1 or data.count(end) != 1:
            raise ValueError("Update component boundaries changed")
        left, right = data.index(start), data.index(end)
        if right <= left:
            raise ValueError("Invalid update component boundaries")
        # The original logo and title, then the product version in place of the
        # Alpha badge and the upstream app version; no effects or updater UI.
        header = (
            ",llr=W.memo(()=>_.jsxs(j2,{children:["
            '_.jsx(E_,{boxSize:"36px",draggable:!1,src:elr}),'
            '_.jsx(Fv,{display:{base:"none",lg:"inherit"},size:"md",children:"chaiNNer"}),'
            '_.jsx(EL,{children:"v' + PRODUCT_VERSION + '"})]}))'
        )
        data = data[:left] + header + data[right:]
        data = replace_once(
            data,
            'const[e,a]=th("checkForUpdatesOnStartup"),[o,l]=th("experimentalFeatures")',
            'const[o,l]=th("experimentalFeatures")',
        )
        data = replace_once(
            data,
            '_.jsx(Fb,{setValue:a,setting:{label:"Check for Update on Start-up",'
            'description:"Toggles checking for updates on start-up."},value:e}),',
            "",
        )
        for forbidden in (
            "api.github.com",
            "/releases/latest",
            "/releases?per_page=",
            "Update Available (",
            'th("checkForUpdatesOnStartup")',
        ):
            if forbidden in data:
                raise ValueError(f"Upstream update behavior remains: {forbidden}")
        # Reviewed interaction repair; keep the installed renderer/toolchain and
        # fail closed if any baseline-specific anchor drifts or is duplicated.
        drop_edits = json.loads(
            Path(__file__).with_name("input-drop.json").read_text(encoding="utf-8")
        )
        if drop_edits["baseline_renderer_sha256"] != BASELINE_HASHES[RENDERER]:
            raise ValueError("Unreviewed drag/drop baseline")
        for edit in drop_edits["edits"]:
            data = replace_once(data, edit["before"], edit["after"])
    else:
        data = replace_once(
            data,
            'checkForUpdatesOnStartup:g("check-upd-on-strtup-2",!0)',
            "checkForUpdatesOnStartup:!1",
        )
        data = replace_once(
            data,
            "y9=A=>{for(const g of vuA)A=g(A);return{...pH,...A}}",
            "y9=A=>{for(const g of vuA)A=g(A);return{...pH,...A,checkForUpdatesOnStartup:!1}}",
        )
        data = replace_once(data, PYTHON_CHECK_UPSTREAM, PYTHON_CHECK_PORT)
        if PYTHON_CHECK_UPSTREAM in data or data.count(PYTHON_CHECK_PORT) != 1:
            raise ValueError("Integrated Python version check was not replaced once")
        for before, after in MAIN_EDITS:
            data = replace_once(data, before, after)
        for forbidden in MAIN_FORBIDDEN:
            if forbidden in data:
                raise ValueError(
                    f"Upstream release or download behavior remains: {forbidden}"
                )
    return tensorrt_types(path, data).encode("utf-8")


def build_ui_overrides(app: Path) -> tuple[dict[str, bytes], list[dict[str, str]]]:
    output = {}
    records = []
    for name in TENSORRT_PATCHES:
        if tensorrt_patch(name)["baseline_sha256"] != BASELINE_HASHES:
            raise ValueError(f"Unreviewed TensorRT patch baseline: {name}")
    for path, baseline_hash in BASELINE_HASHES.items():
        converted = independent_bundle(path, (app / path).read_bytes())
        output[path] = converted
        if path == CSS:
            reason = "TensorRT type colour tokens (light and dark)"
        elif path == RENDERER:
            reason = (
                "Independent edition: no upstream release requests, update UI or re-enabling through old settings"
                f"; v{PRODUCT_VERSION} header without the Alpha badge; reviewed Directory input drops and file/node lock guards"
                "; TensorRT types in the navi scope and the TensorRT accent colour"
                "; a Clear item for the TensorRT nodes and cache clears for removed nodes"
            )
        else:
            reason = (
                "Independent edition: no upstream release requests, update UI or re-enabling through old settings"
                "; integrated Python 3.14.0 or newer is kept, and a missing or older one is an error, never deleted or replaced by the upstream 3.11.5"
                f"; v{PRODUCT_VERSION} shown in About, the startup log and system information (with upstream {UPSTREAM_VERSION}), no Release Notes item; app.getVersion() unchanged"
                "; TensorRT types in the navi scope"
            )
        records.append(
            {
                "destination": path,
                "installed_sha256": baseline_hash,
                "converted_sha256": hashlib.sha256(converted).hexdigest(),
                "reason": reason,
            }
        )
    version = json.loads((app / PACKAGE_JSON).read_text(encoding="utf-8"))["version"]
    if version != UPSTREAM_VERSION:
        raise ValueError(
            f"Unreviewed installed app version {version}; the reviewed bundles show "
            f"upstream {UPSTREAM_VERSION}"
        )
    return output, records
