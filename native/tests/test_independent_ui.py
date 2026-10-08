"""The reviewed UI patcher refuses bundles and anchors it has not reviewed."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import independent_ui
import pytest

# The installed nightly is the pinned UI baseline (independent_ui.BASELINE_HASHES).
INSTALLED_APP = Path(
    os.environ["LOCALAPPDATA"], "chaiNNer", "app-0.25.1-nightly2025-10-21"
)
NODE = Path(r"C:\nvm4w\nodejs\node.exe")
needs_installed_app = pytest.mark.skipif(
    not INSTALLED_APP.is_dir(), reason="the installed chaiNNer baseline is absent"
)

# Just the main-bundle anchors the patcher requires, in the installed bundle's form.
MAIN_SETTINGS = (
    "const pH={checkForUpdatesOnStartup:!0};"
    'x={checkForUpdatesOnStartup:g("check-upd-on-strtup-2",!0)};'
    "y9=A=>{for(const g of vuA)A=g(A);return{...pH,...A}};"
)
UPSTREAM_PYTHON_CHECK = "if(gR.eq(s.version,I))return s}"
MAIN_EDITS = "".join(before for before, _ in independent_ui.MAIN_EDITS)
# The reviewed TensorRT patches' edits; the main-bundle anchors are the types'.
TRT_EDITS = [
    edit
    for name in independent_ui.TENSORRT_PATCHES
    for edit in independent_ui.tensorrt_patch(name)["edits"]
]
TRT_MAIN = "".join(e["before"] for e in TRT_EDITS if e["bundle"] == independent_ui.MAIN)


def patch_main(monkeypatch, python_check, edits=MAIN_EDITS):
    original = (MAIN_SETTINGS + python_check + edits + TRT_MAIN).encode()
    monkeypatch.setitem(
        independent_ui.BASELINE_HASHES,
        independent_ui.MAIN,
        hashlib.sha256(original).hexdigest(),
    )
    return independent_ui.independent_bundle(independent_ui.MAIN, original).decode()


@pytest.fixture(scope="module")
def patched_bundles():
    overrides, _ = independent_ui.build_ui_overrides(INSTALLED_APP)
    return {path: data.decode("utf-8") for path, data in overrides.items()}


def test_ui_patches_fail_closed_on_an_unknown_bundle(tmp_path):
    for path in independent_ui.BASELINE_HASHES:
        bundle = tmp_path / path
        bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle.write_bytes(b"checkForUpdatesOnStartup:!0 // not the reviewed bundle")
    with pytest.raises(ValueError, match="Unreviewed installed UI bundle"):
        independent_ui.build_ui_overrides(tmp_path)


def test_main_keeps_an_integrated_python_of_3_14_or_newer(monkeypatch):
    patched = patch_main(monkeypatch, UPSTREAM_PYTHON_CHECK)
    assert UPSTREAM_PYTHON_CHECK not in patched
    assert "gR.eq(s.version," not in patched
    assert patched.count('if(gR.gte(s.version,"3.14.0"))return s}') == 1


@needs_installed_app
def test_header_shows_the_product_version_without_the_alpha_badge(patched_bundles):
    renderer = patched_bundles[independent_ui.RENDERER]
    start = renderer.index(",llr=W.memo(")
    header = renderer[start : renderer.index(",clr=W.memo(", start)]
    assert header.count('children:"v0.3.2"') == 1
    assert header.count("_.jsx(") == 3  # logo, title and version only
    assert "Alpha" not in header
    assert "VTt" not in header
    assert renderer.count('children:"Alpha"') == 0
    assert renderer.count('["v",VTt]') == 0


@needs_installed_app
def test_no_upstream_update_behavior_remains(patched_bundles):
    renderer = patched_bundles[independent_ui.RENDERER]
    main = patched_bundles[independent_ui.MAIN]
    for forbidden in (
        "api.github.com",
        "/releases/latest",
        "/releases?per_page=",
        "Update Available (",
        'th("checkForUpdatesOnStartup")',
        "checkForUpdatesOnStartup:!0",
        "check-upd-on-strtup",
    ):
        assert forbidden not in renderer, forbidden
        assert forbidden not in main, forbidden
    assert "...A,checkForUpdatesOnStartup:!1}}" in main
    assert 'if(gR.gte(s.version,"3.14.0"))return s}' in main


@pytest.mark.parametrize("python_check", ["", UPSTREAM_PYTHON_CHECK * 2])
def test_main_patch_needs_the_python_check_exactly_once(monkeypatch, python_check):
    with pytest.raises(ValueError, match="exactly one reviewed UI anchor"):
        patch_main(monkeypatch, python_check)


@pytest.mark.parametrize("index", range(len(independent_ui.MAIN_EDITS)))
@pytest.mark.parametrize("times", [0, 2])
def test_main_patch_needs_each_edit_exactly_once(monkeypatch, index, times):
    befores = [before for before, _ in independent_ui.MAIN_EDITS]
    befores[index] *= times
    with pytest.raises(ValueError, match="exactly one reviewed UI anchor"):
        patch_main(monkeypatch, UPSTREAM_PYTHON_CHECK, "".join(befores))


def test_main_patch_fails_if_release_or_download_behavior_remains(monkeypatch):
    with pytest.raises(ValueError, match="behavior remains: Downloading integrated"):
        patch_main(
            monkeypatch,
            UPSTREAM_PYTHON_CHECK,
            MAIN_EDITS + '_g.info("Downloading integrated Python...")',
        )


@needs_installed_app
def test_main_shows_chainner_c_and_keeps_app_get_version(patched_bundles):
    main = patched_bundles[independent_ui.MAIN]
    for shown in (
        'title:"About chaiNNer",message:"chaiNNer v0.3.2",detail:"chaiNNer is an open',
        '_g.info("chaiNNer-C 0.3.2 (upstream 0.25.1-nightly.2025-10-21)"),eP(',
        'd={app:{version:"0.3.2",upstream:"0.25.1-nightly.2025-10-21",packaged:',
        '...xs?[]:[{label:"About chaiNNer",click:C}],{type:"separator"},',
    ):
        assert main.count(shown) == 1, shown
    assert "Release Notes" not in main
    # Saved chains keep the upstream stamp: xf is still app.getVersion().
    assert main.count("Wg.app.getVersion()") == 1
    assert main.count("xf=Wg.app.getVersion()") == 1


@needs_installed_app
def test_the_shown_upstream_version_must_be_the_installed_apps(monkeypatch):
    installed = json.loads(
        (INSTALLED_APP / independent_ui.PACKAGE_JSON).read_text(encoding="utf-8")
    )["version"]
    assert installed == independent_ui.UPSTREAM_VERSION == "0.25.1-nightly.2025-10-21"
    monkeypatch.setattr(independent_ui, "UPSTREAM_VERSION", "0.25.1-nightly2025-10-21")
    with pytest.raises(
        ValueError, match=f"Unreviewed installed app version {installed}"
    ):
        independent_ui.build_ui_overrides(INSTALLED_APP)


@needs_installed_app
def test_main_closes_the_integrated_python_redownload_branch(patched_bundles):
    main = patched_bundles[independent_ui.MAIN]
    start = main.index("xrA=async(A,g)=>{")
    assert main[start : main.index(",qrA=", start)] == (
        "xrA=async(A,g)=>{const B=A5(),{url:Q,version:I}=QP[B],C=IP(A);"
        'if(await KrA(C)){const s=await qU([C]);if(gR.gte(s.version,"3.14.0"))return s}'
        "throw new Error(\"chaiNNer-C's integrated Python is missing or older than "
        '3.14; reinstall chaiNNer-C")}'
    )


@needs_installed_app
@pytest.mark.skipif(not NODE.is_file(), reason="Node is absent")
def test_the_bundle_verifier_passes_the_patches_and_fails_upstreams_main(
    tmp_path, patched_bundles
):
    package = tmp_path / "package"
    for relative, text in patched_bundles.items():
        (package / relative).parent.mkdir(parents=True, exist_ok=True)
        (package / relative).write_bytes(text.encode("utf-8"))
    shutil.copyfile(
        INSTALLED_APP / "resources/app/package.json",
        package / "resources/app/package.json",
    )

    def verify():
        return subprocess.run(
            [
                str(NODE),
                str(
                    Path(independent_ui.__file__).with_name("verify_independent_ui.cjs")
                ),
                "--installed-app",
                str(INSTALLED_APP),
                "--package",
                str(package),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

    passed = verify()
    assert passed.returncode == 0, passed.stdout
    assert "PASS verify_independent_ui: 23 passed, 0 failed" in passed.stdout
    shutil.copyfile(INSTALLED_APP / independent_ui.MAIN, package / independent_ui.MAIN)
    failed = verify()
    assert failed.returncode == 1
    for name in (
        "main edits: each before once in installed",
        "integrated Python: 3.14.0 or newer kept",
    ):
        assert f"FAIL {name}" in failed.stdout, failed.stdout


@pytest.mark.parametrize("data", ["", "anchor anchor"])
def test_anchors_must_occur_exactly_once(data):
    with pytest.raises(ValueError, match="exactly one reviewed UI anchor"):
        independent_ui.replace_once(data, "anchor", "patched")


@needs_installed_app
def test_tensorrt_types_reach_both_scopes_the_accent_and_the_stylesheet(
    patched_bundles,
):
    for path in (independent_ui.RENDERER, independent_ui.MAIN):
        for definition in (
            "struct TensorRTEngine {",
            "enum TrtPrecision { fp32, fp16 }",
            "enum TrtShapeMode { fixed, dynamic }",
            "def convenientUpscaleTrt(engine: TensorRTEngine, image: Image) {",
        ):
            assert patched_bundles[path].count(definition) == 1, (path, definition)
    renderer = patched_bundles[independent_ui.RENDERER]
    assert renderer.count('color:fb("--type-color-tensorrt")') == 1
    css = patched_bundles[independent_ui.CSS]
    assert css.count("--type-color-tensorrt: #76b900}") == 1
    assert css.count("--type-color-tensorrt: #90c41c}") == 1


@needs_installed_app
def test_tensorrt_nodes_get_a_clear_item_and_removed_nodes_clear_their_cache(
    patched_bundles,
):
    renderer = patched_bundles[independent_ui.RENDERER]
    assert renderer.count("{backend:trtClearBackend}=We(Aa)") == 1
    assert (
        renderer.count(
            '(z==="chainner:tensorrt:upscale_image"||z==="chainner:tensorrt:load_engine")'
            "&&_.jsx(Or,{icon:_.jsx(Nv,{}),onClick:()=>{"
            "trtClearBackend.clearNodeCacheIndividual(d).catch(H2.error)},"
            'children:"Clear"})'
        )
        == 1
    )
    assert (
        renderer.count(
            "ne.forEach(I2=>{w.clearNodeCacheIndividual(I2).catch(H2.error)})},[G,g1,k,w])"
        )
        == 1
    )


def test_tensorrt_types_refuse_an_unpinned_baseline(monkeypatch, tmp_path):
    monkeypatch.setitem(independent_ui.BASELINE_HASHES, independent_ui.CSS, "0" * 64)
    with pytest.raises(ValueError, match="Unreviewed TensorRT patch baseline"):
        independent_ui.build_ui_overrides(tmp_path)


@pytest.mark.parametrize("index", range(len(TRT_EDITS)))
def test_tensorrt_types_need_each_anchor(index):
    edit = TRT_EDITS[index]
    with pytest.raises(ValueError, match="exactly one reviewed UI anchor"):
        independent_ui.tensorrt_types(edit["bundle"], edit["before"][:-1])
