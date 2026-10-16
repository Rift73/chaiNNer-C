"""The VapourSynth catalog in chaiNNer's worker: keeps `<storage>\\vapoursynth\\catalog.json`
current (scan.py rescans in a child process when what it lists changed), installs the
user's listed modules first, and turns the catalog and the overlays into Methods with
their families and stable input ids. VapourSynth itself is never imported here.

Storage folder (`<storage>\\vapoursynth`, from CHAINNER_VS_DIR): catalog.json, user\\*.py
(user functions), user_modules.toml (`modules = ["vs_colorfix"]`, pip names) and
overlays\\*.toml. Overlays are TOML tables `[method."<id>"]` with `label`, `family`,
`simple` (parameter names), `labels` (parameter -> plain name), `help`, `hidden`
(parameters left out, keeping their defaults), `annotations` (parameter -> annotation
text, e.g. a catalog enums key; a name the function does not have becomes one more
keyword argument of its **kwargs), `defaults` (parameter -> the value its input
starts with: a number, a text or a dropdown choice's Python source, which the node
then writes) and, for a group, `class_default` (the class or member preselected in its
dropdown, whose key is "class" in `simple` and `labels`); shipped overlays load first,
the user's win. A method with an overlay is listed under "Common" in its node.

Calls and enum members are module-qualified (`import vsdenoise`, then
`vsdenoise.bm3d(...)`; catalog_types.qualified), so that no two imports of a script
shadow each other; a function of the user folder's files also names that folder, which
the script puts on sys.path.

Groups: the same method of GROUP_MIN or more classes of one module
(`Mitchell().scale`, `Catrom().scale`, ...) is one method "<module>.*.<name>" with a
class dropdown, and an enum whose members filter a clip (`Prefilter.DFTTEST(clip)`)
is one method with a member dropdown."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import tomllib
from sanic.log import logger

from dependencies.store import ENV as PIP_ENV
from dependencies.store import PIP_REASON_LINES

from .catalog_types import Method, Param, qualified
from .runner import (
    _API3_NOTICE,
    STDERR_TAIL_LINES,
    VSScriptError,
    vs_env,
    vs_python,
    vspipe,
)
from .scan import SCHEMA, user_module_names

SCAN_PY = Path(__file__).with_name("scan.py")
SCAN_TIMEOUT = 120
# The longest a user module's pip install may take at startup (s).
PIP_TIMEOUT = 900
SHIPPED_OVERLAYS = Path(__file__).with_name("overlays")
CATALOG = "catalog.json"
USER_DIR = "user"
MODULES_FILE = "user_modules.toml"
OVERLAY_DIR = "overlays"

# Input ids below this are the family nodes' fixed inputs (0 main clip, 1 Method,
# 2 Simple/Advanced, 3 Use as filter).
FIRST_INPUT_ID = 1000
# Classes of one module sharing a method that become one method with a class dropdown.
GROUP_MIN = 5
# A group's dropdown, as a parameter name in `simple`, `labels` and input ids.
CLASS = "class"
# Overlay keys a group passes on to each member.
_MEMBER_KEYS = ("hidden", "annotations", "labels", "defaults")
_ID_MASK = 0x7FFFFFFF
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# The family nodes, in menu order.
FAMILIES: tuple[str, ...] = (
    "Load Clip",
    "Clip Properties",
    "Convert Format",
    "Resize",
    "Cut and Splice",
    "Planes",
    "Deinterlace",
    "Denoise",
    "Sharpen and Blur",
    "Anti-alias",
    "Dehalo",
    "Deband and Grain",
    "Masks",
    "Colour",
    "Expression",
    "User Functions",
    "Any Function",
)

# The family of a function of a module that spans many families (vstools, std, ...);
# the ones not listed go to Any Function. A leftover moves here only when its family
# node has no friendlier form of it (the plugins vs-jetpack wraps stay in Any Function).
FUNCTION_FAMILIES: dict[str, str] = {
    **dict.fromkeys(
        [
            "vstools.change_fps",
            "vstools.initialize_clip",
            "vstools.finalize_clip",
            "vstools.Sar.apply",
            "vstools.merge_clip_props",
            "vstools.sc_detect",
            "std.SetFrameProps",
            "std.SetFrameProp",
            "std.RemoveFrameProps",
            "std.CopyFrameProps",
            "std.AssumeFPS",
            "std.SetFieldBased",
            "std.PlaneStats",
            "std.ClipToProp",
            "std.PropToClip",
        ],
        "Clip Properties",
    ),
    **dict.fromkeys(
        ["vstools.depth", "vstools.limiter", "vstools.match_clip", "std.Limiter"],
        "Convert Format",
    ),
    **dict.fromkeys(
        [
            "vstools.Padder.mirror",
            "vstools.Padder.repeat",
            "vstools.Padder.color",
            "vstools.Padder.crop",
            "std.Crop",
            "std.CropAbs",
            "std.CropRel",
            "std.AddBorders",
            "std.FlipHorizontal",
            "std.FlipVertical",
            "std.Transpose",
            "std.Turn90",
            "std.Turn180",
            "std.Turn270",
        ],
        "Resize",
    ),
    **dict.fromkeys(
        ["std.DoubleWeave", "std.SeparateFields", "bwdif.Bwdif"], "Deinterlace"
    ),
    **dict.fromkeys(
        [
            "vstools.replace_ranges",
            "vstools.insert_clip",
            "vstools.replace_every",
            "vstools.remap_frames",
            "vstools.shift_clip",
            "vstools.shift_clip_multi",
            "std.Trim",
            "std.Splice",
            "std.DeleteFrames",
            "std.DuplicateFrames",
            "std.BlankClip",
            "std.SelectEvery",
            "std.Reverse",
            "std.Loop",
            "std.Interleave",
            "std.FreezeFrames",
        ],
        "Cut and Splice",
    ),
    **dict.fromkeys(
        [
            "vstools.split",
            "vstools.join",
            "vstools.plane",
            "vstools.get_y",
            "vstools.get_u",
            "vstools.get_v",
            "vstools.get_r",
            "vstools.get_g",
            "vstools.get_b",
            "vstools.stack_planes",
            "vstools.stack_clips",
            "std.Merge",
            "std.MaskedMerge",
            "std.ShufflePlanes",
            "std.SplitPlanes",
            "std.StackHorizontal",
            "std.StackVertical",
            "std.MakeDiff",
            "std.MergeDiff",
            "std.MakeFullDiff",
            "std.MergeFullDiff",
        ],
        "Planes",
    ),
    **dict.fromkeys(
        ["std.Convolution", "std.BoxBlur", "std.Median"], "Sharpen and Blur"
    ),
    **dict.fromkeys(
        [
            "std.Binarize",
            "std.BinarizeMask",
            "std.InvertMask",
            "std.Prewitt",
            "std.Sobel",
            "std.Minimum",
            "std.Maximum",
            "std.Inflate",
            "std.Deflate",
        ],
        "Masks",
    ),
    **dict.fromkeys(
        ["std.Levels", "std.Invert", "placebo.Tonemap", "vszip.CLAHE"], "Colour"
    ),
    **dict.fromkeys(["std.Expr", "std.Lut", "std.Lut2"], "Expression"),
}

# vs-jetpack's internal helpers, building blocks for writing functions rather than
# filters, and duplicates (padder_ctx is Padder): in no node.
HIDDEN_FUNCTIONS = frozenset(
    {
        *(
            f"vstools.{cls}.{name}"
            for cls in (
                "ProcessVariableFormatClip",
                "ProcessVariableResClip",
                "ProcessVariableResFormatClip",
            )
            for name in ("eval_clip", "get_clip", "normalize", "process")
        ),
        "vstools.AsyncRenderConf.get_clip",
        "vstools.FunctionUtil.return_clip",
        "vstools.Timecodes.assume_vfr",
        "vstools.VideoPackets.apply_props",
        *(f"vstools.padder_ctx.{m}" for m in ("COLOR", "CROP", "MIRROR", "REPEAT")),
    }
)

# The family of everything else from a vs-jetpack module or a plugin namespace.
MODULE_FAMILIES: dict[str, str] = {
    **dict.fromkeys(
        ["vssource", "bs", "ffms2", "lsmas", "d2v", "dvdsrc2"], "Load Clip"
    ),
    "resize": "Convert Format",
    "vskernels": "Resize",
    "vsscale": "Resize",
    "vsdeinterlace": "Deinterlace",
    "vsdenoise": "Denoise",
    "vsrgtools": "Sharpen and Blur",
    "cas": "Sharpen and Blur",
    "vsaa": "Anti-alias",
    "vsdehalo": "Dehalo",
    "vsdeband": "Deband and Grain",
    "vsmasktools": "Masks",
    "vsexprtools": "Expression",
    "akarin": "Expression",
}


def storage_dir() -> Path | None:
    """The VapourSynth storage folder (`<storage>\\vapoursynth`, which the server names
    in CHAINNER_VS_DIR), or None outside the server."""
    value = os.environ.get("CHAINNER_VS_DIR")
    return Path(value) if value else None


def pip_command() -> list[str]:
    """chaiNNer's own pip, as the dependency manager runs it."""
    return [str(vs_python()), "-m", "chainner_pip"]


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def install_user_modules(names: Iterable[str]) -> None:
    """Installs the listed pip packages that are missing. A failed install, or one that
    takes longer than PIP_TIMEOUT, is logged and skips that module only (the scan
    lists it under errors)."""
    for name in names:
        if _version(name) is not None:
            continue
        logger.info("Installing the VapourSynth user module %s with pip.", name)
        try:
            done = subprocess.run(
                [
                    *pip_command(),
                    "install",
                    name,
                    "--disable-chainner_pip-version-check",
                    "--no-warn-script-location",
                    "--no-cache-dir",
                ],
                env=PIP_ENV,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=PIP_TIMEOUT,
                check=False,
                creationflags=_NO_WINDOW,
            )
        except OSError as error:
            logger.warning("Could not run pip to install %s: %s", name, error)
            continue
        except subprocess.TimeoutExpired:
            logger.warning(
                "Installing the VapourSynth user module %s took longer than %d s;"
                " pip was stopped and its functions are skipped.",
                name,
                PIP_TIMEOUT,
            )
            continue
        if done.returncode == 0:
            logger.info("Installed the VapourSynth user module %s.", name)
            continue
        lines = (done.stdout + done.stderr).strip().splitlines()
        logger.warning(
            "Could not install the VapourSynth user module %s; its functions are"
            " skipped. pip said:\n%s",
            name,
            "\n".join(lines[-PIP_REASON_LINES:]),
        )


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _plugins_digest() -> str:
    """The plugin folder's listing (paths, sizes, times): an installed, updated or
    removed plugin changes it."""
    folder = vspipe().parent / "plugins"
    entries: list[str] = []
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for name in sorted(files):
            path = Path(root, name)
            stat = path.stat()
            entries.append(
                f"{path.relative_to(folder)}|{stat.st_size}|{stat.st_mtime_ns}"
            )
    return _digest("\n".join(entries).encode())


def catalog_key(storage: Path, modules: Sequence[str]) -> dict[str, Any]:
    """What the catalog lists depends on; a change means a rescan."""
    user_dir = storage / USER_DIR
    modules_file = storage / MODULES_FILE
    user_files: dict[str, list[int]] = {}
    if user_dir.is_dir():
        for path in sorted(user_dir.glob("*.py")):
            stat = path.stat()
            user_files[path.name] = [stat.st_mtime_ns, stat.st_size]
    return {
        "scanner": _digest(SCAN_PY.read_bytes()),
        "vapoursynth": _version("vapoursynth"),
        "vsjetpack": _version("vsjetpack"),
        "plugins": _plugins_digest(),
        "user_files": user_files,
        "user_modules": {name: _version(name) for name in modules},
        "user_modules_file": (
            _digest(modules_file.read_bytes()) if modules_file.is_file() else None
        ),
    }


def _stored_key(catalog: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(catalog.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return None
    return data.get("key")


def run_scan(out: Path, user_dir: Path, modules_file: Path) -> None:
    """Runs scan.py in a child process of VapourSynth's Python; raises RuntimeError
    with the end of its stderr when it fails or takes longer than SCAN_TIMEOUT s."""
    command = [
        str(vs_python()),
        "-I",
        "-B",
        str(SCAN_PY),
        "--out",
        str(out),
        "--user-dir",
        str(user_dir),
        "--modules-file",
        str(modules_file),
    ]
    try:
        done = subprocess.run(
            command,
            env=vs_env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=SCAN_TIMEOUT,
            check=False,
            creationflags=_NO_WINDOW,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"The scan took longer than {SCAN_TIMEOUT} s.") from error
    if done.returncode != 0:
        stderr = done.stderr.decode("utf-8", "replace").splitlines()
        lines = [line for line in stderr if not _API3_NOTICE.match(line)]
        tail = "\n".join(lines[-STDERR_TAIL_LINES:]).strip()
        raise RuntimeError(tail or f"The scan exited with code {done.returncode}.")


def ensure_catalog(storage: Path) -> Path | None:
    """`<storage>\\catalog.json`, rescanned first when its key changed (VapourSynth or
    vs-jetpack versions, the plugin folder, the user files, user_modules.toml and the
    listed modules' versions); missing user modules are installed first, at each
    start while one is missing.
    None when VapourSynth is not installed, or when the scan failed and there is no
    earlier catalog (a failed scan keeps the earlier one)."""
    try:
        vspipe()
    except VSScriptError:
        logger.info("VapourSynth is not installed: no VapourSynth function catalog.")
        return None
    catalog = storage / CATALOG
    modules_file = storage / MODULES_FILE
    try:
        modules = user_module_names(modules_file)
    except ValueError as error:
        logger.warning("VapourSynth user modules: %s", error)
        modules = []
    key = catalog_key(storage, modules)
    stored = _stored_key(catalog)
    # A listed module that is not installed (its install failed) is tried again at
    # each start; the catalog is rescanned when that changed the key.
    if stored == key and None not in key["user_modules"].values():
        return catalog

    install_user_modules(modules)
    key = catalog_key(storage, modules)
    if stored == key:
        return catalog
    storage.mkdir(parents=True, exist_ok=True)
    scanned = catalog.with_name(CATALOG + ".scan")
    written = catalog.with_name(CATALOG + ".tmp")
    logger.info("Scanning the VapourSynth functions into %s.", catalog)
    try:
        run_scan(scanned, storage / USER_DIR, modules_file)
        data = json.loads(scanned.read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA:
            raise ValueError(
                f"the scan wrote schema {data.get('schema')}, not {SCHEMA}"
            )
        data["key"] = key
        written.write_text(
            json.dumps(data, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(written, catalog)
    except Exception as error:
        logger.error("The VapourSynth function scan failed: %s", error)
        return catalog if catalog.is_file() else None
    finally:
        scanned.unlink(missing_ok=True)
        written.unlink(missing_ok=True)
    for error in data.get("errors", []):
        logger.warning("VapourSynth scan skipped %s", error)
    logger.info("The VapourSynth catalog lists %d functions.", len(data["functions"]))
    return catalog


def overlay_paths(storage: Path) -> list[Path]:
    """The shipped overlays, then the user's (which win)."""
    return [
        *sorted(SHIPPED_OVERLAYS.glob("*.toml")),
        *sorted((storage / OVERLAY_DIR).glob("*.toml")),
    ]


def _overlay_problem(key: str, value: object) -> str | None:
    """Why an overlay key's value cannot be used, or None."""
    if key in ("label", "help", "class_default"):
        return None if isinstance(value, str) else "not a text"
    if key == "family":
        return None if value in FAMILIES else "not one of the node names"
    if key in ("simple", "hidden"):
        names = isinstance(value, list) and all(isinstance(v, str) for v in value)
        return None if names else "not a list of parameter names"
    if key in ("labels", "annotations"):
        texts = isinstance(value, dict) and all(
            isinstance(v, str) for v in value.values()
        )
        return None if texts else "not a table of texts"
    if key == "defaults":
        return None if isinstance(value, dict) else "not a table"
    return "not an overlay key"


def read_overlays(overlays: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """Each method id's overlay table, merged in order: a later file's keys win. A file
    that cannot be read is logged and skipped, and so is a key that cannot be used (an
    earlier file's value of it stays)."""
    merged: dict[str, dict[str, Any]] = {}
    for path in overlays:
        try:
            # utf-8-sig: Notepad and Windows PowerShell may start it with a BOM.
            data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
            logger.warning("Skipping the VapourSynth overlay %s: %s", path, error)
            continue
        tables = data.get("method", {})
        if not isinstance(tables, dict):
            logger.warning(
                "Skipping the VapourSynth overlay %s: no [method] tables", path
            )
            continue
        for method_id, table in tables.items():
            if not isinstance(table, dict):
                continue
            entry = merged.setdefault(method_id, {})
            for key, value in table.items():
                problem = _overlay_problem(key, value)
                if problem is None:
                    entry[key] = value
                else:
                    logger.warning(
                        "Skipping %s of %s in the VapourSynth overlay %s: %r is %s.",
                        key,
                        method_id,
                        path,
                        value,
                        problem,
                    )
    return merged


def family_of(fn: Mapping[str, Any], overlay: Mapping[str, Any] | None) -> str:
    """The family node of a catalog function: the overlay's family, else User
    Functions for a user function (whose file may be named like a plugin namespace,
    resize.py), else the FUNCTION_FAMILIES entry (vstools, std), else the
    MODULE_FAMILIES entry, else Any Function."""
    if overlay and overlay.get("family"):
        return str(overlay["family"])
    if fn["source"] == "user":
        return "User Functions"
    family = FUNCTION_FAMILIES.get(fn["id"]) or MODULE_FAMILIES.get(fn["module"])
    return family or "Any Function"


def _words(name: str) -> str:
    return " ".join(word[:1].upper() + word[1:] for word in name.split("_") if word)


def _label(fn: Mapping[str, Any]) -> str:
    """A plain name: "Mc Degrain", "NNEDI3 Deinterlace", "Merge (std)", and for a user
    function its module too, "Average (mine)" (two files may share a name)."""
    call: str = fn["call"]
    method = call.rsplit(".", 1)[-1]
    if fn["kind"] == "plugin":
        return f"{method} ({fn['module']})"
    if fn["kind"] == "bound":
        return f"{call.split('(', 1)[0]} {_words(method)}"
    if fn["source"] == "user":
        return f"{_words(call)} ({fn['module']})"
    return _words(call)


def _params(
    entries: Sequence[Mapping[str, Any]],
    table: Mapping[str, Any],
    method_id: str,
) -> tuple[Param, ...]:
    """The parameters without the overlay's hidden ones, with its annotations."""
    hidden = set(table.get("hidden", []))
    annotations: Mapping[str, str] = table.get("annotations", {})
    params: list[Param] = []
    for entry in entries:
        name = entry["name"]
        param = Param(
            name,
            annotations.get(name, entry["annotation"]),
            entry["default"],
            entry["kind"],
        )
        if name in hidden:
            if param.default is not None or param.kind == "var_keyword":
                continue
            logger.warning(
                "The overlay hides %s of %s, which has no default; it stays.",
                name,
                method_id,
            )
        params.append(param)
    return tuple(params)


def _with_keywords(
    params: tuple[Param, ...], taken: set[str], table: Mapping[str, Any], method_id: str
) -> tuple[Param, ...]:
    """The overlay's annotations of names the function does not have, as keyword
    arguments of its **kwargs (std.SetFrameProps' _Matrix), before the **kwargs."""
    extra = [
        Param(name, annotation, "None", "keyword")
        for name, annotation in table.get("annotations", {}).items()
        if name not in taken
    ]
    if not extra:
        return params
    if not params or params[-1].kind != "var_keyword":
        logger.warning(
            "The overlay of %s names %s, which it does not take; they are left out.",
            method_id,
            ", ".join(p.name for p in extra),
        )
        return params
    return (*params[:-1], *extra, params[-1])


def _default_simple(params: Sequence[Param]) -> frozenset[str]:
    """The first three parameters that are not clips, and every one without a
    default."""
    named = [p for p in params if p.kind != "var_keyword"]
    first = [p.name for p in named if "VideoNode" not in p.annotation][:3]
    return frozenset([*first, *(p.name for p in named if p.default is None)])


def _method(fn: Mapping[str, Any], overlay: Mapping[str, Any] | None) -> Method:
    table = overlay or {}
    init = _params(fn.get("init_params") or [], table, fn["id"])
    # A method argument named like a constructor argument (NNEDI3's tff) would take
    # the same input: the constructor's covers both.
    init_names = {p.name for p in init}
    params = tuple(
        p for p in _params(fn["params"], table, fn["id"]) if p.name not in init_names
    )
    params = _with_keywords(
        params, {*init_names, *(e["name"] for e in fn["params"])}, table, fn["id"]
    )
    simple = table.get("simple")
    import_line, call = qualified(fn.get("import", ""), fn["call"])
    return Method(
        id=fn["id"],
        family=family_of(fn, overlay),
        label=table.get("label") or _label(fn),
        call=call,
        kind=fn["kind"],
        init_params=init,
        params=params,
        doc=table.get("help") or fn["doc"],
        arg_docs=fn["arg_docs"],
        simple=frozenset(simple)
        if simple is not None
        else _default_simple(init + params),
        import_line=import_line,
        labels=dict(table.get("labels", {})),
        common=overlay is not None,
        defaults=dict(table.get("defaults", {})),
        many=returns_many(fn["returns"]),
    )


def returns_many(returns: str) -> bool:
    """A list of clips ("list[VideoNode]", "Sequence[vs.VideoNode]"), not one clip."""
    text = returns.replace("vs.", "").replace(" ", "")
    return any(f"{s}[VideoNode]" in text for s in ("list", "Sequence", "tuple"))


def _group(
    group_id: str,
    first: Mapping[str, Any],
    members: Sequence[Method],
    table: Mapping[str, Any],
    defaults: tuple[str, str],
) -> Method:
    """One method of `members` (one per class or enum member, each with its
    `choice`), picked in a dropdown ("class"). `defaults`: the method's label and the
    dropdown's when the overlay names none."""
    choices = [m.choice for m in members]
    arg_docs: dict[str, str] = {}
    for member in members:
        arg_docs = {**member.arg_docs, **arg_docs}
    choice = str(table.get("class_default", ""))
    if choice and choice not in choices:
        logger.warning(
            "The overlay of %s preselects %s, which it does not list.", group_id, choice
        )
        choice = ""
    simple = table.get("simple")
    return Method(
        id=group_id,
        family=family_of(first, table),
        label=table.get("label") or defaults[0],
        call="",
        kind="group",
        init_params=(),
        params=(),
        doc=table.get("help") or members[0].doc,
        arg_docs=arg_docs,
        # The dropdown, and what every member shows in Simple mode.
        simple=frozenset(simple)
        if simple is not None
        else frozenset.intersection(*(m.simple for m in members)) | {CLASS},
        labels={CLASS: defaults[1], **table.get("labels", {})},
        members=tuple(members),
        default_choice=choice,
        common=bool(table),
    )


def _class_group(
    key: tuple[str, str],
    fns: Sequence[Mapping[str, Any]],
    tables: Mapping[str, Mapping[str, Any]],
) -> Method:
    """The method `key[1]` of the classes of module `key[0]`, as one group."""
    module, name = key
    group_id = f"{module}.*.{name}"
    table = tables.get(group_id) or {}
    shared = {k: table[k] for k in _MEMBER_KEYS if k in table}
    members: list[Method] = []
    for fn in sorted(fns, key=lambda fn: fn["call"]):
        cls = fn["call"].split("(", 1)[0]
        member = _method(fn, {**shared, **(tables.get(fn["id"]) or {})})
        members.append(replace(member, label=cls, choice=cls))
    return _group(
        group_id, fns[0], members, table, (f"{_words(name)} ({module})", "Class")
    )


def _enum_group(
    fn: Mapping[str, Any], tables: Mapping[str, Mapping[str, Any]]
) -> Method:
    """An enum whose members filter a clip (Prefilter): one group of its members."""
    if not fn["members"]:
        raise ValueError("the enum has no members")
    table = tables.get(fn["id"]) or {}
    base = _method(fn, {k: table[k] for k in _MEMBER_KEYS if k in table})
    name = fn["call"].split(".", 1)[0]
    members = [
        replace(
            base,
            id=f"{fn['id']}.{member}",
            label=member,
            call=base.call.replace("<member>", member),
            kind="function",
            choice=member,
        )
        for member in fn["members"]
    ]
    return _group(fn["id"], fn, members, table, (_words(name), name))


@dataclass(frozen=True)
class Catalog:
    """The Methods of the family nodes and the catalog's enums map (annotation key ->
    {"import", "members": [{"name", "expr"}]})."""

    methods: list[Method]
    enums: Mapping[str, Mapping[str, Any]]


def load_catalog(catalog: Path, overlays: Sequence[Path]) -> Catalog:
    """The catalog's functions as Methods, in catalog order (a group where its first
    member is), with the overlays applied; an entry that does not load is logged and
    skipped."""
    data = json.loads(catalog.read_text(encoding="utf-8"))
    tables = read_overlays(overlays)
    user_dir = catalog.parent / USER_DIR
    by_method: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for fn in data["functions"]:
        if fn.get("kind") == "bound":
            key = (fn["module"], fn["call"].rpartition(".")[2])
            by_method.setdefault(key, []).append(fn)
    methods: list[Method] = []
    for fn in data["functions"]:
        if fn.get("id") in HIDDEN_FUNCTIONS:
            continue
        try:
            key = (fn["module"], fn["call"].rpartition(".")[2])
            if fn["kind"] == "members":
                method = _enum_group(fn, tables)
            elif fn["kind"] != "bound" or len(by_method[key]) < GROUP_MIN:
                method = _method(fn, tables.get(fn["id"]))
            elif by_method[key][0] is fn:
                method = _class_group(key, by_method[key], tables)
            else:
                continue
            methods.append(_with_search_path(method, fn, user_dir))
        except (KeyError, TypeError, ValueError) as error:
            logger.warning("Skipping the catalog entry %s: %r", fn.get("id"), error)
    return Catalog(methods, data.get("enums", {}))


def _with_search_path(method: Method, fn: Mapping[str, Any], user_dir: Path) -> Method:
    """A function of the user folder's files (not of a listed pip module) needs that
    folder on the script's sys.path."""
    if fn["source"] != "user" or not (user_dir / f"{fn['module']}.py").is_file():
        return method
    path = str(user_dir)
    members = tuple(replace(m, search_path=path) for m in method.members)
    return replace(method, search_path=path, members=members)


def current_catalog() -> Catalog:
    """The catalog of the family nodes, or an empty one (only the input and output
    nodes load) outside the server, without VapourSynth, or when no scan
    succeeded."""
    storage = storage_dir()
    catalog = ensure_catalog(storage) if storage is not None else None
    if storage is None or catalog is None:
        return Catalog([], {})
    return load_catalog(catalog, overlay_paths(storage))


def _raw_id(text: str) -> int:
    raw = zlib.crc32(text.encode()) & _ID_MASK
    return raw + FIRST_INPUT_ID if raw < FIRST_INPUT_ID else raw


def input_ids(
    family: str, keys: Iterable[tuple[str, str]]
) -> dict[tuple[str, str], int]:
    """The input id of each (method id, parameter) of a family node: a 31-bit CRC-32 of
    "family/method/param", at least FIRST_INPUT_ID. On a collision the sorted-earlier
    key keeps its hash and the later one rehashes with a "#1", "#2"... suffix. So
    adding methods moves no existing id, unless a new key's hash collides with an
    existing key's and the new key sorts earlier: then the existing key rehashes and
    a saved chain's value for it is not found. With 31-bit hashes and at most about
    450 keys in a node, that is about 5e-5 per node at most."""
    ids: dict[tuple[str, str], int] = {}
    taken: dict[int, tuple[str, str]] = {}
    for key in sorted(set(keys)):
        text = f"{family}/{key[0]}/{key[1]}"
        value = _raw_id(text)
        suffix = 0
        while value in taken:
            suffix += 1
            value = _raw_id(f"{text}#{suffix}")
        if suffix:
            logger.info(
                "Input id collision in %s: %s/%s takes the hash of suffix #%d.",
                family,
                key[0],
                key[1],
                suffix,
            )
        ids[key] = value
        taken[value] = key
    return ids
