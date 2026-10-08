"""Explicit native plugin and TensorRT build commands. Nothing builds on import.

Both subcommands require --execute to launch a build. Otherwise return argv.
Build products are UNVALIDATED until independent runtime parity checks pass.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import time

from .spec import ROOT, SOURCE_SHA256, sha256


def read_verified(directory, filename):
    directory = Path(directory).resolve()
    record = json.loads((directory / filename).read_text(encoding="utf-8"))
    if record["source_sha256"] != SOURCE_SHA256:
        raise ValueError("Source identity mismatch")
    for name, expected in record["files"].items():
        if Path(name).name != name or sha256(directory / name) != expected:
            raise ValueError(f"File identity mismatch: {name}")
    return record


def run_logged(argv, directory, label, timeout):
    # Own a process group/tree and retain failure logs; never retry or kill by name.
    receipt = {"argv": argv, "started": time.time(), "label": label}
    with (directory / f"{label}.log").open("w", encoding="utf-8") as log:
        options = (
            {
                "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.CREATE_NO_WINDOW
            }
            if os.name == "nt"
            else {"start_new_session": True}
        )
        process = subprocess.Popen(
            argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, **options
        )
        receipt["pid"] = process.pid
        record = directory / f"{label}_process.json"
        record.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        try:
            process.wait(timeout=timeout)
        finally:
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        check=True,
                        timeout=30,
                        stdout=log,
                        stderr=log,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    process.wait(timeout=30)
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
            receipt.update(exit=process.returncode, ended=time.time())
            record.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if process.returncode:
        raise RuntimeError(
            f"{label} exited {process.returncode}; see {directory / (label + '.log')}"
        )


def plugins(export_dir, aot_dir, output, trt_root, *, cmake="cmake", execute=False):
    export_dir, aot_dir, output = map(
        lambda p: Path(p).resolve(), (export_dir, aot_dir, output)
    )
    export = read_verified(export_dir, "export.json")
    aot = read_verified(aot_dir, "aot.json")
    spec = export["specialization"]
    if aot["plugin_key"] != spec["plugin_key"] or aot["kernels_sha256"] != sha256(
        ROOT / "scripts/dual_tensorrt/kernels.py"
    ):
        raise ValueError("AOT specialization/source mismatch")
    if output.exists():
        raise FileExistsError(output)
    h, w = spec["trunk"]
    configure = [
        str(cmake),
        "-S",
        str(ROOT / "tensorrt_plugins/dual_tensorrt"),
        "-B",
        str(output),
        f"-DTRT_ROOT={Path(trt_root).resolve().as_posix()}",
        f"-DDUAL_AOT_DIR={aot_dir.as_posix()}",
        f"-DDUAL_C={spec['channels']}",
        f"-DDUAL_H={h}",
        f"-DDUAL_W={w}",
        f"-DDUAL_SM={aot['sm']}",
    ]
    configure += (
        ["-G", "Visual Studio 17 2022", "-A", "x64"]
        if os.name == "nt"
        else ["-DCMAKE_BUILD_TYPE=Release"]
    )
    compile_cmd = [
        str(cmake),
        "--build",
        str(output),
        "--config",
        "Release",
        "--parallel",
        "2",
    ]
    commands = [configure, compile_cmd]
    if not execute:
        return {"execute": False, "commands": commands}
    output.mkdir(parents=True, exist_ok=False)
    (output / "commands.json").write_text(
        json.dumps(commands, indent=2), encoding="utf-8"
    )
    run_logged(configure, output, "configure", 180)
    run_logged(compile_cmd, output, "compile", 1800)
    windows = os.name == "nt"
    libraries = []
    for part in ("Norm", "Core", "Project", "AttentionBarrier"):
        name = f"Dual{part}.dll" if windows else f"libDual{part}.so"
        path = output / ("Release" if windows else "") / name
        # Multi-config MSVC is explicitly used on Windows.
        if not path.is_file():
            raise FileNotFoundError(path)
        libraries.append({"path": str(path), "sha256": sha256(path)})
    record = {
        "schema": 1,
        "status": "built_unvalidated",
        "source_sha256": SOURCE_SHA256,
        "platform": platform.system(),
        "machine": platform.machine(),
        "plugin_key": spec["plugin_key"],
        "sm": aot["sm"],
        "libraries": libraries,
        "aot_manifest_sha256": sha256(aot_dir / "aot.json"),
    }
    (output / "plugins.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def engine(export_dir, plugins_file, output, *, trtexec="trtexec", execute=False):
    export_dir, plugins_file, output = map(
        lambda p: Path(p).resolve(), (export_dir, plugins_file, output)
    )
    export = read_verified(export_dir, "export.json")
    plugins = json.loads(plugins_file.read_text(encoding="utf-8"))
    if (
        plugins["platform"] != platform.system()
        or plugins["machine"] != platform.machine()
        or plugins["source_sha256"] != SOURCE_SHA256
        or plugins["plugin_key"] != export["specialization"]["plugin_key"]
    ):
        raise ValueError("Plugin platform, source or specialization mismatch")
    if output.exists():
        raise FileExistsError(output)
    argv = [
        str(trtexec),
        f"--onnx={export_dir / 'model.onnx'}",
        f"--saveEngine={output / 'model.engine'}",
        "--builderOptimizationLevel=3",
        "--stronglyTyped",
        "--noTF32",
        "--maxAuxStreams=0",
        "--memPoolSize=workspace:8G",
        "--profilingVerbosity=detailed",
        "--skipInference",
        f"--timingCacheFile={output / 'timing.cache'}",
    ]
    for library in plugins["libraries"]:
        path = Path(library["path"])
        if sha256(path) != library["sha256"]:
            raise ValueError(f"Plugin hash mismatch: {path}")
        argv += [f"--dynamicPlugins={path}", f"--setPluginsToSerialize={path}"]
    if not execute:
        return {"execute": False, "argv": argv}
    output.mkdir(parents=True, exist_ok=False)
    (output / "command.json").write_text(json.dumps(argv, indent=2), encoding="utf-8")
    start = time.time()
    run_logged(argv, output, "build", 1800)
    record = {
        "schema": 1,
        "status": "built_unvalidated",
        "platform": platform.system(),
        "source_sha256": SOURCE_SHA256,
        "export_manifest_sha256": sha256(export_dir / "export.json"),
        "plugin_manifest_sha256": sha256(plugins_file),
        "engine_sha256": sha256(output / "model.engine"),
        "launch_to_exit_seconds": time.time() - start,
        "validation": "Build only: fresh-process deserialization and checkpoint-output parity still required",
    }
    (output / "engine.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plugin_parser = sub.add_parser("plugins")
    engine_parser = sub.add_parser("engine")
    for command in (plugin_parser, engine_parser):
        command.add_argument("--export-dir", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--execute", action="store_true")
    plugin_parser.add_argument("--aot-dir", type=Path, required=True)
    plugin_parser.add_argument("--trt-root", type=Path, required=True)
    plugin_parser.add_argument("--cmake", default="cmake")
    engine_parser.add_argument("--plugins-file", type=Path, required=True)
    engine_parser.add_argument("--trtexec", default="trtexec")
    args = vars(parser.parse_args())
    action = args.pop("action")
    print(json.dumps((plugins if action == "plugins" else engine)(**args), indent=2))


if __name__ == "__main__":
    main()
