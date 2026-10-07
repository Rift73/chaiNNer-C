"""Compare isolated oracle/portable image I/O and execution HTTP/SSE graphs.

The default validates the oracle's fixtures only (verify_runtime.oracle_source),
on the provisioned runtime the package was copied from. --include-port
additionally runs the released portable package on the package's runtime. All
images, directories, profiles, source copies,
and outputs are owned by a new report directory. No clipboard, viewers, video,
GPU execution, trained models, or performance measurements are used.
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import traceback
import zlib
from datetime import UTC, datetime
from pathlib import Path

from verify_framework_runtime import compare_runs, run_backend, source_hashes
from verify_runtime import (
    METADATA_COMPARISON,
    ORACLE,
    PROJECT,
    SSE_EVENT_COMPARISON,
    edge,
    make_node,
    oracle_repeat_flags,
    oracle_source,
    write_json,
)

IO_EXECUTION_IDS = {
    "chainner:image:load",
    "chainner:image:save",
    "chainner:image:load_images",
    *(
        "chainner:utility:" + name
        for name in (
            "into_directory",
            "accumulate",
            "logic_operation",
            "conditional",
            "execution_number",
            "range",
            "note",
        )
    ),
}


def fixture_png(channels: int, depth: int, *, opaque: bool = False) -> bytes:
    """Make a tiny deterministic PNG without loading a production image helper."""
    width, height = 13, 9
    maximum = (1 << depth) - 1
    rows = bytearray()
    for y in range(height):
        rows.append(0)  # PNG filter None
        for x in range(width):
            for channel in range(channels):
                value = (x * 7039 + y * 11027 + channel * 17041) & maximum
                if channels == 4 and channel == 3:
                    value = maximum if opaque else (x * maximum) // (width - 1)
                rows.extend(bytes([value]) if depth == 8 else struct.pack(">H", value))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    color_type = {1: 0, 3: 2, 4: 6}[channels]
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, depth, color_type, 0, 0, 0),
        )
        + chunk(b"IDAT", zlib.compress(bytes(rows)))
        + chunk(b"IEND", b"")
    )


def prepare_fixtures(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    for channels in (1, 3, 4):
        for depth in (8, 16):
            (root / f"c{channels}-u{depth}.png").write_bytes(
                fixture_png(channels, depth)
            )
    for depth in (8, 16):
        (root / f"opaque-u{depth}.png").write_bytes(fixture_png(4, depth, opaque=True))
    (root / "caf\u00e9-\u56fe-\U0001f642.png").write_bytes(fixture_png(3, 8))
    tree = root / "sequence"
    for name, channels, depth in (
        ("image1.png", 1, 8),
        ("image2.PNG", 3, 8),
        ("image02.png", 4, 8),
        ("image10.png", 3, 16),
        ("sub/frame3.png", 4, 16),
        ("sub/deep/\u00e94.png", 1, 16),
        (".hidden.png", 3, 8),
    ):
        path = tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fixture_png(channels, depth))
    (tree / "not-an-image.txt").write_text(
        "Ignored by the extension filter", encoding="utf-8"
    )
    (root / "image-directory/folder.png").mkdir(parents=True)
    (root / "empty").mkdir()
    (root / "errors").mkdir()
    (root / "errors/0-corrupt.png").write_bytes(b"Not a valid encoded image")
    (root / "errors/1-valid.png").write_bytes(fixture_png(3, 8))


def fixture_graphs(schemas: dict, assets: Path, output: Path):
    def node(name: str, schema: str, values: dict):
        return make_node(schemas, name, schema, values)

    def save(
        name: str, image: dict, *, values: dict | None = None, node_id: str = "save"
    ):
        return node(
            node_id,
            "chainner:image:save",
            {
                0: image,
                1: str(output / name),
                2: None,
                3: "image",
                4: "png",
                15: "u8",
                **(values or {}),
            },
        )

    def case(
        name: str,
        nodes: list,
        files: list[str],
        *,
        error: bool = False,
        dead: tuple[str, ...] = (),
    ):
        return {
            "name": name,
            "nodes": nodes,
            "files": files,
            "expected_error": error,
            "settle_error_events": error,
            "expected_dead_nodes": list(dead),
        }

    def render_result(
        name: str,
        nodes: list,
        source: str,
        *,
        error: bool = False,
        dead: tuple[str, ...] = (),
    ):
        nodes.extend(
            [
                node(
                    "pattern",
                    "chainner:utility:text_pattern",
                    {0: "Result: {1}", 1: edge(source)},
                ),
                node(
                    "render",
                    "chainner:image:text_as_image",
                    {0: edge("pattern"), 5: 320, 6: 64},
                ),
                save(name, edge("render")),
            ]
        )
        return case(name, nodes, [] if error else ["image.png"], error=error, dead=dead)

    # Image loading covers real channel/depth normalization and alpha removal.
    image_names = [f"c{c}-u{depth}" for c in (1, 3, 4) for depth in (8, 16)]
    image_names.extend(["opaque-u8", "opaque-u16", "caf\u00e9-\u56fe-\U0001f642"])
    for number, image_name in enumerate(image_names):
        name = f"load-image-{number}"
        yield case(
            name,
            [
                node(
                    "load",
                    "chainner:image:load",
                    {0: str(assets / (image_name + ".png"))},
                ),
                save(name, edge("load"), values={15: "u16"}),
            ],
            ["image.png"],
        )
    for name, source in (
        ("load-corrupt", assets / "errors/0-corrupt.png"),
        ("load-missing", assets / "missing.png"),
    ):
        yield case(
            name,
            [
                node("load", "chainner:image:load", {0: str(source)}),
                save(name, edge("load")),
            ],
            [],
            error=True,
        )

    # Each codec's bytes are checked on both repeats and, when requested, across hosts.
    encodings = [
        ("png8", "png", {15: "u8"}, "c4-u16.png"),
        ("png16", "png", {15: "u16"}, "c3-u16.png"),
        ("jpg-default", "jpg", {}, "c3-u8.png"),
        ("jpg-progressive", "jpg", {5: 83, 12: True, 11: 0x111111}, "c3-u8.png"),
        ("jpg-gray", "jpg", {5: 73}, "c1-u8.png"),
        ("gif", "gif", {}, "c4-u8.png"),
        ("bmp", "bmp", {}, "c3-u8.png"),
        ("tga", "tga", {}, "c4-u8.png"),
        ("webp-lossless", "webp", {14: True}, "c4-u8.png"),
        ("webp-lossy", "webp", {14: False, 5: 83}, "c3-u8.png"),
        ("tiff8", "tiff", {16: "u8", 18: 1}, "c4-u8.png"),
        ("tiff16", "tiff", {16: "u16", 18: 5}, "c3-u16.png"),
        ("tiff-float", "tiff", {16: "f32", 18: 8}, "c3-u16.png"),
        ("avif", "avif", {5: 79, 17: "4:4:4"}, "c3-u8.png"),
    ]
    for encoding, extension, options, asset in encodings:
        name = "save-" + encoding
        yield case(
            name,
            [
                node("load", "chainner:image:load", {0: str(assets / asset)}),
                save(name, edge("load"), values={4: extension, **options}),
            ],
            ["image." + extension],
        )
        # GIF saving is supported, but GIF is intentionally absent from Load Image's inputs.
        if extension != "gif":
            reload_name = "reload-" + encoding
            yield case(
                reload_name,
                [
                    node(
                        "load",
                        "chainner:image:load",
                        {0: str(output / name / ("image." + extension))},
                    ),
                    save(reload_name, edge("load"), values={15: "u16"}),
                ],
                ["image.png"],
            )

    top_files = [
        ".hidden.png",
        "image1.png",
        "image2.png",
        "image02.png",
        "image10.png",
    ]
    recursive_files = [*top_files, "sub/frame3.png", "sub/deep/\u00e94.png"]
    sequences = [
        ("default", "sequence", {}, recursive_files, False),
        ("nonrecursive", "sequence", {2: False}, top_files, False),
        ("recursive", "sequence", {2: True}, recursive_files, False),
        ("limit", "sequence", {2: True, 4: True, 5: 1}, [".hidden.png"], False),
        (
            "brace",
            "sequence",
            {1: True, 3: "{image1.png,image10.png,sub/frame3.png}"},
            ["image1.png", "image10.png", "sub/frame3.png"],
            False,
        ),
        (
            "extglob",
            "sequence",
            {1: True, 3: "@(image1|image10).png"},
            ["image1.png", "image10.png"],
            False,
        ),
        (
            "recursive-glob",
            "sequence",
            {1: True, 3: "sub/**/*.png"},
            ["sub/frame3.png", "sub/deep/\u00e94.png"],
            False,
        ),
        (
            "hidden-explicit",
            "sequence",
            {1: True, 3: ".hidden.png"},
            [".hidden.png"],
            False,
        ),
        ("empty", "empty", {}, [], True),
        ("image-directory", "image-directory", {}, [], True),
        ("no-match", "sequence", {1: True, 3: "absent*.png"}, [], True),
        ("fail-fast", "errors", {6: True}, [], True),
        ("deferred-error", "errors", {6: False}, ["1-valid.png"], True),
    ]
    for label, folder, options, files, error in sequences:
        name = "sequence-" + label
        yield case(
            name,
            [
                node(
                    "load-sequence",
                    "chainner:image:load_images",
                    {0: str(assets / folder), **options},
                ),
                save(
                    name,
                    edge("load-sequence"),
                    values={
                        2: edge("load-sequence", 2),
                        3: edge("load-sequence", 3),
                        15: "u16",
                    },
                ),
            ],
            files,
            error=error,
        )

    for operation in ("sum", "prod", "max", "min"):
        for include_start in (False, True):
            for include_end in (False, True):
                name = f"range-{operation}-{int(include_start)}-{int(include_end)}"
                yield render_result(
                    name,
                    [
                        node(
                            "range",
                            "chainner:utility:range",
                            {0: 2, 1: include_start, 2: 5, 3: include_end},
                        ),
                        node(
                            "accumulate",
                            "chainner:utility:accumulate",
                            {0: edge("range"), 1: operation},
                        ),
                    ],
                    "accumulate",
                )
    yield render_result(
        "range-default",
        [
            node("range", "chainner:utility:range", {}),
            node("accumulate", "chainner:utility:accumulate", {0: edge("range")}),
        ],
        "accumulate",
    )
    yield render_result(
        "range-descending",
        [
            node("range", "chainner:utility:range", {0: 5, 2: 2}),
            node("accumulate", "chainner:utility:accumulate", {0: edge("range")}),
        ],
        "accumulate",
        error=True,
    )

    for operation in ("and", "or", "xor", "not"):
        for a in (False, True):
            for b in (False, True):
                name = f"logic-{operation}-{int(a)}-{int(b)}"
                yield render_result(
                    name,
                    [
                        node(
                            "logic",
                            "chainner:utility:logic_operation",
                            {0: operation, 1: a, 2: b},
                        ),
                        node(
                            "choose",
                            "chainner:utility:conditional",
                            {0: edge("logic"), 1: "selected true", 2: "selected false"},
                        ),
                    ],
                    "choose",
                )
    for number in (1, 1234567):
        yield render_result(
            f"execution-number-{number}",
            [node("number", "chainner:utility:execution_number", {0: number})],
            "number",
        )
    for markdown in (False, True):
        yield render_result(
            f"note-{int(markdown)}",
            [
                node(
                    "note",
                    "chainner:utility:note",
                    {0: "# A note\n\u00e9\U0001f642", 1: markdown},
                ),
                node("number", "chainner:utility:execution_number", {0: 7}),
            ],
            "number",
            dead=("note",),
        )

    for label, folders, relative in (
        ("single", {1: "child"}, "child/image.png"),
        ("holes", {1: "one", 3: "two", 10: "caf\u00e9"}, "one/two/caf\u00e9/image.png"),
        ("normalize", {1: "one/../two", 2: "./three"}, "two/three/image.png"),
    ):
        name = "directory-" + label
        yield case(
            name,
            [
                node(
                    "directory",
                    "chainner:utility:into_directory",
                    {0: str(output / name), **folders},
                ),
                node("load", "chainner:image:load", {0: str(assets / "c3-u8.png")}),
                save(name, edge("load"), values={1: edge("directory")}),
            ],
            [relative],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=PROJECT / "out/chaiNNer-C")
    parser.add_argument("--include-port", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=180)
    args = parser.parse_args()
    package = args.package.resolve(strict=True)
    manifest = json.loads(
        (package / "chainner-c-package.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("state") != "complete"
        or manifest["identity"]["destination"] != str(package)
        or manifest["identity"]["project"] != str(PROJECT)
    ):
        raise ValueError(
            "A completed package manifest matching this project/location is required"
        )
    python = package / "python/python/python.exe"
    if not python.is_file() or not (package / "portable").is_file():
        raise ValueError("Independent portable Python runtime missing")
    source, oracle_python, oracle = oracle_source(manifest, ORACLE)
    sources = {"baseline": source}
    pythons = {"baseline": oracle_python, "converted": python}
    if args.include_port:
        sources["converted"] = package / "resources/src"
    root = (
        PROJECT
        / "native/reports"
        / ("io-execution-runtime-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    )
    root.mkdir(parents=True, exist_ok=False)
    protected = {label: source_hashes(path) for label, path in sources.items()}
    report: dict = {
        "utc": datetime.now(UTC).isoformat(),
        "package": str(package),
        "oracle": oracle,
        "success": False,
        "mode": "oracle-versus-port"
        if args.include_port
        else "baseline-fixture-validation-only",
        "cpu_only": True,
        "performance_benchmark": False,
        "model_inference": False,
        "requested_schema_ids": sorted(IO_EXECUTION_IDS),
        "limitations": [
            "Independent backend copies and report-owned fixtures are used; installed files and user profiles are not modified.",
            "Note is accepted in successful request payloads and removed by dead-node optimization; this verifies no backend algorithm or desktop Markdown rendering.",
            "Execution Number preserves the supplied frontend counter. Its desktop increment policy is not exercised by repeated identical HTTP requests.",
            "Load Images retains the complete installed WCMatch/bracex grammar compiler and compiled regex primitives; these cases exercise native discovery, filtering, ordering, iteration, and real image loading when the port is enabled.",
            "Image codecs remain their installed compiled engines. DDS is excluded to avoid implicit GPU encoder policy. GIF saving is covered; GIF is not a registered Load Image format.",
            "Clipboard, viewers, video, trained models, and GPU execution are excluded. Direct protocol/error/concurrency contracts are covered by independent frozen-oracle unit suites.",
            "PNG decoded components, non-PNG encoded bytes and HTTP results require exact parity. Events exclude timings and independent-node scheduling order and compare by the rule that follows.",
            SSE_EVENT_COMPARISON,
            METADATA_COMPARISON,
        ],
        "runs": [],
    }
    try:
        prepare_fixtures(root / "fixtures")
        report["fixture_asset_hashes"] = source_hashes(root / "fixtures")
        for label, source in sources.items():
            copied = root / (label + "-src")
            shutil.copytree(
                source, copied, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
            )
            result = run_backend(
                label,
                copied,
                pythons[label],
                root,
                args.startup_timeout,
                required_schema_ids=IO_EXECUTION_IDS,
                graphs=fixture_graphs,
                synthetic_model_inference=False,
            )
            metadata = json.loads(
                (root / label / "nodes.json").read_text(encoding="utf-8")
            )
            result["registered_schema_count"] = len(metadata["nodes"])
            if result["registered_schema_count"] != 161:
                raise AssertionError(
                    "Expected the complete installed 161-schema catalog"
                )
            report["runs"].append(result)
        if args.include_port:
            report["comparison"] = compare_runs(
                report["runs"][0], report["runs"][1], python
            )
            report["comparison"]["encoded_non_png_bytes_equal"] = report[
                "comparison"
            ].pop("serialized_model_bytes_equal")
            report["success"] = report["comparison"]["success"]
        else:
            report["success"] = report["runs"][0]["success"]
    except Exception as error:
        report["error"], report["traceback"] = str(error), traceback.format_exc()
        report["runs"] = [
            json.loads(path.read_text(encoding="utf-8"))
            for label in sources
            if (path := root / label / "result.json").is_file()
        ]
    finally:
        report["original_source_trees_unchanged"] = {
            label: protected[label] == source_hashes(path)
            for label, path in sources.items()
        }
        report["fixture_assets_unchanged"] = report.get(
            "fixture_asset_hashes"
        ) == source_hashes(root / "fixtures")
        report["all_owned_processes_stopped"] = all(
            run.get("owned_process_exited") and run.get("owned_job_closed")
            for run in report["runs"]
        )
        report["success"] = (
            report["success"]
            and all(report["original_source_trees_unchanged"].values())
            and report["fixture_assets_unchanged"]
            and report["all_owned_processes_stopped"]
        )
        # Flagged, not failed (Consult D-35): the oracle's own attempts ended on
        # different final values of an iterated field.
        report["oracle_repeat_final_values_differ"] = (
            oracle_repeat_flags(report["runs"][0]) if report["runs"] else {}
        )
        write_json(root / "report.json", report)
        print(
            json.dumps(
                {
                    "report": str(root / "report.json"),
                    "success": report["success"],
                    "error": report.get("error"),
                    "oracle_repeat_final_values_differ": sorted(
                        report["oracle_repeat_final_values_differ"]
                    ),
                }
            ),
            flush=True,
        )
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
