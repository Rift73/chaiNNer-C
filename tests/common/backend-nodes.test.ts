import { NonNeverType, SourceDocument, evaluate, parseExpression } from '@chainner/navi';
import { execFileSync } from 'child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'fs';
import os from 'os';
import path from 'path';
import { expect, test } from 'vitest';
import { BackendNodesResponse } from '../../src/common/Backend';
import { CategoryMap } from '../../src/common/CategoryMap';
import {
    Condition,
    InputData,
    InputId,
    NodeSchema,
    OutputId,
    SchemaId,
} from '../../src/common/common-types';
import { checkNodeValidity } from '../../src/common/nodes/checkNodeValidity';
import { testInputCondition } from '../../src/common/nodes/condition';
import { getInputCondition } from '../../src/common/nodes/inputCondition';
import { parseFunctionDefinitions } from '../../src/common/nodes/parseFunctionDefinitions';
import { sortNodes } from '../../src/common/nodes/sort';
import { PassthroughMap } from '../../src/common/PassthroughMap';
import { SchemaInputsMap } from '../../src/common/SchemaInputsMap';
import { SchemaMap } from '../../src/common/SchemaMap';
import { getChainnerScope } from '../../src/common/types/chainner-scope';
import { FunctionDefinition, FunctionInstance } from '../../src/common/types/function';
import { prettyPrintType } from '../../src/common/types/pretty';
import { lazy } from '../../src/common/util';

// The Python that `npm run test:py` uses, and the native module the server needs. Both exist
// only where the runtime is provisioned and the native code is built, as for backend/tests.
const python = path.resolve('native/runtime/cpython-3.14.8/python.exe');
const nativeGraph = path.resolve('backend/src/nodes/impl/_chainner_graph.pyd');

// Writes the server's /nodes reply to argv[1]. The packages load as on every start: a node
// module whose optional dependency is not installed is left out, and the rest still register.
// A module that fails otherwise (an error, or a module of the backend it cannot find) would
// be left out alike, so it fails the dump, naming the module and its error.
const DUMP_NODES = `
import asyncio, importlib.util, os, pathlib, sys, types
out = sys.argv.pop()  # server.py parses the command line when it is imported
sys.path.insert(0, os.getcwd())
# The worker runs server.py: modules find the backend's files by its path.
sys.modules["__main__"].__file__ = os.path.abspath("server.py")
import server
from server_config import ServerConfig
config = ServerConfig(
    port=0,
    close_after_start=False,
    install_builtin_packages=False,
    error_on_failed_node=False,
    storage_dir=None,
    trace=False,
)
# The node-load failures are recorded in the storage folder: the dump's own folder.
context = types.SimpleNamespace(config=config, storage_dir=pathlib.Path(os.path.dirname(out)))
asyncio.run(server.import_packages(context))

def not_installed(error):
    if not isinstance(error, ModuleNotFoundError) or not error.name:
        return False
    return importlib.util.find_spec(error.name.partition(".")[0]) is None

failed = [
    f"{e.module}: {e.error!r}"
    for errors in server.node_loading.errors.values()
    for e in errors
    if not not_installed(e.error)
]
if failed:
    sys.exit("Node modules failed to load:\\n" + "\\n".join(failed))
with open(out, "wb") as f:
    f.write(asyncio.run(server.nodes(None)).body)
`;

const dumpNodes = lazy((): BackendNodesResponse => {
    const dir = mkdtempSync(path.join(os.tmpdir(), 'chaiNNer-nodes-'));
    try {
        const file = path.join(dir, 'nodes.json');
        execFileSync(python, ['-B', '-c', DUMP_NODES, file], {
            cwd: path.resolve('backend/src'),
            // listing the nodes starts no GPU
            env: { ...process.env, CUDA_VISIBLE_DEVICES: '-1', VK_LOADER_DRIVERS_DISABLE: '*' },
            stdio: ['ignore', 'ignore', 'pipe'],
        });
        return JSON.parse(readFileSync(file, 'utf-8')) as BackendNodesResponse;
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});

let dumped: BackendNodesResponse | undefined;
const backendNodes = (): BackendNodesResponse => {
    dumped ??= dumpNodes();
    return dumped;
};

// One node type the frontend cannot evaluate makes it reject the whole node list, and every
// start shows "Unable to process backend nodes.". This runs what BackendContext runs on the
// reply (processBackendResponse and the SchemaInputsMap) over the real schemas.
test.skipIf(!existsSync(python) || !existsSync(nativeGraph))(
    'The frontend accepts every backend node',
    () => {
        const { nodes, categories } = backendNodes();
        expect(nodes.length).toBeGreaterThan(100);

        const categoryMap = new CategoryMap(categories);
        const schemata = new SchemaMap(sortNodes(nodes, categoryMap));
        const functionDefinitions = parseFunctionDefinitions(nodes);
        PassthroughMap.create(functionDefinitions);
        expect(() => new SchemaInputsMap(schemata.schemata)).not.toThrow();

        expect(functionDefinitions.size).toBe(nodes.length);
    },
    300_000
);

// An empty input of Math Expression is no value: the node is valid without it, its output a number.
test.skipIf(!existsSync(python) || !existsSync(nativeGraph))(
    'Math Expression needs its expression and none of its numbers',
    () => {
        const { nodes } = backendNodes();
        const schemaId = 'chainner:utility:math_expression' as SchemaId;
        const schema = nodes.find((n) => n.schemaId === schemaId) as NodeSchema;
        const definition = parseFunctionDefinitions(nodes).get(schemaId) as FunctionDefinition;
        const numbers = schema.inputs.slice(1);
        expect(numbers.map((i) => i.label).join('')).toBe('abcdefghijklmnopqrstuvwxyz');
        expect(numbers.every((i) => i.kind === 'number' && i.optional && i.def === null)).toBe(
            true
        );

        const instance = FunctionInstance.fromPartialInputs(definition, () => undefined);
        expect(instance.outputErrors).toEqual([]);
        expect(prettyPrintType(instance.outputs.get(0 as OutputId) as NonNeverType)).toBe('number');

        const validity = (inputData: InputData) =>
            checkNodeValidity({
                schema,
                inputData,
                connectedInputs: new Set(),
                functionInstance: instance,
                chainLineage: undefined,
                nodeId: 'node',
            }).isValid;
        expect(validity({ [0 as InputId]: 'a + 1' })).toBe(true);
        expect(validity({ [0 as InputId]: 'a + 1', [1 as InputId]: 2 })).toBe(true);
        expect(validity({})).toBe(false);
    },
    300_000
);

// A folder's chain is Load Video's: each output a Save Video reads connects in the same way, to
// the inputs a per-file run feeds (folder_videos.py), and its ids are Load Video's.
test.skipIf(!existsSync(python) || !existsSync(nativeGraph))(
    'Load Videos connects to Save Video as Load Video does',
    () => {
        const functionDefinitions = parseFunctionDefinitions(dumpNodes().nodes);
        const definition = (schemaId: string) => {
            const found = functionDefinitions.get(schemaId as SchemaId);
            if (!found) throw new Error(`${schemaId} is not registered`);
            return found;
        };
        const loadVideos = definition('chainner:image:load_videos');
        const loadVideo = definition('chainner:image:load_video');
        const saveVideo = definition('chainner:image:save_video');
        const connections: readonly (readonly [number, number])[] = [
            [0, 0], // Frame: Image Sequence
            [2, 1], // Video Directory: Directory
            [3, 2], // Name: Video Name
            [4, 14], // FPS
            [5, 15], // Audio Stream
            [9, 23], // Colour Info
        ];
        for (const [output, input] of connections) {
            const type = loadVideos.defaultInstance.outputs.get(output as OutputId);
            expect(type, `output ${output}`).toBeDefined();
            expect(saveVideo.canAssignInput(input as InputId, type!), `input ${input}`).toBe(true);
        }
        const labels = (schema: typeof loadVideo.schema) =>
            schema.outputs.map((o) => [o.id, o.label]);
        expect(labels(loadVideos.schema).slice(0, loadVideo.schema.outputs.length)).toEqual(
            labels(loadVideo.schema)
        );
    },
    300_000
);

const image = (width: number, height: number, channels: number) =>
    evaluate(
        parseExpression(
            new SourceDocument(
                `Image { width: ${width}, height: ${height}, channels: ${channels} }`,
                'backend nodes'
            )
        ),
        getChainnerScope()
    ) as NonNeverType;

// The image size and channels the frontend shows and checks chains with, for nodes whose
// output depends on their inputs.
test.skipIf(!existsSync(python) || !existsSync(nativeGraph))(
    'Crop to Size, Pad to Square, Image Diff, Normal to Height and White Balance declare their outputs',
    () => {
        const { nodes } = backendNodes();
        const definitions = parseFunctionDefinitions(nodes);
        const instance = (
            schemaId: string,
            images: Partial<Record<number, NonNeverType>>,
            values: Partial<Record<number, number>>
        ) => {
            const definition = definitions.get(schemaId as SchemaId) as FunctionDefinition;
            const schema = nodes.find((n) => n.schemaId === schemaId);
            return FunctionInstance.fromPartialInputs(definition, (id: InputId) => {
                const assigned = images[id];
                if (assigned) return assigned;
                const input = schema?.inputs.find((i) => i.id === id);
                const value = values[id] ?? (input && 'def' in input ? input.def : undefined);
                return value === undefined
                    ? undefined
                    : definition.inputDataAdapters.get(id)?.(value as never);
            });
        };
        const output = (
            schemaId: string,
            images: Partial<Record<number, NonNeverType>>,
            values: Partial<Record<number, number>> = {}
        ) => {
            const { outputs, outputErrors } = instance(schemaId, images, values);
            expect(outputErrors).toEqual([]);
            return prettyPrintType(outputs.get(0 as OutputId) as NonNeverType);
        };
        const printed = (width: number, height: number, channels: number) =>
            prettyPrintType(image(width, height, channels));

        const CROP = 'chainner:image:crop_to_size';
        const [KEEP, PAD] = [0, 1];
        const TRANSPARENT = 5;
        const RGB = { 0: image(300, 200, 3) };
        expect(output(CROP, RGB, { 1: 100, 2: 50, 4: KEEP })).toBe(printed(100, 50, 3));
        expect(output(CROP, RGB, { 1: 500, 2: 50, 4: KEEP })).toBe(printed(300, 50, 3));
        expect(output(CROP, RGB, { 1: 500, 2: 900, 4: KEEP })).toBe(printed(300, 200, 3));
        expect(output(CROP, RGB, { 1: 500, 2: 50, 4: PAD })).toBe(printed(500, 50, 3));
        expect(output(CROP, RGB, { 1: 500, 2: 50, 4: PAD, 5: TRANSPARENT })).toBe(
            printed(500, 50, 4)
        );
        expect(output(CROP, RGB, { 1: 100, 2: 50, 4: PAD, 5: TRANSPARENT })).toBe(
            printed(100, 50, 3)
        );

        const SQUARE = 'chainner:image:pad_to_square';
        expect(output(SQUARE, RGB)).toBe(printed(300, 300, 3));
        expect(output(SQUARE, { 0: image(200, 300, 1) })).toBe(printed(300, 300, 1));
        expect(output(SQUARE, RGB, { 2: TRANSPARENT })).toBe(printed(300, 300, 4));
        expect(output(SQUARE, { 0: image(200, 200, 3) }, { 2: TRANSPARENT })).toBe(
            printed(200, 200, 3)
        );

        const DIFF = 'chainner:image:image_diff';
        const [ABSOLUTE, SIGNED, MASK] = [0, 1, 2];
        for (const channels of [1, 3, 4]) {
            const both = { 0: image(30, 20, channels), 1: image(30, 20, channels) };
            const colors = Math.min(channels, 3);
            expect(output(DIFF, both, { 2: ABSOLUTE })).toBe(printed(30, 20, colors));
            expect(output(DIFF, both, { 2: SIGNED })).toBe(printed(30, 20, colors));
            expect(output(DIFF, both, { 2: MASK })).toBe(printed(30, 20, 1));
        }
        for (const other of [image(31, 20, 3), image(30, 21, 3), image(30, 20, 4)]) {
            for (const mode of [ABSOLUTE, SIGNED, MASK]) {
                const { outputErrors } = instance(
                    DIFF,
                    { 0: image(30, 20, 3), 1: other },
                    { 2: mode }
                );
                expect(outputErrors.map((e) => e.outputId)).toContain(0);
            }
        }

        expect(output('chainner:image:normal_to_height', { 0: image(30, 20, 4) })).toBe(
            printed(30, 20, 1)
        );
        for (const channels of [1, 3, 4]) {
            expect(output('chainner:image:white_balance', { 0: image(30, 20, channels) })).toBe(
                printed(30, 20, channels)
            );
        }
    },
    300_000
);

// Conditional takes its Condition from Is Grayscale's output as it is, with no conversion.
test.skipIf(!existsSync(python) || !existsSync(nativeGraph))(
    'Is Grayscale answers with the type that Conditional takes as its Condition',
    () => {
        const definitions = parseFunctionDefinitions(backendNodes().nodes);
        const definition = (schemaId: string) =>
            definitions.get(schemaId as SchemaId) as FunctionDefinition;
        const { outputs, outputErrors } = FunctionInstance.fromPartialInputs(
            definition('chainner:image:is_grayscale'),
            () => undefined
        );
        expect(outputErrors).toEqual([]);
        const answer = prettyPrintType(outputs.get(0 as OutputId) as NonNeverType);
        expect(answer).toBe('bool');
        expect(
            prettyPrintType(
                definition('chainner:utility:conditional').inputDefaults.get(
                    0 as InputId
                ) as NonNeverType
            )
        ).toBe(answer);
    },
    300_000
);

// One automatic precision: no model shows Convert To ONNX's Data Type, and only an ONNX known to
// have no record, or not known yet, leaves Build Engine's and Build Program's Precision to the
// user.
test.skipIf(!existsSync(python) || !existsSync(nativeGraph))(
    'Data Type is hidden for every model, and Precision for any recorded ONNX',
    () => {
        const CONVERT = 'chainner:pytorch:convert_to_onnx';
        const { nodes } = backendNodes();
        const definitions = parseFunctionDefinitions(nodes);
        const type = (expression: string) =>
            evaluate(
                parseExpression(new SourceDocument(expression, 'backend nodes')),
                getChainnerScope()
            ) as NonNeverType;
        const shown = (schemaId: string, label: string, connected?: NonNeverType) => {
            const schema = nodes.find((n) => n.schemaId === schemaId);
            const input = schema?.inputs.find((i) => i.label === label);
            if (!schema || !input) throw new Error(`${schemaId} has no ${label}`);
            const condition = getInputCondition(schema, input.id);
            if (!condition) throw new Error(`${label} of ${schemaId} has no condition`);
            const declared = type(schemaId === CONVERT ? 'PyTorchModel' : 'OnnxModel');
            return testInputCondition(
                condition,
                {},
                schema,
                () => connected ?? declared,
                () => connected !== undefined,
                () => undefined
            );
        };

        const converted = (model: NonNeverType) => {
            const definition = definitions.get(CONVERT as SchemaId) as FunctionDefinition;
            const schema = nodes.find((n) => n.schemaId === CONVERT);
            const { outputs, outputErrors } = FunctionInstance.fromPartialInputs(
                definition,
                (id: InputId) => {
                    if (id === 0) return model;
                    const input = schema?.inputs.find((i) => i.id === id);
                    const value = input && 'def' in input ? input.def : undefined;
                    return value === undefined
                        ? undefined
                        : definition.inputDataAdapters.get(id)?.(value as never);
                }
            );
            expect(outputErrors).toEqual([]);
            return {
                onnx: outputs.get(0 as OutputId) as NonNeverType,
                fpMode: prettyPrintType(outputs.get(1 as OutputId) as NonNeverType),
            };
        };

        const models = [
            'PyTorchModel',
            ...['SwinIR', 'ESRGAN'].map((a) => `PyTorchModel { arch: "${a}" }`),
        ];
        expect(shown(CONVERT, 'Data Type')).toBe(false);
        expect(shown(CONVERT, 'Verify')).toBe(true);
        for (const model of models) {
            expect(shown(CONVERT, 'Data Type', type(model)), model).toBe(false);
            expect(shown(CONVERT, 'Verify', type(model)), model).toBe(true);
        }
        // One of the FP16 architectures can fall back; any other is FP32.
        expect(converted(type('PyTorchModel { arch: "ESRGAN" }')).fpMode).toBe('"fp16" | "fp32"');
        expect(converted(type('PyTorchModel { arch: "NAFNet" }')).fpMode).toBe('"fp32"');
        expect(converted(type('PyTorchModel { arch: "DUAL" }')).fpMode).toBe('"mixed"');
        expect(shown(CONVERT, 'Verify', type('PyTorchModel { arch: "DUAL" }'))).toBe(false);

        // Build Engine's and Build Program's Precision, as test_tensorrt_drft.py and
        // test_migraphx_package.py pin it: their packages load only with their GPUs' libraries.
        const precision: Condition = {
            kind: 'type',
            input: 0 as InputId,
            condition: 'OnnxModel { arch: "unrecorded" }',
            ifNotConnected: true,
        };
        const anySchema = nodes[0];
        const precisionShown = (connected?: NonNeverType) =>
            testInputCondition(
                precision,
                {},
                anySchema,
                () => connected ?? type('OnnxModel'),
                () => connected !== undefined,
                () => undefined
            );
        expect(precisionShown()).toBe(true);
        // A file not read yet, a folder's files
        expect(precisionShown(type('OnnxModel'))).toBe(true);
        expect(precisionShown(type('OnnxModel { arch: "unrecorded" }'))).toBe(true);
        for (const arch of ['SwinIR', 'ESRGAN', 'DUAL']) {
            expect(precisionShown(type(`OnnxModel { arch: "${arch}" }`)), arch).toBe(false);
        }
        // Straight from Convert To ONNX, also before its model is known
        for (const model of models) {
            expect(precisionShown(converted(type(model)).onnx), model).toBe(false);
        }
    },
    300_000
);

// What a node refuses when it runs, by sizes and numbers the frontend already knows, its type
// shows while the chain is edited.
const runRefusals = lazy(() => {
    const { nodes } = backendNodes();
    const definitions = parseFunctionDefinitions(nodes);
    return (
        schemaId: string,
        images: Partial<Record<number, NonNeverType>>,
        values: Partial<Record<number, number | string>> = {}
    ) => {
        const definition = definitions.get(schemaId as SchemaId) as FunctionDefinition;
        const schema = nodes.find((n) => n.schemaId === schemaId);
        const { inputErrors, outputErrors } = FunctionInstance.fromPartialInputs(
            definition,
            (id: InputId) => {
                const assigned = images[id];
                if (assigned) return assigned;
                const input = schema?.inputs.find((i) => i.id === id);
                const value = values[id] ?? (input && 'def' in input ? input.def : undefined);
                return value === undefined || value === null
                    ? undefined
                    : definition.inputDataAdapters.get(id)?.(value as never);
            }
        );
        return [...inputErrors, ...outputErrors];
    };
});

test.skipIf(!existsSync(python) || !existsSync(nativeGraph))(
    'Image Metrics shows that its images must have one shape',
    () => {
        const METRICS = 'chainner:image:image_metrics';
        const refusals = runRefusals();
        expect(refusals(METRICS, { 0: image(30, 20, 3), 1: image(30, 20, 3) })).toEqual([]);
        for (const other of [image(31, 20, 3), image(30, 21, 3), image(30, 20, 4)]) {
            expect(refusals(METRICS, { 0: image(30, 20, 3), 1: other })).not.toEqual([]);
        }
    },
    300_000
);

test.skipIf(!existsSync(python) || !existsSync(nativeGraph))(
    'Merge Channels shows that its images must have one size',
    () => {
        const MERGE = 'chainner:image:merge_channels';
        const refusals = runRefusals();
        const gray = image(30, 20, 1);
        expect(refusals(MERGE, { 0: gray, 1: gray })).toEqual([]);
        expect(refusals(MERGE, { 0: gray, 1: image(31, 20, 1) })).not.toEqual([]);
        expect(refusals(MERGE, { 0: gray, 2: image(30, 21, 1) })).not.toEqual([]);
        expect(refusals(MERGE, { 0: gray, 3: image(29, 20, 3) })).not.toEqual([]);
    },
    300_000
);

test.skipIf(!existsSync(python) || !existsSync(nativeGraph))(
    'Average Color Fix shows that its reference must end up smaller than the image',
    () => {
        const FIX = 'chainner:image:average_color_fix';
        const refusals = runRefusals();
        const reference = image(100, 100, 3);
        // at 12.5 %, the reference is resized to 13x13
        expect(refusals(FIX, { 0: image(14, 14, 3), 1: reference }, { 2: 12.5 })).toEqual([]);
        expect(refusals(FIX, { 0: image(13, 14, 3), 1: reference }, { 2: 12.5 })).not.toEqual([]);
        expect(refusals(FIX, { 0: image(14, 13, 3), 1: reference }, { 2: 12.5 })).not.toEqual([]);
        expect(refusals(FIX, { 0: reference, 1: reference }, { 2: 100 })).not.toEqual([]);
        // the smallest reference is one pixel
        expect(refusals(FIX, { 0: image(2, 2, 3), 1: reference }, { 2: 0 })).toEqual([]);
        expect(refusals(FIX, { 0: image(1, 1, 3), 1: reference }, { 2: 0 })).not.toEqual([]);
    },
    300_000
);

test.skipIf(!existsSync(python) || !existsSync(nativeGraph))(
    'Descale shows that its target must fit the image',
    () => {
        const DESCALE = 'chainner:image:descale';
        const refusals = runRefusals();
        const source = { 0: image(1920, 1080, 3) };
        expect(refusals(DESCALE, source, { 1: 1280, 2: 720 })).toEqual([]);
        expect(refusals(DESCALE, source, { 1: 1920, 2: 1080 })).toEqual([]);
        expect(refusals(DESCALE, source, { 1: 1921, 2: 720 })).not.toEqual([]);
        expect(refusals(DESCALE, source, { 1: 1280, 2: 1081 })).not.toEqual([]);
    },
    300_000
);

test.skipIf(!existsSync(python) || !existsSync(nativeGraph))(
    'Log To Linear shows that Black and White must differ',
    () => {
        const LOG = 'chainner:image:log2lin';
        const refusals = runRefusals();
        const source = { 0: image(30, 20, 3) };
        expect(refusals(LOG, source)).toEqual([]);
        expect(refusals(LOG, source, { 1: 400, 2: 400 })).not.toEqual([]);
        expect(refusals(LOG, source, { 1: 400, 2: 400.5 })).toEqual([]);
    },
    300_000
);

test.skipIf(!existsSync(python) || !existsSync(nativeGraph))(
    'Split Spritesheet shows that its rows and columns must divide the image',
    () => {
        const SPLIT = 'chainner:image:split_spritesheet';
        const refusals = runRefusals();
        const sheet = { 0: image(30, 20, 3) };
        expect(refusals(SPLIT, sheet, { 1: 2, 2: 3 })).toEqual([]);
        expect(refusals(SPLIT, sheet, { 1: 3, 2: 3 })).not.toEqual([]);
        expect(refusals(SPLIT, sheet, { 1: 2, 2: 7 })).not.toEqual([]);
        expect(refusals(SPLIT, sheet, { 1: 40, 2: 1 })).not.toEqual([]);
    },
    300_000
);

// A value that an input's own controls offer, the same input's conversion keeps: a precision too
// coarse for the input's range or step rounds it away, and its type refuses the slider's end.
test.skipIf(!existsSync(python) || !existsSync(nativeGraph))(
    'Every number input accepts the values of its own range and step',
    () => {
        const refusals = runRefusals();
        const problems: string[] = [];
        for (const { schemaId, inputs } of backendNodes().nodes) {
            for (const input of inputs) {
                if (input.kind === 'number' || input.kind === 'slider') {
                    const name = `${schemaId} ${input.label}`;
                    for (const value of [input.min, input.max, input.def]) {
                        if (typeof value === 'number') {
                            const refused = refusals(schemaId, {}, { [input.id]: value });
                            if (refused.some((e) => 'inputId' in e && e.inputId === input.id)) {
                                problems.push(`${name} refuses ${value}`);
                            }
                        }
                    }
                    if (input.precision < 100 && input.controlsStep < 10 ** -input.precision) {
                        problems.push(`${name} steps by less than its precision`);
                    }
                }
            }
        }
        expect(problems).toEqual([]);
    },
    300_000
);

test.skipIf(!existsSync(python) || !existsSync(nativeGraph))(
    'Planes to RGB and RGB to Planes show which planes and matrices they refuse',
    () => {
        const refusals = runRefusals();
        const TO_RGB = 'chainner:image:planes_to_rgb';
        const y = image(30, 21, 1);
        const planes = (u: NonNeverType, v: NonNeverType) => ({ 0: y, 1: u, 2: v });
        const half = image(15, 11, 1);
        expect(refusals(TO_RGB, planes(y, y))).toEqual([]);
        expect(refusals(TO_RGB, planes(half, half))).toEqual([]);
        expect(refusals(TO_RGB, planes(image(8, 6, 1), image(8, 6, 1)))).toEqual([]);
        expect(refusals(TO_RGB, planes(half, y))).not.toEqual([]);
        expect(refusals(TO_RGB, planes(image(7, 11, 1), image(7, 11, 1)))).not.toEqual([]);
        expect(refusals(TO_RGB, planes(image(15, 40, 1), image(15, 40, 1)))).not.toEqual([]);

        const matrix = (value: string) => ({ 4: value });
        expect(refusals(TO_RGB, planes(half, half), matrix('bt709'))).toEqual([]);
        expect(refusals(TO_RGB, planes(y, y), matrix('gbr'))).toEqual([]);
        expect(refusals(TO_RGB, planes(half, half), matrix('gbr'))).not.toEqual([]);
        expect(refusals(TO_RGB, planes(y, y), matrix('ictcp'))).not.toEqual([]);
        expect(refusals(TO_RGB, planes(y, y), matrix('bt2020c'))).not.toEqual([]);

        const TO_PLANES = 'chainner:image:rgb_to_planes';
        const rgb = { 0: image(30, 21, 3) };
        const choice = (m: string, subsampling: string) => ({ 2: m, 7: subsampling });
        expect(refusals(TO_PLANES, rgb, choice('auto', '420'))).toEqual([]);
        expect(refusals(TO_PLANES, rgb, choice('gbr', '444'))).toEqual([]);
        expect(refusals(TO_PLANES, rgb, choice('gbr', '420'))).not.toEqual([]);
        expect(refusals(TO_PLANES, rgb, choice('gbr', '422'))).not.toEqual([]);
        expect(refusals(TO_PLANES, rgb, choice('ictcp', '444'))).not.toEqual([]);
        expect(refusals(TO_PLANES, rgb, choice('bt2020c', '420'))).not.toEqual([]);
    },
    300_000
);
