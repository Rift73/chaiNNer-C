import { Edge, Node } from 'reactflow';
import { expect, test } from 'vitest';
import {
    CategoryId,
    Condition,
    EdgeData,
    GenericInput,
    GroupId,
    InputId,
    IterOutputId,
    NodeData,
    NodeGroupId,
    NodeKind,
    NodeSchema,
    Output,
    OutputId,
    SchemaId,
} from '../../src/common/common-types';
import { checkGroupInputs } from '../../src/common/group-inputs';
import { checkNodeValidity } from '../../src/common/nodes/checkNodeValidity';
import {
    isTautology,
    simplifyCondition,
    testInputCondition,
} from '../../src/common/nodes/condition';
import { ChainLineage } from '../../src/common/nodes/lineage';
import { SchemaMap } from '../../src/common/SchemaMap';
import { stringifySourceHandle, stringifyTargetHandle } from '../../src/common/util';

const LOAD_VIDEO = 'chainner:image:load_video' as SchemaId;
const LOAD_IMAGES = 'chainner:image:load_images' as SchemaId;
const RESIZE = 'test:condition:resize' as SchemaId;
const UPSCALE = 'test:condition:upscale' as SchemaId;

const IMAGE = 0 as InputId;
const STRENGTH = 1 as InputId;
const iteratedByVideo: Condition = { kind: 'iterated', input: IMAGE, source: LOAD_VIDEO };

const genericInput = (id: number, label: string, hasHandle = true): GenericInput => ({
    id: id as InputId,
    type: 'any',
    conversions: [],
    kind: 'generic',
    label,
    optional: false,
    hasHandle,
    hint: false,
    suggest: false,
});
const output = (id: number): Output => ({
    id: id as OutputId,
    type: 'any',
    label: `Output ${id}`,
    kind: 'generic',
    hasHandle: true,
    suggest: false,
});
const makeSchema = (
    schemaId: SchemaId,
    kind: NodeKind,
    inputs: readonly GenericInput[],
    outputCount: number,
    groupLayout: NodeSchema['groupLayout'] = inputs.map((i) => i.id)
): NodeSchema => ({
    name: schemaId,
    category: 'test' as CategoryId,
    nodeGroup: 'test' as NodeGroupId,
    description: '',
    seeAlso: [],
    icon: '',
    kind,
    inputs,
    outputs: Array.from({ length: outputCount }, (_, i) => output(i)),
    groupLayout,
    iteratorInputs: [],
    // a generator's output 0 is its sequence, output 1 is not
    iteratorOutputs:
        kind === 'generator'
            ? [{ id: 0 as IterOutputId, outputs: [0 as OutputId], sequenceType: 'any' }]
            : [],
    suggestions: [],
    schemaId,
    hasSideEffects: false,
    deprecated: false,
    features: [],
});

// Upscale: Image, then Temporal Strength (required) shown only when Image is iterated by
// Load Video.
const upscaleSchema = makeSchema(
    UPSCALE,
    'regularNode',
    [genericInput(0, 'Image'), genericInput(1, 'Temporal Strength')],
    1,
    [
        IMAGE,
        {
            id: 0 as GroupId,
            kind: 'conditional',
            options: { condition: iteratedByVideo },
            items: [STRENGTH],
        },
    ]
);
const schemata = new SchemaMap([
    makeSchema(LOAD_VIDEO, 'generator', [], 2),
    makeSchema(LOAD_IMAGES, 'generator', [], 2),
    makeSchema(RESIZE, 'regularNode', [genericInput(0, 'Image')], 1),
    upscaleSchema,
]);

const id = (n: number) => `00000000-0000-4000-8000-00000000000${n}`;
const node = (n: number, schemaId: SchemaId): Node<NodeData> => ({
    id: id(n),
    position: { x: 0, y: 0 },
    data: { id: id(n), schemaId, inputData: {} },
});
const edge = (source: number, outputId: number, target: number, inputId: number) => {
    const sourceHandle = stringifySourceHandle({
        nodeId: id(source),
        outputId: outputId as OutputId,
    });
    const targetHandle = stringifyTargetHandle({ nodeId: id(target), inputId: inputId as InputId });
    return {
        id: `${sourceHandle}->${targetHandle}`,
        source: id(source),
        sourceHandle,
        target: id(target),
        targetHandle,
    } as Edge<EdgeData>;
};

// Load Video (1) -> Resize (3) -> Upscale (4); Load Images (2) -> Upscale (5); Load Video's
// non-iterated output -> Upscale (6); Upscale (7) unconnected.
const lineage = new ChainLineage(
    schemata,
    [
        node(1, LOAD_VIDEO),
        node(2, LOAD_IMAGES),
        node(3, RESIZE),
        node(4, UPSCALE),
        node(5, UPSCALE),
        node(6, UPSCALE),
        node(7, UPSCALE),
    ],
    [edge(1, 0, 3, 0), edge(3, 0, 4, 0), edge(2, 0, 5, 0), edge(1, 1, 6, 0)]
);
const iteratingSchemaId = (n: number) => (inputId: InputId) =>
    lineage.getIteratingSchemaId({ nodeId: id(n), inputId });

test('getIteratingSchemaId follows the lineage to its generator', () => {
    expect(iteratingSchemaId(4)(IMAGE)).toBe(LOAD_VIDEO);
    expect(iteratingSchemaId(3)(IMAGE)).toBe(LOAD_VIDEO);
    expect(iteratingSchemaId(5)(IMAGE)).toBe(LOAD_IMAGES);
    // a generator's non-iterated output, an unconnected input
    expect(iteratingSchemaId(6)(IMAGE)).toBeUndefined();
    expect(iteratingSchemaId(7)(IMAGE)).toBeUndefined();
    expect(iteratingSchemaId(4)(STRENGTH)).toBeUndefined();
});

test('an iterated condition holds only for its generator', () => {
    const holds = (condition: Condition, n: number) =>
        testInputCondition(
            condition,
            {},
            upscaleSchema,
            () => undefined,
            () => true,
            iteratingSchemaId(n)
        );
    expect([4, 5, 6, 7].map((n) => holds(iteratedByVideo, n))).toEqual([true, false, false, false]);
    const notByVideo: Condition = { kind: 'not', condition: iteratedByVideo };
    expect([4, 5, 6, 7].map((n) => holds(notByVideo, n))).toEqual([false, true, true, true]);
    const either: Condition = {
        kind: 'or',
        items: [iteratedByVideo, { kind: 'iterated', input: IMAGE, source: LOAD_IMAGES }],
    };
    expect([4, 5, 6, 7].map((n) => holds(either, n))).toEqual([true, true, false, false]);
});

test('an iterated condition is no tautology and survives simplification', () => {
    expect(isTautology(iteratedByVideo)).toBe(false);
    expect(isTautology({ kind: 'not', condition: iteratedByVideo })).toBe(false);
    expect(
        simplifyCondition({ kind: 'and', items: [iteratedByVideo, { kind: 'and', items: [] }] })
    ).toEqual(iteratedByVideo);
    expect(
        simplifyCondition({
            kind: 'not',
            condition: { kind: 'not', condition: iteratedByVideo },
        })
    ).toEqual(iteratedByVideo);
});

test('inputs behind an iterated condition are required only while it holds', () => {
    const validity = (n: number, chainLineage: ChainLineage | undefined) =>
        checkNodeValidity({
            schema: upscaleSchema,
            inputData: {},
            connectedInputs: new Set([IMAGE]),
            functionInstance: undefined,
            chainLineage,
            nodeId: id(n),
        });
    expect(validity(4, lineage)).toEqual({
        isValid: false,
        reason: 'Missing required input data: Temporal Strength',
    });
    expect(validity(5, lineage).isValid).toBe(true);
    expect(validity(4, undefined).isValid).toBe(true);
});

test('group checks accept an iterated condition on an input with a handle only', () => {
    const check = (condition: Condition, inputs = upscaleSchema.inputs) => {
        const schema = { ...upscaleSchema, inputs };
        return checkGroupInputs(
            [inputs[1]],
            { id: 0 as GroupId, kind: 'conditional', options: { condition }, items: [STRENGTH] },
            schema
        );
    };
    expect(check(iteratedByVideo).isValid).toBe(true);
    expect(check({ kind: 'iterated', input: 9 as InputId, source: LOAD_VIDEO }).isValid).toBe(
        false
    );
    const handleless = [genericInput(0, 'Image', false), genericInput(1, 'Temporal Strength')];
    expect(check(iteratedByVideo, handleless).isValid).toBe(false);
});
