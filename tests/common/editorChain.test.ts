import { Edge, Node } from 'reactflow';
import { expect, test } from 'vitest';
import {
    CategoryId,
    DropDownInput,
    EdgeData,
    InputId,
    NodeData,
    NodeGroupId,
    NodeSchema,
    NumberInput,
    OutputId,
    SchemaId,
} from '../../src/common/common-types';
import {
    formatDroppedNodes,
    formatReplacedInputs,
    missingSchemata,
    toEditorChain,
} from '../../src/common/nodes/editorChain';
import { toBackendJson } from '../../src/common/nodes/toBackendJson';
import { SchemaMap } from '../../src/common/SchemaMap';

const DESCALE = 'chainner:image:descale' as SchemaId;
const IS_GRAYSCALE = 'chainner:image:is_grayscale' as SchemaId;

const numberInput = (id: number, label: string, def: number): NumberInput => ({
    id: id as InputId,
    type: 'number',
    conversions: [],
    kind: 'number',
    label,
    optional: false,
    hasHandle: true,
    hint: false,
    suggest: false,
    def,
    precision: 0,
    controlsStep: 1,
    hideTrailingZeros: false,
    labelStyle: 'default',
});
const descaleSchema: NodeSchema = {
    name: 'Descale',
    category: 'image_dimension' as CategoryId,
    nodeGroup: 'resize' as NodeGroupId,
    description: '',
    seeAlso: [],
    icon: '',
    kind: 'regularNode',
    inputs: [numberInput(1, 'Width', 1280), numberInput(2, 'Height', 720)],
    outputs: [
        {
            id: 0 as OutputId,
            type: 'Image',
            label: 'Image',
            kind: 'generic',
            hasHandle: true,
            suggest: false,
        },
    ],
    groupLayout: [1 as InputId, 2 as InputId],
    iteratorInputs: [],
    iteratorOutputs: [],
    suggestions: [],
    schemaId: DESCALE,
    hasSideEffects: false,
    deprecated: false,
    features: [],
};
const schemata = new SchemaMap([descaleSchema]);

// inputData as parsed from JSON, which may hold null
const node = (
    id: string,
    schemaId: SchemaId,
    inputData: Record<number, unknown>
): Node<NodeData> => ({
    id,
    type: 'regularNode',
    position: { x: 0, y: 0 },
    data: { id, schemaId, inputData: inputData as NodeData['inputData'] },
});
const edge = (source: string, target: string): Edge<EdgeData> => ({
    id: `${source}-${target}`,
    source,
    sourceHandle: `${source}-0`,
    target,
    targetHandle: `${target}-1`,
    type: 'main',
    animated: true,
});

// Node ids are UUIDs, which handles start with (parseSourceHandle).
const A = '00000000-0000-0000-0000-00000000000a';
const B = '00000000-0000-0000-0000-00000000000b';
const C = '00000000-0000-0000-0000-00000000000c';
const D = '00000000-0000-0000-0000-00000000000d';

// A chain as it is pasted from another chaiNNer: an unknown node, and a known node that lacks an
// input (Height) and has a null one (Width).
const pastedNodes = [
    node(A, DESCALE, { 1: null, 3: 'kept' }),
    node(B, IS_GRAYSCALE, {}),
    node(C, IS_GRAYSCALE, {}),
    node(D, DESCALE, { 1: 1920, 2: 1080 }),
];
const pastedEdges = [edge(B, A), edge(D, A)];

test('Unknown nodes are dropped with their edges', () => {
    const chain = toEditorChain(schemata, pastedNodes, pastedEdges);

    expect(chain.nodes.map((n) => n.id)).toEqual([A, D]);
    expect(chain.edges.map((e) => e.id)).toEqual([`${D}-${A}`]);
    expect(chain.edges[0].animated).toBe(false);
    expect(chain.droppedNodes).toEqual([IS_GRAYSCALE, IS_GRAYSCALE]);
    // the edge of a dropped node goes with it
    expect(chain.droppedEdges).toBe(0);
    // an incomplete load leaves the chain unsaved
    expect(chain.complete).toBe(false);
});

test('Edges that do not fit their nodes are dropped', () => {
    // A chain file is untrusted: each of these made the editor's type evaluation throw after
    // every change (a cycle overflowed its stack).
    const kept = edge(D, A);
    const broken: Edge<EdgeData>[] = [
        { ...edge(A, D), id: 'no handles', sourceHandle: undefined, targetHandle: undefined },
        { ...edge(A, D), id: 'handle of another node', sourceHandle: `${B}-0` },
        { ...edge(A, D), id: 'missing input', targetHandle: `${D}-99` },
        { ...edge(A, D), id: 'missing output', sourceHandle: `${A}-5` },
        { ...edge(A, A), id: 'self loop', targetHandle: `${A}-2` },
    ];
    const cycle = { ...edge(A, D), id: 'closes a cycle', targetHandle: `${D}-2` };
    const second = { ...edge(A, A), id: 'second into one input', targetHandle: `${A}-1` };
    const chain = toEditorChain(
        schemata,
        [pastedNodes[0], pastedNodes[3]],
        [kept, ...broken, cycle, second]
    );

    expect(chain.edges.map((e) => e.id)).toEqual([kept.id]);
    expect(chain.droppedEdges).toBe(broken.length + 2);
    expect(chain.droppedNodes).toEqual([]);
    expect(chain.complete).toBe(false);
});

test('A node without a position or input data gets the origin and its defaults', () => {
    const bare = {
        ...node(A, DESCALE, {}),
        position: undefined,
        data: { id: A, schemaId: DESCALE, inputData: null },
    } as unknown as Node<NodeData>;
    const chain = toEditorChain(schemata, [bare], []);

    expect(chain.nodes[0].position).toEqual({ x: 0, y: 0 });
    expect(chain.nodes[0].data.inputData).toEqual({ 1: 1280, 2: 720 });
    expect(chain.complete).toBe(true);
});

test('An input value that is no number or text gets its default', () => {
    // e.g. an object where a file path goes: the type evaluation of the path threw
    const chain = toEditorChain(
        schemata,
        [node(A, DESCALE, { 1: { path: 'x' }, 2: [1], 3: true })],
        []
    );

    expect(chain.nodes[0].data.inputData).toEqual({ 1: 1280, 2: 720 });
});

const dropdown: DropDownInput = {
    id: 3 as InputId,
    type: 'any',
    conversions: [],
    kind: 'dropdown',
    label: 'Mode',
    optional: false,
    hasHandle: false,
    hint: false,
    suggest: false,
    def: 'near',
    options: [
        { option: 'Near', value: 'near' },
        { option: 'Far', value: 'far' },
        { option: 'Two', value: 2 },
    ],
    preferredStyle: 'dropdown',
    labelStyle: 'default',
    groups: [],
};
const withDropdown = new SchemaMap([
    { ...descaleSchema, inputs: [...descaleSchema.inputs, dropdown] },
]);

test('A value that its dropdown or number input cannot take gets the default', () => {
    // The dropdown of a collapsed node or in a hidden group never shows to correct the value, and
    // the backend rejects it: a tile size option that was removed, or a value of an older input
    // with this id, as in Blend Images from before its overlay position.
    const inputsOf = (inputData: Record<number, unknown>) =>
        toEditorChain(withDropdown, [node(A, DESCALE, inputData)], []).nodes[0].data.inputData;

    expect(inputsOf({ 3: 100 })).toMatchObject({ 3: 'near' });
    expect(inputsOf({ 3: 'D:models' })).toMatchObject({ 3: 'near' });
    // the value as the option has it, not its text
    expect(inputsOf({ 3: '2' })).toMatchObject({ 3: 'near' });
    expect(inputsOf({ 3: 2 })).toMatchObject({ 3: 2 });
    expect(inputsOf({ 3: 'far' })).toMatchObject({ 3: 'far' });

    expect(inputsOf({ 1: '', 2: 'x' })).toMatchObject({ 1: 1280, 2: 720 });
    expect(inputsOf({ 1: 5 })).toMatchObject({ 1: 5 });
    // no such input: kept, as a later version of the chain may have it
    expect(inputsOf({ 9: 'kept' })).toMatchObject({ 9: 'kept' });
});

test('A number in text in a number input becomes that number', () => {
    // a chain from before its inputs' values were numbers, or edited by hand
    const inputsOf = (inputData: Record<number, unknown>) =>
        toEditorChain(schemata, [node(A, DESCALE, inputData)], []).nodes[0].data.inputData;

    expect(inputsOf({ 1: '2', 2: ' 1.5 ' })).toEqual({ 1: 2, 2: 1.5 });

    // no number
    expect(inputsOf({ 1: '', 2: 'x' })).toEqual({ 1: 1280, 2: 720 });
    expect(inputsOf({ 1: 'Infinity', 2: Infinity })).toEqual({ 1: 1280, 2: 720 });
});

test('A number in an input of text gets the default', () => {
    // a file input parses its value as a path when it shows, which throws on a number
    const textInput = (id: number, kind: string, def?: string) =>
        ({ id, kind, def, label: kind, optional: true } as unknown as NodeSchema['inputs'][number]);
    const withTexts = new SchemaMap([
        {
            ...descaleSchema,
            inputs: [
                textInput(3, 'file'),
                textInput(4, 'directory'),
                textInput(5, 'text', 'text'),
                textInput(6, 'color', '{}'),
            ],
        },
    ]);
    const inputsOf = (inputData: Record<number, unknown>) =>
        toEditorChain(withTexts, [node(A, DESCALE, inputData)], []).nodes[0].data.inputData;

    expect(inputsOf({ 3: 5, 4: 5, 5: 5, 6: 5 })).toEqual({
        3: undefined,
        4: undefined,
        5: 'text',
        6: '{}',
    });
    expect(inputsOf({ 3: 'a.png', 4: 'C:\\a', 5: 'b', 6: '{"a":1}' })).toEqual({
        3: 'a.png',
        4: 'C:\\a',
        5: 'b',
        6: '{"a":1}',
    });
});

test('A node of a damaged file runs and takes input by its schema and id', () => {
    const generator = 'chainner:image:load_images' as SchemaId;
    const withGenerator = new SchemaMap([
        descaleSchema,
        { ...descaleSchema, schemaId: generator, kind: 'generator' },
    ]);
    const nodeOf = (changes: Partial<Node<NodeData>>, schemaId = DESCALE) =>
        toEditorChain(withGenerator, [{ ...node(A, schemaId, {}), ...changes }], []).nodes[0];

    // toBackendJson throws on a node without a type, which is the alert on every run
    const untyped = nodeOf({ type: undefined });
    expect(untyped.type).toBe('regularNode');
    expect(() => toBackendJson([untyped], [], withGenerator)).not.toThrow();

    expect(nodeOf({ type: 'unknown' }).type).toBe('regularNode');
    expect(nodeOf({ type: 'regularNode' }, generator).type).toBe('generator');
    expect(nodeOf({ type: 'breakPoint' }).type).toBe('breakPoint');

    const withoutId = { schemaId: DESCALE, inputData: {} } as unknown as NodeData;
    expect(nodeOf({ data: withoutId }).data.id).toBe(A);
});

test('A value that was replaced is listed, and the chain is not complete', () => {
    const replaced = toEditorChain(
        withDropdown,
        [node(A, DESCALE, { 1: 'x', 3: 100 }), node(D, DESCALE, { 3: 100 })],
        []
    );
    expect(replaced.replacedInputs).toEqual([
        { node: 'Descale', input: 'Width', value: 'x' },
        { node: 'Descale', input: 'Mode', value: 100 },
        { node: 'Descale', input: 'Mode', value: 100 },
    ]);
    expect(replaced.complete).toBe(false);
    expect(formatReplacedInputs(replaced.replacedInputs)).toBe(
        '- Descale: Width held "x" (1 node)\n- Descale: Mode held 100 (2 nodes)'
    );

    // Not replaced: no value, a value for no input of the node, one that converts, and a value of
    // a connected input, which does not use it (Save Video's FPS "" of an old chain).
    const kept = toEditorChain(
        withDropdown,
        [node(A, DESCALE, { 1: '', 2: '5', 3: null, 9: 'kept' }), node(D, DESCALE, {})],
        [edge(D, A)]
    );
    expect(kept.nodes[0].data.inputData).toMatchObject({ 1: 1280, 2: 5, 3: 'near', 9: 'kept' });
    expect(kept.replacedInputs).toEqual([]);
    expect(kept.complete).toBe(true);
});

test('Missing and null inputs get their defaults', () => {
    const chain = toEditorChain(schemata, pastedNodes, pastedEdges);

    expect(chain.nodes[0].data.inputData).toEqual({ 1: 1280, 2: 720, 3: 'kept' });
    expect(chain.nodes[1].data.inputData).toEqual({ 1: 1920, 2: 1080 });
});

test('A chain with only known nodes is complete', () => {
    const chain = toEditorChain(schemata, [pastedNodes[0], pastedNodes[3]], [pastedEdges[1]]);

    expect(chain.droppedNodes).toEqual([]);
    // a complete load is the saved state of the file
    expect(chain.complete).toBe(true);
});

test('The alert lists every dropped schema with its count', () => {
    expect(
        formatDroppedNodes([
            IS_GRAYSCALE,
            'chainner:utility:math_expression' as SchemaId,
            IS_GRAYSCALE,
        ])
    ).toBe('- chainner:image:is_grayscale (2 nodes)\n- chainner:utility:math_expression (1 node)');
});

test('A chain that the node list lacks a node of names each missing schema once', () => {
    // opened with the node list of the last start, it waits for the backend's list
    expect(missingSchemata(schemata, pastedNodes)).toEqual([IS_GRAYSCALE]);
    expect(missingSchemata(schemata, [pastedNodes[0], pastedNodes[3]])).toEqual([]);
});
