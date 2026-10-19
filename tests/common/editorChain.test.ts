import { Edge, Node } from 'reactflow';
import { expect, test } from 'vitest';
import {
    CategoryId,
    EdgeData,
    InputId,
    NodeData,
    NodeGroupId,
    NodeSchema,
    NumberInput,
    SchemaId,
} from '../../src/common/common-types';
import {
    formatDroppedNodes,
    missingSchemata,
    toEditorChain,
} from '../../src/common/nodes/editorChain';
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
    outputs: [],
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

// A chain as it is pasted from another chaiNNer: an unknown node, and a known node that lacks an
// input (Height) and has a null one (Width).
const pastedNodes = [
    node('a', DESCALE, { 1: null, 3: 'kept' }),
    node('b', IS_GRAYSCALE, {}),
    node('c', IS_GRAYSCALE, {}),
    node('d', DESCALE, { 1: 1920, 2: 1080 }),
];
const pastedEdges = [edge('b', 'a'), edge('d', 'a')];

test('Unknown nodes are dropped with their edges', () => {
    const chain = toEditorChain(schemata, pastedNodes, pastedEdges);

    expect(chain.nodes.map((n) => n.id)).toEqual(['a', 'd']);
    expect(chain.edges.map((e) => e.id)).toEqual(['d-a']);
    expect(chain.edges[0].animated).toBe(false);
    expect(chain.droppedNodes).toEqual([IS_GRAYSCALE, IS_GRAYSCALE]);
    // an incomplete load leaves the chain unsaved
    expect(chain.complete).toBe(false);
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
