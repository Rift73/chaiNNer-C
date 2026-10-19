import { Type, literal } from '@chainner/navi';
import { Edge, Node } from 'reactflow';
import { expect, test } from 'vitest';
import {
    CategoryId,
    EdgeData,
    GenericInput,
    InputId,
    IterOutputId,
    NodeData,
    NodeGroupId,
    NodeKind,
    NodeSchema,
    OutputId,
    SchemaId,
} from '../../src/common/common-types';
import { ChainLineage } from '../../src/common/nodes/lineage';
import { parseFunctionDefinitions } from '../../src/common/nodes/parseFunctionDefinitions';
import { TypeState } from '../../src/common/nodes/TypeState';
import { SchemaMap } from '../../src/common/SchemaMap';
import { EMPTY_MAP, stringifySourceHandle, stringifyTargetHandle } from '../../src/common/util';

// The type pass (ChainLineage and TypeState) keeps its objects for a chain whose lineages and
// types did not change, so that nothing that depends on them redoes its work.

const LOAD_IMAGES = 'test:type-pass:load_images' as SchemaId;
const LOAD_VIDEO = 'test:type-pass:load_video' as SchemaId;
const RESIZE = 'test:type-pass:resize' as SchemaId;

const input: GenericInput = {
    id: 0 as InputId,
    type: 'any',
    conversions: [],
    kind: 'generic',
    label: 'Image',
    optional: false,
    hasHandle: true,
    hint: false,
    suggest: false,
};
const makeSchema = (schemaId: SchemaId, kind: NodeKind): NodeSchema => ({
    name: schemaId,
    category: 'test' as CategoryId,
    nodeGroup: 'test' as NodeGroupId,
    description: '',
    seeAlso: [],
    icon: '',
    kind,
    inputs: kind === 'generator' ? [] : [input],
    outputs: [
        {
            id: 0 as OutputId,
            type: 'any',
            label: 'Image',
            kind: 'generic',
            hasHandle: true,
            suggest: false,
        },
    ],
    groupLayout: kind === 'generator' ? [] : [input.id],
    iteratorInputs: [],
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
const schemata = new SchemaMap([
    makeSchema(LOAD_IMAGES, 'generator'),
    makeSchema(LOAD_VIDEO, 'generator'),
    makeSchema(RESIZE, 'regularNode'),
]);

const node = (id: string, schemaId: SchemaId, x = 0): Node<NodeData> => ({
    id,
    position: { x, y: 0 },
    data: { id, schemaId, inputData: { [input.id]: x } },
});
const edge = (source: string, target: string, id = `${source}->${target}`): Edge<EdgeData> => ({
    id,
    source,
    sourceHandle: stringifySourceHandle({ nodeId: source, outputId: 0 as OutputId }),
    target,
    targetHandle: stringifyTargetHandle({ nodeId: target, inputId: input.id }),
});

// node IDs are UUIDs: handles keep the node ID as their first 36 characters
const LOAD = '00000000-0000-4000-8000-000000000001';
const A = '00000000-0000-4000-8000-000000000002';
const B = '00000000-0000-4000-8000-000000000003';
const C = '00000000-0000-4000-8000-000000000004';

// Load Images -> Resize A -> Resize B
const nodes = [node(LOAD, LOAD_IMAGES), node(A, RESIZE), node(B, RESIZE)];
const edges = [edge(LOAD, A), edge(A, B)];
const iteratedBy = (lineage: ChainLineage, nodeId: string) =>
    lineage.getIteratingSchemaId({ nodeId, inputId: input.id });

test('the same chain keeps its lineage object', () => {
    const lineage = ChainLineage.create(schemata, nodes, edges);
    expect(iteratedBy(lineage, B)).toBe(LOAD_IMAGES);

    // nodes moved and got other input values; new node and edge objects, new edge IDs
    const moved = nodes.map((n) => node(n.id, n.data.schemaId, 100));
    const copied = edges.map((e) => edge(e.source, e.target, `${e.id}-copy`));

    const again = ChainLineage.create(schemata, moved, copied, lineage);
    expect(again).toBe(lineage);
    expect(iteratedBy(again, B)).toBe(LOAD_IMAGES);
});

test('another chain gets a new lineage', () => {
    const lineage = ChainLineage.create(schemata, nodes, edges);

    const otherSchema = ChainLineage.create(
        schemata,
        [node(LOAD, LOAD_VIDEO), ...nodes.slice(1)],
        edges,
        lineage
    );
    expect(otherSchema).not.toBe(lineage);
    expect(iteratedBy(otherSchema, B)).toBe(LOAD_VIDEO);

    const lessEdges = ChainLineage.create(schemata, nodes, edges.slice(0, 1), lineage);
    expect(lessEdges).not.toBe(lineage);
    expect(iteratedBy(lessEdges, B)).toBeUndefined();

    const moreNodes = ChainLineage.create(schemata, [...nodes, node(C, RESIZE)], edges, lineage);
    expect(moreNodes).not.toBe(lineage);

    const reloaded = new SchemaMap(schemata.schemata);
    expect(ChainLineage.create(reloaded, nodes, edges, lineage)).not.toBe(lineage);
});

const functionDefinitions = parseFunctionDefinitions(schemata.schemata);
const typeState = (
    chainNodes: readonly Node<NodeData>[],
    chainEdges: readonly Edge<EdgeData>[],
    previous?: TypeState,
    outputNarrowing: ReadonlyMap<string, ReadonlyMap<OutputId, Type>> = EMPTY_MAP
): TypeState =>
    TypeState.create(
        new Map(chainNodes.map((n) => [n.id, n])),
        chainEdges,
        outputNarrowing,
        EMPTY_MAP,
        functionDefinitions,
        undefined,
        previous
    );

test('a chain whose types did not change keeps its type state', () => {
    const types = typeState(nodes, edges);

    const moved = nodes.map((n) => node(n.id, n.data.schemaId, 100));
    const copied = edges.map((e) => edge(e.source, e.target, `${e.id}-copy`));
    expect(typeState(moved, copied, types)).toBe(types);
});

test('a changed type, connection or node gives a new type state', () => {
    const types = typeState(nodes, edges);

    // Resize A's output is narrowed: A's and B's (its input) types change, Load's do not
    const narrowed = typeState(
        nodes,
        edges,
        types,
        new Map([[A, new Map([[0 as OutputId, literal(2)]])]])
    );
    expect(narrowed).not.toBe(types);
    expect(narrowed.functions.get(LOAD)).toBe(types.functions.get(LOAD));
    expect(narrowed.functions.get(A)).not.toBe(types.functions.get(A));
    expect(narrowed.functions.get(B)?.inputs.get(input.id)).toEqual(literal(2));

    const lessEdges = typeState(nodes, edges.slice(0, 1), types);
    expect(lessEdges).not.toBe(types);
    expect(lessEdges.edges.isInputConnected(B, input.id)).toBe(false);

    const lessNodes = typeState(nodes.slice(0, 2), edges.slice(0, 1), types);
    expect(lessNodes).not.toBe(types);
    expect(lessNodes.functions.has(B)).toBe(false);
});
