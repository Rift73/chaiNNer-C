import { Edge, Node } from 'reactflow';
import { expect, test } from 'vitest';
import {
    CategoryId,
    EdgeData,
    GenericInput,
    InputData,
    InputId,
    InputValue,
    NodeData,
    NodeGroupId,
    NodeSchema,
    OutputId,
    SchemaId,
} from '../../src/common/common-types';
import { SchemaMap } from '../../src/common/SchemaMap';
import { stringifySourceHandle, stringifyTargetHandle } from '../../src/common/util';
import { computeInputHashes } from '../../src/renderer/hooks/useInputHashes';

const JOIN = 'test:hashes:join' as SchemaId;

const input = (id: number): GenericInput => ({
    id: id as InputId,
    type: 'any',
    conversions: [],
    kind: 'generic',
    label: `Input ${id}`,
    optional: true,
    hasHandle: true,
    hint: false,
    suggest: false,
});
const joinSchema: NodeSchema = {
    name: 'Join',
    category: 'test' as CategoryId,
    nodeGroup: 'test' as NodeGroupId,
    description: '',
    seeAlso: [],
    icon: '',
    kind: 'regularNode',
    inputs: [input(0), input(1)],
    outputs: [
        {
            id: 0 as OutputId,
            type: 'any',
            label: 'Text',
            kind: 'generic',
            hasHandle: true,
            suggest: false,
        },
    ],
    groupLayout: [0 as InputId, 1 as InputId],
    iteratorInputs: [],
    iteratorOutputs: [],
    suggestions: [],
    schemaId: JOIN,
    hasSideEffects: false,
    deprecated: false,
    features: [],
};
const schemata = new SchemaMap([joinSchema]);

// handles keep the node ID as their first 36 characters
const id = (n: number) => `00000000-0000-4000-8000-00000000000${n}`;
/** Input values by input ID. */
type Values = Readonly<Record<number, InputValue>>;
const node = (n: number, values: Values, x = 0): Node<NodeData> => ({
    id: id(n),
    position: { x, y: 0 },
    data: { id: id(n), schemaId: JOIN, inputData: values as InputData },
});
const edge = (source: number, target: number, inputId: number): Edge<EdgeData> => ({
    id: `${source}->${target}`,
    source: id(source),
    sourceHandle: stringifySourceHandle({ nodeId: id(source), outputId: 0 as OutputId }),
    target: id(target),
    targetHandle: stringifyTargetHandle({ nodeId: id(target), inputId: inputId as InputId }),
});

// 1 -> 2 (input 0), 3 unconnected
const nodes = [node(1, { 0: 'a', 1: 2 }), node(2, { 1: 'b' }), node(3, { 0: 'c' })];
const edges = [edge(1, 2, 0)];
const hashes = computeInputHashes(nodes, edges, schemata);

test('a node that only moved keeps its hash', () => {
    const moved = nodes.map((n) => node(Number(n.id.slice(-1)), { ...n.data.inputData }, 50));
    expect(computeInputHashes(moved, edges, schemata)).toEqual(hashes);
});

test('a changed input changes the hash of its node and the nodes after it', () => {
    const changed = computeInputHashes(
        [node(1, { 0: 'a', 1: 3 }), nodes[1], nodes[2]],
        edges,
        schemata
    );
    expect(changed.get(id(1))).not.toBe(hashes.get(id(1)));
    expect(changed.get(id(2))).not.toBe(hashes.get(id(2)));
    expect(changed.get(id(3))).toBe(hashes.get(id(3)));
});

test('a node keeps its hash only while its data and connected hashes stay', () => {
    // node data is replaced on every change, never changed in place; a copy of the chain (new
    // data objects) computes every hash again
    const fresh = (chainNodes: Node<NodeData>[], chainEdges: Edge<EdgeData>[]) =>
        computeInputHashes(structuredClone(chainNodes), chainEdges, schemata);
    const steps: [Node<NodeData>[], Edge<EdgeData>[]][] = [
        [nodes, edges],
        // node 1 changed: node 2 keeps its data, but its connected hash changed
        [[node(1, { 0: 'z', 1: 2 }), nodes[1], nodes[2]], edges],
        [nodes, edges],
        // node 2 without its connection: its own (missing) value again
        [nodes, []],
        // node 3 into node 2's other input
        [nodes, [...edges, edge(3, 2, 1)]],
        // another node list: the same hashes, made again
        [nodes, edges],
    ];
    for (const [chainNodes, chainEdges] of steps) {
        expect(computeInputHashes(chainNodes, chainEdges, schemata)).toEqual(
            fresh(chainNodes, chainEdges)
        );
    }
    expect(computeInputHashes(nodes, edges, new SchemaMap(schemata.schemata))).toEqual(hashes);
});

test('different inputs never read as the same inputs', () => {
    const hashOf = (values: Values, chainEdges: Edge<EdgeData>[] = []) =>
        computeInputHashes([nodes[0], node(2, values)], chainEdges, schemata).get(id(2));

    // a separator inside a value
    expect(hashOf({ 0: 'a;b', 1: 'c' })).not.toBe(hashOf({ 0: 'a', 1: 'b;c' }));
    // no value and an empty one
    expect(hashOf({ 0: '' })).not.toBe(hashOf({}));
    // a connection and a value that reads like the connected node's hash
    expect(hashOf({}, [edge(1, 2, 0)])).not.toBe(hashOf({ 0: hashes.get(id(1)) }));
});
