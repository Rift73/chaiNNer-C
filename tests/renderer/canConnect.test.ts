import { Edge, Node } from 'reactflow';
import { expect, test } from 'vitest';
import {
    CategoryId,
    EdgeData,
    InputId,
    NodeData,
    NodeGroupId,
    NodeSchema,
    OutputId,
    SchemaId,
} from '../../src/common/common-types';
import { ChainLineage } from '../../src/common/nodes/lineage';
import { parseFunctionDefinitions } from '../../src/common/nodes/parseFunctionDefinitions';
import { TypeState } from '../../src/common/nodes/TypeState';
import { SchemaMap } from '../../src/common/SchemaMap';
import { EMPTY_MAP, stringifySourceHandle, stringifyTargetHandle } from '../../src/common/util';
import { canConnect } from '../../src/renderer/helpers/canConnect';

// a node with two inputs and an output of any type
const PASS = 'test:can-connect:pass' as SchemaId;
const schema: NodeSchema = {
    name: 'Pass',
    category: 'test' as CategoryId,
    nodeGroup: 'test' as NodeGroupId,
    description: '',
    seeAlso: [],
    icon: '',
    kind: 'regularNode',
    inputs: [0, 1].map((id) => ({
        id: id as InputId,
        type: 'any',
        conversions: [],
        kind: 'generic',
        label: `In ${id}`,
        optional: true,
        hasHandle: true,
        hint: false,
        suggest: false,
    })),
    outputs: [
        {
            id: 0 as OutputId,
            type: 'any',
            label: 'Out',
            kind: 'generic',
            hasHandle: true,
            suggest: false,
        },
    ],
    groupLayout: [0 as InputId, 1 as InputId],
    iteratorInputs: [],
    iteratorOutputs: [],
    suggestions: [],
    schemaId: PASS,
    hasSideEffects: false,
    deprecated: false,
    features: [],
};
const schemata = new SchemaMap([schema]);
const functionDefinitions = parseFunctionDefinitions(schemata.schemata);

// node IDs are UUIDs: handles keep the node ID as their first 36 characters
const id = (i: number) => `00000000-0000-4000-8000-${i.toString(16).padStart(12, '0')}`;
const node = (i: number): Node<NodeData> => ({
    id: id(i),
    position: { x: 0, y: 0 },
    data: { id: id(i), schemaId: PASS, inputData: {} },
});
const edge = (source: number, target: number, input: number): Edge<EdgeData> => ({
    id: `${source}-${target}-${input}`,
    source: id(source),
    sourceHandle: stringifySourceHandle({ nodeId: id(source), outputId: 0 as OutputId }),
    target: id(target),
    targetHandle: stringifyTargetHandle({ nodeId: id(target), inputId: input as InputId }),
});

const check = (count: number, edges: Edge<EdgeData>[]) => {
    const nodes = Array.from({ length: count }, (_, i) => node(i));
    const types = TypeState.create(
        new Map(nodes.map((n) => [n.id, n])),
        edges,
        EMPTY_MAP,
        EMPTY_MAP,
        functionDefinitions
    );
    const lineage = ChainLineage.create(schemata, nodes, edges);
    return (source: number, target: number, input = 0) =>
        canConnect(
            { nodeId: id(source), outputId: 0 as OutputId },
            { nodeId: id(target), inputId: input as InputId },
            types,
            lineage
        );
};

const loop = 'Connection would create an infinite loop.';

test('a connection that would close a loop is refused', () => {
    // 0 -> 1 -> 2, and 3 alone
    const connect = check(4, [edge(0, 1, 0), edge(1, 2, 0)]);

    expect(connect(2, 0)).toEqual({ isValid: false, reason: loop });
    expect(connect(2, 1, 1)).toEqual({ isValid: false, reason: loop });
    expect(connect(1, 1)).toEqual({ isValid: false, reason: 'Cannot connect a node to itself.' });
    expect(connect(0, 2, 1).isValid).toBe(true);
    expect(connect(3, 0).isValid).toBe(true);
    // the same source against every handle, as while a connection is dragged
    expect([0, 1, 2, 3].map((t) => connect(2, t).isValid)).toEqual([false, false, false, true]);
});

test('other edges give other loops', () => {
    expect(check(3, [edge(0, 1, 0)])(2, 0).isValid).toBe(true);
    // a new list of edges, as after each change in the editor
    expect(check(3, [edge(0, 1, 0), edge(0, 2, 0)])(2, 1).isValid).toBe(true);
    expect(check(3, [edge(0, 1, 0), edge(1, 2, 0)])(2, 0).isValid).toBe(false);
});

// the search each check made before
const reachedBefore = (from: number, to: number, edges: Edge<EdgeData>[]) => {
    const visited = new Set<string>();
    const stack = [id(from)];
    let current;
    // eslint-disable-next-line no-cond-assign
    while ((current = stack.pop())) {
        if (current === id(to)) return true;
        if (!visited.has(current)) {
            visited.add(current);
            for (const e of edges) if (e.target === current) stack.push(e.source);
        }
    }
    return false;
};

test('every loop the search found before is refused, and only those', () => {
    let seed = 3;
    const random = () => {
        seed = (seed * 16807) % 2147483647;
        return seed / 2147483647;
    };
    for (let round = 0; round < 20; round += 1) {
        const count = 12;
        // edges from lower to higher nodes: no loop yet
        const edges: Edge<EdgeData>[] = [];
        for (let target = 1; target < count; target += 1) {
            for (const input of [0, 1]) {
                if (random() < 0.6) edges.push(edge(Math.floor(random() * target), target, input));
            }
        }
        const connect = check(count, edges);
        for (let source = 0; source < count; source += 1) {
            for (let target = 0; target < count; target += 1) {
                if (source !== target) {
                    const validity = connect(source, target);
                    const refused = !validity.isValid && validity.reason === loop;
                    expect(refused).toBe(reachedBefore(source, target, edges));
                }
            }
        }
    }
});
