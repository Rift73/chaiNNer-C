import { Edge } from 'reactflow';
import { expect, test } from 'vitest';
import { EdgeData, InputId, OutputId } from '../../src/common/common-types';
import { getConnectedInputs } from '../../src/common/nodes/connectedInputs';
import { stringifySourceHandle, stringifyTargetHandle } from '../../src/common/util';

// node IDs are UUIDs: handles keep the node ID as their first 36 characters
const id = (i: number) => `00000000-0000-4000-8000-${i.toString(16).padStart(12, '0')}`;
const edge = (source: number, target: number, input: number): Edge<EdgeData> => ({
    id: `${source}-${target}-${input}`,
    source: id(source),
    sourceHandle: stringifySourceHandle({ nodeId: id(source), outputId: 0 as OutputId }),
    target: id(target),
    targetHandle: stringifyTargetHandle({ nodeId: id(target), inputId: input as InputId }),
});

// what each node's validity check read before: one pass over the edges per node
const scanned = (nodeId: string, edges: readonly Edge<EdgeData>[]) =>
    edges.filter((e) => e.target === nodeId && e.targetHandle).map((e) => e.targetHandle);

test('every node gets the inputs its edges connect, in edge order', () => {
    const edges = [
        edge(1, 2, 3),
        edge(1, 2, 0),
        edge(2, 3, 1),
        { ...edge(1, 3, 2), targetHandle: null },
    ];

    expect([...getConnectedInputs(id(2), edges)]).toEqual([3, 0]);
    expect([...getConnectedInputs(id(3), edges)]).toEqual([1]);
    expect(getConnectedInputs(id(1), edges).size).toBe(0);
});

test('other edges give the inputs they connect', () => {
    // as after each change in the editor: a new list of edges
    let edges = [edge(1, 2, 0), edge(3, 2, 1)];
    expect([...getConnectedInputs(id(2), edges)]).toEqual([0, 1]);

    edges = edges.slice(1);
    expect([...getConnectedInputs(id(2), edges)]).toEqual([1]);

    edges = [...edges, edge(4, 2, 5), edge(4, 2, 5)];
    expect([...getConnectedInputs(id(2), edges)]).toEqual([1, 5]);
});

test('the inputs are those that each node read from the edges before', () => {
    let seed = 1;
    const random = () => {
        seed = (seed * 16807) % 2147483647;
        return seed / 2147483647;
    };
    for (let round = 0; round < 50; round += 1) {
        const edges = Array.from({ length: Math.floor(random() * 60) }, () =>
            edge(Math.floor(random() * 20), Math.floor(random() * 20), Math.floor(random() * 6))
        );
        for (let node = 0; node < 20; node += 1) {
            const expected = [...new Set(scanned(id(node), edges))].map((handle) =>
                Number(handle!.substring(37))
            );
            expect([...getConnectedInputs(id(node), edges)]).toEqual(expected);
        }
    }
});
