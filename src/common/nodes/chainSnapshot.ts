import { Edge, Node } from 'reactflow';
import { EdgeData, NodeData } from '../common-types';
import { binaryCompare } from '../util';

/** The value with the keys of every object in it sorted, so equal content stringifies equally. */
const withSortedKeys = (value: unknown): unknown => {
    if (Array.isArray(value)) {
        return value.map(withSortedKeys);
    }
    if (value !== null && typeof value === 'object') {
        return Object.fromEntries(
            Object.entries(value)
                .sort(([a], [b]) => binaryCompare(a, b))
                .map(([key, v]) => [key, withSortedKeys(v)])
        );
    }
    return value;
};

interface NodeText {
    readonly id: string;
    readonly type: string | undefined;
    readonly x: number;
    readonly y: number;
    readonly text: string;
}
/**
 * Each node's part of the snapshot, by its node data, with the rest of the node it was made of.
 * The editor takes a snapshot after every change (twice: for the unsaved mark and the history),
 * and a change replaces the data of the nodes it changes, never changing data in place: the other
 * nodes keep their parts.
 */
const nodeTexts = new WeakMap<NodeData, NodeText>();
const nodeText = (n: Node<NodeData>): string => {
    const { id, type, data } = n;
    const { x, y } = n.position;
    const kept = nodeTexts.get(data);
    if (kept && kept.id === id && kept.type === type && kept.x === x && kept.y === y) {
        return kept.text;
    }
    const text = JSON.stringify(
        withSortedKeys({
            id,
            type,
            position: { x, y },
            data: {
                schemaId: data.schemaId,
                inputData: data.inputData,
                inputHeight: data.inputHeight,
                outputHeight: data.outputHeight,
                nodeWidth: data.nodeWidth,
                isDisabled: data.isDisabled,
                isLocked: data.isLocked,
                isCollapsed: data.isCollapsed,
                isPassthrough: data.isPassthrough,
                nodeName: data.nodeName,
            },
        })
    );
    nodeTexts.set(data, { id, type, x, y, text });
    return text;
};

/**
 * A canonical string of the chain content that a save keeps: two chains have the same snapshot
 * exactly when saving them keeps the same chain.
 *
 * Nodes count with their id, type, position and node data; edges with their ends. What the
 * editor changes without the user changing the chain is left out: selection, dragging, measured
 * sizes, edge animation and ids, the coordinates edges keep in their data, and the order of nodes
 * and edges.
 */
export const chainSnapshot = (
    nodes: readonly Node<NodeData>[],
    edges: readonly Edge<EdgeData>[]
): string => {
    const nodeEntries = nodes
        .map((n) => [n.id, nodeText(n)] as const)
        .sort((a, b) => binaryCompare(a[0], b[0]));
    const edgeEntries = edges
        .map((e) =>
            JSON.stringify([e.source, e.sourceHandle ?? null, e.target, e.targetHandle ?? null])
        )
        .sort(binaryCompare);
    // JSON.stringify(withSortedKeys({ nodes, edges })) of the entries, whose keys sort as here
    return `{"edges":${JSON.stringify(edgeEntries)},"nodes":[${nodeEntries
        .map(([, text]) => text)
        .join(',')}]}`;
};
