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
        .map((n) => ({
            id: n.id,
            type: n.type,
            position: { x: n.position.x, y: n.position.y },
            data: {
                schemaId: n.data.schemaId,
                inputData: n.data.inputData,
                inputHeight: n.data.inputHeight,
                outputHeight: n.data.outputHeight,
                nodeWidth: n.data.nodeWidth,
                isDisabled: n.data.isDisabled,
                isLocked: n.data.isLocked,
                isCollapsed: n.data.isCollapsed,
                isPassthrough: n.data.isPassthrough,
                nodeName: n.data.nodeName,
            },
        }))
        .sort((a, b) => binaryCompare(a.id, b.id));
    const edgeEntries = edges
        .map((e) =>
            JSON.stringify([e.source, e.sourceHandle ?? null, e.target, e.targetHandle ?? null])
        )
        .sort(binaryCompare);
    return JSON.stringify(withSortedKeys({ nodes: nodeEntries, edges: edgeEntries }));
};
