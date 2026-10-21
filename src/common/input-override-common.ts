import { Edge, Node } from 'reactflow';
import { EdgeData, InputData, InputId, NodeData } from './common-types';
import { parseTargetHandle } from './util';

export type InputOverrideId = string & { readonly __input_override_id?: never };

export const createInputOverrideId = (nodeId: string, inputId: InputId): InputOverrideId => {
    if (nodeId.length !== 36)
        throw new Error('Expected node id to be a 36 character hexadecimal UUID.');
    return `#${nodeId}:${inputId}`;
};

/**
 * The graph with the given input values (by node id) for one run: overridden nodes are copies,
 * and connections into overridden inputs are removed. The editor's graph is not changed.
 */
export const withInputOverrides = (
    nodes: readonly Node<NodeData>[],
    edges: readonly Edge<EdgeData>[],
    overrides: ReadonlyMap<string, InputData>
): { nodes: Node<NodeData>[]; edges: Edge<EdgeData>[] } => {
    const missing = [...overrides.keys()].filter((id) => !nodes.some((n) => n.id === id));
    if (missing.length > 0) {
        throw new Error(`Cannot override the inputs of missing node(s) ${missing.join(', ')}.`);
    }

    return {
        nodes: nodes.map((node) => {
            const values = overrides.get(node.id);
            if (values === undefined) return node;
            return {
                ...node,
                data: { ...node.data, inputData: { ...node.data.inputData, ...values } },
            };
        }),
        edges: edges.filter(({ targetHandle }) => {
            if (!targetHandle) return true;
            const { nodeId, inputId } = parseTargetHandle(targetHandle);
            const values = overrides.get(nodeId);
            return values === undefined || !Object.prototype.hasOwnProperty.call(values, inputId);
        }),
    };
};
