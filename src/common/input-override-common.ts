import { Node } from 'reactflow';
import { InputData, InputId, NodeData } from './common-types';

export type InputOverrideId = string & { readonly __input_override_id?: never };

export const createInputOverrideId = (nodeId: string, inputId: InputId): InputOverrideId => {
    if (nodeId.length !== 36)
        throw new Error('Expected node id to be a 36 character hexadecimal UUID.');
    return `#${nodeId}:${inputId}`;
};

/**
 * The nodes with the given input values (by node id) for one run: an overridden node is a copy
 * with the values, and the given nodes are not changed.
 */
export const withInputOverrides = (
    nodes: readonly Node<NodeData>[],
    overrides: ReadonlyMap<string, InputData>
): Node<NodeData>[] => {
    const missing = [...overrides.keys()].filter((id) => !nodes.some((n) => n.id === id));
    if (missing.length > 0) {
        throw new Error(`Cannot override the inputs of missing node(s) ${missing.join(', ')}.`);
    }

    return nodes.map((node) => {
        const values = overrides.get(node.id);
        if (values === undefined) return node;
        return {
            ...node,
            data: { ...node.data, inputData: { ...node.data.inputData, ...values } },
        };
    });
};
