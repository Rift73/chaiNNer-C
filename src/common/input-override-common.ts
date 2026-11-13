import { Edge, Node } from 'reactflow';
import { EdgeData, InputData, InputId, NodeData, OutputId, SchemaId } from './common-types';
import {
    deriveUniqueId,
    parseTargetHandle,
    stringifySourceHandle,
    stringifyTargetHandle,
} from './util';

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

/**
 * The part of the graph a node's outputs come from, for one run: the node, every node it depends
 * on through its connections, and the connections between them. Nodes after it do not run.
 */
export const withOnlyUpTo = (
    nodes: readonly Node<NodeData>[],
    edges: readonly Edge<EdgeData>[],
    nodeId: string
): { nodes: Node<NodeData>[]; edges: Edge<EdgeData>[] } => {
    const kept = new Set([nodeId]);
    const pending = [nodeId];
    for (let id = pending.pop(); id !== undefined; id = pending.pop()) {
        for (const { source, target } of edges) {
            if (target === id && !kept.has(source)) {
                kept.add(source);
                pending.push(source);
            }
        }
    }
    return {
        nodes: nodes.filter((n) => kept.has(n.id)),
        edges: edges.filter((e) => kept.has(e.source) && kept.has(e.target)),
    };
};

const PREVIEW_CLIPS = 'chainner:vapoursynth:preview_clips' as SchemaId;

/**
 * The graph with a Preview Clips node, which the editor's chain does not have, showing the clip of
 * `nodeId`'s output in the viewer. Its id follows from the node's, so that the viewer it opens for
 * the node reloads at the node's next preview, as a Preview Clips node's does.
 */
export const withClipViewer = (
    nodes: readonly Node<NodeData>[],
    edges: readonly Edge<EdgeData>[],
    nodeId: string,
    outputId: OutputId
): { nodes: Node<NodeData>[]; edges: Edge<EdgeData>[] } => {
    const viewerId = deriveUniqueId(`${nodeId} clip viewer`);
    const viewer: Node<NodeData> = {
        id: viewerId,
        type: 'regularNode',
        position: { x: 0, y: 0 },
        data: { id: viewerId, schemaId: PREVIEW_CLIPS, inputData: {} },
    };
    const edge: Edge<EdgeData> = {
        id: deriveUniqueId(`${viewerId} clip`),
        source: nodeId,
        sourceHandle: stringifySourceHandle({ nodeId, outputId }),
        target: viewerId,
        targetHandle: stringifyTargetHandle({ nodeId: viewerId, inputId: 0 as InputId }),
        type: 'main',
        data: {},
    };
    return { nodes: [...nodes, viewer], edges: [...edges, edge] };
};
