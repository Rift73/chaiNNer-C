import { Edge, Node } from 'reactflow';
import { EdgeData, NodeData, SchemaId } from '../common-types';
import { SchemaMap } from '../SchemaMap';

export interface EditorChain {
    readonly nodes: Node<NodeData>[];
    readonly edges: Edge<EdgeData>[];
    /** The schema ID of every dropped node, one entry per node. */
    readonly droppedNodes: readonly SchemaId[];
    /**
     * Whether the editor gets all of the chain. A load that is not complete leaves the chain
     * unsaved, so that saving over the file without the dropped nodes is the user's choice.
     */
    readonly complete: boolean;
}

/**
 * Prepares the nodes and edges of an opened or pasted chain for the editor: nodes whose schema the
 * backend does not have are dropped with their edges, inputs a node lacks (e.g. added after it was
 * saved) get their defaults, and no edge is animated.
 */
export const toEditorChain = (
    schemata: SchemaMap,
    nodes: readonly Node<NodeData>[],
    edges: readonly Edge<EdgeData>[]
): EditorChain => {
    const droppedNodes = nodes
        .filter((node) => !schemata.has(node.data.schemaId))
        .map((node) => node.data.schemaId);
    const validNodes = nodes
        .filter((node) => schemata.has(node.data.schemaId))
        .map((node) => ({
            ...node,
            data: {
                ...node.data,
                inputData: {
                    ...schemata.getDefaultInput(node.data.schemaId),
                    ...Object.fromEntries(
                        Object.entries(node.data.inputData).filter(([, v]) => v != null)
                    ),
                },
            },
        }));
    const validNodeIds = new Set(validNodes.map((n) => n.id));
    const validEdges = edges
        // Filter out any edges that do not have a source or target node associated with it
        .filter((edge) => validNodeIds.has(edge.target) && validNodeIds.has(edge.source))
        // Un-animate all edges, if was accidentally saved when animated
        .map((edge) => (edge.animated ? { ...edge, animated: false } : edge));

    return {
        nodes: validNodes,
        edges: validEdges,
        droppedNodes,
        complete: droppedNodes.length === 0,
    };
};

/**
 * The schema IDs of a chain's nodes that the node list lacks, each once: opened with this list,
 * the chain would lose those nodes.
 */
export const missingSchemata = (
    schemata: SchemaMap,
    nodes: readonly Node<NodeData>[]
): SchemaId[] => [...new Set(nodes.map((n) => n.data.schemaId))].filter((id) => !schemata.has(id));

/**
 * Lists the dropped nodes for an alert: one line per schema ID, with the number of nodes.
 */
export const formatDroppedNodes = (droppedNodes: readonly SchemaId[]): string => {
    const counts = new Map<SchemaId, number>();
    for (const schemaId of droppedNodes) {
        counts.set(schemaId, (counts.get(schemaId) ?? 0) + 1);
    }
    return [...counts]
        .map(([schemaId, count]) => `- ${schemaId} (${count} ${count === 1 ? 'node' : 'nodes'})`)
        .join('\n');
};
