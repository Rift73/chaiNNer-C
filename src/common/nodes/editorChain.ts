import { Edge, Node } from 'reactflow';
import { EdgeData, InputData, NodeData, NodeSchema, SchemaId } from '../common-types';
import { SchemaMap } from '../SchemaMap';
import { parseSourceHandle, parseTargetHandle } from '../util';

export interface EditorChain {
    readonly nodes: Node<NodeData>[];
    readonly edges: Edge<EdgeData>[];
    /** The schema ID of every dropped node, one entry per node. */
    readonly droppedNodes: readonly SchemaId[];
    /**
     * The edges dropped for not fitting their nodes (a chain file is untrusted): one without
     * handles, or whose handles name another node or an input or output the node lacks, a second
     * edge into one input, and an edge that closes a cycle. The editor's type evaluation would
     * throw on each of them, after every change, and a cycle overflows its stack.
     */
    readonly droppedEdges: number;
    /**
     * Whether the editor gets all of the chain. A load that is not complete leaves the chain
     * unsaved, so that saving over the file without the dropped nodes is the user's choice.
     */
    readonly complete: boolean;
}

const isPosition = (position: unknown): boolean => {
    const { x, y } = (position ?? {}) as { x?: unknown; y?: unknown };
    return Number.isFinite(x) && Number.isFinite(y);
};

const fitsNodes = (edge: Edge<EdgeData>, schemaOf: ReadonlyMap<string, NodeSchema>): boolean => {
    const source = schemaOf.get(edge.source);
    const target = schemaOf.get(edge.target);
    if (!source || !target) return false;
    if (typeof edge.sourceHandle !== 'string' || typeof edge.targetHandle !== 'string') {
        return false;
    }
    const output = parseSourceHandle(edge.sourceHandle);
    const input = parseTargetHandle(edge.targetHandle);
    return (
        output.nodeId === edge.source &&
        input.nodeId === edge.target &&
        source.outputs.some((o) => o.id === output.outputId) &&
        target.inputs.some((i) => i.id === input.inputId)
    );
};

/** The edges without the ones that close a cycle (a depth-first walk, without recursion). */
const withoutCycles = (edges: readonly Edge<EdgeData>[]): Edge<EdgeData>[] => {
    const outgoing = new Map<string, Edge<EdgeData>[]>();
    for (const edge of edges) {
        const list = outgoing.get(edge.source) ?? [];
        list.push(edge);
        outgoing.set(edge.source, list);
    }
    const state = new Map<string, 'open' | 'done'>();
    const closing = new Set<Edge<EdgeData>>();
    const walk = (start: string) => {
        state.set(start, 'open');
        const stack: { node: string; next: number }[] = [{ node: start, next: 0 }];
        while (stack.length > 0) {
            const top = stack[stack.length - 1];
            const out = outgoing.get(top.node) ?? [];
            if (top.next === out.length) {
                state.set(top.node, 'done');
                stack.pop();
            } else {
                const edge = out[top.next];
                top.next += 1;
                const seen = state.get(edge.target);
                if (seen === 'open') {
                    closing.add(edge);
                } else if (seen === undefined) {
                    state.set(edge.target, 'open');
                    stack.push({ node: edge.target, next: 0 });
                }
            }
        }
    };
    for (const start of outgoing.keys()) {
        if (!state.has(start)) walk(start);
    }
    return edges.filter((edge) => !closing.has(edge));
};

/**
 * Prepares the nodes and edges of an opened or pasted chain for the editor: nodes whose schema the
 * backend does not have are dropped with their edges, edges that do not fit their nodes are
 * dropped (droppedEdges), inputs a node lacks (e.g. added after it was saved) or holds no number or
 * text in get their defaults, a node without a valid position gets the origin, and no edge is
 * animated.
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
            position: isPosition(node.position) ? node.position : { x: 0, y: 0 },
            data: {
                ...node.data,
                inputData: {
                    ...schemata.getDefaultInput(node.data.schemaId),
                    // Input data holds numbers and texts; any other value (an object from a
                    // crafted file) would throw in the editor's type evaluation of the input.
                    ...Object.fromEntries(
                        // a chain file may hold no input data or null
                        Object.entries(
                            (node.data.inputData as Partial<InputData> | null | undefined) ?? {}
                        ).filter(([, v]) => typeof v === 'number' || typeof v === 'string')
                    ),
                },
            },
        }));
    const schemaOf = new Map(validNodes.map((n) => [n.id, schemata.get(n.data.schemaId)]));
    // An input takes one edge: the first one stays.
    const connected = new Set<string>();
    const fitting = edges.filter((edge) => {
        const handle = edge.targetHandle;
        if (!fitsNodes(edge, schemaOf) || typeof handle !== 'string' || connected.has(handle)) {
            return false;
        }
        connected.add(handle);
        return true;
    });
    const acyclic = withoutCycles(fitting);
    const validEdges = acyclic
        // Un-animate all edges, if was accidentally saved when animated
        .map((edge) => (edge.animated ? { ...edge, animated: false } : edge));
    const droppedEdges =
        edges.filter((edge) => schemaOf.has(edge.source) && schemaOf.has(edge.target)).length -
        validEdges.length;

    return {
        nodes: validNodes,
        edges: validEdges,
        droppedNodes,
        droppedEdges,
        complete: droppedNodes.length === 0 && droppedEdges === 0,
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
