import { Edge, Node } from 'reactflow';
import {
    EdgeData,
    Input,
    InputData,
    InputSchemaValue,
    InputValue,
    NodeData,
    NodeSchema,
    SchemaId,
} from '../common-types';
import { SchemaMap } from '../SchemaMap';
import { parseSourceHandle, parseTargetHandle, stringifyTargetHandle } from '../util';

export interface ReplacedInput {
    /** The name of the node's schema. */
    readonly node: string;
    /** The label of the input. */
    readonly input: string;
    readonly value: unknown;
}

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
    /** The values that their inputs cannot take, now the default. */
    readonly replacedInputs: readonly ReplacedInput[];
    /**
     * Whether the editor gets all of the chain. A load that is not complete leaves the chain
     * unsaved, so that saving over the file without what was dropped or replaced is the user's
     * choice.
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

/**
 * The value in a chain as the input takes it, or undefined if it cannot. A value for no input of
 * the node stays as it is.
 */
const toInputValue = (input: Input | undefined, value: unknown): InputSchemaValue | undefined => {
    // Input data holds numbers and texts; any other value (an object from a crafted file) would
    // throw in the editor's type evaluation of the input.
    if (typeof value !== 'number' && typeof value !== 'string') return undefined;
    // The backend rejects the rest. A dropdown corrects its value only when it shows, and the input
    // of a collapsed node or in a group that its condition hides does not.
    switch (input?.kind) {
        case 'dropdown':
            return input.options.some((o) => o.value === value) ? value : undefined;
        case 'number':
        case 'slider': {
            const number = typeof value === 'string' && value.trim() ? Number(value) : value;
            return typeof number === 'number' && Number.isFinite(number) ? number : undefined;
        }
        // a file input parses its value as a path when it shows, which throws on a number
        case 'file':
        case 'directory':
        case 'text':
        case 'color':
            return typeof value === 'string' ? value : undefined;
        default:
            return value;
    }
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
 * dropped (droppedEdges), inputs a node lacks (e.g. added after it was saved) or holds no value
 * they can take in get their defaults, a node without a valid position gets the origin, a node's
 * type and data id are those of its schema and its id, and no edge is animated.
 */
export const toEditorChain = (
    schemata: SchemaMap,
    nodes: readonly Node<NodeData>[],
    edges: readonly Edge<EdgeData>[]
): EditorChain => {
    const droppedNodes = nodes
        .filter((node) => !schemata.has(node.data.schemaId))
        .map((node) => node.data.schemaId);
    const replacedBy = new Map<string, ReplacedInput>();
    const validNodes = nodes
        .filter((node) => schemata.has(node.data.schemaId))
        .map((node) => {
            const schema = schemata.get(node.data.schemaId);
            const inputData: Record<string | number, InputValue> = {
                ...schemata.getDefaultInput(node.data.schemaId),
            };
            // a chain file may hold no input data or null
            const stored = (node.data.inputData as Partial<InputData> | null | undefined) ?? {};
            for (const [id, value] of Object.entries<unknown>(stored)) {
                const input = schema.inputs.find((i) => String(i.id) === id);
                const inputValue = toInputValue(input, value);
                if (inputValue !== undefined) {
                    inputData[id] = inputValue;
                } else if (input && value != null) {
                    replacedBy.set(stringifyTargetHandle({ nodeId: node.id, inputId: input.id }), {
                        node: schema.name,
                        input: input.label,
                        value,
                    });
                }
            }
            return {
                ...node,
                // The editor shows a node by its type, and the backend runs it by its type: a node
                // without one cannot run. A breakpoint is the one type its schema does not have.
                type: node.type === 'breakPoint' ? node.type : schema.kind,
                position: isPosition(node.position) ? node.position : { x: 0, y: 0 },
                data: {
                    ...node.data,
                    // the inputs of a node go by this one: without it, the node takes no input
                    id: node.id,
                    inputData,
                },
            };
        });
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
    // a connected input does not use its value
    for (const { targetHandle } of validEdges) {
        replacedBy.delete(targetHandle as string);
    }
    const replacedInputs = [...replacedBy.values()];

    return {
        nodes: validNodes,
        edges: validEdges,
        droppedNodes,
        droppedEdges,
        replacedInputs,
        complete: droppedNodes.length === 0 && droppedEdges === 0 && replacedInputs.length === 0,
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

/** The list of an alert: one line per label, with the number of nodes. */
const countedLines = (labels: readonly string[]): string => {
    const counts = new Map<string, number>();
    for (const label of labels) {
        counts.set(label, (counts.get(label) ?? 0) + 1);
    }
    return [...counts]
        .map(([label, count]) => `- ${label} (${count} ${count === 1 ? 'node' : 'nodes'})`)
        .join('\n');
};

/**
 * Lists the dropped nodes for an alert: one line per schema ID, with the number of nodes.
 */
export const formatDroppedNodes = (droppedNodes: readonly SchemaId[]): string =>
    countedLines(droppedNodes);

/**
 * Lists the replaced values for an alert: one line per input and value, with the number of nodes.
 */
export const formatReplacedInputs = (replacedInputs: readonly ReplacedInput[]): string =>
    countedLines(
        replacedInputs.map(
            ({ node, input, value }) => `${node}: ${input} held ${JSON.stringify(value)}`
        )
    );
