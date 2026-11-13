import { Edge, Node } from 'reactflow';
import { expect, test } from 'vitest';
import {
    CategoryId,
    EdgeData,
    GenericInput,
    InputId,
    NodeData,
    NodeGroupId,
    NodeSchema,
    Output,
    OutputId,
    SchemaId,
} from '../../src/common/common-types';
import { optimizeChain } from '../../src/common/nodes/optimize';
import { PassthroughMap } from '../../src/common/PassthroughMap';
import { SchemaMap } from '../../src/common/SchemaMap';
import { FunctionDefinition } from '../../src/common/types/function';
import { stringifySourceHandle, stringifyTargetHandle } from '../../src/common/util';

const input: GenericInput = {
    id: 0 as InputId,
    type: 'any',
    conversions: [],
    kind: 'generic',
    label: 'Value',
    optional: false,
    hasHandle: true,
    hint: false,
    suggest: false,
};
const output = (passthroughOf?: InputId): Output => ({
    id: 0 as OutputId,
    type: 'any',
    label: 'Value',
    kind: 'generic',
    hasHandle: true,
    suggest: false,
    passthroughOf,
});
const schema = (
    schemaId: string,
    inputs: GenericInput[],
    outputs: Output[],
    hasSideEffects: boolean
): NodeSchema => ({
    name: schemaId,
    category: 'test' as CategoryId,
    nodeGroup: 'test' as NodeGroupId,
    description: '',
    seeAlso: [],
    icon: '',
    kind: 'regularNode',
    inputs,
    outputs,
    groupLayout: inputs.map((i) => i.id),
    iteratorInputs: [],
    iteratorOutputs: [],
    suggestions: [],
    schemaId: schemaId as SchemaId,
    hasSideEffects,
    deprecated: false,
    features: [],
});
const schemas = [
    schema('test:source', [], [output()], false),
    schema('test:pass', [input], [output(0 as InputId)], false),
    schema('test:sink', [input], [], true),
];
const schemata = new SchemaMap(schemas);
const passthrough = PassthroughMap.create(
    new Map(schemas.map((s) => [s.schemaId, { schema: s } as FunctionDefinition]))
);

const node = (id: string, schemaId: string, isPassthrough = false): Node<NodeData> => ({
    id,
    type: 'regularNode',
    position: { x: 0, y: 0 },
    data: { id, schemaId: schemaId as SchemaId, inputData: {}, isPassthrough },
});
const edge = (source: string, target: string): Edge<EdgeData> => ({
    id: `${source}-${target}`,
    source,
    target,
    sourceHandle: stringifySourceHandle({ nodeId: source, outputId: 0 as OutputId }),
    targetHandle: stringifyTargetHandle({ nodeId: target, inputId: 0 as InputId }),
});

// a handle is the node's id, which is a UUID, a dash and the input or output id
const uuid = (index: number) => `${String(index).padStart(8, '0')}-0000-0000-0000-000000000000`;

test.each([1, 3, 40])('a chain of %i passthrough nodes is skipped as a whole', (count) => {
    const source = uuid(0);
    const sink = uuid(1);
    const passIds = Array.from({ length: count }, (_, i) => uuid(i + 2));
    const ids = [source, ...passIds, sink];
    const nodes = [
        node(source, 'test:source'),
        ...passIds.map((id) => node(id, 'test:pass', true)),
        node(sink, 'test:sink'),
    ];
    const edges = ids.slice(1).map((id, i) => edge(ids[i], id));

    const optimized = optimizeChain(nodes, edges, schemata, passthrough);

    expect(optimized.nodes.map((n) => n.id)).toEqual([source, sink]);
    const expected = edge(source, sink);
    expect(optimized.edges.map((e) => [e.source, e.sourceHandle, e.targetHandle])).toEqual([
        [source, expected.sourceHandle, expected.targetHandle],
    ]);
});
