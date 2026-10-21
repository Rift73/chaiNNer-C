import { Edge, Node } from 'reactflow';
import { expect, test } from 'vitest';
import {
    CategoryId,
    EdgeData,
    InputId,
    NodeData,
    NodeGroupId,
    NodeSchema,
    OutputId,
    SchemaId,
} from '../../src/common/common-types';
import { withInputOverrides } from '../../src/common/input-override-common';
import { toBackendJson } from '../../src/common/nodes/toBackendJson';
import { SchemaMap } from '../../src/common/SchemaMap';

const node = (id: string, inputData: Record<number, string | number>): Node<NodeData> => ({
    id,
    type: 'generator',
    position: { x: 0, y: 0 },
    data: { id, schemaId: 'chainner:image:load_video' as SchemaId, inputData },
});

test('withInputOverrides sets the values on copies of the overridden nodes', () => {
    const video = node('video', { 0: 'a.mp4', 10: 'all', 11: 0, 12: 9, 13: 120 });
    const other = node('other', { 0: 'b.mp4' });
    const nodes = [video, other];

    const { nodes: overridden } = withInputOverrides(
        nodes,
        [],
        new Map([
            ['video', { [10 as InputId]: 'range', [11 as InputId]: 120, [12 as InputId]: 120 }],
        ])
    );

    expect(overridden[0].data.inputData).toEqual({
        0: 'a.mp4',
        10: 'range',
        11: 120,
        12: 120,
        13: 120,
    });
    // the editor's nodes keep their values, and a node without overrides is the same node
    expect(video.data.inputData).toEqual({ 0: 'a.mp4', 10: 'all', 11: 0, 12: 9, 13: 120 });
    expect(overridden[0]).not.toBe(video);
    expect(overridden[1]).toBe(other);
    expect(nodes).toEqual([video, other]);
});

test('withInputOverrides refuses a node the chain does not have', () => {
    expect(() =>
        withInputOverrides(
            [node('video', {})],
            [],
            new Map([['gone', { [10 as InputId]: 'range' }]])
        )
    ).toThrow('gone');
});

test('connected In and Out serialize as the selected frame when overridden', () => {
    const video = node('00000000-0000-0000-0000-000000000001', {
        10: 'all',
        11: 0,
        12: 999,
        13: 120,
    });
    const number: Node<NodeData> = {
        ...node('00000000-0000-0000-0000-000000000002', {}),
        type: 'regularNode',
        data: {
            id: '00000000-0000-0000-0000-000000000002',
            schemaId: 'chainner:utility:number' as SchemaId,
            inputData: {},
        },
    };
    const numberSchema: NodeSchema = {
        schemaId: number.data.schemaId,
        name: 'Number',
        category: 'utility' as CategoryId,
        nodeGroup: 'value' as NodeGroupId,
        description: '',
        seeAlso: [],
        icon: '',
        kind: 'regularNode',
        inputs: [],
        outputs: [
            {
                id: 0 as OutputId,
                label: 'Number',
                type: 'number',
                kind: 'generic',
                hasHandle: true,
                suggest: false,
            },
        ],
        groupLayout: [],
        iteratorInputs: [],
        iteratorOutputs: [],
        suggestions: [],
        hasSideEffects: false,
        deprecated: false,
        features: [],
    };
    const videoSchema: NodeSchema = {
        ...numberSchema,
        schemaId: video.data.schemaId,
        name: 'Load Video',
        kind: 'generator',
        inputs: [10, 11, 12, 13].map((id) => ({
            id: id as InputId,
            label: String(id),
            type: id === 10 ? 'string' : 'number',
            kind: 'generic',
            conversions: [],
            optional: false,
            hasHandle: true,
            hint: false,
            suggest: false,
        })),
        outputs: [],
    };
    const schemata = new SchemaMap([numberSchema, videoSchema]);
    const nodes = [video, number];
    const edges: Edge<EdgeData>[] = [11, 12, 13].map((id) => ({
        id: `edge-${id}`,
        source: number.id,
        sourceHandle: `${number.id}-0`,
        target: video.id,
        targetHandle: `${video.id}-${id}`,
    }));
    const original = toBackendJson(nodes, edges, schemata)[0];
    expect(original.inputs.slice(1)).toEqual([
        { type: 'edge', id: number.id, index: 0 },
        { type: 'edge', id: number.id, index: 0 },
        { type: 'edge', id: number.id, index: 0 },
    ]);

    const overridden = withInputOverrides(
        nodes,
        edges,
        new Map([
            [video.id, { [10 as InputId]: 'range', [11 as InputId]: 120, [12 as InputId]: 120 }],
        ])
    );
    expect(toBackendJson(overridden.nodes, overridden.edges, schemata)[0].inputs).toEqual([
        { type: 'value', value: 'range' },
        { type: 'value', value: 120 },
        { type: 'value', value: 120 },
        { type: 'edge', id: number.id, index: 0 },
    ]);
    expect(overridden.edges).toEqual([edges[2]]);
    expect(toBackendJson(nodes, edges, schemata)[0]).toEqual(original);
});
