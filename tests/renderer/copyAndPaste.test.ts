import { Edge, Node } from 'reactflow';
import { beforeEach, expect, test, vi } from 'vitest';
import { EdgeData, NodeData, SchemaId } from '../../src/common/common-types';
import { log } from '../../src/common/log';
import { currentMigration } from '../../src/common/migrations';
import { SchemaMap } from '../../src/common/SchemaMap';
import {
    copyToClipboard,
    cutAndCopyToClipboard,
    pasteFromClipboard,
} from '../../src/renderer/helpers/copyAndPaste';

const stubs = vi.hoisted(() => ({ invoke: vi.fn() }));
vi.mock('../../src/renderer/safeIpc', () => ({ ipcRenderer: { invoke: stubs.invoke } }));

const SCHEMA = 'test:copy:node' as SchemaId;
const schemata = new SchemaMap([
    {
        schemaId: SCHEMA,
        name: 'Node',
        inputs: [{ id: 0, kind: 'text', def: 'default', label: 'Text' }],
        outputs: [{ id: 0 }],
        kind: 'regularNode',
    } as unknown as SchemaMap['schemata'][number],
]);
const A = '00000000-0000-0000-0000-00000000000a';
const B = '00000000-0000-0000-0000-00000000000b';
const node = (id: string, selected: boolean): Node<NodeData> => ({
    id,
    type: 'regularNode',
    position: { x: 10, y: 20 },
    selected,
    data: { id, schemaId: SCHEMA, inputData: {} },
});
const edge: Edge<EdgeData> = {
    id: 'edge',
    source: A,
    sourceHandle: `${A}-0`,
    target: B,
    targetHandle: `${B}-0`,
    type: 'main',
};

const paste = async (json: string, schemaMap = schemata) => {
    stubs.invoke.mockResolvedValueOnce({ kind: 'chain', json });
    let nodes: Node<NodeData>[] = [];
    let edges: Edge<EdgeData>[] = [];
    await pasteFromClipboard(
        (update) => {
            nodes = typeof update === 'function' ? update(nodes) : update;
        },
        (update) => {
            edges = typeof update === 'function' ? update(edges) : update;
        },
        vi.fn(),
        vi.fn(),
        { current: null },
        schemaMap,
        vi.fn()
    );
    return { nodes, edges };
};

beforeEach(() => {
    stubs.invoke.mockReset().mockResolvedValue(undefined);
});

test('copy with no node selected keeps what the clipboard holds', () => {
    copyToClipboard([node(A, false), node(B, false)], [edge]);
    expect(stubs.invoke).not.toHaveBeenCalled();

    cutAndCopyToClipboard([node(A, false)], [], vi.fn(), vi.fn());
    expect(stubs.invoke).not.toHaveBeenCalled();

    copyToClipboard([node(A, true), node(B, false)], [edge]);
    expect(stubs.invoke).toHaveBeenCalledOnce();
});

test('a copied chain pastes as new nodes with their connection', async () => {
    copyToClipboard([node(A, true), node(B, true)], [edge]);
    const [, json] = stubs.invoke.mock.calls[0] as [string, string];

    const { nodes, edges } = await paste(json);

    expect(nodes).toHaveLength(2);
    expect(nodes.every((n) => n.selected && n.id === n.data.id)).toBe(true);
    expect(nodes.map((n) => n.id)).not.toContain(A);
    expect(nodes[0].data.inputData).toEqual({ 0: 'default' });
    expect(edges).toEqual([
        expect.objectContaining({
            source: nodes[0].id,
            sourceHandle: `${nodes[0].id}-0`,
            target: nodes[1].id,
            targetHandle: `${nodes[1].id}-0`,
            selected: true,
        }),
    ]);
});

test('a chain says how old it is, and an older one pastes migrated', async () => {
    copyToClipboard([node(A, true)], []);
    const [, json] = stubs.invoke.mock.calls[0] as [string, string];
    expect(JSON.parse(json)).toMatchObject({ migration: currentMigration });

    // Load Video's limit of 25 frames, from before its Frames input
    const LOAD_VIDEO = 'chainner:image:load_video' as SchemaId;
    const withLoadVideo = new SchemaMap([
        {
            schemaId: LOAD_VIDEO,
            inputs: [],
            outputs: [],
            kind: 'generator',
        } as unknown as SchemaMap['schemata'][number],
    ]);
    const older = (migration: number | undefined) =>
        JSON.stringify({
            nodes: [
                {
                    ...node(A, true),
                    data: { id: A, schemaId: LOAD_VIDEO, inputData: { 0: 'a.mp4', 1: 1, 2: 25 } },
                },
            ],
            edges: [],
            migration,
        });

    const migrated = await paste(older(currentMigration - 2), withLoadVideo);
    expect(migrated.nodes[0].data.inputData).toEqual({ 0: 'a.mp4', 10: 'range', 11: 0, 12: 24 });

    // a version that does not say how old its chains are: nothing can be run on it
    const unknown = await paste(older(undefined), withLoadVideo);
    expect(unknown.nodes[0].data.inputData).toEqual({ 0: 'a.mp4', 1: 1, 2: 25 });
});

test('a pasted value that was replaced is logged', async () => {
    const warn = vi.spyOn(log, 'warn');
    const held = {
        nodes: [{ ...node(A, true), data: { id: A, schemaId: SCHEMA, inputData: { 0: 5 } } }],
        edges: [],
    };

    const { nodes } = await paste(JSON.stringify(held));

    expect(nodes[0].data.inputData).toEqual({ 0: 'default' });
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('held 5'));
    warn.mockRestore();
});

test('a chain copied by an older version pastes without what it no longer fits', async () => {
    // a node without a data id or input data, a node this version lacks, and an edge to it
    const old = {
        nodes: [
            { ...node(A, true), data: { schemaId: SCHEMA, inputData: null } },
            { ...node(B, true), data: { id: B, schemaId: 'chainner:image:gone', inputData: {} } },
        ],
        edges: [edge],
    };

    const { nodes, edges } = await paste(JSON.stringify(old));

    expect(nodes).toHaveLength(1);
    expect(nodes[0].data.id).toBe(nodes[0].id);
    expect(edges).toEqual([]);
});
