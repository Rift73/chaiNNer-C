import { Edge, Node } from 'reactflow';
import { expect, test } from 'vitest';
import { EdgeData, InputId, NodeData, SchemaId } from '../../src/common/common-types';
import { chainSnapshot } from '../../src/common/nodes/chainSnapshot';

const LOAD_IMAGE = 'chainner:image:load' as SchemaId;
const SAVE_IMAGE = 'chainner:image:save' as SchemaId;

const node = (id: string, schemaId: SchemaId, data: Partial<NodeData> = {}): Node<NodeData> => ({
    id,
    type: 'regularNode',
    position: { x: 10, y: 20 },
    data: { id, schemaId, inputData: { [0 as InputId]: 'a.png' }, ...data },
});
const edge = (id: string, source: string, target: string): Edge<EdgeData> => ({
    id,
    source,
    sourceHandle: `${source}-0`,
    target,
    targetHandle: `${target}-0`,
    type: 'main',
    data: {},
});

const nodes = [node('a', LOAD_IMAGE), node('b', SAVE_IMAGE)];
const edges = [edge('e1', 'a', 'b')];
const snapshot = chainSnapshot(nodes, edges);

test('what the editor changes on its own leaves the snapshot as it is', () => {
    // a click on a node selects it, React Flow measures it, an edge animates and keeps the
    // coordinates of its ends, and setters that change nothing return new objects
    const touchedNodes = nodes.map(
        (n): Node<NodeData> => ({
            ...n,
            selected: true,
            dragging: false,
            width: 240,
            height: 180,
            positionAbsolute: { x: 10, y: 20 },
            position: { ...n.position },
            data: { ...n.data, inputData: { ...n.data.inputData } },
        })
    );
    const touchedEdges = edges.map(
        (e): Edge<EdgeData> => ({
            ...e,
            id: 'e1-copy',
            animated: true,
            selected: true,
            data: { sourceX: 1, sourceY: 2, targetX: 3, targetY: 4 },
        })
    );
    expect(chainSnapshot(touchedNodes, touchedEdges)).toBe(snapshot);
});

test('the order of nodes, edges and object keys does not count', () => {
    const reordered = [
        node('b', SAVE_IMAGE),
        {
            data: { inputData: { [0 as InputId]: 'a.png' }, schemaId: LOAD_IMAGE, id: 'a' },
            position: { y: 20, x: 10 },
            type: 'regularNode',
            id: 'a',
        },
    ];
    expect(chainSnapshot(reordered, [...edges].reverse())).toBe(snapshot);
});

test('an unset property and an undefined one are the same', () => {
    expect(chainSnapshot([node('a', LOAD_IMAGE, { nodeName: undefined })], [])).toBe(
        chainSnapshot([node('a', LOAD_IMAGE)], [])
    );
});

test('every change a save keeps changes the snapshot', () => {
    const changes: [string, Node<NodeData>[], Edge<EdgeData>[]][] = [
        ['moved node', [{ ...nodes[0], position: { x: 11, y: 20 } }, nodes[1]], edges],
        [
            'input value',
            [node('a', LOAD_IMAGE, { inputData: { [0 as InputId]: 'b.png' } }), nodes[1]],
            edges,
        ],
        [
            'input height',
            [node('a', LOAD_IMAGE, { inputHeight: { [0 as InputId]: 80 } }), nodes[1]],
            edges,
        ],
        ['node width', [node('a', LOAD_IMAGE, { nodeWidth: 300 }), nodes[1]], edges],
        ['disabled', [node('a', LOAD_IMAGE, { isDisabled: true }), nodes[1]], edges],
        ['locked', [node('a', LOAD_IMAGE, { isLocked: true }), nodes[1]], edges],
        ['collapsed', [node('a', LOAD_IMAGE, { isCollapsed: true }), nodes[1]], edges],
        ['passthrough', [node('a', LOAD_IMAGE, { isPassthrough: true }), nodes[1]], edges],
        ['renamed', [node('a', LOAD_IMAGE, { nodeName: 'Input' }), nodes[1]], edges],
        ['removed node', [nodes[0]], []],
        ['added node', [...nodes, node('c', LOAD_IMAGE)], edges],
        ['removed edge', nodes, []],
        ['other handle', nodes, [{ ...edges[0], targetHandle: 'b-1' }]],
    ];
    for (const [change, changedNodes, changedEdges] of changes) {
        expect(chainSnapshot(changedNodes, changedEdges), change).not.toBe(snapshot);
    }
});
