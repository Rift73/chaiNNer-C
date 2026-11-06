import { Edge, Node } from 'reactflow';
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import { EdgeData, NodeData } from '../../src/common/common-types';

const node = (id: string, x: number): Node<NodeData> =>
    ({ id, position: { x, y: 0 }, data: {}, width: 100, height: 50 } as Node<NodeData>);
const nodes = [node('a', 0), node('b', 0)];
const edges = [{ id: 'e', source: 'a', target: 'b' } as Edge<EdgeData>];

describe('formatting a chain', () => {
    beforeEach(() => {
        vi.resetModules();
    });

    afterEach(() => {
        vi.doUnmock('elkjs');
    });

    test('ELK loads on first use and lays out the nodes', async () => {
        const { getLayoutedPositionMap } = await import('../../src/renderer/helpers/graphUtils');

        const positions = await getLayoutedPositionMap(nodes, edges);

        expect(positions.size).toBe(2);
        expect(positions.get('a')!.x).toBeLessThan(positions.get('b')!.x);
    });

    test('a failed load reaches the caller', async () => {
        vi.doMock('elkjs', () => {
            throw new Error('the chunk is missing');
        });
        const { getLayoutedPositionMap } = await import('../../src/renderer/helpers/graphUtils');

        await expect(getLayoutedPositionMap(nodes, edges)).rejects.toThrow();
    });
});
