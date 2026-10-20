import { Node } from 'reactflow';
import { expect, test } from 'vitest';
import { InputId, NodeData, SchemaId } from '../../src/common/common-types';
import { withInputOverrides } from '../../src/common/input-override-common';

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

    const overridden = withInputOverrides(
        nodes,
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
        withInputOverrides([node('video', {})], new Map([['gone', { [10 as InputId]: 'range' }]]))
    ).toThrow('gone');
});
