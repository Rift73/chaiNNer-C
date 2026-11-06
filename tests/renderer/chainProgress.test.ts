import { expect, test } from 'vitest';
import {
    ChainProgress,
    nodeProgressLabel,
    withNodeFinished,
    withNodeProgress,
} from '../../src/renderer/helpers/chainProgress';

const chain: ChainProgress = new Map([['upscale', { progress: 0, weight: 1 }]]);
const own = { progress: 0.37, index: 0, total: 0, eta: 0 };
const items = { progress: 0.5, index: 3, total: 6, eta: 2 };

test('a node the backend left out of the chain changes nothing', () => {
    expect(withNodeProgress(chain, 'gone', 0.5)).toBe(chain);
});

test("a node's own progress ends at 100% with its finish", () => {
    const after = withNodeFinished({ upscale: own, load: items }, 'upscale');
    expect(after.upscale).toEqual({ ...own, progress: 1 });
    expect(after.load).toBe(items);
});

test("a generator's progress and a node without progress keep theirs at a finish", () => {
    const before = { load: items };
    expect(withNodeFinished(before, 'load')).toBe(before);
    expect(withNodeFinished(before, 'upscale')).toBe(before);
});

test('the label of a node without items has no item count', () => {
    expect(nodeProgressLabel(own)).toBe('37%');
    expect(nodeProgressLabel(items)).toBe('3/6 (50%)');
});
