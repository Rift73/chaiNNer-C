import { NonNeverType, SourceDocument, evaluate, parseExpression } from '@chainner/navi';
import { expect, test } from 'vitest';
import { OutputId } from '../../src/common/common-types';
import { getChainnerScope } from '../../src/common/types/chainner-scope';
import { getClipOutput } from '../../src/common/types/util';

const type = (expression: string) =>
    evaluate(
        parseExpression(new SourceDocument(expression, 'test')),
        getChainnerScope()
    ) as NonNeverType;
const outputs = (...types: string[]) => ({
    outputs: new Map(types.map((t, i) => [i as OutputId, type(t)])),
});

test('the Preview entry shows the first output that gives a clip', () => {
    expect(getClipOutput(outputs('Image', 'VideoClip', 'VideoClip'))).toBe(1);
    expect(
        getClipOutput(
            outputs('VideoClip { width: 1920, height: 1080, fps: 24, fieldOrder: "progressive" }')
        )
    ).toBe(0);
});

test('a node without a clip output, or whose output may be a filter, has no Preview entry', () => {
    expect(getClipOutput(outputs('Image', 'number'))).toBeUndefined();
    expect(getClipOutput(outputs('VSFilter'))).toBeUndefined();
    expect(getClipOutput(outputs('VideoClip | VSFilter'))).toBeUndefined();
    // types not known yet
    expect(getClipOutput(undefined)).toBeUndefined();
});
