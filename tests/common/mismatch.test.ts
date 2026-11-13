import { SourceDocument, evaluate, parseExpression } from '@chainner/navi';
import { expect, test } from 'vitest';
import { getChainnerScope } from '../../src/common/types/chainner-scope';
import { clipImageError } from '../../src/common/types/mismatch';

const type = (expression: string) =>
    evaluate(parseExpression(new SourceDocument(expression, 'test')), getChainnerScope());

const image = type('Image { width: 640, height: 480, channels: 3 }');
const clip = type('VideoClip');

test('an image into an input that takes only a clip names the nodes that make a clip', () => {
    for (const definition of ['VideoClip', 'VSFilter | VideoClip']) {
        expect(clipImageError(image, type(definition))).toBe(
            'This input takes a clip, not single images: the node works on the clip as a whole or looks at other frames. Open the video with Load Clip, or turn images into a clip with Frames to Clip.'
        );
    }
});

test('a clip into an input that takes images names Clip Frames', () => {
    for (const definition of ['Image', 'Image { channels: 3 }', 'Image | Color', 'Image | Audio']) {
        expect(clipImageError(clip, type(definition))).toBe(
            'This input takes an image. Add Clip Frames to turn the clip into images.'
        );
    }
});

test('other mismatches keep their own texts', () => {
    // inputs that also take something else, other kinds, and an input that takes nothing
    expect(clipImageError(image, type('VideoClip | Audio'))).toBeUndefined();
    expect(clipImageError(clip, type('Audio'))).toBeUndefined();
    expect(clipImageError(clip, type('Color | Audio'))).toBeUndefined();
    expect(clipImageError(type('Audio'), type('VideoClip'))).toBeUndefined();
    expect(clipImageError(type('Video'), type('Image'))).toBeUndefined();
    expect(clipImageError(image, type('Image { channels: 1 }'))).toBeUndefined();
    expect(clipImageError(image, type('never'))).toBeUndefined();
});
