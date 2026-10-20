import { expect, test } from 'vitest';
import { OutputId } from '../../src/common/common-types';
import {
    clampFrame,
    formatTimecode,
    numericOutputValue,
    stepFrame,
} from '../../src/renderer/helpers/videoSeek';

test('numericOutputValue reads a number broadcast type', () => {
    const types = { 4: 23.976, 12: 1440, 2: 'string', 3: null };
    expect(numericOutputValue(types, 12 as OutputId)).toBe(1440);
    expect(numericOutputValue(types, 4 as OutputId)).toBe(23.976);
    // not a number, no type, no output, no broadcast yet
    expect(numericOutputValue(types, 2 as OutputId)).toBeUndefined();
    expect(numericOutputValue(types, 3 as OutputId)).toBeUndefined();
    expect(numericOutputValue(types, 5 as OutputId)).toBeUndefined();
    expect(numericOutputValue(undefined, 12 as OutputId)).toBeUndefined();
});

test('clampFrame keeps a frame within the video', () => {
    expect(clampFrame(-3, 100)).toBe(0);
    expect(clampFrame(40, 100)).toBe(40);
    expect(clampFrame(40.6, 100)).toBe(41);
    expect(clampFrame(100, 100)).toBe(99);
    expect(clampFrame(5, 1)).toBe(0);
});

test('stepFrame steps by a frame or a second (the frame rate rounded)', () => {
    const count = 1000;
    expect(stepFrame('first', 500, count, 23.976)).toBe(0);
    expect(stepFrame('second-back', 500, count, 23.976)).toBe(476);
    expect(stepFrame('frame-back', 500, count, 23.976)).toBe(499);
    expect(stepFrame('frame-forward', 500, count, 23.976)).toBe(501);
    expect(stepFrame('second-forward', 500, count, 29.97)).toBe(530);
    expect(stepFrame('last', 500, count, 23.976)).toBe(999);

    // the ends hold
    expect(stepFrame('second-back', 10, count, 60)).toBe(0);
    expect(stepFrame('frame-back', 0, count, 60)).toBe(0);
    expect(stepFrame('frame-forward', 999, count, 60)).toBe(999);
    expect(stepFrame('second-forward', 990, count, 60)).toBe(999);
    // a second is at least one frame
    expect(stepFrame('second-forward', 5, count, 0.2)).toBe(6);
});

test('formatTimecode is H:MM:SS.mmm of frame / fps', () => {
    expect(formatTimecode(0, 24)).toBe('0:00:00.000');
    expect(formatTimecode(1, 24)).toBe('0:00:00.042');
    expect(formatTimecode(24, 24)).toBe('0:00:01.000');
    // 1439 / 23.976 = 60.0183...
    expect(formatTimecode(1439, 23.976)).toBe('0:01:00.018');
    expect(formatTimecode(90 * 60 * 30 + 15, 30)).toBe('1:30:00.500');
});
