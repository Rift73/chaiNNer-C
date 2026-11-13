import { expect, test } from 'vitest';
import { Input, NodeSchema, Output, OutputId } from '../../src/common/common-types';
import {
    PAN_ZOOM_FIT,
    SCRUB_PIXELS_PER_FRAME,
    clampFrame,
    formatTimecode,
    getSeekUnderPreview,
    getSeekVideo,
    numericOutputValue,
    scrubFrame,
    seekStepForKey,
    stepFrame,
    zoomAt,
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

test('without a frame rate there is no timecode, and a second is one frame', () => {
    expect(formatTimecode(100, 0)).toBeUndefined();
    expect(stepFrame('second-forward', 100, 1000, 0)).toBe(101);
    expect(stepFrame('second-back', 100, 1000, 0)).toBe(99);
});

test('getSeekVideo needs a frame count and a frame rate, which may be 0', () => {
    expect(getSeekVideo(43000, 30000 / 1001)).toEqual({ frameCount: 43000, fps: 30000 / 1001 });
    expect(getSeekVideo(120.4, 0)).toEqual({ frameCount: 120, fps: 0 });
    expect(getSeekVideo(undefined, 24)).toBeUndefined();
    expect(getSeekVideo(100, undefined)).toBeUndefined();
    expect(getSeekVideo(0, 24)).toBeUndefined();
    expect(getSeekVideo(100, -1)).toBeUndefined();
});

test('stepFrame steps ten frames', () => {
    expect(stepFrame('ten-back', 500, 1000, 24)).toBe(490);
    expect(stepFrame('ten-forward', 500, 1000, 24)).toBe(510);
    expect(stepFrame('ten-back', 4, 1000, 24)).toBe(0);
    expect(stepFrame('ten-forward', 995, 1000, 24)).toBe(999);
});

test('seekStepForKey: Shift steps a second in the node and ten frames maximized', () => {
    expect(seekStepForKey('ArrowLeft', false, 'second')).toBe('frame-back');
    expect(seekStepForKey('ArrowRight', false, 'ten')).toBe('frame-forward');
    expect(seekStepForKey('ArrowLeft', true, 'second')).toBe('second-back');
    expect(seekStepForKey('ArrowRight', true, 'second')).toBe('second-forward');
    expect(seekStepForKey('ArrowLeft', true, 'ten')).toBe('ten-back');
    expect(seekStepForKey('ArrowRight', true, 'ten')).toBe('ten-forward');
    expect(seekStepForKey('Home', true, 'ten')).toBe('first');
    expect(seekStepForKey('End', false, 'second')).toBe('last');
    expect(seekStepForKey('ArrowUp', false, 'ten')).toBeUndefined();
    expect(seekStepForKey('f', false, 'ten')).toBeUndefined();
});

test('an Alt-drag scrubs one frame per 4 pixels, whatever the length', () => {
    expect(SCRUB_PIXELS_PER_FRAME).toBe(4);
    expect(scrubFrame(100, 0, 43000)).toBe(100);
    expect(scrubFrame(100, 3, 43000)).toBe(100);
    expect(scrubFrame(100, 4, 43000)).toBe(101);
    expect(scrubFrame(100, 400, 43000)).toBe(200);
    expect(scrubFrame(100, -7, 43000)).toBe(99);
    expect(scrubFrame(100, 400, 150)).toBe(149);
    expect(scrubFrame(10, -400, 150)).toBe(0);
});

test('zoomAt keeps the point under the pointer and fits again at scale 1', () => {
    const zoomed = zoomAt(PAN_ZOOM_FIT, 2, 100, -50);
    expect(zoomed).toEqual({ scale: 2, x: -100, y: 50 });
    // the picture point under the pointer, (p - offset) / scale, does not move
    const again = zoomAt(zoomed, 1.5, 30, 20);
    expect((30 - again.x) / again.scale).toBeCloseTo((30 - zoomed.x) / zoomed.scale);
    expect((20 - again.y) / again.scale).toBeCloseTo((20 - zoomed.y) / zoomed.scale);
    expect(zoomAt(zoomed, 0.25, 0, 0)).toBe(PAN_ZOOM_FIT);
    expect(zoomAt(PAN_ZOOM_FIT, 1000, 0, 0).scale).toBe(32);
});

test('the seek bar and its Frames, In and Out go under the first large image output', () => {
    const seek = {
        id: 13,
        kind: 'video-seek',
        framesInput: 10,
        inInput: 11,
        outInput: 12,
    } as unknown as Input;
    const input = (id: number) => ({ id, kind: 'number' } as unknown as Input);
    const [frames, inInput, outInput] = [input(10), input(11), input(12)];
    const preview = { id: 11, kind: 'large-image' } as unknown as Output;
    const schema = (inputs: Input[], outputs: Output[]) =>
        ({ inputs, outputs } as unknown as NodeSchema);
    // in the order Frames, In, Out, whatever the schema's order
    expect(getSeekUnderPreview(schema([outInput, seek, frames, inInput], [preview]))).toEqual({
        input: seek,
        outputId: 11,
        companions: [frames, inInput, outInput],
    });
    // a companion the schema lacks is left out
    expect(getSeekUnderPreview(schema([seek, frames], [preview]))?.companions).toEqual([frames]);
    expect(getSeekUnderPreview(schema([seek], []))).toBeUndefined();
    expect(getSeekUnderPreview(schema([], [preview]))).toBeUndefined();
});
