import { evaluate, isNumericLiteral } from '@chainner/navi';
import { OutputId, OutputTypes } from '../../common/common-types';
import { log } from '../../common/log';
import { getChainnerScope } from '../../common/types/chainner-scope';
import { fromJson } from '../../common/types/json';
import { assertNever } from '../../common/util';

/**
 * The number an output's broadcast type holds (a node's Frame Count or FPS from its last run), or
 * `undefined` when the type is not one finite number.
 */
export const numericOutputValue = (
    types: OutputTypes | undefined,
    outputId: OutputId
): number | undefined => {
    const expression = types?.[outputId];
    if (expression == null) return undefined;
    try {
        const type = evaluate(fromJson(expression), getChainnerScope());
        if (isNumericLiteral(type) && Number.isFinite(type.value)) return type.value;
    } catch (error) {
        log.error(error);
    }
    return undefined;
};

/** A frame number clamped to the frames of a video: 0..frameCount - 1. */
export const clampFrame = (frame: number, frameCount: number): number =>
    Math.min(Math.max(0, Math.round(frame)), Math.max(0, frameCount - 1));

export type SeekStep =
    | 'first'
    | 'second-back'
    | 'frame-back'
    | 'frame-forward'
    | 'second-forward'
    | 'last';

/**
 * The frame a step of the seek bar goes to from `frame`, within the video's frames. A second is
 * the frame rate rounded (at least 1 frame).
 */
export const stepFrame = (
    step: SeekStep,
    frame: number,
    frameCount: number,
    fps: number
): number => {
    const second = Math.max(1, Math.round(fps));
    switch (step) {
        case 'first':
            return 0;
        case 'second-back':
            return clampFrame(frame - second, frameCount);
        case 'frame-back':
            return clampFrame(frame - 1, frameCount);
        case 'frame-forward':
            return clampFrame(frame + 1, frameCount);
        case 'second-forward':
            return clampFrame(frame + second, frameCount);
        case 'last':
            return clampFrame(frameCount - 1, frameCount);
        default:
            return assertNever(step);
    }
};

/** A frame's time in the video, frame / fps, as H:MM:SS.mmm (rounded to the millisecond). */
export const formatTimecode = (frame: number, fps: number): string => {
    const ms = Math.round((frame / fps) * 1000);
    const hours = Math.floor(ms / 3_600_000);
    const minutes = Math.floor(ms / 60_000) % 60;
    const seconds = Math.floor(ms / 1000) % 60;
    const pad = (n: number, digits: number) => String(n).padStart(digits, '0');
    return `${hours}:${pad(minutes, 2)}:${pad(seconds, 2)}.${pad(ms % 1000, 3)}`;
};
