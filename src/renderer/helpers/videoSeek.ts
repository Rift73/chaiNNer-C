import { evaluate, isNumericLiteral } from '@chainner/navi';
import {
    Input,
    NodeSchema,
    OutputId,
    OutputTypes,
    VideoSeekInput,
} from '../../common/common-types';
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

export interface SeekVideo {
    readonly frameCount: number;
    /** 0 for a video without a constant frame rate (a VapourSynth clip with a variable one). */
    readonly fps: number;
}

/**
 * The video a seek bar seeks, from its node's reported Frame Count and FPS; `undefined` until both
 * are known, and for a video without frames.
 */
export const getSeekVideo = (
    frameCount: number | undefined,
    fps: number | undefined
): SeekVideo | undefined => {
    if (frameCount === undefined || fps === undefined || frameCount < 1 || fps < 0) {
        return undefined;
    }
    return { frameCount: Math.round(frameCount), fps };
};

/** A frame number clamped to the frames of a video: 0..frameCount - 1. */
export const clampFrame = (frame: number, frameCount: number): number =>
    Math.min(Math.max(0, Math.round(frame)), Math.max(0, frameCount - 1));

export type SeekStep =
    | 'first'
    | 'second-back'
    | 'ten-back'
    | 'frame-back'
    | 'frame-forward'
    | 'ten-forward'
    | 'second-forward'
    | 'last';

/**
 * The frame a step of the seek bar goes to from `frame`, within the video's frames. A second is
 * the frame rate rounded (at least 1 frame, so 1 for an FPS of 0); a ten step is 10 frames.
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
        case 'ten-back':
            return clampFrame(frame - 10, frameCount);
        case 'frame-back':
            return clampFrame(frame - 1, frameCount);
        case 'frame-forward':
            return clampFrame(frame + 1, frameCount);
        case 'ten-forward':
            return clampFrame(frame + 10, frameCount);
        case 'second-forward':
            return clampFrame(frame + second, frameCount);
        case 'last':
            return clampFrame(frameCount - 1, frameCount);
        default:
            return assertNever(step);
    }
};

/**
 * A frame's time in the video, frame / fps, as H:MM:SS.mmm (rounded to the millisecond);
 * `undefined` without a frame rate (an FPS of 0).
 */
export const formatTimecode = (frame: number, fps: number): string | undefined => {
    if (!(fps > 0)) return undefined;
    const ms = Math.round((frame / fps) * 1000);
    const hours = Math.floor(ms / 3_600_000);
    const minutes = Math.floor(ms / 60_000) % 60;
    const seconds = Math.floor(ms / 1000) % 60;
    const pad = (n: number, digits: number) => String(n).padStart(digits, '0');
    return `${hours}:${pad(minutes, 2)}:${pad(seconds, 2)}.${pad(ms % 1000, 3)}`;
};

/**
 * The step a key makes: Left/Right a frame, Home/End the first/last frame. With Shift, Left/Right
 * step a second in the node and ten frames in the maximized preview (`shiftStep`).
 */
export const seekStepForKey = (
    key: string,
    shiftKey: boolean,
    shiftStep: 'second' | 'ten'
): SeekStep | undefined => {
    switch (key) {
        case 'ArrowLeft':
            if (!shiftKey) return 'frame-back';
            return shiftStep === 'second' ? 'second-back' : 'ten-back';
        case 'ArrowRight':
            if (!shiftKey) return 'frame-forward';
            return shiftStep === 'second' ? 'second-forward' : 'ten-forward';
        case 'Home':
            return 'first';
        case 'End':
            return 'last';
        default:
            return undefined;
    }
};

export interface SeekUnderPreview {
    readonly input: VideoSeekInput;
    readonly outputId: OutputId;
    /** The inputs the bar's Set start and Set end set (Frames, Start, End), in that order. */
    readonly companions: readonly Input[];
}

/**
 * The seek input drawn under a node's Preview instead of among its inputs, with its companions:
 * the node's first seek input, when the node has a large image output (the first one) to draw it
 * under.
 */
export const getSeekUnderPreview = (schema: NodeSchema): SeekUnderPreview | undefined => {
    const output = schema.outputs.find((o) => o.kind === 'large-image');
    const input = schema.inputs.find((i): i is VideoSeekInput => i.kind === 'video-seek');
    if (!output || !input) return undefined;
    const companions = [input.framesInput, input.inInput, input.outInput].flatMap(
        (id) => schema.inputs.find((i) => i.id === id) ?? []
    );
    return { input, outputId: output.id, companions };
};

/** Pixels of an Alt-drag on the maximized preview per frame, whatever the video's length. */
export const SCRUB_PIXELS_PER_FRAME = 4;

/** The frame an Alt-drag of `dx` pixels from `startFrame` goes to. */
export const scrubFrame = (startFrame: number, dx: number, frameCount: number): number =>
    clampFrame(startFrame + Math.trunc(dx / SCRUB_PIXELS_PER_FRAME), frameCount);

export interface PanZoom {
    readonly scale: number;
    /** The picture's offset from the centre of its view, in pixels. */
    readonly x: number;
    readonly y: number;
}
export const PAN_ZOOM_FIT: PanZoom = { scale: 1, x: 0, y: 0 };
const MIN_SCALE = 1;
const MAX_SCALE = 32;

/**
 * Zooms by `factor` about a point given from the centre of the view, so the picture's point under
 * the pointer stays under it. The scale stays within 1 (fit to the view) and 32; at 1 the picture
 * is centred again.
 */
export const zoomAt = (view: PanZoom, factor: number, px: number, py: number): PanZoom => {
    const scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, view.scale * factor));
    if (scale === MIN_SCALE) return PAN_ZOOM_FIT;
    const ratio = scale / view.scale;
    return { scale, x: px - (px - view.x) * ratio, y: py - (py - view.y) * ratio };
};
