import { evaluate, isNumericLiteral } from '@chainner/navi';
import {
    EdgeData,
    Input,
    InputData,
    InputId,
    NodeData,
    NodeSchema,
    OutputId,
    OutputTypes,
    SchemaId,
    VideoSeekInput,
} from '../../common/common-types';
import { log } from '../../common/log';
import { getConnectedInputs } from '../../common/nodes/connectedInputs';
import { getChainnerScope } from '../../common/types/chainner-scope';
import { fromJson } from '../../common/types/json';
import { assertNever } from '../../common/util';
import type { Edge } from 'reactflow';

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

/** The video a node's seek bar seeks, from the Frame Count and FPS its last run reported. */
export const getNodeSeekVideo = (
    types: OutputTypes | undefined,
    input: Pick<VideoSeekInput, 'frameCountOutput' | 'fpsOutput'>
): SeekVideo | undefined =>
    getSeekVideo(
        numericOutputValue(types, input.frameCountOutput),
        numericOutputValue(types, input.fpsOutput)
    );

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

/** A node's seek bar: its first seek input. */
export const getSeekInput = (schema: NodeSchema): VideoSeekInput | undefined =>
    schema.inputs.find((i): i is VideoSeekInput => i.kind === 'video-seek');

/**
 * The seek input drawn under a node's Preview instead of among its inputs, with its companions:
 * the node's first seek input, when the node has a large image output (the first one) to draw it
 * under.
 */
export const getSeekUnderPreview = (schema: NodeSchema): SeekUnderPreview | undefined => {
    const output = schema.outputs.find((o) => o.kind === 'large-image');
    const input = getSeekInput(schema);
    if (!output || !input) return undefined;
    const companions = [input.framesInput, input.inInput, input.outInput].flatMap(
        (id) => schema.inputs.find((i) => i.id === id) ?? []
    );
    return { input, outputId: output.id, companions };
};

/** The value of a seek input's `framesInput` enum for a frame range (Start..End). */
export const FRAMES_RANGE = 'range';

/**
 * The inputs that run a seek bar's node on frame `n` alone, for one run: Frames a Range from `n`
 * to `n`. The node keeps its own Frames, Start and End.
 */
export const oneFrameOverride = (
    input: Pick<VideoSeekInput, 'framesInput' | 'inInput' | 'outInput'>,
    n: number
): InputData => ({
    [input.framesInput]: FRAMES_RANGE,
    [input.inInput]: n,
    [input.outInput]: n,
});

/** The frame a node's seek bar shows: its seek value, or the input's default when unset. */
const seekValue = (node: NodeData, input: VideoSeekInput): number => {
    const value = node.inputData[input.id];
    return typeof value === 'number' ? value : input.def;
};

/**
 * The overrides of "Run this frame": every node with a seek bar runs one frame, so the loops a
 * chain joins run one frame each. A node in `pressed` runs the frame given for it (the pressed
 * Load Video's, or every bar a View Image's seek bar moves). Any other node runs the frame its own
 * bar shows: its seek value (unset: the input's default, as the bar shows), clamped to its video
 * when its last run reported the Frame Count; unclamped, a frame past the end fails the run (Start
 * past the last frame) instead of running another one. A node whose seek input is connected shows
 * no frame of its own and keeps its own Frames.
 */
export const runThisFrameOverrides = (
    nodes: readonly NodeData[],
    edges: readonly Edge<EdgeData>[],
    getSchema: (schemaId: SchemaId) => NodeSchema,
    getTypes: (nodeId: string) => OutputTypes | undefined,
    pressed: ReadonlyMap<string, number>
): Map<string, InputData> => {
    const entries = nodes.flatMap((node): [string, InputData][] => {
        const input = getSeekInput(getSchema(node.schemaId));
        if (input === undefined) return [];
        const pressedFrame = pressed.get(node.id);
        if (pressedFrame !== undefined) return [[node.id, oneFrameOverride(input, pressedFrame)]];
        if (getConnectedInputs(node.id, edges).has(input.id)) return [];
        const frame = seekValue(node, input);
        const video = getNodeSeekVideo(getTypes(node.id), input);
        const shown = video ? clampFrame(frame, video.frameCount) : frame;
        return [[node.id, oneFrameOverride(input, shown)]];
    });
    return new Map(entries);
};

/** A seek bar that a node's seek control moves. */
export interface UpstreamSeek {
    readonly nodeId: string;
    readonly input: VideoSeekInput;
    /** The frame its bar shows. */
    readonly frame: number;
    /** Its video, once its last run reported the Frame Count and FPS. */
    readonly video: SeekVideo | undefined;
}

/** The nodes before node `nodeId`: every node reached by walking the edges backwards from it. */
const nodesBefore = (nodeId: string, edges: readonly Edge<EdgeData>[]): Set<string> => {
    const sources = new Map<string, string[]>();
    for (const e of edges) {
        const list = sources.get(e.target);
        if (list) list.push(e.source);
        else sources.set(e.target, [e.source]);
    }
    const reached = new Set<string>();
    const stack = [...(sources.get(nodeId) ?? [])];
    for (let id = stack.pop(); id !== undefined; id = stack.pop()) {
        if (!reached.has(id)) {
            reached.add(id);
            stack.push(...(sources.get(id) ?? []));
        }
    }
    reached.delete(nodeId);
    return reached;
};

/**
 * The seek bars before node `nodeId`, which its seek control moves: every node reached by walking
 * the edges backwards from the node whose seek input is not connected and which is not locked, in
 * the order of `nodes`. A connected seek input shows no frame of its own; a locked node keeps its
 * frame (Run this frame still runs it).
 */
export const getUpstreamSeeks = (
    nodeId: string,
    nodes: readonly NodeData[],
    edges: readonly Edge<EdgeData>[],
    getSchema: (schemaId: SchemaId) => NodeSchema,
    getTypes: (nodeId: string) => OutputTypes | undefined
): UpstreamSeek[] => {
    const reached = nodesBefore(nodeId, edges);
    return nodes.flatMap((node): UpstreamSeek[] => {
        if (!reached.has(node.id) || node.isLocked) return [];
        const input = getSeekInput(getSchema(node.schemaId));
        if (input === undefined || getConnectedInputs(node.id, edges).has(input.id)) return [];
        return [
            {
                nodeId: node.id,
                input,
                frame: seekValue(node, input),
                video: getNodeSeekVideo(getTypes(node.id), input),
            },
        ];
    });
};

/**
 * Whether a loop source before node `nodeId` runs all its items in Run this frame: a node of the
 * generator kind without a seek bar (Load Images, a range, a spritesheet's tiles) or whose seek
 * input is connected. Its loop then joins one frame of a video with all of its own items, which
 * the run refuses.
 */
export const hasUpstreamFullLoop = (
    nodeId: string,
    nodes: readonly NodeData[],
    edges: readonly Edge<EdgeData>[],
    getSchema: (schemaId: SchemaId) => NodeSchema
): boolean => {
    const reached = nodesBefore(nodeId, edges);
    return nodes.some((node) => {
        if (!reached.has(node.id)) return false;
        const schema = getSchema(node.schemaId);
        if (schema.kind !== 'generator') return false;
        const input = getSeekInput(schema);
        return input === undefined || getConnectedInputs(node.id, edges).has(input.id);
    });
};

/**
 * The video a seek control of several bars seeks: as many frames as the shortest known video, the
 * first known one's frame rate; `undefined` until a bar's video is known.
 */
export const getUpstreamSeekVideo = (seeks: readonly UpstreamSeek[]): SeekVideo | undefined => {
    const videos = seeks.flatMap((s) => s.video ?? []);
    if (videos.length === 0) return undefined;
    return {
        frameCount: Math.min(...videos.map((v) => v.frameCount)),
        fps: videos[0].fps,
    };
};

/**
 * The seek values that move every bar to frame `n`: each bar's frame clamped to its own video
 * when it is known.
 */
export const seekAllTo = (
    seeks: readonly UpstreamSeek[],
    n: number
): { readonly nodeId: string; readonly inputId: InputId; readonly frame: number }[] =>
    seeks.map((s) => ({
        nodeId: s.nodeId,
        inputId: s.input.id,
        frame: s.video ? clampFrame(n, s.video.frameCount) : Math.max(0, Math.round(n)),
    }));

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
 * the pointer stays under it. The scale stays within 1 (fit to the view) and 32, or from a view
 * below the fit (a small picture at its real size) within that view's scale and 32; at 1 the
 * picture is centred again.
 */
export const zoomAt = (view: PanZoom, factor: number, px: number, py: number): PanZoom => {
    const minScale = Math.min(MIN_SCALE, view.scale);
    const scale = Math.min(MAX_SCALE, Math.max(minScale, view.scale * factor));
    if (scale === MIN_SCALE) return PAN_ZOOM_FIT;
    const ratio = scale / view.scale;
    return { scale, x: px - (px - view.x) * ratio, y: py - (py - view.y) * ratio };
};
