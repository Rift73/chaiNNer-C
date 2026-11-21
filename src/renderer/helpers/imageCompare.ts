import { evaluate, isStringLiteral } from '@chainner/navi';
import {
    EdgeData,
    ImageCompare,
    InputData,
    InputId,
    NodeData,
    NodeSchema,
    OutputId,
    OutputTypes,
    SchemaId,
    Size,
} from '../../common/common-types';
import { withInputOverrides } from '../../common/input-override-common';
import { log } from '../../common/log';
import { ImagePreviewBackground } from '../../common/settings/settings';
import { getChainnerScope } from '../../common/types/chainner-scope';
import { fromJson } from '../../common/types/json';
import { PanZoom, getUpstreamSeeks, hasUpstreamFullLoop, zoomAt } from './videoSeek';
import type { Edge, Node } from 'reactflow';

/** One of the previews a large image output's broadcast has. */
export interface PreviewImage {
    readonly width: number;
    readonly height: number;
    readonly url: string;
}
/**
 * The preview a view `realWidth` px wide shows: the smallest at least that wide, else the
 * largest there is (a gesture's frame may have only the previews of the views on screen then).
 */
export const pickImage = (previews: readonly PreviewImage[], realWidth: number): PreviewImage => {
    const sorted = [...previews].sort((a, b) => a.width - b.width);
    const found = sorted.find((preview) => preview.width >= realWidth);
    return found ?? sorted[sorted.length - 1];
};

export const DEFAULT_SPLIT = 0.5;
/** The split's step for a Left or Right key. */
export const SPLIT_KEY_STEP = 0.01;

/**
 * The large image outputs a node shows in one viewer: all of them when there are two or more,
 * none otherwise. The viewer takes the place of the first; the others draw nothing.
 */
export const getCompareOutputs = (schema: NodeSchema): readonly OutputId[] => {
    const outputs = schema.outputs.filter((o) => o.kind === 'large-image').map((o) => o.id);
    return outputs.length >= 2 ? outputs : [];
};

/** The longest name an image of the viewer takes. */
export const IMAGE_NAME_MAX_LENGTH = 64;

/** The inputs whose images a node's viewer shows (the input with its output's id): these can be named. */
export const getNameableInputs = (schema: NodeSchema): ReadonlySet<InputId> => {
    const outputs = new Set<number>(getCompareOutputs(schema));
    return new Set(schema.inputs.filter((i) => outputs.has(i.id)).map((i) => i.id));
};

/** A name with its spaces collapsed, cut to the longest name. */
const cleanName = (name: string): string =>
    name.trim().replace(/\s+/g, ' ').slice(0, IMAGE_NAME_MAX_LENGTH).trim();

/**
 * What an image is called: the name fed to it (`fed`, cleaned as a typed name is), else the name
 * given to it, else its input's label.
 */
export const imageLabel = (label: string, name: string | undefined, fed?: string): string => {
    const fedName = fed === undefined ? '' : cleanName(fed);
    const trimmed = name?.trim();
    return fedName || trimmed || label;
};

/**
 * The outputs that give a viewer's images their fed names, by the image's output: the node's text
 * outputs without a handle, the first naming the first image, and so on. Each holds the text of
 * the input with its id (View Image's Name inputs) as its broadcast type. None without a viewer.
 */
export const getNameOutputs = (schema: NodeSchema): ReadonlyMap<OutputId, OutputId> => {
    const images = getCompareOutputs(schema);
    const names = schema.outputs.filter((o) => o.kind === 'generic' && !o.hasHandle);
    return new Map(
        images.flatMap((id, k): [OutputId, OutputId][] => {
            const name = names.at(k);
            return name ? [[id, name.id]] : [];
        })
    );
};

/**
 * The name fed to an image: the text its name output's broadcast type holds from the last run,
 * while the name output's input is connected; `undefined` otherwise.
 */
export const fedImageName = (
    types: OutputTypes | undefined,
    nameOutputId: OutputId | undefined,
    connectedInputs: ReadonlySet<InputId>
): string | undefined => {
    if (nameOutputId === undefined || !connectedInputs.has(nameOutputId as number as InputId)) {
        return undefined;
    }
    const expression = types?.[nameOutputId];
    if (expression == null) return undefined;
    try {
        const type = evaluate(fromJson(expression), getChainnerScope());
        if (isStringLiteral(type)) return type.value;
    } catch (error) {
        log.error(error);
    }
    return undefined;
};

/**
 * The choice with the image of `inputId` named `name`, its spaces collapsed and cut to the longest
 * name; an empty name, or the input's own label, removes the name.
 */
export const withImageName = (
    choice: ImageCompare | undefined,
    inputId: InputId,
    name: string,
    label: string
): ImageCompare => {
    const clean = cleanName(name);
    const others = Object.entries(choice?.names ?? {}).filter(([id]) => Number(id) !== inputId);
    const named = clean && clean.toUpperCase() !== label.toUpperCase();
    const names = named ? [...others, [String(inputId), clean]] : others;
    return {
        ...choice,
        names: names.length > 0 ? (Object.fromEntries(names) as ImageCompare['names']) : undefined,
    };
};

export interface ComparePair {
    readonly a: OutputId;
    readonly b: OutputId;
}

/**
 * The images to compare, from the saved choice and the images there are: a saved image that is
 * there stays, the others are the image with the lowest id and the next one that is not the other
 * side. Fewer than two images compare nothing.
 */
export const getComparePair = (
    available: readonly OutputId[],
    choice: ImageCompare | undefined
): ComparePair | undefined => {
    if (available.length < 2) return undefined;
    const sorted = [...available].sort((x, y) => x - y);
    const has = (id: OutputId | undefined): id is OutputId =>
        id !== undefined && sorted.includes(id);
    const savedA = has(choice?.a) ? choice?.a : undefined;
    const savedB = has(choice?.b) && choice?.b !== savedA ? choice?.b : undefined;
    const a = savedA ?? sorted.find((id) => id !== savedB)!;
    const b = savedB ?? sorted.find((id) => id !== a)!;
    return { a, b };
};

/**
 * The image on side A of a viewer whose connected images are `connected`: the pair's A
 * (getComparePair), or the one image when there is one.
 */
export const getSideA = (
    connected: readonly OutputId[],
    choice: ImageCompare | undefined
): OutputId | undefined => getComparePair(connected, choice)?.a ?? connected[0];

/**
 * The inputs of a viewer's node that leave out its images but those of `kept` for one run: the
 * input of each other image and its name input have no value, so the run takes neither of their
 * connections (withInputOverrides).
 */
const withoutImagesBut = (schema: NodeSchema, kept: readonly OutputId[]): InputData => {
    const names = getNameOutputs(schema);
    return Object.fromEntries(
        getCompareOutputs(schema)
            .filter((id) => !kept.includes(id))
            .flatMap((id) => {
                const name = names.get(id);
                return name === undefined ? [id] : [id, name];
            })
            .map((id) => [id, undefined])
    ) as InputData;
};

/**
 * What the seek bar of viewer `viewerId` renders: the inputs of the viewer's node for a render up
 * to it (renderPreview). A move of the bar (`shown`: its side A image) leaves out the images but
 * that one and those the node requires, which it cannot run without (View Image's Image 1), so
 * that only what feeds these runs; the images rendered keep their names, the others lose theirs.
 * The end of a gesture (`shown`: 'all') leaves out none: every image connected renders, as Run
 * this frame renders them. `undefined` when the render shows nothing new or cannot run: no seek
 * bar the viewer moves (getUpstreamSeeks) feeds the side A image (any image, for 'all'), or a loop
 * source that runs all its items (hasUpstreamFullLoop) feeds an image the render runs.
 */
export const getSeekRender = (
    viewerId: string,
    shown: OutputId | 'all',
    nodes: readonly Node<NodeData>[],
    edges: readonly Edge<EdgeData>[],
    getSchema: (schemaId: SchemaId) => NodeSchema
): InputData | undefined => {
    const viewer = nodes.find((n) => n.id === viewerId);
    if (viewer === undefined) return undefined;
    const schema = getSchema(viewer.data.schemaId);
    const data = nodes.map((n) => n.data);
    const edgesWith = (inputs: InputData) =>
        withInputOverrides(nodes, edges, new Map([[viewerId, inputs]])).edges;

    const kept = shown === 'all' ? getCompareOutputs(schema) : [shown];
    // which bars feed the image does not depend on their videos
    const seeksOfShown = getUpstreamSeeks(
        viewerId,
        data,
        edgesWith(withoutImagesBut(schema, kept)),
        getSchema,
        () => undefined
    );
    if (seeksOfShown.length === 0) return undefined;

    const required = getCompareOutputs(schema).filter(
        (id) => schema.inputs.find((i) => (i.id as number) === id)?.optional === false
    );
    const inputs = withoutImagesBut(schema, [...kept, ...required]);
    return hasUpstreamFullLoop(viewerId, data, edgesWith(inputs), getSchema) ? undefined : inputs;
};

/** The pair with `side` set to `id`; choosing the other side's image swaps the two. */
export const chooseCompareSide = (
    pair: ComparePair,
    side: keyof ComparePair,
    id: OutputId
): ComparePair => {
    if (side === 'a') return id === pair.b ? { a: id, b: pair.a } : { ...pair, a: id };
    return id === pair.a ? { a: pair.b, b: id } : { ...pair, b: id };
};

/** A split position within the picture: 0 (all B) to 1 (all A); anything else is the middle. */
export const clampSplit = (split: number | undefined): number => {
    if (split === undefined || !Number.isFinite(split)) return DEFAULT_SPLIT;
    return Math.min(1, Math.max(0, split));
};

/** The split under a pointer at `x`, for a picture from `left` that is `width` wide. */
export const splitAt = (x: number, left: number, width: number): number =>
    width > 0 ? clampSplit((x - left) / width) : DEFAULT_SPLIT;

/** The split a Left or Right key moves to, or undefined for another key. */
export const splitForKey = (split: number, key: string): number | undefined => {
    if (key === 'ArrowLeft') return clampSplit(split - SPLIT_KEY_STEP);
    if (key === 'ArrowRight') return clampSplit(split + SPLIT_KEY_STEP);
    return undefined;
};

/** The width an image of `image`'s size is drawn at when it is contained in a box of `box`. */
export const containedWidth = (box: Readonly<Size>, image: Readonly<Size>): number =>
    Math.min(box.width, (box.height * image.width) / image.height);

/**
 * Whether the maximized view draws an image with nearest-neighbour pixels: when one pixel of the
 * preview it shows covers at least 2 device pixels. Widths on the screen are in CSS pixels, `dpr`
 * device pixels each.
 */
export const isPixelated = (displayWidth: number, previewWidth: number, dpr: number): boolean =>
    displayWidth * dpr >= 2 * previewWidth;

/** The zoom as a whole percentage of the image's real size, in device pixels. */
export const realSizePercent = (displayWidth: number, realWidth: number, dpr: number): number =>
    Math.round(((displayWidth * dpr) / realWidth) * 100);

/**
 * The view that shows an image at its real size (one pixel per device pixel), keeping the point
 * at the centre of the view in place; `fitWidth` is the image's width in CSS pixels at the fit
 * (scale 1). A small image's real size is below the fit.
 */
export const realSizeView = (
    view: PanZoom,
    fitWidth: number,
    realWidth: number,
    dpr: number
): PanZoom => {
    const scale = realWidth / (fitWidth * dpr);
    if (scale >= 1) return zoomAt(view, scale / view.scale, 0, 0);
    const ratio = scale / view.scale;
    return { scale, x: view.x * ratio, y: view.y * ratio };
};

const BACKGROUND_ORDER: readonly ImagePreviewBackground[] = ['theme', 'white', 'black'];

/** The background the viewer's button goes to next: theme, white, black, theme again. */
export const nextPreviewBackground = (background: ImagePreviewBackground): ImagePreviewBackground =>
    BACKGROUND_ORDER[(BACKGROUND_ORDER.indexOf(background) + 1) % BACKGROUND_ORDER.length];

/** The empty area's colour: the place's own theme colour, or the solid colour chosen. */
export const emptyAreaColor = (background: ImagePreviewBackground, themeColor: string): string =>
    background === 'theme' ? themeColor : background;

// https://stackoverflow.com/a/65129916/7595472
const CHECKERBOARD = 'repeating-conic-gradient(#CDCDCD 0% 25%, #FFFFFF 0% 50%) 50% / 20px 20px';

/**
 * What is drawn behind an image's own pixels: with the theme's background a checkerboard for an
 * image with alpha (nothing otherwise), with a solid one that colour.
 */
export const imageBackdrop = (background: ImagePreviewBackground, channels: number): string => {
    if (background !== 'theme') return background;
    return channels === 4 ? CHECKERBOARD : '';
};
