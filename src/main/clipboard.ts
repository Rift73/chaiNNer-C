import { ClipboardItem, clipboard } from 'electron/main';
import { readClipboardImageAsPng, storeClipboardPng } from './clipboardImage';

/**
 * A copied chain: its JSON as UTF-8 under the raw clipboard format `application/chainner.chain`,
 * where chaiNNer has always put it (with `clipboard.writeBuffer` up to Electron 43), so chains
 * copy between this and older versions both ways.
 */
const CHAIN_FORMAT = 'electron application/osclipboard;format="application/chainner.chain"';

// The clipboard types pasted as an image, the first one found. Chromium reads each as the
// clipboard's image.
const IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/gif', 'image/bmp', 'image/tiff'];

export type ClipboardPaste =
    | { kind: 'chain'; json: string }
    | { kind: 'image'; path: string }
    | { kind: 'other'; types: string[] };

export const writeChainToClipboard = async (json: string): Promise<void> => {
    await clipboard.write([new ClipboardItem({ [CHAIN_FORMAT]: json })]);
};

/**
 * What a paste finds on the clipboard: a copied chain, else an image (stored in `dir` as a PNG
 * file), else just the clipboard's types.
 */
export const readClipboardForPaste = async (dir: string): Promise<ClipboardPaste> => {
    const items = await clipboard.read();

    const chainItem = items.find((item) => item.types.includes(CHAIN_FORMAT));
    if (chainItem) {
        const blob = await chainItem.getType(CHAIN_FORMAT);
        return { kind: 'chain', json: await blob.text() };
    }

    const imageItem = items.find((item) => IMAGE_TYPES.some((type) => item.types.includes(type)));
    if (imageItem) {
        const imageType = IMAGE_TYPES.find((type) => imageItem.types.includes(type))!;
        const png = await readClipboardImageAsPng(imageItem, imageType);
        return { kind: 'image', path: await storeClipboardPng(dir, png) };
    }

    return { kind: 'other', types: items.flatMap((item) => item.types) };
};
