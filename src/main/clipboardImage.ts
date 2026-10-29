import { createHash } from 'crypto';
import { nativeImage } from 'electron/common';
import { ClipboardItem, clipboard } from 'electron/main';
import fs from 'fs/promises';
import path from 'path';
import { writeFileAtomic } from './util';

// The first 8 bytes of every PNG file
const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

// Raw clipboard formats that hold the copying app's own PNG file where Chromium's image read
// does not look: Windows apps that register "image/png" (Chromium reads "PNG") and Linux apps
// that offer "PNG" (Chromium reads "image/png").
const APP_PNG_FORMATS = [
    'electron application/osclipboard;format="PNG"',
    'electron application/osclipboard;format="image/png"',
];

const PNG_DATA_URL_PREFIX = 'data:image/png;base64,';

const isPng = (data: Buffer): boolean =>
    data.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE);

const hasTransparency = (bitmap: Buffer): boolean => {
    // 4 bytes per pixel, alpha last
    for (let i = 3; i < bitmap.length; i += 4) {
        if (bitmap[i] !== 255) return true;
    }
    return false;
};

const readBytes = async (item: Electron.ClipboardItem, type: string): Promise<Buffer> => {
    const blob = (await item.getType(type)) as Blob;
    return Buffer.from(await blob.arrayBuffer());
};

/**
 * Returns the clipboard's image as a PNG file. `type` is one of `item`'s image types.
 *
 * Chromium reads a clipboard image as a PNG file: the copying app's own (Windows' "PNG" format,
 * Linux's "image/png"), else the bitmap, encoded. Decoded, the image is premultiplied, so a PNG
 * encoded from it has black under transparent pixels and rounded colour under semi-transparent
 * ones (upstream chaiNNer #1511). An image with transparency is therefore pasted as the PNG file
 * read, or as the app's own PNG under a format Chromium does not read. An opaque image is
 * re-encoded, which makes it RGB: the app's own PNG may have an all-opaque alpha channel or
 * 16-bit samples, which would change what Load Image outputs.
 */
export const readClipboardImageAsPng = async (
    item: Electron.ClipboardItem,
    type: string
): Promise<Buffer> => {
    const read = await readBytes(item, type);
    const image = nativeImage.createFromBuffer(read);
    if (!hasTransparency(image.toBitmap())) {
        return image.toPNG();
    }
    const appPngs = await Promise.all(
        APP_PNG_FORMATS.filter((format) => item.types.includes(format)).map((format) =>
            readBytes(item, format)
        )
    );
    return appPngs.find(isPng) ?? read;
};

/**
 * Puts the image of a PNG data URL (a screenshot) on the clipboard. Chromium writes it from the
 * decoded bitmap (on Windows as a PNG file and a DIB), as Electron's former `clipboard.writeImage`
 * did.
 */
export const writePngDataUrlToClipboard = async (url: string): Promise<void> => {
    if (!url.startsWith(PNG_DATA_URL_PREFIX)) {
        throw new Error('Expected a PNG data URL');
    }
    const png = Buffer.from(url.slice(PNG_DATA_URL_PREFIX.length), 'base64');
    await clipboard.write([new ClipboardItem({ 'image/png': new Blob([png]) })]);
};

/**
 * Stores a pasted PNG in `dir` and returns its path. The file is named by its content: pasting the
 * same image again reuses the file (and with its path, the Load Image node's cached result), and
 * it is written through a temporary file, so a file with the name is always complete.
 */
export const storeClipboardPng = async (dir: string, png: Buffer): Promise<string> => {
    const hash = createHash('sha256').update(png).digest('hex').slice(0, 32);
    const file = path.join(dir, `chaiNNer-clipboard-${hash}.png`);
    const stored = await fs.stat(file).then(
        (stats) => stats.size === png.length,
        () => false
    );
    if (!stored) {
        await writeFileAtomic(file, png);
    }
    return file;
};
