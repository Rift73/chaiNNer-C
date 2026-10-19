import { createHash } from 'crypto';
import fs from 'fs/promises';
import path from 'path';
import { writeFileAtomic } from './util';

// The first 8 bytes of every PNG file
const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

// Clipboard formats that hold the copying app's own PNG file: Windows apps register "PNG",
// Linux (and some Windows apps) use the MIME type.
const PNG_FORMATS = ['PNG', 'image/png'];

const hasTransparency = (bitmap: Buffer): boolean => {
    // 4 bytes per pixel, alpha last
    for (let i = 3; i < bitmap.length; i += 4) {
        if (bitmap[i] !== 255) return true;
    }
    return false;
};

interface ImageClipboard {
    readImage: () => Pick<Electron.NativeImage, 'toBitmap' | 'toPNG'>;
    readBuffer: (format: string) => Buffer;
}

/**
 * Returns the clipboard's image as a PNG file.
 *
 * `readImage()` stores the image premultiplied, so its PNG has black under transparent pixels and
 * rounded colour under semi-transparent ones (upstream chaiNNer #1511). An image with transparency
 * is therefore taken from the PNG the copying app put on the clipboard, when there is one. An
 * opaque image keeps `readImage()`'s PNG, which is RGB: the app's own PNG may have an all-opaque
 * alpha channel or 16-bit samples, which would change what Load Image outputs.
 *
 * The bitmap (a full copy of the image) is read for its alpha only when the app's PNG exists.
 */
export const readClipboardImageAsPng = (clipboard: ImageClipboard): Buffer => {
    let appPng: Buffer | undefined;
    for (const format of PNG_FORMATS) {
        const png = clipboard.readBuffer(format);
        if (png.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE)) {
            appPng = png;
            break;
        }
    }
    const image = clipboard.readImage();
    if (appPng && hasTransparency(image.toBitmap())) {
        return appPng;
    }
    return image.toPNG();
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
