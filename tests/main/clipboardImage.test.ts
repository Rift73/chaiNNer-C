import { mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'fs';
import fsPromises from 'fs/promises';
import os from 'os';
import path from 'path';
import { afterEach, expect, test, vi } from 'vitest';
import {
    readClipboardImageAsPng,
    storeClipboardPng,
    writePngDataUrlToClipboard,
} from '../../src/main/clipboardImage';

const { decoded, written } = vi.hoisted(() => ({
    // the image `nativeImage.createFromBuffer` decodes from each file: its bitmap and its PNG
    decoded: new Map<string, { bitmap: Buffer; png: Buffer }>(),
    written: [] as unknown[][],
}));

vi.mock('electron/common', () => ({
    nativeImage: {
        createFromBuffer: (data: Buffer) => {
            const image = decoded.get(data.toString('latin1'));
            return {
                toBitmap: () => image?.bitmap ?? Buffer.alloc(0),
                toPNG: () => image?.png ?? Buffer.alloc(0),
            };
        },
    },
}));

vi.mock('electron/main', () => {
    class ClipboardItem {
        readonly items: Record<string, unknown>;

        constructor(items: Record<string, unknown>) {
            this.items = items;
        }
    }
    return {
        ClipboardItem,
        clipboard: {
            write: (items: unknown[]) => {
                written.push(items);
                return Promise.resolve();
            },
        },
    };
});

afterEach(() => {
    decoded.clear();
    written.length = 0;
});

const pngFile = (content: string) =>
    Buffer.concat([
        Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
        Buffer.from(content),
    ]);
const appPng = pngFile("the copying app's PNG");
const readPng = pngFile("the PNG Chromium's image read gives");
const reencodedPng = pngFile('the decoded image, encoded again');

// a decoded bitmap: 4 bytes per pixel, alpha last
const bitmap = (...alphas: number[]) => Buffer.from(alphas.flatMap((alpha) => [0, 200, 0, alpha]));

const osFormat = (name: string) => `electron application/osclipboard;format="${name}"`;

// A clipboard item as `clipboard.read()` gives it: its types, and their payloads as Blobs
const clipboardItem = (formats: Partial<Record<string, Buffer>>) => {
    const getType = vi.fn((type: string) => {
        const data = formats[type];
        return data
            ? Promise.resolve(new Blob([data], { type }))
            : Promise.reject(new Error(`The type '${type}' was not found in the ClipboardItem`));
    });
    return {
        getType,
        item: { types: Object.keys(formats), getType } as unknown as Electron.ClipboardItem,
    };
};

const decodesTo = (file: Buffer, alphas: number[]) => {
    decoded.set(file.toString('latin1'), { bitmap: bitmap(...alphas), png: reencodedPng });
};

// upstream chaiNNer #1511: the decoded image is premultiplied, so a PNG encoded from it has black
// under transparent pixels
test.each([0, 128, 254])(
    'an image with transparency (alpha %i) is pasted as the PNG file read',
    async (alpha) => {
        // Windows: the app's "PNG" format; Linux: its "image/png"
        decodesTo(appPng, [255, alpha]);
        const { item } = clipboardItem({ 'image/png': appPng });
        expect(await readClipboardImageAsPng(item, 'image/png')).toEqual(appPng);
    }
);

test.each([osFormat('PNG'), osFormat('image/png')])(
    'an image with transparency is pasted as the PNG the app put under %s',
    async (format) => {
        decodesTo(readPng, [0]);
        const { item } = clipboardItem({ 'image/png': readPng, [format]: appPng });
        expect(await readClipboardImageAsPng(item, 'image/png')).toEqual(appPng);
    }
);

test('"PNG" is preferred to "image/png"', async () => {
    decodesTo(readPng, [0]);
    const { item } = clipboardItem({
        'image/png': readPng,
        [osFormat('PNG')]: appPng,
        [osFormat('image/png')]: pngFile('other'),
    });
    expect(await readClipboardImageAsPng(item, 'image/png')).toEqual(appPng);
});

test('an app format that is not a PNG file is passed over', async () => {
    decodesTo(readPng, [0]);
    const { item } = clipboardItem({
        'image/png': readPng,
        [osFormat('PNG')]: Buffer.from('BM, a bitmap file'),
    });
    expect(await readClipboardImageAsPng(item, 'image/png')).toEqual(readPng);
});

test('an opaque image is encoded again, as before', async () => {
    decodesTo(appPng, [255, 255]);
    const { item, getType } = clipboardItem({
        'image/png': appPng,
        [osFormat('image/png')]: appPng,
    });
    expect(await readClipboardImageAsPng(item, 'image/png')).toEqual(reencodedPng);

    // the app's formats are read only for an image with transparency
    expect(getType.mock.calls).toEqual([['image/png']]);
});

test('an image read under another image type is pasted as a PNG', async () => {
    const jpeg = Buffer.from('a JPEG file');
    decodesTo(jpeg, [255]);
    const { item } = clipboardItem({ 'image/jpeg': jpeg });
    expect(await readClipboardImageAsPng(item, 'image/jpeg')).toEqual(reencodedPng);
});

test('a screenshot goes on the clipboard as its PNG file', async () => {
    const screenshot = pngFile('a screenshot');
    await writePngDataUrlToClipboard(`data:image/png;base64,${screenshot.toString('base64')}`);

    expect(written).toHaveLength(1);
    const [items] = written as [{ items: Record<string, Blob> }[]];
    expect(items).toHaveLength(1);
    expect(Object.keys(items[0].items)).toEqual(['image/png']);
    expect(Buffer.from(await items[0].items['image/png'].arrayBuffer())).toEqual(screenshot);
});

test('only a PNG data URL is put on the clipboard', async () => {
    await expect(writePngDataUrlToClipboard('data:image/jpeg;base64,AAAA')).rejects.toThrow(
        'Expected a PNG data URL'
    );
    expect(written).toHaveLength(0);
});

test('a pasted image is stored once, by its content, and complete', async () => {
    const dir = mkdtempSync(path.join(os.tmpdir(), 'chaiNNer-clipboard-test-'));
    try {
        const first = await storeClipboardPng(dir, appPng);
        expect(readFileSync(first)).toEqual(appPng);

        // the same image again: the same file, not written again
        const { mtimeMs } = statSync(first);
        const rename = vi.spyOn(fsPromises, 'rename');
        expect(await storeClipboardPng(dir, Buffer.from(appPng))).toBe(first);
        expect(rename).not.toHaveBeenCalled();
        expect(statSync(first).mtimeMs).toBe(mtimeMs);
        rename.mockRestore();

        // another image: another file
        const second = await storeClipboardPng(dir, readPng);
        expect(second).not.toBe(first);
        expect(readFileSync(second)).toEqual(readPng);
        expect(readdirSync(dir).sort()).toEqual(
            [path.basename(first), path.basename(second)].sort()
        );

        // a cut-off file of an image is replaced whole
        writeFileSync(first, appPng.subarray(0, 4));
        expect(await storeClipboardPng(dir, appPng)).toBe(first);
        expect(readFileSync(first)).toEqual(appPng);
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});
