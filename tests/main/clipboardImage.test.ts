import { expect, test } from 'vitest';
import { readClipboardImageAsPng } from '../../src/main/clipboardImage';

const pngFile = (content: string) =>
    Buffer.concat([
        Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
        Buffer.from(content),
    ]);
const appPng = pngFile("the copying app's PNG");
const readImagePng = pngFile("readImage()'s PNG");

// readImage()'s bitmap: 4 bytes per pixel, alpha last
const bitmap = (...alphas: number[]) => Buffer.from(alphas.flatMap((alpha) => [0, 200, 0, alpha]));

const fakeClipboard = (pixels: Buffer, formats: Partial<Record<string, Buffer>>) => ({
    readImage: () => ({ toBitmap: () => pixels, toPNG: () => readImagePng }),
    readBuffer: (format: string) => formats[format] ?? Buffer.alloc(0),
});

// upstream chaiNNer #1511: readImage() is premultiplied, so its PNG has black under transparent pixels
test.each([
    ['PNG', 0],
    ['PNG', 128],
    ['image/png', 254],
])(
    'an image with transparency is pasted from the PNG the app put under "%s" (alpha %i)',
    (format, alpha) => {
        const clipboard = fakeClipboard(bitmap(255, alpha), { [format]: appPng });
        expect(readClipboardImageAsPng(clipboard)).toBe(appPng);
    }
);

test('"PNG" is read before "image/png"', () => {
    const clipboard = fakeClipboard(bitmap(0), { PNG: appPng, 'image/png': pngFile('other') });
    expect(readClipboardImageAsPng(clipboard)).toBe(appPng);
});

test("an opaque image keeps readImage()'s PNG, as before", () => {
    const clipboard = fakeClipboard(bitmap(255, 255), { PNG: appPng, 'image/png': appPng });
    expect(readClipboardImageAsPng(clipboard)).toBe(readImagePng);
});

test("without a PNG from the app, an image with transparency keeps readImage()'s PNG", () => {
    expect(readClipboardImageAsPng(fakeClipboard(bitmap(0), {}))).toBe(readImagePng);
    const notPng = fakeClipboard(bitmap(0), { PNG: Buffer.from('BM, a bitmap file') });
    expect(readClipboardImageAsPng(notPng)).toBe(readImagePng);
});
