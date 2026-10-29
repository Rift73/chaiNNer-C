import { mkdtempSync, readFileSync, rmSync } from 'fs';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { readClipboardForPaste, writeChainToClipboard } from '../../src/main/clipboard';

const { clipboardItems, written } = vi.hoisted(() => ({
    // what `clipboard.read()` gives: each item's types and payloads
    clipboardItems: [] as Record<string, Buffer>[],
    written: [] as unknown[][],
}));

vi.mock('electron/common', () => ({
    nativeImage: {
        createFromBuffer: (data: Buffer) => ({
            toBitmap: () => Buffer.from([0, 0, 0, 255]),
            toPNG: () => data,
        }),
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
            read: () =>
                Promise.resolve(
                    clipboardItems.map((formats) => ({
                        types: Object.keys(formats),
                        getType: (type: string) =>
                            Promise.resolve(new Blob([formats[type]], { type })),
                    }))
                ),
            write: (items: unknown[]) => {
                written.push(items);
                return Promise.resolve();
            },
        },
    };
});

// Electron 44 and later read and write a raw platform clipboard format under this type. Up to
// Electron 43, chaiNNer wrote the format with `clipboard.writeBuffer('application/chainner.chain',
// Buffer.from(json))` and read it with `clipboard.readBuffer`.
const CHAIN_TYPE = 'electron application/osclipboard;format="application/chainner.chain"';

const chainJson = JSON.stringify({
    nodes: [{ id: 'a', data: { inputData: { 0: 'C:\\画像\\über.png' } } }],
    edges: [],
});

let dir: string;
beforeEach(() => {
    dir = mkdtempSync(path.join(os.tmpdir(), 'chaiNNer-clipboard-test-'));
});
afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
    clipboardItems.length = 0;
    written.length = 0;
});

test('a chain is copied as its JSON under the format older versions read', async () => {
    await writeChainToClipboard(chainJson);

    expect(written).toHaveLength(1);
    const [items] = written as [{ items: Record<string, unknown> }[]];
    expect(items).toHaveLength(1);
    // a string payload is written as UTF-8: the bytes older versions wrote and read
    expect(items[0].items).toEqual({ [CHAIN_TYPE]: chainJson });
});

test('a chain copied by an older version is pasted', async () => {
    clipboardItems.push({ [CHAIN_TYPE]: Buffer.from(chainJson) });
    expect(await readClipboardForPaste(dir)).toEqual({ kind: 'chain', json: chainJson });
});

test('a chain is pasted before an image', async () => {
    clipboardItems.push({
        'image/png': Buffer.from('a PNG file'),
        [CHAIN_TYPE]: Buffer.from('{}'),
    });
    expect(await readClipboardForPaste(dir)).toEqual({ kind: 'chain', json: '{}' });
});

test('an image is pasted as a PNG file in the folder given', async () => {
    const png = Buffer.from('a PNG file');
    clipboardItems.push({ 'text/html': Buffer.from('<img>'), 'image/png': png });

    const paste = await readClipboardForPaste(dir);
    expect(paste.kind).toBe('image');
    const file = (paste as { path: string }).path;
    expect(path.dirname(file)).toBe(dir);
    expect(readFileSync(file)).toEqual(png);
});

test('a clipboard without a chain or an image gives its types', async () => {
    clipboardItems.push({ 'text/plain': Buffer.from('text'), 'text/html': Buffer.from('<b>') });
    expect(await readClipboardForPaste(dir)).toEqual({
        kind: 'other',
        types: ['text/plain', 'text/html'],
    });

    clipboardItems.length = 0;
    expect(await readClipboardForPaste(dir)).toEqual({ kind: 'other', types: [] });
});
