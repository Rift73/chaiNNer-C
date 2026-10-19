import * as fs from 'fs';
import fsPromises from 'fs/promises';
import * as os from 'os';
import * as path from 'path';
import { afterEach, expect, test, vi } from 'vitest';
import { currentMigration } from '../../src/common/migrations';
import { RawSaveFile, SaveFile } from '../../src/main/SaveFile';

const dataDir = path.join(__dirname, '..', 'data');

test('Read save file with a UTF-8 byte order mark', async () => {
    // Windows PowerShell 5.1 writes one with `Set-Content -Encoding utf8`
    const filePath = path.join(dataDir, 'crop.chn');
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'save-file-test-'));
    try {
        const bomPath = path.join(dir, 'crop.chn');
        fs.writeFileSync(bomPath, `\uFEFF${fs.readFileSync(filePath, 'utf-8')}`, 'utf-8');

        expect(await SaveFile.read(bomPath)).toEqual(await SaveFile.read(filePath));
    } finally {
        fs.rmSync(dir, { recursive: true, force: true });
    }
});

test('Load Image Pairs index migrates to one edge per input', () => {
    const pairs = '00000000-0000-0000-0000-000000000001';
    const math = '00000000-0000-0000-0000-000000000002';
    const content = {
        nodes: [
            {
                id: pairs,
                type: 'generator',
                position: { x: 0, y: 0 },
                data: {
                    id: pairs,
                    schemaId: 'chainner:image:load_image_pairs',
                    inputData: { 0: 'a', 1: 'b', 2: 0, 3: 10, 4: 0 },
                },
            },
            {
                id: math,
                type: 'regularNode',
                position: { x: 400, y: 0 },
                data: { id: math, schemaId: 'chainner:utility:math', inputData: { 1: 'add' } },
            },
        ],
        edges: [
            {
                id: 'index-edge',
                source: pairs,
                sourceHandle: `${pairs}-8`,
                target: math,
                targetHandle: `${math}-0`,
                type: 'main',
            },
        ],
        viewport: { x: 0, y: 0, zoom: 1 },
    };
    // a v0.24 chain: the last migration splits Load Image Pairs into two Load Images
    const parsed = SaveFile.parse(
        JSON.stringify({ version: '0.24.0', content, migration: currentMigration - 1 })
    );

    const toMath = parsed.edges.filter((e) => e.target === math);
    expect(toMath).toHaveLength(1);
    expect(parsed.nodes.find((n) => n.id === toMath[0].source)?.data.nodeName).toBe(
        'Load Images A'
    );
    expect(toMath[0].sourceHandle).toBe(`${toMath[0].source}-4`);
});

afterEach(() => {
    vi.restoreAllMocks();
});

const errnoError = (code: string): NodeJS.ErrnoException =>
    Object.assign(new Error(`${code}: rename`), { code });

/** Calls `fn` with a new empty folder, removed afterwards. */
const withTempDir = async (fn: (dir: string) => Promise<void>): Promise<void> => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'save-file-test-'));
    try {
        await fn(dir);
    } finally {
        fs.rmSync(dir, { recursive: true, force: true });
    }
};

const savedContent = (file: string): unknown =>
    (JSON.parse(fs.readFileSync(file, 'utf-8')) as RawSaveFile).content;
const expectedContent = async (): Promise<unknown> => {
    const saveData = await SaveFile.read(path.join(dataDir, 'crop.chn'));
    return (JSON.parse(SaveFile.stringify(saveData, '0.0.0-test')) as RawSaveFile).content;
};

test('Write replaces the chain and leaves no temporary file', async () => {
    const saveData = await SaveFile.read(path.join(dataDir, 'crop.chn'));
    await withTempDir(async (dir) => {
        const file = path.join(dir, 'chain.chn');
        fs.writeFileSync(file, 'old chain', 'utf-8');

        await SaveFile.write(file, saveData, '0.0.0-test');

        expect(savedContent(file)).toEqual(await expectedContent());
        expect(fs.readdirSync(dir)).toEqual(['chain.chn']);
    });
});

const canSymlink = ((): boolean => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'save-file-test-'));
    try {
        fs.writeFileSync(path.join(dir, 'a'), '');
        fs.symlinkSync(path.join(dir, 'a'), path.join(dir, 'b'), 'file');
        return true;
    } catch {
        // Windows without Developer Mode or the privilege to create symbolic links
        return false;
    } finally {
        fs.rmSync(dir, { recursive: true, force: true });
    }
})();

test.skipIf(!canSymlink)('Write through a link updates the file it points to', async () => {
    const saveData = await SaveFile.read(path.join(dataDir, 'crop.chn'));
    await withTempDir(async (dir) => {
        const real = path.join(dir, 'real.chn');
        const link = path.join(dir, 'link.chn');
        fs.writeFileSync(real, 'old chain', 'utf-8');
        fs.symlinkSync(real, link, 'file');

        await SaveFile.write(link, saveData, '0.0.0-test');

        expect(fs.lstatSync(link).isSymbolicLink()).toBe(true);
        expect(savedContent(real)).toEqual(await expectedContent());
        expect(fs.readdirSync(dir).sort()).toEqual(['link.chn', 'real.chn']);
    });
});

for (const code of ['EPERM', 'EBUSY']) {
    test(`Write falls back to writing in place when the rename fails with ${code}`, async () => {
        const saveData = await SaveFile.read(path.join(dataDir, 'crop.chn'));
        await withTempDir(async (dir) => {
            const file = path.join(dir, 'chain.chn');
            fs.writeFileSync(file, 'old chain', 'utf-8');
            const rename = vi.spyOn(fsPromises, 'rename').mockRejectedValueOnce(errnoError(code));

            await SaveFile.write(file, saveData, '0.0.0-test');

            expect(rename).toHaveBeenCalledOnce();
            expect(savedContent(file)).toEqual(await expectedContent());
            expect(fs.readdirSync(dir)).toEqual(['chain.chn']);
        });
    });
}

test('Write keeps the old chain when the rename fails otherwise', async () => {
    const saveData = await SaveFile.read(path.join(dataDir, 'crop.chn'));
    await withTempDir(async (dir) => {
        const file = path.join(dir, 'chain.chn');
        fs.writeFileSync(file, 'old chain', 'utf-8');
        vi.spyOn(fsPromises, 'rename').mockRejectedValueOnce(errnoError('EIO'));

        await expect(SaveFile.write(file, saveData, '0.0.0-test')).rejects.toThrow('EIO');

        expect(fs.readFileSync(file, 'utf-8')).toBe('old chain');
        expect(fs.readdirSync(dir)).toEqual(['chain.chn']);
    });
});

for (const file of fs.readdirSync(dataDir)) {
    const filePath = path.join(dataDir, file);

    test(`Read save file ${file}`, async () => {
        const parsed = await SaveFile.read(filePath);
        expect(parsed).toMatchSnapshot();
    });
    test(`Write save file ${file}`, async () => {
        const json = SaveFile.stringify(await SaveFile.read(filePath), '0.0.0-test');
        const obj = JSON.parse(json) as RawSaveFile;
        delete obj.migration;
        delete obj.timestamp;
        expect(obj).toMatchSnapshot();
    });
}
