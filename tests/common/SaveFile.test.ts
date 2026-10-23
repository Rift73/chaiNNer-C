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
    // a v0.24 chain: the second to last migration splits Load Image Pairs into two Load Images
    const parsed = SaveFile.parse(
        JSON.stringify({ version: '0.24.0', content, migration: currentMigration - 2 })
    );

    const toMath = parsed.edges.filter((e) => e.target === math);
    expect(toMath).toHaveLength(1);
    expect(parsed.nodes.find((n) => n.id === toMath[0].source)?.data.nodeName).toBe(
        'Load Images A'
    );
    expect(toMath[0].sourceHandle).toBe(`${toMath[0].source}-4`);
});

test("Load Video's Use limit migrates to a frame range", () => {
    const limited = '00000000-0000-0000-0000-000000000001';
    const unlimited = '00000000-0000-0000-0000-000000000002';
    const old = '00000000-0000-0000-0000-000000000003';
    const number = '00000000-0000-0000-0000-000000000004';
    const loadVideo = (id: string, inputData: Record<number, string | number>) => ({
        id,
        type: 'generator',
        position: { x: 0, y: 0 },
        data: { id, schemaId: 'chainner:image:load_video', inputData },
    });
    const content = {
        nodes: [
            loadVideo(limited, { 0: 'a.mp4', 1: 1, 2: 25, 3: 'rgb' }),
            loadVideo(unlimited, { 0: 'b.mp4', 1: 0, 2: 25, 3: 'rgb' }),
            // older chains have only the path
            loadVideo(old, { 0: 'c.mp4' }),
            {
                id: number,
                type: 'regularNode',
                position: { x: -400, y: 0 },
                data: { id: number, schemaId: 'chainner:utility:number', inputData: { 0: 25 } },
            },
        ],
        edges: [
            {
                id: 'limit-edge',
                source: number,
                sourceHandle: `${number}-0`,
                target: unlimited,
                targetHandle: `${unlimited}-2`,
                type: 'main',
            },
        ],
        viewport: { x: 0, y: 0, zoom: 1 },
    };
    // the last migration replaces Use limit and Limit (inputs 1 and 2) with Frames, In and Out
    const parsed = SaveFile.parse(
        JSON.stringify({ version: '0.25.0', content, migration: currentMigration - 1 })
    );

    const inputsOf = (id: string) => parsed.nodes.find((n) => n.id === id)?.data.inputData;
    expect(inputsOf(limited)).toEqual({ 0: 'a.mp4', 3: 'rgb', 10: 'range', 11: 0, 12: 24 });
    expect(inputsOf(unlimited)).toEqual({ 0: 'b.mp4', 3: 'rgb', 10: 'all' });
    expect(inputsOf(old)).toEqual({ 0: 'c.mp4', 10: 'all' });
    expect(inputsOf(number)).toEqual({ 0: 25 });
    // Limit is gone, and the edge into it with it
    expect(parsed.edges).toEqual([]);
});

test.each([
    ['not JSON', 'nodes: []', 'it is not JSON'],
    ['no chain', { content: 5 }, 'it holds no chain'],
    ['nodes that are no list', { content: { nodes: 5, edges: [] } }, 'it has no list of nodes'],
    [
        'edges that are no list',
        { content: { nodes: [], edges: {} } },
        'it has no list of connections',
    ],
    [
        'a node without an id',
        { content: { nodes: [{ data: {} }], edges: [] } },
        'its node 1 has no',
    ],
    [
        'a node without a node type',
        { content: { nodes: [{ id: 'a', data: { schemaId: 5 } }], edges: [] } },
        'its node 1 has no id or node type',
    ],
    ['an edge that is null', { content: { nodes: [], edges: [null] } }, 'its connection 1 is'],
    // an older file: the migrations fail on it
    ['an old file without nodes', { content: {}, migration: 20 }, 'damaged: '],
])('A file with %s is a clear error', (_, file, reason) => {
    const text =
        typeof file === 'string'
            ? file
            : JSON.stringify({ version: '0.25.1', migration: currentMigration, ...file });
    expect(() => SaveFile.parse(text)).toThrow(
        'The file is not a chaiNNer chain, or it is damaged: '
    );
    expect(() => SaveFile.parse(text)).toThrow(reason);
});

test('A file without a view opens with the default view', () => {
    const content = { nodes: [], edges: [], viewport: { x: 'a', y: 0, zoom: 1 } };
    const parsed = SaveFile.parse(
        JSON.stringify({ version: '0.25.1', migration: currentMigration, content })
    );
    expect(parsed.viewport).toEqual({ x: 0, y: 0, zoom: 1 });
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

test('Overlapping Save As writes keep request order even when the first path lookup is delayed', async () => {
    const saveData = await SaveFile.read(path.join(dataDir, 'crop.chn'));
    await withTempDir(async (dir) => {
        const file = path.join(dir, 'new-chain.chn');
        let release = () => {};
        const blocked = new Promise<void>((resolve) => {
            release = resolve;
        });
        const realpath = fsPromises.realpath.bind(fsPromises);
        const lookup = vi.spyOn(fsPromises, 'realpath').mockImplementationOnce(async (p) => {
            await blocked;
            return realpath(p);
        });
        const write = vi.spyOn(fsPromises, 'writeFile');
        const first = SaveFile.write(file, saveData, '0.0.0-test');
        const secondData = { ...saveData, viewport: { x: 2, y: 0, zoom: 1 } };
        const second = SaveFile.write(file, secondData, '0.0.0-test');
        try {
            await vi.waitFor(() => expect(lookup).toHaveBeenCalled());
            expect(lookup).toHaveBeenCalledOnce();
            expect(write).not.toHaveBeenCalled();
        } finally {
            release();
            await Promise.all([first, second]);
        }

        expect(savedContent(file)).toMatchObject({ viewport: secondData.viewport });
        expect(fs.readdirSync(dir)).toEqual(['new-chain.chn']);
    });
});

test('A delayed rename blocks the same path but lets another path save', async () => {
    const saveData = await SaveFile.read(path.join(dataDir, 'crop.chn'));
    await withTempDir(async (dir) => {
        const file = path.join(dir, 'chain.chn');
        fs.writeFileSync(file, 'old chain');
        let release = () => {};
        const blocked = new Promise<void>((resolve) => {
            release = resolve;
        });
        const rename = fsPromises.rename.bind(fsPromises);
        const renames = vi.spyOn(fsPromises, 'rename').mockImplementationOnce(async (from, to) => {
            await blocked;
            return rename(from, to);
        });
        const first = SaveFile.write(file, saveData, '0.0.0-test');
        const secondData = { ...saveData, viewport: { x: 2, y: 0, zoom: 1 } };
        try {
            await vi.waitFor(() => expect(renames).toHaveBeenCalledOnce());
            // Use another spelling of the same absolute path, including Windows case folding.
            const samePath = path.join(dir, 'unused', '..', 'chain.chn');
            const second = SaveFile.write(
                process.platform === 'win32' ? samePath.toUpperCase() : samePath,
                secondData,
                '0.0.0-test'
            );
            try {
                await SaveFile.write(path.join(dir, 'other.chn'), saveData, '0.0.0-test');
                expect(renames).toHaveBeenCalledTimes(2);
                expect(fs.readFileSync(file, 'utf-8')).toBe('old chain');
            } finally {
                release();
                await second;
            }
        } finally {
            release();
            await first;
        }

        expect(savedContent(file)).toMatchObject({ viewport: secondData.viewport });
        expect(fs.readdirSync(dir).sort()).toEqual(['chain.chn', 'other.chn']);
    });
});

test('A failed queued save rejects its caller and allows later saves of the path', async () => {
    const saveData = await SaveFile.read(path.join(dataDir, 'crop.chn'));
    await withTempDir(async (dir) => {
        const file = path.join(dir, 'chain.chn');
        vi.spyOn(fsPromises, 'rename').mockRejectedValueOnce(errnoError('EIO'));
        const first = SaveFile.write(file, saveData, '0.0.0-test');
        const secondData = { ...saveData, viewport: { x: 2, y: 0, zoom: 1 } };
        const second = SaveFile.write(file, secondData, '0.0.0-test');

        await Promise.all([expect(first).rejects.toThrow('EIO'), second]);
        expect(savedContent(file)).toMatchObject({ viewport: secondData.viewport });
        await SaveFile.write(file, saveData, '0.0.0-test');
        expect(savedContent(file)).toEqual(await expectedContent());
        expect(fs.readdirSync(dir)).toEqual(['chain.chn']);
    });
});

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

test.skipIf(!canSymlink).each(['existing file link', 'new file through a directory link'])(
    'Delayed alias resolution reserves writes in arrival order for the same target: %s',
    async (kind) => {
        const saveData = await SaveFile.read(path.join(dataDir, 'crop.chn'));
        await withTempDir(async (dir) => {
            const target = path.join(dir, 'chain.chn');
            const existingFile = kind === 'existing file link';
            const link = path.join(dir, existingFile ? 'link.chn' : 'alias');
            if (existingFile) {
                fs.writeFileSync(target, 'old chain');
                fs.symlinkSync(target, link, 'file');
            } else {
                fs.symlinkSync(dir, link, process.platform === 'win32' ? 'junction' : 'dir');
            }
            let releaseLookup = () => {};
            let releaseRename = () => {};
            const blockedLookup = new Promise<void>((resolve) => {
                releaseLookup = resolve;
            });
            const blockedRename = new Promise<void>((resolve) => {
                releaseRename = resolve;
            });
            const realpath = fsPromises.realpath.bind(fsPromises);
            const lookup = vi.spyOn(fsPromises, 'realpath').mockImplementationOnce(async (p) => {
                await blockedLookup;
                return realpath(p);
            });
            const rename = fsPromises.rename.bind(fsPromises);
            const renames = vi
                .spyOn(fsPromises, 'rename')
                .mockImplementationOnce(async (from, to) => {
                    await blockedRename;
                    return rename(from, to);
                });
            const first = SaveFile.write(
                existingFile ? link : path.join(link, 'chain.chn'),
                saveData,
                '0.0.0-test'
            );
            const secondData = { ...saveData, viewport: { x: 2, y: 0, zoom: 1 } };
            const second = SaveFile.write(target, secondData, '0.0.0-test');
            try {
                await vi.waitFor(() => expect(lookup).toHaveBeenCalled());
                expect(lookup).toHaveBeenCalledOnce();
                releaseLookup();
                await vi.waitFor(() => {
                    expect(lookup).toHaveBeenCalledTimes(existingFile ? 2 : 4);
                    expect(renames).toHaveBeenCalled();
                });
                expect(renames).toHaveBeenCalledOnce();
                if (existingFile) expect(fs.readFileSync(target, 'utf-8')).toBe('old chain');
                else expect(fs.existsSync(target)).toBe(false);
            } finally {
                releaseLookup();
                releaseRename();
                await Promise.all([first, second]);
            }
            expect(savedContent(target)).toMatchObject({ viewport: secondData.viewport });
            expect(fs.lstatSync(link).isSymbolicLink()).toBe(true);
            expect(fs.readdirSync(dir).sort()).toEqual([path.basename(link), 'chain.chn'].sort());
        });
    }
);

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
    test(`Write retries a rename that fails once with ${code}`, async () => {
        const saveData = await SaveFile.read(path.join(dataDir, 'crop.chn'));
        await withTempDir(async (dir) => {
            const file = path.join(dir, 'chain.chn');
            fs.writeFileSync(file, 'old chain', 'utf-8');
            const rename = vi.spyOn(fsPromises, 'rename').mockRejectedValueOnce(errnoError(code));

            await SaveFile.write(file, saveData, '0.0.0-test');

            expect(rename).toHaveBeenCalledTimes(2);
            expect(savedContent(file)).toEqual(await expectedContent());
            expect(fs.readdirSync(dir)).toEqual(['chain.chn']);
        });
    });

    test(`Write falls back to writing in place when the rename keeps failing with ${code}`, async () => {
        const saveData = await SaveFile.read(path.join(dataDir, 'crop.chn'));
        await withTempDir(async (dir) => {
            const file = path.join(dir, 'chain.chn');
            fs.writeFileSync(file, 'old chain', 'utf-8');
            vi.spyOn(fsPromises, 'rename').mockRejectedValue(errnoError(code));

            await SaveFile.write(file, saveData, '0.0.0-test');

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
