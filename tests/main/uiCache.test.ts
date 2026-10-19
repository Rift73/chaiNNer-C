import { mkdir, mkdtemp, readdir, rm, writeFile } from 'fs/promises';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { BackendData } from '../../src/common/Backend';
import { CategoryId, PythonInfo, Version } from '../../src/common/common-types';

vi.mock('electron/main', () => ({
    app: {
        getVersion: () => '0.24.1',
        getGPUInfo: () =>
            Promise.resolve({
                gpuDevice: [{ vendorId: 4318, deviceId: 8708, driverVersion: '32.0.15.8180' }],
            }),
    },
}));
vi.mock('../../src/main/buildInfo', () => ({
    buildInfo: { tree: 'a1b2c3', date: '2026-10-19T09:00:00+08:00', dirty: false, packaged: true },
}));

let directory: string;
let python: PythonInfo;
let sitePackages: string;
let cacheFile: string;

beforeEach(async () => {
    directory = await mkdtemp(path.join(os.tmpdir(), 'chaiNNer-ui-cache-test-'));
    const isWindows = process.platform === 'win32';
    python = {
        python: path.join(directory, isWindows ? 'python/python.exe' : 'python/bin/python3.14'),
        version: '3.14.8' as Version,
    };
    sitePackages = isWindows
        ? path.join(directory, 'python/Lib/site-packages')
        : path.join(directory, 'python/lib/python3.14/site-packages');
    await mkdir(path.join(sitePackages, 'numpy-2.5.3.dist-info'), { recursive: true });
    const { getUiCacheFile } = await import('../../src/main/uiCache');
    cacheFile = getUiCacheFile(directory);
});

afterEach(async () => {
    await rm(directory, { recursive: true, force: true });
});

const data: BackendData = [
    {
        nodes: [],
        categories: [],
        categoriesMissingNodes: ['pytorch' as CategoryId],
    },
    [],
];

const newCache = async () => {
    const { UiCache } = await import('../../src/main/uiCache');
    return new UiCache(cacheFile, () => Promise.resolve(python), path.join(directory, 'backend'));
};

test('the node list written at one start is read at the next', async () => {
    const first = await newCache();
    await expect(first.read()).resolves.toBeUndefined();
    await first.write(data);
    // the write leaves no temporary file behind
    expect(await readdir(path.dirname(cacheFile))).toEqual(['backend-ui.json']);

    await expect((await newCache()).read()).resolves.toEqual(data);
});

test('a cache of another set of installed distributions is not used', async () => {
    await (await newCache()).write(data);

    // e.g. the dependency manager installed PyTorch since
    await mkdir(path.join(sitePackages, 'torch-2.14.1.dist-info'));
    const next = await newCache();
    await expect(next.read()).resolves.toBeUndefined();

    // the live list then rewrites the cache with the new key
    await next.write(data);
    await expect((await newCache()).read()).resolves.toEqual(data);
});

// The folders Electron's Chromium keeps in userData, which is the app's root folder. Chromium
// clears and replaces them on its own, e.g. its HTTP cache `Cache` after an unclean exit.
const CHROMIUM_FOLDERS = [
    'blob_storage',
    'Cache',
    'Code Cache',
    'Crashpad',
    'databases',
    'DawnCache',
    'DawnGraphiteCache',
    'DawnWebGPUCache',
    'Dictionaries',
    'GPUCache',
    'GrShaderCache',
    'IndexedDB',
    'Local Storage',
    'Network',
    'Partitions',
    'Service Worker',
    'Session Storage',
    'ShaderCache',
    'Shared Dictionary',
    'SharedStorage',
    'VideoDecodeStats',
    'WebStorage',
];

test('the cache is kept outside every folder Chromium manages', async () => {
    const { UI_CACHE_FOLDER } = await import('../../src/main/uiCache');
    // Windows' file names ignore case
    const chromium = CHROMIUM_FOLDERS.map((folder) => folder.toLowerCase());
    expect(chromium).not.toContain(UI_CACHE_FOLDER.toLowerCase());
    expect(path.relative(directory, path.dirname(cacheFile))).toBe(UI_CACHE_FOLDER);
});

test('an unreadable cache file is no cache', async () => {
    await mkdir(path.dirname(cacheFile), { recursive: true });
    await writeFile(cacheFile, '{"key": "abc", "nodes": ');
    await expect((await newCache()).read()).resolves.toBeUndefined();
});
