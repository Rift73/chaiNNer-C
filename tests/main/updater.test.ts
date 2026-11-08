import { execFileSync, spawnSync } from 'child_process';
import { createHash } from 'crypto';
import { existsSync } from 'fs';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'fs/promises';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { crc32 } from 'zlib';
import {
    LINUX_INSTALLER,
    WINDOWS_INSTALLER,
    WINDOWS_PARAMETERS,
    expectedSha256,
    findReleaseAsset,
    findReleaseRoot,
    installerArguments,
    stageUpdate,
    takeUpdateResult,
} from '../../src/main/updater';

const { paths, fetchRelease, download } = vi.hoisted(() => ({
    paths: { downloads: '', userData: '' },
    fetchRelease: vi.fn(),
    download: vi.fn(),
}));

vi.mock('electron/main', () => ({
    app: {
        isPackaged: true,
        getPath: (name: 'downloads' | 'userData' | 'exe') =>
            name === 'exe' ? process.execPath : paths[name],
        whenReady: () => Promise.resolve(),
    },
    net: { fetch: fetchRelease },
}));
vi.mock('../../src/main/python/integratedPython', () => ({ download }));

const asset = (name: string, digest?: string) => ({
    name,
    size: 1,
    digest,
    browser_download_url: `https://example.invalid/${name}`,
});
const release = {
    tag_name: 'v0.3.11',
    name: 'chaiNNer-C v0.3.11',
    assets: [asset('chaiNNer-C-v0.3.11-win-x64.zip'), asset('chaiNNer-C-v0.3.11-linux-x64.zip')],
};
const executable = path.basename(process.execPath);

let scratch: string;
beforeEach(async () => {
    scratch = await mkdtemp(path.join(os.tmpdir(), 'updater-test-'));
    paths.downloads = path.join(scratch, 'downloads');
    paths.userData = path.join(scratch, 'user-data');
    await mkdir(paths.downloads);
    await mkdir(paths.userData);
});
afterEach(async () => {
    vi.clearAllMocks();
    await rm(scratch, { recursive: true, force: true });
});

test('each platform takes its own zip, and no other platform has one', () => {
    expect(findReleaseAsset(release, 'win32', 'x64').name).toBe('chaiNNer-C-v0.3.11-win-x64.zip');
    expect(findReleaseAsset(release, 'linux', 'x64').name).toBe('chaiNNer-C-v0.3.11-linux-x64.zip');
    expect(() => findReleaseAsset(release, 'darwin', 'x64')).toThrow('no zip for darwin x64');
    expect(() => findReleaseAsset(release, 'win32', 'arm64')).toThrow('no zip for win32 arm64');
});

test("an asset without GitHub's SHA-256 is not installed", () => {
    const hex = 'ab'.repeat(32);
    expect(expectedSha256(asset('a.zip', `sha256:${hex.toUpperCase()}`))).toBe(hex);
    expect(() => expectedSha256(asset('a.zip'))).toThrow('no SHA-256');
    expect(() => expectedSha256(asset('a.zip', 'md5:00'))).toThrow('no SHA-256');
});

const writeTree = async (root: string, files: Record<string, string>) => {
    await Promise.all(
        Object.entries(files).map(async ([name, text]) => {
            await mkdir(path.dirname(path.join(root, name)), { recursive: true });
            await writeFile(path.join(root, name), text);
        })
    );
};

test("a release's app is at the zip's root or in its one top folder", async () => {
    const windows = path.join(scratch, 'windows');
    await writeTree(windows, { [executable]: 'exe', 'resources/app/main.js': 'app' });
    expect(await findReleaseRoot(windows, executable)).toBe(windows);

    const linux = path.join(scratch, 'linux');
    await writeTree(linux, {
        [`chaiNNer-linux-x64/${executable}`]: 'exe',
        'chaiNNer-linux-x64/resources/app/main.js': 'app',
    });
    expect(await findReleaseRoot(linux, executable)).toBe(path.join(linux, 'chaiNNer-linux-x64'));

    const other = path.join(scratch, 'other');
    await writeTree(other, { 'readme.txt': 'not an app' });
    await expect(findReleaseRoot(other, executable)).rejects.toThrow('not a chaiNNer-C build');
});

/** A zip of these files, stored without compression. */
const storedZip = (files: Record<string, string>): Buffer => {
    const locals: Buffer[] = [];
    const centrals: Buffer[] = [];
    let offset = 0;
    for (const [name, text] of Object.entries(files)) {
        const nameBytes = Buffer.from(name);
        const data = Buffer.from(text);
        const crc = crc32(data);
        const local = Buffer.alloc(30);
        local.writeUInt32LE(0x04034b50, 0);
        local.writeUInt16LE(20, 4);
        local.writeUInt32LE(crc, 14);
        local.writeUInt32LE(data.length, 18);
        local.writeUInt32LE(data.length, 22);
        local.writeUInt16LE(nameBytes.length, 26);
        const central = Buffer.alloc(46);
        central.writeUInt32LE(0x02014b50, 0);
        central.writeUInt16LE(20, 4);
        central.writeUInt16LE(20, 6);
        central.writeUInt32LE(crc, 16);
        central.writeUInt32LE(data.length, 20);
        central.writeUInt32LE(data.length, 24);
        central.writeUInt16LE(nameBytes.length, 28);
        central.writeUInt32LE(offset, 42);
        locals.push(local, nameBytes, data);
        centrals.push(central, nameBytes);
        offset += local.length + nameBytes.length + data.length;
    }
    const directory = Buffer.concat(centrals);
    const end = Buffer.alloc(22);
    end.writeUInt32LE(0x06054b50, 0);
    end.writeUInt16LE(Object.keys(files).length, 8);
    end.writeUInt16LE(Object.keys(files).length, 10);
    end.writeUInt32LE(directory.length, 12);
    end.writeUInt32LE(offset, 16);
    return Buffer.concat([...locals, directory, end]);
};

const releaseZip = () =>
    storedZip({
        [executable]: 'new exe',
        'resources/app/main.js': 'new app',
        'resources/src/run.py': 'new backend',
    });

const answerWith = (zip: Buffer, digest: string) => {
    const name =
        process.platform === 'win32'
            ? 'chaiNNer-C-v0.3.11-win-x64.zip'
            : 'chaiNNer-C-v0.3.11-linux-x64.zip';
    fetchRelease.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ ...release, assets: [asset(name, digest)] }),
    });
    download.mockImplementation(async (_url: string, file: string) => {
        await writeFile(file, zip);
    });
    return name;
};

test('a checked download is unpacked, and the zip stays in Downloads until it is installed', async () => {
    const zip = releaseZip();
    const name = answerWith(zip, `sha256:${createHash('sha256').update(zip).digest('hex')}`);
    const progress: number[] = [];
    const update = await stageUpdate((p) => progress.push(p));
    try {
        expect(update.release).toBe('chaiNNer-C v0.3.11');
        expect(update.zip).toBe(path.join(paths.downloads, name));
        expect(existsSync(update.zip)).toBe(true);
        expect(existsSync(`${update.zip}.part`)).toBe(false);
        expect(await readFile(path.join(update.root, 'resources/app/main.js'), 'utf-8')).toBe(
            'new app'
        );
        expect(progress.at(-1)).toBe(100);
    } finally {
        await rm(update.unpacked, { recursive: true, force: true });
    }
});

test('a download that does not match GitHub is removed and not unpacked', async () => {
    const name = answerWith(releaseZip(), `sha256:${'0'.repeat(64)}`);
    await expect(stageUpdate(() => {})).rejects.toThrow('does not match');
    expect(existsSync(path.join(paths.downloads, name))).toBe(false);
    expect(existsSync(path.join(paths.downloads, `${name}.part`))).toBe(false);
});

test("the last update's result is read once", async () => {
    const file = path.join(paths.userData, 'update-result.json');
    await writeFile(file, '﻿{"ok":false,"release":"chaiNNer-C v0.3.11","message":"m","zip":"z"}');
    expect(await takeUpdateResult()).toEqual({
        ok: false,
        release: 'chaiNNer-C v0.3.11',
        message: 'm',
        zip: 'z',
    });
    expect(await takeUpdateResult()).toBeUndefined();
});

/** A process id that has ended. */
const endedProcessId = (): number => {
    const child = spawnSync(process.execPath, ['-e', 'process.stdout.write(String(process.pid))'], {
        encoding: 'utf-8',
        env: { ...process.env, ELECTRON_RUN_AS_NODE: '1' },
    });
    return Number(child.stdout);
};

test.runIf(process.platform === 'win32')(
    'the Windows installer copies the release over the install and leaves the rest',
    async () => {
        const target = path.join(scratch, 'install');
        const unpacked = path.join(scratch, 'unpacked');
        const source = path.join(unpacked, 'release');
        const zip = path.join(paths.downloads, 'release.zip');
        const result = path.join(paths.userData, 'update-result.json');
        await writeTree(target, {
            'chaiNNer-C.exe': 'old exe',
            'resources/app/main.js': 'old app',
            'resources/app/dropped.js': 'a file the release no longer has',
            'resources/src/run.py': 'old backend',
            'resources/src/extra.py': 'outside the app bundle: stays',
            'python/python.exe': 'integrated python: stays',
            'settings.json': 'settings: stay',
        });
        await writeTree(source, {
            'chaiNNer-C.exe': 'new exe',
            'resources/app/main.js': 'new app',
            'resources/src/run.py': 'new backend',
        });
        await writeFile(zip, 'zip');
        const script = path.join(scratch, 'install.ps1');
        await writeFile(script, WINDOWS_INSTALLER, 'utf-8');

        const update = { release: 'chaiNNer-C v0.3.11', zip, unpacked, root: source };
        const args = installerArguments(update, target, result, '');
        args[0] = String(endedProcessId());
        execFileSync(
            'powershell.exe',
            [
                '-NoProfile',
                '-NonInteractive',
                '-ExecutionPolicy',
                'Bypass',
                '-File',
                script,
                ...WINDOWS_PARAMETERS.flatMap((name, i) => [`-${name}`, args[i]]),
            ],
            { stdio: 'ignore', timeout: 120_000 }
        );

        const read = (name: string) => readFile(path.join(target, name), 'utf-8');
        expect(await read('chaiNNer-C.exe')).toBe('new exe');
        expect(await read('resources/app/main.js')).toBe('new app');
        expect(existsSync(path.join(target, 'resources/app/dropped.js'))).toBe(false);
        expect(await read('resources/src/run.py')).toBe('new backend');
        expect(await read('resources/src/extra.py')).toBe('outside the app bundle: stays');
        expect(await read('python/python.exe')).toBe('integrated python: stays');
        expect(await read('settings.json')).toBe('settings: stay');
        expect(existsSync(zip)).toBe(false);
        expect(existsSync(unpacked)).toBe(false);
        expect(existsSync(script)).toBe(false);
        expect(JSON.parse(await readFile(result, 'utf-8'))).toMatchObject({
            ok: true,
            release: 'chaiNNer-C v0.3.11',
        });
    },
    150_000
);

test.runIf(process.platform === 'linux')(
    'the Linux installer copies the release over the install and leaves the rest',
    async () => {
        const target = path.join(scratch, 'install');
        const unpacked = path.join(scratch, 'unpacked');
        const source = path.join(unpacked, 'chaiNNer-linux-x64');
        const zip = path.join(paths.downloads, 'release.zip');
        const result = path.join(paths.userData, 'update-result.json');
        await writeTree(target, {
            chaiNNer: 'old exe',
            'resources/app/main.js': 'old app',
            'resources/app/dropped.js': 'a file the release no longer has',
            'resources/src/run.py': 'old backend',
            'python/bin/python3': 'integrated python: stays',
        });
        await writeTree(source, {
            chaiNNer: 'new exe',
            'resources/app/main.js': 'new app',
            'resources/src/run.py': 'new backend',
        });
        await writeFile(zip, 'zip');
        const script = path.join(scratch, 'install.sh');
        await writeFile(script, LINUX_INSTALLER, { encoding: 'utf-8', mode: 0o755 });

        const update = { release: 'chaiNNer-C "v0.3.11"', zip, unpacked, root: source };
        const args = installerArguments(update, target, result, '');
        args[0] = String(endedProcessId());
        execFileSync('/bin/sh', [script, ...args], { stdio: 'ignore', timeout: 120_000 });

        const read = (name: string) => readFile(path.join(target, name), 'utf-8');
        expect(await read('chaiNNer')).toBe('new exe');
        expect(await read('resources/app/main.js')).toBe('new app');
        expect(existsSync(path.join(target, 'resources/app/dropped.js'))).toBe(false);
        expect(await read('resources/src/run.py')).toBe('new backend');
        expect(await read('python/bin/python3')).toBe('integrated python: stays');
        expect(existsSync(zip)).toBe(false);
        expect(existsSync(unpacked)).toBe(false);
        expect(existsSync(script)).toBe(false);
        expect(JSON.parse(await readFile(result, 'utf-8'))).toMatchObject({
            ok: true,
            release: 'chaiNNer-C "v0.3.11"',
        });
    },
    150_000
);
