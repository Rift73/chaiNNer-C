import { execFile, spawnSync } from 'child_process';
import { createHash } from 'crypto';
import { existsSync } from 'fs';
import { appendFile, chmod, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'fs/promises';
import os from 'os';
import path from 'path';
import { afterAll, afterEach, beforeAll, beforeEach, expect, test, vi } from 'vitest';
import { crc32 } from 'zlib';
import { delay } from '../../src/common/util';
import { UI_CACHE_FOLDER } from '../../src/main/uiCache';
import {
    INSTALLED_LIST,
    INSTALL_DATA_NAMES,
    JOURNAL,
    LINUX_INSTALLER,
    LOCK,
    STAGED_LIST,
    UPDATE_DONE,
    UPDATE_NEW,
    UPDATE_OLD,
    WINDOWS_INSTALLER,
    checkUnfinishedUpdate,
    discardUpdate,
    entryState,
    expectedSha256,
    findReleaseAsset,
    findReleaseRoot,
    journalText,
    parseJournal,
    parseReleaseFiles,
    parseStagedList,
    planSwap,
    processStartTime,
    removeUnpackedReleases,
    showUpdateAtStart,
    stageUpdate,
    stagedListText,
    startInstaller,
    takeUpdateResult,
} from '../../src/main/updater';

const { paths, fetchRelease, download, spawnInstaller, fakeChild, showMessageBox } = vi.hoisted(
    () => {
        /** A spawned process that reports `event` (its start, or a failure to start) later, as Node. */
        const child = (event: 'spawn' | 'error') => {
            const listeners = new Map<string, (value?: unknown) => void>();
            setTimeout(() =>
                listeners.get(event)?.(
                    event === 'error' ? new Error('spawn powershell.exe ENOENT') : undefined
                )
            );
            const spawned = {
                once: (name: string, listener: (value?: unknown) => void) => {
                    listeners.set(name, listener);
                    return spawned;
                },
                unref: () => {},
            };
            return spawned;
        };
        return {
            // the install the app runs from: platform.ts reads it once, at its import
            paths: {
                downloads: '',
                userData: '',
                // no character of the install's path may end a string or start a command in the
                // installers (a double quote cannot be in a Windows path)
                install: `${
                    process.env.TEMP ?? process.env.TMPDIR ?? '/tmp'
                }/updater-test-install-${process.pid} [v0.3] it's $HOME${'`'}x; a & (b) é${
                    process.platform === 'win32' ? '' : ' "q"'
                }`,
            },
            fetchRelease: vi.fn(),
            download: vi.fn(),
            spawnInstaller: vi.fn((): unknown => child('spawn')),
            fakeChild: child,
            showMessageBox: vi.fn(),
        };
    }
);

vi.mock('child_process', async (importOriginal) => ({
    ...(await importOriginal<typeof import('child_process')>()),
    spawn: spawnInstaller,
}));

vi.mock('electron/main', () => ({
    app: {
        isPackaged: true,
        getPath: (name: 'downloads' | 'userData' | 'exe') =>
            name === 'exe' ? `${paths.install}/chaiNNer-C.exe` : paths[name],
        whenReady: () => Promise.resolve(),
    },
    dialog: { showErrorBox: vi.fn(), showMessageBox },
    net: { fetch: fetchRelease },
}));
vi.mock('../../src/main/python/integratedPython', () => ({ download }));

const windows = process.platform === 'win32';
const windowsOnly = test.runIf(windows);
const linuxOnly = test.runIf(process.platform === 'linux');

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
/** Where the program the installer starts again writes that it ran: beside the install. */
const launchedFile = () =>
    path.join(path.dirname(paths.install), `updater-test-launched-${process.pid}.txt`);

/** Runs `run` for each item, one after another. */
const inTurn = <T>(items: readonly T[], run: (item: T) => Promise<void>): Promise<void> =>
    items.reduce(async (previous, item) => {
        await previous;
        await run(item);
    }, Promise.resolve());
/** 1 to `n`. */
const upTo = (n: number) => Array.from({ length: n }, (_, i) => i + 1);

let scratch: string;
beforeEach(async () => {
    scratch = await mkdtemp(path.join(os.tmpdir(), 'updater-test-'));
    paths.downloads = path.join(scratch, 'downloads');
    paths.userData = path.join(scratch, 'user-data');
    await mkdir(paths.downloads);
    await mkdir(paths.userData);
    await rm(paths.install, { recursive: true, force: true, maxRetries: 50, retryDelay: 100 });
    await mkdir(paths.install, { recursive: true });
});
afterEach(async () => {
    vi.clearAllMocks();
    // retried: the relaunched program of the installer test may still run in its folder
    await rm(scratch, { recursive: true, force: true, maxRetries: 50, retryDelay: 100 });
    await rm(paths.install, { recursive: true, force: true, maxRetries: 50, retryDelay: 100 });
    await rm(launchedFile(), { force: true });
});

/**
 * Records every console host window Windows opens (OpenConsole.exe, WindowsTerminal.exe), which a
 * console program started without a hidden console gets: the tests start none.
 */
const WATCH_CONSOLES = String.raw`param([string]$Out)
$query = "SELECT * FROM __InstanceCreationEvent WITHIN 0.1 WHERE TargetInstance ISA 'Win32_Process' AND (TargetInstance.Name = 'OpenConsole.exe' OR TargetInstance.Name = 'WindowsTerminal.exe')"
Register-CimIndicationEvent -Query $query -SourceIdentifier consoles
[IO.File]::WriteAllText($Out, '')
while ($true) {
    $event = Wait-Event -SourceIdentifier consoles -Timeout 1
    if ($event) {
        $process = $event.SourceEventArgs.NewEvent.TargetInstance
        [IO.File]::AppendAllText($Out, "$($process.Name) $($process.ProcessId) $($process.CommandLine)" + [char]10)
        Remove-Event -EventIdentifier $event.EventIdentifier
    }
}
`;
let consoleWatch: { watcher: ReturnType<typeof execFile>; out: string; folder: string } | undefined;
beforeAll(async () => {
    if (!windows) {
        return;
    }
    const folder = await mkdtemp(path.join(os.tmpdir(), 'updater-test-consoles-'));
    const script = path.join(folder, 'watch.ps1');
    const out = path.join(folder, 'consoles.txt');
    await writeFile(script, WATCH_CONSOLES, 'utf-8');
    const watcher = execFile(
        'powershell.exe',
        [
            '-NoProfile',
            '-NonInteractive',
            '-ExecutionPolicy',
            'Bypass',
            '-File',
            script,
            '-Out',
            out,
        ],
        { windowsHide: true }
    );
    consoleWatch = { watcher, out, folder };
    await vi.waitFor(() => expect(existsSync(out)).toBe(true), { timeout: 30_000 });
}, 60_000);
afterAll(async () => {
    if (!consoleWatch) {
        return;
    }
    // the last window an event could still report
    await delay(1500);
    spawnSync('taskkill', ['/PID', String(consoleWatch.watcher.pid), '/T', '/F'], {
        stdio: 'ignore',
        windowsHide: true,
    });
    const windowsOpened = await readFile(consoleWatch.out, 'utf-8');
    await rm(consoleWatch.folder, {
        recursive: true,
        force: true,
        maxRetries: 50,
        retryDelay: 100,
    });
    expect(windowsOpened, 'a test opened a console window on the desktop').toBe('');
}, 60_000);

/** Writes these files below `root`; a name ending in `/` is an empty folder. */
const writeTree = async (root: string, files: Record<string, string>) => {
    await inTurn(Object.entries(files), async ([name, text]) => {
        if (name.endsWith('/')) {
            await mkdir(path.join(root, name), { recursive: true });
            return;
        }
        await mkdir(path.dirname(path.join(root, name)), { recursive: true });
        await writeFile(path.join(root, name), text);
    });
};

/**
 * Every file below `root` (relative, with `/`) and its text, and every empty folder (ending in
 * `/`); not the lock, which an installer may hold open (the tests check it on their own).
 */
const treeOf = async (root: string): Promise<Record<string, string>> => {
    const tree: Record<string, string> = {};
    const walk = async (folder: string, prefix: string) => {
        const entries = await readdir(folder, { withFileTypes: true });
        if (entries.length === 0 && prefix) {
            tree[prefix] = '';
        }
        await inTurn(entries, async (entry) => {
            if (!prefix && entry.name === LOCK) {
                return;
            }
            const full = path.join(folder, entry.name);
            if (entry.isDirectory()) {
                await walk(full, `${prefix}${entry.name}/`);
            } else {
                tree[`${prefix}${entry.name}`] = await readFile(full, 'utf-8');
            }
        });
    };
    await walk(root, '');
    return tree;
};

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
        // a name ending in / is a folder: the MS-DOS directory attribute
        central.writeUInt32LE(name.endsWith('/') ? 0x10 : 0, 38);
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

const answerWith = (
    zip: Buffer,
    digest = `sha256:${createHash('sha256').update(zip).digest('hex')}`
) => {
    const name = windows ? 'chaiNNer-C-v0.3.11-win-x64.zip' : 'chaiNNer-C-v0.3.11-linux-x64.zip';
    fetchRelease.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ ...release, assets: [asset(name, digest)] }),
    });
    download.mockImplementation(async (_url: string, file: string) => {
        await writeFile(file, zip);
    });
    return name;
};

const NODES = 'resources/src/packages/chaiNNer_standard';
const listOf = (names: string[]) => `${names.join('\n')}\n`;
/** Two releases to update between, and the renames a whole update between them makes. */
interface ReleasePair {
    old: Record<string, string>;
    new: Record<string, string>;
    renames: number;
}
const sameLayoutList = listOf([
    executable,
    'ffmpeg.dll',
    'icudtl.dat',
    'locales',
    'release-files.txt',
    'resources',
    'swiftshader',
]);
/**
 * Two releases as real ones mostly are: the same layout, so the same release-files.txt; entries
 * the same in both (a file, a folder, a folder with no files); the executable, a DLL and the
 * contents of resources (the same set of files) changed.
 */
const SAME_LAYOUT: ReleasePair = {
    old: {
        [executable]: 'old exe',
        'ffmpeg.dll': 'old ffmpeg',
        'icudtl.dat': 'the same in both releases',
        'locales/en-US.pak': 'a locale the same in both',
        'swiftshader/': '',
        'release-files.txt': sameLayoutList,
        'resources/app/main.js': 'old app',
        'resources/src/run.py': 'old backend',
        [`${NODES}/node.py`]: 'old node',
    },
    new: {
        [executable]: 'new exe',
        'ffmpeg.dll': 'new ffmpeg',
        'icudtl.dat': 'the same in both releases',
        'locales/en-US.pak': 'a locale the same in both',
        'swiftshader/': '',
        'release-files.txt': sameLayoutList,
        // the zip's marker: an install that keeps its data elsewhere stays so
        portable: '',
        'resources/app/main.js': 'new app',
        'resources/src/run.py': 'new backend',
        [`${NODES}/node.py`]: 'new node',
    },
    // seven entries both have, two renames each
    renames: 14,
};
/** Two releases that share no entry's bytes: one entry dropped, one added, files dropped in resources. */
const ALL_DIFFERENT: ReleasePair = {
    old: {
        [executable]: 'old exe',
        'ffmpeg.dll': 'old ffmpeg',
        'old-only.dll': 'a file the new release dropped',
        'release-files.txt': listOf([
            executable,
            'ffmpeg.dll',
            'old-only.dll',
            'resources',
            'release-files.txt',
        ]),
        'resources/app/main.js': 'old app',
        'resources/app/dropped.js': 'a file the new bundle dropped',
        'resources/src/run.py': 'old backend',
        [`${NODES}/dropped_node.py`]: 'a node the new release no longer has',
    },
    new: {
        [executable]: 'new exe',
        'ffmpeg.dll': 'new ffmpeg',
        'new-only.dll': 'a file only the new release has',
        'release-files.txt': listOf([
            executable,
            'ffmpeg.dll',
            'new-only.dll',
            'resources',
            'release-files.txt',
        ]),
        portable: '',
        'resources/app/main.js': 'new app',
        'resources/src/run.py': 'new backend',
        [`${NODES}/node.py`]: 'new node',
    },
    // four entries both have, one dropped, one added
    renames: 10,
};
/** The pair the tests update between: SAME_LAYOUT unless a test chooses another. */
let pair = SAME_LAYOUT;
let oldRelease = pair.old;
let newRelease = pair.new;
const usePair = (chosen: ReleasePair) => {
    pair = chosen;
    oldRelease = chosen.old;
    newRelease = chosen.new;
};
beforeEach(() => usePair(SAME_LAYOUT));
/** The install's own data, which no update touches, and a program the installer starts again. */
const userData = (): Record<string, string> => ({
    'python/python.exe': 'integrated python',
    'settings.json': 'settings',
    'backend-storage/cache.bin': 'storage',
    'logs/main.log': 'log',
    ...(windows
        ? { 'launch.cmd': `@echo launched> "%~dp0..\\${path.basename(launchedFile())}"\r\n` }
        : {
              'launch.sh': `#!/bin/sh\necho launched > "$(dirname "$0")/../${path.basename(
                  launchedFile()
              )}"\n`,
          }),
});
const before = () => ({ ...userData(), ...oldRelease });
const after = () => {
    const { portable, ...files } = newRelease;
    return { ...userData(), ...files };
};
const launchName = windows ? 'launch.cmd' : 'launch.sh';
const journalFile = () => path.join(paths.install, JOURNAL);
const installerScript = () =>
    path.join(paths.install, windows ? 'finish-update.ps1' : 'finish-update.sh');
const finishFiles = windows
    ? ['finish-update.ps1', 'finish-update.cmd', 'update-not-finished.txt', JOURNAL]
    : ['finish-update.sh', 'update-not-finished.txt', JOURNAL];

/** The start's question when the install was changed during an update. */
const CHANGED = 'The install was changed during an update';

/** The old release and the install's data, as an install holds them. */
const installOld = async () => {
    await writeTree(paths.install, before());
    if (!windows) {
        await chmod(path.join(paths.install, launchName), 0o755);
    }
};

/** Stages `files` as the latest release, into an install with the old release. */
const stage = async (files: Record<string, string> = newRelease) => {
    await installOld();
    answerWith(storedZip(files));
    return stageUpdate(() => {});
};

/** The app's part: the old release installed, the new one staged, the installer's files written. */
const prepare = async (files: Record<string, string> = newRelease) => {
    await rm(paths.install, { recursive: true, force: true, maxRetries: 50, retryDelay: 100 });
    await rm(launchedFile(), { force: true });
    const update = await stage(files);
    await startInstaller(update);
    // the program started again: the test's script, not the test's own runtime
    const text = await readFile(journalFile(), 'utf-8');
    await writeFile(journalFile(), text.replace(/^launch\t.*$/m, `launch\t${launchName}`));
    return update;
};

/** Runs a PowerShell script; its exit code and output. */
const powershell = (args: string[]) =>
    spawnSync(
        'powershell.exe',
        ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', ...args],
        { encoding: 'utf-8', timeout: 120_000, windowsHide: true }
    );

/** A process id that has ended. */
const endedProcessId = (): number => {
    const child = spawnSync(process.execPath, ['-e', 'process.stdout.write(String(process.pid))'], {
        encoding: 'utf-8',
        windowsHide: true,
        env: { ...process.env, ELECTRON_RUN_AS_NODE: '1' },
    });
    return Number(child.stdout);
};

/** Runs the install's installer as the app starts it, waiting for a process that has ended. */
const runInstaller = () =>
    windows
        ? powershell([installerScript(), '-ProcessId', String(endedProcessId())])
        : spawnSync('/bin/sh', [installerScript(), String(endedProcessId())], {
              encoding: 'utf-8',
              timeout: 120_000,
          });

const readResult = async () =>
    JSON.parse(await readFile(path.join(paths.userData, 'update-result.json'), 'utf-8')) as {
        ok: boolean;
        message: string;
        release: string;
    };

/** The install after a whole update: the new release and the install's own data, nothing else. */
const expectUpdated = async (zip: string) => {
    expect(await treeOf(paths.install)).toEqual(after());
    expect(existsSync(path.join(paths.install, LOCK))).toBe(false);
    expect(existsSync(zip)).toBe(false);
    expect(await readResult()).toMatchObject({ ok: true, release: 'chaiNNer-C v0.3.11' });
    await vi.waitFor(() => expect(existsSync(launchedFile())).toBe(true), { timeout: 10_000 });
};

/** Ends the program a test started and all it started. */
const endTree = (pid: number | undefined) => {
    spawnSync('taskkill', ['/PID', String(pid), '/T', '/F'], {
        stdio: 'ignore',
        windowsHide: true,
    });
};

/** A terminal open in resources\src: resources cannot be renamed while it runs. */
const holdResources = async () => {
    const holder = execFile('cmd.exe', ['/d', '/c', 'ping -n 120 127.0.0.1 > nul'], {
        cwd: path.join(paths.install, 'resources', 'src'),
        windowsHide: true,
    });
    await delay(500);
    return holder;
};

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

test("a release's app is at the zip's root or in its one top folder", async () => {
    const root = path.join(scratch, 'windows');
    await writeTree(root, { [executable]: 'exe', 'resources/app/main.js': 'app' });
    expect(await findReleaseRoot(root, executable)).toBe(root);

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

const plan = (overrides: Partial<Parameters<typeof planSwap>[0]> = {}) =>
    planSwap({
        installEntries: [
            'app.exe',
            'ffmpeg.dll',
            'old-only.dll',
            'resources',
            'release-files.txt',
            'python',
            'settings.json',
            'portable',
        ],
        stagedEntries: [
            'app.exe',
            'ffmpeg.dll',
            'new-only.dll',
            'resources',
            'release-files.txt',
            'portable',
        ],
        newFiles: ['app.exe', 'ffmpeg.dll', 'new-only.dll', 'resources', 'release-files.txt'],
        oldFiles: ['app.exe', 'ffmpeg.dll', 'old-only.dll', 'resources', 'release-files.txt'],
        executable: 'app.exe',
        ...overrides,
    });

test('the swap takes every entry either release owns, resources next to last and the executable last', () => {
    expect(plan()).toEqual([
        { name: 'ffmpeg.dll', old: true, new: true },
        { name: 'new-only.dll', old: false, new: true },
        // a file the new release dropped goes, and nothing takes its place
        { name: 'old-only.dll', old: true, new: false },
        { name: 'release-files.txt', old: true, new: true },
        { name: 'resources', old: true, new: true },
        { name: 'app.exe', old: true, new: true },
    ]);
});

test("the swap never names the install's data or the zip's portable marker", () => {
    const named = plan().map(({ name }) => name);
    for (const data of ['python', 'settings.json', 'portable']) {
        expect(named).not.toContain(data);
    }
});

test('an entry the installed release owns, and the install lacks, is not moved out', () => {
    expect(plan({ installEntries: ['app.exe', 'resources', 'release-files.txt'] })).toContainEqual({
        name: 'ffmpeg.dll',
        old: false,
        new: true,
    });
});

test("a new entry whose name the install has but its release does not own is the user's: refused", () => {
    expect(() =>
        plan({ installEntries: ['app.exe', 'resources', 'release-files.txt', 'new-only.dll'] })
    ).toThrow('not part of chaiNNer-C');
});

test("a release that names the install's data, the update's files or a path is refused", () => {
    for (const name of [
        ...INSTALL_DATA_NAMES,
        '..',
        '..\\x',
        '../x',
        'a/b',
        String.raw`C:\Windows`,
        '/etc',
        "it's.dll",
        'a"b',
        'a\nb',
        'a b',
        'a$b',
        'a;b',
        'a`b',
    ]) {
        expect(() =>
            plan({ newFiles: ['app.exe', 'resources', 'release-files.txt', name] })
        ).toThrow('may not replace');
        expect(() => plan({ oldFiles: ['app.exe', name] })).toThrow('may not replace');
    }
});

test('the app and Chromium keep their data under names no release may have', () => {
    for (const name of [
        'python',
        'settings.json',
        'settings',
        'settings_old',
        'backend-storage',
        'logs',
        'ffmpeg',
        UI_CACHE_FOLDER,
        'update-result.json',
        'portable',
        JOURNAL,
        STAGED_LIST,
        LOCK,
        UPDATE_NEW,
        UPDATE_OLD,
        UPDATE_DONE,
        INSTALLED_LIST,
        // the backend's caches in portable mode (auto_engine.py)
        'cache',
    ]) {
        expect(INSTALL_DATA_NAMES).toContain(name);
    }
});

test.runIf(windows)(
    "on Windows a release's name differs from the install's data only by case: refused",
    () => {
        for (const name of ['Logs', 'CACHE', 'Python', 'Settings.json']) {
            expect(() =>
                plan({ newFiles: ['app.exe', 'resources', 'release-files.txt', name] })
            ).toThrow('may not replace');
        }
        expect(() =>
            plan({
                installEntries: ['app.exe', 'resources', 'release-files.txt', 'models'],
                stagedEntries: ['app.exe', 'resources', 'release-files.txt', 'Models'],
                newFiles: ['app.exe', 'resources', 'release-files.txt', 'Models'],
            })
        ).toThrow('adds "Models"');
    }
);

test('a staged release that misses an entry it lists, or the three every release has, is refused', () => {
    expect(() => plan({ stagedEntries: ['ffmpeg.dll', 'resources', 'release-files.txt'] })).toThrow(
        'not whole'
    );
    for (const required of ['app.exe', 'resources', 'release-files.txt']) {
        expect(() =>
            plan({
                newFiles: [
                    'ffmpeg.dll',
                    ...['app.exe', 'resources', 'release-files.txt'].filter((n) => n !== required),
                ],
            })
        ).toThrow(`does not list ${required}`);
    }
});

test('the list of release files is one name per line', () => {
    expect(parseReleaseFiles('a.dll\r\nresources\n\n  b  \n')).toEqual(['a.dll', 'resources', 'b']);
});

test("an entry's state is read from where it is", () => {
    const both = { name: 'e', old: true, new: true };
    expect(entryState(both, true, false, true)).toBe('start');
    expect(entryState(both, false, true, true)).toBe('mid');
    expect(entryState(both, true, true, false)).toBe('end');
    expect(entryState(both, false, true, false)).toBe('broken');
    expect(entryState(both, true, true, true)).toBe('broken');
    const dropped = { name: 'e', old: true, new: false };
    expect(entryState(dropped, true, false, false)).toBe('start');
    expect(entryState(dropped, false, true, false)).toBe('end');
    expect(entryState(dropped, false, false, false)).toBe('broken');
    const added = { name: 'e', old: false, new: true };
    expect(entryState(added, false, false, true)).toBe('start');
    expect(entryState(added, true, false, false)).toBe('end');
    expect(entryState(added, true, false, true)).toBe('broken');
});

const journalStart = {
    app: { pid: 12, start: '638123' },
    release: 'chaiNNer-C\tv0.3.12',
    zip: 'C:\\Downloads\\r.zip',
    result: 'C:\\update-result.json',
    launch: 'chaiNNer.exe',
    executable: 'chaiNNer.exe',
    staged: UPDATE_NEW,
    entries: [
        { name: 'a.dll', old: true, new: true },
        { name: 'resources', old: true, new: true },
        { name: 'chaiNNer.exe', old: true, new: true },
    ],
};

test('the journal holds a closed plan, and reads back the lines written after it', () => {
    const text = journalText(journalStart);
    expect(parseJournal(text)).toEqual({
        ...journalStart,
        release: 'chaiNNer-C v0.3.12',
        resumes: [],
    });
    const later = `${text}moved\ta.dll\t.update-old\\a.dll\nresume\tforward:mid\nundo\twhy\nfailed\tstuck\n`;
    expect(parseJournal(later)).toMatchObject({
        undo: 'why',
        failed: 'stuck',
        resumes: ['forward:mid'],
    });
    expect(() => journalText({ ...journalStart, zip: 'a\nb' })).toThrow('control character');
    expect(() => journalText({ ...journalStart, launch: '..\\x.exe' })).toThrow(
        'not a plain file name'
    );
    expect(() => journalText({ ...journalStart, staged: '..' })).toThrow('not in .update-new');
});

test('a journal cut at any line, empty or zero-filled is not a plan', () => {
    const lines = journalText(journalStart).split('\n');
    // every cut before the end marker's line is complete
    upTo(lines.length - 2).forEach((kept) => {
        expect(() => parseJournal(`${lines.slice(0, kept).join('\n')}\n`)).toThrow('not complete');
    });
    expect(() => parseJournal('')).toThrow('not complete');
    expect(() => parseJournal('\0'.repeat(4096))).toThrow('not complete');
    // a plan with an entry after the end marker, or a count that does not match
    expect(() => parseJournal(journalText(journalStart).replace('end\t3', 'end\t2'))).toThrow(
        'not complete'
    );
});

test('the staged list holds every file with its size and digest, closed by an end marker', () => {
    const list = {
        owner: { pid: 3, start: '9' },
        files: [{ path: 'resources/app/main.js', size: 7, sha256: 'a'.repeat(64) }],
    };
    const text = stagedListText(list);
    expect(parseStagedList(text)).toEqual(list);
    expect(() => parseStagedList(text.split('\n').slice(0, 3).join('\n'))).toThrow('not complete');
    expect(() =>
        stagedListText({ ...list, files: [{ path: '../x', size: 1, sha256: 'a'.repeat(64) }] })
    ).toThrow('cannot record');
});

test('a checked download is unpacked into the install, every file listed, and the zip stays in Downloads', async () => {
    const progress: number[] = [];
    await installOld();
    answerWith(storedZip(newRelease));
    const update = await stageUpdate((p) => progress.push(p));
    expect(update.release).toBe('chaiNNer-C v0.3.11');
    expect(path.dirname(update.zip)).toBe(paths.downloads);
    expect(existsSync(update.zip)).toBe(true);
    expect(existsSync(`${update.zip}.part`)).toBe(false);
    expect(update.unpacked).toBe(path.join(paths.install, UPDATE_NEW));
    expect(progress.at(-1)).toBe(100);
    const list = parseStagedList(await readFile(path.join(paths.install, STAGED_LIST), 'utf-8'));
    expect(list.owner.pid).toBe(process.pid);
    // the entries the release owns: not the zip's portable marker
    expect(list.files.map((file) => file.path).sort()).toEqual(
        Object.keys(newRelease)
            .filter((name) => name !== 'portable' && !name.endsWith('/'))
            .sort()
    );
    expect(list.files.find((file) => file.path === 'resources/app/main.js')).toEqual({
        path: 'resources/app/main.js',
        size: 7,
        sha256: createHash('sha256').update('new app').digest('hex'),
    });
    // the probe of the installer left nothing
    expect(
        (await readdir(paths.install)).filter((name) => name.startsWith('update-probe'))
    ).toEqual([]);
});

test("a file of the release's name the user keeps in Downloads is neither replaced nor removed", async () => {
    const name = answerWith(storedZip(newRelease));
    const kept = path.join(paths.downloads, name);
    await writeFile(kept, "the user's own copy");
    await installOld();
    const update = await stageUpdate(() => {});
    expect(update.zip).not.toBe(kept);
    await discardUpdate(update);
    expect(existsSync(update.zip)).toBe(false);
    expect(await readFile(kept, 'utf-8')).toBe("the user's own copy");
    expect(await treeOf(paths.install)).toEqual(before());
});

// e.g. an install in Program Files: the installer would retry every file it may not replace
test('an install that cannot be written is refused before anything is downloaded', async () => {
    answerWith(storedZip(newRelease));
    await rm(paths.install, { recursive: true, force: true });
    await writeFile(paths.install, 'a file where the install folder should be');
    await expect(stageUpdate(() => {})).rejects.toThrow('cannot write to');
    expect(fetchRelease).not.toHaveBeenCalled();
    expect(download).not.toHaveBeenCalled();
});

test('an install without the list of its files is refused before anything is downloaded', async () => {
    answerWith(storedZip(newRelease));
    await writeTree(paths.install, { [executable]: 'old exe' });
    await expect(stageUpdate(() => {})).rejects.toThrow('no list of the files it owns');
    expect(download).not.toHaveBeenCalled();
});

test('a download that does not match GitHub is removed and not unpacked', async () => {
    await installOld();
    answerWith(storedZip(newRelease), `sha256:${'0'.repeat(64)}`);
    await expect(stageUpdate(() => {})).rejects.toThrow('does not match');
    expect(await readdir(paths.downloads)).toEqual([]);
    expect(await treeOf(paths.install)).toEqual(before());
});

test('a release that is not a chaiNNer-C build, or reaches outside its folder, leaves nothing', async () => {
    await installOld();
    await inTurn(
        [{ 'readme.txt': 'not an app' }, { ...newRelease, '../outside.txt': 'x' }] as Record<
            string,
            string
        >[],
        async (files) => {
            answerWith(storedZip(files));
            await expect(stageUpdate(() => {})).rejects.toThrow();
            expect(await readdir(paths.downloads)).toEqual([]);
            expect(await treeOf(paths.install)).toEqual(before());
        }
    );
    expect(existsSync(path.join(path.dirname(paths.install), 'outside.txt'))).toBe(false);
});

test('a write refused while staging leaves the install as it was', async () => {
    await installOld();
    // a file, then a folder of the same name: unpacking it fails halfway
    answerWith(storedZip({ ...newRelease, 'resources/app': 'a file', 'resources/app/x.js': 'x' }));
    await expect(stageUpdate(() => {})).rejects.toThrow();
    expect(await treeOf(paths.install)).toEqual(before());
    expect(await readdir(paths.downloads)).toEqual([]);
});

test('an installer that cannot run on this PC stops the update before it starts, saying why', async () => {
    await installOld();
    answerWith(storedZip(newRelease));
    const { PATH } = process.env;
    // no PowerShell or shell tools to be found: the probe fails as a policy or a missing tool would
    process.env.PATH = '';
    try {
        await expect(stageUpdate(() => {})).rejects.toThrow('cannot update itself on this PC');
    } finally {
        process.env.PATH = PATH;
    }
    expect(await treeOf(paths.install)).toEqual(before());
    expect(await readdir(paths.downloads)).toEqual([]);
});

test('an installer that cannot be started is reported at the next start, and its zip is kept for an install by hand', async () => {
    const update = await stage();
    spawnInstaller.mockImplementationOnce(() => fakeChild('error'));
    await expect(startInstaller(update)).rejects.toThrow('ENOENT');
    expect(finishFiles.filter((file) => existsSync(path.join(paths.install, file)))).toEqual([]);
    expect(await takeUpdateResult()).toMatchObject({
        ok: false,
        release: 'chaiNNer-C v0.3.11',
        message: expect.stringContaining('ENOENT') as unknown,
        zip: update.zip,
    });
    removeUnpackedReleases();
    expect(existsSync(update.zip)).toBe(true);
    // the next start clears the staged release its dead owner left
    expect(await checkUnfinishedUpdate()).toBeUndefined();
    expect(await treeOf(paths.install)).toEqual(before());
});

test("an entry of the user's that a new release would replace stops Update now, naming it and the way out", async () => {
    await installOld();
    await writeTree(paths.install, { 'models/mine.pth': "the user's model" });
    answerWith(
        storedZip({
            ...newRelease,
            'models/default.pth': 'a model the release ships',
            'release-files.txt': listOf([executable, 'resources', 'release-files.txt', 'models']),
        })
    );
    const refusal = stageUpdate(() => {});
    await expect(refusal).rejects.toThrow('adds "models"');
    await expect(refusal).rejects.toThrow('Move "models" out of the install folder');
    expect(await treeOf(paths.install)).toEqual({
        ...before(),
        'models/mine.pth': "the user's model",
    });
    expect(await readdir(paths.downloads)).toEqual([]);
});

test('a read-only lock file that no installer holds is no running update: it is removed and the app starts', async () => {
    await installOld();
    const lock = path.join(paths.install, LOCK);
    await writeFile(lock, '');
    await chmod(lock, 0o444);
    expect(await checkUnfinishedUpdate()).toBeUndefined();
    expect(existsSync(lock)).toBe(false);
});

test('a start without a journal removes what a whole update replaced, never putting it back', async () => {
    await writeTree(paths.install, {
        ...after(),
        [`${UPDATE_DONE}/old-only.dll`]: 'what the new release dropped',
        [`${UPDATE_DONE}/resources/app/main.js`]: 'old app',
    });
    expect(await checkUnfinishedUpdate()).toBeUndefined();
    expect(await treeOf(paths.install)).toEqual(after());
});

test('quitting removes a release this process staged and does not install, not the one it installs', async () => {
    const left = await stage();
    removeUnpackedReleases();
    expect(existsSync(left.zip)).toBe(false);
    expect(await treeOf(paths.install)).toEqual(before());

    answerWith(storedZip(newRelease));
    const installed = await stageUpdate(() => {});
    await startInstaller(installed);
    removeUnpackedReleases();
    expect(existsSync(installed.unpacked)).toBe(true);
    expect(existsSync(path.join(paths.install, STAGED_LIST))).toBe(true);
    expect(existsSync(installed.zip)).toBe(true);
    await rm(installed.zip, { force: true });
});

test("the last update's result is read once", async () => {
    const file = path.join(paths.userData, 'update-result.json');
    await writeFile(
        file,
        '\uFEFF{"ok":false,"release":"chaiNNer-C v0.3.11","message":"m","zip":"z"}'
    );
    expect(await takeUpdateResult()).toEqual({
        ok: false,
        release: 'chaiNNer-C v0.3.11',
        message: 'm',
        zip: 'z',
    });
    expect(await takeUpdateResult()).toBeUndefined();
});

test('the installer, its journal, its staged list and the finish files sit in the install', async () => {
    const update = await prepare();
    expect(await readdir(paths.install)).toEqual(
        expect.arrayContaining([...finishFiles, STAGED_LIST, UPDATE_NEW])
    );
    expect(await readFile(installerScript(), 'utf-8')).toBe(
        windows ? WINDOWS_INSTALLER : LINUX_INSTALLER
    );
    expect(await readFile(path.join(paths.install, 'update-not-finished.txt'), 'utf-8')).toContain(
        update.zip
    );
    const journal = parseJournal(await readFile(journalFile(), 'utf-8'));
    expect(journal.app.pid).toBe(process.pid);
    expect(journal.app.start).toBe(processStartTime(process.pid));
    expect(journal.entries.map(({ name }) => name).slice(-2)).toEqual(['resources', executable]);
});

test("a release that names the install's data is refused at Update now, before anything is written", async () => {
    await expect(
        stage({
            ...newRelease,
            'settings.json': 'a release that would replace the settings',
            'release-files.txt': listOf([
                executable,
                'resources',
                'release-files.txt',
                'settings.json',
            ]),
        })
    ).rejects.toThrow('may not replace');
    expect(existsSync(journalFile())).toBe(false);
    expect(await treeOf(paths.install)).toEqual(before());
    expect(await readdir(paths.downloads)).toEqual([]);
});

test('a start without a journal removes what an update left, puts back what has no counterpart, and starts', async () => {
    await writeTree(paths.install, {
        ...before(),
        [`${UPDATE_OLD}/ffmpeg.dll`]: 'an old copy: the install has its own',
        [`${UPDATE_OLD}/lost.dll`]: 'the only copy',
        [`${UPDATE_NEW}/new.dll`]: 'left',
    });
    expect(await checkUnfinishedUpdate()).toBeUndefined();
    expect(await treeOf(paths.install)).toEqual({ ...before(), 'lost.dll': 'the only copy' });
    expect(spawnInstaller).not.toHaveBeenCalled();
});

/** Runs `test` while another process runs; it gets that process's mark as a journal line holds it. */
const whileAnotherRuns = async (run: (mark: string) => Promise<void>) => {
    const other = execFile(process.execPath, ['-e', 'setTimeout(() => {}, 60000)'], {
        windowsHide: true,
        env: { ...process.env, ELECTRON_RUN_AS_NODE: '1' },
    });
    try {
        await vi.waitFor(() => expect(processStartTime(other.pid ?? 0)).toBeDefined());
        await run(`${other.pid ?? 0}\t${processStartTime(other.pid ?? 0) ?? ''}`);
    } finally {
        other.kill();
    }
};

test('a start leaves alone a release another running instance stages', async () => {
    await stage();
    const listFile = path.join(paths.install, STAGED_LIST);
    const text = await readFile(listFile, 'utf-8');
    await whileAnotherRuns(async (mark) => {
        await writeFile(listFile, text.replace(/^owner\t.*$/m, `owner\t${mark}`));
        expect(await checkUnfinishedUpdate()).toBeUndefined();
        // and this process may not stage over it
        answerWith(storedZip(newRelease));
        await expect(stageUpdate(() => {})).rejects.toThrow('Another chaiNNer-C');
    });
    expect(existsSync(path.join(paths.install, UPDATE_NEW))).toBe(true);
    expect(existsSync(listFile)).toBe(true);
});

test('a start after a journal cut short or emptied, with nothing moved aside, starts and says so', async () => {
    await inTurn(['', '\0'.repeat(4096), 'cut'], async (content) => {
        await prepare();
        const full = await readFile(journalFile(), 'utf-8');
        await writeFile(
            journalFile(),
            content === 'cut' ? full.slice(0, full.indexOf('\nend\t') + 1) : content
        );
        expect(await checkUnfinishedUpdate()).toBeUndefined();
        expect(await treeOf(paths.install)).toEqual(before());
        expect(await readResult()).toMatchObject({
            ok: false,
            message: expect.stringContaining('cut short before it changed anything') as unknown,
        });
    });
});

test('a first start after a cut-short update resumes it, saying so; a second with no progress tells how to finish', async () => {
    await prepare();
    spawnInstaller.mockClear();
    const resume = await checkUnfinishedUpdate();
    expect(resume?.title).toBe('Finishing the update');
    // no recorded failure: no way by hand in the message
    expect(resume?.text).not.toContain('To finish it:');
    // the installer starts once the message is closed: an open message never holds it waiting
    expect(spawnInstaller).not.toHaveBeenCalled();
    await resume?.afterwards?.();
    expect(spawnInstaller).toHaveBeenCalledTimes(1);
    const message = await checkUnfinishedUpdate();
    expect(message?.title).toBe('The update did not finish');
    expect(message?.text).toContain('changed nothing');
    expect(message?.text).toContain('start chaiNNer-C again');
    expect(spawnInstaller).toHaveBeenCalledTimes(1);
});

test('after an installer said why it stopped, the start resumes while the install proves, again after a second failure, and asks when it does not', async () => {
    await prepare();
    const reason = 'chaiNNer-C did not close within 10 minutes.';
    await appendFile(journalFile(), `installer\t1\nfailed\t${reason}\n`);
    spawnInstaller.mockClear();
    const first = await checkUnfinishedUpdate();
    expect(first?.title).toBe('Finishing the update');
    expect(first?.text).toContain(reason);
    // a failure that may come back: the way by hand, as the fallback
    expect(first?.text).toContain('If it stops again: To finish it:');
    await first?.afterwards?.();
    expect(spawnInstaller).toHaveBeenCalledTimes(1);
    expect((await readFile(journalFile(), 'utf-8')).endsWith('checked\n')).toBe(true);
    // the resumed installer failed the same way: offered again, not "no progress"
    await appendFile(journalFile(), `installer\t2\nfailed\t${reason}\n`);
    const second = await checkUnfinishedUpdate();
    expect(second?.title).toBe('Finishing the update');
    await second?.afterwards?.();
    expect(spawnInstaller).toHaveBeenCalledTimes(2);
    // an installer that never ran after that resume: told, not started again
    expect((await checkUnfinishedUpdate())?.text).toContain('changed nothing');
    // and an install changed from outside after a failure: asked
    await appendFile(journalFile(), `installer\t3\nfailed\t${reason}\n`);
    await writeTree(paths.install, { 'resources/app/main.js': 'changed from outside' });
    expect((await checkUnfinishedUpdate())?.title).toBe(CHANGED);
    expect(spawnInstaller).toHaveBeenCalledTimes(2);
});

test('a start while the app that started the installer still runs tells so and touches nothing', async () => {
    await prepare();
    const text = await readFile(journalFile(), 'utf-8');
    await whileAnotherRuns(async (mark) => {
        await writeFile(journalFile(), text.replace(/^app\t.*$/m, `app\t${mark}`));
        const tree = await treeOf(paths.install);
        expect((await checkUnfinishedUpdate())?.title).toBe('chaiNNer-C is being updated');
        expect(spawnInstaller).toHaveBeenCalledTimes(1);
        expect(await treeOf(paths.install)).toEqual(tree);
    });
});

windowsOnly(
    'the Windows installer swaps the release in, leaves the data, drops what the release dropped, and starts the app',
    async () => {
        await inTurn([SAME_LAYOUT, ALL_DIFFERENT], async (chosen) => {
            usePair(chosen);
            const update = await prepare();
            expect(runInstaller().status).toBe(0);
            await expectUpdated(update.zip);
        });
    },
    150_000
);

/** Holds the install's lock from another process, as a running installer does. */
const HOLD_LOCK = String.raw`param([string]$Path)
$lock = [IO.File]::Open($Path, 'OpenOrCreate', 'ReadWrite', 'None')
[Console]::Out.WriteLine('held')
Start-Sleep -Seconds 60
`;

windowsOnly(
    'while an installer holds the lock, a start says so and a second installer leaves at once, changing nothing',
    async () => {
        await prepare();
        const script = path.join(scratch, 'hold-lock.ps1');
        await writeFile(script, HOLD_LOCK, 'utf-8');
        const holder = execFile(
            'powershell.exe',
            [
                '-NoProfile',
                '-ExecutionPolicy',
                'Bypass',
                '-File',
                script,
                '-Path',
                path.join(paths.install, LOCK),
            ],
            { windowsHide: true }
        );
        try {
            await new Promise<void>((resolve) => {
                holder.stdout?.on('data', () => resolve());
            });
            const tree = await treeOf(paths.install);
            spawnInstaller.mockClear();
            expect((await checkUnfinishedUpdate())?.title).toBe('chaiNNer-C is being updated');
            expect(spawnInstaller).not.toHaveBeenCalled();
            const second = runInstaller();
            expect(second.status).toBe(0);
            // a finish script run by hand says why it did nothing
            expect(second.stdout).toContain('Another installer is finishing this update');
            expect(await treeOf(paths.install)).toEqual(tree);
        } finally {
            endTree(holder.pid);
        }
    },
    150_000
);

windowsOnly(
    'two installers started at once: one swaps, the other leaves, and the install is the new release',
    async () => {
        await inTurn(upTo(3), async () => {
            const update = await prepare();
            const run = () =>
                new Promise<void>((resolve) => {
                    execFile(
                        'powershell.exe',
                        [
                            '-NoProfile',
                            '-NonInteractive',
                            '-ExecutionPolicy',
                            'Bypass',
                            '-File',
                            installerScript(),
                        ],
                        { timeout: 120_000, windowsHide: true },
                        () => resolve()
                    );
                });
            await Promise.all([run(), run()]);
            await expectUpdated(update.zip);
        });
    },
    300_000
);

windowsOnly(
    'a rename that stays refused undoes the update: the install is as it was, and the old version starts',
    async () => {
        const update = await prepare();
        const holder = await holdResources();
        try {
            expect(runInstaller().status).toBe(0);
        } finally {
            endTree(holder.pid);
        }
        expect(await treeOf(paths.install)).toEqual(before());
        expect(await readResult()).toMatchObject({
            ok: false,
            message: expect.stringContaining(
                'The update was undone: resources could not be moved'
            ) as unknown,
        });
        expect(existsSync(update.zip)).toBe(true);
        await vi.waitFor(() => expect(existsSync(launchedFile())).toBe(true), { timeout: 10_000 });
    },
    150_000
);

windowsOnly(
    'a staged file removed or cut short, or the staged list gone, stops the update with nothing changed',
    async () => {
        await inTurn(['removed', 'cut', 'list'], async (damage) => {
            await prepare();
            const staged = path.join(paths.install, UPDATE_NEW);
            if (damage === 'removed') await rm(path.join(staged, executable));
            if (damage === 'cut')
                await writeFile(path.join(staged, 'resources', 'app', 'main.js'), 'new');
            if (damage === 'list') await rm(path.join(paths.install, STAGED_LIST));
            expect(runInstaller().status).toBe(0);
            expect(await treeOf(paths.install)).toEqual(before());
            expect((await readResult()).ok).toBe(false);
        });
    },
    300_000
);

/**
 * Runs the Windows installer with its journal's `moved` lines counted: the process ends as it is
 * about to record its `KillAfter`th rename, which it has made.
 */
const KILLED_AFTER = String.raw`param([string]$Installer, [string]$Count, [int]$KillAfter)
function Add-Content {
    param([string]$LiteralPath, [string]$Value, [string]$Encoding)
    if ($Value.StartsWith('moved' + [char]9)) {
        [IO.File]::AppendAllText($Count, 'x')
        if ([IO.File]::ReadAllText($Count).Length -ge $KillAfter) { [Environment]::Exit(9) }
    }
    Microsoft.PowerShell.Management\Add-Content -LiteralPath $LiteralPath -Value $Value -Encoding $Encoding
}
& $Installer @args
`;

const killedAfter = async (k: number) => {
    const wrapper = path.join(scratch, 'killed-after.ps1');
    await writeFile(wrapper, KILLED_AFTER, 'utf-8');
    // a count of its own: a test may kill at the same rename twice
    const count = path.join(scratch, `count-${k}.txt`);
    await rm(count, { force: true });
    const run = powershell([
        wrapper,
        '-Installer',
        installerScript(),
        '-Count',
        count,
        '-KillAfter',
        String(k),
        '-ProcessId',
        String(endedProcessId()),
    ]);
    expect(run.status).toBe(9);
};

windowsOnly(
    'an installer killed after any rename is resumed by the next start to the whole update, whatever its record lines say',
    async () => {
        const kills = [SAME_LAYOUT, ALL_DIFFERENT].flatMap((chosen) =>
            upTo(chosen.renames).map((k) => ({ chosen, k }))
        );
        await inTurn(kills, async ({ chosen, k }) => {
            usePair(chosen);
            const update = await prepare();
            await killedAfter(k);
            // the journal's record lines do not steer a resume: double some, drop others
            const text = await readFile(journalFile(), 'utf-8');
            const lines = text.split('\n');
            const moved = lines.filter((line) => line.startsWith('moved\t'));
            await writeFile(
                journalFile(),
                k % 2 === 0
                    ? `${text}${moved.join('\n')}\n`
                    : `${lines.filter((line) => !line.startsWith('moved\t')).join('\n')}`
            );
            spawnInstaller.mockClear();
            if (k === chosen.renames) {
                // every rename made: the new release is in place, and the start clears up
                expect(await checkUnfinishedUpdate()).toBeUndefined();
                expect(await treeOf(paths.install)).toEqual(after());
                expect(await readResult()).toMatchObject({ ok: true });
                return;
            }
            const resume = await checkUnfinishedUpdate();
            expect(resume?.title).toBe('Finishing the update');
            await resume?.afterwards?.();
            const [program, args] = spawnInstaller.mock.calls[0] as unknown as [string, string[]];
            expect(program).toBe('powershell.exe');
            expect(args).toContain(installerScript());
            // what the start started, run here
            expect(runInstaller().status).toBe(0);
            await expectUpdated(update.zip);
        });
    },
    600_000
);

windowsOnly(
    'finish-update.cmd after a kill refuses until the start has checked the install, then finishes',
    async () => {
        const cmd = () =>
            spawnSync('cmd.exe', ['/d', '/c', '.\\finish-update.cmd'], {
                cwd: paths.install,
                encoding: 'utf-8',
                timeout: 120_000,
                windowsHide: true,
            });
        await inTurn(upTo(pair.renames - 1), async (k) => {
            const update = await prepare();
            await killedAfter(k);
            const found = await treeOf(paths.install);
            const refused = cmd();
            expect(refused.stdout).toContain('Start chaiNNer-C to finish this update');
            expect(refused.status).toBe(0);
            expect(await treeOf(paths.install)).toEqual(found);
            // the start checks the install and starts the installer; run here by hand instead
            const resume = await checkUnfinishedUpdate();
            expect(resume?.title).toBe('Finishing the update');
            await resume?.afterwards?.();
            const run = cmd();
            expect(`${run.stdout}${run.stderr}`).toBe('');
            expect(run.status).toBe(0);
            await expectUpdated(update.zip);
        });
    },
    600_000
);

windowsOnly(
    'an installer killed at any rename of an undo is resumed to the install as it was',
    async () => {
        // ten renames before resources, which is held, then ten undone
        await inTurn(upTo(10), async (j) => {
            const update = await prepare();
            const holder = await holdResources();
            try {
                await killedAfter(10 + j);
            } finally {
                endTree(holder.pid);
            }
            expect(parseJournal(await readFile(journalFile(), 'utf-8')).undo).toContain(
                'resources could not be moved'
            );
            // resources could be moved now: the checked resume still undoes
            const resume = await checkUnfinishedUpdate();
            expect(resume?.title).toBe('Finishing the update');
            await resume?.afterwards?.();
            expect(runInstaller().status).toBe(0);
            expect(await treeOf(paths.install)).toEqual(before());
            expect(await readResult()).toMatchObject({ ok: false });
            expect(existsSync(update.zip)).toBe(true);
        });
    },
    600_000
);

windowsOnly(
    'the staged zip unpacked by hand over an interrupted update, at any kill point: cleared up and started, with no question',
    async () => {
        await inTurn(upTo(pair.renames), async (k) => {
            await prepare();
            await killedAfter(k);
            // the start asked first (or would have) and the user chose Quit: nothing was changed
            const { portable, ...files } = newRelease;
            await writeTree(paths.install, files);
            expect(await checkUnfinishedUpdate()).toBeUndefined();
            const tree = await treeOf(paths.install);
            expect(tree).toEqual(after());
            expect(await readResult()).toMatchObject({ ok: true });
        });
    },
    600_000
);

windowsOnly(
    'the staged zip unpacked over resources not yet swapped, where the old release had files the new one dropped: asked',
    async () => {
        usePair(ALL_DIFFERENT);
        await inTurn([3, 7, 9], async (k) => {
            await prepare();
            await killedAfter(k);
            const { portable, ...files } = newRelease;
            await writeTree(paths.install, files);
            const answer = await checkUnfinishedUpdate();
            if (k < 7) {
                // the unpacking left the old release's files in resources beside the new ones (a
                // removed node would still load): not the staged release byte for byte
                expect(answer?.title).toBe(CHANGED);
                return;
            }
            expect(answer).toBeUndefined();
            expect(await treeOf(paths.install)).toMatchObject(after());
        });
    },
    300_000
);

/** What the update leaves in the install, by its own names: what Keep removes. */
const updateFiles = (name: string) => /^(?:\.update-|update-|finish-)/.test(name);

windowsOnly(
    'another release unpacked over a half-swapped install, completely or not: the start asks, and never keeps or resumes by itself',
    async () => {
        const newer: Record<string, string> = {
            [executable]: 'newer exe',
            'ffmpeg.dll': 'newer ffmpeg',
            'release-files.txt': listOf([
                executable,
                'ffmpeg.dll',
                'resources',
                'release-files.txt',
            ]),
            'resources/app/main.js': 'newer app',
            'resources/src/run.py': 'newer backend',
        };
        // an unpacking that stopped part-way: the list (stored early, by name) and one file
        const partial: Record<string, string> = {
            'release-files.txt': newer['release-files.txt'],
            'resources/app/main.js': 'newer app',
        };
        await inTurn(
            [3, 7, 9].flatMap((k) => [
                { k, files: newer },
                { k, files: partial },
            ]),
            async ({ k, files }) => {
                await prepare();
                await killedAfter(k);
                await writeTree(paths.install, files);
                const found = await treeOf(paths.install);
                spawnInstaller.mockClear();
                const question = await checkUnfinishedUpdate();
                expect(question?.title).toBe(CHANGED);
                expect(question?.keep).toBeDefined();
                expect(question?.afterwards).toBeUndefined();
                // Quit: nothing is touched, and the next start asks again
                expect(spawnInstaller).not.toHaveBeenCalled();
                expect(await treeOf(paths.install)).toEqual(found);
                expect((await checkUnfinishedUpdate())?.title).toBe(CHANGED);
                expect(await treeOf(paths.install)).toEqual(found);
                if (files === newer) {
                    // Keep: what is left of the update goes, the install is otherwise as found
                    await question?.keep?.();
                    const kept = Object.fromEntries(
                        Object.entries(found).filter(([name]) => !updateFiles(name))
                    );
                    expect(await treeOf(paths.install)).toEqual(kept);
                    expect(await readResult()).toMatchObject({ ok: true });
                    expect(await checkUnfinishedUpdate()).toBeUndefined();
                }
            }
        );
        // an install nobody unpacked over is not taken for one: the half-swapped install resumes
        await prepare();
        await killedAfter(7);
        expect((await checkUnfinishedUpdate())?.title).toBe('Finishing the update');
    },
    600_000
);

test('the start question: "Keep this release" keeps it, "Quit" leaves everything', async () => {
    const keep = vi.fn(() => Promise.resolve());
    const question = { title: 't', text: 'x', keep };
    showMessageBox.mockResolvedValueOnce({ response: 0 });
    expect(await showUpdateAtStart(question)).toBe(true);
    expect(keep).toHaveBeenCalledTimes(1);
    expect(showMessageBox).toHaveBeenCalledWith(
        expect.objectContaining({ buttons: ['Keep this release', 'Quit'], cancelId: 1 })
    );
    showMessageBox.mockResolvedValueOnce({ response: 1 });
    expect(await showUpdateAtStart(question)).toBe(false);
    expect(keep).toHaveBeenCalledTimes(1);
});

/**
 * Asserts that the start, finding the install changed from outside, asks and changes nothing:
 * no installer, no resume, no undo, the same question at the next start.
 */
const expectAsked = async () => {
    const found = await treeOf(paths.install);
    spawnInstaller.mockClear();
    const question = await checkUnfinishedUpdate();
    expect(question?.title).toBe(CHANGED);
    expect(question?.keep).toBeDefined();
    expect(question?.afterwards).toBeUndefined();
    expect(spawnInstaller).not.toHaveBeenCalled();
    expect(await treeOf(paths.install)).toEqual(found);
    expect((await checkUnfinishedUpdate())?.title).toBe(CHANGED);
    expect(await treeOf(paths.install)).toEqual(found);
    return question;
};

windowsOnly(
    'a third release with the same list unpacked over a half swap: the start asks, and changes nothing',
    async () => {
        // the same layout, other bytes: the list itself is byte-identical to the staged one's
        const third: Record<string, string> = {
            [executable]: 'third exe',
            'ffmpeg.dll': 'third ffmpeg',
            'icudtl.dat': 'third icu',
            'release-files.txt': newRelease['release-files.txt'],
            'resources/app/main.js': 'third app',
            'resources/src/run.py': 'third backend',
            [`${NODES}/node.py`]: 'third node',
        };
        await prepare();
        await killedAfter(5);
        await writeTree(paths.install, third);
        await expectAsked();
        // the finish script, by hand, refuses too
        expect(runInstaller().stdout).toContain('Start chaiNNer-C to finish this update');
    },
    150_000
);

windowsOnly(
    'an unpack that stopped before resources, a file changed deep in resources, or a release over an update not started: asked',
    async () => {
        await inTurn(
            [
                // only the executable and a DLL of a third release replaced
                { k: 3, files: { [executable]: 'third exe', 'ffmpeg.dll': 'third ffmpeg' } },
                // one file changed deep inside the old resources, not yet swapped
                { k: 3, files: { [`${NODES}/node.py`]: 'changed from outside' } },
                // a newer zip with the same list over an update that had not started
                {
                    k: 0,
                    files: {
                        'resources/app/main.js': 'newer app',
                        'release-files.txt': newRelease['release-files.txt'],
                    },
                },
            ] as { k: number; files: Record<string, string> }[],
            async ({ k, files }) => {
                await prepare();
                if (k > 0) {
                    await killedAfter(k);
                }
                await writeTree(paths.install, files);
                await expectAsked();
            }
        );
    },
    300_000
);

windowsOnly(
    'records the start cannot read prove nothing: it asks, with no error and no resume',
    async () => {
        await inTurn(
            [
                async () => rm(path.join(paths.install, STAGED_LIST)),
                async () => writeFile(path.join(paths.install, INSTALLED_LIST), 'garbage'),
            ],
            async (damage) => {
                await prepare();
                await killedAfter(3);
                await damage();
                const question = await expectAsked();
                expect(question?.text).toContain('records of the two releases cannot be read');
            }
        );
    },
    300_000
);

/** Runs the Windows installer, holding a file open from the moment the journal says swapped. */
const HELD_AFTER_SWAP = String.raw`param([string]$Installer, [string]$Hold)
function Add-Content {
    param([string]$LiteralPath, [string]$Value, [string]$Encoding)
    Microsoft.PowerShell.Management\Add-Content -LiteralPath $LiteralPath -Value $Value -Encoding $Encoding
    if ($Value -ceq 'swapped') { $script:held = [IO.File]::Open($Hold, 'Open', 'Read', 'None') }
}
& $Installer @args
`;

windowsOnly(
    'a file held open in what the update replaced: the update succeeds, the app starts, and the next start removes it, putting nothing back',
    async () => {
        const update = await prepare();
        const wrapper = path.join(scratch, 'held-after-swap.ps1');
        await writeFile(wrapper, HELD_AFTER_SWAP, 'utf-8');
        const run = powershell([
            wrapper,
            '-Installer',
            installerScript(),
            '-Hold',
            path.join(paths.install, UPDATE_OLD, executable),
            '-ProcessId',
            String(endedProcessId()),
        ]);
        expect(run.status).toBe(0);
        expect(await readResult()).toMatchObject({ ok: true, release: 'chaiNNer-C v0.3.11' });
        await vi.waitFor(() => expect(existsSync(launchedFile())).toBe(true), { timeout: 10_000 });
        expect(existsSync(journalFile())).toBe(false);
        expect(existsSync(update.zip)).toBe(false);
        // the handle closed with the installer: the next start removes what is left
        expect(await checkUnfinishedUpdate()).toBeUndefined();
        expect(await treeOf(paths.install)).toEqual(after());
    },
    150_000
);

windowsOnly(
    'an installer cut short after the swap, during its clean-up, reports the new release, not an undo',
    async () => {
        const update = await prepare();
        expect(runInstaller().status).toBe(0);
        // as if cut short with the new release whole: the journal says swapped, the staged list
        // is gone, what it replaced still lies aside
        await writeTree(paths.install, {
            [`${UPDATE_DONE}/old-only.dll`]: 'what the new release dropped',
        });
        const result = path.join(paths.userData, 'update-result.json');
        await writeFile(
            journalFile(),
            `${journalText({
                app: { pid: endedProcessId(), start: '1' },
                release: 'chaiNNer-C v0.3.11',
                zip: path.join(paths.downloads, 'gone.zip'),
                result,
                launch: launchName,
                executable,
                staged: UPDATE_NEW,
                entries: [{ name: executable, old: true, new: true }],
            })}swapped\nchecked\n`
        );
        await writeFile(installerScript(), WINDOWS_INSTALLER, 'utf-8');
        await rm(launchedFile(), { force: true });
        expect(runInstaller().status).toBe(0);
        expect(await treeOf(paths.install)).toEqual(after());
        expect(await readResult()).toMatchObject({ ok: true });
        expect(existsSync(update.zip)).toBe(false);
    },
    150_000
);

windowsOnly(
    'the Windows installer changes nothing and starts nothing while the app still runs',
    async () => {
        const update = await prepare();
        // the 10 minutes' wait, shortened
        await writeFile(
            installerScript(),
            WINDOWS_INSTALLER.replace('AddMinutes(10)', 'AddSeconds(2)'),
            'utf-8'
        );
        // this process runs throughout: the app that does not close
        expect(powershell([installerScript(), '-ProcessId', String(process.pid)]).status).toBe(0);
        expect(await treeOf(paths.install)).toEqual(before());
        const result = await readResult();
        expect(result.ok).toBe(false);
        expect(result.message).toContain('did not close');
        expect(result.message).toContain('stays at');
        expect(existsSync(update.zip)).toBe(true);
        await delay(2000);
        expect(existsSync(launchedFile())).toBe(false);
    },
    150_000
);

windowsOnly(
    'a journal cut at any line, or one that names a path or an odd name, is refused before anything is renamed',
    async () => {
        await prepare();
        const full = await readFile(journalFile(), 'utf-8');
        const lines = full.split('\n');
        const cuts = upTo(lines.length - 2).map((kept) => `${lines.slice(0, kept).join('\n')}\n`);
        const bad = [
            `entry\t..\\x\t1\t1`,
            `entry\tC:\\Windows\t1\t0`,
            `entry\tit's.dll\t1\t0`,
            `entry\ta"b.dll\t1\t0`,
            `entry\tffmpeg\n.dll\t1\t0`,
            `entry\tffmpeg.dll\t0\t0`,
            'launch\tC:\\Windows\\System32\\calc.exe',
            'staged\t.update-new\\..\\..',
        ].map((line) => full.replace(/^entry\t/m, `${line}\nentry\t`));
        await inTurn([...cuts, ...bad], async (text) => {
            await prepare();
            await writeFile(journalFile(), text);
            expect(runInstaller().status).toBe(0);
            // the old release in place, nothing moved aside (a cut journal is refused and left)
            const tree = await treeOf(paths.install);
            expect(tree).toMatchObject(before());
            expect(Object.keys(tree).filter((name) => name.startsWith(`${UPDATE_OLD}/`))).toEqual(
                []
            );
        });
    },
    600_000
);

/**
 * Runs the Windows installer with Get-CimInstance replaced: its first `failures` calls fail as a
 * broken WMI does (an error, no processes); every call is counted in `calls`.
 */
const WMI_FAILING = String.raw`param([string]$Installer, [string]$Calls, [int]$Failures)
function Get-CimInstance {
    [CmdletBinding()] param([Parameter(Position = 0)][string]$ClassName)
    Add-Content -LiteralPath $Calls -Value $ClassName
    if (@(Get-Content -LiteralPath $Calls).Count -le $Failures) {
        Write-Error 'The RPC server is unavailable.'
        return
    }
    CimCmdlets\Get-CimInstance $ClassName
}
& $Installer @args
`;

windowsOnly(
    'the Windows installer waits while the process list cannot be read',
    async () => {
        const update = await prepare();
        const calls = path.join(scratch, 'calls.txt');
        const wrapper = path.join(scratch, 'wmi-failing.ps1');
        await writeFile(wrapper, WMI_FAILING, 'utf-8');
        const run = powershell([
            wrapper,
            '-Installer',
            installerScript(),
            '-Calls',
            calls,
            '-Failures',
            '2',
            '-ProcessId',
            String(endedProcessId()),
        ]);
        expect(run.status).toBe(0);
        // the swap waited for the first answer: the third call
        expect(
            (await readFile(calls, 'utf-8')).trim().split(/\r?\n/).length
        ).toBeGreaterThanOrEqual(3);
        await expectUpdated(update.zip);
    },
    150_000
);

/**
 * A `mv` first on PATH that ends the installer right after the `KILL_AFTER`th rename: the
 * installer, and its command substitution if the shell made one for this mv.
 */
const KILLING_MV = `#!/bin/sh
/bin/mv "$@" || exit $?
echo moved >> "$COUNT"
[ "$(wc -l < "$COUNT")" -ge "$KILL_AFTER" ] || exit 0
p=$PPID
while grep -qa finish-update "/proc/$p/cmdline" 2>/dev/null; do
    parent=$(cut -d ' ' -f 4 "/proc/$p/stat"); kill -9 "$p"; p=$parent
done
exit 0
`;

linuxOnly(
    'the Linux installer swaps the release in, and one killed after any rename is resumed to the whole update',
    async () => {
        const update = await prepare();
        expect(runInstaller().status).toBe(0);
        await expectUpdated(update.zip);

        const bin = path.join(scratch, 'bin');
        await mkdir(bin, { recursive: true });
        await writeFile(path.join(bin, 'mv'), KILLING_MV, { mode: 0o755 });
        await inTurn(upTo(pair.renames), async (k) => {
            const staged = await prepare();
            spawnSync('/bin/sh', [installerScript(), String(endedProcessId())], {
                env: {
                    ...process.env,
                    PATH: `${bin}:${process.env.PATH ?? ''}`,
                    COUNT: path.join(scratch, `count-${k}`),
                    KILL_AFTER: String(k),
                },
                timeout: 120_000,
            });
            expect(existsSync(journalFile())).toBe(true);
            if (k === pair.renames) {
                expect(await checkUnfinishedUpdate()).toBeUndefined();
                expect(await treeOf(paths.install)).toEqual(after());
                return;
            }
            const resume = await checkUnfinishedUpdate();
            expect(resume?.title).toBe('Finishing the update');
            await resume?.afterwards?.();
            expect(runInstaller().status).toBe(0);
            await expectUpdated(staged.zip);
        });
    },
    600_000
);
