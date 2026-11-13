import { ChildProcess, execFile, spawn, spawnSync } from 'child_process';
import { createHash, randomUUID } from 'crypto';
import decompress from 'decompress';
import { app, dialog, net } from 'electron/main';
import {
    closeSync,
    createReadStream,
    existsSync,
    openSync,
    readFileSync,
    readdirSync,
    renameSync,
    rmSync,
} from 'fs';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import { log } from '../common/log';
import { lazy } from '../common/util';
import installDataNames from './install-data-names.json';
import { SupportedPlatform, currentExecutableDir, getPlatform } from './platform';
import { download } from './python/integratedPython';
import { readTextFile } from './util';

const REPOSITORY = 'Rift73/chaiNNer-C';

/** The fields of a GitHub release asset the update reads. */
export interface ReleaseAsset {
    name: string;
    size: number;
    digest?: string | null;
    browser_download_url: string;
}

export interface Release {
    tag_name: string;
    name: string | null;
    assets: ReleaseAsset[];
}

/** The release's zip for each platform the releases have. */
const ASSET_SUFFIX: Partial<Record<SupportedPlatform, string>> = {
    win32: '-win-x64.zip',
    linux: '-linux-x64.zip',
};

export const findReleaseAsset = (
    release: Release,
    platform: SupportedPlatform,
    arch: string
): ReleaseAsset => {
    const suffix = ASSET_SUFFIX[platform];
    const asset =
        suffix && arch === 'x64' ? release.assets.find((a) => a.name.endsWith(suffix)) : undefined;
    if (!asset) {
        throw new Error(`${release.tag_name} has no zip for ${platform} ${arch}.`);
    }
    return asset;
};

/** The SHA-256 GitHub reports for an asset, as lowercase hex. */
export const expectedSha256 = (asset: ReleaseAsset): string => {
    const match = /^sha256:([0-9a-f]{64})$/i.exec(asset.digest ?? '');
    if (!match) {
        throw new Error(`GitHub reports no SHA-256 for ${asset.name}, so it cannot be checked.`);
    }
    return match[1].toLowerCase();
};

export const fileSha256 = (file: string): Promise<string> =>
    new Promise((resolve, reject) => {
        const hash = createHash('sha256');
        createReadStream(file)
            .on('data', (chunk) => hash.update(chunk))
            .on('error', reject)
            .on('end', () => resolve(hash.digest('hex')));
    });

/**
 * The folder of an unpacked release that holds the app: the zip's top folder when it has only
 * one (the Linux zip), else the zip's root (the Windows zip). It must hold resources/app and the
 * executable this process runs.
 */
export const findReleaseRoot = async (unpacked: string, executable: string): Promise<string> => {
    const entries = await fs.readdir(unpacked, { withFileTypes: true });
    const root =
        entries.length === 1 && entries[0].isDirectory()
            ? path.join(unpacked, entries[0].name)
            : unpacked;
    for (const required of [path.join('resources', 'app'), executable]) {
        if (!existsSync(path.join(root, required))) {
            throw new Error(`The release has no ${required}: it is not a chaiNNer-C build.`);
        }
    }
    return root;
};

/** The install's folder a release is unpacked into before it is installed. */
export const UPDATE_NEW = '.update-new';
/** The install's folder the replaced entries are moved to while an update is installed. */
export const UPDATE_OLD = '.update-old';
/** UPDATE_OLD once the new release is whole: only removed, never put back. */
export const UPDATE_DONE = '.update-done';
/** In UPDATE_OLD: the new release is whole, and what is in it is only removed. */
const REPLACED_MARK = '.replaced';
/** The plan of an update in progress, in the install; it goes once the install is whole. */
export const JOURNAL = 'update-journal.txt';
/** Every staged file with its size, and the process that staged them; beside the journal. */
export const STAGED_LIST = 'update-staged.txt';
/** Every file of the installed release's entries, with its size and digest, written with the journal. */
export const INSTALLED_LIST = 'update-installed.txt';
/** One installer at a time: the file an installer holds while it runs. */
export const LOCK = 'update.lock';
/** What a user whose app does not start reads, beside the journal. */
export const NOTICE = 'update-not-finished.txt';
/** The top-level entries of the install a release owns, one per line, shipped in the release. */
export const RELEASE_FILES = 'release-files.txt';
/** What a probe of the installer's needs writes, and removes. */
const PROBE = 'update-probe';

/**
 * The install's entries the app and Chromium keep their data in (the integrated Python,
 * settings, storage, logs, caches) and the update's own files: a release may never name one.
 * packaging (forge.config.js) checks every release against the same list.
 */
export const INSTALL_DATA_NAMES: readonly string[] = installDataNames;

/** The installer, in the install beside the journal, and what a user runs to finish the update. */
const finishFiles = (platform: SupportedPlatform): { script: string; runner: string } =>
    platform === 'win32'
        ? { script: 'finish-update.ps1', runner: 'finish-update.cmd' }
        : { script: 'finish-update.sh', runner: 'finish-update.sh' };

/**
 * The names an update may rename or start: plain top-level names of letters, digits and `._+-`,
 * not starting with a dot. Release entries come from a download and the journal from a file on
 * disk: no other name gets near a rename, and both installers check the journal by the same rule.
 */
const PLAIN_NAME = /^[a-z\d][\w.+-]*$/i;

const checkPlainName = (name: string, what: string): string => {
    if (!PLAIN_NAME.test(name)) {
        throw new Error(`The update names ${what} "${name}", which is not a plain file name.`);
    }
    return name;
};

const hasControl = (value: string): boolean =>
    [...value].some((char) => char < ' ' || char === '\x7f');

/** The names of a RELEASE_FILES list. */
export const parseReleaseFiles = (text: string): string[] =>
    text
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line !== '');

/** A process, told apart from a later one that reuses its id by its start time. */
export interface ProcessMark {
    pid: number;
    start: string;
}

/** A process's start time as both installers record it, or undefined when it does not run. */
export const processStartTime = (pid: number): string | undefined => {
    // read from a file on disk: only a positive process id goes into the query
    if (!Number.isSafeInteger(pid) || pid <= 0) {
        return undefined;
    }
    if (getPlatform() === 'win32') {
        const query = spawnSync(
            'powershell.exe',
            [
                '-NoProfile',
                '-NonInteractive',
                '-Command',
                `(Get-Process -Id ${pid} -ErrorAction Stop).StartTime.ToUniversalTime().Ticks`,
            ],
            { encoding: 'utf-8', windowsHide: true }
        );
        return query.status === 0 ? query.stdout.trim() : undefined;
    }
    try {
        const stat = readFileSync(`/proc/${pid}/stat`, 'utf-8');
        // the fields after the command's name, which may hold spaces; the 22nd is the start time
        return stat.slice(stat.lastIndexOf(')') + 2).split(' ')[19];
    } catch (error) {
        if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
            return undefined;
        }
        throw error;
    }
};

const ownMark = lazy(
    (): ProcessMark => ({ pid: process.pid, start: processStartTime(process.pid) ?? '' })
);

/** Whether the process still runs; a mark without a start time never does. */
const isRunning = (mark: ProcessMark): boolean =>
    mark.start !== '' && processStartTime(mark.pid) === mark.start;

/** One top-level entry of an update: whether the installed release has it, and the new one. */
export interface PlanEntry {
    name: string;
    old: boolean;
    new: boolean;
}

export interface SwapPlan {
    /** The install's top-level entries. */
    installEntries: readonly string[];
    /** The staged release's top-level entries. */
    stagedEntries: readonly string[];
    /** The entries the new release owns: its RELEASE_FILES. */
    newFiles: readonly string[];
    /** The entries the installed release owns: its RELEASE_FILES. */
    oldFiles: readonly string[];
    /** The name of the executable. */
    executable: string;
}

/**
 * The entries an update swaps, in order: the others first, then `resources` (the app is out of
 * place only between its two renames), the executable last. Only an entry the installed release
 * owns is ever moved out; a new entry whose name the install has but its release does not own is
 * the user's, and the update is refused. Neither release may name the install's data.
 */
export const planSwap = ({
    installEntries,
    stagedEntries,
    newFiles,
    oldFiles,
    executable,
}: SwapPlan): PlanEntry[] => {
    // Windows' names do not differ by case: "Logs" is the install's "logs"
    const fold = (name: string) => (getPlatform() === 'win32' ? name.toLowerCase() : name);
    const data = new Set(INSTALL_DATA_NAMES.map(fold));
    for (const name of [...newFiles, ...oldFiles]) {
        if (data.has(fold(name)) || !PLAIN_NAME.test(name)) {
            throw new Error(
                `The release claims the entry "${name}", which an update may not replace.`
            );
        }
    }
    checkPlainName(executable, 'the executable');
    const installed = new Set(installEntries.map(fold));
    const staged = new Set(stagedEntries.map(fold));
    const ownedBefore = new Set(oldFiles.map(fold));
    for (const required of [executable, 'resources', RELEASE_FILES]) {
        if (!newFiles.includes(required)) {
            throw new Error(`The new release does not list ${required} among its files.`);
        }
    }
    for (const name of newFiles) {
        if (!staged.has(fold(name))) {
            throw new Error(`The staged release has no ${name}, which it lists: it is not whole.`);
        }
        if (installed.has(fold(name)) && !ownedBefore.has(fold(name))) {
            throw new Error(
                `The new release adds "${name}", but this install already has an entry of that name that is not part of chaiNNer-C (yours, or left by an install by hand), so the update is not installed. Move "${name}" out of the install folder and run Update now again, or unpack the new release into a new folder by hand.`
            );
        }
    }
    const owned = new Set([...newFiles, ...oldFiles]);
    const last = ['resources', executable];
    const order = [...[...owned].filter((name) => !last.includes(name)).sort(), ...last];
    return order
        .map((name) => ({
            name,
            old: oldFiles.includes(name) && installed.has(fold(name)),
            new: newFiles.includes(name),
        }))
        .filter((entry) => entry.old || entry.new);
};

/**
 * Where an entry stands, from what the install holds: `start` (the old release's, the new one
 * staged), `mid` (the old one moved aside), `end` (the new release's), `broken` (none of them).
 * Both installers decide every rename from this, never from what the journal says was done.
 */
export type EntryState = 'start' | 'mid' | 'end' | 'broken';

export const entryState = (
    entry: PlanEntry,
    inInstall: boolean,
    inOld: boolean,
    inStaged: boolean
): EntryState => {
    const at = (install: boolean, old: boolean, staged: boolean) =>
        inInstall === install && inOld === old && inStaged === staged;
    if (entry.old && entry.new) {
        if (at(true, false, true)) return 'start';
        if (at(false, true, true)) return 'mid';
        if (at(true, true, false)) return 'end';
    } else if (entry.old) {
        if (inInstall && !inOld) return 'start';
        if (!inInstall && inOld) return 'end';
    } else {
        if (!inInstall && inStaged) return 'start';
        if (inInstall && !inStaged) return 'end';
    }
    return 'broken';
};

const JOURNAL_HEADER = 'chaiNNer-C update journal 2';
const STAGED_HEADER = 'chaiNNer-C staged release';

/** What the app writes into the journal before it starts the installer. */
export interface JournalStart {
    app: ProcessMark;
    release: string;
    zip: string;
    result: string;
    /** The install's entry the installer starts afterwards. */
    launch: string;
    executable: string;
    /** The staged release's folder, relative to the install. */
    staged: string;
    entries: readonly PlanEntry[];
}

/** The journal as an installer or a start leaves it: the closed plan, then their lines. */
export interface Journal extends JournalStart {
    /** Why the update is being undone. */
    undo?: string;
    /** Why the last installer gave up. */
    failed?: string;
    /** What the install looked like at each start that resumed the update. */
    resumes: string[];
    /** The new release is whole: only what it replaced is left to remove. */
    swapped?: boolean;
    /** Whether an installer said why it stopped after the last start that resumed the update. */
    failedSinceResume?: boolean;
}

/** The journal: a header, one field per line (tab-separated), the entries, an end marker. */
export const journalText = (start: JournalStart): string => {
    const field = (value: string) => {
        if (hasControl(value)) {
            throw new Error(`An update cannot record "${value}": it holds a control character.`);
        }
        return value;
    };
    const staging = start.staged.split(/[\\/]/);
    if (
        staging[0] !== UPDATE_NEW ||
        staging.length > 2 ||
        (staging.length === 2 && !PLAIN_NAME.test(staging[1]))
    ) {
        throw new Error(`The release is staged in "${start.staged}", not in ${UPDATE_NEW}.`);
    }
    return [
        JOURNAL_HEADER,
        `app\t${start.app.pid}\t${field(start.app.start)}`,
        `release\t${[...start.release].map((char) => (hasControl(char) ? ' ' : char)).join('')}`,
        `zip\t${field(start.zip)}`,
        `result\t${field(start.result)}`,
        `launch\t${checkPlainName(start.launch, 'the program to start')}`,
        `executable\t${checkPlainName(start.executable, 'the executable')}`,
        `staged\t${start.staged}`,
        ...start.entries.map(
            (entry) =>
                `entry\t${checkPlainName(entry.name, 'the entry')}\t${Number(entry.old)}\t${Number(
                    entry.new
                )}`
        ),
        `end\t${start.entries.length}`,
        '',
    ].join('\n');
};

const INCOMPLETE_JOURNAL = "The update's journal is not complete.";

/** The journal; throws when it is not complete (no header, no end marker, or a cut plan). */
export const parseJournal = (text: string): Journal => {
    const lines = text.split(/\r?\n/);
    if (lines[0] !== JOURNAL_HEADER) {
        throw new Error(INCOMPLETE_JOURNAL);
    }
    const values = new Map<string, string[]>();
    const entries: PlanEntry[] = [];
    const journal: Omit<Journal, keyof JournalStart> = { resumes: [] };
    let ended = false;
    for (const line of lines.slice(1)) {
        const [key, ...fields] = line.split('\t');
        if (ended) {
            const [said] = fields;
            if (key === 'undo') journal.undo = said;
            if (key === 'failed') {
                journal.failed = said;
                journal.failedSinceResume = true;
            }
            if (key === 'resume') {
                journal.resumes.push(said);
                journal.failedSinceResume = false;
            }
            if (key === 'swapped') journal.swapped = true;
        } else if (key === 'entry') {
            const [name, old, isNew] = fields;
            if (
                fields.length !== 3 ||
                !PLAIN_NAME.test(name) ||
                !['0', '1'].includes(old) ||
                !['0', '1'].includes(isNew) ||
                old + isNew === '00'
            ) {
                throw new Error(INCOMPLETE_JOURNAL);
            }
            entries.push({ name, old: old === '1', new: isNew === '1' });
        } else if (key === 'end') {
            if (fields[0] !== String(entries.length)) {
                throw new Error(INCOMPLETE_JOURNAL);
            }
            ended = true;
        } else if (fields.length > 0) {
            values.set(key, fields);
        }
    }
    const value = (key: string) => values.get(key)?.[0] ?? '';
    const [pid = '0', start = ''] = values.get('app') ?? [];
    if (!ended || value('staged') === '' || value('executable') === '') {
        throw new Error(INCOMPLETE_JOURNAL);
    }
    return {
        ...journal,
        app: { pid: Number(pid), start },
        release: value('release'),
        zip: value('zip'),
        result: value('result'),
        launch: value('launch'),
        executable: value('executable'),
        staged: value('staged'),
        entries,
    };
};

/** A staged file: its path below the staged release's folder (with `/`), size and SHA-256. */
export interface StagedFile {
    path: string;
    size: number;
    sha256: string;
}

/** What the app staged, and which process staged it. */
export interface StagedList {
    owner: ProcessMark;
    files: readonly StagedFile[];
}

export const stagedListText = ({ owner, files }: StagedList): string => {
    for (const { path: file } of files) {
        const parts = file.split('/');
        if (
            hasControl(file) ||
            /[\\:]/.test(file) ||
            parts.some((part) => part === '' || part === '.' || part === '..')
        ) {
            throw new Error(`The release holds the file "${file}", which an update cannot record.`);
        }
    }
    return [
        STAGED_HEADER,
        `owner\t${owner.pid}\t${owner.start}`,
        ...files.map((file) => `file\t${file.path}\t${file.size}\t${file.sha256}`),
        `end\t${files.length}`,
        '',
    ].join('\n');
};

/** The staged list; throws when it is not complete. */
export const parseStagedList = (text: string): StagedList => {
    const lines = text.split(/\r?\n/).filter((line) => line !== '');
    const [header, owner, ...rest] = lines;
    const end = rest.pop();
    // a list of one line has no owner line
    const [ownerKey, pid, start = ''] = String(owner).split('\t');
    const files = rest.map((line) => {
        const [key, file, size, sha256] = line.split('\t');
        if (key !== 'file' || !/^\d+$/.test(size) || !/^[\da-f]{64}$/.test(sha256)) {
            throw new Error("The list of the staged release's files is not complete.");
        }
        return { path: file, size: Number(size), sha256 };
    });
    if (header !== STAGED_HEADER || ownerKey !== 'owner' || end !== `end\t${files.length}`) {
        throw new Error("The list of the staged release's files is not complete.");
    }
    return { owner: { pid: Number(pid), start }, files };
};

/** The install's staged list, or undefined (logged) when there is none or it is not whole. */
const readStagedList = (target: string): StagedList | undefined => {
    try {
        return parseStagedList(readFileSync(path.join(target, STAGED_LIST), 'utf-8'));
    } catch (error) {
        log.warn(`The list of the staged update cannot be read: ${String(error)}`);
        return undefined;
    }
};

/** Writes a file so that it is whole on disk or not there: a temporary file, flushed, renamed. */
const writeDurably = async (file: string, text: string): Promise<void> => {
    const temporary = `${file}.tmp`;
    const handle = await fs.open(temporary, 'w');
    try {
        await handle.writeFile(text, 'utf-8');
        await handle.sync();
    } finally {
        await handle.close();
    }
    await fs.rename(temporary, file);
};

/** The updates this process staged and did not discard, by unpacked folder: their zip and folder go when it quits. */
const stagedUpdates = new Map<string, string>();

/** A release downloaded, checked and unpacked into the install, ready to be swapped in. */
export interface StagedUpdate {
    release: string;
    zip: string;
    unpacked: string;
    root: string;
}

const fetchLatestRelease = async (): Promise<Release> => {
    const response = await net.fetch(`https://api.github.com/repos/${REPOSITORY}/releases/latest`, {
        headers: { Accept: 'application/vnd.github+json', 'User-Agent': 'chaiNNer-C' },
    });
    if (!response.ok) {
        throw new Error(`GitHub answered ${response.status} ${response.statusText}.`);
    }
    return (await response.json()) as Release;
};

/**
 * Rejects when this process cannot create a file in the install: the release could be neither
 * unpacked there nor swapped in (e.g. an install in Program Files).
 */
const checkInstallWritable = async (target: string): Promise<void> => {
    const probe = path.join(target, `chaiNNer-C-update-check-${randomUUID()}`);
    try {
        await fs.writeFile(probe, '');
    } catch (error) {
        throw new Error(
            `chaiNNer-C cannot write to ${target}, so it cannot update itself there (${String(
                error
            )}). Install the new release by hand, or move chaiNNer-C to a folder you can write to.`
        );
    }
    await fs.rm(probe, { force: true });
};

/**
 * Unpacks a release's zip into `folder`, one file after another: a write that fails stops it, and
 * nothing is written after the caller removes the folder (decompress writes all files at once and
 * keeps writing after the first failure). Only files and folders inside `folder` are written.
 */
const unpackRelease = async (zip: string, folder: string): Promise<void> => {
    const files = await decompress(zip);
    await files.reduce(async (previous, file) => {
        await previous;
        const destination = path.resolve(folder, file.path);
        const inside = path.relative(folder, destination);
        if (inside === '' || inside.startsWith('..') || path.isAbsolute(inside)) {
            throw new Error(`The release's zip holds "${file.path}", which is outside its folder.`);
        }
        if (file.type === 'directory') {
            await fs.mkdir(destination, { recursive: true });
        } else if (file.type === 'file') {
            await fs.mkdir(path.dirname(destination), { recursive: true });
            // the permission bits; an entry without them (a zip made on Windows) gets the usual ones
            await fs.writeFile(destination, file.data, { mode: file.mode % 0o1000 || 0o666 });
        } else {
            throw new Error(
                `The release's zip holds the ${file.type} "${file.path}": only files are unpacked.`
            );
        }
    }, Promise.resolve());
};

/** Every file below `root`, with its size and SHA-256, by its path below `root` with `/`, in order. */
const listFiles = async (root: string): Promise<StagedFile[]> => {
    const entries = await fs.readdir(root, { recursive: true, withFileTypes: true });
    const files = entries
        .filter((entry) => entry.isFile())
        .map((entry) => path.join(entry.parentPath, entry.name))
        .sort();
    // one file after another: a release has hundreds
    return files.reduce<Promise<StagedFile[]>>(async (previous, full) => {
        const listed = await previous;
        listed.push({
            path: path.relative(root, full).split(path.sep).join('/'),
            size: (await fs.stat(full)).size,
            sha256: await fileSha256(full),
        });
        return listed;
    }, Promise.resolve([]));
};

/** The files of the top-level entry `name` below `base`, as the records hold them (`name/...`). */
const listEntry = async (base: string, name: string): Promise<StagedFile[]> => {
    const full = path.join(base, name);
    if ((await fs.stat(full)).isDirectory()) {
        return (await listFiles(full)).map((file) => ({ ...file, path: `${name}/${file.path}` }));
    }
    return [{ path: name, size: (await fs.stat(full)).size, sha256: await fileSha256(full) }];
};

/** The program and arguments that run an installer script the way the app starts it. */
const installerCommand = (script: string, waitFor?: number): [string, string[]] =>
    getPlatform() === 'win32'
        ? [
              'powershell.exe',
              [
                  '-NoProfile',
                  '-NonInteractive',
                  '-ExecutionPolicy',
                  'Bypass',
                  '-WindowStyle',
                  'Hidden',
                  '-File',
                  script,
                  ...(waitFor === undefined ? [] : ['-ProcessId', String(waitFor)]),
              ],
          ]
        : ['/bin/sh', [script, ...(waitFor === undefined ? [] : [String(waitFor)])]];

/** What the installer needs and a probe tries, in the install: a lock, the journal's reads and appends, a rename, the process list. */
export const WINDOWS_PROBE = String.raw`$ErrorActionPreference = 'Stop'
$folder = Join-Path $PSScriptRoot 'update-probe'
[void][IO.Directory]::CreateDirectory($folder)
$lock = [IO.File]::Open((Join-Path $folder 'lock'), 'OpenOrCreate', 'ReadWrite', 'None')
[IO.File]::WriteAllText((Join-Path $folder 'a'), 'probe')
Add-Content -LiteralPath (Join-Path $folder 'a') -Value 'line' -Encoding UTF8
[void][IO.File]::ReadAllLines((Join-Path $folder 'a'))
[IO.Directory]::Move((Join-Path $folder 'a'), (Join-Path $folder 'b'))
[void](New-Object IO.FileInfo (Join-Path $folder 'b')).Length
[void]@(Get-CimInstance Win32_Process -ErrorAction Stop)
[void][Diagnostics.Process]::GetCurrentProcess().StartTime.ToUniversalTime().Ticks
$lock.Close()
Remove-Item -LiteralPath $folder -Recurse -Force
'ready'
`;

export const LINUX_PROBE = String.raw`#!/bin/sh
set -e
folder="$(cd "$(dirname "$0")" && pwd)/update-probe"
mkdir -p "$folder"
exec 9>"$folder/lock"
flock -n 9
printf 'probe\n' > "$folder/a"
mv -T "$folder/a" "$folder/b"
wc -c < "$folder/b" > /dev/null
grep -c probe "$folder/b" > /dev/null
cut -d ' ' -f 22 "/proc/$$/stat" > /dev/null
exec 9>&-
rm -rf "$folder"
echo ready
`;

/**
 * Rejects, saying why, when the installer cannot run on this PC (e.g. a policy that stops
 * PowerShell scripts, or no flock): the probe runs as the installer would, from the install.
 */
const probeInstaller = async (target: string): Promise<void> => {
    const windows = getPlatform() === 'win32';
    const script = path.join(target, windows ? 'update-probe.ps1' : 'update-probe.sh');
    await fs.writeFile(script, windows ? WINDOWS_PROBE : LINUX_PROBE, { mode: 0o755 });
    try {
        const [program, args] = installerCommand(script);
        const answer = await new Promise<{ failure?: Error; output: string }>((resolve) => {
            execFile(
                program,
                args,
                { encoding: 'utf-8', windowsHide: true, timeout: 60_000 },
                (failure, stdout, stderr) =>
                    resolve({ failure: failure ?? undefined, output: `${stdout}${stderr}` })
            );
        });
        if (answer.failure || answer.output.trim() !== 'ready') {
            throw new Error(
                `chaiNNer-C cannot update itself on this PC: its installer (${program}) cannot run here (${
                    answer.failure?.message ?? answer.output.trim()
                }). Install the new release by hand.`
            );
        }
    } finally {
        await fs.rm(script, { force: true });
        await fs.rm(path.join(target, PROBE), { recursive: true, force: true });
    }
};

/** The entries that swap the staged update into the install `target`. */
const planUpdate = async (update: StagedUpdate, target: string): Promise<PlanEntry[]> => {
    const readList = async (folder: string): Promise<string[]> => {
        const file = path.join(folder, RELEASE_FILES);
        if (!existsSync(file)) {
            throw new Error(
                `${folder} has no list of its release's files (${RELEASE_FILES}): the update is not installed. Install the new release by hand.`
            );
        }
        return parseReleaseFiles(await readTextFile(file));
    };
    return planSwap({
        installEntries: await fs.readdir(target),
        stagedEntries: await fs.readdir(update.root),
        newFiles: await readList(update.root),
        oldFiles: await readList(target),
        executable: path.basename(process.execPath),
    });
};

/**
 * Downloads the latest release's zip for this platform into the Downloads folder, checks it
 * against the SHA-256 GitHub reports, unpacks it into the install's UPDATE_NEW folder (on the
 * install's drive, so the swap only renames), lists every staged file, and checks that the
 * installer can run.
 */
export const stageUpdate = async (
    onProgress: (percentage: number) => void
): Promise<StagedUpdate> => {
    if (!app.isPackaged) {
        throw new Error('Only a packaged chaiNNer-C can update itself.');
    }
    const target = currentExecutableDir;
    await checkInstallWritable(target);
    if (existsSync(path.join(target, JOURNAL))) {
        throw new Error('An update of this install is already waiting for chaiNNer-C to close.');
    }
    if (!existsSync(path.join(target, RELEASE_FILES))) {
        throw new Error(
            `This chaiNNer-C has no list of the files it owns (${RELEASE_FILES}), so it cannot update itself safely. Unpack the new release over its folder by hand.`
        );
    }
    const listFile = path.join(target, STAGED_LIST);
    const owner = existsSync(listFile) ? readStagedList(target)?.owner : undefined;
    if (owner && owner.pid !== process.pid && isRunning(owner)) {
        throw new Error('Another chaiNNer-C is preparing an update of this install.');
    }
    const release = await fetchLatestRelease();
    const asset = findReleaseAsset(release, getPlatform(), os.arch());
    const sha256 = expectedSha256(asset);

    // a name of its own: a file of the asset's name the user keeps in Downloads is left alone
    const zip = path.join(
        app.getPath('downloads'),
        `${asset.name.replace(/\.zip$/i, '')}-update-${randomUUID()}.zip`
    );
    const partial = `${zip}.part`;
    await download(asset.browser_download_url, partial, (percentage) =>
        onProgress(percentage * 0.9)
    );
    const actual = await fileSha256(partial);
    if (actual !== sha256) {
        await fs.rm(partial, { force: true });
        throw new Error(`${asset.name} does not match GitHub's SHA-256: the download is damaged.`);
    }
    await fs.rename(partial, zip);

    const unpacked = path.join(target, UPDATE_NEW);
    try {
        // what a crashed staging left; the list says which process stages
        await fs.rm(unpacked, { recursive: true, force: true });
        await writeDurably(listFile, stagedListText({ owner: ownMark(), files: [] }));
        await fs.mkdir(unpacked);
        stagedUpdates.set(unpacked, zip);
        await unpackRelease(zip, unpacked);
        const root = await findReleaseRoot(unpacked, path.basename(process.execPath));
        const releaseFiles = path.join(root, RELEASE_FILES);
        if (!existsSync(releaseFiles)) {
            throw new Error(`The release has no ${RELEASE_FILES}: it cannot be installed.`);
        }
        // the files of the entries the release owns (not the zip's portable marker); last: a
        // whole list means a whole staged release
        const owned = parseReleaseFiles(await readTextFile(releaseFiles));
        const files = (await listFiles(root)).filter((file) =>
            owned.includes(file.path.split('/')[0])
        );
        await writeDurably(listFile, stagedListText({ owner: ownMark(), files }));
        await probeInstaller(target);
        const update = { release: release.name ?? release.tag_name, zip, unpacked, root };
        // what the install would refuse (an entry of the user's in the way), said now, in the app
        await planUpdate(update, target);
        onProgress(100);
        return update;
    } catch (error) {
        await fs.rm(unpacked, { recursive: true, force: true });
        await fs.rm(listFile, { force: true });
        stagedUpdates.delete(unpacked);
        await fs.rm(zip, { force: true });
        throw error;
    }
};

/** Removes a staged update that will not be installed. */
export const discardUpdate = async (update: StagedUpdate): Promise<void> => {
    await fs.rm(update.unpacked, { recursive: true, force: true });
    await fs.rm(path.join(path.dirname(update.unpacked), STAGED_LIST), { force: true });
    stagedUpdates.delete(update.unpacked);
    await fs.rm(update.zip, { force: true });
};

/**
 * Swaps a staged release into the install once every process run from the install has ended.
 * One installer at a time: it holds LOCK, which the system releases when it ends, and a second
 * one leaves at once. It reads the journal's closed plan (header to end marker), checks every
 * name in it and every staged file against STAGED_LIST, then takes each entry to the new
 * release's state, deciding each rename from what the install holds, so a run cut short (a kill,
 * a power loss) is resumed exactly. A rename refused for 5 seconds, or anything not as planned,
 * turns it to undo (recorded): every entry back to the old release's state, in reverse. Once the
 * install holds one whole release, the replaced or unused entries, the journal and its files go,
 * the zip too after a successful update, the result goes to the journal's `result` as JSON, and
 * the app starts again. When it can neither finish nor undo, it records why (`failed`) and leaves
 * the journal for the next start or a run by hand; when nothing was changed (the app did not
 * close, a plan that is not whole), the update is dropped and the zip is kept for an install by
 * hand.
 */
export const WINDOWS_INSTALLER = String.raw`param([int]$ProcessId = 0)
$ErrorActionPreference = 'Stop'
$install = $PSScriptRoot
$journal = Join-Path $install 'update-journal.txt'
try {
    $lock = [IO.File]::Open((Join-Path $install 'update.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
} catch {
    # in use: another installer runs (0x20, a sharing violation); else the lock cannot be taken
    if (($_.Exception.InnerException.HResult -band 0xFFFF) -eq 0x20) {
        Write-Host 'Another installer is finishing this update; this one changed nothing.'
        exit 0
    }
    $reason = "update.lock cannot be opened ($($_.Exception.Message -replace '\s+', ' ')): delete it, then run this again."
    Write-Host $reason
    if ([IO.File]::Exists($journal)) { Add-Content -LiteralPath $journal -Value ("failed" + [char]9 + $reason) -Encoding UTF8 }
    exit 1
}
# another installer finished the update
if (-not [IO.File]::Exists($journal)) {
    $lock.Close()
    exit 0
}
# The app checks, by digest, that the install is in a state the update itself made before it
# starts an installer, and says so as the journal's last line; a run by hand after anything else
# (an installer cut short, the install changed) would decide by sizes alone, so it does nothing.
$last = @([IO.File]::ReadAllLines($journal))
if ($last.Count -eq 0 -or $last[-1] -cne 'checked') {
    Write-Host 'Start chaiNNer-C to finish this update: it checks the install first. Nothing was changed.'
    $lock.Close()
    exit 0
}
# the check is used up at once: a run after this one, however this one ends, needs a new check
Add-Content -LiteralPath $journal -Value ('installer' + [char]9 + $PID) -Encoding UTF8
$tab = [string][char]9
$name = '[A-Za-z0-9][A-Za-z0-9._+-]*'
function Add-Journal([string]$Line) { Add-Content -LiteralPath $journal -Value $Line -Encoding UTF8 }
function Format-Line([string]$Text) { ($Text -replace '\s+', ' ').Trim() }
function Test-Entry([string]$Path) { [IO.File]::Exists($Path) -or [IO.Directory]::Exists($Path) }
# a failure is reported in the result
function Remove-Leftover([string]$Path) {
    try {
        if (Test-Entry $Path) { Remove-Item -LiteralPath $Path -Recurse -Force }
    } catch {
        $script:message = ("$script:message $Path could not be removed: " + $_.Exception.Message).Trim()
    }
}
function Move-Entry([string]$From, [string]$To) {
    $source = Join-Path $install $From
    $destination = Join-Path $install $To
    if (Test-Entry $destination) { throw "$($From) cannot be moved to $($To), which is there already." }
    for ($try = 1; ; $try++) {
        try {
            # one rename, of a file or a folder; not Move-Item, which moves a folder it may not
            # rename file by file and stops halfway
            [IO.Directory]::Move($source, $destination)
            break
        } catch {
            if ($try -ge 10) { throw "$($From) could not be moved to $($To): $($_.Exception.Message)" }
            Start-Sleep -Milliseconds 500
        }
    }
    # for whoever reads the journal: resuming reads the install, not these lines
    Add-Journal "moved$tab$From$tab$To"
}
function Get-State($Entry) {
    $i = Test-Entry (Join-Path $install $Entry.Name)
    $o = Test-Entry (Join-Path $install (Join-Path '.update-old' $Entry.Name))
    $n = Test-Entry (Join-Path $install (Join-Path $staged $Entry.Name))
    if ($Entry.Old -and $Entry.New) {
        if ($i -and -not $o -and $n) { return 'start' }
        if (-not $i -and $o -and $n) { return 'mid' }
        if ($i -and $o -and -not $n) { return 'end' }
    } elseif ($Entry.Old) {
        if ($i -and -not $o) { return 'start' }
        if (-not $i -and $o) { return 'end' }
    } else {
        if (-not $i -and $n) { return 'start' }
        if ($i -and -not $n) { return 'end' }
    }
    return 'broken'
}
function Step-Forward($Entry) {
    $state = Get-State $Entry
    if ($state -eq 'start' -and $Entry.Old) {
        Move-Entry $Entry.Name (Join-Path '.update-old' $Entry.Name)
        $state = if ($Entry.New) { 'mid' } else { 'end' }
    }
    if ($state -eq 'mid' -or ($state -eq 'start' -and -not $Entry.Old)) {
        Move-Entry (Join-Path $staged $Entry.Name) $Entry.Name
        $state = 'end'
    }
    if ($state -ne 'end') { throw "$($Entry.Name) is neither the old release's nor the new one's." }
}
# the entry as the old release had it, whether or not its new one is still staged
function Test-OldPlace($Entry) {
    $i = Test-Entry (Join-Path $install $Entry.Name)
    if ($Entry.Old) { return $i -and -not (Test-Entry (Join-Path $install (Join-Path '.update-old' $Entry.Name))) }
    return -not $i
}
function Step-Back($Entry) {
    if (Test-OldPlace $Entry) { return }
    $state = Get-State $Entry
    if ($state -eq 'end' -and $Entry.New) {
        Move-Entry $Entry.Name (Join-Path $staged $Entry.Name)
        $state = if ($Entry.Old) { 'mid' } else { 'start' }
    }
    if ($state -eq 'mid' -or ($state -eq 'end' -and -not $Entry.New)) {
        Move-Entry (Join-Path '.update-old' $Entry.Name) $Entry.Name
    }
    if (-not (Test-OldPlace $Entry)) { throw "$($Entry.Name) is neither the new release's nor the old one's." }
}
# every staged file there and whole, where its entry stands, and the release's own three
function Test-Staged {
    $incomplete = "The list of the staged release's files is not complete."
    $lines = [IO.File]::ReadAllLines((Join-Path $install 'update-staged.txt'))
    if ($lines.Count -lt 3 -or $lines[0] -cne 'chaiNNer-C staged release' -or $lines[-1] -cne "end$tab$($lines.Count - 3)") { throw $incomplete }
    $byName = @{}
    foreach ($entry in $entries) { $byName[$entry.Name] = $entry }
    $states = @{}
    $found = @{ 'release-files.txt' = $false; 'resources' = $false; $executable = $false }
    for ($k = 2; $k -lt $lines.Count - 1; $k++) {
        $fields = $lines[$k].Split($tab)
        if ($fields.Count -ne 4 -or $fields[0] -cne 'file' -or $fields[2] -notmatch '^\d+$') { throw $incomplete }
        $parts = $fields[1].Split('/')
        if (@($parts | Where-Object { $_ -in '', '.', '..' -or $_ -match '[\\:]' }).Count -gt 0) {
            throw "The staged release names the file '$($fields[1])', which is outside it."
        }
        $entry = $byName[$parts[0]]
        if (-not $entry -or -not $entry.New) { throw "The staged file $($fields[1]) belongs to no entry of the update." }
        if (-not $states.ContainsKey($entry.Name)) { $states[$entry.Name] = Get-State $entry }
        $base = if ($states[$entry.Name] -eq 'end') { $install } else { Join-Path $install $staged }
        $file = New-Object IO.FileInfo (Join-Path $base $fields[1])
        if (-not $file.Exists -or $file.Length -ne [long]$fields[2]) {
            throw "The staged release is incomplete: $($fields[1]) is missing or not whole."
        }
        if ($found.ContainsKey($entry.Name)) { $found[$entry.Name] = $true }
    }
    foreach ($key in @($found.Keys)) { if (-not $found[$key]) { throw "The staged release has no $key." } }
}

$entries = @()
$swapped = $false
$staged = '.update-new'
$executable = ''
$zip = ''
$result = ''
$undo = ''
$closed = $false
$resolved = $false
$ok = $false
$message = ''
try {
    # the journal is a file on disk: nothing it names is used before it is checked, and a plan
    # without its end marker is not one
    $incomplete = "The update's journal is not complete."
    $lines = [IO.File]::ReadAllLines($journal)
    if ($lines.Count -eq 0 -or $lines[0] -cne 'chaiNNer-C update journal 2') { throw $incomplete }
    $values = @{}
    $ended = $false
    foreach ($line in $lines) {
        $fields = $line.Split($tab)
        if ($ended) {
            if ($fields[0] -ceq 'undo') { $undo = $fields[1] }
            if ($fields[0] -ceq 'swapped') { $swapped = $true }
        } elseif ($fields[0] -ceq 'entry') {
            if ($fields.Count -ne 4 -or $fields[1] -cnotmatch "^$name$" -or $fields[2] -notin '0', '1' -or $fields[3] -notin '0', '1' -or ($fields[2] -eq '0' -and $fields[3] -eq '0')) {
                throw $incomplete
            }
            $entries += [pscustomobject]@{ Name = $fields[1]; Old = $fields[2] -eq '1'; New = $fields[3] -eq '1' }
        } elseif ($fields[0] -ceq 'end') {
            if ($fields.Count -ne 2 -or $fields[1] -cne [string]$entries.Count) { throw $incomplete }
            $ended = $true
        } elseif ($fields.Count -ge 2) {
            $values[$fields[0]] = $fields[1]
        }
    }
    if (-not $ended) { throw $incomplete }
    if (-not ([IO.Path]::IsPathRooted($values['result']) -and [IO.Path]::GetFileName($values['result']) -ceq 'update-result.json')) {
        throw "The update's journal names the result '$($values['result'])', which is not the app's result file."
    }
    $result = $values['result']
    if (-not ([IO.Path]::IsPathRooted($values['zip']) -and ([string]$values['zip']).EndsWith('.zip', [StringComparison]::OrdinalIgnoreCase))) {
        throw "The update's journal names the zip '$($values['zip'])', which is not a zip's full path."
    }
    $zip = $values['zip']
    foreach ($key in 'launch', 'executable') {
        if ([string]$values[$key] -cnotmatch "^$name$") { throw "The update's journal names the $key '$($values[$key])', which is not an entry of the install." }
    }
    $executable = $values['executable']
    if ([string]$values['staged'] -cnotmatch "^\.update-new(?:[\\/]$name)?$") {
        throw "The update's journal names the staged release '$($values['staged'])', which is not in .update-new."
    }
    $staged = $values['staged']
    $deadline = (Get-Date).AddMinutes(10)
    $prefix = $install.TrimEnd('\') + '\'
    while ($true) {
        # the app, every process run from the install, and every process running its resources
        # (the backend's workers, whose Python may live elsewhere); not this one. A process list
        # that cannot be read counts as the app still running.
        $running = $true
        $queryError = ''
        try {
            $running = @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object {
                $_.ProcessId -ne $PID -and (
                    ($ProcessId -gt 0 -and $_.ProcessId -eq $ProcessId) -or
                    ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) -or
                    ($_.CommandLine -and $_.CommandLine.IndexOf($prefix + 'resources', [StringComparison]::OrdinalIgnoreCase) -ge 0)
                )
            }).Count -gt 0
        } catch {
            $queryError = $_.Exception.Message
        }
        if (-not $running) { break }
        if ((Get-Date) -gt $deadline) {
            $reason = 'chaiNNer-C did not close within 10 minutes.'
            if ($queryError) { $reason += " The list of processes could not be read: $queryError" }
            throw $reason
        }
        Start-Sleep -Milliseconds 500
    }
    $closed = $true
    if ($swapped) {
        # the new release is whole: only what it replaced is left to remove
        $ok = $true
    } elseif (-not $undo) {
        try {
            Test-Staged
            [void][IO.Directory]::CreateDirectory((Join-Path $install '.update-old'))
            foreach ($entry in $entries) { Step-Forward $entry }
            $ok = $true
        } catch {
            $undo = Format-Line $_.Exception.Message
            Add-Journal "undo$tab$undo"
        }
    }
    if ($ok -and -not $swapped) {
        # from here on, a run of this journal only removes what the update replaced
        Add-Journal 'swapped'
        $swapped = $true
    }
    if ($undo) {
        for ($k = $entries.Count - 1; $k -ge 0; $k--) { Step-Back $entries[$k] }
        $message = "The update was undone: $undo"
    }
    $resolved = $true
} catch {
    $message = $_.Exception.Message
}
# What a whole update replaced is only ever removed: .update-old becomes .update-done (renamed,
# retried as every rename), or, when a file held open inside (e.g. a virus scan of the old exe)
# keeps it from being renamed, it is marked so and left for the next start's removal. Whether it
# was left.
function Set-Replaced {
    $old = Join-Path $install '.update-old'
    $done = Join-Path $install '.update-done'
    if (-not (Test-Entry $old)) { return $false }
    if (-not (Test-Entry $done)) {
        for ($try = 1; $try -le 10; $try++) {
            try {
                [IO.Directory]::Move($old, $done)
                return $false
            } catch {
                Start-Sleep -Milliseconds 500
            }
        }
    }
    [IO.File]::WriteAllText((Join-Path $old '.replaced'), '')
    $script:message = ("$script:message What the update replaced stays in $old until the next start removes it.").Trim()
    return $true
}
# nothing was changed: every entry as the old release had it, and nothing moved aside
$old = Join-Path $install '.update-old'
$untouched = -not $swapped -and (-not [IO.Directory]::Exists($old) -or @([IO.Directory]::GetFileSystemEntries($old)).Count -eq 0)
foreach ($entry in $entries) { if (-not (Test-OldPlace $entry)) { $untouched = $false } }
$aborted = -not $resolved -and $untouched
if ($resolved) {
    if (-not ($ok -and (Set-Replaced))) { Remove-Leftover $old }
    Remove-Leftover (Join-Path $install '.update-done')
    Remove-Leftover (Join-Path $install '.update-new')
    if ($ok) { Remove-Leftover $zip }
} elseif ($aborted) {
    $message = "$message Nothing was changed."
    if ($zip) { $message += " The downloaded release stays at $zip for an install by hand." }
    Remove-Leftover (Join-Path $install '.update-new')
} else {
    Add-Journal "failed$tab$(Format-Line $message)"
}
if ($result) {
    $json = @{ ok = $ok; release = $values['release']; message = $message; zip = $zip } | ConvertTo-Json -Compress
    [IO.File]::WriteAllText($result, $json)
}
if ($resolved -or $aborted) {
    Remove-Leftover (Join-Path $install 'update-not-finished.txt')
    Remove-Leftover (Join-Path $install 'finish-update.cmd')
    Remove-Leftover (Join-Path $install 'update-staged.txt')
    Remove-Leftover (Join-Path $install 'update-installed.txt')
    Remove-Leftover $journal
    $lock.Close()
    # an app that did not close still runs: it is not started a second time
    if ($resolved -and $values['launch']) {
        # not Start-Process, which reads its paths as wildcards: a [ in the install's path fails
        # it. Started directly with no console window: the app's own window is unaffected, and a
        # console program never flashes one.
        $start = New-Object Diagnostics.ProcessStartInfo (Join-Path $install $values['launch'])
        $start.WorkingDirectory = $install
        $start.UseShellExecute = $false
        $start.CreateNoWindow = $true
        [void][Diagnostics.Process]::Start($start)
    }
    Remove-Leftover (Join-Path $install 'update.lock')
    Remove-Leftover $PSCommandPath
}
`;

/** WINDOWS_INSTALLER's steps in POSIX shell, for Linux; the process to wait for is its argument. */
export const LINUX_INSTALLER = String.raw`#!/bin/sh
pid=$1
case "$pid" in '' | *[!0-9]*) pid=0 ;; esac
install=$(cd "$(dirname "$0")" && pwd)
journal="$install/update-journal.txt"
exec 9>>"$install/update.lock" || {
    reason="update.lock cannot be opened: delete it, then run this again."
    echo "$reason" >&2
    [ -f "$journal" ] && printf 'failed\t%s\n' "$reason" >> "$journal"
    exit 1
}
flock -n 9 || { echo 'Another installer is finishing this update; this one changed nothing.' >&2; exit 0; }
# another installer finished the update
[ -f "$journal" ] || exit 0
# The app checks, by digest, that the install is in a state the update itself made before it
# starts an installer, and says so as the journal's last line; a run by hand after anything else
# (an installer cut short, the install changed) would decide by sizes alone, so it does nothing.
if [ "$(tail -n 1 "$journal")" != checked ]; then
    echo 'Start chaiNNer-C to finish this update: it checks the install first. Nothing was changed.' >&2
    exit 0
fi
# the check is used up at once: a run after this one, however this one ends, needs a new check
printf 'installer\t%s\n' "$$" >> "$journal"
tab=$(printf '\t')
name='[A-Za-z0-9][A-Za-z0-9._+-]*'
plain() { printf '%s\n' "$1" | grep -Eqx "$name"; }
line() { printf '%s' "$1" | tr '\t\r\n' '   '; }
exists() { [ -e "$1" ] || [ -L "$1" ]; }
# the closed plan: the journal up to its end marker
plan() { sed -n "1,/^end$tab/p" "$journal"; }
value() { plan | grep "^$1$tab" | tail -n 1 | cut -f 2; }
entries() { plan | grep "^entry$tab"; }
# a failure is reported in the result
remove_leftover() {
    if exists "$1" && ! out=$(rm -rf "$1" 2>&1); then message="$message $1 could not be removed: $(line "$out")"; fi
}
move_entry() {
    if exists "$install/$2"; then error="$1 cannot be moved to $2, which is there already."; return 1; fi
    tries=1
    # -T: never into a folder that is there
    until out=$(mv -T "$install/$1" "$install/$2" 2>&1); do
        if [ "$tries" -ge 10 ]; then error="$1 could not be moved to $2: $(line "$out")"; return 1; fi
        sleep 0.5; tries=$((tries + 1))
    done
    # for whoever reads the journal: resuming reads the install, not these lines
    printf 'moved\t%s\t%s\n' "$1" "$2" >> "$journal"
}
# where entry $1 (old $2, new $3) stands: start, mid, end or broken
state() {
    i=0; o=0; n=0
    exists "$install/$1" && i=1
    exists "$install/.update-old/$1" && o=1
    exists "$install/$staged/$1" && n=1
    case "$2$3" in
        11) case "$i$o$n" in 101) echo start ;; 011) echo mid ;; 110) echo end ;; *) echo broken ;; esac ;;
        10) case "$i$o" in 10) echo start ;; 01) echo end ;; *) echo broken ;; esac ;;
        *) case "$i$n" in 01) echo start ;; 10) echo end ;; *) echo broken ;; esac ;;
    esac
}
forward() {
    s=$(state "$1" "$2" "$3")
    if [ "$s" = start ] && [ "$2" = 1 ]; then
        move_entry "$1" ".update-old/$1" || return 1
        if [ "$3" = 1 ]; then s=mid; else s=end; fi
    fi
    if [ "$s" = mid ] || { [ "$s" = start ] && [ "$2" = 0 ]; }; then
        move_entry "$staged/$1" "$1" || return 1
        s=end
    fi
    [ "$s" = end ] || { error="$1 is neither the old release's nor the new one's."; return 1; }
}
# entry $1 (old $2) as the old release had it, whether or not its new one is still staged
old_place() {
    if [ "$2" = 1 ]; then exists "$install/$1" && ! exists "$install/.update-old/$1"; else ! exists "$install/$1"; fi
}
back() {
    old_place "$1" "$2" && return 0
    s=$(state "$1" "$2" "$3")
    if [ "$s" = end ] && [ "$3" = 1 ]; then
        move_entry "$1" "$staged/$1" || return 1
        if [ "$2" = 1 ]; then s=mid; else s=start; fi
    fi
    if [ "$s" = mid ] || { [ "$s" = end ] && [ "$3" = 0 ]; }; then
        move_entry ".update-old/$1" "$1" || return 1
    fi
    old_place "$1" "$2" || { error="$1 is neither the new release's nor the old one's."; return 1; }
}
# every entry, in the plan's order (or, with "back", the reverse), through $1
each() {
    if [ "$2" = back ]; then list=$(entries | sed -n '1!G;h;$p'); else list=$(entries); fi
    while IFS="$tab" read -r key e o n; do
        [ -n "$key" ] || continue
        "$1" "$e" "$o" "$n" || return 1
    done <<EOF
$list
EOF
}
# the journal is a file on disk: nothing it names is used before it is checked, and a plan
# without its end marker is not one
check_journal() {
    error="The update's journal is not complete."
    [ "$(head -n 1 "$journal")" = 'chaiNNer-C update journal 2' ] || return 1
    [ "$(plan | grep -c "^end$tab")" = 1 ] || return 1
    [ "$(plan | tail -n 1 | cut -f 2)" = "$(entries | grep -c .)" ] || return 1
    list=$(entries)
    while IFS="$tab" read -r key e o n rest; do
        [ -n "$key" ] || continue
        plain "$e" && [ -z "$rest" ] || return 1
        case "$o$n" in 01 | 10 | 11) ;; *) return 1 ;; esac
    done <<EOF
$list
EOF
    result=$(value result)
    case "$result" in /*/update-result.json) ;; *) error="The update's journal names the result '$result', which is not the app's result file."; result=""; return 1 ;; esac
    zip=$(value zip)
    case "$zip" in /*.zip) ;; *) error="The update's journal names the zip '$zip', which is not a zip's full path."; zip=""; return 1 ;; esac
    launch=$(value launch); executable=$(value executable)
    plain "$launch" || { error="The update's journal names the launch '$launch', which is not an entry of the install."; return 1; }
    plain "$executable" || { error="The update's journal names the executable '$executable', which is not an entry of the install."; return 1; }
    staged=$(value staged)
    if [ "$staged" != .update-new ] && ! printf '%s\n' "$staged" | grep -Eqx "\.update-new/$name"; then
        error="The update's journal names the staged release '$staged', which is not in .update-new."; staged=.update-new; return 1
    fi
}
# every staged file there and whole, where its entry stands, and the release's own three
check_staged() {
    stagedlist="$install/update-staged.txt"
    error="The list of the staged release's files is not complete."
    total=$(grep -c . "$stagedlist" 2>/dev/null) || return 1
    [ "$(head -n 1 "$stagedlist")" = 'chaiNNer-C staged release' ] || return 1
    [ "$(tail -n 1 "$stagedlist")" = "end$tab$((total - 3))" ] || return 1
    has_files=0; has_resources=0; has_executable=0; last=""
    files=$(grep "^file$tab" "$stagedlist")
    while IFS="$tab" read -r key rel size hash; do
        [ -n "$key" ] || continue
        case "$size" in '' | *[!0-9]*) return 1 ;; esac
        [ -n "$hash" ] || return 1
        case "/$rel/" in */../* | */./* | *//* | *\\*) error="The staged release names the file '$rel', which is outside it."; return 1 ;; esac
        top=$(printf '%s\n' "$rel" | cut -d / -f 1)
        if [ "$top" != "$last" ]; then
            entry=$(entries | awk -F "$tab" -v top="$top" '$2 == top')
            [ -n "$entry" ] && [ "$(printf '%s\n' "$entry" | cut -f 4)" = 1 ] || { error="The staged file $rel belongs to no entry of the update."; return 1; }
            if [ "$(state "$top" "$(printf '%s\n' "$entry" | cut -f 3)" 1)" = end ]; then base=$install; else base="$install/$staged"; fi
            last=$top
        fi
        [ -f "$base/$rel" ] && found=$(wc -c < "$base/$rel") && [ "$((found))" -eq "$size" ] || { error="The staged release is incomplete: $rel is missing or not whole."; return 1; }
        [ "$rel" = release-files.txt ] && has_files=1
        [ "$top" = resources ] && has_resources=1
        [ "$rel" = "$executable" ] && has_executable=1
    done <<EOF
$files
EOF
    [ "$has_files$has_resources$has_executable" = 111 ] || { error="The staged release has no release-files.txt, resources or $executable."; return 1; }
}
# the app, every process run from the install, and every process running its resources (the
# backend's workers, whose Python may live elsewhere); not this one
running() {
    # no process id (a run by hand): 0 would name this process group
    [ "$pid" -gt 0 ] && kill -0 "$pid" 2>/dev/null && return 0
    for proc in /proc/[0-9]*; do
        [ "$proc" = "/proc/$$" ] && continue
        case "$(readlink "$proc/exe" 2>/dev/null)" in "$install"/*) return 0 ;; esac
        tr '\0' ' ' 2>/dev/null < "$proc/cmdline" | grep -qF "$install/resources/" && return 0
    done
    return 1
}
message=""; closed=false; resolved=false; ok=false; zip=""; result=""; undo=""; staged=.update-new
swapped=0
if check_journal; then
    undo=$(sed -n "/^end$tab/,\$p" "$journal" | grep "^undo$tab" | tail -n 1 | cut -f 2)
    swapped=$(sed -n "/^end$tab/,\$p" "$journal" | grep -c '^swapped$')
    waited=0
    while running; do
        if [ "$waited" -ge 1200 ]; then message="chaiNNer-C did not close within 10 minutes."; break; fi
        sleep 0.5; waited=$((waited + 1))
    done
else
    message=$error
fi
if [ -z "$message" ]; then
    closed=true
    if [ "$swapped" != 0 ]; then
        # the new release is whole: only what it replaced is left to remove
        ok=true
    elif [ -z "$undo" ]; then
        if check_staged && mkdir -p "$install/.update-old" && each forward; then
            ok=true
        else
            undo=$(line "$error"); printf 'undo\t%s\n' "$undo" >> "$journal"
        fi
    fi
    if [ "$ok" = true ] && [ "$swapped" = 0 ]; then
        # from here on, a run of this journal only removes what the update replaced
        printf 'swapped\n' >> "$journal"; swapped=1
        exists "$install/.update-done" || mv -T "$install/.update-old" "$install/.update-done"
    fi
    if [ -n "$undo" ]; then
        if each back back; then resolved=true; message="The update was undone: $undo"; else message=$error; fi
    else
        resolved=true
    fi
fi
# nothing was changed: every entry as the old release had it, and nothing moved aside
untouched=true
[ "$swapped" = 0 ] || untouched=false
[ -n "$(ls -A "$install/.update-old" 2>/dev/null)" ] && untouched=false
list=$(entries 2>/dev/null)
while IFS="$tab" read -r key e o n; do
    [ -n "$key" ] || continue
    # the plan's valid entries: a journal that is not checked may hold others
    plain "$e" || continue
    case "$o$n" in 01 | 10 | 11) ;; *) continue ;; esac
    old_place "$e" "$o" || untouched=false
done <<EOF
$list
EOF
aborted=false
[ "$resolved" = false ] && [ "$untouched" = true ] && aborted=true
if [ "$resolved" = true ]; then
    remove_leftover "$install/.update-old"
    remove_leftover "$install/.update-done"
    remove_leftover "$install/.update-new"
    [ "$ok" = true ] && remove_leftover "$zip"
elif [ "$aborted" = true ]; then
    message="$message Nothing was changed."
    [ -n "$zip" ] && message="$message The downloaded release stays at $zip for an install by hand."
    remove_leftover "$install/.update-new"
else
    printf 'failed\t%s\n' "$(line "$message")" >> "$journal"
fi
escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }
if [ -n "$result" ]; then
    printf '{"ok":%s,"release":"%s","message":"%s","zip":"%s"}' "$ok" "$(escape "$(value release)")" "$(escape "$(line "$message")")" "$(escape "$zip")" > "$result"
fi
if [ "$resolved" = true ] || [ "$aborted" = true ]; then
    remove_leftover "$install/update-not-finished.txt"
    remove_leftover "$install/update-staged.txt"
    remove_leftover "$install/update-installed.txt"
    remove_leftover "$journal"
    exec 9>&-
    # an app that did not close still runs: it is not started a second time
    if [ "$resolved" = true ] && [ -n "$launch" ]; then (cd "$install" && nohup "$install/$launch" >/dev/null 2>&1 &); fi
    rm -f "$install/update.lock" "$0"
fi
`;

/** The install's entry the app starts again with after an update: Linux's launcher script when the install has it. */
const launchTarget = (target: string): string =>
    getPlatform() === 'linux' && existsSync(path.join(target, 'chaiNNer.sh'))
        ? 'chaiNNer.sh'
        : path.basename(process.execPath);

/** Where the installer writes its result, which the next start reports. */
export const updateResultPath = (): string =>
    path.join(app.getPath('userData'), 'update-result.json');

/** The result of an update the installer finished before this start. */
export interface UpdateResult {
    ok: boolean;
    release: string;
    message: string;
    zip: string;
}

/**
 * The zip the quit keeps (each staging has its own): the one an installer this process started
 * swaps in, or one whose install was refused, kept for an install by hand.
 */
let keptZip: string | undefined;

/**
 * Removes, when the app quits, the updates this process staged and did not discard (their zip,
 * unpacked folder and list): all but the kept one's zip, and the folder an installer swaps in.
 */
export const removeUnpackedReleases = (): void => {
    for (const [folder, zip] of stagedUpdates) {
        if (zip !== keptZip) {
            try {
                rmSync(folder, { recursive: true, force: true });
                rmSync(path.join(path.dirname(folder), STAGED_LIST), { force: true });
                rmSync(zip, { force: true });
            } catch (error) {
                log.error(`Unable to remove the staged update in ${folder}.`, error);
            }
        }
    }
};

const writeResult = async (result: UpdateResult): Promise<void> => {
    await fs.writeFile(updateResultPath(), JSON.stringify(result));
};

/** Starts the install's installer, detached, to run once the process `waitFor` has ended. */
const spawnInstaller = async (target: string, waitFor: number): Promise<void> => {
    const [program, args] = installerCommand(
        path.join(target, finishFiles(getPlatform()).script),
        waitFor
    );
    const child: ChildProcess = spawn(program, args, {
        detached: true,
        stdio: 'ignore',
        windowsHide: true,
    });
    // a program that cannot start (e.g. no powershell.exe) is reported by an event, later
    await new Promise<void>((resolve, reject) => {
        child.once('spawn', resolve);
        child.once('error', reject);
    });
    child.unref();
};

/** What a user reads in NOTICE and in the app's messages: how to finish the update by hand. */
const finishSteps = (target: string, zip: string): string =>
    `To finish it: close every program that uses ${target}, then start chaiNNer-C again. If it does not start, unpack ${
        zip || 'the new release'
    } over that folder, completely, and start chaiNNer-C: it clears up the rest.`;

/**
 * Writes the update's plan (durably, last), the installer and the files a user can finish it
 * with into the install, then starts the installer, detached, to run once this process has ended.
 * The caller ends the app. Rejects when the update may not be installed or the installer does not
 * start, and removes those files then; why goes to the result the next start reports, and the zip
 * stays for an install by hand.
 */
export const startInstaller = async (update: StagedUpdate): Promise<void> => {
    const target = currentExecutableDir;
    const platform = getPlatform();
    const { script, runner } = finishFiles(platform);
    const written = [script, runner, NOTICE, INSTALLED_LIST, JOURNAL].map((name) =>
        path.join(target, name)
    );
    try {
        const journal = journalText({
            app: ownMark(),
            release: update.release,
            zip: update.zip,
            result: updateResultPath(),
            launch: launchTarget(target),
            executable: path.basename(process.execPath),
            staged: path.relative(target, update.root),
            entries: await planUpdate(update, target),
        });
        const entries = parseJournal(`${journal}`).entries.filter((entry) => entry.old);
        const installed = await entries.reduce<Promise<StagedFile[]>>(async (previous, entry) => {
            const listed = await previous;
            listed.push(...(await listEntry(target, entry.name)));
            return listed;
        }, Promise.resolve([]));
        await writeDurably(
            path.join(target, INSTALLED_LIST),
            stagedListText({ owner: ownMark(), files: installed })
        );
        if (platform === 'win32') {
            await fs.writeFile(path.join(target, script), WINDOWS_INSTALLER, 'utf-8');
            await fs.writeFile(
                path.join(target, runner),
                // (goto) ends the reading of this file, which the installer removes, first
                `@(goto) 2>nul & powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0${script}"\r\n`
            );
        } else {
            await fs.writeFile(path.join(target, script), LINUX_INSTALLER, {
                encoding: 'utf-8',
                mode: 0o755,
            });
        }
        await fs.writeFile(
            path.join(target, NOTICE),
            `An update of chaiNNer-C to ${update.release} did not finish. ${finishSteps(
                target,
                update.zip
            )}\n`
        );
        // last: a journal always has its installer and its staged list beside it
        // the install is as the update found it: checked, for the installer
        await writeDurably(path.join(target, JOURNAL), `${journal}checked\n`);
        await spawnInstaller(target, process.pid);
    } catch (error) {
        // the journal first: without it the other files mean nothing
        await written.reverse().reduce(async (previous, file) => {
            await previous;
            await fs.rm(file, { force: true });
        }, Promise.resolve());
        keptZip = update.zip;
        await writeResult({
            ok: false,
            release: update.release,
            message: error instanceof Error ? error.message : String(error),
            zip: update.zip,
        });
        throw error;
    }
    keptZip = update.zip;
    log.info(`Installing ${update.release} into ${target} after chaiNNer-C closes`);
};

/** Whether an installer holds the install's lock: one runs. */
/**
 * Whether an installer holds the install's lock (`held`), or not (`free`), or what is wrong with
 * a lock file no installer holds and that cannot be removed. Only a file in use by another process
 * is an installer; any other error opening a left lock file (e.g. a read-only one) is not, and the
 * file is removed.
 */
const lockState = (target: string): 'held' | 'free' | { problem: string } => {
    const lock = path.join(target, LOCK);
    if (!existsSync(lock)) {
        return 'free';
    }
    if (getPlatform() === 'win32') {
        // an installer holds it open with no sharing
        try {
            closeSync(openSync(lock, 'r+'));
            return 'free';
        } catch (error) {
            if ((error as NodeJS.ErrnoException).code === 'EBUSY') {
                return 'held';
            }
            log.warn(`${lock} cannot be opened, and no installer holds it: ${String(error)}`);
        }
        try {
            rmSync(lock, { force: true });
            return 'free';
        } catch (error) {
            return {
                problem: `${lock} is left from an update and cannot be removed (${String(
                    error
                )}). Delete it, then start chaiNNer-C again.`,
            };
        }
    }
    const probe = spawnSync('flock', ['-n', lock, 'true'], { stdio: 'ignore' });
    if (probe.error) {
        // no flock: no installer could have taken the lock either
        log.warn(`The update lock cannot be checked: ${String(probe.error)}`);
        return 'free';
    }
    return probe.status === 0 ? 'free' : 'held';
};

/** Where each entry of the journal's plan stands in the install. */
export const entryStates = (target: string, journal: Journal): EntryState[] =>
    journal.entries.map((entry) =>
        entryState(
            entry,
            existsSync(path.join(target, entry.name)),
            existsSync(path.join(target, UPDATE_OLD, entry.name)),
            existsSync(path.join(target, journal.staged, entry.name))
        )
    );

/** Whether an entry below `base` holds exactly these files (the same set, sizes and digests). */
const entryIs = async (
    base: string,
    name: string,
    recorded: readonly StagedFile[]
): Promise<boolean> => {
    if (!existsSync(path.join(base, name))) {
        return false;
    }
    const actual = await listEntry(base, name);
    const key = (file: StagedFile) => `${file.path}\t${file.size}\t${file.sha256}`;
    const want = new Set(recorded.map(key));
    return actual.length === want.size && actual.every((file) => want.has(key(file)));
};

/** Where the install stands, by digest: one release's whole, an update's own state, or neither. */
type InstallProof = 'staged' | 'producible' | { unproven: string };

/**
 * The states an update makes of an entry (whether the installed and the staged release have it;
 * whose copy is in the install, in UPDATE_OLD and in the staged release's folder), and where that
 * leaves it: before its renames, between them, after them. Forward and undo pass the same states.
 */
const PRODUCIBLE: Partial<Record<string, 'start' | 'mid' | 'end'>> = {
    '11 old - new': 'start',
    '11 - old new': 'mid',
    '11 new old -': 'end',
    '10 old - -': 'start',
    '10 - old -': 'end',
    '01 - - new': 'start',
    '01 new - -': 'end',
};

/**
 * Whether every entry of the update is in a state the update itself could have produced: in the
 * install, the installed release's copy (byte for byte), the staged one's, or absent with its
 * counterpart in the update's own folders, each of those also byte for byte. Anything else (a
 * release unpacked over it, wholly or not, a file changed or removed) is unproven, and so are
 * records that are missing or damaged. After the swap (`swapped`), only the install itself counts:
 * what it replaced is being removed.
 */
const proveInstall = async (
    target: string,
    journal: Journal,
    swapped: boolean
): Promise<InstallProof> => {
    let installed: StagedFile[];
    let staged: StagedList;
    try {
        installed = [
            ...parseStagedList(readFileSync(path.join(target, INSTALLED_LIST), 'utf-8')).files,
        ];
        staged = parseStagedList(readFileSync(path.join(target, STAGED_LIST), 'utf-8'));
    } catch (error) {
        return {
            unproven: `the update's records of the two releases cannot be read (${String(error)})`,
        };
    }
    const filesOf = (files: readonly StagedFile[], name: string) =>
        files.filter((file) => file.path === name || file.path.startsWith(`${name}/`));
    // Whose copy of an entry a folder can be: none (-), the installed release's, the staged
    // one's, or another. An entry the same in both releases (byte for byte, or with no files at
    // all) is either copy, and the state decides which it is.
    const whose = async (
        base: string,
        name: string,
        copies: { old?: readonly StagedFile[]; new?: readonly StagedFile[] }
    ): Promise<string[]> => {
        if (!existsSync(path.join(base, name))) {
            return ['-'];
        }
        const could: string[] = [];
        if (copies.old && (await entryIs(base, name, copies.old))) {
            could.push('old');
        }
        if (copies.new && (await entryIs(base, name, copies.new))) {
            could.push('new');
        }
        return could.length > 0 ? could : ['other'];
    };
    const old = path.join(target, UPDATE_OLD);
    const stagedRoot = path.join(target, journal.staged);
    // whose copies each entry in the install can be
    const installs: string[][] = [];
    const states = await journal.entries.reduce<Promise<string[]>>(async (previous, entry) => {
        const found = await previous;
        const was = entry.old ? filesOf(installed, entry.name) : undefined;
        const will = entry.new ? filesOf(staged.files, entry.name) : undefined;
        const install = await whose(target, entry.name, { old: was, new: will });
        installs.push(install);
        if (swapped) {
            found.push(install.includes(entry.new ? 'new' : '-') ? 'end' : 'other');
            return found;
        }
        const aside = await whose(old, entry.name, { old: was });
        const next = await whose(stagedRoot, entry.name, { new: will });
        const kind = `${Number(entry.old)}${Number(entry.new)}`;
        // the entry is as the update left it if some reading of its copies is a state it makes
        const made = install
            .flatMap((i) =>
                aside.flatMap((o) => next.map((n) => PRODUCIBLE[`${kind} ${i} ${o} ${n}`]))
            )
            .find((state) => state !== undefined);
        found.push(made ?? 'other');
        return found;
    }, Promise.resolve([]));
    // every entry of the staged release in the install, byte for byte (e.g. its zip unpacked by
    // hand): the update is in place, whatever else lies about
    if (journal.entries.every((entry, i) => !entry.new || installs[i].includes('new'))) {
        return 'staged';
    }
    const changed = journal.entries.filter((_, i) => states[i] === 'other').map(({ name }) => name);
    if (changed.length > 0) {
        return {
            unproven: `${changed.join(', ')} ${
                changed.length === 1 ? 'is' : 'are'
            } neither the update's nor either release's`,
        };
    }
    return 'producible';
};

/** Removes, at a start, a file or folder an update left; a failure is logged. */
const removeAtStart = (file: string): void => {
    try {
        rmSync(file, { recursive: true, force: true });
    } catch (error) {
        log.error(`Unable to remove ${file}, which an update left.`, error);
    }
};

/**
 * Removes what an update left in the install: its journal and files, UPDATE_DONE (what a whole
 * update replaced), UPDATE_NEW unless a running process stages there, the lock unless an
 * installer holds it. An entry of UPDATE_OLD (an update that did not finish) goes when the install
 * has its counterpart and is put back when it has none (with `deleteOld`, all go: the new release
 * is in place).
 */
const clearUpdate = (target: string, deleteOld: boolean): void => {
    const owner = readStagedList(target)?.owner;
    const staging = owner !== undefined && owner.pid !== process.pid && isRunning(owner);
    const old = path.join(target, UPDATE_OLD);
    const done = path.join(target, UPDATE_DONE);
    // what a whole update replaced and could not rename away: renamed away now (renamed first, so
    // a removal cut short never leaves it unmarked), never put back
    if (existsSync(path.join(old, REPLACED_MARK))) {
        removeAtStart(done);
        try {
            renameSync(old, done);
        } catch (error) {
            log.warn(`What the last update replaced cannot be removed yet: ${String(error)}`);
        }
    }
    if (existsSync(old) && !existsSync(path.join(old, REPLACED_MARK))) {
        for (const name of readdirSync(old)) {
            if (!deleteOld && !existsSync(path.join(target, name))) {
                try {
                    renameSync(path.join(old, name), path.join(target, name));
                    log.info(`Put ${name} back into ${target} from ${UPDATE_OLD}.`);
                } catch (error) {
                    log.error(`Unable to put ${name} back from ${UPDATE_OLD}.`, error);
                }
            }
        }
        removeAtStart(old);
    }
    const files = [
        UPDATE_DONE,
        INSTALLED_LIST,
        JOURNAL,
        `${JOURNAL}.tmp`,
        NOTICE,
        PROBE,
        'update-probe.ps1',
        'update-probe.sh',
    ];
    files.push(...Object.values(finishFiles('win32')), finishFiles('linux').script);
    if (!staging) {
        files.push(UPDATE_NEW, STAGED_LIST, `${STAGED_LIST}.tmp`);
    }
    if (lockState(target) === 'free') {
        files.push(LOCK);
    }
    files.forEach((file) => removeAtStart(path.join(target, file)));
};

/** What the start tells the user before the app ends, and what it does once that is closed. */
export interface UpdateMessage {
    title: string;
    text: string;
    afterwards?: () => Promise<void>;
    /** Offered as "Keep this release" beside "Quit": what keeping does; the app then starts again. */
    keep?: () => Promise<void>;
}

/**
 * Run at every start, before anything else: an update that did not finish keeps the app from
 * starting on a half-swapped install. Undefined when the app may start: no journal (what an
 * update left is cleared), a journal that is not whole while nothing was moved aside, or an
 * install that already holds the new release (unpacked by hand); the app then reports it as an
 * update result. Otherwise a message, and the app ends: while an installer runs; when its
 * installer gave up, or a resume made no progress, with how to finish it; else after starting
 * the installer again, which finishes or undoes the update and starts the app.
 */
export const checkUnfinishedUpdate = async (): Promise<UpdateMessage | undefined> => {
    const target = currentExecutableDir;
    const journalFile = path.join(target, JOURNAL);
    const running: UpdateMessage = {
        title: 'chaiNNer-C is being updated',
        text: `An update is being installed into ${target}. Start chaiNNer-C again once it has finished.`,
    };
    const lock = lockState(target);
    if (lock === 'held') {
        return running;
    }
    if (lock !== 'free') {
        return { title: 'The update lock is in the way', text: lock.problem };
    }
    if (!existsSync(journalFile)) {
        clearUpdate(target, false);
        return undefined;
    }
    let journal: Journal;
    try {
        journal = parseJournal(await readTextFile(journalFile));
    } catch (error) {
        if (existsSync(path.join(target, UPDATE_OLD))) {
            return {
                title: 'The update did not finish',
                text: `${String(error)}\n\n${finishSteps(target, '')}`,
            };
        }
        clearUpdate(target, false);
        await writeResult({
            ok: false,
            release: '',
            message:
                'An update was cut short before it changed anything, so chaiNNer-C is as it was.',
            zip: '',
        });
        return undefined;
    }
    if (journal.app.pid !== process.pid && isRunning(journal.app)) {
        return running;
    }
    // before any resume or undo: the install, by digest, as the update could have left it
    const started = Date.now();
    const proof = await proveInstall(target, journal, journal.swapped === true);
    log.info(`Checked the install for the interrupted update in ${Date.now() - started} ms`);
    if (proof === 'staged') {
        clearUpdate(target, true);
        await writeResult({
            ok: true,
            release: journal.release,
            message: '',
            zip: journal.zip,
        });
        return undefined;
    }
    if (proof !== 'producible') {
        // Neither release can be completed from such a state, and only the user knows what the
        // install now holds: asked, never kept, resumed or undone by itself. Quit leaves
        // everything as found, so the question comes back at the next start.
        return {
            title: 'The install was changed during an update',
            text: `An update to ${journal.release} was interrupted, and ${target} was changed since by something other than the update (${proof.unproven}): another release unpacked over it, completely or not, or files changed or removed. chaiNNer-C cannot tell which release it now holds, or whether it is whole.\n\nChoose "Keep this release" if you unpacked a release's zip over that folder completely: what is left of the interrupted update is removed, and chaiNNer-C starts. If chaiNNer-C does not start after this, unpack the release's zip again, completely. Otherwise choose "Quit", unpack the release's zip again, completely, over that folder, and start chaiNNer-C.`,
            keep: async () => {
                clearUpdate(target, true);
                await writeResult({
                    ok: true,
                    release: 'the release unpacked by hand',
                    message: '',
                    zip: journal.zip,
                });
            },
        };
    }
    const steps = finishSteps(target, journal.zip);
    const { script } = finishFiles(getPlatform());
    if (!existsSync(path.join(target, script))) {
        return {
            title: 'The update did not finish',
            text: `The update's installer is missing from ${target}.\n\n${steps}`,
        };
    }
    const progress = `${journal.undo === undefined ? 'forward' : 'undo'}:${entryStates(
        target,
        journal
    ).join(',')}`;
    // an installer that was started and never ran (nothing written since): not started again; one
    // that stopped and said why (failed) is, as often as the install still proves
    if (journal.resumes.at(-1) === progress && !journal.failedSinceResume) {
        return {
            title: 'The update did not finish',
            text: `The update to ${journal.release} could not be finished: the last attempt changed nothing.\n\n${steps}`,
        };
    }
    return {
        title: 'Finishing the update',
        // a recorded failure may come back: the way by hand, should this attempt stop too
        text:
            journal.failed === undefined
                ? `The update to ${journal.release} was cut short. chaiNNer-C finishes it once you close this message, and starts again by itself.`
                : `The update to ${journal.release} did not finish: the last attempt stopped (${journal.failed}). chaiNNer-C tries again once you close this message, and starts again by itself.\n\nIf it stops again: ${steps}`,
        // the installer starts once the message is closed: it never waits on an open message
        afterwards: async () => {
            await fs.appendFile(journalFile, `resume\t${progress}\nchecked\n`);
            log.info(`Finishing the update to ${journal.release} that was cut short`);
            await spawnInstaller(target, process.pid);
        },
    };
};

/**
 * Shows what checkUnfinishedUpdate has to tell, once the app is ready; with a choice, whether
 * "Keep this release" was chosen (and its keeping done).
 */
export const showUpdateAtStart = async ({ title, text, keep }: UpdateMessage): Promise<boolean> => {
    await app.whenReady();
    if (!keep) {
        dialog.showErrorBox(title, text);
        return false;
    }
    const { response } = await dialog.showMessageBox({
        type: 'question',
        title,
        message: title,
        detail: text,
        buttons: ['Keep this release', 'Quit'],
        defaultId: 1,
        cancelId: 1,
        noLink: true,
    });
    if (response !== 0) {
        return false;
    }
    await keep();
    return true;
};

/** The last update's result, once: the file is removed after it is read. */
export const takeUpdateResult = async (): Promise<UpdateResult | undefined> => {
    const file = updateResultPath();
    try {
        return JSON.parse(await readTextFile(file)) as UpdateResult;
    } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
            log.warn(`Could not read the last update's result: ${String(error)}`);
        }
        return undefined;
    } finally {
        await fs.rm(file, { force: true });
    }
};
