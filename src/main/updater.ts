import { spawn } from 'child_process';
import { createHash } from 'crypto';
import decompress from 'decompress';
import { app, net } from 'electron/main';
import { createReadStream, existsSync } from 'fs';
import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import { log } from '../common/log';
import { SupportedPlatform, currentExecutableDir, getPlatform } from './platform';
import { download } from './python/integratedPython';

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

/** A release downloaded, checked and unpacked, ready to be copied over the install. */
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
 * Downloads the latest release's zip for this platform into the Downloads folder, checks it
 * against the SHA-256 GitHub reports, and unpacks it into a temporary folder.
 */
export const stageUpdate = async (
    onProgress: (percentage: number) => void
): Promise<StagedUpdate> => {
    if (!app.isPackaged) {
        throw new Error('Only a packaged chaiNNer-C can update itself.');
    }
    const release = await fetchLatestRelease();
    const asset = findReleaseAsset(release, getPlatform(), os.arch());
    const sha256 = expectedSha256(asset);

    const zip = path.join(app.getPath('downloads'), asset.name);
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

    const unpacked = await fs.mkdtemp(path.join(os.tmpdir(), 'chaiNNer-C-update-'));
    try {
        await decompress(zip, unpacked);
        const root = await findReleaseRoot(unpacked, path.basename(process.execPath));
        onProgress(100);
        return { release: release.name ?? release.tag_name, zip, unpacked, root };
    } catch (error) {
        await fs.rm(unpacked, { recursive: true, force: true });
        throw error;
    }
};

/** Removes a staged update that will not be installed. */
export const discardUpdate = async (update: StagedUpdate): Promise<void> => {
    await fs.rm(update.unpacked, { recursive: true, force: true });
    await fs.rm(update.zip, { force: true });
};

/**
 * Copies a release over an install once every process run from the install has ended. Every
 * file the release has is written; the app bundle (resources/app) is mirrored, so files a release
 * dropped go too. Nothing else in the install is touched: the integrated Python and its packages,
 * settings and storage stay. The zip and the unpacked copy are removed after a successful copy;
 * after a failure the zip stays for a manual install. The result goes to `Result` as JSON, and
 * `Launch`, if given, starts again either way.
 */
export const WINDOWS_INSTALLER = String.raw`param(
    [int]$ProcessId, [string]$Source, [string]$Target, [string]$Zip, [string]$Unpacked,
    [string]$Release, [string]$Result, [string]$Launch
)
$ErrorActionPreference = 'Stop'
$ok = $false
$message = ''
try {
    $deadline = (Get-Date).AddMinutes(10)
    $prefix = $Target.TrimEnd('\') + '\'
    while ($true) {
        # the app, every process run from the install, and every process running its resources
        # (the backend's workers, whose Python may live elsewhere); not this one
        $running = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
            $_.ProcessId -ne $PID -and (
                $_.ProcessId -eq $ProcessId -or
                ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) -or
                ($_.CommandLine -and $_.CommandLine.IndexOf($prefix + 'resources', [StringComparison]::OrdinalIgnoreCase) -ge 0)
            )
        })
        if ($running.Count -eq 0) { break }
        if ((Get-Date) -gt $deadline) { throw 'chaiNNer-C did not close within 10 minutes; nothing was changed.' }
        Start-Sleep -Milliseconds 500
    }
    & robocopy $Source $Target /E /IS /IT /R:30 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Copying the release into $Target failed (robocopy $LASTEXITCODE)." }
    & robocopy (Join-Path $Source 'resources\app') (Join-Path $Target 'resources\app') /MIR /IS /IT /R:30 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Copying the app bundle into $Target failed (robocopy $LASTEXITCODE)." }
    $ok = $true
} catch {
    $message = $_.Exception.Message
}
if ($ok) {
    Remove-Item -LiteralPath $Zip -Force -ErrorAction SilentlyContinue
}
Remove-Item -LiteralPath $Unpacked -Recurse -Force -ErrorAction SilentlyContinue
$json = @{ ok = $ok; release = $Release; message = $message; zip = $Zip } | ConvertTo-Json -Compress
[IO.File]::WriteAllText($Result, $json)
if ($Launch) { Start-Process -FilePath $Launch -WorkingDirectory $Target }
Remove-Item -LiteralPath $MyInvocation.MyCommand.Path -Force -ErrorAction SilentlyContinue
`;

/** WINDOWS_INSTALLER's steps in POSIX shell, for Linux; its arguments in the same order. */
export const LINUX_INSTALLER = String.raw`#!/bin/sh
pid="$1"; source="$2"; target="$3"; zip="$4"; unpacked="$5"; release="$6"; result="$7"; launch="$8"
ok=false
message=""
# the app, every process run from the install, and every process running its resources (the
# backend's workers, whose Python may live elsewhere); not this one
running() {
    kill -0 "$pid" 2>/dev/null && return 0
    for proc in /proc/[0-9]*; do
        [ "$proc" = "/proc/$$" ] && continue
        case "$(readlink "$proc/exe" 2>/dev/null)" in "$target"/*) return 0 ;; esac
        tr '\0' ' ' < "$proc/cmdline" 2>/dev/null | grep -qF "$target/resources/" && return 0
    done
    return 1
}
waited=0
while running; do
    if [ "$waited" -ge 1200 ]; then message="chaiNNer-C did not close within 10 minutes; nothing was changed."; break; fi
    sleep 0.5; waited=$((waited + 1))
done
if [ -z "$message" ]; then
    if cp -a "$source"/. "$target"/ && rm -rf "$target/resources/app" && cp -a "$source/resources/app" "$target/resources/app"; then
        ok=true
    else
        message="Copying the release into $target failed."
    fi
fi
[ "$ok" = true ] && rm -f "$zip"
rm -rf "$unpacked"
escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }
printf '{"ok":%s,"release":"%s","message":"%s","zip":"%s"}' "$ok" "$(escape "$release")" "$(escape "$message")" "$(escape "$zip")" > "$result"
if [ -n "$launch" ]; then (cd "$target" && nohup "$launch" >/dev/null 2>&1 &); fi
rm -f "$0"
`;

/** WINDOWS_INSTALLER's parameters, in the order installerArguments gives them. */
export const WINDOWS_PARAMETERS = [
    'ProcessId',
    'Source',
    'Target',
    'Zip',
    'Unpacked',
    'Release',
    'Result',
    'Launch',
] as const;

/** The installer's arguments, in its parameters' order. */
export const installerArguments = (
    update: StagedUpdate,
    target: string,
    result: string,
    launch: string
): string[] => [
    String(process.pid),
    update.root,
    target,
    update.zip,
    update.unpacked,
    update.release,
    result,
    launch,
];

/** What the app starts again after an update: Linux's launcher script when the install has it. */
const launchTarget = (target: string): string => {
    const script = path.join(target, 'chaiNNer.sh');
    return getPlatform() === 'linux' && existsSync(script) ? script : process.execPath;
};

/** Where the installer writes its result, which the next start reports. */
export const updateResultPath = (): string =>
    path.join(app.getPath('userData'), 'update-result.json');

/**
 * Starts the installer for a staged update, detached, to run once this process has ended. The
 * caller ends the app.
 */
export const startInstaller = async (update: StagedUpdate): Promise<void> => {
    const target = currentExecutableDir;
    const args = installerArguments(update, target, updateResultPath(), launchTarget(target));
    const scriptDirectory = await fs.mkdtemp(path.join(os.tmpdir(), 'chaiNNer-C-installer-'));
    if (getPlatform() === 'win32') {
        const script = path.join(scriptDirectory, 'install.ps1');
        await fs.writeFile(script, WINDOWS_INSTALLER, 'utf-8');
        const named = WINDOWS_PARAMETERS.flatMap((name, i) => [`-${name}`, args[i]]);
        spawn(
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
                ...named,
            ],
            { detached: true, stdio: 'ignore', windowsHide: true }
        ).unref();
    } else {
        const script = path.join(scriptDirectory, 'install.sh');
        await fs.writeFile(script, LINUX_INSTALLER, { encoding: 'utf-8', mode: 0o755 });
        spawn('/bin/sh', [script, ...args], { detached: true, stdio: 'ignore' }).unref();
    }
    log.info(`Installing ${update.release} into ${target} after chaiNNer-C closes`);
};

/** The result of an update the installer finished before this start. */
export interface UpdateResult {
    ok: boolean;
    release: string;
    message: string;
    zip: string;
}

/** The last update's result, once: the file is removed after it is read. */
export const takeUpdateResult = async (): Promise<UpdateResult | undefined> => {
    const file = updateResultPath();
    try {
        const text = (await fs.readFile(file, 'utf-8')).replace(/^\ufeff/, '');
        return JSON.parse(text) as UpdateResult;
    } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
            log.warn(`Could not read the last update's result: ${String(error)}`);
        }
        return undefined;
    } finally {
        await fs.rm(file, { force: true });
    }
};
