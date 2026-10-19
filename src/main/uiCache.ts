import { createHash } from 'crypto';
import { app } from 'electron/main';
import fs from 'fs/promises';
import path from 'path';
import { BackendData } from '../common/Backend';
import { PythonInfo } from '../common/common-types';
import { log } from '../common/log';
import { lazy } from '../common/util';
import { chainnerCVersion, isDevelopmentBuild } from '../common/version';
import { buildInfo } from './buildInfo';
import { readTextFile, writeFileAtomic } from './util';

/**
 * The folder of the cache in the app's root folder, which is Electron's userData too. Not
 * `cache`: on Windows that is Chromium's HTTP cache folder `Cache`, which Chromium clears and
 * replaces (e.g. after an unclean exit), and with it the file.
 */
export const UI_CACHE_FOLDER = 'ui-cache';

/** Where the node list cache of the app with this root folder is kept. */
export const getUiCacheFile = (rootDir: string): string =>
    path.join(rootDir, UI_CACHE_FOLDER, 'backend-ui.json');

interface UiCacheFile {
    key: string;
    nodes: BackendData[0];
    packages: BackendData[1];
}

const sha256 = (value: string): string => createHash('sha256').update(value).digest('hex');

/** The size and modification time of every Python file and native module of the backend. */
const fingerprintBackend = async (backendDir: string): Promise<string> => {
    const entries: string[] = [];
    const walk = async (dir: string): Promise<void> => {
        const children = await fs.readdir(dir, { withFileTypes: true });
        await Promise.all(
            children.map(async (child) => {
                const childPath = path.join(dir, child.name);
                if (child.isDirectory()) {
                    await walk(childPath);
                } else if (/\.(?:py|pyd|so)$/.test(child.name)) {
                    const { size, mtimeMs } = await fs.stat(childPath);
                    entries.push(`${path.relative(backendDir, childPath)}:${size}:${mtimeMs}`);
                }
            })
        );
    };
    await walk(backendDir);
    return sha256(entries.sort().join('\n'));
};

/**
 * The installed distributions of the integrated Python, by their metadata folder names, which
 * hold each distribution's name and version.
 */
const listDistributions = async ({ python, version }: PythonInfo): Promise<string[]> => {
    const [major, minor] = version.split('.');
    const sitePackages =
        process.platform === 'win32'
            ? path.join(path.dirname(python), 'Lib', 'site-packages')
            : path.join(
                  path.dirname(path.dirname(python)),
                  'lib',
                  `python${major}.${minor}`,
                  'site-packages'
              );
    const names = await fs.readdir(sitePackages).catch((): string[] => []);
    return names.filter((name) => /\.(?:dist|egg)-info$/.test(name)).sort();
};

/** How long the GPUs may take to list before the key goes without them. */
const GPU_INFO_TIMEOUT_MS = 1000;

/**
 * The GPUs and their drivers, which decide e.g. whether the TensorRT nodes exist. The editor
 * waits for the cache at its start, so a GPU process that does not answer counts as no GPUs.
 */
const listGpus = async (): Promise<string[]> => {
    let timeout: NodeJS.Timeout | undefined;
    try {
        const info = (await Promise.race([
            app.getGPUInfo('basic'),
            new Promise((_, reject) => {
                timeout = setTimeout(
                    () => reject(new Error('The GPU information did not arrive in time.')),
                    GPU_INFO_TIMEOUT_MS
                );
            }),
        ])) as {
            gpuDevice?: { vendorId: number; deviceId: number; driverVersion?: string }[];
        };
        return (info.gpuDevice ?? []).map(
            (gpu) => `${gpu.vendorId}:${gpu.deviceId}:${gpu.driverVersion ?? ''}`
        );
    } catch (error) {
        log.warn('Unable to list the GPUs for the node list cache.', error);
        return [];
    } finally {
        clearTimeout(timeout);
    }
};

/**
 * The editor's copy of the backend's node list (`BackendData`), kept in `file` between starts so
 * the editor can show the nodes at once while the backend starts.
 *
 * The copy belongs to what decides the node list: the app and backend build, the Python and its
 * installed distributions, the GPUs and their drivers, and `CUDA_VISIBLE_DEVICES`. A copy made
 * with any of them different is not used. The editor checks the copy against the live list
 * anyway, so a factor this misses costs a moment of a stale list, not a wrong one.
 */
export class UiCache {
    private readonly file: string;

    private readonly getPython: () => Promise<PythonInfo>;

    /** The key parts that cannot change while the app runs. */
    private readonly fixedKey: () => Promise<string>;

    /** The content of the file as last read or written, `undefined` if there is none. */
    private onDisk: Promise<string | undefined>;

    private writing: Promise<void> = Promise.resolve();

    /** `getPython` gives the backend's Python, once the backend has started. */
    constructor(file: string, getPython: () => Promise<PythonInfo>, backendDir: string) {
        this.file = file;
        this.getPython = getPython;
        this.fixedKey = lazy(async () => {
            const [pythonInfo, build, gpus] = await Promise.all([
                getPython(),
                isDevelopmentBuild(buildInfo) ? fingerprintBackend(backendDir) : buildInfo.tree,
                listGpus(),
            ]);
            return JSON.stringify({
                app: app.getVersion(),
                chainnerC: chainnerCVersion,
                build,
                python: pythonInfo,
                gpus,
                cudaVisibleDevices: process.env.CUDA_VISIBLE_DEVICES ?? null,
            });
        });
        this.onDisk = readTextFile(file).catch(() => undefined);
    }

    private async getKey(): Promise<string> {
        // the distributions change when the dependency manager installs or removes a package
        const [fixedKey, distributions] = await Promise.all([
            this.fixedKey(),
            this.getPython().then(listDistributions),
        ]);
        return sha256(JSON.stringify([fixedKey, distributions]));
    }

    /** The cached node list, if there is one and it belongs to this app, Python and system. */
    async read(): Promise<BackendData | undefined> {
        const content = await this.onDisk;
        if (content === undefined) {
            return undefined;
        }
        const key = await this.getKey();
        try {
            const cache = JSON.parse(content) as Partial<UiCacheFile>;
            if (cache.key !== key || !cache.nodes || !Array.isArray(cache.packages)) {
                log.info('The node list cache belongs to another build or system; not used.');
                return undefined;
            }
            return [cache.nodes, cache.packages];
        } catch (error) {
            log.warn('Unable to read the node list cache.', error);
            return undefined;
        }
    }

    /**
     * Keeps the live node list for the next start. Writes only when the file would change; the
     * promise never rejects (a failure is logged).
     */
    write(data: BackendData): Promise<void> {
        this.writing = this.writing
            .then(async () => {
                const cache: UiCacheFile = {
                    key: await this.getKey(),
                    nodes: data[0],
                    packages: data[1],
                };
                const content = JSON.stringify(cache);
                if (content !== (await this.onDisk)) {
                    await writeFileAtomic(this.file, content);
                    this.onDisk = Promise.resolve(content);
                }
            })
            .catch((error) => log.error('Unable to write the node list cache.', error));
        return this.writing;
    }
}
