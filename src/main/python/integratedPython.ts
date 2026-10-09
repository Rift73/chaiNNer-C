import decompress from 'decompress';
import { app, session } from 'electron/main';
import fs from 'fs/promises';
import path from 'path';
import semver from 'semver';
import { PythonInfo } from '../../common/common-types';
import { log } from '../../common/log';
import { isArmMac } from '../env';
import { SupportedPlatform, getPlatform } from '../platform';
import { checkFileExists } from '../util';
import { checkPythonPaths } from './checkPythonPaths';

interface PythonDownload {
    url: string;
    version: string;
    path: string;
}

// python-build-standalone 20261003, CPython 3.14.8: the interpreter of native/python-stack.lock.txt,
// whose header records the Windows archive's SHA-256.
const downloads: Record<SupportedPlatform, PythonDownload> = {
    linux: {
        url: 'https://github.com/astral-sh/python-build-standalone/releases/download/20261003/cpython-3.14.8+20261003-x86_64-unknown-linux-gnu-install_only.tar.gz',
        version: '3.14.8',
        path: 'python/bin/python3.14',
    },
    darwin: {
        url: isArmMac
            ? 'https://github.com/astral-sh/python-build-standalone/releases/download/20261003/cpython-3.14.8+20261003-aarch64-apple-darwin-install_only.tar.gz'
            : 'https://github.com/astral-sh/python-build-standalone/releases/download/20261003/cpython-3.14.8+20261003-x86_64-apple-darwin-install_only.tar.gz',
        version: '3.14.8',
        path: 'python/bin/python3.14',
    },
    win32: {
        url: 'https://github.com/astral-sh/python-build-standalone/releases/download/20261003/cpython-3.14.8+20261003-x86_64-pc-windows-msvc-install_only.tar.gz',
        version: '3.14.8',
        path: 'python/python.exe',
    },
};

/** An environment variable as pip reads it: the lower-case name first, then the upper-case one. */
const getProxyEnv = (name: string): string | undefined =>
    process.env[name] || process.env[name.toUpperCase()];

/**
 * Downloads `url` to `filePath` with Electron's network stack, which follows the system's proxy
 * settings (on Windows the Internet settings, PAC scripts included). A proxy in https_proxy or
 * else all_proxy, and no_proxy, if set, take precedence, as they do for pip.
 */
const download = async (
    url: string,
    filePath: string,
    onProgress: (percentage: number) => void
) => {
    // the CLI gets here before Electron is ready, and the network stack needs it
    await app.whenReady();

    let downloadSession = session.defaultSession;
    const proxy = getProxyEnv('https_proxy') || getProxyEnv('all_proxy');
    if (proxy) {
        downloadSession = session.fromPartition('integrated-python-download');
        // Chromium takes scheme://host:port; the variable may also have a path or no scheme
        const { protocol, host } = new URL(proxy.includes('://') ? proxy : `http://${proxy}`);
        await downloadSession.setProxy({
            proxyRules: `${protocol}//${host}`,
            proxyBypassRules: getProxyEnv('no_proxy'),
        });
    }

    const response = await downloadSession.fetch(url);
    if (!response.ok || !response.body) {
        throw new Error(
            `Downloading ${url} failed: HTTP ${response.status} ${response.statusText}`
        );
    }
    const totalBytes = Number(response.headers.get('content-length'));
    let receivedBytes = 0;

    await fs.mkdir(path.dirname(filePath), { recursive: true });
    const file = await fs.open(filePath, 'w');
    try {
        await response.body.pipeTo(
            new WritableStream<Uint8Array>({
                write: async (chunk) => {
                    await file.write(chunk);
                    receivedBytes += chunk.byteLength;
                    if (totalBytes > 0) {
                        onProgress((receivedBytes / totalBytes) * 100);
                    }
                },
            })
        );
    } catch (error) {
        await file.close();
        await fs.rm(filePath, { force: true });
        throw error;
    }
    await file.close();
};

const extractPython = async (
    directory: string,
    tarPath: string,
    onProgress: (percent: number) => void
) => {
    const files = await decompress(tarPath);
    const totalFiles = files.length;
    let doneCounter = 0;

    await Promise.all(
        files.map(async (file) => {
            const filePath = path.join(directory, file.path);
            await fs.mkdir(path.dirname(filePath), { recursive: true });
            await fs.writeFile(filePath, file.data);
            const percentageComplete = (doneCounter / totalFiles) * 100;
            doneCounter += 1;
            onProgress(percentageComplete);
        })
    );
};

const getIntegratedPythonExecutable = (directory: string): string => {
    const platform = getPlatform();
    const { path: relativePath } = downloads[platform];
    return path.resolve(path.join(directory, relativePath));
};

/**
 * Retrieves the path of the python executable of the integrated python installation.
 *
 * If the installation does not exist, python will be downloaded and installed.
 */
export const getIntegratedPython = async (
    directory: string,
    onProgress: (percentage: number, stage: 'download' | 'extract') => void
): Promise<PythonInfo> => {
    const platform = getPlatform();
    const { url, version } = downloads[platform];

    const pythonPath = getIntegratedPythonExecutable(directory);
    const pythonBinExists = await checkFileExists(pythonPath);

    if (pythonBinExists) {
        const pythonInfo = await checkPythonPaths([pythonPath]);
        // CPython 3.14: this table's version or a newer one is kept; an older one (upstream's
        // 3.11.5) is removed below and replaced by the download.
        if (semver.gte(pythonInfo.version, version)) {
            return pythonInfo;
        }
    }

    // Invalid version, remove legacy integrated python
    const legacyPythonFolder = path.resolve(path.join(directory, '/python'));
    await fs.rm(legacyPythonFolder, { recursive: true, force: true });

    log.info(`Integrated Python not found at ${pythonPath}`);

    const tarName = 'python.tar.gz';
    const tarPath = path.join(directory, tarName);

    log.info('Downloading integrated Python...');
    onProgress(0, 'download');
    await download(url, tarPath, (percentage) => onProgress(percentage, 'download'));

    log.info('Extracting integrated Python...');
    onProgress(0, 'extract');
    await extractPython(directory, tarPath, (percentage) => onProgress(percentage, 'extract'));

    log.info('Removing downloaded files...');
    await fs.rm(tarPath);

    if (platform === 'linux' || platform === 'darwin') {
        log.info('Granting permissions for integrated python...');
        try {
            await fs.chmod(pythonPath, 0o7777);
        } catch (error) {
            log.warn(error);
        }
    }

    return checkPythonPaths([pythonPath]);
};
