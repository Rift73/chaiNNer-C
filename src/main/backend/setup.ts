import { t } from 'i18next';
import path from 'path';
import portfinder from 'portfinder';
import { PythonInfo } from '../../common/common-types';
import { log } from '../../common/log';
import { CriticalError } from '../../common/ui/error';
import { ProgressToken } from '../../common/ui/progress';
import { getBackendStorageFolder } from '../platform';
import { getIntegratedPython } from '../python/integratedPython';
import { BackendProcess, BorrowedBackendProcess, OwnedBackendProcess } from './process';

const getValidPort = async () => {
    log.info('Attempting to check for a port...');
    const port = await portfinder.getPortPromise();
    if (!port) {
        log.error('An open port could not be found');

        throw new CriticalError({
            title: 'No open port',
            message:
                'This error should never happen, but if it does it means you are running a lot of servers on your computer that just happen to be in the port range I look for. Quit some of those and then this will work.',
        });
    }
    log.info(`Port found: ${port}`);

    return port;
};

// chaiNNer-C runs only on its integrated CPython 3.14: its native modules link python314.dll and
// the host imports packages that a system Python would not have.
const getIntegratedPythonInfo = async (
    token: ProgressToken,
    integratedPythonFolder: string
): Promise<PythonInfo> => {
    log.info('Attempting to check integrated Python env...');

    try {
        return await getIntegratedPython(integratedPythonFolder, (percentage, stage) => {
            token.submitProgress({
                status:
                    stage === 'download'
                        ? t('setup.downloadingPython', 'Downloading Integrated Python...')
                        : t('setup.extractingPython', 'Extracting downloaded files...'),
                totalProgress: stage === 'download' ? 0.3 : 0.4,
                statusProgress: percentage / 100,
            });
        });
    } catch (error) {
        log.error(error);

        enum Action {
            Retry,
            Crash,
        }
        let action: Action = Action.Crash as Action;
        await token.submitInterrupt({
            type: 'warning',
            title: 'Unable to install integrated Python',
            message:
                'chaiNNer needs a stable internet connection to install its integrated Python environment. Please make sure you have a stable internet connection and try again.' +
                '\n\nchaiNNer requires a valid Python environment to run. Please choose one of the following options:',
            options: [
                {
                    title: 'Retry to install integrated Python',
                    action: {
                        type: 'run',
                        action: () => {
                            action = Action.Retry;
                        },
                    },
                },
                {
                    title: 'Exit',
                    action: {
                        type: 'run',
                        action: () => {
                            action = Action.Crash;
                        },
                    },
                },
            ],
        });

        if (action === Action.Retry) {
            return getIntegratedPythonInfo(token, integratedPythonFolder);
        }

        throw new CriticalError({
            title: 'Unable to install integrated Python',
            message:
                `chaiNNer was unable to install its integrated Python environment.` +
                ` Please ensure that your computer is connected to the internet and that chaiNNer has access to the network.`,
        });
    }
};
const getPythonInfo = async (token: ProgressToken, rootDir: string) => {
    const pythonInfo = await getIntegratedPythonInfo(
        token,
        path.normalize(path.join(rootDir, 'python'))
    );

    log.info(`Final Python binary: ${pythonInfo.python}`);
    log.info(pythonInfo);

    return pythonInfo;
};

const spawnBackend = (port: number, pythonInfo: PythonInfo) => {
    try {
        const backend = OwnedBackendProcess.spawn({
            port,
            python: pythonInfo,
            storageDir: getBackendStorageFolder(),
        });

        return backend;
    } catch (error) {
        log.error('Error spawning backend.', error);
        throw new CriticalError({ message: 'Unable to start backend.' });
    }
};

const setupOwnedBackend = async (
    token: ProgressToken,
    rootDir: string
): Promise<OwnedBackendProcess> => {
    token.submitProgress({
        status: t('setup.checkingPort', 'Checking for available port...'),
        totalProgress: 0.1,
    });
    const port = await getValidPort();

    token.submitProgress({
        status: t('setup.checkingPython', 'Checking system environment for valid Python...'),
        totalProgress: 0.3,
    });
    const pythonInfo = await getPythonInfo(token, rootDir);

    token.submitProgress({
        status: t('setup.startingBackend', 'Starting up backend process...'),
        totalProgress: 0.7,
    });
    return spawnBackend(port, pythonInfo);
};

const setupBorrowedBackend = async (
    token: ProgressToken,
    url: string
): Promise<BorrowedBackendProcess> => {
    log.info(`Attempting to setup backend from ${url}...`);

    token.submitProgress({
        status: t('setup.startingBackend', 'Starting up backend process...'),
        totalProgress: 0.8,
    });
    return BorrowedBackendProcess.fromUrl(url);
};

export const setupBackend = async (
    token: ProgressToken,
    // Ignored, and always off since the settings migration: chaiNNer-C runs only on its
    // integrated Python. They stay until the callers in cli/run.ts and gui/main-window.ts drop them.
    useSystemPython: boolean,
    systemPythonLocation: string | undefined | null,
    rootDir: string,
    remoteBackend: string | undefined
): Promise<BackendProcess> => {
    token.submitProgress({ totalProgress: 0 });

    const backend = remoteBackend
        ? await setupBorrowedBackend(token, remoteBackend)
        : await setupOwnedBackend(token, rootDir);

    token.submitProgress({ totalProgress: 1 });
    return backend;
};
