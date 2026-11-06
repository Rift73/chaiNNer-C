import { app, dialog } from 'electron/main';
import { t } from 'i18next';
import { log } from '../../common/log';
import { OwnedBackendProcess } from '../backend/process';
import { getLogsFolder } from '../platform';
import { ipcMain } from '../safeIpc';
import { SettingsWriter } from '../setting-storage';

/**
 * Ends the app with exit code 1 once the settings are written: `app.exit` skips the quit events,
 * and with them the write of settings changed shortly before (still waiting for the writer's
 * delay).
 */
const exitAfterFailure = (settingsWriter: SettingsWriter): void => {
    settingsWriter
        .flush()
        .finally(() => app.exit(1))
        .catch(log.error);
};

/**
 * Ends the app when the backend it owns fails: on an unexpected error, and when the backend stops
 * before the renderer reported it ready (the user would see only the splash screen, which would
 * wait forever for a backend that has stopped).
 */
export const exitOnBackendFailure = (
    backend: OwnedBackendProcess,
    settingsWriter: SettingsWriter
): void => {
    backend.addErrorListener((error) => {
        dialog.showMessageBoxSync({
            type: 'error',
            title: t('backendFailure.unexpectedErrorTitle', 'Unexpected Error'),
            message: t(
                'backendFailure.unexpectedError',
                'The Python backend encountered an unexpected error. ChaiNNer will now exit. Error: {{error}}',
                { error: String(error) }
            ),
        });
        exitAfterFailure(settingsWriter);
    });

    let backendReady = false;
    ipcMain.once('backend-ready', () => {
        backendReady = true;
    });
    backend.addExitListener(({ code, signal, stderrTail }) => {
        if (backendReady) {
            return;
        }
        const exitStatus =
            code !== null
                ? t('backendFailure.exitCode', 'exit code {{code}}', { code })
                : t('backendFailure.signal', 'signal {{signal}}', { signal: String(signal) });
        const lastOutput = stderrTail
            ? `\n\n${t('backendFailure.lastOutput', 'Its last output:')}\n${stderrTail}`
            : '';
        dialog.showMessageBoxSync({
            type: 'error',
            title: t('backendFailure.unableToStart', 'Unable to start the backend'),
            message: `${t(
                'backendFailure.stoppedWhileStarting',
                'The Python backend stopped while chaiNNer was starting ({{status}}). ChaiNNer will now exit.',
                { status: exitStatus }
            )}${lastOutput}\n\n${t('backendFailure.fullLog', 'The full log is in {{folder}}.', {
                folder: getLogsFolder(),
            })}`,
        });
        exitAfterFailure(settingsWriter);
    });
};
