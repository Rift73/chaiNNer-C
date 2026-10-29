import { mkdtempSync, readFileSync, rmSync } from 'fs';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { defaultSettings } from '../../src/common/settings/settings';
import { OwnedBackendProcess } from '../../src/main/backend/process';
import { exitOnBackendFailure } from '../../src/main/gui/backendFailure';
import { SettingsWriter } from '../../src/main/setting-storage';

const { appExit, readyListeners } = vi.hoisted(() => ({
    appExit: vi.fn(),
    readyListeners: [] as (() => void)[],
}));

vi.mock('electron/main', () => ({
    app: { exit: appExit, getPath: () => os.tmpdir() },
    dialog: { showMessageBoxSync: vi.fn() },
    ipcMain: {
        once: (channel: string, listener: () => void) => {
            if (channel === 'backend-ready') readyListeners.push(listener);
        },
    },
}));

type ErrorListener = Parameters<OwnedBackendProcess['addErrorListener']>[0];
type ExitListener = Parameters<OwnedBackendProcess['addExitListener']>[0];

const fakeBackend = () => {
    const errorListeners: ErrorListener[] = [];
    const exitListeners: ExitListener[] = [];
    const backend = {
        addErrorListener: (listener: ErrorListener) => errorListeners.push(listener),
        addExitListener: (listener: ExitListener) => exitListeners.push(listener),
    } as unknown as OwnedBackendProcess;
    return { backend, errorListeners, exitListeners };
};

let rootDir: string;
beforeEach(() => {
    rootDir = mkdtempSync(path.join(os.tmpdir(), 'chaiNNer-backend-failure-test-'));
});
afterEach(() => {
    rmSync(rootDir, { recursive: true, force: true });
    appExit.mockReset();
    readyListeners.length = 0;
});

// A setting changed just before the failure waits for the writer's delay (an hour here).
const changedSettings = () => {
    const settingsWriter = new SettingsWriter(rootDir, 60 * 60 * 1000);
    const settings = { ...defaultSettings, theme: 'light' };
    settingsWriter.save(settings);
    return { settingsWriter, settings };
};

const settingsOnDisk = () =>
    JSON.parse(readFileSync(path.join(rootDir, 'settings.json'), 'utf-8')) as unknown;

test('an unexpected backend error ends the app after the settings are written', async () => {
    const { settingsWriter, settings } = changedSettings();
    const { backend, errorListeners } = fakeBackend();
    exitOnBackendFailure(backend, settingsWriter);

    let writtenAtExit: unknown;
    appExit.mockImplementation(() => {
        writtenAtExit = settingsOnDisk();
    });
    errorListeners.forEach((listener) => listener(new Error('broken pipe')));

    await vi.waitFor(() => expect(appExit).toHaveBeenCalledWith(1));
    expect(writtenAtExit).toEqual(settings);
});

test('a backend that stops while the app starts ends the app after the settings are written', async () => {
    const { settingsWriter, settings } = changedSettings();
    const { backend, exitListeners } = fakeBackend();
    exitOnBackendFailure(backend, settingsWriter);

    let writtenAtExit: unknown;
    appExit.mockImplementation(() => {
        writtenAtExit = settingsOnDisk();
    });
    exitListeners.forEach((listener) => listener({ code: 1, signal: null, stderrTail: '' }));

    await vi.waitFor(() => expect(appExit).toHaveBeenCalledWith(1));
    expect(writtenAtExit).toEqual(settings);
});

test('a backend that stops after it was ready does not end the app', async () => {
    const { settingsWriter } = changedSettings();
    const { backend, exitListeners } = fakeBackend();
    exitOnBackendFailure(backend, settingsWriter);

    readyListeners.forEach((listener) => listener());
    exitListeners.forEach((listener) => listener({ code: 1, signal: null, stderrTail: '' }));

    await settingsWriter.flush();
    expect(appExit).not.toHaveBeenCalled();
});
