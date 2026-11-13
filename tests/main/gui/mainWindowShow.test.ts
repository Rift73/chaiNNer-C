import { EventEmitter } from 'events';
import { Mock, afterEach, beforeEach, expect, test, vi } from 'vitest';
import { ChainnerSettings } from '../../../src/common/settings/settings';
import { OpenArguments } from '../../../src/main/arguments';
import { createMainWindow } from '../../../src/main/gui/main-window';
import { ipcMain } from '../../../src/main/safeIpc';

const { windows, backend, protocol } = vi.hoisted(() => ({
    windows: [] as FakeWindow[],
    protocol: { handle: vi.fn(), registerSchemesAsPrivileged: vi.fn() },
    // the backend the window gets, if it comes up
    backend: { started: new Promise<unknown>(() => {}) },
}));

interface FakeWindow extends EventEmitter {
    readonly webContents: EventEmitter;
    readonly show: ReturnType<typeof vi.fn>;
    readonly maximize: ReturnType<typeof vi.fn>;
    readonly loadURL: ReturnType<typeof vi.fn>;
    readonly options: { webPreferences: Record<string, unknown> };
}

vi.mock('electron/common', () => ({ shell: {} }));
vi.mock('electron/main', async () => {
    const { EventEmitter: Emitter } = await import('events');
    const emitter = () =>
        Object.assign(new Emitter(), { handle: vi.fn(), setWindowOpenHandler: vi.fn() });
    return {
        BrowserWindow: class extends Emitter {
            webContents = emitter();

            show = vi.fn();

            maximize = vi.fn();

            isDestroyed = vi.fn(() => false);

            loadURL = vi.fn(() => new Promise(() => {}));

            constructor(readonly options: unknown) {
                super();
                windows.push(this as unknown as FakeWindow);
            }
        },
        app: {
            getPath: () => 'documents',
            getVersion: () => '0.0.0',
            on: vi.fn(),
            requestSingleInstanceLock: () => true,
        },
        dialog: {},
        ipcMain: emitter(),
        nativeTheme: { shouldUseDarkColors: true },
        net: { fetch: vi.fn() },
        powerSaveBlocker: {},
        protocol,
    };
});
vi.mock('eventsource', () => ({ default: vi.fn() }));
vi.mock('electron-log', () => ({ default: { info: vi.fn(), error: vi.fn(), warn: vi.fn() } }));
vi.mock('../../../src/main/backend/process', () => ({
    getBackendPath: () => 'backend/run.py',
    stopBackendOnQuit: vi.fn(),
}));
// By default the backend never comes up: the window's showing does not wait for it.
vi.mock('../../../src/main/backend/setup', () => ({ setupBackend: () => backend.started }));
vi.mock('../../../src/main/buildInfo', () => ({ buildInfo: {} }));
vi.mock('../../../src/main/clipboard', () => ({}));
vi.mock('../../../src/main/clipboardImage', () => ({}));
vi.mock('../../../src/main/fileWatcher', () => ({ addBrowserWindow: vi.fn() }));
vi.mock('../../../src/main/platform', () => ({
    getRootDir: () => 'root',
    installDir: 'install',
}));
vi.mock('../../../src/main/SaveFile', () => ({
    openSaveFile: (path: string) => Promise.resolve({ kind: 'Success', path }),
}));
vi.mock('../../../src/main/setting-storage', () => ({ SettingsWriter: vi.fn() }));
vi.mock('../../../src/main/uiCache', () => ({ UiCache: vi.fn(), getUiCacheFile: vi.fn() }));
vi.mock('../../../src/main/gui/backendFailure', () => ({ exitOnBackendFailure: vi.fn() }));
vi.mock('../../../src/main/gui/dialogs', () => ({ showOpenDialog: vi.fn() }));
vi.mock('../../../src/main/gui/menu', () => ({ setMainMenu: vi.fn() }));

const args: OpenArguments = {
    command: 'open',
    devtools: false,
    file: undefined,
    refresh: false,
    remoteBackend: undefined,
};

const settings = (maximized: boolean) =>
    ({ lastWindowSize: { width: 800, height: 600, maximized } } as unknown as ChainnerSettings);

const openWindow = (platform: NodeJS.Platform, maximized = false): FakeWindow => {
    Object.defineProperty(process, 'platform', { value: platform, configurable: true });
    createMainWindow(args, settings(maximized)).catch((error) => {
        throw error;
    });
    return windows[windows.length - 1];
};

const { platform } = process;

beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal('MAIN_WINDOW_VITE_DEV_SERVER_URL', undefined);
    vi.stubGlobal('MAIN_WINDOW_VITE_NAME', 'main_window');
});

afterEach(() => {
    Object.defineProperty(process, 'platform', { value: platform, configurable: true });
    vi.unstubAllGlobals();
    vi.useRealTimers();
});

test('on Linux, a load that finishes shows the window when ready-to-show never fires', () => {
    const window = openWindow('linux', true);
    expect(window.show).not.toHaveBeenCalled();

    window.webContents.emit('did-finish-load');

    expect(window.show).toHaveBeenCalledOnce();
    expect(window.maximize).toHaveBeenCalledOnce();
});

test('on Linux, the window shows once whichever event fires first', () => {
    const first = openWindow('linux', true);
    first.webContents.emit('did-finish-load');
    first.emit('ready-to-show');
    expect(first.show).toHaveBeenCalledOnce();
    expect(first.maximize).toHaveBeenCalledOnce();

    const second = openWindow('linux');
    second.emit('ready-to-show');
    second.webContents.emit('did-finish-load');
    expect(second.show).toHaveBeenCalledOnce();
    expect(second.maximize).not.toHaveBeenCalled();
});

test.each<NodeJS.Platform>(['win32', 'darwin'])(
    'on %s, only ready-to-show shows the window',
    (other) => {
        const window = openWindow(other, true);

        expect(window.webContents.listenerCount('did-finish-load')).toBe(0);
        window.webContents.emit('did-finish-load');
        expect(window.show).not.toHaveBeenCalled();

        window.emit('ready-to-show');
        expect(window.show).toHaveBeenCalledOnce();
        expect(window.maximize).toHaveBeenCalledOnce();
    }
);

test('the built renderer loads over its scheme, not from file://, keeping its code cache', () => {
    protocol.handle.mockClear();
    const window = openWindow('win32');

    expect(window.loadURL).toHaveBeenCalledWith('chainner-ui://app/index.html');
    expect(protocol.handle).toHaveBeenCalledOnce();
    expect(window.options.webPreferences.v8CacheOptions).toBe('bypassHeatCheck');
});

test('the dev server loads when there is one, and the scheme serves nothing', () => {
    protocol.handle.mockClear();
    vi.stubGlobal('MAIN_WINDOW_VITE_DEV_SERVER_URL', 'http://localhost:5173');
    const window = openWindow('win32');

    expect(window.loadURL).toHaveBeenCalledWith('http://localhost:5173');
    expect(protocol.handle).not.toHaveBeenCalled();
});

/** The last handler registered for `channel`. */
const handlerOf = (channel: string): (() => unknown) => {
    const mocked = ipcMain as unknown as { handle: Mock<[string, () => unknown]> };
    const { calls } = mocked.handle.mock;
    const handler = calls.filter(([name]) => name === channel).at(-1)?.[1];
    if (!handler) throw new Error(`No handler for `);
    return handler;
};
/** The answer the window's page gets from `channel` (on each start, for the start-up ones). */
const answer = (channel: string) => handlerOf(channel)();

// View > Reload starts the page again, which asks again: the chain on screen is newer by then.
test.each([
    ['the chain the app was opened with', { ...args, file: 'work.chn' }, {}],
    ['the Startup Template', args, { startupTemplate: 'template.chn' }],
])('%s opens at the first start of the page only', async (_, openArgs, extra) => {
    Object.defineProperty(process, 'platform', { value: 'win32', configurable: true });
    createMainWindow(openArgs, { ...settings(false), ...extra } as ChainnerSettings).catch(
        (error) => {
            throw error;
        }
    );

    expect(await answer('get-auto-open')).toMatchObject({ kind: 'Success' });
    expect(await answer('get-auto-open')).toBeUndefined();
});

test('an update is refused before its download on a backend the app did not start', async () => {
    backend.started = Promise.resolve({ owned: false, url: 'http://127.0.0.1:8000', python: {} });
    Object.defineProperty(process, 'platform', { value: 'win32', configurable: true });
    createMainWindow(args, settings(false)).catch(() => {});
    // registered once the backend is up
    await vi.waitFor(() => expect(() => handlerOf('download-update')).not.toThrow());

    await expect(answer('download-update')).rejects.toThrow('did not start');
});
