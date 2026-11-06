import { EventEmitter } from 'events';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { ChainnerSettings } from '../../../src/common/settings/settings';
import { OpenArguments } from '../../../src/main/arguments';
import { createMainWindow } from '../../../src/main/gui/main-window';

const { windows } = vi.hoisted(() => ({ windows: [] as FakeWindow[] }));

interface FakeWindow extends EventEmitter {
    readonly webContents: EventEmitter;
    readonly show: ReturnType<typeof vi.fn>;
    readonly maximize: ReturnType<typeof vi.fn>;
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

            loadFile = vi.fn(() => new Promise(() => {}));

            constructor() {
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
        powerSaveBlocker: {},
    };
});
vi.mock('eventsource', () => ({ default: vi.fn() }));
vi.mock('electron-log', () => ({ default: { info: vi.fn(), error: vi.fn(), warn: vi.fn() } }));
vi.mock('../../../src/main/backend/process', () => ({
    getBackendPath: () => 'backend/run.py',
    stopBackendOnQuit: vi.fn(),
}));
// The backend never comes up: the window's showing does not wait for it.
vi.mock('../../../src/main/backend/setup', () => ({ setupBackend: () => new Promise(() => {}) }));
vi.mock('../../../src/main/buildInfo', () => ({ buildInfo: {} }));
vi.mock('../../../src/main/clipboard', () => ({}));
vi.mock('../../../src/main/clipboardImage', () => ({}));
vi.mock('../../../src/main/fileWatcher', () => ({ addBrowserWindow: vi.fn() }));
vi.mock('../../../src/main/platform', () => ({
    getRootDir: () => 'root',
    installDir: 'install',
}));
vi.mock('../../../src/main/SaveFile', () => ({ openSaveFile: vi.fn() }));
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
