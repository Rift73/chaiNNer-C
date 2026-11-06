import { mkdtemp, rm } from 'fs/promises';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

const { app, loaded, runChainInCli, state } = vi.hoisted(() => {
    const hoistedState = { dir: '' };
    return {
        state: hoistedState,
        loaded: new Set<string>(),
        runChainInCli: vi.fn(() => Promise.resolve()),
        app: {
            isPackaged: false,
            disableHardwareAcceleration: vi.fn(),
            exit: vi.fn(),
            getPath: () => hoistedState.dir,
            getVersion: () => '0.0.0',
            on: vi.fn(),
            quit: vi.fn(),
            requestSingleInstanceLock: () => true,
            setPath: vi.fn(),
        },
    };
});

const electron = {
    app,
    BrowserWindow: class {},
    Menu: { buildFromTemplate: vi.fn(), setApplicationMenu: vi.fn() },
    clipboard: {},
    dialog: {},
    ipcMain: { handle: vi.fn(), on: vi.fn() },
    nativeImage: {},
    nativeTheme: {},
    powerSaveBlocker: {},
    shell: {},
};
vi.mock('electron', () => electron);
vi.mock('electron/main', () => electron);
vi.mock('electron/common', () => electron);

const electronLog = {
    catchErrors: vi.fn(),
    debug: vi.fn(),
    error: vi.fn(),
    errorHandler: { startCatching: vi.fn() },
    info: vi.fn(),
    initialize: vi.fn(),
    transports: { console: {}, file: {} },
};
vi.mock('electron-log', () => ({ default: electronLog }));
vi.mock('electron-log/main', () => ({ default: electronLog }));

vi.mock('../../src/main/cli/run', () => {
    loaded.add('command-line run');
    return { runChainInCli };
});
vi.mock('reactflow', () => {
    loaded.add('React Flow');
    return {};
});
vi.mock('@chainner/navi', () => {
    loaded.add('type engine');
    return {};
});
vi.mock('rregex/lib/web', () => {
    loaded.add('regex WebAssembly');
    return { default: () => Promise.resolve(), RRegex: class {} };
});

const start = async (...args: string[]) => {
    process.argv = ['electron', 'main.js', ...args];
    await import('../../src/main/main');
};

describe('main process start', () => {
    const { argv } = process;

    beforeEach(async () => {
        state.dir = await mkdtemp(path.join(os.tmpdir(), 'chaiNNer-main-test-'));
        loaded.clear();
        vi.clearAllMocks();
        vi.resetModules();
        vi.stubGlobal('BUILD_TREE', '');
        vi.stubGlobal('BUILD_DATE', '');
        vi.stubGlobal('BUILD_DIRTY', false);
    });

    afterEach(async () => {
        vi.unstubAllGlobals();
        process.argv = argv;
        await rm(state.dir, { recursive: true, force: true });
    });

    test('opening a window loads neither the command-line run nor what only it needs', async () => {
        await start();

        expect(app.on).toHaveBeenCalledWith('ready', expect.any(Function));
        expect([...loaded]).toEqual([]);
    });

    test('run loads the command-line run and gives it the arguments', async () => {
        await start('run', 'chain.chn');

        await vi.waitFor(() => expect(app.exit).toHaveBeenCalledWith(0));
        expect(runChainInCli).toHaveBeenCalledTimes(1);
        expect(runChainInCli).toHaveBeenCalledWith(
            expect.objectContaining({ command: 'run', file: 'chain.chn' })
        );
    });

    test('a failed run exits with an error', async () => {
        runChainInCli.mockRejectedValueOnce(new Error('failed'));

        await start('run', 'chain.chn');

        await vi.waitFor(() => expect(app.exit).toHaveBeenCalledWith(1));
    });
});
