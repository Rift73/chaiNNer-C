import { mkdtemp, rm } from 'fs/promises';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';

const {
    app,
    checkUnfinishedUpdate,
    loaded,
    removeUnpackedReleases,
    runChainInCli,
    showUpdateAtStart,
    state,
} = vi.hoisted(() => {
    const hoistedState = { dir: '' };
    return {
        state: hoistedState,
        loaded: new Set<string>(),
        checkUnfinishedUpdate: vi.fn((): Promise<unknown> => Promise.resolve(undefined)),
        removeUnpackedReleases: vi.fn(),
        showUpdateAtStart: vi.fn((): Promise<boolean> => Promise.resolve(false)),
        runChainInCli: vi.fn(() => Promise.resolve()),
        app: {
            isPackaged: false,
            disableHardwareAcceleration: vi.fn(),
            exit: vi.fn(),
            getPath: () => hoistedState.dir,
            getVersion: () => '0.0.0',
            on: vi.fn(),
            quit: vi.fn(),
            relaunch: vi.fn(),
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
    net: {},
    powerSaveBlocker: {},
    protocol: { registerSchemesAsPrivileged: vi.fn() },
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

vi.mock('../../src/main/updater', () => ({
    checkUnfinishedUpdate,
    discardUpdate: vi.fn(),
    removeUnpackedReleases,
    showUpdateAtStart,
    stageUpdate: vi.fn(),
    startInstaller: vi.fn(),
    takeUpdateResult: vi.fn(),
}));

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

        await vi.waitFor(() => expect(app.on).toHaveBeenCalledWith('ready', expect.any(Function)));
        expect([...loaded]).toEqual([]);
    });

    test('an update that did not finish ends the start before a window or a run, after its message', async () => {
        const message = { title: 'The update did not finish', text: 'why' };
        checkUnfinishedUpdate.mockResolvedValueOnce(message);

        await start('run', 'chain.chn');

        // a run kept from running fails, as a failed run does
        await vi.waitFor(() => expect(app.exit).toHaveBeenCalledWith(1));
        expect(app.quit).not.toHaveBeenCalled();
        expect(showUpdateAtStart).toHaveBeenCalledWith(message);
        expect(app.on).not.toHaveBeenCalledWith('ready', expect.any(Function));
        expect(runChainInCli).not.toHaveBeenCalled();
        expect(app.relaunch).not.toHaveBeenCalled();
    });

    test('a release the user keeps at the start question starts the app again, with nothing left to resume', async () => {
        const afterwards = vi.fn(() => Promise.resolve());
        checkUnfinishedUpdate.mockResolvedValueOnce({ title: 't', text: 'x', afterwards });
        showUpdateAtStart.mockResolvedValueOnce(true);

        await start();

        await vi.waitFor(() => expect(app.quit).toHaveBeenCalledTimes(1));
        expect(app.relaunch).toHaveBeenCalledTimes(1);
        expect(afterwards).not.toHaveBeenCalled();
    });

    test('a release kept at the start of a command-line run does not repeat the run in another process', async () => {
        checkUnfinishedUpdate.mockResolvedValueOnce({ title: 't', text: 'x' });
        showUpdateAtStart.mockResolvedValueOnce(true);

        await start('run', 'chain.chn');

        await vi.waitFor(() => expect(app.exit).toHaveBeenCalledWith(1));
        expect(app.relaunch).not.toHaveBeenCalled();
        expect(runChainInCli).not.toHaveBeenCalled();
    });

    test('run loads the command-line run and gives it the arguments', async () => {
        await start('run', 'chain.chn');

        await vi.waitFor(() => expect(app.exit).toHaveBeenCalledWith(0));
        expect(runChainInCli).toHaveBeenCalledTimes(1);
        expect(runChainInCli).toHaveBeenCalledWith(
            expect.objectContaining({ command: 'run', file: 'chain.chn' })
        );
    });

    test('quitting removes the releases the updater unpacked, and nothing else', async () => {
        await start();

        expect(app.on).toHaveBeenCalledWith('quit', removeUnpackedReleases);
    });

    test('a failed run exits with an error', async () => {
        runChainInCli.mockRejectedValueOnce(new Error('failed'));

        await start('run', 'chain.chn');

        await vi.waitFor(() => expect(app.exit).toHaveBeenCalledWith(1));
    });
});
