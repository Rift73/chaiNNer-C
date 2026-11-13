import { beforeEach, expect, test, vi } from 'vitest';
import { OpenArguments } from '../../../src/main/arguments';
import { createGuiApp } from '../../../src/main/gui/create';

const { Menu, app, instanceLock, protocol } = vi.hoisted(() => ({
    Menu: { setApplicationMenu: vi.fn() },
    protocol: { registerSchemesAsPrivileged: vi.fn() },
    app: {
        disableHardwareAcceleration: vi.fn(),
        on: vi.fn(),
        quit: vi.fn(),
        requestSingleInstanceLock: vi.fn(),
    },
    instanceLock: { held: true },
}));

vi.mock('electron/main', () => ({ Menu, app, dialog: {}, net: {}, protocol }));
vi.mock('electron-log', () => ({ default: { errorHandler: { startCatching: vi.fn() } } }));
vi.mock('../../../src/main/gui/main-window', () => ({ createMainWindow: vi.fn() }));
vi.mock('../../../src/main/setting-storage', () => ({
    readSettings: () => ({ allowMultipleInstances: false, hardwareAcceleration: true }),
}));

const args: OpenArguments = {
    command: 'open',
    devtools: false,
    file: undefined,
    refresh: false,
    remoteBackend: undefined,
};

beforeEach(() => {
    vi.clearAllMocks();
    app.requestSingleInstanceLock.mockImplementation(() => instanceLock.held);
});

test('no default menu gets built: the application menu is empty until the window sets its own', () => {
    instanceLock.held = true;

    createGuiApp(args);

    expect(Menu.setApplicationMenu).toHaveBeenCalledOnce();
    expect(Menu.setApplicationMenu).toHaveBeenCalledWith(null);
    expect(app.on).toHaveBeenCalledWith('ready', expect.any(Function));
});

test('the renderer scheme is registered before the app can be ready, as Electron requires', () => {
    instanceLock.held = true;

    createGuiApp(args);

    expect(protocol.registerSchemesAsPrivileged).toHaveBeenCalledOnce();
    const [registered] = protocol.registerSchemesAsPrivileged.mock.invocationCallOrder;
    const [listening] = app.on.mock.invocationCallOrder;
    expect(registered).toBeLessThan(listening);
});

test('a second instance quits without touching the menu', () => {
    instanceLock.held = false;

    createGuiApp(args);

    expect(app.quit).toHaveBeenCalledOnce();
    expect(Menu.setApplicationMenu).not.toHaveBeenCalled();
});
