import { expect, test, vi } from 'vitest';
import { setMainMenu } from '../../../src/main/gui/menu';
import { BrowserWindowWithSafeIpc } from '../../../src/main/safeIpc';
import type { MenuItemConstructorOptions } from 'electron/main';

const { Menu } = vi.hoisted(() => ({
    Menu: { buildFromTemplate: vi.fn(), setApplicationMenu: vi.fn() },
}));

vi.mock('electron/main', () => ({
    Menu,
    app: { getPath: () => '', isPackaged: true },
    dialog: {},
}));
vi.mock('electron/common', () => ({ shell: {} }));
vi.mock('../../../src/main/buildInfo', () => ({ buildInfo: {} }));
vi.mock('../../../src/main/systemInfo', () => ({}));

test('only the first nine recent files get a shortcut, CmdOrCtrl+1 to CmdOrCtrl+9', () => {
    const recent = Array.from({ length: 20 }, (_, i) => `chain ${i}.chn`);
    setMainMenu({
        mainWindow: {} as BrowserWindowWithSafeIpc,
        // oldest first, as the renderer sends it
        menuData: { openRecentRev: [...recent].reverse() },
        enabled: true,
    });

    const [template] = Menu.buildFromTemplate.mock.calls[0] as [MenuItemConstructorOptions[]];
    const file = template.find((item) => item.label === 'File');
    const openRecent = (file?.submenu as MenuItemConstructorOptions[]).find(
        (item) => item.label === 'Open Recent'
    );
    const accelerators = (openRecent?.submenu as MenuItemConstructorOptions[])
        .filter((item) => item.label?.endsWith('.chn'))
        .map((item) => item.accelerator);

    expect(accelerators).toEqual([
        ...Array.from({ length: 9 }, (_, i) => `CmdOrCtrl+${i + 1}`),
        ...Array<undefined>(11).fill(undefined),
    ]);
});
