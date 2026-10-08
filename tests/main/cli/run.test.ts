import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import { getBackend } from '../../../src/common/Backend';
import { delay } from '../../../src/common/util';
import { RunArguments } from '../../../src/main/arguments';
import { setupBackend } from '../../../src/main/backend/setup';
import { runChainInCli } from '../../../src/main/cli/run';

vi.mock('electron/main', () => ({ app: { exit: vi.fn() } }));
vi.mock('../../../src/common/Backend', async (importOriginal) => ({
    ...(await importOriginal<typeof import('../../../src/common/Backend')>()),
    getBackend: vi.fn(),
}));
vi.mock('../../../src/common/util', async (importOriginal) => ({
    ...(await importOriginal<typeof import('../../../src/common/util')>()),
    delay: vi.fn(),
}));
vi.mock('../../../src/main/backend/setup', () => ({ setupBackend: vi.fn() }));
vi.mock('../../../src/main/platform', () => ({ getRootDir: () => 'root' }));
vi.mock('../../../src/main/setting-storage', async () => {
    const { defaultSettings } = await import('../../../src/common/settings/settings');
    return { readSettings: () => defaultSettings };
});

const args: RunArguments = {
    command: 'run',
    file: 'chain.chn',
    overrideFile: undefined,
    remoteBackend: undefined,
    refresh: false,
};

const mockBackend = (backend: Record<string, unknown>) => {
    vi.mocked(getBackend).mockReturnValue(backend as unknown as ReturnType<typeof getBackend>);
};
const mockBackendProcess = (backendProcess: object) => {
    vi.mocked(setupBackend).mockResolvedValue(
        backendProcess as Awaited<ReturnType<typeof setupBackend>>
    );
};

describe('runChainInCli', () => {
    let now = 0;

    beforeEach(() => {
        now = 0;
        vi.spyOn(Date, 'now').mockImplementation(() => now);
        vi.mocked(delay).mockImplementation((ms) => {
            now += ms;
            return Promise.resolve();
        });
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    test('retries with a capped back-off for 5 minutes', async () => {
        mockBackendProcess({ owned: false, url: 'http://127.0.0.1:1' });
        mockBackend({ nodes: () => Promise.reject(new Error('connect ECONNREFUSED')) });

        const running = runChainInCli(args);
        await expect(running).rejects.toThrow('Unable to connect to backend server');

        const sleeps = vi.mocked(delay).mock.calls.map(([ms]) => ms);
        expect(sleeps.slice(0, 3)).toEqual([1, 2, 4]);
        expect(Math.max(...sleeps)).toBe(250);
        expect(now).toBeGreaterThanOrEqual(5 * 60 * 1000);
        expect(now).toBeLessThan(5 * 60 * 1000 + 250);
        await expect(running).rejects.toThrow(
            'Unable to connect to backend server: Error: connect ECONNREFUSED'
        );
    });
});
