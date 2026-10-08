import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import { getBackend } from '../../../src/common/Backend';
import { CategoryId, NodeGroupId, NodeSchema, SchemaId } from '../../../src/common/common-types';
import { delay } from '../../../src/common/util';
import { RunArguments } from '../../../src/main/arguments';
import { setupBackend } from '../../../src/main/backend/setup';
import { Exit } from '../../../src/main/cli/exit';
import { runChainInCli } from '../../../src/main/cli/run';

const { exit, readSaveFile } = vi.hoisted(() => ({ exit: vi.fn(), readSaveFile: vi.fn() }));

vi.mock('electron/main', () => ({ app: { exit } }));
vi.mock('eventsource', () => ({
    default: class {
        addEventListener = vi.fn();

        close = vi.fn();
    },
}));
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
vi.mock('../../../src/main/SaveFile', () => ({ SaveFile: { read: readSaveFile } }));
vi.mock('../../../src/main/setting-storage', async () => {
    const { defaultSettings } = await import('../../../src/common/settings/settings');
    return { readSettings: () => defaultSettings };
});

const schema: NodeSchema = {
    name: 'Effect',
    category: 'test' as CategoryId,
    nodeGroup: 'test' as NodeGroupId,
    description: '',
    seeAlso: [],
    icon: '',
    kind: 'regularNode',
    inputs: [],
    outputs: [],
    groupLayout: [],
    iteratorInputs: [],
    iteratorOutputs: [],
    suggestions: [],
    schemaId: 'test:effect' as SchemaId,
    hasSideEffects: true,
    deprecated: false,
    features: [],
};

const args: RunArguments = {
    command: 'run',
    file: 'chain.chn',
    overrideFile: undefined,
    remoteBackend: undefined,
    refresh: false,
};

const createOwnedBackendProcess = () => {
    const errorListeners: ((error: Error) => void)[] = [];
    return {
        owned: true as const,
        url: 'http://127.0.0.1:1',
        errorListeners,
        addErrorListener: (listener: (error: Error) => void) => {
            errorListeners.push(listener);
        },
        clearErrorListeners: vi.fn(() => {
            errorListeners.length = 0;
        }),
        tryKill: vi.fn(() => Promise.resolve()),
    };
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
    let sigintListeners = 0;

    beforeEach(() => {
        now = 0;
        vi.spyOn(Date, 'now').mockImplementation(() => now);
        vi.mocked(delay).mockImplementation((ms) => {
            now += ms;
            return Promise.resolve();
        });
        readSaveFile.mockResolvedValue({
            nodes: [
                {
                    id: 'node',
                    type: 'regularNode',
                    position: { x: 0, y: 0 },
                    data: { id: 'node', schemaId: schema.schemaId, inputData: {} },
                },
            ],
            edges: [],
            viewport: { x: 0, y: 0, zoom: 1 },
            tamperedWith: false,
        });
        sigintListeners = process.listenerCount('SIGINT');
    });

    afterEach(() => {
        expect(process.listenerCount('SIGINT')).toBe(sigintListeners);
        vi.restoreAllMocks();
        exit.mockReset();
    });

    test('stops the backend after a successful run', async () => {
        const backendProcess = createOwnedBackendProcess();
        mockBackendProcess(backendProcess);
        const run = vi.fn().mockResolvedValue({ type: 'success' });
        mockBackend({ nodes: () => Promise.resolve({ nodes: [schema] }), run });

        await runChainInCli(args);

        expect(run).toHaveBeenCalledTimes(1);
        expect(backendProcess.tryKill).toHaveBeenCalledTimes(1);
    });

    test('stops the backend when the chain fails', async () => {
        const backendProcess = createOwnedBackendProcess();
        mockBackendProcess(backendProcess);
        mockBackend({
            nodes: () => Promise.resolve({ nodes: [schema] }),
            run: () =>
                Promise.resolve({
                    type: 'error',
                    message: 'Error',
                    exception: 'boom',
                    source: null,
                }),
        });

        await expect(runChainInCli(args)).rejects.toBeInstanceOf(Exit);

        expect(backendProcess.tryKill).toHaveBeenCalledTimes(1);
    });

    test('retries with a capped back-off for 5 minutes, then stops the backend', async () => {
        const backendProcess = createOwnedBackendProcess();
        mockBackendProcess(backendProcess);
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
        expect(backendProcess.tryKill).toHaveBeenCalledTimes(1);
    });

    test('stops the backend, then exits, on Ctrl+C', async () => {
        const backendProcess = createOwnedBackendProcess();
        mockBackendProcess(backendProcess);
        let finishRun = () => {};
        const run = vi.fn(
            () =>
                new Promise((resolve) => {
                    finishRun = () => resolve({ type: 'success' });
                })
        );
        mockBackend({ nodes: () => Promise.resolve({ nodes: [schema] }), run });

        const running = runChainInCli(args);
        await vi.waitFor(() => expect(run).toHaveBeenCalled());
        process.emit('SIGINT');
        await vi.waitFor(() => expect(exit).toHaveBeenCalledWith(1));

        expect(backendProcess.tryKill).toHaveBeenCalledTimes(1);
        expect(backendProcess.tryKill.mock.invocationCallOrder[0]).toBeLessThan(
            exit.mock.invocationCallOrder[0]
        );

        finishRun();
        await running;
        expect(backendProcess.tryKill).toHaveBeenCalledTimes(1);
    });

    test('stops the backend, then exits, when the backend process fails', async () => {
        const backendProcess = createOwnedBackendProcess();
        mockBackendProcess(backendProcess);
        let finishRun = () => {};
        const run = vi.fn(
            () =>
                new Promise((resolve) => {
                    finishRun = () => resolve({ type: 'success' });
                })
        );
        mockBackend({ nodes: () => Promise.resolve({ nodes: [schema] }), run });

        const running = runChainInCli(args);
        await vi.waitFor(() => expect(run).toHaveBeenCalled());
        backendProcess.errorListeners[0](new Error('spawn failed'));
        await vi.waitFor(() => expect(exit).toHaveBeenCalledWith(1));

        expect(backendProcess.tryKill).toHaveBeenCalledTimes(1);

        finishRun();
        await running;
    });

    test('leaves a remote backend running', async () => {
        mockBackendProcess({ owned: false, url: 'http://127.0.0.1:1' });
        const run = vi.fn().mockResolvedValue({ type: 'success' });
        mockBackend({ nodes: () => Promise.resolve({ nodes: [schema] }), run });

        await runChainInCli({ ...args, remoteBackend: 'http://127.0.0.1:1' });

        expect(run).toHaveBeenCalledTimes(1);
    });
});
