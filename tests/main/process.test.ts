import { mkdir, mkdtemp, rm, writeFile } from 'fs/promises';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { delay } from '../../src/common/util';

const { shutdown } = vi.hoisted(() => ({ shutdown: vi.fn<[], Promise<void>>() }));

vi.mock('electron/main', () => ({ app: { getAppPath: () => '' } }));
vi.mock('../../src/common/Backend', () => ({ getBackend: () => ({ shutdown }) }));

// The backend is started as `<python> <resourcesPath>/src/run.py <port>`. Node stands in for
// Python here, and run.py is a script that behaves like the backend under test.
let resourcesPath: string;

const spawnBackend = async (script: string) => {
    await mkdir(path.join(resourcesPath, 'src'));
    await writeFile(path.join(resourcesPath, 'src', 'run.py'), script);

    const { OwnedBackendProcess } = await import('../../src/main/backend/process');
    return OwnedBackendProcess.spawn({
        port: 1,
        python: { python: process.execPath, version: '3.14.8' },
    });
};

beforeEach(async () => {
    resourcesPath = await mkdtemp(path.join(os.tmpdir(), 'chaiNNer-process-test-'));
    Object.defineProperty(process, 'resourcesPath', { value: resourcesPath, configurable: true });
});

afterEach(async () => {
    Reflect.deleteProperty(process, 'resourcesPath');
    vi.resetModules();
    shutdown.mockReset();
    await rm(resourcesPath, { recursive: true, force: true });
});

test('a backend that stops on its own is reported with its exit code and last output', async () => {
    const backend = await spawnBackend(`
        for (let i = 1; i <= 15; i += 1) {
            process.stderr.write('line ' + i + '\\n');
        }
        process.stderr.write("ModuleNotFoundError: No module named 'numpy'\\n");
        process.exitCode = 3;
    `);

    const exit = await new Promise((resolve) => {
        backend.addExitListener(resolve);
    });

    expect(exit).toEqual({
        code: 3,
        signal: null,
        stderrTail: [
            ...Array.from({ length: 11 }, (_, i) => `line ${i + 5}`),
            "ModuleNotFoundError: No module named 'numpy'",
        ].join('\n'),
    });
});

test('a backend stopped by kill() is not reported, even if it exits during the shutdown request', async () => {
    // like the real backend, the process exits by itself while kill() waits for /shutdown
    shutdown.mockImplementation(() => delay(1000));
    const backend = await spawnBackend('setTimeout(() => {}, 100);');
    const exitListener = vi.fn();
    backend.addExitListener(exitListener);

    await backend.kill();

    expect(shutdown).toHaveBeenCalledOnce();
    expect(exitListener).not.toHaveBeenCalled();
});
