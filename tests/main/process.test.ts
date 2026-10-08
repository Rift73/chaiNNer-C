import { mkdir, mkdtemp, readFile, rm, writeFile } from 'fs/promises';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { PythonInfo } from '../../src/common/common-types';
import { delay } from '../../src/common/util';
import { SHUTDOWN_TIMEOUT_MS } from '../../src/main/backend/process';

const { shutdown, pythonInfo } = vi.hoisted(() => ({
    shutdown: vi.fn<[], Promise<void>>(),
    pythonInfo: vi.fn<[], Promise<PythonInfo>>(),
}));

vi.mock('electron/main', () => ({ app: { getAppPath: () => '' } }));
vi.mock('../../src/common/Backend', () => ({ getBackend: () => ({ shutdown, pythonInfo }) }));

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
    vi.useRealTimers();
    Reflect.deleteProperty(process, 'resourcesPath');
    vi.resetModules();
    shutdown.mockReset();
    pythonInfo.mockReset();
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

// A backend that runs until it is killed, and says which process it is.
const runForeverScript = `
    require('fs').writeFileSync(require('path').join(__dirname, 'pid'), String(process.pid));
    setInterval(() => {}, 1000);
`;

const readPid = async (): Promise<number> => {
    const pidFile = path.join(resourcesPath, 'src', 'pid');
    return vi.waitFor(async () => Number(await readFile(pidFile, 'utf-8')), { timeout: 5000 });
};

const isRunning = (pid: number): boolean => {
    try {
        process.kill(pid, 0);
        return true;
    } catch {
        return false;
    }
};

const expectStopped = async (pid: number) => {
    try {
        await vi.waitFor(() => expect(isRunning(pid)).toBe(false), { timeout: 5000 });
    } finally {
        // don't leave the process behind when the test fails
        if (isRunning(pid)) {
            process.kill(pid);
        }
    }
};

test('kill() ends the process when the backend cannot be asked to shut down', async () => {
    // e.g. the backend is still installing its dependencies and not listening yet
    shutdown.mockRejectedValue(new Error('connect ECONNREFUSED 127.0.0.1:1'));
    const backend = await spawnBackend(runForeverScript);
    const pid = await readPid();

    await backend.kill();

    expect(shutdown).toHaveBeenCalledOnce();
    await expectStopped(pid);
});

test('kill() ends the process when the shutdown request gets no answer', async () => {
    shutdown.mockImplementation(() => new Promise(() => {}));
    const backend = await spawnBackend(runForeverScript);
    const pid = await readPid();

    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    const killing = backend.kill();
    await vi.advanceTimersByTimeAsync(SHUTDOWN_TIMEOUT_MS);
    await killing;
    vi.useRealTimers();

    await expectStopped(pid);
});

test('a borrowed backend asks for its Python only until the backend answers', async () => {
    const python: PythonInfo = { python: 'python.exe', version: '3.14.8' };
    // the backend is still starting at the first try
    pythonInfo.mockRejectedValueOnce(new Error('connect ECONNREFUSED')).mockResolvedValue(python);

    const { BorrowedBackendProcess } = await import('../../src/main/backend/process');
    const backend = await BorrowedBackendProcess.fromUrl('http://127.0.0.1:8000');

    expect(backend.python).toEqual(python);
    expect(pythonInfo).toHaveBeenCalledTimes(2);
});
