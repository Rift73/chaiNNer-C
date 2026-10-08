import { mkdtemp, readdir, rm } from 'fs/promises';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

const { defaultSession, proxySession, fromPartition } = vi.hoisted(() => {
    const createSession = () => ({
        fetch: vi.fn<[string], Promise<Response>>(),
        setProxy: vi.fn(() => Promise.resolve()),
    });
    const proxy = createSession();
    return {
        defaultSession: createSession(),
        proxySession: proxy,
        fromPartition: vi.fn(() => proxy),
    };
});

vi.mock('electron/main', () => ({
    app: { whenReady: () => Promise.resolve(), getPath: () => '' },
    session: { defaultSession, fromPartition },
}));

let directory: string;

beforeEach(async () => {
    directory = await mkdtemp(path.join(os.tmpdir(), 'chaiNNer-python-download-test-'));
});

afterEach(async () => {
    vi.unstubAllEnvs();
    vi.resetModules();
    vi.clearAllMocks();
    await rm(directory, { recursive: true, force: true });
});

test('HTTPS_PROXY routes the download through that proxy', async () => {
    vi.stubEnv('HTTPS_PROXY', '127.0.0.1:3128/');
    vi.stubEnv('NO_PROXY', 'localhost');
    proxySession.fetch.mockResolvedValue(
        new Response('Not Found', { status: 404, statusText: 'Not Found' })
    );

    const { getIntegratedPython } = await import('../../src/main/python/integratedPython');

    await expect(getIntegratedPython(directory, () => {})).rejects.toThrow(
        /^Downloading https:\/\/github\.com\/.+ failed: HTTP 404 Not Found$/
    );
    expect(fromPartition).toHaveBeenCalledWith('integrated-python-download');
    expect(proxySession.setProxy).toHaveBeenCalledWith({
        proxyRules: 'http://127.0.0.1:3128',
        proxyBypassRules: 'localhost',
    });
    expect(defaultSession.fetch).not.toHaveBeenCalled();
});

test('without HTTPS_PROXY the download reports progress and removes a partial file', async () => {
    vi.stubEnv('HTTPS_PROXY', '');
    let pulls = 0;
    const body = new ReadableStream<Uint8Array>({
        pull: (controller) => {
            pulls += 1;
            if (pulls === 1) {
                controller.enqueue(new Uint8Array(400));
            } else {
                controller.error(new Error('net::ERR_CONNECTION_CLOSED'));
            }
        },
    });
    defaultSession.fetch.mockResolvedValue(
        new Response(body, { headers: { 'content-length': '1000' } })
    );

    const { getIntegratedPython } = await import('../../src/main/python/integratedPython');
    const progress: [number, string][] = [];

    await expect(
        getIntegratedPython(directory, (percentage, stage) => progress.push([percentage, stage]))
    ).rejects.toThrow('net::ERR_CONNECTION_CLOSED');
    expect(fromPartition).not.toHaveBeenCalled();
    expect(progress).toEqual([
        [0, 'download'],
        [40, 'download'],
    ]);
    expect(await readdir(directory)).toEqual([]);
});
