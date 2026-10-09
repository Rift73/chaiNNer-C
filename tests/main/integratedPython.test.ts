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

// pip reads the lower-case name first; on Windows the two names are one variable, so the
// lower-case one is cleared before the upper-case one is set.
test('HTTPS_PROXY routes the download through that proxy', async () => {
    vi.stubEnv('https_proxy', '');
    vi.stubEnv('HTTPS_PROXY', '127.0.0.1:3128/');
    vi.stubEnv('no_proxy', '');
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

test('ALL_PROXY is the proxy when no HTTPS proxy is set', async () => {
    vi.stubEnv('https_proxy', '');
    vi.stubEnv('HTTPS_PROXY', '');
    vi.stubEnv('all_proxy', '');
    vi.stubEnv('ALL_PROXY', 'socks5://127.0.0.1:1080');
    proxySession.fetch.mockResolvedValue(new Response('Not Found', { status: 404 }));

    const { getIntegratedPython } = await import('../../src/main/python/integratedPython');

    await expect(getIntegratedPython(directory, () => {})).rejects.toThrow(/HTTP 404/);
    expect(proxySession.setProxy).toHaveBeenCalledWith(
        expect.objectContaining({ proxyRules: 'socks5://127.0.0.1:1080' })
    );
});

test.skipIf(process.platform === 'win32')(
    'the lower-case proxy variables take precedence, as for pip',
    async () => {
        vi.stubEnv('https_proxy', 'http://127.0.0.1:3128');
        vi.stubEnv('HTTPS_PROXY', 'http://127.0.0.1:8080');
        vi.stubEnv('no_proxy', 'localhost');
        vi.stubEnv('NO_PROXY', 'example.com');
        proxySession.fetch.mockResolvedValue(new Response('Not Found', { status: 404 }));

        const { getIntegratedPython } = await import('../../src/main/python/integratedPython');

        await expect(getIntegratedPython(directory, () => {})).rejects.toThrow(/HTTP 404/);
        expect(proxySession.setProxy).toHaveBeenCalledWith({
            proxyRules: 'http://127.0.0.1:3128',
            proxyBypassRules: 'localhost',
        });
    }
);

test('without a proxy variable the download reports progress and removes a partial file', async () => {
    for (const name of ['https_proxy', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']) {
        vi.stubEnv(name, '');
    }
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
