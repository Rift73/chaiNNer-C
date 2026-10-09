import { EventEmitter } from 'events';
import { mkdtemp, readdir, rm } from 'fs/promises';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';

const { defaultSession, proxySession, fromPartition, request } = vi.hoisted(() => {
    const createSession = () => ({ setProxy: vi.fn(() => Promise.resolve()) });
    const proxy = createSession();
    return {
        defaultSession: createSession(),
        proxySession: proxy,
        fromPartition: vi.fn(() => proxy),
        request: vi.fn(),
    };
});

vi.mock('electron/main', () => ({
    app: { whenReady: () => Promise.resolve(), getPath: () => '' },
    net: { request },
    session: { defaultSession, fromPartition },
}));

const incomingMessage = (
    statusCode: number,
    statusMessage: string,
    headers: Record<string, string> = {}
) => Object.assign(new EventEmitter(), { statusCode, statusMessage, headers });

/**
 * Answers the download's request with `response`; with `challenge`, the server or proxy first asks
 * for credentials, and the returned mock is the callback that receives them.
 */
const respondWith = (
    response: ReturnType<typeof incomingMessage>,
    challenge?: { isProxy: boolean }
) => {
    const login = vi.fn();
    request.mockImplementation(() => {
        const clientRequest = new EventEmitter();
        return Object.assign(clientRequest, {
            end: () => {
                setImmediate(() => {
                    if (challenge) {
                        clientRequest.emit('login', challenge, login);
                    }
                    clientRequest.emit('response', response);
                });
            },
        });
    });
    return login;
};

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
    const login = respondWith(incomingMessage(404, 'Not Found'), { isProxy: true });

    const { getIntegratedPython } = await import('../../src/main/python/integratedPython');

    await expect(getIntegratedPython(directory, () => {})).rejects.toThrow(
        /^Downloading https:\/\/github\.com\/.+ failed: HTTP 404 Not Found$/
    );
    expect(fromPartition).toHaveBeenCalledWith('integrated-python-download');
    expect(proxySession.setProxy).toHaveBeenCalledWith({
        proxyRules: 'http://127.0.0.1:3128',
        proxyBypassRules: 'localhost',
    });
    expect(request).toHaveBeenCalledWith(expect.objectContaining({ session: proxySession }));
    // no credentials to offer: the challenge is cancelled
    expect(login).toHaveBeenCalledWith();
});

test("the proxy's credentials answer its challenge, as for pip", async () => {
    vi.stubEnv('https_proxy', '');
    vi.stubEnv('HTTPS_PROXY', 'http://chai%40nner:p%3Ass@127.0.0.1:3128');
    const login = respondWith(incomingMessage(404, 'Not Found'), { isProxy: true });

    const { getIntegratedPython } = await import('../../src/main/python/integratedPython');

    await expect(getIntegratedPython(directory, () => {})).rejects.toThrow(/HTTP 404/);
    expect(proxySession.setProxy).toHaveBeenCalledWith(
        expect.objectContaining({ proxyRules: 'http://127.0.0.1:3128' })
    );
    expect(login).toHaveBeenCalledWith('chai@nner', 'p:ss');
});

test("a server's challenge gets no proxy credentials", async () => {
    vi.stubEnv('https_proxy', '');
    vi.stubEnv('HTTPS_PROXY', 'http://user:pass@127.0.0.1:3128');
    const login = respondWith(incomingMessage(401, 'Unauthorized'), { isProxy: false });

    const { getIntegratedPython } = await import('../../src/main/python/integratedPython');

    await expect(getIntegratedPython(directory, () => {})).rejects.toThrow(/HTTP 401/);
    expect(login).toHaveBeenCalledWith();
});

test('ALL_PROXY is the proxy when no HTTPS proxy is set', async () => {
    vi.stubEnv('https_proxy', '');
    vi.stubEnv('HTTPS_PROXY', '');
    vi.stubEnv('all_proxy', '');
    vi.stubEnv('ALL_PROXY', 'socks5://127.0.0.1:1080');
    respondWith(incomingMessage(404, 'Not Found'));

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
        respondWith(incomingMessage(404, 'Not Found'));

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
    const response = incomingMessage(200, 'OK', { 'content-length': '1000' });
    respondWith(response);

    const { getIntegratedPython } = await import('../../src/main/python/integratedPython');
    const progress: [number, string][] = [];

    const downloading = getIntegratedPython(directory, (percentage, stage) =>
        progress.push([percentage, stage])
    );
    await vi.waitFor(() => expect(response.listenerCount('error')).toBe(1));
    response.emit('data', new Uint8Array(400));
    await vi.waitFor(() => expect(progress).toContainEqual([40, 'download']));
    response.emit('error', new Error('net::ERR_CONNECTION_CLOSED'));

    await expect(downloading).rejects.toThrow('net::ERR_CONNECTION_CLOSED');
    expect(fromPartition).not.toHaveBeenCalled();
    expect(request).toHaveBeenCalledWith(expect.objectContaining({ session: defaultSession }));
    expect(progress).toEqual([
        [0, 'download'],
        [40, 'download'],
    ]);
    expect(await readdir(directory)).toEqual([]);
});
