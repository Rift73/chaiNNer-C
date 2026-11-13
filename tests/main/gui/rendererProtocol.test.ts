import path from 'path';
import { fileURLToPath } from 'url';
import { beforeEach, describe, expect, test, vi } from 'vitest';
import {
    getRendererFile,
    registerRendererScheme,
    serveRenderer,
} from '../../../src/main/gui/rendererProtocol';

const { net, protocol } = vi.hoisted(() => ({
    net: { fetch: vi.fn() },
    protocol: { handle: vi.fn(), registerSchemesAsPrivileged: vi.fn() },
}));

vi.mock('electron/main', () => ({ net, protocol }));

const root = path.resolve('install dir #1', 'renderer', 'main_window');
const url = (rest: string) => `chainner-ui://app/${rest}`;

const handler = () => {
    const pageUrl = serveRenderer(root);
    const [scheme, handle] = protocol.handle.mock.calls[0] as [
        string,
        (request: { url: string }) => Promise<Response> | Response
    ];
    return { handle, pageUrl, scheme };
};

const attacks = [
    '..%2f..%2fsecret.txt',
    '..%2Fsecret.txt',
    '%2e%2e%2fsecret.txt',
    'assets/..%2f..%2fsecret.txt',
    '..%5c..%5csecret.txt',
    '%2e%2e%5csecret.txt',
    '..\\..\\secret.txt',
    '..',
    '..%2f',
    '%2e%2e',
    'bad%E0%A4%A',
    'index.js%00.png',
    '',
];

beforeEach(() => {
    vi.clearAllMocks();
});

describe('the scheme', () => {
    test('is standard and asks for the code cache, which needs standard', () => {
        registerRendererScheme();

        const [schemes] = protocol.registerSchemesAsPrivileged.mock.calls[0] as [
            { scheme: string; privileges: Record<string, boolean> }[]
        ];
        expect(schemes).toEqual([
            {
                scheme: 'chainner-ui',
                privileges: {
                    standard: true,
                    secure: true,
                    supportFetchAPI: true,
                    corsEnabled: true,
                    codeCache: true,
                },
            },
        ]);
    });
});

describe('getRendererFile', () => {
    test.each([
        ['index.html', 'index.html'],
        ['index.js', 'index.js'],
        ['rregex.wasm', 'rregex.wasm'],
        ['index.js?v=1#x', 'index.js'],
        ['Roboto%20Mono.ttf', 'Roboto Mono.ttf'],
        ['assets/a.png', path.join('assets', 'a.png')],
        ['..hidden.js', '..hidden.js'],
    ])('%s is a file of the renderer', (rest, expected) => {
        expect(getRendererFile(root, url(rest))).toBe(path.join(root, expected));
    });

    test.each(attacks)('%j is refused', (rest) => {
        expect(getRendererFile(root, url(rest))).toBeUndefined();
    });

    test.each(['../../secret.txt', '%2E%2E/%2E%2E/secret.txt'])(
        'the URL parser folds %s, so it stays in the renderer',
        (rest) => {
            expect(getRendererFile(root, url(rest))).toBe(path.join(root, 'secret.txt'));
        }
    );

    test.each(['//server/share/secret.txt', '%2F%2Fserver%2Fshare%2Fsecret.txt', 'C:/secret.txt'])(
        'the absolute path in %j stays under the renderer',
        (rest) => {
            const file = getRendererFile(root, url(rest));

            expect(file === undefined || file.startsWith(root + path.sep)).toBe(true);
        }
    );

    test('another host is refused', () => {
        expect(getRendererFile(root, 'chainner-ui://other/index.html')).toBeUndefined();
    });
});

describe('serveRenderer', () => {
    test('returns the page on the scheme it handles', () => {
        const { pageUrl, scheme } = handler();

        expect(scheme).toBe('chainner-ui');
        expect(pageUrl).toBe('chainner-ui://app/index.html');
        expect(getRendererFile(root, pageUrl)).toBe(path.join(root, 'index.html'));
    });

    test('hands back the fetch of the file untouched, from a path with a space and a #', () => {
        const response = new Response('');
        net.fetch.mockReturnValue(response);
        const { handle } = handler();

        expect(handle({ url: url('index.js') })).toBe(response);

        const [fetched] = net.fetch.mock.calls[0] as [string];
        expect(fileURLToPath(fetched)).toBe(path.join(root, 'index.js'));
    });

    test.each(attacks)('refuses %j without touching a file', async (rest) => {
        const { handle } = handler();

        const response = await handle({ url: url(rest) });

        expect(response.status).toBe(404);
        expect(net.fetch).not.toHaveBeenCalled();
    });
});
