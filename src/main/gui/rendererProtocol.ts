import { net, protocol } from 'electron/main';
import path from 'path';
import { pathToFileURL } from 'url';
import { log } from '../../common/log';

const SCHEME = 'chainner-ui';
const HOST = 'app';

/**
 * Chromium keeps no V8 code cache for `file://` pages, so the window's scripts compile again on
 * every start. A standard scheme with the `codeCache` privilege gets one, checked against a hash of
 * the source text, so an update cannot run on an old entry. `secure` keeps the page a secure context
 * as `file://` is, which `navigator.clipboard` needs.
 *
 * The scheme is part of the page's origin: renaming it orphans what the origin has stored.
 *
 * Must be called before the app is ready.
 */
export const registerRendererScheme = (): void => {
    protocol.registerSchemesAsPrivileged([
        {
            scheme: SCHEME,
            privileges: {
                standard: true,
                secure: true,
                supportFetchAPI: true,
                corsEnabled: true,
                codeCache: true,
            },
        },
    ]);
};

/**
 * The file in `root` that a URL of the scheme stands for, or undefined if it stands for nothing
 * in there.
 */
export const getRendererFile = (root: string, url: string): string | undefined => {
    const { host, pathname } = new URL(url);
    if (host !== HOST) return undefined;

    let decoded;
    try {
        decoded = decodeURIComponent(pathname);
    } catch {
        // Malformed percent-encoding names no file. The caller logs the refusal.
        return undefined;
    }
    if (decoded.includes('\0')) return undefined;

    // The leading dot keeps a decoded absolute path, such as one of a UNC share, under the root.
    const file = path.resolve(root, `.${decoded}`);
    const relative = path.relative(root, file);
    const isInside =
        relative !== '' &&
        relative !== '..' &&
        !relative.startsWith(`..${path.sep}`) &&
        !path.isAbsolute(relative);
    return isInside ? file : undefined;
};

/**
 * Serves the built renderer in `root`, and nothing else, over the scheme. Returns the URL of its
 * page. Call once the app is ready.
 */
export const serveRenderer = (root: string): string => {
    protocol.handle(SCHEME, (request) => {
        const file = getRendererFile(root, request.url);
        if (file === undefined) {
            log.warn(`Refused a request for a file outside the renderer: ${request.url}`);
            return new Response('Not found', { status: 404 });
        }
        // Returned as it is: Electron gives an untouched fetch to its own loader, with the MIME
        // type Chromium knows for the extension. Module scripts and WebAssembly streaming need it.
        return net.fetch(pathToFileURL(file).toString());
    });
    return `${SCHEME}://${HOST}/index.html`;
};
