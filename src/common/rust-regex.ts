import wasmUrl from 'rregex/lib/rregex.wasm?url';
import init, { RRegex as _rr } from 'rregex/lib/web';
import { log } from './log';

let loaded = false;

/**
 * Resolves once the regex WebAssembly has loaded, in the background: an RRegex made before then
 * fails, so the types that need one are not known until then (isRegexLoaded).
 */
export const regexLoaded: Promise<void> = init(wasmUrl).then(() => {
    loaded = true;
}, log.error);

export const isRegexLoaded = (): boolean => loaded;

export class RRegex extends _rr {}
