import { FSWatcher } from 'chokidar';
import { log } from '../common/log';
import { BrowserWindowWithSafeIpc } from './safeIpc';

type EventType = 'add' | 'change' | 'unlink';

// How many times each watched file is watched
const watchCounts = new Map<string, number>();

const listenerWindows: Set<BrowserWindowWithSafeIpc> = new Set();

const watcher = new FSWatcher({
    disableGlobbing: true,
    awaitWriteFinish: true,
    ignoreInitial: true,
    // chokidar only handles the errors of persistent watches. A non-persistent watch throws an
    // uncaught error when a watched file's folder is deleted (EPERM) or its drive goes away.
    persistent: true,
});

const sendEvent = (window: BrowserWindowWithSafeIpc, eventType: EventType, path: string) => {
    window.webContents.send('file-changed', eventType, path);
};

const callListeners = (event: EventType, path: string) => {
    for (const window of listenerWindows) {
        sendEvent(window, event, path);
    }
};

watcher.on('error', (e) => log.error(e));
watcher.on('add', (path) => callListeners('add', path));
watcher.on('change', (path) => callListeners('change', path));
watcher.on('unlink', (path) => callListeners('unlink', path));

export const addBrowserWindow = (window: BrowserWindowWithSafeIpc) => {
    listenerWindows.add(window);
};

export const addFile = (path: string) => {
    const count = watchCounts.get(path) ?? 0;
    if (count === 0) {
        watcher.add(path);
    }
    watchCounts.set(path, count + 1);
};

export const addFiles = (paths: readonly string[]) => {
    for (const path of paths) {
        addFile(path);
    }
};

export const removeFile = (path: string) => {
    const count = watchCounts.get(path);
    if (count === 1) {
        watchCounts.delete(path);
        watcher.unwatch(path);
    } else if (count !== undefined) {
        watchCounts.set(path, count - 1);
    }
};

export const removeFiles = (paths: readonly string[]) => {
    for (const path of paths) {
        removeFile(path);
    }
};
