import { mkdtempSync, rmSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import path from 'path';
import { expect, test, vi } from 'vitest';
import { delay } from '../../src/common/util';
import { addBrowserWindow, addFile, removeFile } from '../../src/main/fileWatcher';
import { BrowserWindowWithSafeIpc } from '../../src/main/safeIpc';

// Deleting the folder of a watched file makes Windows report EPERM for the watch (upstream chaiNNer
// #3302; #3013 is the same with UNKNOWN). That error must not be uncaught: vitest fails the run
// on any uncaught exception.
test('a watched file whose folder is deleted reports unlink, not an uncaught error', async () => {
    const events: unknown[][] = [];
    addBrowserWindow({
        webContents: { send: (...args: unknown[]) => events.push(args) },
    } as unknown as BrowserWindowWithSafeIpc);

    const dir = mkdtempSync(path.join(tmpdir(), 'file-watcher-test-'));
    const file = path.join(dir, 'image.png');
    writeFileSync(file, 'x');
    addFile(file);
    await delay(500);

    rmSync(dir, { recursive: true, force: true });
    await vi.waitFor(() => expect(events).toContainEqual(['file-changed', 'unlink', file]), {
        timeout: 5000,
    });
    await delay(500);

    removeFile(file);
});
