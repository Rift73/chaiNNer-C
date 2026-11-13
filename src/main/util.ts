import {
    closeSync,
    constants,
    fsyncSync,
    mkdirSync,
    openSync,
    renameSync,
    rmSync,
    writeFileSync,
} from 'fs';
import fs from 'fs/promises';
import path from 'path';
import { delay } from '../common/util';

export const checkFileExists = (file: string): Promise<boolean> =>
    fs.access(file, constants.F_OK).then(
        () => true,
        () => false
    );

/**
 * Reads a UTF-8 text file without its byte order mark, if it has one (e.g. a file written by
 * Windows PowerShell 5.1 with `-Encoding utf8`).
 */
export const readTextFile = async (file: string): Promise<string> => {
    const text = await fs.readFile(file, { encoding: 'utf-8' });
    return text.replace(/^\uFEFF/, '');
};

let tempSequence = 0;

/**
 * A temporary file beside `file`, unique to this call: two writes of one file that overlap (in
 * this process or another) never share a temporary file.
 */
const tempFileFor = (file: string): string => {
    tempSequence += 1;
    return `${file}.chainner-c-${process.pid}-${tempSequence}.tmp`;
};

/**
 * Windows refuses a rename onto a file for a moment while another rename replaces it, or while a
 * reader holds it open without delete sharing; such a rename is retried for up to a second.
 */
const RENAME_RETRY_MS = 1000;
const isTransientRenameError = (error: unknown): boolean =>
    ['EPERM', 'EACCES', 'EBUSY'].includes((error as NodeJS.ErrnoException).code ?? '');

const renameRetrying = async (
    from: string,
    to: string,
    deadline = Date.now() + RENAME_RETRY_MS,
    wait = 5
): Promise<void> => {
    try {
        await fs.rename(from, to);
    } catch (error) {
        if (!isTransientRenameError(error) || Date.now() >= deadline) {
            throw error;
        }
        await delay(wait);
        await renameRetrying(from, to, deadline, Math.min(wait * 2, 100));
    }
};

const renameRetryingSync = (from: string, to: string): void => {
    const deadline = Date.now() + RENAME_RETRY_MS;
    for (let wait = 5; ; wait = Math.min(wait * 2, 100)) {
        try {
            renameSync(from, to);
            return;
        } catch (error) {
            if (!isTransientRenameError(error) || Date.now() >= deadline) {
                throw error;
            }
            Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, wait);
        }
    }
};

/**
 * Writes a file through a temporary file in the same folder and a rename, so a reader finds
 * either the old content or the new one, never a partial file (a crash or a second instance
 * writing at the same time). The content is flushed to the disk before the rename: a rename can
 * reach it first, and a power loss would then leave the name with no content.
 */
export const writeFileAtomic = async (
    file: string,
    content: string | Uint8Array
): Promise<void> => {
    await fs.mkdir(path.dirname(file), { recursive: true });
    const temp = tempFileFor(file);
    try {
        const handle = await fs.open(temp, 'w');
        try {
            await handle.writeFile(content, 'utf-8');
            await handle.sync();
        } finally {
            await handle.close();
        }
        await renameRetrying(temp, file);
    } catch (error) {
        await fs.rm(temp, { force: true });
        throw error;
    }
};

/** `writeFileAtomic` for a write that must be done before the app goes on (e.g. at its start). */
export const writeFileAtomicSync = (file: string, content: string): void => {
    mkdirSync(path.dirname(file), { recursive: true });
    const temp = tempFileFor(file);
    try {
        const fd = openSync(temp, 'w');
        try {
            writeFileSync(fd, content, 'utf-8');
            fsyncSync(fd);
        } finally {
            closeSync(fd);
        }
        renameRetryingSync(temp, file);
    } catch (error) {
        rmSync(temp, { force: true });
        throw error;
    }
};

/** Whether `file` is `folder` or inside it (case-insensitively on Windows, as its paths are). */
export const isInFolder = (file: string, folder: string): boolean => {
    const relative = path.relative(folder, file);
    return !path.isAbsolute(relative) && relative.split(path.sep)[0] !== '..';
};

/**
 * Calls `attempt` until it resolves or `deadline` (a Date.now() time) has passed, waiting 1 ms after
 * the first failure and twice as long after each next one, at most 250 ms. Rejects with the last
 * attempt's error.
 */
export const retryUntil = async <T>(
    attempt: () => Promise<T>,
    deadline: number,
    wait = 1
): Promise<T> => {
    try {
        return await attempt();
    } catch (error) {
        if (Date.now() >= deadline) {
            throw error;
        }
        await delay(Math.min(wait, deadline - Date.now()));
        return retryUntil(attempt, deadline, Math.min(wait * 2, 250));
    }
};
