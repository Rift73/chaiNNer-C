import { constants, mkdirSync, renameSync, rmSync, writeFileSync } from 'fs';
import fs from 'fs/promises';
import path from 'path';

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

/**
 * Writes a file through a temporary file in the same folder and a rename, so a reader finds
 * either the old content or the new one, never a partial file (a crash or a second instance
 * writing at the same time).
 */
export const writeFileAtomic = async (
    file: string,
    content: string | Uint8Array
): Promise<void> => {
    await fs.mkdir(path.dirname(file), { recursive: true });
    const temp = `${file}.${process.pid}.tmp`;
    try {
        await fs.writeFile(temp, content, 'utf-8');
        await fs.rename(temp, file);
    } catch (error) {
        await fs.rm(temp, { force: true });
        throw error;
    }
};

/** `writeFileAtomic` for a write that must be done before the app goes on (e.g. at its start). */
export const writeFileAtomicSync = (file: string, content: string): void => {
    mkdirSync(path.dirname(file), { recursive: true });
    const temp = `${file}.${process.pid}.tmp`;
    try {
        writeFileSync(temp, content, 'utf-8');
        renameSync(temp, file);
    } catch (error) {
        rmSync(temp, { force: true });
        throw error;
    }
};
