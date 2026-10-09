import { constants } from 'fs';
import fs from 'fs/promises';

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
