import deepEqual from 'fast-deep-equal';
import { existsSync, readFileSync, renameSync } from 'fs';
import { readFile } from 'fs/promises';
import { LocalStorage } from 'node-localstorage';
import path from 'path';
import { log } from '../common/log';
import { migrateOldStorageSettings, migrateSettings } from '../common/settings/migration';
import { ChainnerSettings, defaultSettings } from '../common/settings/settings';
import { getRootDir } from './platform';
import { writeFileAtomic, writeFileAtomicSync } from './util';

const getSettingsJson = (rootDir: string): string => path.join(rootDir, 'settings.json');

const stringifySettings = (settings: ChainnerSettings): string => JSON.stringify(settings, null, 4);

/**
 * The settings object in the content of settings.json, or undefined if it holds none (e.g. a
 * file cut short while an older version wrote it in place). A byte order mark (a file written by
 * Windows PowerShell 5.1 with `-Encoding utf8`) is not part of the JSON.
 */
const parseSettingsJson = (content: string): Partial<ChainnerSettings> | undefined => {
    let value: unknown;
    try {
        value = JSON.parse(content.replace(/^\uFEFF/, ''));
    } catch {
        return undefined;
    }
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
        return undefined;
    }
    return value as Partial<ChainnerSettings>;
};

export const readSettings = (rootDir: string = getRootDir()): ChainnerSettings => {
    const settingsJson = getSettingsJson(rootDir);
    if (existsSync(settingsJson)) {
        // settings.json
        const partialSettings = parseSettingsJson(readFileSync(settingsJson, 'utf-8'));
        if (partialSettings === undefined) {
            // keep the file for the user next to the new one, and start with the defaults
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const invalidCopy = `${settingsJson}.invalid-${timestamp}`;
            renameSync(settingsJson, invalidCopy);
            log.error(
                `${settingsJson} holds no valid settings. It was moved to ${invalidCopy}, and the default settings are used.`
            );
            return { ...defaultSettings };
        }
        const originalPartialSettings = { ...partialSettings };
        const settings = migrateSettings(partialSettings);

        if (!deepEqual(originalPartialSettings, settings)) {
            // write settings if they aren't up to date
            writeFileAtomicSync(settingsJson, stringifySettings(settings));
        }

        return settings;
    }

    // legacy settings
    const storagePath = path.join(rootDir, 'settings');
    if (!existsSync(storagePath)) {
        // neither settings.json nor old settings exist, so this is a fresh install
        return { ...defaultSettings };
    }

    const storage = new LocalStorage(storagePath);
    const partialSettings = migrateOldStorageSettings({
        keys: Array.from({ length: storage.length }, (_, i) => storage.key(i)),
        getItem: (key: string) => storage.getItem(key),
    });
    const settings = migrateSettings(partialSettings);

    // write a new settings.json we'll use form now on
    writeFileAtomicSync(settingsJson, stringifySettings(settings));
    // don't delete the old settings in case we need to revert
    renameSync(storagePath, path.join(rootDir, 'settings_old'));

    return settings;
};

/**
 * Writes the settings to settings.json in the background.
 *
 * A save waits `delay` ms, so a burst of changes (e.g. resizing the window) is one write. Writes
 * run one at a time, each through a temporary file and a rename, so a crash leaves the old
 * settings or the new ones, never a partial file. Settings equal to the file's are not written.
 */
export class SettingsWriter {
    private readonly settingsJson: string;

    private readonly delay: number;

    private pending: ChainnerSettings | undefined;

    private timer: NodeJS.Timeout | undefined;

    private writing: Promise<void> = Promise.resolve();

    private unfinishedWrites = 0;

    constructor(rootDir: string = getRootDir(), delay = 1000) {
        this.settingsJson = getSettingsJson(rootDir);
        this.delay = delay;
    }

    /** Whether every save so far is written (or failed and was logged). */
    get idle(): boolean {
        return this.pending === undefined && this.unfinishedWrites === 0;
    }

    /** Saves the settings once the delay is over; a later save before then replaces them. */
    save(settings: ChainnerSettings): void {
        this.pending = settings;
        this.timer ??= setTimeout(() => {
            this.flush().catch(log.error);
        }, this.delay);
    }

    /**
     * Writes the waiting save now. Resolves once every save so far is written; a write that fails
     * is logged and does not reject.
     */
    flush(): Promise<void> {
        clearTimeout(this.timer);
        this.timer = undefined;
        const settings = this.pending;
        this.pending = undefined;
        if (settings !== undefined) {
            this.unfinishedWrites += 1;
            this.writing = this.writing.then(() => this.write(settings));
        }
        return this.writing;
    }

    private async write(settings: ChainnerSettings): Promise<void> {
        try {
            const content = stringifySettings(settings);
            const current = await readFile(this.settingsJson, 'utf-8').catch(() => undefined);
            if (current !== content) {
                await writeFileAtomic(this.settingsJson, content);
            }
        } catch (error) {
            log.error('Unable to save settings.', error);
        } finally {
            this.unfinishedWrites -= 1;
        }
    }
}
