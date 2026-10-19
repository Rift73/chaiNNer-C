import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'fs';
import fsPromises from 'fs/promises';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { ChainnerSettings, defaultSettings } from '../../src/common/settings/settings';
import { SettingsWriter, readSettings } from '../../src/main/setting-storage';

vi.mock('electron/main', () => ({
    app: { getPath: () => os.tmpdir() },
}));

let rootDir: string;
let settingsJson: string;

beforeEach(() => {
    rootDir = mkdtempSync(path.join(os.tmpdir(), 'chaiNNer-settings-test-'));
    settingsJson = path.join(rootDir, 'settings.json');
});
afterEach(() => {
    vi.restoreAllMocks();
    rmSync(rootDir, { recursive: true, force: true });
});

const fileText = (settings: ChainnerSettings): string => JSON.stringify(settings, null, 4);
const withTheme = (theme: string): ChainnerSettings => ({ ...defaultSettings, theme });

test.each([
    ['cut short', '{\n    "theme": "dark",\n    "snapTo'],
    ['empty', ''],
    ['not an object', 'null'],
    ['a list', '[]'],
])('a settings.json that is %s starts with the defaults and keeps the file', (_, content) => {
    writeFileSync(settingsJson, content, 'utf-8');

    expect(readSettings(rootDir)).toEqual(defaultSettings);

    const files = readdirSync(rootDir);
    expect(files).toHaveLength(1);
    expect(files[0]).toMatch(/^settings\.json\.invalid-[\dT-]+Z$/);
    expect(readFileSync(path.join(rootDir, files[0]), 'utf-8')).toBe(content);
});

test('a settings.json with a byte order mark is read', () => {
    // as Windows PowerShell 5.1 writes it with `-Encoding utf8`
    writeFileSync(settingsJson, `\uFEFF${fileText(withTheme('default-light'))}`, 'utf-8');

    expect(readSettings(rootDir)).toEqual(withTheme('default-light'));
    expect(readdirSync(rootDir)).toEqual(['settings.json']);
});

test('outdated settings are migrated and written through a temporary file', () => {
    const { storage, ...partial } = withTheme('light');
    writeFileSync(settingsJson, JSON.stringify(partial), 'utf-8');

    const settings = readSettings(rootDir);

    expect(settings.storage).toEqual(storage);
    expect(readFileSync(settingsJson, 'utf-8')).toBe(fileText(settings));
    expect(readdirSync(rootDir)).toEqual(['settings.json']);
});

test('a burst of saves is one write of the last settings', async () => {
    const rename = vi.spyOn(fsPromises, 'rename');
    const writer = new SettingsWriter(rootDir, 60_000);

    writer.save(withTheme('dark'));
    writer.save(withTheme('light'));
    expect(writer.idle).toBe(false);
    await writer.flush();

    expect(writer.idle).toBe(true);
    expect(rename).toHaveBeenCalledOnce();
    expect(readFileSync(settingsJson, 'utf-8')).toBe(fileText(withTheme('light')));
    expect(readdirSync(rootDir)).toEqual(['settings.json']);
});

test('a save is written once the delay is over', async () => {
    const writer = new SettingsWriter(rootDir, 10);

    writer.save(withTheme('light'));

    await vi.waitFor(() => {
        expect(existsSync(settingsJson)).toBe(true);
        expect(writer.idle).toBe(true);
    });
    expect(readFileSync(settingsJson, 'utf-8')).toBe(fileText(withTheme('light')));
});

test('settings equal to the file are not written again', async () => {
    writeFileSync(settingsJson, fileText(withTheme('light')), 'utf-8');
    const rename = vi.spyOn(fsPromises, 'rename');
    const writer = new SettingsWriter(rootDir, 60_000);

    writer.save(withTheme('light'));
    await writer.flush();

    expect(rename).not.toHaveBeenCalled();
});

test('a failed write keeps the old file, does not reject, and later saves are written', async () => {
    writeFileSync(settingsJson, fileText(withTheme('dark')), 'utf-8');
    vi.spyOn(fsPromises, 'rename').mockRejectedValueOnce(
        Object.assign(new Error('EIO: rename'), { code: 'EIO' })
    );
    const writer = new SettingsWriter(rootDir, 60_000);

    writer.save(withTheme('light'));
    await writer.flush();
    expect(writer.idle).toBe(true);
    expect(readFileSync(settingsJson, 'utf-8')).toBe(fileText(withTheme('dark')));
    expect(readdirSync(rootDir)).toEqual(['settings.json']);

    writer.save(withTheme('light'));
    await writer.flush();
    expect(readFileSync(settingsJson, 'utf-8')).toBe(fileText(withTheme('light')));
});

test('a save during a write is written after it', async () => {
    const writer = new SettingsWriter(rootDir, 60_000);

    writer.save(withTheme('dark'));
    const first = writer.flush();
    writer.save(withTheme('light'));
    const second = writer.flush();
    await Promise.all([first, second]);

    expect(writer.idle).toBe(true);
    expect(readFileSync(settingsJson, 'utf-8')).toBe(fileText(withTheme('light')));
});
