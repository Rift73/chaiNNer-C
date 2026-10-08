import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { expect, test } from 'vitest';
import { RawSaveFile, SaveFile } from '../../src/main/SaveFile';

const dataDir = path.join(__dirname, '..', 'data');

test('Read save file with a UTF-8 byte order mark', async () => {
    // Windows PowerShell 5.1 writes one with `Set-Content -Encoding utf8`
    const filePath = path.join(dataDir, 'crop.chn');
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'save-file-test-'));
    try {
        const bomPath = path.join(dir, 'crop.chn');
        fs.writeFileSync(bomPath, `\uFEFF${fs.readFileSync(filePath, 'utf-8')}`, 'utf-8');

        expect(await SaveFile.read(bomPath)).toEqual(await SaveFile.read(filePath));
    } finally {
        fs.rmSync(dir, { recursive: true, force: true });
    }
});

for (const file of fs.readdirSync(dataDir)) {
    const filePath = path.join(dataDir, file);

    test(`Read save file ${file}`, async () => {
        const parsed = await SaveFile.read(filePath);
        expect(parsed).toMatchSnapshot();
    });
    test(`Write save file ${file}`, async () => {
        const json = SaveFile.stringify(await SaveFile.read(filePath), '0.0.0-test');
        const obj = JSON.parse(json) as RawSaveFile;
        delete obj.migration;
        delete obj.timestamp;
        expect(obj).toMatchSnapshot();
    });
}
