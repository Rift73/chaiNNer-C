import { mkdtempSync, rmSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import path from 'path';
import { expect, test } from 'vitest';
import { readOverrideFile } from '../../src/main/input-override';

test('readOverrideFile reads a file with a UTF-8 byte order mark', async () => {
    // Windows PowerShell 5.1 writes one with `Out-File -Encoding utf8` and `Set-Content -Encoding utf8`
    const dir = mkdtempSync(path.join(tmpdir(), 'input-override-test-'));
    try {
        const file = path.join(dir, 'override.json');
        const inputs = { '#00000000-0000-0000-0000-000000000000:0': 'C:\\images\\a.png' };
        writeFileSync(file, `\uFEFF${JSON.stringify({ inputs })}\r\n`, 'utf-8');

        expect(await readOverrideFile(file)).toEqual({ inputs });
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});
